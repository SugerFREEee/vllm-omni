"""Minimal CosyVoice3 DiT pipeline for Omni diffusion benchmarks."""

import os
import time
from typing import Iterable, List, Tuple

import torch
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.model_executor.models.cosyvoice3 import CosyVoice3DiTConfig, CosyVoice3DiTVllm


class _DummyVAE(nn.Module):
    """Placeholder VAE so Omni hooks can toggle slicing/tiling attributes."""

    def __init__(self):
        super().__init__()
        self.use_slicing = False
        self.use_tiling = False


class CosyVoice3Pipeline(nn.Module):
    """Pipeline that exposes CosyVoice3 DiT through the Omni diffusion engine.

    This pipeline mirrors the lightweight benchmarking harness:
    it runs CosyVoice3 DiT on randomly generated inputs to measure latency.
    The cache backend (cache-dit / tea_cache) is managed by Omni's workers.
    """

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()

        tf_params = od_config.tf_model_config.params if od_config.tf_model_config else {}
        self.model_config = CosyVoice3DiTConfig(
            hidden_size=tf_params.get("hidden_size", 1024),
            num_hidden_layers=tf_params.get("num_hidden_layers", 22),
            num_attention_heads=tf_params.get("num_attention_heads", 16),
            mel_dim=tf_params.get("mel_dim", 80),
            enable_cache_dit=False,
            cache_Fn=tf_params.get("cache_Fn", 8),
            cache_Bn=tf_params.get("cache_Bn", 0),
            cache_threshold=tf_params.get("cache_threshold", 0.08),
            cache_warmup_steps=tf_params.get("cache_warmup_steps", 8),
            num_inference_steps=tf_params.get("num_inference_steps", 28),
            enable_taylorseer=tf_params.get("enable_taylorseer", False),
            taylorseer_order=tf_params.get("taylorseer_order", 1),
        )

        flow_path = self._resolve_flow_path(od_config.model)
        self.model = CosyVoice3DiTVllm.from_pretrained(flow_path, config=self.model_config)
        self.model = self.model.to(device=self.device, dtype=od_config.dtype)
        self.model.eval()
        self.transformer = self.model.dit
        self.vae = _DummyVAE()

        # Disable the default HF loader since we manually load weights.
        self.weights_sources: List = []

        self.default_seq_len = tf_params.get("seq_len", 200)
        self.default_batch_size = tf_params.get("batch_size", 1)
        self.default_num_warmup = tf_params.get("num_warmup", 0)

    @staticmethod
    def _resolve_flow_path(model_path: str) -> str:
        if os.path.isdir(model_path):
            candidate = os.path.join(model_path, "flow.pt")
            if os.path.exists(candidate):
                return candidate
        return model_path

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """Weights already loaded via from_pretrained."""
        return set()

    def _collect_cached_steps(self) -> int:
        try:
            import cache_dit

            stats = cache_dit.summary(self.transformer, details=False)
            if stats and isinstance(stats, list) and hasattr(stats[0], "cached_steps"):
                return len(stats[0].cached_steps)
        except Exception:
            pass
        return 0

    def forward(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        extra = request.extra or {}
        batch_size = int(extra.get("batch_size", self.default_batch_size))
        seq_len = int(extra.get("seq_len", self.default_seq_len))
        num_warmup = int(extra.get("num_warmup", self.default_num_warmup))
        dtype = self.od_config.dtype or next(self.transformer.parameters()).dtype

        num_steps = request.num_inference_steps or self.model_config.num_inference_steps
        device = self.device

        def _rand():
            return torch.randn(batch_size, seq_len, self.model_config.mel_dim, device=device, dtype=dtype)

        hidden_states = _rand()
        condition_vector = _rand()
        speaker_embedding = _rand()
        timesteps = torch.zeros(batch_size, device=device, dtype=dtype)

        # Warmup
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = self.model(
                    hidden_states=hidden_states,
                    condition_vector=condition_vector,
                    speaker_embedding=speaker_embedding,
                    timesteps=timesteps,
                )

        # TTFT
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = self.model(
                hidden_states=hidden_states,
                condition_vector=condition_vector,
                speaker_embedding=speaker_embedding,
                timesteps=timesteps,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        ttft = time.perf_counter() - start

        # Per-step timing
        step_times: List[float] = []
        for step in range(num_steps):
            t = step / max(num_steps - 1, 1)
            timesteps = torch.full((batch_size,), t * 1000, device=device, dtype=dtype)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = self.model(
                    hidden_states=hidden_states,
                    condition_vector=condition_vector,
                    speaker_embedding=speaker_embedding,
                    timesteps=timesteps,
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - start)

        avg_time = sum(step_times) / len(step_times) if step_times else 0.0
        total_time = sum(step_times)
        cached_steps = self._collect_cached_steps()

        payload = [
            {
                "ttft": ttft,
                "avg_time": avg_time,
                "total_time": total_time,
                "cached_steps": cached_steps,
                "step_times": step_times,
            }
        ]
        return DiffusionOutput(output=payload)
