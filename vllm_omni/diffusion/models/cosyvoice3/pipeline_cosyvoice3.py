"""Minimal CosyVoice3 DiT pipeline for Omni diffusion benchmarks.

Real inference mode implements CosyVoice3's CFM (flow matching) Euler solver with
classifier-free guidance, matching `cosyvoice/flow/flow_matching.py`.
"""

import os
import time
from typing import Iterable, List, Tuple

import torch
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.request import OmniDiffusionRequest
from .cosyvoice3_config import CosyVoice3DiTConfig
from .cosyvoice3_dit_vllm import CosyVoice3DiTVllm


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
        # Force float32 for stability and to match CosyVoice reference implementation.
        self.model = self.model.to(device=self.device, dtype=torch.float32)
        self.model.eval()
        self.transformer = self.model.dit
        self.vae = _DummyVAE()

        # Disable the default HF loader since we manually load weights.
        self.weights_sources: List = []

        self.default_seq_len = tf_params.get("seq_len", 200)
        self.default_batch_size = tf_params.get("batch_size", 1)
        self.default_num_warmup = tf_params.get("num_warmup", 0)

        # Match CosyVoice3 `cfm_params.content` defaults from the shipped cosyvoice3.yaml.
        self._t_scheduler = tf_params.get("t_scheduler", "cosine")
        self._inference_cfg_rate = float(tf_params.get("inference_cfg_rate", 0.7))

        # Match `CausalConditionalCFM` deterministic noise (seed=0, shape [1,80,50*300]).
        gen = torch.Generator(device=self.device).manual_seed(0)
        self.register_buffer(
            "_rand_noise",
            torch.randn(
                (1, self.model_config.mel_dim, 50 * 300),
                generator=gen,
                device=self.device,
                dtype=torch.float32,
            ),
            persistent=False,
        )

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

        # Check if this is a real inference request or benchmark
        is_benchmark = extra.get("benchmark_mode", False)

        if is_benchmark:
            # Original benchmark mode
            return self._forward_benchmark(request)
        else:
            # Real inference mode
            return self._forward_inference(request)

    def _forward_benchmark(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        """Original benchmark implementation"""
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

    def _forward_inference(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        """Real inference implementation using actual DiT diffusion

        CosyVoice uses Conditional Flow Matching (CFM), where the ODE starts from
        the condition mu (not random noise) and flows to the target mel spectrogram.
        """
        extra = request.extra or {}

        # Extract real inputs from extra (all in [batch, dim, seq] format from cosy_server)
        # - condition_vector corresponds to `mu` in CosyVoice CFM (encoder output)
        # - cond corresponds to prompt mel + zeros (conditioning)
        mu = extra.get("condition_vector")  # [batch, mel_dim, seq]
        spks = extra.get("speaker_embedding")  # [batch, spk_dim]
        cond = extra.get("cond")  # [batch, mel_dim, seq]
        seq_len = extra.get("seq_len")
        batch_size = extra.get("batch_size", 1)

        if mu is None or spks is None or cond is None:
            raise ValueError("condition_vector, speaker_embedding, and cond must be provided in extra for real inference")

        # Convert to tensors (float32 only)
        if not isinstance(mu, torch.Tensor):
            mu = torch.tensor(mu, dtype=torch.float32)
        if not isinstance(spks, torch.Tensor):
            spks = torch.tensor(spks, dtype=torch.float32)
        if not isinstance(cond, torch.Tensor):
            cond = torch.tensor(cond, dtype=torch.float32)

        mu = mu.to(self.device, dtype=torch.float32)
        spks = spks.to(self.device, dtype=torch.float32)
        cond = cond.to(self.device, dtype=torch.float32)

        if mu.dim() == 2:
            mu = mu.unsqueeze(0)
        if spks.dim() == 1:
            spks = spks.unsqueeze(0)
        if cond.dim() == 2:
            cond = cond.unsqueeze(0)

        if seq_len is None:
            seq_len = int(mu.shape[2])

        num_steps = int(request.num_inference_steps or self.model_config.num_inference_steps)

        # Build t_span in [0,1], then apply cosine scheduler if configured.
        t_span = torch.linspace(0.0, 1.0, num_steps + 1, device=self.device, dtype=torch.float32)
        if self._t_scheduler == "cosine":
            t_span = 1.0 - torch.cos(t_span * 0.5 * torch.pi)

        # Deterministic initial noise (matches `CausalConditionalCFM.rand_noise` usage).
        z = self._rand_noise[:, :, :seq_len].to(self.device)

        # Full-length mask (no padding). Match CosyVoice mask shape (B, 1, T).
        mask = torch.ones((batch_size, 1, seq_len), device=self.device, dtype=torch.float32)

        # Euler solver with classifier-free guidance:
        # run estimator on batch=2, where the second sample has mu/spks/cond dropped.
        x = z.expand(batch_size, -1, -1).contiguous()

        # Pre-allocate inputs to match CosyVoice memory layout constraints.
        x_in = torch.zeros((2 * batch_size, self.model_config.mel_dim, seq_len), device=self.device, dtype=torch.float32)
        mask_in = torch.zeros((2 * batch_size, 1, seq_len), device=self.device, dtype=torch.float32)
        mu_in = torch.zeros((2 * batch_size, self.model_config.mel_dim, seq_len), device=self.device, dtype=torch.float32)
        t_in = torch.zeros((2 * batch_size,), device=self.device, dtype=torch.float32)
        spks_in = torch.zeros((2 * batch_size, self.model_config.mel_dim), device=self.device, dtype=torch.float32)
        cond_in = torch.zeros((2 * batch_size, self.model_config.mel_dim, seq_len), device=self.device, dtype=torch.float32)

        t = t_span[0]
        dt = (t_span[1] - t_span[0]).item()

        with torch.no_grad():
            for step in range(1, len(t_span)):
                x_in[:] = x.repeat(2, 1, 1)
                mask_in[:] = mask.repeat(2, 1, 1)
                mu_in[:batch_size] = mu
                spks_in[:batch_size] = spks
                cond_in[:batch_size] = cond
                t_in[:] = t

                # Call the real DiT estimator (signature matches CosyVoice).
                dphi_dt = self.model.dit(
                    x=x_in,
                    mask=mask_in,
                    mu=mu_in,
                    t=t_in,
                    spks=spks_in,
                    cond=cond_in,
                    streaming=False,
                )

                guided, cfg = torch.split(dphi_dt, [batch_size, batch_size], dim=0)
                guided = (1.0 + self._inference_cfg_rate) * guided - self._inference_cfg_rate * cfg
                x = x + dt * guided

                t = t_span[step]
                if step < len(t_span) - 1:
                    dt = (t_span[step + 1] - t).item()

        return DiffusionOutput(output=x.float())
