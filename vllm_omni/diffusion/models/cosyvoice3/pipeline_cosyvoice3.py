"""Minimal CosyVoice3 DiT pipeline for Omni diffusion benchmarks.

Real inference mode implements CosyVoice3's CFM (flow matching) Euler solver with
classifier-free guidance, matching `cosyvoice/flow/flow_matching.py`.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.request import OmniDiffusionRequest
from .cosyvoice3_config import CosyVoice3DiTConfig
from .cosyvoice3_dit_vllm import CosyVoice3DiTVllm
from .utils.mask import make_pad_mask, add_optional_chunk_mask


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

        log_dir_env = os.environ.get("COSYVOICE_CACHE_LOG_DIR")
        if log_dir_env:
            self.cache_log_dir = Path(log_dir_env)
        else:
            self.cache_log_dir = Path(__file__).resolve().parents[4] / "cache_logs"

        backend = getattr(self.od_config, "cache_backend", "none") if self.od_config else "none"
        requested_cache_logging = (
            getattr(self.od_config, "enable_cache_logging", None) if self.od_config else None
        )
        requested_mode = getattr(self.od_config, "cache_logging_mode", None) if self.od_config else None
        self._cache_logging_mode = self._determine_cache_logging_mode(
            backend=backend,
            mode=requested_mode,
            flag=requested_cache_logging,
        )
        self._cache_summary_enabled = self._cache_logging_mode in ("return", "on", "json")
        self._cache_summary_emit_logs = self._cache_logging_mode in ("on", "json")
        self._json_cache_logging_enabled = self._cache_logging_mode == "json"

        if self._json_cache_logging_enabled:
            self.cache_log_dir.mkdir(parents=True, exist_ok=True)
        self._layer_names: List[str] = []
        self._block_hook_handles: List[Any] = []
        self._prev_block_outputs: List[Optional[torch.Tensor]] = []
        self._layer_diff_buffer: List[Optional[Dict[str, Any]]] = []
        self._per_layer_step_records: List[Dict[str, Any]] = []
        self._capture_block_diffs = False
        self._current_diff_step: Optional[int] = None
        if self._json_cache_logging_enabled:
            self._install_layer_cache_hooks()

    @staticmethod
    def _resolve_flow_path(model_path: str) -> str:
        if os.path.isdir(model_path):
            candidate = os.path.join(model_path, "flow.pt")
            if os.path.exists(candidate):
                return candidate
        return model_path

    @staticmethod
    def _determine_cache_logging_mode(
        backend: str,
        mode: Optional[str],
        flag: Optional[bool],
    ) -> str:
        """Resolve requested cache logging mode with sane fallbacks."""
        valid_modes = {"off", "return", "on", "json"}
        normalized_mode = None
        if mode is not None:
            candidate = str(mode).lower()
            if candidate in valid_modes:
                normalized_mode = candidate
            else:
                print(f"[CosyVoice3] Unknown cache logging mode '{mode}', defaulting based on backend")
        if normalized_mode:
            return normalized_mode

        if flag is True:
            return "json"
        if flag is False:
            return "off"

        if backend in ("cache_dit", "cache-dit"):
            return "return"
        return "off"

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """Weights already loaded via from_pretrained."""
        return set()

    def _collect_cached_steps(self, *, logging: bool) -> Optional[int]:
        try:
            import cache_dit

            stats = cache_dit.summary(self.transformer, details=False, logging=logging)
            if stats and isinstance(stats, list) and hasattr(stats[0], "cached_steps"):
                return len(stats[0].cached_steps)
        except Exception:
            pass
        return 0

    def _maybe_collect_cached_steps(self) -> Optional[int]:
        """Collect cached steps if the current logging mode requires it."""
        if not self._cache_summary_enabled:
            return None
        return self._collect_cached_steps(logging=self._cache_summary_emit_logs)

    def _install_layer_cache_hooks(self) -> None:
        """Attach forward hooks to each DiT block for per-layer diff logging."""
        if self._block_hook_handles:
            return
        blocks = list(self.transformer.transformer_blocks)
        self._layer_names = [f"transformer_block_{idx}" for idx in range(len(blocks))]
        self._prev_block_outputs = [None] * len(blocks)
        self._layer_diff_buffer = [None] * len(blocks)
        for idx, block in enumerate(blocks):
            handle = block.register_forward_hook(self._create_block_logging_hook(idx))
            self._block_hook_handles.append(handle)

    def _create_block_logging_hook(self, layer_idx: int):
        """Create a hook that tracks per-layer residual diffs between diffusion steps."""

        def _hook(module, inputs, output):  # pylint: disable=unused-argument
            if not self._capture_block_diffs:
                return
            tensor = output
            if isinstance(tensor, (tuple, list)) and tensor:
                tensor = tensor[0]
            if not isinstance(tensor, torch.Tensor):
                return
            with torch.no_grad():
                current = tensor.detach()
                prev = self._prev_block_outputs[layer_idx]
                abs_diff = None
                rel_diff = None
                has_previous = prev is not None and isinstance(prev, torch.Tensor)
                if has_previous and prev.shape == current.shape:
                    diff_tensor = torch.abs(current - prev)
                    abs_diff = float(torch.mean(diff_tensor).item())
                    prev_abs = torch.mean(torch.abs(prev)).item()
                    if prev_abs > 1e-12:
                        rel_diff = abs_diff / prev_abs
                self._layer_diff_buffer[layer_idx] = {
                    "value": rel_diff,
                    "abs_value": abs_diff,
                    "executed": True,
                    "has_previous": has_previous,
                }
                self._prev_block_outputs[layer_idx] = current.clone()

        return _hook

    def _reset_block_logging_state(self, reset_records: bool = True) -> None:
        """Reset cached per-layer logging buffers."""
        if not self._json_cache_logging_enabled:
            return
        num_layers = len(self._layer_names)
        self._prev_block_outputs = [None] * num_layers
        self._layer_diff_buffer = [None] * num_layers
        self._current_diff_step = None
        self._capture_block_diffs = False
        if reset_records:
            self._per_layer_step_records = []

    def _start_block_logging_step(self, step_idx: int) -> None:
        if not self._json_cache_logging_enabled:
            return
        self._current_diff_step = step_idx
        self._layer_diff_buffer = [None] * len(self._layer_names)
        self._capture_block_diffs = True

    def _finish_block_logging_step(self, step_idx: int) -> None:
        if not self._json_cache_logging_enabled or self._current_diff_step != step_idx:
            return
        layer_logs: List[Dict[str, Any]] = []
        for idx, name in enumerate(self._layer_names):
            entry = self._layer_diff_buffer[idx]
            if entry is None:
                layer_logs.append(
                    {
                        "layer": name,
                        "value": None,
                        "abs_value": None,
                        "executed": False,
                        "has_previous": self._prev_block_outputs[idx] is not None,
                        "cached": True,
                    }
                )
            else:
                layer_logs.append(
                    {
                        "layer": name,
                        "value": entry["value"],
                        "abs_value": entry.get("abs_value"),
                        "executed": True,
                        "has_previous": entry["has_previous"],
                        "cached": False,
                    }
                )
        self._per_layer_step_records.append(
            {
                "step": step_idx + 1,
                "layers": layer_logs,
            }
        )
        self._capture_block_diffs = False
        self._current_diff_step = None

    @staticmethod
    def _step_order_key(step_label: Any) -> Tuple[int, str]:
        """Extract numeric component from step label for stable sorting."""
        if isinstance(step_label, str):
            # step labels look like "step_0", "cfg_step_3", etc.
            for token in step_label.replace("-", "_").split("_"):
                if token.isdigit():
                    return int(token), step_label
        try:
            return int(step_label), str(step_label)
        except (TypeError, ValueError):
            return 0, str(step_label)

    @staticmethod
    def _serialize_diff_value(value: Any) -> Any:
        """Convert cache_dit residual diff values into JSON-serializable floats/lists."""
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.item())
            return [float(v) for v in value.detach().cpu().flatten().tolist()]
        if isinstance(value, (list, tuple)):
            return [CosyVoice3Pipeline._serialize_diff_value(v) for v in value]
        try:
            return float(value)
        except (TypeError, ValueError):
            return value

    @classmethod
    def _format_residual_entries(cls, residuals: Dict[str, Any] | None) -> List[Dict[str, Any]]:
        """Convert residual diff dict into a sorted list for logging."""
        if not residuals:
            return []
        entries: List[Dict[str, Any]] = []
        sorted_items = sorted(residuals.items(), key=lambda item: cls._step_order_key(item[0]))
        for step_key, diff_value in sorted_items:
            entries.append(
                {
                    "step": str(step_key),
                    "value": cls._serialize_diff_value(diff_value),
                }
            )
        return entries

    def _log_cache_residual_diffs(self, num_steps: int, seq_len: int, batch_size: int) -> None:
        """Dump per-layer residual diff statistics into a timestamped log file."""
        if not self._json_cache_logging_enabled:
            return
        backend = getattr(self.od_config, "cache_backend", "none")
        if backend not in ("cache_dit", "cache-dit"):
            return

        try:
            import cache_dit
        except Exception as exc:
            print(f"[CosyVoice3] cache_dit not available for logging residual diffs: {exc}")
            return

        try:
            stats_list = cache_dit.summary(self.transformer, logging=False)
        except Exception as exc:
            print(f"[CosyVoice3] Failed to collect cache summaries: {exc}")
            return

        if not stats_list:
            return

        cache_cfg = getattr(self.od_config, "cache_config", None)
        cfg_summary = None
        if cache_cfg is not None:
            cfg_summary = {
                "Fn_compute_blocks": getattr(cache_cfg, "Fn_compute_blocks", None),
                "Bn_compute_blocks": getattr(cache_cfg, "Bn_compute_blocks", None),
                "residual_diff_threshold": getattr(cache_cfg, "residual_diff_threshold", None),
                "max_warmup_steps": getattr(cache_cfg, "max_warmup_steps", None),
                "max_continuous_cached_steps": getattr(cache_cfg, "max_continuous_cached_steps", None),
                "enable_taylorseer": getattr(cache_cfg, "enable_taylorseer", None),
                "taylorseer_order": getattr(cache_cfg, "taylorseer_order", None),
            }

        now = datetime.utcnow()
        timestamp_label = now.strftime("%Y%m%d-%H%M%S_%f")
        log_payload: Dict[str, Any] = {
            "timestamp": f"{now.isoformat()}Z",
            "num_inference_steps": num_steps,
            "sequence_length": seq_len,
            "batch_size": batch_size,
            "cache_config": cfg_summary,
            "layers": [],
        }

        for idx, stats in enumerate(stats_list):
            cache_options = stats.cache_options or {}
            layer_name = cache_options.get("name") or cache_options.get("cache_name")
            if not layer_name:
                layer_name = f"layer_{idx}"

            residual_entries = self._format_residual_entries(stats.residual_diffs)
            cfg_residual_entries = self._format_residual_entries(stats.cfg_residual_diffs)

            if not residual_entries and not cfg_residual_entries:
                continue

            layer_entry: Dict[str, Any] = {
                "name": layer_name,
                "cached_steps": list(stats.cached_steps) if stats.cached_steps else [],
                "cfg_cached_steps": list(stats.cfg_cached_steps) if stats.cfg_cached_steps else [],
                "residual_diffs": residual_entries,
                "cfg_residual_diffs": cfg_residual_entries,
            }
            log_payload["layers"].append(layer_entry)

        if not log_payload["layers"]:
            return

        if self._json_cache_logging_enabled and self._per_layer_step_records:
            log_payload["per_layer_residual_diffs"] = self._per_layer_step_records

        log_path = self.cache_log_dir / f"cache_residual_{timestamp_label}.log"
        try:
            with open(log_path, "w", encoding="utf-8") as handle:
                json.dump(log_payload, handle, ensure_ascii=True, indent=2)
        except Exception as exc:
            print(f"[CosyVoice3] Failed to write cache residual diffs log: {exc}")
        finally:
            # Release buffered tensors and per-layer logs after dumping.
            self._reset_block_logging_state(reset_records=True)

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
        cached_steps = self._maybe_collect_cached_steps()

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
        batch_size = int(extra.get("batch_size", 1))

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
        else:
            seq_len = int(seq_len)

        num_steps = int(request.num_inference_steps or self.model_config.num_inference_steps)

        # Build t_span in [0,1], then apply cosine scheduler if configured.
        t_span = torch.linspace(0.0, 1.0, num_steps + 1, device=self.device, dtype=torch.float32)
        if self._t_scheduler == "cosine":
            t_span = 1.0 - torch.cos(t_span * 0.5 * torch.pi)

        # Deterministic initial noise (matches `CausalConditionalCFM.rand_noise` usage).
        # Add temperature parameter support
        temperature = 1.0  # Default temperature
        z = self._rand_noise[:, :, :seq_len].to(self.device) * temperature

        # Generate proper mask using make_pad_mask
        # Create token_len_total (assuming no padding in this case)
        token_len_total = torch.tensor([seq_len], dtype=torch.int32, device=self.device).repeat(batch_size)
        mask = (~make_pad_mask(token_len_total)).unsqueeze(1).to(self.device)

        # Euler solver with classifier-free guidance:
        # run estimator on batch=2, where the second sample has mu/spks/cond dropped.
        x = z.expand(batch_size, -1, -1).contiguous()

        # Pre-allocate inputs with matching dtype
        dtype = mu.dtype if mu is not None else torch.float32
        x_in = torch.zeros((2 * batch_size, self.model_config.mel_dim, seq_len), device=self.device, dtype=dtype)
        mask_in = torch.zeros((2 * batch_size, 1, seq_len), device=self.device, dtype=dtype)
        mu_in = torch.zeros((2 * batch_size, self.model_config.mel_dim, seq_len), device=self.device, dtype=dtype)
        t_in = torch.zeros((2 * batch_size,), device=self.device, dtype=dtype)
        spks_in = torch.zeros((2 * batch_size, self.model_config.mel_dim), device=self.device, dtype=dtype)
        cond_in = torch.zeros((2 * batch_size, self.model_config.mel_dim, seq_len), device=self.device, dtype=dtype)

        # Initialize time step (same as original)
        t = t_span[0]
        dt = (t_span[1] - t_span[0]).item()

        if self._json_cache_logging_enabled:
            self._reset_block_logging_state(reset_records=True)

        with torch.no_grad():
            for step in range(1, len(t_span)):
                step_idx = step - 1
                if self._json_cache_logging_enabled:
                    self._start_block_logging_step(step_idx)

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

                if self._json_cache_logging_enabled:
                    self._finish_block_logging_step(step_idx)

                # Update time step same as original
                t = t + dt
                if step < len(t_span) - 1:
                    dt = t_span[step + 1] - t

        # Collect cached steps for real inference too
        cached_steps = self._maybe_collect_cached_steps()
        if self._json_cache_logging_enabled:
            self._log_cache_residual_diffs(num_steps=num_steps, seq_len=seq_len, batch_size=batch_size)

        # Return both mel and cached steps in a dict
        return DiffusionOutput(output={
            "mel": x.float(),
            "cached_steps": cached_steps
        })
