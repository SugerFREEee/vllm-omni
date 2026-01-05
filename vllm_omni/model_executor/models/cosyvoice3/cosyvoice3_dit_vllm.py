"""
CosyVoice3 DiT vLLM Wrapper

This module provides a vLLM-compatible wrapper for the CosyVoice3 DiT model,
with optional cache-dit acceleration support.
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    from .cosyvoice3_dit_model import CosyVoice3DiT
    from .cosyvoice3_config import CosyVoice3DiTConfig
except ImportError:
    # Fallback for standalone testing
    from cosyvoice3_dit_model import CosyVoice3DiT
    from cosyvoice3_config import CosyVoice3DiTConfig


class CosyVoice3DiTVllm(nn.Module):
    """
    vLLM-compatible wrapper for CosyVoice3 DiT model.

    This wrapper:
    1. Wraps the standalone CosyVoice3DiT model
    2. Adapts inputs from vLLM format to DiT format
    3. Optionally enables cache-dit acceleration
    4. Provides vLLM-compatible forward interface
    """

    def __init__(self, config: CosyVoice3DiTConfig):
        super().__init__()
        self.config = config

        # Create standalone DiT model
        self.dit = CosyVoice3DiT(
            dim=config.hidden_size,
            depth=config.num_hidden_layers,
            heads=config.num_attention_heads,
            dim_head=config.dim_head,
            ff_mult=config.ff_mult,
            mel_dim=config.mel_dim,
            mu_dim=config.mu_dim,
            spk_dim=config.spk_dim,
            out_channels=config.out_channels,
        )

        # Cache-DiT setup
        self.enable_cache = config.enable_cache_dit
        self.cache_adapter = None
        self._cache_initialized = False

        if self.enable_cache:
            self._setup_cache_dit()

    def _setup_cache_dit(self):
        """
        Setup cache-dit acceleration.

        This method:
        1. Creates a BlockAdapter for the DiT transformer blocks
        2. Configures DBCacheConfig with user-specified parameters
        3. Optionally enables TaylorSeer calibrator
        4. Enables cache acceleration
        """
        try:
            import cache_dit
            from cache_dit import (
                BlockAdapter,
                ForwardPattern,
                DBCacheConfig,
                TaylorSeerCalibratorConfig
            )

            print(f"[CosyVoice3] Setting up cache-dit acceleration...")

            # Create BlockAdapter
            self.cache_adapter = BlockAdapter(
                pipe=None,  # Transformer-only interface
                transformer=self.dit,
                blocks=self.dit.blocks,
                forward_pattern=ForwardPattern.Pattern_3,  # Single input/output
            )

            # Configure cache
            cache_config = DBCacheConfig(
                Fn_compute_blocks=min(self.config.cache_Fn, len(self.dit.blocks)),
                Bn_compute_blocks=self.config.cache_Bn,
                residual_diff_threshold=self.config.cache_threshold,
                max_warmup_steps=self.config.cache_warmup_steps,
                max_cached_steps=-1,  # Unlimited
                max_continuous_cached_steps=-1,  # Unlimited
                num_inference_steps=self.config.num_inference_steps,  # Required for Transformer-only
            )

            # TaylorSeer calibrator (optional)
            calibrator_config = None
            if self.config.enable_taylorseer:
                calibrator_config = TaylorSeerCalibratorConfig(
                    enable_calibrator=True,
                    taylorseer_order=self.config.taylorseer_order,
                )
                print(f"[CosyVoice3] TaylorSeer calibrator enabled (order={self.config.taylorseer_order})")

            # Enable cache
            cache_dit.enable_cache(
                self.cache_adapter,
                cache_config=cache_config,
                calibrator_config=calibrator_config
            )

            self._cache_initialized = True

            print(f"[CosyVoice3] Cache-dit enabled: "
                  f"F{cache_config.Fn_compute_blocks}B{cache_config.Bn_compute_blocks}, "
                  f"threshold={cache_config.residual_diff_threshold}")

        except Exception as e:
            print(f"[CosyVoice3] Failed to enable cache-dit: {e}")
            print(f"[CosyVoice3] Continuing without cache acceleration")
            self.enable_cache = False
            self._cache_initialized = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        condition_vector: torch.Tensor,
        speaker_embedding: torch.Tensor,
        timesteps: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with vLLM-compatible interface.

        Args:
            hidden_states: [batch, seq_len, mel_dim] - Noised mel features
            condition_vector: [batch, seq_len, mel_dim] - Condition (mu)
            speaker_embedding: [batch, seq_len, mel_dim] - Speaker embedding
            timesteps: [batch] - Diffusion timesteps (0-1000)

        Returns:
            output: [batch, seq_len, mel_dim] - Predicted noise/velocity
        """
        # Input adaptation: vLLM format -> DiT format
        x = hidden_states
        mu = condition_vector
        spk = speaker_embedding

        # Call standalone DiT
        output = self.dit(x, timesteps, mu, spk)

        return output

    def refresh_cache_context(self, num_inference_steps: Optional[int] = None):
        """
        Refresh cache-dit context for new inference.

        This should be called before each new diffusion sequence.

        Args:
            num_inference_steps: Number of inference steps (optional override)
        """
        if not self.enable_cache or self.cache_adapter is None:
            return

        try:
            import cache_dit
            steps = num_inference_steps or self.config.num_inference_steps
            cache_dit.refresh_context(
                self.cache_adapter,
                num_inference_steps=steps,
                verbose=False
            )
        except Exception as e:
            print(f"[CosyVoice3] Failed to refresh cache context: {e}")

    def get_cache_stats(self):
        """
        Get cache statistics.

        Returns:
            Cache statistics dict or None if cache is not enabled
        """
        if not self.enable_cache or self.cache_adapter is None:
            return None

        try:
            import cache_dit
            stats = cache_dit.summary(self.cache_adapter, details=False)
            return stats
        except Exception as e:
            print(f"[CosyVoice3] Failed to get cache stats: {e}")
            return None

    def disable_cache(self):
        """Disable cache-dit acceleration"""
        if not self.enable_cache or self.cache_adapter is None:
            return

        try:
            import cache_dit
            cache_dit.disable_cache(self.cache_adapter)
            self.enable_cache = False
            print("[CosyVoice3] Cache-dit disabled")
        except Exception as e:
            print(f"[CosyVoice3] Failed to disable cache: {e}")

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[CosyVoice3DiTConfig] = None,
        **kwargs
    ):
        """
        Load model from pretrained weights.

        Args:
            model_path: Path to flow.pt checkpoint
            config: Model configuration (optional)
            **kwargs: Additional arguments

        Returns:
            Loaded model instance
        """
        if config is None:
            config = CosyVoice3DiTConfig()

        model = cls(config)

        # Load weights
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            model.dit.load_state_dict(state_dict, strict=False)
            print(f"[CosyVoice3] Loaded weights from: {model_path}")
        except Exception as e:
            print(f"[CosyVoice3] Failed to load weights: {e}")
            print(f"[CosyVoice3] Using random initialization")

        return model
