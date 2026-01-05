"""CosyVoice3 DiT Configuration"""

from transformers import PretrainedConfig


class CosyVoice3DiTConfig(PretrainedConfig):
    """
    Configuration class for CosyVoice3 DiT model.

    This configuration stores all parameters for the DiT (Diffusion Transformer)
    model extracted from CosyVoice3, including cache-dit acceleration settings.
    """

    model_type = "cosyvoice3_dit"

    def __init__(
        self,
        # Model architecture parameters
        hidden_size=1024,
        num_hidden_layers=22,
        num_attention_heads=16,
        dim_head=64,
        ff_mult=2,
        mel_dim=80,
        mu_dim=80,
        spk_dim=80,
        out_channels=80,
        dropout=0.0,
        # Cache-DiT acceleration parameters
        enable_cache_dit=False,
        cache_Fn=8,
        cache_Bn=0,
        cache_threshold=0.08,
        cache_warmup_steps=8,
        num_inference_steps=28,
        # TaylorSeer calibrator parameters
        enable_taylorseer=False,
        taylorseer_order=1,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Model architecture
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.dim_head = dim_head
        self.ff_mult = ff_mult
        self.mel_dim = mel_dim
        self.mu_dim = mu_dim
        self.spk_dim = spk_dim
        self.out_channels = out_channels
        self.dropout = dropout

        # Cache-DiT configuration
        self.enable_cache_dit = enable_cache_dit
        self.cache_Fn = cache_Fn
        self.cache_Bn = cache_Bn
        self.cache_threshold = cache_threshold
        self.cache_warmup_steps = cache_warmup_steps
        self.num_inference_steps = num_inference_steps

        # TaylorSeer calibrator
        self.enable_taylorseer = enable_taylorseer
        self.taylorseer_order = taylorseer_order

    @property
    def input_dim(self):
        """Total input dimension (mel + mu + spk)"""
        return self.mel_dim + self.mu_dim + self.spk_dim
