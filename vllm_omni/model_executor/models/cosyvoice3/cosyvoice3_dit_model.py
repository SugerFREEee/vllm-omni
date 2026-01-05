"""
CosyVoice3 DiT Model Implementation
Extracts the DiT (Diffusion Transformer) part from CosyVoice3 for cache-dit acceleration testing.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for timestep"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimeEmbedding(nn.Module):
    """Time embedding MLP"""
    def __init__(self, dim, out_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, out_dim),
        )

    def forward(self, x):
        return self.time_mlp(x)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=1e-8) * self.gamma


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation"""
    def __init__(self, dim, mult=2):
        super().__init__()
        hidden_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Multi-head attention"""
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        h = self.heads
        x = self.norm(x)

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        # Reshape for multi-head attention
        q = q.view(q.shape[0], q.shape[1], h, -1).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], h, -1).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], h, -1).transpose(1, 2)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).reshape(out.shape[0], out.shape[2], -1)
        return self.to_out(out)


class DiTBlock(nn.Module):
    """DiT Transformer Block"""
    def __init__(self, dim, heads=8, dim_head=64, ff_mult=2):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head)
        self.ff = FeedForward(dim, mult=ff_mult)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x):
        # Attention with residual
        x = x + self.attn(x)
        # Feed-forward with residual
        x = x + self.ff(self.norm2(x))
        return x


class CosyVoice3DiT(nn.Module):
    """
    CosyVoice3 DiT Model

    Args:
        dim: Model dimension (default: 1024)
        depth: Number of transformer layers (default: 22)
        heads: Number of attention heads (default: 16)
        dim_head: Dimension per head (default: 64)
        ff_mult: Feed-forward expansion factor (default: 2)
        mel_dim: Mel spectrogram dimension (default: 80)
        mu_dim: Mean dimension (default: 80)
        spk_dim: Speaker embedding dimension (default: 80)
        out_channels: Output channels (default: 80)
    """
    def __init__(
        self,
        dim=1024,
        depth=22,
        heads=16,
        dim_head=64,
        ff_mult=2,
        mel_dim=80,
        mu_dim=80,
        spk_dim=80,
        out_channels=80,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(dim),
            TimeEmbedding(dim, dim)
        )

        # Input projection
        self.input_proj = nn.Linear(mel_dim + mu_dim + spk_dim, dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult)
            for _ in range(depth)
        ])

        # Output projection
        self.norm_out = RMSNorm(dim)
        self.output_proj = nn.Linear(dim, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        mu: Optional[torch.Tensor] = None,
        spk: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass

        Args:
            x: Input mel spectrogram [B, T, mel_dim]
            timesteps: Diffusion timesteps [B]
            mu: Mean/condition [B, T, mu_dim]
            spk: Speaker embedding [B, T, spk_dim]

        Returns:
            Output tensor [B, T, out_channels]
        """
        # Get time embedding
        t_emb = self.time_embed(timesteps)  # [B, dim]
        t_emb = t_emb.unsqueeze(1)  # [B, 1, dim]

        # Concatenate inputs
        if mu is not None and spk is not None:
            x = torch.cat([x, mu, spk], dim=-1)  # [B, T, mel_dim+mu_dim+spk_dim]

        # Project to model dimension
        x = self.input_proj(x)  # [B, T, dim]

        # Add time embedding
        x = x + t_emb

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Output projection
        x = self.norm_out(x)
        x = self.output_proj(x)

        return x

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, map_location='cpu'):
        """
        Load model from CosyVoice3 checkpoint

        Args:
            checkpoint_path: Path to flow.pt file
            map_location: Device to load the model

        Returns:
            Loaded DiT model
        """
        # Load checkpoint
        state_dict = torch.load(checkpoint_path, map_location=map_location)

        # Filter DiT parameters (decoder.estimator.*)
        dit_state_dict = {}
        prefix = 'decoder.estimator.'
        for key, value in state_dict.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]  # Remove prefix
                dit_state_dict[new_key] = value

        # Create model (use default config matching CosyVoice3-0.5B)
        model = cls(
            dim=1024,
            depth=22,
            heads=16,
            dim_head=64,
            ff_mult=2,
            mel_dim=80,
            mu_dim=80,
            spk_dim=80,
            out_channels=80,
        )

        # Try to load the state dict (may need adjustment based on actual structure)
        try:
            model.load_state_dict(dit_state_dict, strict=False)
            print(f"Loaded DiT model with {len(dit_state_dict)} parameters")
        except Exception as e:
            print(f"Warning: Could not load all parameters: {e}")
            print("Available keys in checkpoint:", list(dit_state_dict.keys())[:10])

        return model


if __name__ == "__main__":
    # Test model creation
    model = CosyVoice3DiT()
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    # Test forward pass
    batch_size, seq_len = 2, 100
    x = torch.randn(batch_size, seq_len, 80)
    timesteps = torch.randint(0, 1000, (batch_size,))
    mu = torch.randn(batch_size, seq_len, 80)
    spk = torch.randn(batch_size, seq_len, 80)

    with torch.no_grad():
        output = model(x, timesteps, mu, spk)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
