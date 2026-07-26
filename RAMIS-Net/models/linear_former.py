import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RoPE(nn.Module):
    """Rotary Position Embedding (RoPE) for 3D tensors.

    Applies rotary positional encoding to attention queries and keys.
    """

    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Compute the inverse frequencies for RoPE
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, spatial_dims):
        """Generate rotation matrices for given spatial dimensions.

        Args:
            spatial_dims: tuple of (D, H, W) spatial dimensions

        Returns:
            tuple of (sin, cos) rotation components
        """
        D, H, W = spatial_dims
        device = self.inv_freq.device

        # Create position indices for each dimension
        pos_d = torch.arange(D, device=device, dtype=self.inv_freq.dtype)
        pos_h = torch.arange(H, device=device, dtype=self.inv_freq.dtype)
        pos_w = torch.arange(W, device=device, dtype=self.inv_freq.dtype)

        # Compute frequencies for each dimension
        freqs_d = torch.einsum("i,j->ij", pos_d, self.inv_freq)
        freqs_h = torch.einsum("i,j->ij", pos_h, self.inv_freq)
        freqs_w = torch.einsum("i,j->ij", pos_w, self.inv_freq)

        # Interleave cos and sin
        emb_d = torch.cat([freqs_d, freqs_d], dim=-1)
        emb_h = torch.cat([freqs_h, freqs_h], dim=-1)
        emb_w = torch.cat([freqs_w, freqs_w], dim=-1)

        # Combine spatial embeddings by averaging
        emb = (emb_d.mean(0, keepdim=True) + emb_h.mean(0, keepdim=True) + emb_w.mean(0, keepdim=True)) / 3

        sin = emb.sin()
        cos = emb.cos()

        return sin, cos


class GateLinearAttentionNoSilu(nn.Module):
    """Gated Linear Attention without SiLU activation.

    A linear attention mechanism with gating, designed for efficient 3D processing.
    """

    def __init__(self, dim, num_heads=1, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Gate parameters
        self.gate = nn.Linear(dim, dim)

    def forward(self, x, sin, cos):
        """Apply gated linear attention with rotary embeddings.

        Args:
            x: input tensor of shape (B, C, D, H, W)
            sin: sine component of rotary embeddings
            cos: cosine component of rotary embeddings

        Returns:
            tuple of (attention output, context)
        """
        B, C, D, H, W = x.shape

        # Reshape for attention computation
        x_flat = x.flatten(2).transpose(1, 2)  # (B, D*H*W, C)

        # Generate query, key, value
        qkv = self.qkv(x_flat)
        qkv = qkv.reshape(B, D*H*W, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each (B, num_heads, D*H*W, head_dim)

        # Apply rotary embeddings to q and k
        sin_expanded = sin.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, -1)
        cos_expanded = cos.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, -1)

        # RoPE rotation
        q = self._apply_rope(q, sin_expanded, cos_expanded)
        k = self._apply_rope(k, sin_expanded, cos_expanded)

        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Compute context for potential use in other layers
        context = attn.mean(dim=1)  # Average across heads

        # Apply attention to values
        x_attn = (attn @ v).transpose(1, 2).reshape(B, D*H*W, C)

        # Apply gating
        gate = torch.sigmoid(self.gate(x_attn))
        x_attn = x_attn * gate

        # Project output
        x_out = self.proj(x_attn)
        x_out = self.proj_drop(x_out)

        # Reshape back to original shape
        x_out = x_out.transpose(1, 2).reshape(B, C, D, H, W)

        return x_out, context

    @staticmethod
    def _apply_rope(x, sin, cos):
        """Apply rotary embeddings to x.

        Args:
            x: tensor of shape (B, num_heads, seq_len, head_dim)
            sin: sine embeddings
            cos: cosine embeddings

        Returns:
            rotated tensor
        """
        # Reshape x for rotation: (B, num_heads, seq_len, head_dim // 2, 2)
        *batch, seq_len, dim = x.shape
        x = x.reshape(*batch, seq_len, dim // 2, 2)

        # Apply rotation using complex number representation
        x_rot = x.clone()
        x_rot[..., 0] = x[..., 0] * cos - x[..., 1] * sin
        x_rot[..., 1] = x[..., 0] * sin + x[..., 1] * cos

        return x_rot.reshape(*batch, seq_len, dim)
