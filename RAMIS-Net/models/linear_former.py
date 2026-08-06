"""Rank-Aware Linear Attention (RALA) for 3D feature volumes."""

import torch
import torch.nn as nn
from einops import rearrange


class RoPE(nn.Module):
    """Generate 3D rotary positional encodings for query and key features."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 4))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer("angle", angle)
        self.proj1 = nn.Linear(int(embed_dim / 2 * 3), self.embed_dim)
        self.proj2 = nn.Linear(int(embed_dim / 2 * 3), self.embed_dim)
        self.act = nn.ReLU()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, slen: tuple[int]):
        """Return sine and cosine embeddings for a (depth, height, width) grid."""
        depth, height, width = slen

        depth_positions = torch.arange(depth).to(self.angle)
        height_positions = torch.arange(height).to(self.angle)
        width_positions = torch.arange(width).to(self.angle)

        sin_depth = torch.sin(depth_positions[:, None] * self.angle[None, :])
        sin_height = torch.sin(height_positions[:, None] * self.angle[None, :])
        sin_width = torch.sin(width_positions[:, None] * self.angle[None, :])
        sin_depth = sin_depth.unsqueeze(1).unsqueeze(1).repeat(1, height, width, 1)
        sin_height = sin_height.unsqueeze(0).unsqueeze(2).repeat(depth, 1, width, 1)
        sin_width = sin_width.unsqueeze(0).unsqueeze(0).repeat(depth, height, 1, 1)
        sin = torch.cat([sin_depth, sin_height, sin_width], dim=-1)

        cos_depth = torch.cos(depth_positions[:, None] * self.angle[None, :])
        cos_height = torch.cos(height_positions[:, None] * self.angle[None, :])
        cos_width = torch.cos(width_positions[:, None] * self.angle[None, :])
        cos_depth = cos_depth.unsqueeze(1).unsqueeze(1).repeat(1, height, width, 1)
        cos_height = cos_height.unsqueeze(0).unsqueeze(2).repeat(depth, 1, width, 1)
        cos_width = cos_width.unsqueeze(0).unsqueeze(0).repeat(depth, height, 1, 1)
        cos = torch.cat([cos_depth, cos_height, cos_width], dim=-1)

        sin = self.act(self.norm1(self.proj1(sin.flatten(0, 2))))
        cos = self.act(self.norm2(self.proj2(cos.flatten(0, 2))))
        return sin, cos


def rotate_every_two(x):
    """Rotate adjacent channel pairs for rotary positional encoding."""
    even_components = x[:, :, :, ::2]
    odd_components = x[:, :, :, 1::2]
    rotated_features = torch.stack([-odd_components, even_components], dim=-1)
    return rotated_features.flatten(-2)


def theta_shift(x, sin, cos):
    """Apply rotary positional encoding."""
    return (x * cos) + (rotate_every_two(x) * sin)


class GateLinearAttentionNoSilu(nn.Module):
    """High-Rank Linear Attention used in the RARA module."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.qkvo = nn.Conv3d(dim, dim * 4, 1)
        self.elu = nn.ELU()
        self.lepe = nn.Conv3d(dim, dim, 5, 1, 2, groups=dim)
        self.proj = nn.Conv3d(dim, dim, 1)

    def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
        """Return the rank-enriched attention output and REC matrix."""
        batch_size, channels, depth, height, width = x.shape
        num_tokens = depth * height * width

        qkvo_features = self.qkvo(x)
        qkv_features = qkvo_features[:, :3 * self.dim, :, :, :]
        channel_modulator = qkvo_features[:, 3 * self.dim:, :, :, :]
        local_positional_encoding = self.lepe(qkv_features[:, 2 * self.dim:, :, :, :])

        query, key, value = rearrange(
            qkv_features,
            "b (m n c) d h w -> m b n (d h w) c",
            m=3,
            n=self.num_heads,
        )

        # Positive kernel mapping used by linear attention.
        query = self.elu(query) + 1.0
        key = self.elu(key) + 1.0

        # Q_g and alpha_j in Eqs. (2)-(4): global-query-guided context weighting.
        global_query = query.mean(dim=-2, keepdim=True)
        context_weights = self.scale * global_query @ key.transpose(-1, -2)
        context_weights = torch.softmax(context_weights, dim=-1).transpose(-1, -2)
        weighted_key = key * context_weights * num_tokens

        position_encoded_query = theta_shift(query, sin, cos)
        position_encoded_key = theta_shift(weighted_key, sin, cos)

        normalization_factor = 1 / (
            query @ weighted_key.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6
        )
        rank_enriched_context = (
            position_encoded_key.transpose(-2, -1) * (num_tokens ** -0.5)
        ) @ (value * (num_tokens ** -0.5))

        rank_enriched_output = position_encoded_query @ rank_enriched_context * normalization_factor
        rank_enriched_output = rearrange(
            rank_enriched_output,
            "b n (d h w) c -> b (n c) d h w",
            d=depth,
            h=height,
            w=width,
        )
        rank_enriched_output = rank_enriched_output + local_positional_encoding

        attention_output = self.proj(rank_enriched_output * channel_modulator)
        return attention_output, rank_enriched_context
