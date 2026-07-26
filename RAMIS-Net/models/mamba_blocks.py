import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from einops.layers.torch import Rearrange
from .base_modules import SELayer


class MambaLayer(nn.Module):
    """Mamba layer with SE attention and skip connections."""
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim // 4,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))
        self.se = SELayer(input_dim // 4)

    def forward(self, x):
        """
        input:  b c d h w
        output: b c' d h w
        """
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)

        # Process chunk 1
        M1 = self.mamba(x1)  # B L C/4
        M2 = self.skip_scale * x1  # B L C/4
        M1 = M1.transpose(-1, -2).reshape(B, self.input_dim // 4, *img_dims)
        M2 = M2.transpose(-1, -2).reshape(B, self.input_dim // 4, *img_dims)
        M = torch.cat([M1, M2], dim=1)  # B C/2 d h w
        M = self.se(M)
        M1 = M[0] * M1
        M2 = M[1] * M2
        x_mamba1 = M1 + M2
        x_mamba1 = x_mamba1.reshape(B, C // 4, n_tokens).transpose(-1, -2)

        # Process chunk 2
        M1 = self.mamba(x2)
        M2 = self.skip_scale * x2
        M1 = M1.transpose(-1, -2).reshape(B, self.input_dim // 4, *img_dims)
        M2 = M2.transpose(-1, -2).reshape(B, self.input_dim // 4, *img_dims)
        M = torch.cat([M1, M2], dim=1)
        M = self.se(M)
        M1 = M[0] * M1
        M2 = M[1] * M2
        x_mamba2 = M1 + M2
        x_mamba2 = x_mamba2.reshape(B, C // 4, n_tokens).transpose(-1, -2)

        # Process chunk 3
        M1 = self.mamba(x3)
        M2 = self.skip_scale * x3
        M1 = M1.transpose(-1, -2).reshape(B, self.input_dim // 4, *img_dims)
        M2 = M2.transpose(-1, -2).reshape(B, self.input_dim // 4, *img_dims)
        M = torch.cat([M1, M2], dim=1)
        M = self.se(M)
        M1 = M[0] * M1
        M2 = M[1] * M2
        x_mamba3 = M1 + M2
        x_mamba3 = x_mamba3.reshape(B, C // 4, n_tokens).transpose(-1, -2)

        # Process chunk 4
        M1 = self.mamba(x4)
        M2 = self.skip_scale * x4
        M1 = M1.transpose(-1, -2).reshape(B, self.input_dim // 4, *img_dims)
        M2 = M2.transpose(-1, -2).reshape(B, self.input_dim // 4, *img_dims)
        M = torch.cat([M1, M2], dim=1)
        M = self.se(M)
        M1 = M[0] * M1
        M2 = M[1] * M2
        x_mamba4 = M1 + M2
        x_mamba4 = x_mamba4.reshape(B, C // 4, n_tokens).transpose(-1, -2)

        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=2)
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)

        return out


def get_mamba_layer(
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        stride: int = 1
):
    """Factory function to create a mamba layer."""
    mamba_layer = MambaLayer(input_dim=in_channels, output_dim=out_channels)
    if stride != 1:
        return nn.Sequential(mamba_layer)
    return mamba_layer


class ResMambaBlock(nn.Module):
    """Residual Mamba block with MLPs."""
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            kernel_size: int = 3,
            DownSample=False,
    ) -> None:

        super().__init__()
        self.DownSample = DownSample

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")

        self.mamba1 = get_mamba_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels
        )
        self.mamba2 = get_mamba_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels
        )

        self.mlp_norm1 = nn.LayerNorm(in_channels)
        self.mlp_norm2 = nn.LayerNorm(in_channels)

        from .base_modules import MixFFN_skip
        self.mlp1 = MixFFN_skip(in_channels, in_channels * 4)
        self.mlp2 = MixFFN_skip(in_channels, in_channels * 4)

    def forward(self, x, D, H, W, state="train"):
        """
        input: b c h w d
        output: b c h w d
        """

        x_init = x
        x = self.mamba1(x)
        x = x + x_init
        x = Rearrange('b c d h w -> b (d h w) c')(x)
        x = x + self.mlp1(self.mlp_norm1(x), D, H, W, state=state)
        x = Rearrange('b (d h w) c -> b c d h w', d=D, h=H, w=W)(x)

        x_init = x
        x = self.mamba2(x)
        x = x + x_init
        x = Rearrange('b c d h w -> b (d h w) c')(x)
        x = x + self.mlp2(self.mlp_norm2(x), D, H, W, state=state)
        x = Rearrange('b (d h w) c -> b c d h w', d=D, h=H, w=W)(x)

        if self.DownSample:
            return x  # b c d h w
        else:
            return x


class EfficientAttention3D(nn.Module):
    """Efficient 3D Attention mechanism."""

    def __init__(self, in_channels, key_channels, value_channels, head_count=1, recon_mode=False):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        self.recon_mode = recon_mode

        self.keys = nn.Conv3d(in_channels, key_channels, 1)
        self.queries = nn.Conv3d(in_channels, key_channels, 1)
        self.values = nn.Conv3d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv3d(value_channels, in_channels, 1)

    def forward(self, input_, CLS):
        n, c, d, h, w = input_.size()

        keys = self.keys(input_).reshape((n, self.key_channels, d * h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, d * h * w)
        values = self.values(input_).reshape((n, self.value_channels, d * h * w))

        if CLS is not None:
            keys = torch.cat((CLS, keys), dim=-1)
            queries = torch.cat((CLS, queries), dim=-1)
            values = torch.cat((CLS, values), dim=-1)

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]

            context = key @ value.transpose(1, 2)  # kv
            attended_value = (context.transpose(1, 2) @ query)

            if CLS is not None:
                CLS, attended_value = attended_value[..., :4], attended_value[..., 4:]

            attended_value = attended_value.reshape(n, head_value_channels, d, h, w)  # n*v
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)

        if CLS is not None:
            return attention, context, CLS
        return attention, context
