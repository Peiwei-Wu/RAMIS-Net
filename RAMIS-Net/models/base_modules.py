import torch
import torch.nn as nn
from monai.networks.blocks.convolutions import Convolution


class DWConv_3D(nn.Module):
    """Depthwise separable 3D convolution."""
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, bias: bool = False):
        super(DWConv_3D, self).__init__()
        self.depth_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=in_channels,
                                 strides=stride, kernel_size=kernel_size, bias=bias, conv_only=True, groups=in_channels)
        self.point_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels,
                                 strides=stride, kernel_size=1, bias=bias, conv_only=True, groups=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class My_DWConv(nn.Module):
    """Custom depthwise convolution for 2D input."""
    def __init__(self):
        super(My_DWConv, self).__init__()
        self.dwconv = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1, groups=3, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 9, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(9),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.dwconv(x)
        return x


class SELayer(nn.Module):
    """Squeeze-and-Excitation layer for 3D."""
    def __init__(self, channel):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.channel = channel
        self.fc2 = nn.Sequential(
            nn.Linear(self.channel * 2, self.channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel, self.channel * 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        input: B C/2 d h w
        output: tuple of 2 tensors
        """
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # B C/2
        y = self.fc2(y).view(b, c, 1, 1, 1)
        y = torch.chunk(y, chunks=2, dim=1)
        return y


class DWConv(nn.Module):
    """Depthwise 3D convolution."""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, D, H, W) -> torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, D, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)


class MixFFN_skip(nn.Module):
    """Mixed Feed-Forward Network with skip connection."""
    def __init__(self, c1, c2, dropout_ratio=0.1):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x: torch.Tensor, D, H, W, state="train") -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), D, H, W) + self.fc1(x)))
        out = self.fc2(ax)
        return out


class MixFFN(nn.Module):
    """Mixed Feed-Forward Network."""
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: torch.Tensor, D, H, W) -> torch.Tensor:
        ax = self.act(self.dwconv(self.fc1(x), D, H, W))
        out = self.fc2(ax)
        return out


class MLP_FFN(nn.Module):
    """MLP Feed-Forward Network."""
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x, D, H, W):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class PatchEmbed3D(nn.Module):
    """3D Patch Embedding."""
    def __init__(self, img_size=(128, 128, 128), patch_size=(4, 4, 4), in_chans=3, embed_dim=96, norm_layer=None,
                 stride=4, padding=1):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[1] // patch_size[1]]
        self.patches_resolution = patches_resolution

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """
        input:  B, C, D, H, W
        Output: B, D, H, W, C
        """
        import torch.nn.functional as F

        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.view(-1, D, Wh, Ww, self.embed_dim)
            _, D, H, W, C = x.shape

        return x, D, H, W
