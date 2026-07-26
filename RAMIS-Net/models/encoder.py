import numpy as np
import torch
import torch.nn as nn
from timm.layers import trunc_normal_
from einops.layers.torch import Rearrange

from .base_modules import PatchEmbed3D, MixFFN, MixFFN_skip, MLP_FFN
from .mamba_blocks import ResMambaBlock, EfficientAttention3D
from .linear_former import RoPE, GateLinearAttentionNoSilu


class EfficientTransformerBlock3D(nn.Module):
    """
    Efficient 3D Transformer block.
    input:  B, D, H, W, C
    Output: B, D, H, W, C
    """

    def __init__(self, in_dim, key_dim, value_dim, head_count=1, token_mlp='mix_skip', recon_mode=False,
                 isLF=False, layerscale=False, layer_init_value=1e-6, isDrop_path=False, drop_path=0.):
        super().__init__()
        from timm.layers import DropPath

        self.pos = nn.Conv3d(in_dim, in_dim, 3, 1, 1, groups=in_dim)
        self.norm1 = nn.LayerNorm(in_dim)
        self.isLF = isLF
        self.layerscale = layerscale
        self.isDrop_path = isDrop_path
        if isLF:
            self.rope = RoPE(in_dim, head_count)
            self.attn = GateLinearAttentionNoSilu(dim=in_dim, num_heads=head_count)
            if isDrop_path:
                self.drop_path = DropPath(drop_path)
            if layerscale:
                self.gamma_1 = nn.Parameter(layer_init_value * torch.ones(1, in_dim, 1, 1, 1), requires_grad=True) # b c d h w
        else:
            self.attn = EfficientAttention3D(in_channels=in_dim, key_channels=key_dim,
                                         value_channels=value_dim, head_count=1, recon_mode=recon_mode)

        self.norm2 = nn.LayerNorm(in_dim)
        if token_mlp == 'mix':
            self.mlp = MixFFN(in_dim, int(in_dim * 4))
        elif token_mlp == 'mix_skip':
            self.mlp = MixFFN_skip(in_dim, int(in_dim * 4))
        else:
            self.mlp = MLP_FFN(in_dim, int(in_dim * 4))

    def forward(self, x: torch.Tensor, D, H, W, CLS=None, state="train") -> torch.Tensor:
        shortcut = x # b d h w d
        x = Rearrange('b d h w c -> b c d h w', d=D, h=H, w=W)(x)
        if self.isLF:
            x = x + self.pos(x)
        norm_1 = self.norm1(Rearrange('b c d h w -> b d h w c', d=D, h=H, w=W)(x))
        norm_1 = Rearrange('b d h w c -> b c d h w', d=D, h=H, w=W)(norm_1)

        if CLS is not None:
            attn, context, CLS = self.attn(norm_1, CLS=CLS)
        else:
            if self.isLF:
                # Rotational position encoding
                sin, cos = self.rope((D, H, W))
                if self.isDrop_path:
                    attn, context = self.drop_path(self.attn(norm_1, sin, cos))
                else:
                    attn, context = self.attn(norm_1, sin, cos)
                if self.layerscale:
                    attn = self.gamma_1 * attn
            else:
                attn, context = self.attn(norm_1, CLS=CLS)
        attn = Rearrange('b c d h w -> b d h w c')(attn)

        tx = shortcut + attn
        tx = Rearrange('b d h w c -> b (d h w) c')(tx)

        mx = tx + self.mlp(self.norm2(tx), D, H, W)
        mx = Rearrange('b (d h w) c -> b d h w c', d=D, h=H, w=W)(mx)

        return mx, context, CLS


class Encoder(nn.Module):
    """Multi-stage 3D encoder with transformers and mamba blocks."""
    def __init__(self, img_size, in_dim, key_dim, value_dim, layers, patch_sizes, in_chans=4,
                 norm_layer=nn.LayerNorm, patch_norm=True, head_count=1, token_mlp='mix_skip', use_ablation_module=False):
        super().__init__()

        spatial_dims = 3
        strides = [(4, 4, 4), (2, 2, 2), (2, 2, 2)]
        padding = [(0, 0, 0), (0, 0, 1), (0, 0, 1)]

        self.use_ablation_module = use_ablation_module

        self.patch_embed1 = PatchEmbed3D(img_size=img_size, patch_size=patch_sizes[0], in_chans=in_chans,
                                         embed_dim=in_dim[0], norm_layer=norm_layer if patch_norm else None,
                                         stride=strides[0], padding=padding[0])
        self.patch_embed2 = PatchEmbed3D(img_size=np.floor_divide(img_size, 4), patch_size=patch_sizes[1],
                                         in_chans=in_dim[0],
                                         embed_dim=in_dim[1], norm_layer=norm_layer if patch_norm else None,
                                         stride=strides[1], padding=padding[1])
        self.patch_embed3 = PatchEmbed3D(img_size=np.floor_divide(img_size, 8), patch_size=patch_sizes[2],
                                         in_chans=in_dim[1],
                                         embed_dim=in_dim[2], norm_layer=norm_layer if patch_norm else None,
                                         stride=strides[2], padding=padding[2])

        # transformer encoder stage 1
        self.block1 = EfficientTransformerBlock3D(in_dim[0], key_dim[0], value_dim[0], head_count, token_mlp, isLF=True, layerscale=True)
        self.block1_2 = EfficientTransformerBlock3D(in_dim[0], key_dim[0], value_dim[0], head_count, token_mlp, isLF=True, layerscale=True)
        self.mamba1 = ResMambaBlock(spatial_dims, in_dim[0], DownSample=True)
        self.norm1 = nn.LayerNorm(in_dim[0])

        # transformer encoder stage 2
        self.block2 = EfficientTransformerBlock3D(in_dim[1], key_dim[1], value_dim[1], head_count, token_mlp, isLF=True, layerscale=True)
        self.block2_2 = EfficientTransformerBlock3D(in_dim[1], key_dim[1], value_dim[1], head_count, token_mlp, isLF=True, layerscale=True)
        self.mamba2 = ResMambaBlock(spatial_dims, in_dim[1], DownSample=True)
        self.norm2 = nn.LayerNorm(in_dim[1])

        # transformer encoder stage 3
        self.block3 = nn.ModuleList([
            EfficientTransformerBlock3D(in_dim[2], key_dim[2], value_dim[2], head_count, token_mlp)
            for _ in range(layers[2])])
        self.norm3 = nn.LayerNorm(in_dim[2])

        self.cls_token = nn.Parameter(torch.zeros(1, in_dim[2], 4))
        trunc_normal_(self.cls_token, std=.02)

    def forward(self, x: torch.Tensor, state="train") -> torch.Tensor:
        B = x.shape[0]
        outs = []
        context_att = []

        # stage 1
        x, D, H, W = self.patch_embed1(x)
        x, context, _ = self.block1(x, D, H, W, state=state)
        context_att.append(context)
        x, context, _ = self.block1_2(x, D, H, W, state=state)
        context_att.append(context)
        x = Rearrange('b d h w c -> b c d h w')(x)
        x = self.mamba1(x, D, H, W, state=state)
        x = Rearrange('b c d h w -> b d h w c')(x)
        x = self.norm1(x)
        outs.append(x)

        # stage 2
        x = Rearrange('b d h w c -> b c d h w')(x)
        x, D, H, W = self.patch_embed2(x)
        x, context, _ = self.block2(x, D, H, W, state=state)
        context_att.append(context)
        x, context, _ = self.block2_2(x, D, H, W, state=state)
        context_att.append(context)
        x = Rearrange('b d h w c -> b c d h w')(x)
        x = self.mamba2(x, D, H, W, state=state)
        x = Rearrange('b c d h w -> b d h w c')(x)
        x = self.norm2(x)
        outs.append(x)

        # stage 3
        x = Rearrange('b d h w c -> b c d h w')(x)
        x, D, H, W = self.patch_embed3(x)
        # token loss
        cls_tokens = self.cls_token.expand(B, -1, -1)
        for blk in self.block3:
            x, context, cls_tokens = blk(x, D, H, W, cls_tokens, state=state)
        x = self.norm3(x)
        x = Rearrange('b d h w c -> b (d h w) c')(x)
        outs.append(x)

        return outs, context_att, cls_tokens
