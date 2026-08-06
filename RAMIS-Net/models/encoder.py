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
    Encoder block used by RARA or MISA.

    Input/Output: (batch, depth, height, width, channels).
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
        residual = x
        x = Rearrange('b d h w c -> b c d h w', d=D, h=H, w=W)(x)
        if self.isLF:
            x = x + self.pos(x)
        normalized_features = self.norm1(Rearrange('b c d h w -> b d h w c', d=D, h=H, w=W)(x))
        normalized_features = Rearrange('b d h w c -> b c d h w', d=D, h=H, w=W)(normalized_features)

        if CLS is not None:
            attention_features, relational_context, CLS = self.attn(normalized_features, CLS=CLS)
        else:
            if self.isLF:
                # RALA produces rank-enriched attention features and the REC matrix.
                sin, cos = self.rope((D, H, W))
                if self.isDrop_path:
                    attention_features, relational_context = self.drop_path(
                        self.attn(normalized_features, sin, cos)
                    )
                else:
                    attention_features, relational_context = self.attn(normalized_features, sin, cos)
                if self.layerscale:
                    attention_features = self.gamma_1 * attention_features
            else:
                attention_features, relational_context = self.attn(normalized_features, CLS=CLS)
        attention_features = Rearrange('b c d h w -> b d h w c')(attention_features)

        attention_residual = residual + attention_features
        attention_residual = Rearrange('b d h w c -> b (d h w) c')(attention_residual)

        block_output = attention_residual + self.mlp(
            self.norm2(attention_residual), D, H, W
        )
        block_output = Rearrange('b (d h w) c -> b d h w c', d=D, h=H, w=W)(block_output)

        return block_output, relational_context, CLS


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
        batch_size = x.shape[0]
        multi_scale_features = []
        rank_enriched_contexts = []

        # stage 1
        x, D, H, W = self.patch_embed1(x)
        x, rank_enriched_context, _ = self.block1(x, D, H, W, state=state)
        rank_enriched_contexts.append(rank_enriched_context)
        x, rank_enriched_context, _ = self.block1_2(x, D, H, W, state=state)
        rank_enriched_contexts.append(rank_enriched_context)
        x = Rearrange('b d h w c -> b c d h w')(x)
        x = self.mamba1(x, D, H, W, state=state)
        x = Rearrange('b c d h w -> b d h w c')(x)
        x = self.norm1(x)
        multi_scale_features.append(x)

        # stage 2
        x = Rearrange('b d h w c -> b c d h w')(x)
        x, D, H, W = self.patch_embed2(x)
        x, rank_enriched_context, _ = self.block2(x, D, H, W, state=state)
        rank_enriched_contexts.append(rank_enriched_context)
        x, rank_enriched_context, _ = self.block2_2(x, D, H, W, state=state)
        rank_enriched_contexts.append(rank_enriched_context)
        x = Rearrange('b d h w c -> b c d h w')(x)
        x = self.mamba2(x, D, H, W, state=state)
        x = Rearrange('b c d h w -> b d h w c')(x)
        x = self.norm2(x)
        multi_scale_features.append(x)

        # stage 3
        x = Rearrange('b d h w c -> b c d h w')(x)
        x, D, H, W = self.patch_embed3(x)
        # MISA modality-semantic vectors.
        modality_semantic_vectors = self.cls_token.expand(batch_size, -1, -1)
        for blk in self.block3:
            x, relational_context, modality_semantic_vectors = blk(
                x, D, H, W, modality_semantic_vectors, state=state
            )
        x = self.norm3(x)
        x = Rearrange('b d h w c -> b (d h w) c')(x)
        multi_scale_features.append(x)

        return multi_scale_features, rank_enriched_contexts, modality_semantic_vectors
