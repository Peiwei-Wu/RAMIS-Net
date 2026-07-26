import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from .encoder import EfficientTransformerBlock3D


class PatchExpand(nn.Module):
    """Patch expansion layer."""
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim_scale = dim_scale
        self.dim = dim
        self.expand = nn.Linear(dim, 4 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, D*H*W, C
        """
        D, H, W = self.input_resolution
        x = x.flatten(2)
        x = self.expand(x)
        B, L, C = x.shape

        x = x.view(B, D, H, W, C)
        x = rearrange(x, 'b d h w (p1 p2 p3 c)-> b (d p1) (h p2) (w p3) c', p1=self.dim_scale, p2=self.dim_scale,
                      p3=self.dim_scale, c=C // 8)
        x = self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Module):
    """Final patch expansion layer with 4x upsampling."""
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 4 * 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, D*H*W, C
        """
        D, H, W = self.input_resolution

        x = self.expand(x)
        B, L, C = x.shape

        x = x.view(B, D, H, W, C)
        x = rearrange(x, 'b d h w (p1 p2 p3 c)-> b (d p1) (h p2) (w p3) c', p1=self.dim_scale, p2=self.dim_scale,
                      p3=self.dim_scale,
                      c=C // (self.dim_scale ** 3))

        x = self.norm(x)

        return x


class MyDecoderLayer(nn.Module):
    """Decoder layer with optional transformer and upsampling."""
    def __init__(self, input_size, in_out_chan, head_count, token_mlp_mode, n_class=9,
                 norm_layer=nn.LayerNorm, is_last=False, recon_mode=False):
        super().__init__()

        self.recon_mode = recon_mode
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]

        self.is_last = is_last

        if not is_last:
            self.concat_linear = nn.Linear(dims * 2, out_dim)
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.concat_linear = nn.Linear(dims * 4, out_dim)
            if recon_mode:
                self.layer_up = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False)
            else:
                self.layer_up = FinalPatchExpand_X4(input_resolution=input_size, dim=out_dim, dim_scale=4,
                                                    norm_layer=norm_layer)
            self.last_layer = nn.Conv3d(out_dim, n_class, 1, bias=False)

        if self.recon_mode == False:
            self.layer_former_1 = EfficientTransformerBlock3D(out_dim, key_dim, value_dim, head_count,
                                                              token_mlp_mode, recon_mode=recon_mode)
        self.layer_former_2 = EfficientTransformerBlock3D(out_dim, key_dim, value_dim, head_count,
                                                          token_mlp_mode, recon_mode=recon_mode)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x1, x2=None, first=False, CLS=None):

        if first:
            out = self.layer_up(x1)
        else:
            b, d, h, w, c = x2.shape

            cat_x = torch.cat([x1, x2], dim=-1)
            cat_x = cat_x.view(b, -1, cat_x.shape[-1])
            cat_linear_x = self.concat_linear(cat_x)
            cat_linear_x = Rearrange('b (d h w) c -> b d h w c', b=b, h=h, w=w)(cat_linear_x)

            if self.recon_mode == False:
                cat_linear_x, _, _ = self.layer_former_1(cat_linear_x, d, h, w, CLS=CLS)
            tran_layer_2, _, _ = self.layer_former_2(cat_linear_x, d, h, w, CLS=CLS)
            tran_layer_2 = Rearrange('b d h w c -> b (d h w) c')(tran_layer_2)

            if self.is_last:
                if self.recon_mode:
                    tran_layer_2 = Rearrange('b (d h w) c -> b c d h w', b=b, h=h, w=w)(tran_layer_2)
                out = self.layer_up(tran_layer_2)
                if not self.recon_mode:
                    out = Rearrange('b d h w c -> b c d h w')(out)
                out = self.last_layer(out)
            else:
                out = self.layer_up(tran_layer_2)

        return out
