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
            decoder_output = self.layer_up(x1)
        else:
            batch_size, depth, height, width, channels = x2.shape

            skip_fused_features = torch.cat([x1, x2], dim=-1)
            skip_fused_features = skip_fused_features.view(
                batch_size, -1, skip_fused_features.shape[-1]
            )
            projected_features = self.concat_linear(skip_fused_features)
            projected_features = Rearrange(
                'b (d h w) c -> b d h w c',
                b=batch_size,
                h=height,
                w=width,
            )(projected_features)

            if self.recon_mode == False:
                projected_features, _, _ = self.layer_former_1(
                    projected_features, depth, height, width, CLS=CLS
                )
            decoded_features, _, _ = self.layer_former_2(
                projected_features, depth, height, width, CLS=CLS
            )
            decoded_features = Rearrange('b d h w c -> b (d h w) c')(decoded_features)

            if self.is_last:
                if self.recon_mode:
                    decoded_features = Rearrange(
                        'b (d h w) c -> b c d h w',
                        b=batch_size,
                        h=height,
                        w=width,
                    )(decoded_features)
                decoder_output = self.layer_up(decoded_features)
                if not self.recon_mode:
                    decoder_output = Rearrange('b d h w c -> b c d h w')(decoder_output)
                decoder_output = self.last_layer(decoder_output)
            else:
                decoder_output = self.layer_up(decoded_features)

        return decoder_output
