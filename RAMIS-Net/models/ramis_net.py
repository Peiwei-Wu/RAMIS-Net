import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import MyDecoderLayer


class RAMISNet(nn.Module):
    """RAMIS-Net: Multi-modal brain segmentation model with encoder-decoder architecture."""
    def __init__(self, model_mode, img_size=(128, 160, 192), num_classes=4, in_chans=4, head_count=1,
                 token_mlp_mode="mix_skip", use_ablation_module=False):
        super().__init__()

        # Encoder configuration
        in_dim = [64, 128, 320]
        key_dim = [64, 128, 320]
        value_dim = [64, 128, 320]
        layers = [2, 2, 2]
        patch_sizes = [(4, 4, 4), (3, 3, 3), (3, 3, 3), (3, 3, 3)]

        self.enc = Encoder(img_size, in_dim, key_dim, value_dim, layers, patch_sizes,
                           in_chans=in_chans, head_count=1, token_mlp='mix_skip', use_ablation_module=use_ablation_module)

        # Decoder configuration
        d_base_feat_size = [4, 5, 6]
        in_out_chan = [[32, 64, 64, 64], [144, 128, 128, 128],
                       [288, 320, 320, 320]]  # [dim, out_dim, key_dim, value_dim]

        self.decoder_2 = MyDecoderLayer((d_base_feat_size[0] * 2, d_base_feat_size[1] * 2, d_base_feat_size[2] * 2),
                                        in_out_chan[2], head_count, token_mlp_mode, n_class=num_classes,
                                        recon_mode=False)

        self.decoder_1 = MyDecoderLayer((d_base_feat_size[0] * 4, d_base_feat_size[1] * 4, d_base_feat_size[2] * 4),
                                        in_out_chan[1], head_count, token_mlp_mode, n_class=num_classes,
                                        recon_mode=False)

        self.decoder_0 = MyDecoderLayer((d_base_feat_size[0] * 8, d_base_feat_size[1] * 8, d_base_feat_size[2] * 8),
                                        in_out_chan[0], head_count, token_mlp_mode, n_class=num_classes,
                                        is_last=True, recon_mode=False)

        self.model_mode = model_mode
        if self.model_mode == 'full':
            self.decoder_recon = MyDecoderLayer(
                (d_base_feat_size[0] * 8, d_base_feat_size[1] * 8, d_base_feat_size[2] * 8),
                in_out_chan[0], head_count, token_mlp_mode, n_class=num_classes,
                is_last=True, recon_mode=True)
            self.cls_projection = nn.Linear(in_out_chan[2][-1], in_out_chan[0][-1])

    def forward(self, x, state="train"):

        enc_out, enc_context_att, CLS = self.enc(x, state=state)
        CLS = CLS.permute(0, 2, 1)

        # Decode stage 2
        tmp_2 = self.decoder_2(enc_out[2], first=True)

        # Decode stage 1
        tmp_1 = self.decoder_1(tmp_2, enc_out[1], first=False)

        # Decode stage 0
        tmp_seg = self.decoder_0(tmp_1, enc_out[0], first=False)

        uout = torch.sigmoid(tmp_seg)

        # Reconstruction stage (if model_mode is 'full')
        if self.model_mode == 'full':
            proj_CLS = self.cls_projection(CLS).permute(0, 2, 1)
            tmp_recon = self.decoder_recon(tmp_1, enc_out[0], first=False, CLS=proj_CLS)
            return uout, enc_context_att, CLS, tmp_recon

        return uout, enc_context_att, CLS, []
