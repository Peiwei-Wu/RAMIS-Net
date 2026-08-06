from einops.layers.torch import Rearrange
from importlib import reload
import argparse
import torch
from dataset import make_data_loaders
from models import RAMISNet
import eval_utils

from eval_utils import *
import warnings
import numpy as np

warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")

use_cuda = torch.cuda.is_available()
parser = argparse.ArgumentParser(description='RAMIS-Net')

parser.add_argument('--task_name', type=str, default='RAMIS-Net',
                    help='task name')
parser.add_argument('--saved_model_path', type=str,
                    default='./new_results/RAMIS-Net/weights/best_model_weights.pth',
                    help='Pre-trained model path')

parser.add_argument('--datasets', type=str, default='BRATS 2020', help='the type of dataset [BRATS 2018, BRATS 2020]')
parser.add_argument('--path_to_data', type=str, default='./datasets/MICCAI_BraTS2020_TrainingData/', help='path to dataset')

parser.add_argument('--modalities', type=str, nargs='*', default=['flair', 't1', 't1ce', 't2'],
                    help='List of modalities need to be used for training and evaluating the model')
parser.add_argument('--n_missing_modalities', type=int, default=1,
                    help='number of modalities for the missing path')

parser.add_argument('--number_classes', type=int, default=4,
                    help='number of classes in the target dataset')
parser.add_argument('--batch_size_tr', type=int, default=1,
                    help='batch size for train')
parser.add_argument('--batch_size_va', type=int, default=1,
                    help='batch size for validation')
parser.add_argument('--progress_p', type=float, default=0.1,
                    help='progress report frequency')
parser.add_argument('--inputshape', default=[160, 192, 128],
                    help='input shape')

parser.add_argument('--missing_in_chans', type=int, default=1,
                    help='missing modality input channels')
parser.add_argument('--full_in_chans', type=int, default=4,
                    help='full modality input channels')

parser.add_argument('--generate_seg_figure', type=str, default='store_false',
                    help='whether to generate seg figure')

args = parser.parse_args(args=[])
args.modalities = ['flair', 't1', 't1ce', 't2']
loaders = make_data_loaders(args)
train_loader = loaders['train']
eval_loader = loaders['eval']
test_loader = loaders['test']

torch.backends.cudnn.enabled = False


def build_model(inp_shape, num_classes, missing_in_chans):
    return RAMISNet(
        model_mode='missing',
        img_size=inp_shape,
        num_classes=num_classes,
        in_chans=missing_in_chans,
        head_count=1,
        token_mlp_mode="mix_skip",
        use_ablation_module=True
    ).cuda()


def load_model(model_missing, saved_model_path):
    print("Constructing model from saved file... ")
    checkpoint = torch.load(saved_model_path, map_location='cuda')
    model_missing.load_state_dict(checkpoint["model_missing"])
    return model_missing


class RunningAverages:
    """Track independent running means for evaluation metrics."""

    def __init__(self):
        self.totals = {}
        self.counts = {}

    def update(self, **values):
        for metric_name, metric_value in values.items():
            self.totals[metric_name] = self.totals.get(metric_name, 0.0) + metric_value
            self.counts[metric_name] = self.counts.get(metric_name, 0) + 1

    def mean(self, metric_name):
        if metric_name not in self.totals:
            raise KeyError(f"Metric '{metric_name}' has not been updated.")
        return self.totals[metric_name] / self.counts[metric_name]


model_missing = build_model(
    args.inputshape,
    args.number_classes,
    args.missing_in_chans
)
model_missing = load_model(model_missing, args.saved_model_path)
model_missing.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
model_missing = model_missing.to(device)

loaders = {
    'train': train_loader,
    'eval': eval_loader,
    'test': test_loader
}

# Running Dice and HD95 metrics.
test_metrics = RunningAverages()

# Main evaluation loop
for phase in ['test']:
    loader = loaders[phase]
    batch_count = 0
    for batch_id, (batch_x, batch_y) in enumerate(loader):
        batch_count += 1
        batch_x, batch_y = batch_x.contiguous().to(device), batch_y.contiguous().to(device)

        batch_x = Rearrange('b c h w d -> b c d h w')(batch_x)
        batch_y = Rearrange('b c h w d -> b c d h w')(batch_y)

        with torch.no_grad():
            miss_modality = batch_x[:, 0:1, :, :, :].contiguous()
            output_missing = model_missing(miss_modality)
            pred = output_missing[0]

        if args.generate_seg_figure == 'store_true':
            if batch_id == 10:
                if args.datasets == "BRATS 2018":
                    val_sc_miss, val_wt_miss, val_et_miss, val_ct_miss = measure_dice_score(pred, batch_y,
                                                                                            thresh=[0.5, 0.5, 0.5],
                                                                                            wt_j=3, ct_j=2, et_j=None,
                                                                                            slice_idx=130,
                                                                                            save_path="./new_results/2025_BRATS2018/0_1_with_bv.png",
                                                                                            save_path_gt="./2025_BRATS2018/groundtruth_with_bv.png",
                                                                                            base_volume=batch_x[0, 0],
                                                                                            alpha=0.6)
                    save_modal_slices(batch_x=batch_x, z=130,
                                      save_dir="./new_results/2025_BRATS2018/")
                elif args.datasets == "BRATS 2020":
                    val_sc_miss, val_wt_miss, val_et_miss, val_ct_miss = measure_dice_score(pred, batch_y,
                                                                                            thresh=[0.5, 0.5, 0.5],
                                                                                            wt_j=3, ct_j=2, et_j=None,
                                                                                            slice_idx=140,
                                                                                            save_path="./new_results/2026_BRATS2020/0_1_with_bv.png",
                                                                                            save_path_gt="./new_results/2026_BRATS2020/groundtruth_with_bv.png",
                                                                                            base_volume=batch_x[0, 0],
                                                                                            alpha=0.6)
                    save_modal_slices(batch_x=batch_x, z=140,
                                      save_dir="./new_results/2026_BRATS2020/")
        else:
            val_dice_miss, val_wt_miss, val_et_miss, val_ct_miss = measure_dice_score(pred, batch_y, thresh=[0.5, 0.5, 0.5], wt_j=3, ct_j=2, et_j=None)
            val_hd95_miss, hd95_WT, hd95_TC, hd95_ET = np.array(cal_hd95(pred, batch_y, thresh=[0.5, 0.5, 0.5], wt_j=3, ct_j=2, et_j=None))

            test_metrics.update(
                dice_missing=val_dice_miss,
                dice_wt=val_wt_miss,
                dice_tc=val_ct_miss,
                dice_et=val_et_miss,
                hd95_missing=val_hd95_miss,
                hd95_wt=hd95_WT,
                hd95_tc=hd95_TC,
                hd95_et=hd95_ET,
            )

    if args.generate_seg_figure != 'store_true':
        dice_missing = test_metrics.mean('dice_missing')
        dice_wt = test_metrics.mean('dice_wt')
        dice_tc = test_metrics.mean('dice_tc')
        dice_et = test_metrics.mean('dice_et')
        hd95_missing = test_metrics.mean('hd95_missing')
        hd95_wt = test_metrics.mean('hd95_wt')
        hd95_tc = test_metrics.mean('hd95_tc')
        hd95_et = test_metrics.mean('hd95_et')

    print(f'✓ Completed {batch_count} batches')
    if args.generate_seg_figure != 'store_true':
        print(f'### Val DSC  missing: {dice_missing}, WT: {dice_wt}, TC: {dice_tc}, ET: {dice_et}')
        print(f'### Val HD95 missing: {hd95_missing}, WT: {hd95_wt}, TC: {hd95_tc}, ET: {hd95_et}')
