import os
import sys
import logging
import datetime
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from functions.adamw import AdamW
from models import RAMISNet
from dataset import make_data_loaders
from losses import get_losses, DiceLoss
from utils import measure_dice_score
from configs import get_args
import warnings
import random
import numpy as np

# Set random seeds
random.seed(3407)
np.random.seed(3407)
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
torch.cuda.manual_seed_all(3407)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get command line arguments
args = get_args()
task_name = args.task_name

os.makedirs(args.path_to_log + task_name, exist_ok=True)
log_dir = os.path.join(args.path_to_log, task_name, "weights")

date_and_time = datetime.datetime.now()
logging.basicConfig(
    filename=args.path_to_log + task_name + f"/{task_name}" + str(date_and_time).replace(":", "_") + "_log.txt",
    level=logging.INFO,
    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info(f'{args}')


# Load data
loaders = make_data_loaders(args)
for phase in ['train', 'eval']:
    loader = loaders[phase]
    total = len(loader)
    logging.info(f'Number of {phase} subjects: {total}')

def build_model(inp_shape, num_classes, full_in_chans, missing_in_chans):
    model_full = RAMISNet(model_mode='full', img_size=inp_shape, num_classes=num_classes, in_chans=full_in_chans,
                           head_count=1, token_mlp_mode="mix_skip").cuda()
    model_missing = RAMISNet(model_mode='missing', img_size=inp_shape, num_classes=num_classes,
                              in_chans=missing_in_chans, head_count=1, token_mlp_mode="mix_skip").cuda()
    return model_full, model_missing


def load_old_model(model_full, model_missing, optimizer, saved_model_path):
    print("Constructing model from saved file... ")
    checkpoint = torch.load(saved_model_path)
    model_full.load_state_dict(checkpoint["model_full"])
    model_missing.load_state_dict(checkpoint["model_missing"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint["epochs"]
    return model_full, model_missing, optimizer, epoch


class PolyLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_epoch, power=0.9, last_epoch=-1):
        self.max_epoch = max_epoch
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1 - self.last_epoch / self.max_epoch) ** self.power for base_lr in self.base_lrs]


def make_optimizer_double(model1, model2):
    optimizer = AdamW([
            {'params': model1.parameters()},
            {'params': model2.parameters()}],
            lr=args.lr,
            weight_decay=args.weight_decay)
    scheduler = PolyLR(optimizer, max_epoch=args.n_epochs, power=args.power)
    return optimizer, scheduler


# Build and load models
model_full, model_missing = build_model(inp_shape=args.inputshape, num_classes=args.number_classes,
                                        full_in_chans=args.full_in_chans, missing_in_chans=args.missing_in_chans)
optimizer, scheduler = make_optimizer_double(model_full, model_missing)


# Loss functions
losses = get_losses()
criteria = DiceLoss()
mse_loss = torch.nn.MSELoss()
L1_loss = torch.nn.L1Loss()
L2_loss = torch.nn.MSELoss()
loss_cosine = nn.CosineEmbeddingLoss()

epoch = 0
epoch_init = epoch
n_epochs = args.n_epochs
iter_num = 0
best_dice = 0.0


# Training loop
for epoch in range(epoch_init, n_epochs):
    scheduler.step()

    train_loss = 0.0
    enc_loss_value = 0.0
    cls_loss_value = 0.0
    recon_loss_value = 0.0
    dice_full_loss_value = 0.0
    dice_missing_loss_value = 0.0
    consistency_loss_value = 0.0

    val_scores_full = 0.0
    val_scores_miss = 0.0
    val_loss_wt = 0.0
    val_loss_et = 0.0
    val_loss_ct = 0.0

    for phase in ['train', 'eval']:
        loader = loaders[phase]
        total = len(loader)
        if __name__ == '__main__':
            for batch_id, (batch_x, batch_y) in enumerate(loader):
                iter_num = iter_num + 1
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                # Rearrange tensor dimensions to (B, C, D, H, W)
                batch_x = Rearrange('b c h w d -> b c d h w')(batch_x)
                batch_y = Rearrange('b c h w d -> b c d h w')(batch_y)

                with torch.set_grad_enabled(phase == 'train'):
                    output_full, enc_context_att_full, CLS_full, recon_full = model_full(batch_x[:, 0:])

                    miss_modality = batch_x[:, 0:1, :, :, :]
                    output_missing, enc_context_att_missing, CLS_missing, _ = model_missing(miss_modality)

                    # MISA Loss
                    if args.context_loss == 'L1':
                        loss_context_att = (args.context_loss_l1_coef * L1_loss(enc_context_att_full[0],
                                                                                enc_context_att_missing[0]) +
                                            args.context_loss_l2_coef * L1_loss(enc_context_att_full[1],
                                                                                enc_context_att_missing[1]) +
                                            args.context_loss_l3_coef * L1_loss(enc_context_att_full[2],
                                                                                enc_context_att_missing[2]) +
                                            args.context_loss_l4_coef * L1_loss(enc_context_att_full[3],
                                                                                enc_context_att_missing[3]))
                    else:
                        # Cosine similarity alignment
                        enc_context_att_full[0] = enc_context_att_full[0].reshape(1, -1)
                        loss_context_att = (
                                    args.context_loss_l1_coef * loss_cosine(enc_context_att_full[0].reshape(1, -1),
                                                                            enc_context_att_missing[0].reshape(1, -1),
                                                                            torch.tensor([1]).to(device)) +
                                    args.context_loss_l2_coef * loss_cosine(enc_context_att_full[1].reshape(1, -1),
                                                                            enc_context_att_missing[1].reshape(1, -1),
                                                                            torch.tensor([1]).to(device)) +
                                    args.context_loss_l3_coef * loss_cosine(enc_context_att_full[2].reshape(1, -1),
                                                                            enc_context_att_missing[2].reshape(1, -1),
                                                                            torch.tensor([1]).to(device)) +
                                    args.context_loss_l4_coef * loss_cosine(enc_context_att_full[3].reshape(1, -1),
                                                                            enc_context_att_missing[3].reshape(1, -1),
                                                                            torch.tensor([1]).to(device))
                        )

                    # Dice and consistency loss
                    loss_dc, loss_miss_dc, consistency_loss = losses['co_loss'](output_full, output_missing, batch_y,
                                                                                epoch)

                    # Token loss
                    if args.token_loss == 'L1':
                        cls_token_loss = L1_loss(CLS_full, CLS_missing)
                    else:
                        cls_token_loss = L2_loss(CLS_full, CLS_missing)

                    # Reconstruction loss
                    if args.recon_loss == 'L1':
                        recon_loss = L1_loss(recon_full, batch_x[:, 0:])

                    # Total loss
                    tot_loss = (args.weight_full_coef * loss_dc +
                                args.weight_missing_coef * loss_miss_dc +
                                args.consistency_coef * consistency_loss +
                                args.context_loss_full_coef * loss_context_att +
                                args.token_loss_coef * cls_token_loss +
                                args.recon_loss_coef * recon_loss)

                    if phase == 'train':
                        optimizer.zero_grad()
                        tot_loss.backward()

                        train_loss += tot_loss.item()
                        enc_loss_value += (args.context_loss_full_coef * loss_context_att.item())
                        dice_full_loss_value += (args.weight_full_coef * loss_dc.item())
                        dice_missing_loss_value += (args.weight_missing_coef * loss_miss_dc.item())
                        consistency_loss_value += (args.consistency_coef * consistency_loss.item())
                        cls_loss_value += (args.token_loss_coef * cls_token_loss.item())
                        recon_loss_value += (args.recon_loss_coef * recon_loss.item())

                if phase == 'train':
                    optimizer.step()
                    if (batch_id + 1) % 20 == 0:
                        logging.info(
                            f'Epoch: {epoch + 1}|| iteration: {batch_id + 1}|| Training loss: {(train_loss / (batch_id + 1)):.5f}|| Encoder loss: {(enc_loss_value / (batch_id + 1)):.7f}|| DSC loss full: {(dice_full_loss_value / (batch_id + 1)):.5f}|| DSC loss missing: {(dice_missing_loss_value / (batch_id + 1)):.5f}|| Consistency loss: {(consistency_loss_value / (batch_id + 1)):.5f}|| CLS loss: {(cls_loss_value / (batch_id + 1)):.6f}|| Recon loss: {(recon_loss_value / (batch_id + 1)):.7f}')
                else:
                    val_scores_full_t, val_loss_full_wt, val_loss_full_et, val_loss_full_ct = measure_dice_score(
                        output_full, batch_y)
                    val_scores_miss_t, val_loss_missing_wt_t, val_loss_missing_et_t, val_loss_missing_ct_t = measure_dice_score(
                        output_missing, batch_y)

                    val_scores_full += val_scores_full_t
                    val_scores_miss += val_scores_miss_t
                    val_loss_wt += val_loss_missing_wt_t
                    val_loss_et += val_loss_missing_et_t
                    val_loss_ct += val_loss_missing_ct_t

            if phase == 'train':
                logging.info(f'### Epoch {epoch + 1} overall training loss>> {train_loss / (batch_id + 1)}')

            elif phase == 'eval':
                dice_full = (val_scores_full / (batch_id + 1))
                dice_missing = (val_scores_miss / (batch_id + 1))
                dice_wt = (val_loss_wt / (batch_id + 1))
                dice_et = (val_loss_et / (batch_id + 1))
                dice_ct = (val_loss_ct / (batch_id + 1))

                # Save model
                state = {}
                state['model_full'] = model_full.state_dict()
                state['model_missing'] = model_missing.state_dict()
                state['optimizer'] = optimizer.state_dict()
                state['epochs'] = epoch
                file_name = log_dir + '/model_weights.pth'
                torch.save(state, file_name)

                if dice_missing > best_dice:
                    torch.save(state, log_dir + '/best_model_weights.pth')
                    best_dice = dice_missing
                logging.info(f'### Best Val DSC missing >> {best_dice}')
                logging.info(
                    f'### Epoch {epoch + 1}, Val DSC full: {dice_full}, Val DSC missing: {dice_missing}, WT: {dice_wt}, CT: {dice_ct}, ET: {dice_et}')

