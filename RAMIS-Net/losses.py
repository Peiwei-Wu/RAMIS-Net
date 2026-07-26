import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch, consistency = 10, consistency_rampup = 20.0):
    return consistency * sigmoid_rampup(epoch, consistency_rampup)


def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)


def dice_loss(input, target):
    """soft dice loss"""
    with torch.cuda.amp.autocast(enabled=False):
        eps = 1e-6
        iflat = input.reshape(-1)
        tflat = target.reshape(-1)
        intersection = (iflat * tflat).sum()

    return 1 - 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum() + eps)


def gram_matrix(input):
    a, b, c, d, e = input.size()
    features = input.view(a * b, c * d * e)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d * e)


def unet_Co_loss(batch_pred_full, batch_pred_missing, batch_y, epoch):
    loss_dict = {}

    loss_fn = DiceCELoss3D(weight_ce=0.5, weight_dice=0.5)
    loss_dict['ed_dc_loss']  = loss_fn(batch_pred_full[:, 0], batch_y[:, 0])
    loss_dict['net_dc_loss'] = loss_fn(batch_pred_full[:, 1], batch_y[:, 1])
    loss_dict['et_dc_loss']  = loss_fn(batch_pred_full[:, 2], batch_y[:, 2])

    loss_dict['ed_miss_dc_loss']  = loss_fn(batch_pred_missing[:, 0], batch_y[:, 0])
    loss_dict['net_miss_dc_loss'] = loss_fn(batch_pred_missing[:, 1], batch_y[:, 1])
    loss_dict['et_miss_dc_loss']  = loss_fn(batch_pred_missing[:, 2], batch_y[:, 2])

    loss_dict['loss_dc'] = loss_dict['ed_dc_loss'] + loss_dict['net_dc_loss'] + loss_dict['et_dc_loss']
    loss_dict['loss_miss_dc'] = loss_dict['ed_miss_dc_loss'] + loss_dict['net_miss_dc_loss'] + loss_dict['et_miss_dc_loss']

    loss_dict['ed_mse_loss']  = F.mse_loss(batch_pred_full[:, 0], batch_pred_missing[:, 0], reduction='mean')
    loss_dict['net_mse_loss'] = F.mse_loss(batch_pred_full[:, 1], batch_pred_missing[:, 1], reduction='mean')
    loss_dict['et_mse_loss']  = F.mse_loss(batch_pred_full[:, 2], batch_pred_missing[:, 2], reduction='mean')
    loss_dict['consistency_loss'] = loss_dict['ed_mse_loss'] + loss_dict['net_mse_loss'] + loss_dict['et_mse_loss']

    weight_consistency = get_current_consistency_weight(epoch)

    return loss_dict['loss_dc'], loss_dict['loss_miss_dc'], weight_consistency * loss_dict['consistency_loss']


def simple_loss(batch_pred, batch_y):
    loss_dict = {}
    loss_dict['ed_dc_loss']  = dice_loss(batch_pred[:, 0], batch_y[:, 0])
    loss_dict['net_dc_loss'] = dice_loss(batch_pred[:, 1], batch_y[:, 1])
    loss_dict['et_dc_loss']  = dice_loss(batch_pred[:, 2], batch_y[:, 2])
    loss = loss_dict['ed_dc_loss'] + loss_dict['net_dc_loss'] + loss_dict['et_dc_loss']
    return loss


def get_losses():
    losses = {}
    losses['co_loss'] = unet_Co_loss
    return losses


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        prediction = torch.Tensor(prediction)
        target = torch.Tensor(target)
        iflat = prediction.reshape(-1)
        tflat = target.reshape(-1)
        intersection = (iflat * tflat).sum()

        return (2.0 * intersection + self.smooth) / (iflat.sum() + tflat.sum() + self.smooth)


class DiceCELoss3D(nn.Module):
    def __init__(self, weight_ce=0.001, weight_dice=1.0):
        super(DiceCELoss3D, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce_loss = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        if targets.dtype == torch.float16:
            targets = targets.type(torch.float32)
        if preds.dtype == torch.float16:
            preds = preds.type(torch.float32)
        dice = dice_loss(preds, targets)
        ce = self.ce_loss(preds, targets)
        total_loss = self.weight_ce * ce + self.weight_dice * dice
        return total_loss
