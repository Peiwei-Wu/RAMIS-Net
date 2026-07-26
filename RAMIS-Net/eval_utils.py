import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_closing, binary_fill_holes

from losses import DiceLoss
import os
from medpy import metric
from medpy.metric import hd95

criteria = DiceLoss()


def morph_op(msk_pred, j):
    # Morphological closing and hole filling to reduce spurious predictions
    out = msk_pred.copy()
    se = np.ones((j+1, j+1), dtype=bool)

    for z in range(out.shape[-1]):
        sl = out[..., z].astype(bool)
        sl = binary_closing(sl, structure=se)
        sl = binary_fill_holes(sl, structure=se)
        out[..., z] = sl.astype(out.dtype)

    return out

def get_mask(seg_volume, thresh):
    seg_volume = seg_volume.detach().cpu().numpy()
    seg_volume = np.squeeze(seg_volume)

    wt_pred = seg_volume[0]
    tc_pred = seg_volume[1]
    et_pred = seg_volume[2]

    mask = np.zeros_like(wt_pred)
    mask[wt_pred > thresh[0]] = 2
    mask[tc_pred > thresh[1]] = 1
    mask[et_pred > thresh[2]] = 4
    mask = mask.astype("uint8")
    return mask


def eval_dice_metrics(gt, pred, wt_j, ct_j, et_j):
    wt_pred = np.where(pred > 0, 1, 0)
    if (np.sum(wt_pred) > 20) and (wt_j is not None):
        wt_pred = morph_op(wt_pred, wt_j)
    loss_wt = criteria(np.where(gt > 0, 1, 0), wt_pred)

    ct_pred = np.where(pred == 1, 1, 0) + np.where(pred == 4, 1, 0)
    if (np.sum(ct_pred) > 20) and (ct_j is not None):
        ct_pred = morph_op(ct_pred, ct_j)
    loss_ct = criteria(np.where(gt == 1, 1, 0) + np.where(gt == 4, 1, 0), ct_pred)

    et_pred = np.where(pred == 4, 1, 0)
    if (np.sum(et_pred) > 20) and (et_j is not None):
        et_pred = morph_op(et_pred, et_j)
    loss_et = criteria(np.where(gt == 4, 1, 0), et_pred)

    return loss_wt, loss_et, loss_ct


def _pick_best_slice(mask_3d):
    areas = [(mask_3d[..., z] > 0).sum() for z in range(mask_3d.shape[-1])]
    return int(np.argmax(areas)) if max(areas) > 0 else mask_3d.shape[-1] // 2


def measure_dice_score(batch_pred, batch_y, thresh, wt_j=None, ct_j=None, et_j=None,
                       slice_idx=None, save_path=None, save_path_gt=None, base_volume=None, alpha=0.85):
    pred = get_mask(batch_pred,  thresh=thresh)
    gt   = get_mask(batch_y,     thresh=[0.5, 0.5, 0.5])

    loss_wt, loss_et, loss_ct = eval_dice_metrics(gt, pred, wt_j, ct_j, et_j)
    score = (loss_wt + loss_et + loss_ct) / 3.0

    z = _pick_best_slice(pred) if slice_idx is None else int(slice_idx)
    m2d_pred = pred[..., z].astype(np.uint8)
    m2d_gt   = gt[...,   z].astype(np.uint8)

    if base_volume is not None:
        if hasattr(base_volume, "detach"):
            base_volume = base_volume.detach().cpu().numpy()
        img2d = base_volume[..., z].astype(np.float32)
        vmin, vmax = np.percentile(img2d, 1), np.percentile(img2d, 99)
        img2d = np.clip(img2d, vmin, vmax)
        img2d = (img2d - img2d.min()) / (img2d.max() - img2d.min() + 1e-8)
    else:
        img2d = np.zeros_like(m2d_pred, dtype=np.float32)

    def make_overlay(m2d):
        H, W = m2d.shape
        ov = np.zeros((H, W, 4), dtype=np.float32)

        for cls, rgb in [(2,(0,0,1)), (4,(1,0,0)), (1,(0,1,0))]:
            mask = (m2d == cls)
            ov[mask, :3] = rgb
            ov[mask, 3]  = 1.0

        return ov

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(5,6))
        plt.imshow(img2d, cmap="gray", interpolation='nearest')
        plt.imshow(make_overlay(m2d_pred), alpha=alpha, interpolation='nearest')
        for cls, color in [(2,'blue'), (4,'red'), (1,'green')]:
            plt.contour((m2d_pred==cls).astype(np.uint8), levels=[0.5], colors=[color], linewidths=1.5)
        plt.axis('off')
        plt.savefig(save_path, dpi=220, bbox_inches='tight', pad_inches=0)
        plt.close()

    if save_path_gt is not None:
        os.makedirs(os.path.dirname(save_path_gt), exist_ok=True)
        plt.figure(figsize=(5,6))
        plt.imshow(img2d, cmap="gray", interpolation='nearest')
        plt.imshow(make_overlay(m2d_gt), alpha=alpha, interpolation='nearest')

        for cls, color in [(2,'blue'), (4,'red'), (1,'green')]:
            plt.contour((m2d_gt==cls).astype(np.uint8), levels=[0.5], colors=[color], linewidths=1.5)
        plt.axis('off')
        plt.savefig(save_path_gt, dpi=220, bbox_inches='tight', pad_inches=0)
        plt.close()

    return score, loss_wt, loss_et, loss_ct


def save_modal_slices(batch_x, z, save_dir="./"):
    os.makedirs(save_dir, exist_ok=True)

    for c in range(batch_x.shape[1]):
        img2d = batch_x[0, c, :, :, z]
        img2d = img2d.detach().cpu().numpy()

        plt.figure(figsize=(6.4, 5.12), dpi=220)
        plt.imshow(img2d, cmap="gray")
        plt.axis("off")
        save_path = os.path.join(save_dir, f"init{c+1}.png")
        plt.savefig(save_path, dpi=220, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"Saved {save_path}")


def eval_hd95_metrics(gt, pred, wt_j, ct_j, et_j):
    wt_pred = np.where(pred > 0, 1, 0)
    wt_pred = morph_op(wt_pred, wt_j)
    hd95_WT = compute_BraTS_HD95(np.where(gt > 0, 1, 0), wt_pred)

    tc_pred = np.where(pred == 1, 1, 0) + np.where(pred == 4, 1, 0)
    tc_pred = morph_op(tc_pred, ct_j)
    hd95_TC = compute_BraTS_HD95(np.where(gt == 1, 1, 0) + np.where(gt == 4, 1, 0), tc_pred)

    et_pred = np.where(pred == 4, 1, 0)
    if (np.sum(et_pred) > 20) and (et_j is not None):
        et_pred = morph_op(et_pred, et_j)
    hd95_ET = compute_BraTS_HD95(np.where(gt == 4, 1, 0), et_pred)

    return hd95_WT, hd95_TC, hd95_ET


def compute_BraTS_HD95(ref, pred):
    """Hausdorff distance"""
    num_ref = np.sum(ref)
    num_pred = np.sum(pred)
    if num_ref == 0:
        if num_pred == 0:
            return 0
        else:
            return 373.12866
    elif num_pred == 0 and num_ref != 0:
        return 373.12866
    else:
        return hd95(pred, ref, (1, 1, 1))


def cal_hd95(batch_pred, batch_y, thresh, wt_j=None, ct_j=None, et_j=None, slice_idx=None):
    pred = get_mask(batch_pred, thresh=thresh)
    gt   = get_mask(batch_y, thresh=[0.5, 0.5, 0.5])

    hd95_WT, hd95_TC, hd95_ET = eval_hd95_metrics(gt=gt, pred=pred, wt_j=wt_j, ct_j=ct_j, et_j=et_j)
    avg_hd95 = (hd95_WT + hd95_TC + hd95_ET) / 3

    return avg_hd95, hd95_WT, hd95_TC, hd95_ET
