import torch
import nibabel as nib
import glob
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

class BraTS(Dataset):

    def __init__(self, patients_dir, crop_size, modes, train=True, normalization = True):
        self.patients_dir = patients_dir
        self.modes = modes
        self.train = train
        self.crop_size = crop_size
        self.normalization = normalization

    def __len__(self):
        return len(self.patients_dir)

    def __getitem__(self, index):
        patient_dir = self.patients_dir[index]
        patient_id = os.path.split(patient_dir)[-1]

        modes = list(self.modes) + ["seg"]
        paths = []
        for mode in modes:
            p = os.path.join(patient_dir, f"{patient_id}_{mode}.nii")
            if not os.path.exists(p):
                return self.__getitem__((index + 1) % len(self.patients_dir))
            paths.append(p)

        volumes = []
        for mode, p in zip(modes, paths):
            vol = nib.load(p).get_fdata()
            if mode != "seg" and self.normalization:
                vol = self.normlize(vol)
            volumes.append(vol)

        seg_volume = volumes[-1]
        volumes = volumes[:-1]
        volume, seg_volume = self.aug_sample(volumes, seg_volume)
        # Labels: 0=background, 1=necrotic/non-enhanced, 2=peritumoral edema, 4=enhancing tumor
        bg_volume = (seg_volume == 0)
        net_volume= (seg_volume == 1)
        ed_volume = (seg_volume == 2)
        et_volume = (seg_volume == 4)

        seg_volume = [ed_volume, net_volume, et_volume, bg_volume]
        seg_volume = np.concatenate(seg_volume, axis=0).astype("float32")

        return (torch.tensor(volume.copy(), dtype=torch.float),
                torch.tensor(seg_volume.copy(), dtype=torch.float))


    def aug_sample(self, volumes, mask):
        """Prepare batch (N, H, W, D) -> (channel, h, w, d)"""
        x = np.stack(volumes, axis=0)
        y = np.expand_dims(mask, axis=0)

        if self.train:
            x, y = self.random_crop(x, y)
            if random.random() < 0.5:
                x = np.flip(x, axis=1)
                y = np.flip(y, axis=1)
            if random.random() < 0.5:
                x = np.flip(x, axis=2)
                y = np.flip(y, axis=2)
            if random.random() < 0.5:
                x = np.flip(x, axis=3)
                y = np.flip(y, axis=3)
        else:
            x, y = self.center_crop(x, y)

        return x, y

    def random_crop(self, x, y):
        crop_size = self.crop_size
        height, width, depth = x.shape[-3:]
        sx = random.randint(0, height - crop_size[0] - 1)
        sy = random.randint(0, width - crop_size[1] - 1)
        sz = random.randint(0, depth - crop_size[2] - 1)
        crop_volume = x[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]
        crop_seg = y[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]

        return crop_volume, crop_seg

    def center_crop(self, x, y):
        crop_size = self.crop_size
        height, width, depth = x.shape[-3:]
        sx = (height - crop_size[0] - 1) // 2
        sy = (width - crop_size[1] - 1) // 2
        sz = (depth - crop_size[2] - 1) // 2
        crop_volume = x[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]
        crop_seg = y[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]

        return crop_volume, crop_seg

    def normlize(self, x):
        return (x - x.min()) / (x.max() - x.min())


    def normlize_brain(self, x, epsilon=1e-8):
        average        = x[np.nonzero(x)].mean()
        std            = x[np.nonzero(x)].std() + epsilon
        mask           = x>0
        sub_mean       = np.where(mask, x-average, x)
        x_normalized   = np.where(mask, sub_mean/std, x)
        return x_normalized


def split_dataset(data_root, validation_p, test_p, datasets_type, seed=42):
    if datasets_type == 'BRATS 2018':
        patients_dir = glob.glob(os.path.join(data_root, "*GG", "Brats18*"))
    elif datasets_type == 'BRATS 2020':
        patients_dir = glob.glob(os.path.join(data_root, "BraTS20*"))
    else:
        raise ValueError(f"Unknown dataset type: {datasets_type}")

    random.seed(seed)
    np.random.seed(seed)
    random.shuffle(patients_dir)

    N_test = int(len(patients_dir) * test_p)
    N_val = int(len(patients_dir) * validation_p)

    # Split dataset: test | val | train
    test_patients_list = patients_dir[:N_test]
    val_patients_list = patients_dir[N_test:N_test + N_val]
    train_patients_list = patients_dir[N_test + N_val:]

    return train_patients_list, val_patients_list, test_patients_list


def make_data_loaders(config):
    train_list, val_list, test_list = split_dataset(config.path_to_data, float(config.validation_p), float(config.test_p), config.datasets)
    crop_size = np.zeros((3))
    crop_size[0] = config.inputshape[0]
    crop_size[1] = config.inputshape[1]
    crop_size[2] = config.inputshape[2]
    crop_size    = crop_size.astype(np.uint16)
    crop_size    = (160, 192, 128)

    train_ds = BraTS(train_list, crop_size=crop_size, modes=config.modalities, train=True,  normalization = True)
    val_ds   = BraTS(val_list,   crop_size=crop_size, modes=config.modalities, train=False, normalization = True)
    test_ds  = BraTS(test_list,  crop_size=crop_size, modes=config.modalities, train=False, normalization = True)

    loaders = {}
    loaders['train'] = DataLoader(train_ds, batch_size=int(config.batch_size_tr),
                                  num_workers=4,
                                  pin_memory=True,
                                  shuffle=True)
    loaders['eval'] = DataLoader(val_ds, batch_size=int(config.batch_size_va),
                                  num_workers=4,
                                  pin_memory=True,
                                  shuffle=False)
    loaders['test'] = DataLoader(test_ds, batch_size=int(config.batch_size_tr),
                                  num_workers=4,
                                  pin_memory=True,
                                  shuffle=True)
    
    return loaders
