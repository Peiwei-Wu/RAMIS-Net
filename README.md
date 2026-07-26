<!-- 徽章区域（项目状态、版本、依赖等） -->
<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-1.12%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/Version-1.0-blue" alt="Version">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" alt="Status">
</p>

# 🧠 RAMIS-Net: Hierarchical collaborative distillation for brain tumor segmentation with incomplete modality

> *A PyTorch implementation of RAMIS‑Net for medical image segmentation with support for missing modalities.*  
> RAMIS‑Net improves brain tumor segmentation under incomplete MRI conditions, especially in the **TC** and **ET** regions, by hierarchically distilling complete modality knowledge into missing modality paths through **RARA**, **MISA**, and task‑level alignment.

---

## 📷 Overview

<img width="1015" height="619" alt="image" src="https://github.com/user-attachments/assets/02b99604-1879-40f7-9970-31418a740725" />

---

## 📂 Project Structure

```
RAMIS-Net/
├── configs/
│   ├── __init__.py
│   └── args.py                 # Training arguments configuration
├── models/
│   ├── __init__.py
│   ├── base_modules.py         # Foundational layers (conv, FFN, etc.)
│   ├── mamba_blocks.py         # Mamba-based components
│   ├── encoder.py              # Multi-stage encoder
│   ├── decoder.py              # Decoder layers
│   ├── linear_former.py        # RoPE and attention mechanisms
│   └── ramis_net.py            # Main model
├── functions/
│   ├── adamw.py               # AdamW optimizer
│   └── DropPath.py            # Stochastic depth
├── train.py                    # Training script
├── dataset.py                  # Data loading and preprocessing
├── losses.py                   # Loss functions
├── utils.py                    # Utility functions
├── eval_utils.py              # Evaluation metrics
└── validate.py                # Validation script
```

---

## ⚙️ Installation & Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Training

```bash
python train.py \
    --task_name RAMIS-Net \
    --path_to_data ./datasets/MICCAI_BraTS2020_TrainingData/ \
    --datasets BRATS\ 2020 \
    --n_epochs 300 \
    --batch_size_tr 1 \
    --batch_size_va 1 \
    --lr 0.00025
```

### Configuration

All training arguments are defined in `configs/args.py`. Key parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--task_name` | Project name | `RAMIS-Net` |
| `--datasets` | Dataset type – `'BRATS 2018'` or `'BRATS 2020'` | – |
| `--path_to_data` | Path to dataset directory | – |
| `--n_epochs` | Number of training epochs | `300` |
| `--batch_size_tr` | Training batch size | – |
| `--batch_size_va` | Validation batch size | – |
| `--lr` | Learning rate | `0.00025` |
| `--modalities` | Input modalities | `flair t1 t1ce t2` |

### Evaluation

```bash
python validate.py \
    --saved_model_path ./new_results/RAMIS-Net/weights/best_model_weights.pth \
    --datasets BRATS\ 2020 \
    --path_to_data ./datasets/MICCAI_BraTS2020_TrainingData/
```

---

## 🗂️ Data Format

Expected directory structure:

```
MICCAI_BraTS2020_TrainingData/
├── BraTS20_Training_001/
│   ├── BraTS20_Training_001_flair.nii
│   ├── BraTS20_Training_001_t1.nii
│   ├── BraTS20_Training_001_t1ce.nii
│   ├── BraTS20_Training_001_t2.nii
│   └── BraTS20_Training_001_seg.nii
├── BraTS20_Training_002/
└── ...
```

**Label mapping** in segmentation files:

- `0` – Background
- `1` – Necrotic/Non‑enhancing tumor (NCR/NET)
- `2` – Peritumoral edema (ED)
- `4` – Enhancing tumor (ET)

---

## 📊 Output

After training, the model saves the following files:

- `./new_results/RAMIS-Net/weights/model_weights.pth` – Latest checkpoint  
- `./new_results/RAMIS-Net/weights/best_model_weights.pth` – Best validation model  
- `./new_results/RAMIS-Net/[timestamp]_log.txt` – Training logs

---

## 📄 License

This project is open source and available under the **MIT License**.

---

## 🤝 Acknowledgments

- Built with [PyTorch](https://pytorch.org/)
- Uses [Mamba](https://github.com/state-spaces/mamba) state‑space models
- Inspired by the [timm](https://github.com/rwightman/pytorch-image-models) library
- [MONAI](https://monai.io/) for medical imaging utilities

---

## 📬 Support

For issues and questions, please refer to the project documentation or contact the authors.

---

**Last Updated**: July 2026  
**Version**: 1.0
