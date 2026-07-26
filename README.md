# RAMIS-Net: Multi-Modal Medical Image Segmentation

A PyTorch implementation of RAMIS-Net for medical image segmentation with support for missing modalities. This network combines Mamba blocks with transformer-based attention mechanisms for robust segmentation of brain tumors in multi-modal MRI data.

## Overview

RAMIS-Net is designed to handle incomplete multi-modal medical imaging datasets by learning from both complete and incomplete modality combinations. The architecture leverages:

- **Mamba Blocks**: State-space models for efficient sequence processing
- **Rotary Position Embedding (RoPE)**: For improved spatial encoding
- **Multi-stage Encoder-Decoder**: With skip connections for fine-grained segmentation
- **Linear Attention**: For efficient long-range dependencies

## Features

- Support for multi-modal MRI data (FLAIR, T1, T1ce, T2)
- Handles missing modalities gracefully
- Modular architecture for easy customization
- Training with BraTS 2018/2020 datasets
- Comprehensive evaluation metrics (Dice, HD95)

## Project Structure

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

## Requirements

```
torch>=1.13.0
torchvision>=0.14.0
numpy
nibabel
einops
timm
mamba-ssm
monai
scipy
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

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

- `--task_name`: Project name (default: RAMIS-Net)
- `--datasets`: Dataset type - 'BRATS 2018' or 'BRATS 2020'
- `--path_to_data`: Path to dataset directory
- `--n_epochs`: Number of training epochs (default: 300)
- `--batch_size_tr`: Training batch size
- `--batch_size_va`: Validation batch size
- `--lr`: Learning rate (default: 0.00025)
- `--modalities`: Input modalities (default: flair t1 t1ce t2)

### Evaluation

```bash
python validate.py \
    --saved_model_path ./new_results/RAMIS-Net/weights/best_model_weights.pth \
    --datasets BRATS\ 2020 \
    --path_to_data ./datasets/MICCAI_BraTS2020_TrainingData/
```

## Data Format

Expected data directory structure:
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

Labels in segmentation files:
- 0: Background
- 1: Necrotic/Non-enhancing tumor (NCR/NET)
- 2: Peritumoral edema (ED)
- 4: Enhancing tumor (ET)

## Model Architecture

### Encoder
- Multi-stage patch embedding with progressively increasing receptive fields
- Combination of efficient transformers and Mamba blocks
- Skip connections for feature preservation

### Decoder
- Progressive upsampling with transformer-based refinement
- Feature fusion from encoder skip connections
- Final segmentation head with 4 output channels

## Training Details

- **Optimizer**: AdamW with weight decay 1e-5
- **Scheduler**: Polynomial learning rate decay (power=0.75)
- **Loss Function**: Weighted combination of:
  - Dice loss (full + missing modality branches)
  - Context alignment loss (RARA)
  - Token consistency loss (MISA)
  - Reconstruction loss
  - Consistency loss between full and missing branches

- **Input Size**: 128 × 160 × 192
- **Random Seed**: 3407 (fixed for reproducibility)

## Output

After training, the model saves:
- `./new_results/RAMIS-Net/weights/model_weights.pth` - Latest checkpoint
- `./new_results/RAMIS-Net/weights/best_model_weights.pth` - Best validation model
- `./new_results/RAMIS-Net/[timestamp]_log.txt` - Training logs

## Citation

If you use RAMIS-Net in your research, please cite this work:

```bibtex
@article{ramis-net,
  title={RAMIS-Net: Medical Image Segmentation with Handling Missing Modalities},
  year={2026}
}
```

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/)
- Uses [Mamba](https://github.com/state-spaces/mamba) state-space models
- Inspired by [timm](https://github.com/rwightman/pytorch-image-models) library
- [MONAI](https://monai.io/) for medical imaging utilities

## Support

For issues and questions, please refer to the project documentation or contact the authors.

---

**Last Updated**: July 2026  
**Version**: 1.0
