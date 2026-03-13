# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Animatable Gaussians** (CVPR 2024) — a human avatar modeling system that combines 2D StyleGAN-based CNNs with 3D Gaussian splatting. Given multi-view RGB video and SMPL-X body model fits, it learns pose-dependent Gaussian attributes for animatable, high-fidelity avatars.

## Setup

```bash
pip install -r requirements.txt
# Install custom CUDA modules:
pip install gaussians/diff_gaussian_rasterization_depth_alpha/
pip install network/styleunet/
# Download SMPL-X model files → ./smpl_files/smplx/
```

## Key Commands

**Data Preprocessing** (run in order):
```bash
python -m gen_data.gen_weight_volume -c configs/{subject}/template.yaml
python -m gen_data.gen_pos_maps -c configs/{subject}/avatar.yaml
```

**Template reconstruction** (optional, for loose garments):
```bash
python main_template.py -c configs/{subject}/template.yaml
```

**Avatar training:**
```bash
python main_avatar.py -c configs/avatarrex_zzr/avatar.yaml --mode=train
```

**Avatar testing/animation:**
```bash
python main_avatar.py -c configs/avatarrex_zzr/avatar.yaml --mode=test
```

## Architecture

### Data Flow

```
Multi-view RGB + SMPL-X fits
    → Preprocess (gen_weight_volume, gen_pos_maps)
    → AvatarNet training (800k iterations)
    → Test: load poses → render animations
```

### Core Components

**`network/avatar.py` — `AvatarNet`**: Central model. Maintains a canonical `GaussianModel` initialized from the SMPL mesh. Three `DualStyleUNet` modules generate pose-dependent attributes:
- `color_net` → per-Gaussian RGB
- `position_net` → per-Gaussian position offsets
- `other_net` → per-Gaussian opacity, scale, rotation

The forward flow: `get_pose_map()` → project SMPL body to 2D front/back maps → pass through StyleUNets → `transform_cano2live()` applies LBS deformation → render with custom CUDA rasterizer.

**`gaussians/gaussian_model.py` — `GaussianModel`**: Stores 3D Gaussian primitives (`_xyz`, `_features_dc`, `_scaling`, `_rotation`, `_opacity`) with activation functions (exp for scale, sigmoid for opacity). Serializes to/from PLY.

**`gaussians/gaussian_renderer.py` — `render3()`**: Custom CUDA-accelerated rasterization via `diff_gaussian_rasterization_depth_alpha`. Outputs RGB + depth + alpha.

**`main_avatar.py` — `AvatarTrainer`**: Orchestrates training phases:
1. **Pretrain**: Initialize Gaussians to match canonical SMPL
2. **Main training**: 800k iterations, random background, image cropping after 300k iter
3. **Finetune** (optional): Color network only for pose generalization
- Loss: L1 + LPIPS + offset regularization
- LR: Adam with cosine annealing from `lr_init` (default 5e-4)
- Checkpoints: every 50k iterations → `./results/{subject}/avatar/`

**`dataset/`**: `MvRgbDatasetAvatarReX` loads multi-view RGB + masks + calibration + SMPL params. `PoseDataset` loads animation sequences for testing. Supports AvatarReX, ActorsHQ, THuman4.0, AMASS.

**`network/styleunet/`**: StyleGAN2-based U-Net used for conditional Gaussian map generation. Takes pose maps as input, outputs per-pixel Gaussian attributes.

**`config.py`**: Loads YAML configs and merges with CLI arguments. All training hyperparameters live in per-subject YAML files under `configs/`.

### 2D Canonical Map Parameterization

Gaussians are stored as 2D front/back maps aligned with the SMPL UV parameterization. Each pixel corresponds to one 3D Gaussian. This enables use of 2D CNNs (StyleUNet) for generating pose-dependent attributes while maintaining 3D spatial consistency via LBS.

### Supported Datasets

- **AvatarReX**: 4 subjects, 16 RGB cameras, 1500×2048 — see `AVATARREX_DATASET.md`
- **ActorsHQ**: configs in `configs/actor01/` through `configs/actor08/`
- **THuman4.0**: configs in `configs/subject00/` through `configs/subject02/`
- **AMASS**: pose sequences for animation testing (`configs/awesome_amass_poses.yaml`)
