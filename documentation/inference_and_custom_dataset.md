# Inference & Custom Dataset Guide

## Running Inference

### Prerequisites

#### 1. Download the Dataset

The project uses the **AvatarReX** dataset. Download the subject you want:

| Subject | Frames | Size | Link |
|---|---|---|---|
| avatarrex_zzr | 2001 | ~21 GB | [Google Drive](https://drive.google.com/file/d/1sCQJ3YU-F3lY9p_HYNIQbT7QyfVKy0HT/view?usp=sharing) |
| avatarrex_zxc | 1801 | ~12 GB | [Google Drive](https://drive.google.com/file/d/1pY1qRj2n6b2YOCmZRVM1D--CXKR02qXU/view?usp=sharing) |
| avatarrex_lbn1 | 1901 | ~11 GB | [Google Drive](https://drive.google.com/file/d/1DuESdA5YwvJKapyo7i_KoQxKHHFWzi-w/view?usp=sharing) |
| avatarrex_lbn2 | 1871 | ~16 GB | [Google Drive](https://drive.google.com/file/d/1J7ITsYhuWlqhoIkmYni8dL2KJw-wmcy_/view?usp=sharing) |

> **License**: The AvatarReX dataset is for **non-commercial research only**. By downloading you agree to the terms in `AVATARREX_DATASET.md`.

Each downloaded archive contains, per subject:
```
avatarrex_{subject}/
├── {cam_name}/                   # one folder per camera (16 cameras total)
│   ├── 00000000.jpg              # RGB frames at 1500×2048, 30 fps
│   └── mask/pha/00000000.jpg     # foreground alpha masks
├── calibration_full.json         # camera calibration for all 16 cameras
├── smpl_params.npz               # SMPL-X body pose fits for all frames
└── missing_img_files.txt         # list of frames lost during capture
```

#### 2. Two Key Files Required for Inference

Both files below come with the downloaded dataset and are **required** even when running inference with new poses.

---

**`smpl_params.npz`** — SMPL-X body model parameters

This file encodes the subject's body shape and every recorded motion as compact numerical vectors. It contains:

| Key | Shape | Description |
|---|---|---|
| `betas` | `(1, 10)` | **Body shape coefficients** — define the subject's proportions (height, weight, limb lengths). These are the same for all frames. |
| `global_orient` | `(N, 3)` | Root rotation per frame, as an axis-angle vector |
| `body_pose` | `(N, 63)` | 21 joint rotations per frame (axis-angle, 3 values each) |
| `transl` | `(N, 3)` | Root translation in world space (meters) per frame |
| `jaw_pose` | `(N, 3)` | Jaw rotation per frame |
| `expression` | `(N, 10)` | Facial expression coefficients per frame |
| `left_hand_pose` | `(N, 45)` | 15 hand joint rotations per frame |
| `right_hand_pose` | `(N, 45)` | 15 hand joint rotations per frame |

During inference the `betas` field is used to reconstruct the subject's canonical body mesh, which is the foundation all Gaussians are placed on. Without it the system cannot produce the correct body shape.

---

**`calibration_full.json`** — Camera calibration data

This file defines where each of the 16 cameras is positioned and how they project 3D points onto 2D images. It contains one entry per camera:

```json
{
  "cam_name": {
    "R": [[...], [...], [...]],   // 3×3 rotation matrix — camera orientation in world space
    "T": [tx, ty, tz],            // 3D translation — camera position in world space (meters)
    "K": [[fx,0,cx],[0,fy,cy],[0,0,1]],  // intrinsic matrix — focal length and principal point
    "H": 2048,                    // image height in pixels
    "W": 1500                     // image width in pixels
  }
}
```

During inference this file is used when `view_setting: camera` is set, so the renderer knows the exact camera frustum to render from. It is also needed during training to reproject rendered Gaussians back onto each training image.

---

#### 3. Preprocessed Data (Optional Shortcut)

Running the two preprocessing scripts (`gen_weight_volume`, `gen_pos_maps`) can take significant time. Pre-processed versions for some subjects are available:

| Subject | Link |
|---|---|
| avatarrex_zzr | [Google Drive](https://drive.google.com/file/d/1o5tIisBAhYxCl81SUZ4HGaEKyslCBD16/view?usp=sharing) |
| avatarrex_lbn1 | [Google Drive](https://drive.google.com/file/d/1RDM3v5P4XF6Sp88EusDvokw-yHg6Je0C/view?usp=sharing) |
| avatarrex_lbn2 | [Google Drive](https://drive.google.com/file/d/1AuITI1KDHG4MbaNplnzmkcYDwii_Q419/view?usp=sharing) |

Place the extracted contents inside the subject's data folder alongside the raw dataset files.

#### 4. Pretrained Checkpoints (Skip Training)

To run inference without training from scratch, download a pretrained model:

| Subject | Link |
|---|---|
| avatarrex_zzr | [Google Drive](https://drive.google.com/file/d/1lR_O9m0J_lwc8POA_UtCDM9LsTWOIu4m/view?usp=sharing) |
| avatarrex_lbn1 | [Google Drive](https://drive.google.com/file/d/1P-s-RcJ5_Z7ZVSzjjl-xhPCExqN8td7S/view?usp=sharing) |
| avatarrex_lbn2 | [Google Drive](https://drive.google.com/file/d/1KakiePoLpV3Wa0QGtnzrt8MAhZbNQi6n/view?usp=sharing) |

Extract each checkpoint under `results/{subject}/avatar/` and point `prev_ckpt` in your config to it.

---

#### 5. Trained Checkpoint (if training yourself)

After training, checkpoints are saved to:
```
results/{subject}/avatar/batch_XXXXXX/
```

### Basic Test Command

```bash
python main_avatar.py -c configs/{subject}/avatar.yaml --mode=test
```

### Test Configuration

In your `avatar.yaml`, the `test:` section controls inference behavior:

```yaml
test:
  dataset: MvRgbDatasetAvatarReX     # same dataset class used for training
  data:
    data_dir: /path/to/subject/data
    subject_name: my_subject
    frame_range: [0, 500]            # which frames to render

  # Optional: drive with NEW poses (otherwise replays training frames)
  pose_data:
    data_path: /path/to/poses.npz
    frame_range: [0, 300]

  prev_ckpt: ./results/my_subject/avatar/batch_700000   # checkpoint to load

  # Camera view options: camera | free | free_bird | front | back | moving | cano
  view_setting: free
  render_view_idx: 0        # used only when view_setting: camera
  img_scale: 1.0            # render resolution multiplier

  # Optional outputs (all default false)
  render_skeleton: false
  save_tex_map: false
  save_ply: false

  # Pose PCA (helps generalize to unseen poses; -1 to disable)
  n_pca: 20
  sigma_pca: 2.0
```

### Output Structure

Results are written to:
```
test_results/{subject}/{ckpt_name}/{dataset}_{view_setting}/batch_{N}/
├── rgb_map/         # Rendered RGB frames (8-bit JPG)
├── mask_map/        # Alpha masks (PNG)
├── live_skeleton/   # Skeleton overlays (if render_skeleton: true)
├── cano_tex_map/    # Canonical texture maps (if save_tex_map: true)
└── posed_gaussians/ # Per-frame Gaussian PLYs (if save_ply: true)
```

### View Settings

| Setting | Description |
|---|---|
| `camera` | Render from a specific training camera (`render_view_idx`) |
| `free` | 360° orbit around the subject |
| `free_bird` | 360° orbit from an elevated angle |
| `front` / `back` | Fixed front or back view |
| `moving` | Camera follows subject position |
| `cano` | Render in canonical (T-pose) space |

### Animating with Custom Poses

Set `pose_data.data_path` to a pose file. Supported formats:

**`.npz` (SMPL-X format)**:
```
body_pose:        (N, 63)
global_orient:    (N, 3)
transl:           (N, 3)
betas:            (1, 10)
jaw_pose:         (N, 3)       # optional
expression:       (N, 10)      # optional
left_hand_pose:   (N, 45)      # optional
right_hand_pose:  (N, 45)      # optional
```

**AMASS `.npz`** (detected automatically):
```
poses:   (N, 156)   # [global_orient(3), body_pose(66), hands(90)]
trans:   (N, 3)
```

**AIST++ `.pkl`**: keys `smpl_poses`, `smpl_trans`

---

## Using Your Own Dataset

### What You Need to Implement / Prepare

#### 1. Capture Setup Requirements

- **Multi-view RGB video**: minimum ~4 cameras recommended; the more the better
- **Synchronized frames** across all cameras
- **Static background** or background matting for masks
- **Camera calibration**: intrinsics (fx, fy, cx, cy) and extrinsics (R, T) per camera

#### 2. SMPL-X Parameter Fitting

For every frame, you need SMPL-X body pose parameters. Use an off-the-shelf fitter such as:
- **PyMAF-X** — multi-view aware, recommended
- **PIXIE** — expressive body + face + hands
- **OSX / ExPose** — handles hands and face well

The fitter output must be saved as `smpl_params.npz`:

```python
import numpy as np
np.savez('smpl_params.npz',
    betas           = np.zeros((1, 10)),    # shape params (same across all frames)
    global_orient   = np.zeros((N, 3)),     # root rotation, axis-angle
    body_pose       = np.zeros((N, 63)),    # 21 joints x 3, axis-angle
    transl          = np.zeros((N, 3)),     # root translation (meters)
    jaw_pose        = np.zeros((N, 3)),     # jaw rotation
    expression      = np.zeros((N, 10)),    # face expression
    left_hand_pose  = np.zeros((N, 45)),    # 15 joints x 3
    right_hand_pose = np.zeros((N, 45)),
)
```

#### 3. Data Directory Structure

```
data/my_subject/
├── smpl_params.npz
├── calibration.json          # camera params (see format below)
├── images/
│   ├── cam00/
│   │   ├── 00000000.jpg
│   │   └── ...
│   ├── cam01/
│   │   └── ...
│   └── ...
└── masks/
    ├── cam00/
    │   ├── 00000000.png      # binary alpha mask, 0 or 255
    │   └── ...
    └── ...
```

Calibration JSON format (THuman4 style):
```json
{
  "cam00": {
    "R": [[...], [...], [...]],
    "T": [[tx], [ty], [tz]],
    "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "H": 1080,
    "W": 1920
  },
  "cam01": { "..." }
}
```

- `R`: 3×3 rotation matrix (world → camera)
- `T`: 3×1 translation vector in meters
- `K`: 3×3 camera intrinsic matrix

#### 4. Implement a Dataset Class

Copy and modify `MvRgbDatasetTHuman4` (the simplest existing implementation) in `dataset/dataset_mv_rgb.py`:

```python
class MvRgbDatasetCustom(MvRgbDatasetBase):

    def load_cam_data(self):
        """Parse calibration.json and populate required camera arrays."""
        import json
        with open(os.path.join(self.data_dir, 'calibration.json')) as f:
            calib = json.load(f)

        self.cam_names = sorted(calib.keys())
        self.view_num = len(self.cam_names)
        self.extr_mats, self.intr_mats = [], []
        self.img_heights, self.img_widths = [], []

        for name in self.cam_names:
            c = calib[name]
            R = np.array(c['R'], dtype=np.float32)
            T = np.array(c['T'], dtype=np.float32).reshape(3, 1)
            extr = np.eye(4, dtype=np.float32)
            extr[:3, :3] = R
            extr[:3, 3:] = T
            self.extr_mats.append(extr)
            self.intr_mats.append(np.array(c['K'], dtype=np.float32))
            self.img_heights.append(c['H'])
            self.img_widths.append(c['W'])

    def load_color_mask_images(self, pose_idx, view_idx):
        """Return (color_img, mask_img) as uint8 HxWx3 and HxW arrays."""
        cam = self.cam_names[view_idx]
        img_path  = os.path.join(self.data_dir, 'images', cam, f'{pose_idx:08d}.jpg')
        mask_path = os.path.join(self.data_dir, 'masks',  cam, f'{pose_idx:08d}.png')
        color_img = cv2.imread(img_path)[..., ::-1]           # BGR -> RGB
        mask_img  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return color_img, mask_img
```

Then make sure the class is importable from wherever `main_avatar.py` resolves dataset names (look for `eval(cfg.dataset)` or a dict lookup in that file).

#### 5. Run Preprocessing

```bash
# Step 1 — LBS weight volume (reads betas from smpl_params.npz)
python -m gen_data.gen_weight_volume -c configs/my_subject/template.yaml

# Step 2 — Per-frame SMPL position maps (reads all frames from smpl_params.npz)
python -m gen_data.gen_pos_maps -c configs/my_subject/avatar.yaml
```

After this, your data directory will contain:
```
data/my_subject/
├── cano_weight_volume.npz
└── smpl_pos_map/
    ├── cano_smpl_pos_map.exr
    ├── cano_smpl_nml_map.exr
    ├── init_pts_lbs.npy
    ├── 00000000.exr
    └── ...
```

> **Linux note**: `gen_weight_volume.py` internally calls `PointInterpolant.exe`, a Windows binary
> for RBF interpolation. On Linux you need to replace this call with a compatible solver, for
> example `scipy.interpolate.RBFInterpolator`, or build the binary from its source.

#### 6. Create Config Files

`configs/my_subject/template.yaml` (used only by `gen_weight_volume`):
```yaml
data:
  subject_name: my_subject
  data_dir: ./data/my_subject
```

`configs/my_subject/avatar.yaml`:
```yaml
mode: train

train:
  dataset: MvRgbDatasetCustom
  data:
    subject_name: my_subject
    data_dir: ./data/my_subject
    frame_range: [0, 1000, 1]         # adjust to your total frame count
    used_cam_ids: [0, 1, 2, 3]        # indices into cam_names list
    load_smpl_pos_map: true
  net_ckpt_dir: ./results/my_subject/avatar
  lr_init: 0.0005
  loss_weight:
    l1: 1.0
    lpips: 0.1
    offset: 0.005
  random_bg_color: true
  num_workers: 4

test:
  dataset: MvRgbDatasetCustom
  data:
    data_dir: ./data/my_subject
    frame_range: [0, 100]
    subject_name: my_subject
  view_setting: free
  img_scale: 1.0
  n_pca: 20
  prev_ckpt: ./results/my_subject/avatar/batch_700000

model:
  with_viewdirs: true
  random_style: false
```

#### 7. Train and Test

```bash
# Train (~800k iterations)
python main_avatar.py -c configs/my_subject/avatar.yaml --mode=train

# Render / animate
python main_avatar.py -c configs/my_subject/avatar.yaml --mode=test
```

---

### Summary Checklist

| Step | What to do |
|---|---|
| Capture | Multi-view synchronized video + background matting |
| Calibrate cameras | Per-camera intrinsics + extrinsics (R, T) → `calibration.json` |
| Fit SMPL-X | Run PyMAF-X / PIXIE on your footage → `smpl_params.npz` |
| Organize files | `images/camXX/`, `masks/camXX/`, `calibration.json` |
| Implement dataset class | Subclass `MvRgbDatasetBase`, implement `load_cam_data` + `load_color_mask_images` |
| Preprocess | `gen_weight_volume` then `gen_pos_maps` |
| Write config | `avatar.yaml` pointing to your data and class |
| Train | `--mode=train` (~800k iterations) |
| Animate | `--mode=test` with optional `pose_data` for new motion sequences |
