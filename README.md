# 📌 Evaluating Hybrid Visual Odometry Under Scene Dynamics

A comprehensive benchmarking framework for evaluating classical, deep, and hybrid visual odometry (VO) pipelines under varying levels of scene dynamics.

## 📋 Project Overview

Visual Odometry (VO) estimates camera motion from image sequences. While classical geometry-based pipelines (e.g., ORB-SLAM) excel in static environments, they struggle when dynamic objects dominate the scene. Deep learning components (SuperPoint, DISK, SuperGlue, LightGlue) offer improved robustness but often overfit to training domains.

This project systematically evaluates **classical, deep, and hybrid VO pipelines** across **indoor and outdoor datasets** with **controlled scene dynamics** to understand:
- How hybrid systems behave as scene dynamics increase
- Which components contribute most to robustness
- Trade-offs between accuracy, robustness, and runtime

### Research Question
👉 *How do hybrid VO systems behave as scene dynamics increase, and which components contribute most to robustness?*

### Project Goals
- ✅ Evaluate different VO pipelines (classical, deep, hybrid) under controlled scene dynamics
- ✅ Compare detectors: **ORB**, **SuperPoint**, **DISK**
- ✅ Compare matchers: **kNN**, **SuperGlue**, **LightGlue**
- ✅ Assess dynamic-object masking: **optical flow** (classical) vs **Fast-SCNN** (deep)
- ✅ Quantify robustness using **ATE**, **RPE**, **inlier ratio**, **match count**, and **runtime**

---

## 🔧 Pipeline Architecture

### Classical Pipeline
```
Image → ORB Detector → kNN Matcher → Essential Matrix + RANSAC → Pose
```
- **Detector:** ORB (Oriented FAST and Rotated BRIEF)
- **Matcher:** kNN with Lowe's ratio test
- **Pose Estimation:** Essential matrix decomposition with RANSAC
- **Masking:** None

### Hybrid Pipeline (Future)
```
Image → Deep Detector → Deep Matcher → Essential Matrix + RANSAC → Pose
         (SuperPoint/DISK)  (SuperGlue/LightGlue)    (Classical)
```
- Combines learned feature detection/matching with classical geometry
- Optional dynamic-object masking for improved robustness


### Pipeline Components

**Detectors:**
- `ORBDetector` - Classical ORB feature detector ✅
- `SuperPointDetector` - Learned detector & descriptor ✅
- `DISKDetector` - Learned detector & descriptor (to be implemented)

**Matchers:**
- `KNNMatcher` - Brute-force kNN matching with ratio test ✅
- `SuperGlueMatcher` - Learned matcher ✅
- `LightGlueMatcher` - Learned matcher (to be implemented)

**Pose Estimation:**
- `PoseEstimator` - Essential matrix estimation with RANSAC
- Monocular pipeline (scale ambiguity handled via alignment)

**Masking (Future):**
- `OpticalFlowMask` - Classical optical flow-based masking
- `FastSCNNMask` - Deep semantic segmentation-based masking



## 🛣️ Planned Pipelines

We are expanding the benchmark to cover seven pipelines spanning classical, hybrid, and fully-deep configurations:

1) ORB + kNN — classical baseline; strong in static scenes, degrades with dynamics. ✅
2) ORB + SuperGlue — isolates the benefit of a deep matcher with a classical detector. 🚧
3) ORB + LightGlue — lighter deep matcher vs SuperGlue under the same detector. 🚧
4) SuperPoint + kNN — deep detector with classical matching to gauge learned features alone. ✅
5) SuperPoint + SuperGlue — fully deep, heavy pipeline for maximum robustness. ✅
6) SuperPoint + LightGlue — fully deep, efficient alternative for runtime vs accuracy trade-offs. 🚧
7) DISK + LightGlue — alternative learned detector paired with a deep matcher to compare SuperPoint vs DISK. 🚧

---

## 📁 Repository Structure

```
4776_proj/
├── detectors/              # Feature detectors
│   ├── orb_detector.py      # ORB detector ✅
│   ├── superpoint_infer.py  # SuperPoint detector ✅
│   └── disk_infer.py        # DISK detector 🚧
│
├── matchers/                # Feature matchers
│   ├── knn_matcher.py       # kNN matcher ✅
│   ├── superglue_infer.py   # SuperGlue matcher ✅
│   └── lightglue_infer.py   # LightGlue matcher 🚧
│
├── masking/                 # Dynamic object masking
│   ├── opticalflow_mask.py  # Optical flow masking (to be implemented)
│   └── fastscnn_infer.py    # Fast-SCNN masking (to be implemented)
│
├── geometry/                # Geometric operations
│   └── pose_estimation.py   # Essential matrix + RANSAC pose estimation
│
├── eval/                    # Evaluation tools
│   ├── dataset_loader_tum.py    # TUM dataset loader
│   ├── groundtruth_loader.py     # Ground truth trajectory loader
│   ├── metrics.py                 # ATE, RPE computation
│   ├── align.py                  # Trajectory alignment (Umeyama)
│   └── plots.py                  # Visualization utilities
│
├── config/                  # Pipeline configuration files
│   ├── classical_orb_knn.yaml        # Classical ORB + kNN ✅
│   ├── hybrid_superpoint_knn.yaml    # SuperPoint + kNN ✅
│   └── hybrid_superpoint_superglue.yaml  # SuperPoint + SuperGlue ✅
│
├── data/                    # Datasets (not tracked in git)
│   ├── TUM/                 # TUM RGB-D sequences
│   └── KITTI/               # KITTI odometry sequences
│
├── main.py                  # Main entry point for all pipelines
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Create conda environment
conda create -n vo-benchmark python=3.9
conda activate vo-benchmark

# Install dependencies
pip install -r requirements.txt

# Install EVO for trajectory evaluation (optional)
pip install evo --upgrade
```

### 2. Download Datasets

**TUM RGB-D:**
```bash
mkdir -p data/TUM
cd data/TUM

# Download sequences
curl -O https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz
curl -O https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_xyz.tgz
curl -O https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_halfsphere.tgz

# Extract
for f in *.tgz; do tar -xvf "$f"; done
```

**KITTI Odometry:**
```bash
mkdir -p data/KITTI
cd data/KITTI
curl -O https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip
unzip data_odometry_gray.zip
# Keep only sequences 00/, 05/, 09/
```

### 3. Run a Pipeline

```bash
# Classical baseline (ORB + kNN)
python main.py --config config/classical_orb_knn.yaml \
               --dataset TUM --sequence rgbd_dataset_freiburg1_desk

# Hybrid pipeline (SuperPoint + kNN)
python main.py --config config/hybrid_superpoint_knn.yaml \
               --dataset TUM --sequence rgbd_dataset_freiburg1_desk

# Fully deep pipeline (SuperPoint + SuperGlue)
python main.py --config config/hybrid_superpoint_superglue.yaml \
               --dataset TUM --sequence rgbd_dataset_freiburg1_desk
```

**Note:** Evaluation, visualization, and trajectory saving are **automatic** - no flags needed!

---

## 💻 Command-Line Usage

### Basic Syntax

```bash
python main.py --config <config_file> [--dataset <dataset>] [--sequence <sequence>] [--save <path>]
```

### Required Arguments

- `--config <path>` - Path to YAML config file defining the pipeline method (detector/matcher/masking)

### Optional Arguments

- `--dataset <name>` - Override dataset from config (`TUM` or `KITTI`)
- `--sequence <name>` - Override sequence from config
- `--save <path>` - Override trajectory save path (auto-generated by default)

### Automatic Features

The pipeline **automatically** performs:
- ✅ **Trajectory saving** - Saved to `output/{config_name}_{sequence_name}/trajectory.txt`
- ✅ **Evaluation** - Runs if ground truth is available, saves metrics to JSON
- ✅ **Visualization** - Generates trajectory and error plots automatically

### Examples

**Run classical pipeline (automatic evaluation & visualization):**
```bash
python main.py --config config/classical_orb_knn.yaml \
               --dataset TUM --sequence rgbd_dataset_freiburg1_desk
```

**Run SuperPoint + kNN:**
```bash
python main.py --config config/hybrid_superpoint_knn.yaml \
               --dataset TUM --sequence rgbd_dataset_freiburg1_desk
```

**Run SuperPoint + SuperGlue:**
```bash
python main.py --config config/hybrid_superpoint_superglue.yaml \
               --dataset TUM --sequence rgbd_dataset_freiburg1_desk
```

**Run on different TUM sequences:**
```bash
# Static scene
python main.py --config config/classical_orb_knn.yaml \
               --dataset TUM --sequence rgbd_dataset_freiburg1_desk

# Low dynamics
python main.py --config config/classical_orb_knn.yaml \
               --dataset TUM --sequence rgbd_dataset_freiburg3_walking_xyz

# High dynamics
python main.py --config config/classical_orb_knn.yaml \
               --dataset TUM --sequence rgbd_dataset_freiburg3_walking_halfsphere
```

**Run on KITTI:**
```bash
python main.py --config config/classical_orb_knn.yaml \
               --dataset KITTI --sequence 09
```

**Batch processing (loop over sequences):**
```bash
for seq in rgbd_dataset_freiburg1_desk \
          rgbd_dataset_freiburg3_walking_xyz \
          rgbd_dataset_freiburg3_walking_halfsphere; do
    python main.py --config config/classical_orb_knn.yaml \
                   --dataset TUM --sequence $seq
done
```

---

## ⚙️ Configuration Files

### Config Structure

Config files define the **pipeline method** (which detector/matcher/masking to use), while command-line arguments specify **where to run** (dataset + sequence).

**Example: `config/classical_orb_knn.yaml`**
```yaml
# Detector configuration
detector: orb
detector_params:
  nfeatures: 2000

# Matcher configuration
matcher: knn
matcher_params:
  ratio: 0.75

# Masking (none for classical)
masking: none

# Dataset (can be overridden via CLI)
dataset: TUM
sequence: rgbd_dataset_freiburg1_desk
data_root: data/TUM

# Pose estimation parameters
pose_estimation:
  ransac_threshold: 0.999
  ransac_confidence: 1.0
  min_matches: 8

# Evaluation parameters
evaluation:
  max_time_diff: 0.02
  rpe_delta: 1

# Output configuration
output:
  trajectory_file: trajectory.txt
  trajectory_plot: trajectory.png
  errors_plot: errors.png
  plot_dpi: 150
```

### Available Configs

- `config/classical_orb_knn.yaml` - Classical: ORB + kNN ✅
- `config/hybrid_superpoint_knn.yaml` - Hybrid: SuperPoint + kNN ✅
- `config/hybrid_superpoint_superglue.yaml` - Hybrid: SuperPoint + SuperGlue ✅
- `config/hybrid_superpoint_lightglue.yaml` - Hybrid: SuperPoint + LightGlue 🚧
- `config/hybrid_superpoint_lightglue_mask.yaml` - Hybrid with masking 🚧

### Design Philosophy

This structure allows you to:
- ✅ Define methods once in YAML files
- ✅ Run any method on any dataset/sequence via CLI
- ✅ Organize results by `(pipeline_name, dataset, sequence)` without creating dozens of config files
- ✅ Easily compare methods across sequences

---

## 📊 Datasets

### TUM RGB-D (Indoor)

| Sequence | Dynamics | Description |
|----------|----------|-------------|
| `rgbd_dataset_freiburg1_desk` | Static | Desk scene, minimal motion |
| `rgbd_dataset_freiburg3_walking_xyz` | Low | Person walking in straight line |
| `rgbd_dataset_freiburg3_walking_halfsphere` | High | Person walking in half-sphere pattern |

**Expected structure:**
```
data/TUM/
├── rgbd_dataset_freiburg1_desk/
│   ├── rgb.txt
│   ├── groundtruth.txt
│   └── rgb/
└── ...
```

### KITTI Odometry (Outdoor)

| Sequence | Dynamics | Description |
|----------|----------|-------------|
| `00` | Static | Highway, minimal traffic |
| `05` | Medium | Urban with moderate traffic |
| `09` | High | Dense urban traffic |

**Expected structure:**
```
data/KITTI/
├── 00/
├── 05/
└── 09/
```

---

## 📈 Evaluation Pipeline

The project evaluates each method's output through a standardized pipeline that compares estimated trajectories against ground truth. Here's how it works:

### Evaluation Workflow

```
Estimated Trajectory → Timestamp Sync → Trajectory Alignment → Metric Computation → Results
```

### Step-by-Step Process

**1. Trajectory Generation**
- Each pipeline method (ORB+kNN, SuperPoint+LightGlue, etc.) processes the image sequence
- Produces an estimated trajectory: list of 4×4 pose matrices `[T₀, T₁, ..., Tₙ]`
- Each pose represents camera position and orientation at that frame
- Timestamps are recorded for each pose

**2. Ground Truth Loading**
- Loads ground truth trajectory from dataset (e.g., `groundtruth.txt` for TUM)
- Ground truth format: `timestamp tx ty tz qx qy qz qw` (position + quaternion)
- Converts quaternions to rotation matrices, builds 4×4 pose matrices

**3. Timestamp Synchronization**
- Matches estimated poses to ground truth poses by finding closest timestamps
- Only includes pose pairs within `max_time_diff` (default: 0.02 seconds)
- Ensures both trajectories have the same length and correspond to the same time points
- **Function:** `sync_trajectories()` in `eval/groundtruth_loader.py`

**4. Trajectory Alignment (Umeyama Algorithm)**
- Monocular VO has **scale ambiguity** (translation is normalized)
- Aligns estimated trajectory to ground truth using similarity transformation:
  - **Rotation** (R): Aligns coordinate frames
  - **Scale** (s): Recovers true scale from ground truth
  - **Translation** (t): Centers trajectories
- Transformation: `T_aligned = s·R·T_est + t`
- **Function:** `align_trajectories()` in `eval/groundtruth_loader.py`

**5. Metric Computation**

#### Absolute Trajectory Error (ATE)
Measures the absolute difference between estimated and ground truth positions:
- **RMSE**: `√(mean(||p_est - p_gt||²))` - Root mean square error
- **Mean**: `mean(||p_est - p_gt||)` - Average absolute error
- **Max**: `max(||p_est - p_gt||)` - Maximum absolute error
- **Function:** `compute_ate()` in `eval/metrics.py`

#### Relative Pose Error (RPE)
Measures error in relative motion between consecutive frames:
- For each frame pair `(i, i+δ)`:
  - Compute relative motion in estimated trajectory: `T_est_rel = T_est[i]⁻¹ · T_est[i+δ]`
  - Compute relative motion in ground truth: `T_gt_rel = T_gt[i]⁻¹ · T_gt[i+δ]`
  - Error: `T_error = T_gt_rel⁻¹ · T_est_rel`
- **Translation RMSE**: `√(mean(||t_error||²))` - Relative translation error
- **Rotation RMSE**: `√(mean(angle(R_error)²))` - Relative rotation error (degrees)
- **Function:** `compute_rpe()` in `eval/metrics.py`

**6. Runtime Metrics (Per-Frame)**
- **Match count**: Number of feature matches found per frame
- **Inlier ratio**: Percentage of matches surviving RANSAC filtering
- **Tracking failures**: Frames where pose estimation failed (insufficient matches)

**7. Visualization (Optional)**
- **3D Trajectory Plot**: Overlays estimated and ground truth trajectories
- **Error Plots**: ATE and RPE errors over time
- **Function:** `plot_trajectory()`, `plot_errors()` in `eval/plots.py`

### Usage

Evaluation runs automatically when ground truth is available:

```bash
python main.py --config config/classical_orb_knn.yaml \
               --dataset TUM --sequence rgbd_dataset_freiburg1_desk
```

**Output:**
```
📊 Evaluating against ground truth...
   Estimated: 613 poses
   Ground truth: 2335 poses
   After sync: 613 matched poses
   Alignment scale: 0.5234

📈 Absolute Trajectory Error (ATE):
   RMSE: 0.0234 m
   Mean: 0.0198 m
   Max:  0.0456 m

📈 Relative Pose Error (RPE):
   Translation RMSE: 0.0123 m
   Rotation RMSE:    1.23 deg
```

### Comparing Methods

To compare different methods, run each pipeline and collect metrics:

```bash
# Classical baseline
python main.py --config config/classical_orb_knn.yaml \
               --dataset TUM --sequence rgbd_dataset_freiburg1_desk

# SuperPoint + kNN
python main.py --config config/hybrid_superpoint_knn.yaml \
               --dataset TUM --sequence rgbd_dataset_freiburg1_desk

# SuperPoint + SuperGlue
python main.py --config config/hybrid_superpoint_superglue.yaml \
               --dataset TUM --sequence rgbd_dataset_freiburg1_desk
```

Results are automatically organized by `(pipeline_name, dataset, sequence)` in the `output/` directory for systematic comparison.

---

## 💾 Output Files

All output files are automatically saved to organized directories in the `output/` folder. The directory structure is:

```
output/
├── {config_name}_{sequence_name}/
│   ├── trajectory.txt      # Estimated trajectory (TUM format)
│   ├── metrics.json        # Evaluation metrics
│   ├── trajectory.png      # 3D trajectory visualization
│   └── errors.png          # Error plots (ATE & RPE)
```

**Example:** `output/hybrid_superpoint_knn_rgbd_dataset_freiburg1_desk/`

### Metrics File (`output/{config_name}_{sequence_name}/metrics.json`)
Saved automatically when evaluation is performed. Contains:
- **Absolute Trajectory Error (ATE)**: RMSE, mean, max (in meters)
- **Relative Pose Error (RPE)**: Translation and rotation RMSE
- **Runtime Metrics**: Average matches per frame, average inlier ratio
- **Metadata**: Pipeline name, dataset, sequence, timestamp

**Location:** `output/{config_name}_{sequence_name}/metrics.json` (auto-generated, configurable via `output.metrics_file` in config)

**Example:**
```json
{
  "timestamp": "2025-12-02T19:30:45.123456",
  "pipeline": "classical_orb_knn",
  "dataset": "TUM",
  "sequence": "rgbd_dataset_freiburg1_desk",
  "num_frames": 613,
  "absolute_trajectory_error": {
    "rmse": 0.0234,
    "mean": 0.0198,
    "max": 0.0456,
    "unit": "meters"
  },
  "relative_pose_error": {
    "translation_rmse": 0.0123,
    "rotation_rmse": 1.23,
    "translation_unit": "meters",
    "rotation_unit": "degrees"
  },
  "runtime_metrics": {
    "avg_matches_per_frame": 1250.5,
    "avg_inlier_ratio": 0.85
  }
}
```

### Visualization Plots
- **`output/{config_name}_{sequence_name}/trajectory.png`**: 3D plot showing estimated vs ground truth trajectories
- **`output/{config_name}_{sequence_name}/errors.png`**: Error plots showing ATE and RPE over time

**Location:** Auto-generated based on config name and sequence (configurable via `output.trajectory_plot` and `output.errors_plot` in config)

### Trajectory File
- **Format**: TUM format (`timestamp tx ty tz qx qy qz qw`)
- **Location**: `output/{config_name}_{sequence_name}/trajectory.txt` (auto-generated)
- **Usage**: Can be used with EVO toolkit for additional analysis
- **Override**: Use `--save <path>` to specify a custom path

### Output Organization

All outputs are automatically organized by pipeline and sequence:
- Each run creates a directory: `output/{config_name}_{sequence_name}/`
- All related files (trajectory, metrics, plots) are saved together
- Easy to compare results across different pipelines on the same sequence

**Example output structure:**
```
output/
├── classical_orb_knn_rgbd_dataset_freiburg1_desk/
│   ├── trajectory.txt
│   ├── metrics.json
│   ├── trajectory.png
│   └── errors.png
├── hybrid_superpoint_knn_rgbd_dataset_freiburg1_desk/
│   ├── trajectory.txt
│   ├── metrics.json
│   ├── trajectory.png
│   └── errors.png
└── hybrid_superpoint_superglue_rgbd_dataset_freiburg1_desk/
    ├── trajectory.txt
    ├── metrics.json
    ├── trajectory.png
    └── errors.png
```

**Note:** The `output/` folder is automatically created and is ignored by git (added to `.gitignore`).

---

## 🔧 Implementation Details

### SuperPoint Detector
- **L2-normalized descriptors** for proper distance computation
- **Keypoint bounds validation** to filter out-of-bounds features
- **Empty keypoint handling** for robust edge case management
- **Score-based keypoint size** for better visualization

### SuperGlue Matcher
- **Indoor/outdoor weights** - Automatically selects appropriate weights based on dataset type
  - Indoor weights (ScanNet-trained) for TUM RGB-D
  - Outdoor weights (MegaDepth-trained) for KITTI
- **Keypoint scores integration** - Properly passes SuperPoint scores to SuperGlue
- **Configurable match threshold** - Default 0.2 for optimal balance

### Pose Estimation
- **RANSAC validation** - Ensures confidence values are within valid range (0 < conf < 1)
- **Robust error handling** - Gracefully handles edge cases and failed pose estimates

## 📌 Expected Findings

Based on the research goals, we expect:

- ✅ **Deep matchers** (SuperGlue / LightGlue) improve robustness in moderately dynamic scenes
- ✅ **Hybrid pipelines** (SuperPoint + LightGlue) offer the best balance of robustness and runtime
- ✅ **Dynamic masking** significantly stabilizes pose estimation under high dynamics
- ✅ **Classical ORB + kNN** works well in static scenes but degrades quickly with motion and occlusion

---

## 👥 Authors

- Hongyuan Kang
- Zhengbin Lu
- Hanzhi Bian
- Yujia Zhai

**Columbia University — COMS 4776 (Fall 2025)**

---

## 📄 License

MIT License

---

## 🔗 References

- TUM RGB-D Dataset: https://vision.in.tum.de/data/datasets/rgbd-dataset
- KITTI Odometry: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
- EVO Toolkit: https://github.com/MichaelGrupp/evo
