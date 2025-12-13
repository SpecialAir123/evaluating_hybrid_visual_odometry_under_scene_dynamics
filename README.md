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
- `ORBDetector` - Classical ORB feature detector
- `SuperPoint` - Learned detector & descriptor (to be implemented)
- `DISK` - Learned detector & descriptor (to be implemented)

**Matchers:**
- `KNNMatcher` - Brute-force kNN matching with ratio test
- `SuperGlue` - Learned matcher (to be implemented)
- `LightGlue` - Learned matcher (to be implemented)

**Pose Estimation:**
- `PoseEstimator` - Essential matrix estimation with RANSAC
- Monocular pipeline (scale ambiguity handled via alignment)

**Masking (Future):**
- `OpticalFlowMask` - Classical optical flow-based masking
- `FastSCNNMask` - Deep semantic segmentation-based masking



## 🛣️ Planned Pipelines

We are expanding the benchmark to cover seven pipelines spanning classical, hybrid, and fully-deep configurations:

1) ORB + kNN — classical baseline; strong in static scenes, degrades with dynamics.  (Done)
2) ORB + SuperGlue — isolates the benefit of a deep matcher with a classical detector.  
3) ORB + LightGlue — lighter deep matcher vs SuperGlue under the same detector.  
4) SuperPoint + kNN — deep detector with classical matching to gauge learned features alone.  
5) SuperPoint + SuperGlue — fully deep, heavy pipeline for maximum robustness.  
6) SuperPoint + LightGlue — fully deep, efficient alternative for runtime vs accuracy trade-offs.  
7) DISK + LightGlue — alternative learned detector paired with a deep matcher to compare SuperPoint vs DISK.

---

## 📁 Repository Structure

```
4776_proj/
├── detectors/              # Feature detectors
│   ├── orb_detector.py      # ORB detector (implemented)
│   ├── superpoint_infer.py  # SuperPoint (to be implemented)
│   └── disk_infer.py        # DISK (to be implemented)
│
├── matchers/                # Feature matchers
│   ├── knn_matcher.py       # kNN matcher (implemented)
│   ├── superglue_infer.py   # SuperGlue (to be implemented)
│   └── lightglue_infer.py   # LightGlue (to be implemented)
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
│   └── classical_orb_knn.yaml   # Classical ORB + kNN config
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

### 3. Run Baseline

```bash
python main.py --config config/classical_orb_knn.yaml \
               --dataset TUM --sequence rgbd_dataset_freiburg1_desk
```

---

## 💻 Command-Line Usage

### Basic Syntax

```bash
python main.py --config <config_file> [--dataset <dataset>] [--sequence <sequence>] [options]
```

### Required Arguments

- `--config <path>` - Path to YAML config file defining the pipeline method (detector/matcher/masking)

### Optional Arguments

- `--dataset <name>` - Override dataset from config (`TUM` or `KITTI`)
- `--sequence <name>` - Override sequence from config
- `--eval` - Evaluate against ground truth and compute metrics
- `--visualize` - Generate trajectory and error plots
- `--save <path>` - Save trajectory to file (TUM format)

### Examples

**Run classical pipeline on TUM static sequence:**
```bash
python main.py --config config/classical_orb_knn.yaml \
               --dataset TUM --sequence rgbd_dataset_freiburg1_desk
```

**Run with evaluation and visualization:**
```bash
python main.py --config config/classical_orb_knn.yaml \
               --dataset TUM --sequence rgbd_dataset_freiburg1_desk \
               --eval --visualize
```

**Run on different TUM sequences:**
```bash
# Static scene
python main.py --config config/classical_orb_knn.yaml \
               --dataset TUM --sequence rgbd_dataset_freiburg1_desk --eval

# Low dynamics
python main.py --config config/classical_orb_knn.yaml \
               --dataset TUM --sequence rgbd_dataset_freiburg3_walking_xyz --eval

# High dynamics
python main.py --config config/classical_orb_knn.yaml \
               --dataset TUM --sequence rgbd_dataset_freiburg3_walking_halfsphere --eval
```

**Run on KITTI:**
```bash
python main.py --config config/classical_orb_knn.yaml \
               --dataset KITTI --sequence 09 --eval
```

**Batch processing (loop over sequences):**
```bash
for seq in rgbd_dataset_freiburg1_desk \
          rgbd_dataset_freiburg3_walking_xyz \
          rgbd_dataset_freiburg3_walking_halfsphere; do
    python main.py --config config/classical_orb_knn.yaml \
                   --dataset TUM --sequence $seq --eval
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

- `config/classical_orb_knn.yaml` - Classical: ORB + kNN (✅ implemented)
- `config/hybrid_superpoint_superglue.yaml` - Hybrid: SuperPoint + SuperGlue (🚧 to be implemented)
- `config/hybrid_superpoint_lightglue_mask.yaml` - Hybrid with masking (🚧 to be implemented)

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

## 📈 Evaluation Metrics

The framework computes the following metrics when `--eval` is used:

### Absolute Trajectory Error (ATE)
- **RMSE** - Root mean square error of absolute trajectory
- **Mean** - Average absolute error
- **Max** - Maximum absolute error

### Relative Pose Error (RPE)
- **Translation RMSE** - Relative translation error
- **Rotation RMSE** - Relative rotation error (degrees)

### Additional Metrics
- **Inlier ratio** - Percentage of matches surviving RANSAC
- **Match count** - Number of feature matches per frame
- **Tracking failures** - Frames where pose estimation failed
- **Runtime (FPS)** - Processing speed

Trajectory alignment uses the **Umeyama algorithm** to handle scale ambiguity in monocular VO.

---

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
