# 📌 Evaluating Hybrid Visual Odometry Under Scene Dynamics

Hybrid Visual Odometry (VO) combines classical geometry-based pipelines with modern deep feature detectors and matchers. While classical VO (e.g., ORB-SLAM-style pipelines) performs well in static environments, it suffers when dynamic elements dominate the scene. Deep neural components such as SuperPoint, DISK, SuperGlue, and LightGlue improve feature robustness but often overfit to training domains.

This project systematically benchmarks **classical, deep, and hybrid VO pipelines** under **varying levels of scene dynamics**, using both indoor and outdoor datasets.

## 🚀 Project Goals

- Evaluate different VO pipelines (classical, deep, hybrid) under controlled scene dynamics
- Compare detectors: **ORB**, **SuperPoint**, **DISK**
- Compare matchers: **kNN**, **SuperGlue**, **LightGlue**
- Assess dynamic-object masking: **optical flow** (classical) vs **Fast-SCNN** (deep)
- Quantify robustness using **ATE**, **RPE**, **inlier ratio**, **match count**, and **runtime**

**Research Question:**  
👉 *How do hybrid VO systems behave as scene dynamics increase, and which components contribute most to robustness?*

## 📁 Repository Structure

```
evaluating_hybrid_visual_odometry_under_scene_dynamics/
├── detectors/
│   ├── orb_detector.py
│   ├── superpoint_infer.py
│   └── disk_infer.py
├── matchers/
│   ├── knn_matcher.py
│   ├── superglue_infer.py
│   └── lightglue_infer.py
├── masking/
│   ├── opticalflow_mask.py
│   └── fastscnn_infer.py
├── geometry/
│   └── pose_estimation.py
├── eval/
│   ├── metrics.py
│   ├── align.py
│   └── plots.py
├── config/
│   └── pipeline.yaml
├── scripts/
│   └── download_data.sh
├── main.py
├── requirements.txt
└── README.md
```

## 📦 Installation

### 1. Create environment
```bash
conda create -n vo-benchmark python=3.9
conda activate vo-benchmark
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install EVO (for trajectory evaluation)
```bash
pip install evo --upgrade
```

## 📊 Datasets

We evaluate VO pipelines on indoor and outdoor benchmarks that cover static, moderately dynamic, and highly dynamic motion.

### TUM RGB-D (Indoor)
- `fr1/desk` — static
- `fr3/walking_xyz` — low dynamics
- `fr3/walking_halfsphere` — high dynamics

### KITTI Odometry (Outdoor)
- `00` — mostly static
- `05` — medium dynamics
- `09` — high dynamics with dense traffic

**Expected directory layout:**
```
data/
├── TUM/
│   ├── fr1/desk/
│   ├── fr3/walking_xyz/
│   └── fr3/walking_halfsphere/
└── KITTI/
    ├── 00/
    ├── 05/
    └── 09/
```

## 📥 Downloading the Datasets

### TUM RGB-D
```bash
mkdir -p data/TUM
cd data/TUM

curl -O https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz
curl -O https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_xyz.tgz
curl -O https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_halfsphere.tgz

# Extract
tar -xvf rgbd_dataset_freiburg1_desk.tgz
tar -xvf rgbd_dataset_freiburg3_walking_xyz.tgz
tar -xvf rgbd_dataset_freiburg3_walking_halfsphere.tgz
```

### KITTI Odometry
```bash
mkdir -p data/KITTI
cd data/KITTI

curl -O https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip
unzip data_odometry_gray.zip

# Only keep sequences 00/, 05/, 09/ (delete the rest to save disk space)
```

### Automated Script
```bash
chmod +x scripts/download_data.sh
./scripts/download_data.sh
```

## 🧠 VO Pipelines Evaluated

### Classical
- ORB detector
- kNN matching
- Essential Matrix + RANSAC
- Monocular pipeline (no loop closure)

### Deep Components
- **Detectors:** SuperPoint, DISK (learned detectors & descriptors)
- **Matchers:** SuperGlue, LightGlue (learned matchers)
- **Masking:** Fast-SCNN (dynamic-region removal)

### Hybrid Pipeline
- Deep detector + Deep matcher
- Classical geometric pose estimation
- Optional dynamic-object masking

## 📈 Metrics

We measure:
- **Absolute Trajectory Error (ATE)**
- **Relative Pose Error (RPE)**
- **Scale drift**
- **Inlier ratio**
- **Number of matches**
- **Tracking failures**
- **Runtime (FPS)**

Trajectory alignment uses the Umeyama method (via EVO toolkit).

## 🧪 Running Experiments

```bash
python main.py --config config/pipeline.yaml
```

**Example `pipeline.yaml`:**
```yaml
detector: superpoint
matcher: lightglue
masking: fastscnn
dataset: TUM
sequence: fr3/walking_halfsphere
```

## 📌 Expected Findings

- Deep matchers (SuperGlue / LightGlue) improve robustness in moderately dynamic scenes
- Hybrid pipelines (SuperPoint + LightGlue) offer the best balance of robustness and runtime
- Dynamic masking significantly stabilizes pose estimation under high dynamics
- Classical ORB + kNN works well in static scenes but degrades quickly with motion and occlusion

## 👥 Authors

- Hongyuan Kang
- Zhengbin Lu
- Hanzhi Bian
- Yujia Zhai

Columbia University — COMS 4776 (Fall 2025)

## 📄 License

MIT License
