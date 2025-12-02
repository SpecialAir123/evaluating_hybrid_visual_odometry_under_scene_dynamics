📌 Evaluating Hybrid Visual Odometry Under Scene Dynamics

Hybrid Visual Odometry (VO) combines classical geometry-based pipelines with modern deep feature detectors and matchers. While classical VO (e.g., ORB-SLAM-style pipelines) performs well in static environments, it suffers when dynamic elements dominate the scene. Deep neural components such as SuperPoint, DISK, SuperGlue, and LightGlue improve feature robustness but often overfit to training domains.

This project systematically benchmarks classical, deep, and hybrid VO pipelines under varying levels of scene dynamics, using both indoor and outdoor datasets.

🚀 Project Goals

Evaluate different VO pipelines (classical, deep, hybrid) under controlled scene dynamics.

Compare detectors: ORB, SuperPoint, DISK.

Compare matchers: kNN, SuperGlue, LightGlue.

Assess dynamic-object masking: optical flow (classical) vs Fast-SCNN (deep).

Quantify robustness using ATE, RPE, inlier ratio, match count, and runtime.

Ultimately, we aim to answer:
👉 How do hybrid VO systems behave as scene dynamics increase, and which components contribute most to robustness?

📁 Repository Structure
evaluating_hybrid_visual_odometry_under_scene_dynamics/
 ├── detectors/
 │    ├── orb_detector.py
 │    ├── superpoint_infer.py
 │    └── disk_infer.py
 ├── matchers/
 │    ├── knn_matcher.py
 │    ├── superglue_infer.py
 │    └── lightglue_infer.py
 ├── masking/
 │    ├── opticalflow_mask.py
 │    └── fastscnn_infer.py
 ├── geometry/
 │    └── pose_estimation.py
 ├── eval/
 │    ├── metrics.py        # ATE, RPE, drift, inlier stats
 │    ├── align.py
 │    └── plots.py
 ├── config/
 │    └── pipeline.yaml     # configure detector, matcher, mask, scenario
 ├── scripts/
 │    └── download_data.sh  # optional dataset downloader
 ├── main.py                # main evaluation runner
 ├── requirements.txt
 └── README.md

📦 Installation
1. Create environment
conda create -n vo-benchmark python=3.9
conda activate vo-benchmark

2. Install dependencies
pip install -r requirements.txt

3. (Optional) Install EVO for trajectory evaluation
pip install evo --upgrade

📊 Datasets

We evaluate VO pipelines on indoor and outdoor benchmarks covering static to highly dynamic scenes.

TUM RGB-D (Indoor)

fr1/desk — static

fr3/walking_xyz — low dynamics

fr3/walking_halfsphere — high dynamics

KITTI Odometry (Outdoor)

00 — mostly static

05 — medium dynamics

09 — high dynamics with dense traffic

Place them under data/ in this structure:

data/
 ├── TUM/fr3/walking_halfsphere/
 └── KITTI/09/

🧠 VO Pipelines Evaluated
Classical

ORB detector

kNN matching

Essential matrix + RANSAC

No loop closure (for fairness)

Deep Components

SuperPoint, DISK (detectors & descriptors)

SuperGlue, LightGlue (learned matchers)

Fast-SCNN for dynamic object masking

Hybrid

Classical geometry (5-point RANSAC)

Deep detector + matcher

Optional dynamic masking

📈 Metrics

We evaluate each pipeline using:

Absolute Trajectory Error (ATE)

Relative Pose Error (RPE)

Inlier ratio / number of matches

Tracking failures

Runtime (FPS)

Trajectory alignment uses the Umeyama alignment (via EVO).

🧪 Running Experiments

Run a full VO experiment using:

python main.py --config config/pipeline.yaml


Example pipeline.yaml:

detector: superpoint
matcher: lightglue
masking: fastscnn
dataset: TUM
sequence: fr3/walking_halfsphere

📌 Expected Findings

Based on prior research and early observations:

Deep matchers (SuperGlue / LightGlue) improve robustness under moderate dynamics.

Hybrid pipelines (SuperPoint + LightGlue) offer the best balance of accuracy and runtime.

Dynamic-region masking helps significantly in highly dynamic scenes.

Classical pipelines perform well in static scenes but degrade quickly as motion increases.

👥 Authors

Hongyuan Kang

Zhengbin Lu

Hanzhi Bian

Yujia Zhai

Columbia University — COMS 4776 (Fall 2025)