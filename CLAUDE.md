# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A benchmarking framework for evaluating **classical, deep learning, and hybrid visual odometry (VO) pipelines** under varying scene dynamics. The core research question: *How do hybrid VO systems behave as scene dynamics increase, and which components contribute most to robustness?*

The project implements a modular monocular VO pipeline where different **detectors** (ORB, SuperPoint), **matchers** (kNN, LightGlue), and **masking methods** (optical flow, semantic segmentation) can be mixed and matched via YAML configuration files.

## Environment Setup

**Active environment**: `vis` (Python 3.9)

```bash
# Activate environment
conda activate vis

# Test installation
python test_installation.py

# Verify all modules load correctly
python -c "from detectors.superpoint_infer import SuperPoint; from matchers.lightglue_infer import LightGlue; print('✓ All imports successful')"
```

## Running the Pipeline

### Basic Command Structure

```bash
python main.py --config <config.yaml> [--dataset TUM|KITTI] [--sequence <name>] [--eval] [--visualize] [--save <path>]
```

- **Config file** defines the method (detector/matcher/masking)
- **CLI args** override dataset/sequence for easy experimentation
- **--eval** computes ATE/RPE metrics against ground truth
- **--visualize** generates trajectory and error plots

### Example Commands

```bash
# Classical pipeline (ORB + kNN)
python main.py --config config/classical_orb_knn.yaml --dataset TUM --sequence rgbd_dataset_freiburg1_desk

# Hybrid pipeline with evaluation
python main.py --config config/hybrid_superpoint_lightglue.yaml --dataset TUM --sequence rgbd_dataset_freiburg1_desk --eval --visualize

# With dynamic masking (for scenes with moving objects)
python main.py --config config/hybrid_superpoint_lightglue_mask.yaml --dataset TUM --sequence rgbd_dataset_freiburg3_walking_xyz --eval
```

## Architecture

### Pipeline Flow

The main processing loop in `main.py` follows this sequence:

```
1. Load config → Initialize components (detector, matcher, mask_generator, pose_estimator)
2. For each frame pair:
   a. Generate dynamic mask (optional) - identifies moving objects
   b. Detect features (via detector or adapter.detect())
   c. Apply mask to filter keypoints (optional)
   d. Match features
   e. Estimate pose via Essential Matrix + RANSAC
   f. Update trajectory (accumulate relative poses)
3. Evaluate: sync timestamps → align trajectories (Umeyama) → compute ATE/RPE
```

### Critical Design Patterns

#### 1. LightGlue Adapter Pattern
LightGlue requires keypoint **locations** in addition to descriptors (unlike traditional matchers). Two integration approaches:

- **LightGlueMatcherAdapter** (recommended): Wraps the detector, caches keypoints automatically
  ```python
  matcher = LightGlueMatcherAdapter(detector, features='superpoint')
  kp1, desc1 = matcher.detect(img1)  # Caches kp1
  kp2, desc2 = matcher.detect(img2)  # Caches kp2
  matches = matcher.match(desc1, desc2)  # Uses cached keypoints
  ```

- **Direct usage**: Manually pass keypoints
  ```python
  matcher = LightGlue(features='superpoint')
  matches = matcher(desc1, desc2, kpts1, kpts2)
  ```

**In main.py**: The `use_lightglue_adapter` flag (lines 116-131) switches between these modes. When True, detection happens via `matcher.detect()` instead of `detector()`.

#### 2. Monocular Scale Ambiguity Handling
Monocular VO has inherent scale ambiguity. The pipeline handles this through:

1. **Pose estimation** (geometry/pose_estimation.py): Essential matrix gives rotation R and unit-norm translation t
2. **Trajectory accumulation**: Relative poses are chained: `current_pose = current_pose @ T`
3. **Evaluation alignment** (eval/align.py): Uses Umeyama algorithm to find optimal scale factor s, rotation R, translation t that aligns estimated trajectory to ground truth

**Camera intrinsics**: Hard-coded in `get_tum_intrinsics()` for freiburg1/2/3 cameras. Essential for Essential Matrix decomposition.

#### 3. Masking Integration
Masking filters out keypoints on dynamic objects to improve robustness:

- **Interface**: All mask generators return binary mask (H, W) where 1=static, 0=dynamic
- **Application**: `mask_generator.apply_mask_to_keypoints(kpts, mask)` returns filtered keypoints and valid indices
- **Descriptor filtering**: Must slice descriptors using the same indices to maintain correspondence

**In main.py** (lines 207-216): When masking is enabled, both keypoints AND descriptors are filtered before matching.

### Component Interfaces

All components follow consistent call signatures to enable plug-and-play configuration:

**Detectors** (`detectors/`):
```python
kpts, descs = detector(image)  # Returns (List[cv2.KeyPoint], np.ndarray)
```

**Matchers** (`matchers/`):
```python
matches = matcher(desc1, desc2)  # Returns List[cv2.DMatch]
```

**Mask Generators** (`masking/`):
```python
mask = mask_generator(image)  # Returns np.ndarray (H, W), dtype=uint8
kpts_filtered, indices = mask_generator.apply_mask_to_keypoints(kpts, mask)
```

**Pose Estimator** (`geometry/pose_estimation.py`):
```python
R, t, inlier_mask = pose_estimator.estimate(kp1, kp2, matches)
```

## Configuration System

### YAML Structure

Config files define the **method**, CLI args specify **where to run**:

```yaml
detector: superpoint | orb
detector_params:
  device: cuda | cpu | mps
  max_keypoints: 2048
  keypoint_threshold: 0.005

matcher: lightglue | knn
matcher_params:
  features: superpoint  # Must match detector type for LightGlue
  device: cuda | cpu
  filter_threshold: 0.1

use_lightglue_adapter: true  # Use adapter pattern (recommended)

masking: none | opticalflow | opticalflow_advanced | semantic | hybrid
masking_params:
  # Optical flow
  flow_method: farneback | dis
  threshold: 2.0
  # Semantic
  model: deeplabv3 | fcn | maskrcnn
  dataset: coco | cityscapes

pose_estimation:
  ransac_threshold: 0.999
  ransac_confidence: 1.0
  min_matches: 8

evaluation:
  max_time_diff: 0.02  # Max timestamp difference for trajectory sync
  rpe_delta: 1         # Frame delta for RPE computation
```

### Available Configurations

- `classical_orb_knn.yaml`: Classical ORB + kNN baseline
- `hybrid_superpoint_lightglue.yaml`: Deep detector/matcher, no masking
- `hybrid_superpoint_lightglue_mask.yaml`: Full hybrid with optical flow masking

## Dataset Structure

### TUM RGB-D Expected Layout
```
data/TUM/rgbd_dataset_freiburg1_desk/
├── rgb.txt              # timestamp filename
├── groundtruth.txt      # timestamp tx ty tz qx qy qz qw
└── rgb/
    ├── 1234567890.123456.png
    └── ...
```

**TUMDataset** (`eval/dataset_loader_tum.py`) parses `rgb.txt` to load frames in timestamp order.

### Test Sequences by Scene Dynamics

- **Static**: `rgbd_dataset_freiburg1_desk` (desk scene, minimal motion)
- **Low dynamics**: `rgbd_dataset_freiburg3_walking_xyz` (person walking in straight line)
- **High dynamics**: `rgbd_dataset_freiburg3_walking_halfsphere` (person walking in half-sphere)

Use these to evaluate how different pipelines degrade with increasing motion.

## Evaluation Pipeline

When `--eval` is specified (main.py lines 260-324):

1. **Load ground truth**: `load_tum_groundtruth()` parses TUM format (timestamp tx ty tz qx qy qz qw)
2. **Timestamp synchronization**: `sync_trajectories()` matches estimated poses to ground truth by nearest timestamp (within `max_time_diff`)
3. **Trajectory alignment**: `align_trajectories()` computes optimal similarity transform (scale + SE(3)) via Umeyama algorithm
4. **Metric computation**:
   - **ATE** (Absolute Trajectory Error): RMSE of aligned trajectory positions
   - **RPE** (Relative Pose Error): Relative translation/rotation error over `rpe_delta` frames

**Key insight**: Alignment handles scale ambiguity. Always align before computing metrics.

## Model Loading

### SuperPoint
- First attempts `torch.hub.load('magicleap/SuperGluePretrainedNetwork', 'superpoint')`
- Falls back to local `SuperPointNet` implementation if hub fails
- First run downloads ~9MB model weights (requires internet)

### LightGlue
- First attempts `torch.hub.load('cvg/LightGlue', 'lightglue', features=...)`
- Falls back to `from lightglue import LightGlue` (pip package)
- If neither works, instructs user: `pip install git+https://github.com/cvg/LightGlue.git`

### Semantic Segmentation
- Uses `torchvision.models.segmentation` (built-in, no extra install)
- Models: `deeplabv3_resnet101`, `fcn_resnet101`, `maskrcnn_resnet50_fpn`
- First run downloads model weights automatically

## Performance Considerations

### Masking Trade-offs

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| none | Fastest | Degrades in dynamics | Static scenes only |
| opticalflow | Fast (~10% overhead) | Good | Real-time, moderate dynamics |
| semantic | Slow (~3x slower) | Best for known objects | Offline, urban/indoor |
| hybrid | Slowest (~4x slower) | Most robust | Research, high accuracy needed |

**Recommendation**: Start with `opticalflow` for dynamic scenes. Use `semantic` only if you need precision and have GPU compute.

### Feature Detector Speed

- **ORB**: ~30 FPS (CPU), suitable for real-time
- **SuperPoint**: ~10-15 FPS (GPU), ~2-3 FPS (CPU)
  - Reduce `max_keypoints` (2048 → 1024) for 2x speedup with minimal accuracy loss

### GPU vs CPU

The config `device` parameter affects:
- SuperPoint feature extraction
- LightGlue matching
- Semantic segmentation

**Apple Silicon**: Use `device: mps` if PyTorch supports it, otherwise `device: cpu` (ARM-optimized)

## Common Development Tasks

### Adding a New Detector

1. Implement in `detectors/new_detector.py`:
   ```python
   class NewDetector:
       def __init__(self, **params): ...
       def __call__(self, image):  # Must return (kpts, descs)
           return keypoints, descriptors
   ```

2. Register in `detectors/__init__.py`

3. Add to `main.py` initialization (line 101-111):
   ```python
   elif detector_type == "newdetector":
       detector = NewDetector(**detector_params)
   ```

4. Create config file `config/pipeline_newdetector.yaml`

### Adding a New Matcher

Follow same pattern as detector. **Important**: If matcher needs keypoint locations (like LightGlue), consider implementing an adapter or update main loop to pass locations.

### Adding a New Masking Method

1. Implement in `masking/new_mask.py` with interface:
   ```python
   class NewMask:
       def __call__(self, image):  # Returns binary mask
           return mask
       def apply_mask_to_keypoints(self, keypoints, mask):
           return filtered_kpts, valid_indices
   ```

2. Register in main.py masking initialization (lines 140-163)

## Troubleshooting

### "Could not load SuperPoint from torch hub"
- Code automatically falls back to local implementation
- Check internet connection if first run
- Verify PyTorch version compatibility

### "CUDA out of memory"
- Reduce `max_keypoints` in config
- Switch to `device: cpu`
- Disable semantic masking (use `opticalflow` instead)

### "Ground truth not found"
- Check dataset structure matches TUM format
- Verify `groundtruth.txt` exists in sequence directory
- Run without `--eval` to test VO pipeline only

### Masking causes too few matches
- Increase masking `threshold` (optical flow)
- Reduce `erosion_kernel` (semantic segmentation)
- Set `masking: none` to disable

## Key Files Reference

- **main.py**: Main entry point, orchestrates entire pipeline
- **geometry/pose_estimation.py**: Essential matrix estimation (core VO algorithm)
- **eval/align.py**: Umeyama alignment for handling scale ambiguity
- **eval/metrics.py**: ATE/RPE computation
- **detectors/superpoint_infer.py**: SuperPoint implementation with torch.hub integration
- **matchers/lightglue_infer.py**: LightGlue + adapter pattern for keypoint caching
- **masking/opticalflow_mask.py**: Classical flow-based masking (fast)
- **masking/semantic_mask.py**: Deep learning segmentation-based masking (accurate)

## References

- **SuperPoint**: DeTone et al., CVPR 2018
- **LightGlue**: Lindenberger et al., ICCV 2023
- **TUM RGB-D**: https://vision.in.tum.de/data/datasets/rgbd-dataset
- **EVO toolkit**: https://github.com/MichaelGrupp/evo
