# 使用指南：SuperPoint、LightGlue 和 Masking

本文档介绍如何使用新实现的 SuperPoint、LightGlue 和动态物体 masking 功能。

## 📦 安装依赖

首先确保安装了所有必需的依赖：

```bash
pip install -r requirements.txt

# 可选：安装 LightGlue（如果 torch.hub 加载失败）
pip install git+https://github.com/cvg/LightGlue.git

# 可选：如果需要使用 Fast-SCNN（轻量级语义分割）
pip install git+https://github.com/Tramac/Fast-SCNN-pytorch.git
```

## 🚀 快速开始

### 1. 基础混合管道：SuperPoint + LightGlue

使用深度学习特征检测器和匹配器：

```bash
python main.py --config config/hybrid_superpoint_lightglue.yaml \
               --dataset TUM --sequence rgbd_dataset_freiburg1_desk \
               --eval --visualize
```

### 2. 带 Masking 的混合管道

在动态场景中使用 optical flow masking 提高鲁棒性：

```bash
python main.py --config config/hybrid_superpoint_lightglue_mask.yaml \
               --dataset TUM --sequence rgbd_dataset_freiburg3_walking_xyz \
               --eval --visualize
```

## 📝 配置文件详解

### SuperPoint 检测器配置

```yaml
detector: superpoint
detector_params:
  device: cuda              # 'cuda' 或 'cpu'
  max_keypoints: 2048       # 最大特征点数量，-1 表示不限制
  keypoint_threshold: 0.005 # 特征点检测阈值
  nms_radius: 4             # 非极大值抑制半径
```

**参数说明：**
- `max_keypoints`: 控制检测的特征点数量。更多特征点可能提高准确性，但会增加计算时间
- `keypoint_threshold`: 降低此值会检测到更多特征点（包括一些弱特征）
- `nms_radius`: 控制特征点之间的最小距离

### LightGlue 匹配器配置

```yaml
matcher: lightglue
matcher_params:
  features: superpoint      # 特征类型，需与检测器匹配
  device: cuda              # 'cuda' 或 'cpu'
  filter_threshold: 0.1     # 匹配置信度阈值

use_lightglue_adapter: true # 使用适配器（推荐）
```

**参数说明：**
- `features`: 必须与使用的检测器类型匹配（'superpoint', 'disk', 'sift' 等）
- `filter_threshold`: 越低匹配越宽松，越高越严格
- `use_lightglue_adapter`: 建议设为 `true`，因为 LightGlue 需要特征点位置信息

### Masking 配置选项

#### 选项 1：Optical Flow Masking（经典方法）

适用于：实时性能要求高的场景

```yaml
masking: opticalflow
masking_params:
  flow_method: farneback    # 'farneback' 或 'dis'
  threshold: 2.0            # 光流偏差阈值（像素）
  min_flow_magnitude: 1.0   # 最小光流幅度
```

**优点：**
- 计算速度快
- 不需要预训练模型
- 适合室内外各种场景

**缺点：**
- 对光照变化敏感
- 可能误判快速相机运动

#### 选项 2：Advanced Optical Flow（带单应性估计）

适用于：有大型动态物体的场景

```yaml
masking: opticalflow_advanced
masking_params:
  flow_method: farneback
  threshold: 3.0
  use_homography: true
```

**优点：**
- 更鲁棒的相机运动估计
- 更好地处理大型动态物体

#### 选项 3：Semantic Segmentation Masking（深度学习）

适用于：已知物体类别的场景（如城市道路、室内）

```yaml
masking: semantic
masking_params:
  model: deeplabv3         # 'deeplabv3', 'fcn', 'maskrcnn'
  dataset: coco            # 'coco' 或 'cityscapes'
  device: cuda
  erosion_kernel: 5        # 膨胀核大小
```

**支持的模型：**
- `deeplabv3`: 准确，速度适中
- `fcn`: 速度较快，准确度略低
- `maskrcnn`: 实例分割，最准确但最慢

**数据集选择：**
- `coco`: 适合一般场景（室内、室外）
- `cityscapes`: 专门针对城市街道场景

**优点：**
- 准确识别已知动态物体
- 不依赖光流

**缺点：**
- 计算开销大
- 依赖预训练模型的类别
- 无法识别训练集外的物体

#### 选项 4：Hybrid Masking（结合两种方法）

适用于：需要高精度且计算资源充足的场景

```yaml
masking: hybrid
masking_params:
  flow_threshold: 2.0
  semantic_model: deeplabv3
  dataset: coco
  device: cuda
```

**优点：**
- 结合两种方法的优势
- 减少误报（如停放的车辆）

**缺点：**
- 计算开销最大

## 🎯 使用场景推荐

### 静态场景（如办公室、桌面）
```bash
# 不需要 masking，使用基础混合管道即可
python main.py --config config/hybrid_superpoint_lightglue.yaml \
               --sequence rgbd_dataset_freiburg1_desk
```

### 低动态场景（有少量移动物体）
```bash
# 使用 optical flow masking
python main.py --config config/hybrid_superpoint_lightglue_mask.yaml \
               --sequence rgbd_dataset_freiburg3_walking_xyz
```

编辑配置文件，设置：
```yaml
masking: opticalflow
masking_params:
  threshold: 2.0
```

### 高动态场景（多个移动物体）
```bash
# 使用 semantic segmentation 或 hybrid masking
python main.py --config config/hybrid_superpoint_lightglue_mask.yaml \
               --sequence rgbd_dataset_freiburg3_walking_halfsphere
```

编辑配置文件，设置：
```yaml
masking: semantic
masking_params:
  model: deeplabv3
  dataset: coco
```

或使用混合方法：
```yaml
masking: hybrid
```

### 城市街道场景（KITTI）
```bash
python main.py --config config/hybrid_superpoint_lightglue_mask.yaml \
               --dataset KITTI --sequence 09
```

编辑配置文件，使用 Cityscapes 训练的模型：
```yaml
masking: semantic
masking_params:
  model: deeplabv3
  dataset: cityscapes  # 针对城市场景优化
```

## 🔧 性能优化建议

### GPU 内存不足？

1. **减少特征点数量：**
```yaml
detector_params:
  max_keypoints: 1024  # 从 2048 降低到 1024
```

2. **使用 CPU：**
```yaml
detector_params:
  device: cpu
matcher_params:
  device: cpu
```

3. **简化 masking：**
```yaml
masking: opticalflow  # 而不是 semantic 或 hybrid
```

### 追求速度？

1. **使用经典管道：**
```bash
python main.py --config config/classical_orb_knn.yaml
```

2. **减少特征点：**
```yaml
detector_params:
  max_keypoints: 512
```

3. **使用简单 masking：**
```yaml
masking: opticalflow
masking_params:
  flow_method: dis  # DIS 比 Farneback 更快
```

### 追求准确度？

1. **增加特征点：**
```yaml
detector_params:
  max_keypoints: 4096
  keypoint_threshold: 0.003  # 更敏感
```

2. **使用严格匹配：**
```yaml
matcher_params:
  filter_threshold: 0.2  # 提高阈值
```

3. **使用 hybrid masking：**
```yaml
masking: hybrid
```

## 📊 批量实验示例

运行不同配置的批量实验：

```bash
#!/bin/bash

# TUM 数据集的所有序列
SEQUENCES=(
    "rgbd_dataset_freiburg1_desk"
    "rgbd_dataset_freiburg3_walking_xyz"
    "rgbd_dataset_freiburg3_walking_halfsphere"
)

# 测试不同配置
CONFIGS=(
    "config/classical_orb_knn.yaml"
    "config/hybrid_superpoint_lightglue.yaml"
    "config/hybrid_superpoint_lightglue_mask.yaml"
)

for config in "${CONFIGS[@]}"; do
    for seq in "${SEQUENCES[@]}"; do
        echo "Running: $config on $seq"
        python main.py --config "$config" \
                       --dataset TUM --sequence "$seq" \
                       --eval --save "results/${config}_${seq}.txt"
    done
done
```

## ⚠️ 常见问题

### 1. "Could not load SuperPoint from torch hub"

**解决方案：**
代码会自动使用本地实现。如果还有问题，检查网络连接或手动下载模型权重。

### 2. "LightGlue package not found"

**解决方案：**
```bash
pip install git+https://github.com/cvg/LightGlue.git
```

### 3. "CUDA out of memory"

**解决方案：**
- 减少 `max_keypoints`
- 使用 `device: cpu`
- 使用更简单的 masking 方法

### 4. Masking 导致匹配点太少

**解决方案：**
- 增大 `threshold` 参数（对于 optical flow）
- 减小 `erosion_kernel`（对于 semantic）
- 或者完全禁用 masking：`masking: none`

### 5. 速度太慢

**解决方案：**
- 使用 GPU（`device: cuda`）
- 减少特征点数量
- 使用 optical flow masking 而不是 semantic
- 考虑使用经典管道（ORB + kNN）

## 📈 预期性能

基于我们的测试（大致估计）：

| 配置 | ATE (m) | 速度 (FPS) | GPU 内存 |
|------|---------|-----------|----------|
| ORB + kNN | 高 | ~30 | 最小 |
| SuperPoint + LightGlue | 中 | ~10-15 | ~2GB |
| SuperPoint + LightGlue + Optical Flow | 中-低 | ~8-12 | ~2GB |
| SuperPoint + LightGlue + Semantic | 低 | ~5-8 | ~4GB |
| SuperPoint + LightGlue + Hybrid | 最低 | ~3-5 | ~4GB |

**注意：** 实际性能取决于硬件、场景复杂度和参数设置。

## 🎓 参考文献

- **SuperPoint**: DeTone et al., "SuperPoint: Self-Supervised Interest Point Detection and Description", CVPR 2018
- **LightGlue**: Lindenberger et al., "LightGlue: Local Feature Matching at Light Speed", ICCV 2023
- **DeepLabV3**: Chen et al., "Rethinking Atrous Convolution for Semantic Image Segmentation", arXiv 2017

## 💬 获取帮助

如果遇到问题：
1. 检查此文档的"常见问题"部分
2. 查看 README.md 了解项目整体结构
3. 检查配置文件格式是否正确
4. 确保数据集路径正确
