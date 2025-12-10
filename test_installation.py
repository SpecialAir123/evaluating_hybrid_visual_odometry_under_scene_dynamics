#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本：验证所有依赖和模块是否正确安装
"""

import sys
import numpy as np

def test_basic_imports():
    """测试基础库导入"""
    print("=" * 60)
    print("测试基础库导入...")
    print("=" * 60)

    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV 导入失败: {e}")
        return False

    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy 导入失败: {e}")
        return False

    try:
        import matplotlib
        print(f"✓ Matplotlib: {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib 导入失败: {e}")
        return False

    try:
        import scipy
        print(f"✓ SciPy: {scipy.__version__}")
    except ImportError as e:
        print(f"✗ SciPy 导入失败: {e}")
        return False

    try:
        import yaml
        print(f"✓ PyYAML: {yaml.__version__}")
    except ImportError as e:
        print(f"✗ PyYAML 导入失败: {e}")
        return False

    return True


def test_deep_learning():
    """测试深度学习库"""
    print("\n" + "=" * 60)
    print("测试深度学习库...")
    print("=" * 60)

    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"  CUDA 可用: {torch.cuda.is_available()}")
        print(f"  MPS 可用 (Apple Silicon): {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else 'N/A'}")
    except ImportError as e:
        print(f"✗ PyTorch 导入失败: {e}")
        return False

    try:
        import torchvision
        print(f"✓ TorchVision: {torchvision.__version__}")
    except ImportError as e:
        print(f"✗ TorchVision 导入失败: {e}")
        return False

    return True


def test_project_modules():
    """测试项目模块导入"""
    print("\n" + "=" * 60)
    print("测试项目模块...")
    print("=" * 60)

    try:
        from detectors.orb_detector import ORBDetector
        print("✓ ORBDetector")
    except ImportError as e:
        print(f"✗ ORBDetector 导入失败: {e}")
        return False

    try:
        from detectors.superpoint_infer import SuperPoint
        print("✓ SuperPoint")
    except ImportError as e:
        print(f"✗ SuperPoint 导入失败: {e}")
        return False

    try:
        from matchers.knn_matcher import KNNMatcher
        print("✓ KNNMatcher")
    except ImportError as e:
        print(f"✗ KNNMatcher 导入失败: {e}")
        return False

    try:
        from matchers.lightglue_infer import LightGlue, LightGlueMatcherAdapter
        print("✓ LightGlue, LightGlueMatcherAdapter")
    except ImportError as e:
        print(f"✗ LightGlue 导入失败: {e}")
        return False

    try:
        from masking.opticalflow_mask import OpticalFlowMask
        print("✓ OpticalFlowMask")
    except ImportError as e:
        print(f"✗ OpticalFlowMask 导入失败: {e}")
        return False

    try:
        from masking.semantic_mask import SemanticSegmentationMask
        print("✓ SemanticSegmentationMask")
    except ImportError as e:
        print(f"✗ SemanticSegmentationMask 导入失败: {e}")
        return False

    try:
        from geometry.pose_estimation import PoseEstimator
        print("✓ PoseEstimator")
    except ImportError as e:
        print(f"✗ PoseEstimator 导入失败: {e}")
        return False

    return True


def test_detector_instantiation():
    """测试检测器实例化"""
    print("\n" + "=" * 60)
    print("测试检测器实例化...")
    print("=" * 60)

    try:
        from detectors.orb_detector import ORBDetector
        detector = ORBDetector(nfeatures=500)
        print("✓ ORBDetector 实例化成功")

        # 测试在随机图像上运行
        img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        kpts, descs = detector(img)
        print(f"  检测到 {len(kpts)} 个特征点")

    except Exception as e:
        print(f"✗ ORBDetector 实例化失败: {e}")
        return False

    try:
        from detectors.superpoint_infer import SuperPoint
        print("✓ SuperPoint 类加载成功")
        print("  注意: SuperPoint 需要下载预训练模型，首次运行可能需要一些时间")

    except Exception as e:
        print(f"✗ SuperPoint 加载失败: {e}")
        return False

    return True


def test_masking():
    """测试 masking 模块"""
    print("\n" + "=" * 60)
    print("测试 Masking 模块...")
    print("=" * 60)

    try:
        from masking.opticalflow_mask import OpticalFlowMask
        mask_gen = OpticalFlowMask(flow_method='farneback')
        print("✓ OpticalFlowMask 实例化成功")

        # 测试在随机图像上运行
        img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        mask = mask_gen(img)  # 第一帧
        print(f"  生成 mask shape: {mask.shape if mask is not None else 'None (第一帧)'}")

    except Exception as e:
        print(f"✗ OpticalFlowMask 测试失败: {e}")
        return False

    return True


def test_config_files():
    """测试配置文件"""
    print("\n" + "=" * 60)
    print("测试配置文件...")
    print("=" * 60)

    import yaml
    import os

    config_files = [
        'config/classical_orb_knn.yaml',
        'config/hybrid_superpoint_lightglue.yaml',
        'config/hybrid_superpoint_lightglue_mask.yaml'
    ]

    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    cfg = yaml.safe_load(f)
                print(f"✓ {config_file}")
            except Exception as e:
                print(f"✗ {config_file} 解析失败: {e}")
                return False
        else:
            print(f"⚠ {config_file} 不存在")

    return True


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("开始测试安装...")
    print("=" * 60 + "\n")

    all_passed = True

    # 运行所有测试
    all_passed &= test_basic_imports()
    all_passed &= test_deep_learning()
    all_passed &= test_project_modules()
    all_passed &= test_detector_instantiation()
    all_passed &= test_masking()
    all_passed &= test_config_files()

    # 总结
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有测试通过！环境配置成功。")
        print("\n下一步:")
        print("1. 下载 TUM 数据集（见 README.md）")
        print("2. 运行基础测试:")
        print("   python main.py --config config/classical_orb_knn.yaml --dataset TUM --sequence rgbd_dataset_freiburg1_desk")
    else:
        print("❌ 部分测试失败，请检查上述错误信息")
        print("\n常见问题:")
        print("1. 如果 PyTorch 相关错误，尝试重新安装: pip install torch torchvision")
        print("2. 如果模块导入失败，确保在项目根目录运行此脚本")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
