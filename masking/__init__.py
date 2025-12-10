"""Dynamic object masking modules for robust visual odometry.

This module provides different approaches to mask out dynamic objects
in the scene to improve pose estimation robustness.
"""

from .opticalflow_mask import OpticalFlowMask
from .semantic_mask import SemanticSegmentationMask

__all__ = ['OpticalFlowMask', 'SemanticSegmentationMask']
