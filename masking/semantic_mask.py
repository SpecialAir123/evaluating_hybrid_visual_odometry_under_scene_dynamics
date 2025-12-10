import cv2
import torch
import numpy as np


class SemanticSegmentationMask:
    """Dynamic object masking using semantic segmentation.

    This deep learning approach uses a semantic segmentation model to identify
    potentially dynamic objects (people, cars, bikes, etc.) and masks them out.

    Supports multiple segmentation models:
    - DeepLabV3+ (torchvision)
    - Fast-SCNN (lightweight for real-time)
    - Mask R-CNN (instance segmentation)
    """

    # Classes that are typically dynamic (COCO/Cityscapes)
    DYNAMIC_CLASSES_COCO = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
        6: 'bus', 7: 'train', 8: 'truck', 17: 'cat', 18: 'dog',
        19: 'horse', 20: 'sheep', 21: 'cow'
    }

    DYNAMIC_CLASSES_CITYSCAPES = {
        11: 'person', 12: 'rider', 13: 'car', 14: 'truck',
        15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle'
    }

    def __init__(self, model='deeplabv3', dataset='coco', device='cuda',
                 dynamic_classes=None, erosion_kernel=5):
        """Initialize semantic segmentation masking.

        Args:
            model: 'deeplabv3', 'fcn', or 'maskrcnn'
            dataset: 'coco' or 'cityscapes' (determines class labels)
            device: 'cuda' or 'cpu'
            dynamic_classes: Custom list of class IDs to mask, or None for default
            erosion_kernel: Size of morphological erosion kernel (to expand masks)
        """
        self.model_name = model
        self.dataset = dataset
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.erosion_kernel = erosion_kernel

        # Set dynamic classes
        if dynamic_classes is not None:
            self.dynamic_classes = set(dynamic_classes)
        elif dataset == 'cityscapes':
            self.dynamic_classes = set(self.DYNAMIC_CLASSES_CITYSCAPES.keys())
        else:  # coco
            self.dynamic_classes = set(self.DYNAMIC_CLASSES_COCO.keys())

        # Load model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

        # Erosion kernel for mask refinement
        if erosion_kernel > 0:
            self.kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (erosion_kernel, erosion_kernel)
            )
        else:
            self.kernel = None

    def _load_model(self):
        """Load pretrained segmentation model."""
        if self.model_name == 'deeplabv3':
            # DeepLabV3 with ResNet101 backbone
            from torchvision.models.segmentation import deeplabv3_resnet101
            model = deeplabv3_resnet101(pretrained=True)
            return model

        elif self.model_name == 'fcn':
            # FCN with ResNet101 backbone
            from torchvision.models.segmentation import fcn_resnet101
            model = fcn_resnet101(pretrained=True)
            return model

        elif self.model_name == 'maskrcnn':
            # Mask R-CNN (instance segmentation)
            from torchvision.models.detection import maskrcnn_resnet50_fpn
            model = maskrcnn_resnet50_fpn(pretrained=True)
            return model

        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def __call__(self, image):
        """Generate dynamic object mask for the image.

        Args:
            image: Input image (H, W, 3) in BGR format

        Returns:
            mask: Binary mask (H, W) where 1=static, 0=dynamic
        """
        # Convert BGR to RGB
        if len(image.shape) == 2:
            # Grayscale to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image_rgb.shape[:2]

        # Preprocess image
        input_tensor = self._preprocess(image_rgb)

        # Run inference
        with torch.no_grad():
            if self.model_name == 'maskrcnn':
                output = self.model([input_tensor])[0]
                seg_mask = self._process_maskrcnn_output(output, (h, w))
            else:
                output = self.model(input_tensor)['out'][0]
                seg_mask = self._process_segmentation_output(output, (h, w))

        # Create static mask (invert: 1=static, 0=dynamic)
        static_mask = (1 - seg_mask).astype(np.uint8)

        # Apply morphological operations to expand dynamic regions slightly
        # This helps ensure dynamic objects are fully masked
        if self.kernel is not None:
            seg_mask_expanded = cv2.dilate(seg_mask, self.kernel, iterations=1)
            static_mask = (1 - seg_mask_expanded).astype(np.uint8)

        return static_mask

    def _preprocess(self, image_rgb):
        """Preprocess image for model input.

        Args:
            image_rgb: RGB image (H, W, 3)

        Returns:
            tensor: Preprocessed tensor (1, 3, H, W)
        """
        # Convert to float and normalize
        image_float = image_rgb.astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_normalized = (image_float - mean) / std

        # Convert to tensor
        tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).float()
        tensor = tensor.unsqueeze(0).to(self.device)

        return tensor

    def _process_segmentation_output(self, output, original_size):
        """Process output from semantic segmentation models (DeepLab, FCN).

        Args:
            output: Model output tensor (C, H, W)
            original_size: (height, width) of original image

        Returns:
            mask: Binary mask (H, W) where 1=dynamic, 0=static
        """
        # Get class predictions
        predictions = output.argmax(0).cpu().numpy()  # (H, W)

        # Resize to original size if needed
        if predictions.shape != original_size:
            predictions = cv2.resize(predictions, (original_size[1], original_size[0]),
                                     interpolation=cv2.INTER_NEAREST)

        # Create binary mask for dynamic classes
        dynamic_mask = np.zeros(original_size, dtype=np.uint8)
        for class_id in self.dynamic_classes:
            dynamic_mask[predictions == class_id] = 1

        return dynamic_mask

    def _process_maskrcnn_output(self, output, original_size):
        """Process output from Mask R-CNN (instance segmentation).

        Args:
            output: Dictionary with 'boxes', 'labels', 'masks', 'scores'
            original_size: (height, width) of original image

        Returns:
            mask: Binary mask (H, W) where 1=dynamic, 0=static
        """
        dynamic_mask = np.zeros(original_size, dtype=np.uint8)

        # Filter by score threshold
        score_threshold = 0.5
        high_scores = output['scores'] > score_threshold

        labels = output['labels'][high_scores]
        masks = output['masks'][high_scores]

        # Combine masks for dynamic classes
        for i, label in enumerate(labels):
            if label.item() in self.dynamic_classes:
                mask = masks[i, 0].cpu().numpy()
                mask = cv2.resize(mask, (original_size[1], original_size[0]))
                dynamic_mask[mask > 0.5] = 1

        return dynamic_mask

    def apply_mask_to_keypoints(self, keypoints, mask):
        """Filter out keypoints in dynamic regions.

        Args:
            keypoints: List of cv2.KeyPoint objects
            mask: Binary mask (H, W) where 1=static, 0=dynamic

        Returns:
            filtered_keypoints: List of keypoints in static regions
            valid_indices: Indices of valid keypoints
        """
        filtered_kpts = []
        valid_indices = []

        for i, kp in enumerate(keypoints):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            # Check bounds
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                # Check if keypoint is in static region
                if mask[y, x] > 0:
                    filtered_kpts.append(kp)
                    valid_indices.append(i)

        return filtered_kpts, valid_indices


class FastSCNNMask(SemanticSegmentationMask):
    """Fast-SCNN based masking for real-time performance.

    Fast-SCNN is a lightweight semantic segmentation model designed for
    real-time applications. This class provides a wrapper for Fast-SCNN.

    Note: Requires separate installation of Fast-SCNN implementation.
    You can use: https://github.com/Tramac/Fast-SCNN-pytorch
    """

    def __init__(self, checkpoint_path=None, dataset='cityscapes', device='cuda',
                 dynamic_classes=None, erosion_kernel=5):
        """Initialize Fast-SCNN masking.

        Args:
            checkpoint_path: Path to Fast-SCNN checkpoint
            dataset: 'cityscapes' (Fast-SCNN is trained on Cityscapes)
            device: 'cuda' or 'cpu'
            dynamic_classes: Custom list of dynamic class IDs
            erosion_kernel: Size of erosion kernel
        """
        self.checkpoint_path = checkpoint_path
        self.dataset = dataset
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.erosion_kernel = erosion_kernel

        # Set dynamic classes
        if dynamic_classes is not None:
            self.dynamic_classes = set(dynamic_classes)
        else:
            self.dynamic_classes = set(self.DYNAMIC_CLASSES_CITYSCAPES.keys())

        # Load model
        try:
            self.model = self._load_fastscnn()
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Warning: Could not load Fast-SCNN: {e}")
            print("Falling back to DeepLabV3...")
            # Fallback to DeepLabV3
            super().__init__(model='deeplabv3', dataset=dataset, device=device,
                           dynamic_classes=dynamic_classes, erosion_kernel=erosion_kernel)
            return

        # Erosion kernel
        if erosion_kernel > 0:
            self.kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (erosion_kernel, erosion_kernel)
            )
        else:
            self.kernel = None

    def _load_fastscnn(self):
        """Load Fast-SCNN model."""
        try:
            # Try importing Fast-SCNN
            from models.fast_scnn import FastSCNN
            model = FastSCNN(num_classes=19)  # Cityscapes has 19 classes

            if self.checkpoint_path is not None:
                # Load checkpoint if provided
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                model.load_state_dict(checkpoint['state_dict'])

            return model
        except ImportError:
            raise ImportError(
                "Fast-SCNN not found. Please install from: "
                "https://github.com/Tramac/Fast-SCNN-pytorch or use 'deeplabv3' instead."
            )


class HybridMask:
    """Hybrid masking combining optical flow and semantic segmentation.

    This combines both approaches:
    - Semantic segmentation identifies potentially dynamic objects
    - Optical flow verifies which ones are actually moving

    This reduces false positives (e.g., parked cars marked as dynamic).
    """

    def __init__(self, flow_threshold=2.0, semantic_model='deeplabv3',
                 dataset='coco', device='cuda'):
        """Initialize hybrid masking.

        Args:
            flow_threshold: Optical flow threshold
            semantic_model: Semantic segmentation model name
            dataset: Dataset for segmentation model
            device: 'cuda' or 'cpu'
        """
        from .opticalflow_mask import OpticalFlowMask

        self.flow_mask = OpticalFlowMask(threshold=flow_threshold)
        self.semantic_mask = SemanticSegmentationMask(
            model=semantic_model, dataset=dataset, device=device
        )

    def __call__(self, image):
        """Compute hybrid mask.

        Args:
            image: Input image

        Returns:
            mask: Binary mask where 1=static, 0=dynamic
        """
        # Get flow-based mask
        flow_mask = self.flow_mask(image)

        # Get semantic mask
        semantic_mask = self.semantic_mask(image)

        # Combine: mark as dynamic if BOTH methods agree
        # (or if semantic says dynamic AND flow is non-trivial)
        # This is conservative but reduces false positives
        combined_mask = np.logical_and(flow_mask, semantic_mask).astype(np.uint8)

        return combined_mask

    def reset(self):
        """Reset flow mask state."""
        self.flow_mask.reset()
