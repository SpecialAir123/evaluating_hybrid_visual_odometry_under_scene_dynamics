import cv2
import torch
import numpy as np


class LightGlue:
    """LightGlue feature matcher.

    Reference: LightGlue: Local Feature Matching at Light Speed
    Paper: https://arxiv.org/abs/2306.13643

    This implementation uses the official pretrained model optimized for SuperPoint features.
    """

    default_config = {
        'depth_confidence': 0.95,  # Early stopping confidence threshold
        'width_confidence': 0.99,  # Point pruning confidence threshold
        'filter_threshold': 0.1,  # Match confidence threshold
        'flash': True,  # Use flash attention if available
    }

    def __init__(self, features='superpoint', config=None, device='cuda'):
        """Initialize LightGlue matcher.

        Args:
            features: Type of features to match ('superpoint', 'disk', 'aliked', 'sift')
            config: Dictionary with configuration parameters
            device: 'cuda' or 'cpu'
        """
        self.config = {**self.default_config, **(config or {})}
        self.features = features
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'

        # Load pretrained model
        self.net = self._build_model()
        self.net.to(self.device)
        self.net.eval()

    def _build_model(self):
        """Build LightGlue network."""
        try:
            # Try loading from torch hub
            extractor = self.features.lower()
            model = torch.hub.load('cvg/LightGlue', 'lightglue',
                                    features=extractor, pretrained=True,
                                    force_reload=False)
            return model
        except Exception as e:
            print(f"Warning: Could not load LightGlue from torch hub: {e}")
            print("Attempting to load using pip package...")
            try:
                # Try importing from installed package
                from lightglue import LightGlue as LG
                return LG(features=self.features, **self.config)
            except ImportError:
                print("Error: LightGlue package not found. Please install it:")
                print("  pip install git+https://github.com/cvg/LightGlue.git")
                raise

    def __call__(self, desc1, desc2, kpts1=None, kpts2=None):
        """Match descriptors between two images.

        Args:
            desc1: Descriptors from image 1, shape (N1, D)
            desc2: Descriptors from image 2, shape (N2, D)
            kpts1: Keypoints from image 1, shape (N1, 2) or list of cv2.KeyPoint
            kpts2: Keypoints from image 2, shape (N2, 2) or list of cv2.KeyPoint

        Returns:
            matches: List of cv2.DMatch objects for compatibility with existing pipeline
        """
        if desc1 is None or desc2 is None:
            return []

        if len(desc1) == 0 or len(desc2) == 0:
            return []

        # Convert to numpy arrays if needed
        if isinstance(desc1, torch.Tensor):
            desc1 = desc1.cpu().numpy()
        if isinstance(desc2, torch.Tensor):
            desc2 = desc2.cpu().numpy()

        # Extract keypoint coordinates
        if kpts1 is None or kpts2 is None:
            # If keypoints not provided, create dummy coordinates
            kpts1_np = np.zeros((len(desc1), 2), dtype=np.float32)
            kpts2_np = np.zeros((len(desc2), 2), dtype=np.float32)
        else:
            # Convert cv2.KeyPoint to numpy array if needed
            if isinstance(kpts1, list) and len(kpts1) > 0 and isinstance(kpts1[0], cv2.KeyPoint):
                kpts1_np = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts1], dtype=np.float32)
            else:
                kpts1_np = np.array(kpts1, dtype=np.float32)

            if isinstance(kpts2, list) and len(kpts2) > 0 and isinstance(kpts2[0], cv2.KeyPoint):
                kpts2_np = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts2], dtype=np.float32)
            else:
                kpts2_np = np.array(kpts2, dtype=np.float32)

        # Convert to torch tensors
        kpts1_tensor = torch.from_numpy(kpts1_np).float().to(self.device)[None]  # (1, N1, 2)
        kpts2_tensor = torch.from_numpy(kpts2_np).float().to(self.device)[None]  # (1, N2, 2)
        desc1_tensor = torch.from_numpy(desc1).float().to(self.device)[None]  # (1, N1, D)
        desc2_tensor = torch.from_numpy(desc2).float().to(self.device)[None]  # (1, N2, D)

        # Prepare input dictionary
        data = {
            'keypoints0': kpts1_tensor,
            'keypoints1': kpts2_tensor,
            'descriptors0': desc1_tensor.transpose(1, 2),  # (1, D, N1)
            'descriptors1': desc2_tensor.transpose(1, 2),  # (1, D, N2)
        }

        # Run matching
        with torch.no_grad():
            pred = self.net(data)

        # Extract matches
        matches0 = pred['matches0'][0].cpu().numpy()  # (N1,) indices into desc2, -1 if no match
        matching_scores = pred['matching_scores0'][0].cpu().numpy()  # (N1,)

        # Convert to cv2.DMatch format for compatibility
        cv_matches = []
        for idx1, idx2 in enumerate(matches0):
            if idx2 >= 0:  # Valid match
                distance = 1.0 - matching_scores[idx1]  # Convert confidence to distance
                match = cv2.DMatch(_queryIdx=idx1, _trainIdx=int(idx2),
                                   _distance=float(distance))
                cv_matches.append(match)

        return cv_matches


class LightGlueWrapper:
    """Wrapper class that accepts keypoints directly during matching.

    This is useful when the matcher needs access to keypoint locations,
    which is the case for LightGlue (unlike traditional descriptor matchers).
    """

    def __init__(self, features='superpoint', config=None, device='cuda'):
        """Initialize LightGlue wrapper.

        Args:
            features: Type of features to match ('superpoint', 'disk', etc.)
            config: Configuration dictionary
            device: 'cuda' or 'cpu'
        """
        self.matcher = LightGlue(features=features, config=config, device=device)
        self.kpts1_cache = None
        self.kpts2_cache = None

    def set_keypoints(self, kpts1, kpts2):
        """Set keypoints for the next matching call.

        This is a workaround since the existing pipeline passes only descriptors
        to the matcher, but LightGlue needs keypoint locations too.
        """
        self.kpts1_cache = kpts1
        self.kpts2_cache = kpts2

    def __call__(self, desc1, desc2):
        """Match descriptors using cached keypoints.

        Args:
            desc1: Descriptors from image 1
            desc2: Descriptors from image 2

        Returns:
            matches: List of cv2.DMatch objects
        """
        return self.matcher(desc1, desc2, self.kpts1_cache, self.kpts2_cache)


class LightGlueMatcherAdapter:
    """Adapter that modifies the detector to also cache keypoints for LightGlue.

    Usage in main.py:
        detector = SuperPoint()
        matcher = LightGlueMatcherAdapter(detector, device='cuda')

        # In the loop:
        kp1, desc1 = matcher.detect(prev_img)  # This caches kp1
        kp2, desc2 = matcher.detect(img)       # This caches kp2
        matches = matcher.match(desc1, desc2)   # Uses cached keypoints
    """

    def __init__(self, detector, features='superpoint', config=None, device='cuda'):
        """Initialize adapter.

        Args:
            detector: The feature detector (e.g., SuperPoint instance)
            features: Type of features ('superpoint', 'disk', etc.)
            config: LightGlue configuration
            device: 'cuda' or 'cpu'
        """
        self.detector = detector
        self.matcher = LightGlue(features=features, config=config, device=device)
        self.kpts_history = []  # Store last 2 keypoint sets

    def detect(self, image):
        """Detect features and cache keypoints.

        Args:
            image: Input image

        Returns:
            keypoints: List of cv2.KeyPoint
            descriptors: Numpy array of descriptors
        """
        kpts, descs = self.detector(image)
        self.kpts_history.append(kpts)
        if len(self.kpts_history) > 2:
            self.kpts_history.pop(0)
        return kpts, descs

    def match(self, desc1, desc2):
        """Match descriptors using cached keypoints.

        Args:
            desc1: Descriptors from first image
            desc2: Descriptors from second image

        Returns:
            matches: List of cv2.DMatch objects
        """
        if len(self.kpts_history) < 2:
            print("Warning: Not enough cached keypoints, returning empty matches")
            return []

        kpts1 = self.kpts_history[-2]
        kpts2 = self.kpts_history[-1]

        return self.matcher(desc1, desc2, kpts1, kpts2)

    def __call__(self, desc1, desc2):
        """Alias for match() to maintain compatibility with matcher interface.

        Args:
            desc1: Descriptors from first image
            desc2: Descriptors from second image

        Returns:
            matches: List of cv2.DMatch objects
        """
        return self.match(desc1, desc2)
