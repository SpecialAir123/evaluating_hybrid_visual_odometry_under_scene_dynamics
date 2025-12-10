import cv2
import numpy as np


class OpticalFlowMask:
    """Dynamic object masking using optical flow analysis.

    This classical approach identifies dynamic objects by analyzing optical flow
    inconsistencies. Points with flow that deviates significantly from the
    dominant motion (camera ego-motion) are marked as dynamic.

    The method:
    1. Computes dense optical flow between consecutive frames
    2. Estimates dominant flow (ego-motion) using RANSAC on flow vectors
    3. Marks pixels with flow deviating from dominant motion as dynamic
    4. Returns a binary mask where 1=static, 0=dynamic
    """

    def __init__(self, flow_method='farneback', threshold=2.0, min_flow_magnitude=1.0):
        """Initialize optical flow masking.

        Args:
            flow_method: 'farneback' or 'dis' (Dense Inverse Search)
            threshold: Flow deviation threshold (in pixels) to mark as dynamic
            min_flow_magnitude: Minimum flow magnitude to consider (filter noise)
        """
        self.flow_method = flow_method
        self.threshold = threshold
        self.min_flow_magnitude = min_flow_magnitude
        self.prev_gray = None

        # Initialize flow algorithm
        if flow_method == 'farneback':
            self.flow_params = dict(
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
        elif flow_method == 'dis':
            self.flow_computer = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        else:
            raise ValueError(f"Unknown flow method: {flow_method}")

    def __call__(self, image):
        """Compute dynamic object mask for current frame.

        Args:
            image: Current frame (grayscale or BGR)

        Returns:
            mask: Binary mask (H, W) where 1=static, 0=dynamic
                  Returns None on first frame (no previous frame to compare)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # First frame: just store and return full valid mask
        if self.prev_gray is None:
            self.prev_gray = gray
            return np.ones_like(gray, dtype=np.uint8)

        # Compute optical flow
        flow = self._compute_flow(self.prev_gray, gray)

        # Compute flow magnitude
        flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

        # Filter out very small flow (noise/static regions)
        valid_flow_mask = flow_magnitude > self.min_flow_magnitude

        # Estimate dominant flow (ego-motion) using RANSAC
        dominant_flow = self._estimate_dominant_flow(flow, valid_flow_mask)

        # Compute deviation from dominant flow
        flow_deviation = flow - dominant_flow[None, None, :]
        deviation_magnitude = np.sqrt(flow_deviation[..., 0]**2 + flow_deviation[..., 1]**2)

        # Create mask: static (1) if deviation is small, dynamic (0) if large
        static_mask = (deviation_magnitude < self.threshold).astype(np.uint8)

        # Also mark regions with very small flow as static (likely background)
        static_mask[~valid_flow_mask] = 1

        # Update previous frame
        self.prev_gray = gray

        return static_mask

    def _compute_flow(self, prev_gray, gray):
        """Compute dense optical flow between two frames.

        Args:
            prev_gray: Previous frame (grayscale)
            gray: Current frame (grayscale)

        Returns:
            flow: Optical flow (H, W, 2) with (dx, dy) at each pixel
        """
        if self.flow_method == 'farneback':
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, **self.flow_params
            )
        elif self.flow_method == 'dis':
            flow = self.flow_computer.calc(prev_gray, gray, None)
        else:
            raise ValueError(f"Unknown flow method: {self.flow_method}")

        return flow

    def _estimate_dominant_flow(self, flow, valid_mask):
        """Estimate dominant flow using RANSAC (robust to outliers).

        Args:
            flow: Dense flow field (H, W, 2)
            valid_mask: Binary mask indicating valid flow pixels

        Returns:
            dominant_flow: (2,) array with median flow (dx, dy)
        """
        # Extract valid flow vectors
        valid_flow = flow[valid_mask]

        if len(valid_flow) < 10:
            # Not enough valid flow, return zero
            return np.array([0.0, 0.0])

        # Simple approach: use median (robust to outliers)
        # For camera motion, most of the scene should follow ego-motion
        dominant_flow = np.median(valid_flow, axis=0)

        # Alternative: Could use RANSAC to fit affine model, but median is simpler
        # and works well for our purpose

        return dominant_flow

    def reset(self):
        """Reset internal state (e.g., when starting a new sequence)."""
        self.prev_gray = None

    def apply_mask_to_keypoints(self, keypoints, mask):
        """Filter out keypoints that fall in dynamic regions.

        Args:
            keypoints: List of cv2.KeyPoint objects
            mask: Binary mask (H, W) where 1=static, 0=dynamic

        Returns:
            filtered_keypoints: List of keypoints in static regions
            valid_indices: Indices of valid keypoints in original list
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


class OpticalFlowMaskAdvanced(OpticalFlowMask):
    """Advanced optical flow masking with homography-based ego-motion estimation.

    This variant uses sparse feature matching + homography to better estimate
    camera ego-motion, making it more robust to large dynamic objects.
    """

    def __init__(self, flow_method='farneback', threshold=3.0, min_flow_magnitude=1.0,
                 use_homography=True):
        """Initialize advanced optical flow masking.

        Args:
            flow_method: 'farneback' or 'dis'
            threshold: Flow deviation threshold
            min_flow_magnitude: Minimum flow magnitude to consider
            use_homography: If True, use homography for ego-motion estimation
        """
        super().__init__(flow_method, threshold, min_flow_magnitude)
        self.use_homography = use_homography
        self.prev_keypoints = None

        # Feature detector for sparse matching (used in homography estimation)
        self.detector = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def __call__(self, image, keypoints=None):
        """Compute dynamic mask using homography-based ego-motion.

        Args:
            image: Current frame
            keypoints: Optional pre-detected keypoints for homography

        Returns:
            mask: Binary mask where 1=static, 0=dynamic
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # First frame
        if self.prev_gray is None:
            self.prev_gray = gray
            if self.use_homography:
                kp, desc = self.detector.detectAndCompute(gray, None)
                self.prev_keypoints = kp
                self.prev_descriptors = desc
            return np.ones_like(gray, dtype=np.uint8)

        # Compute optical flow
        flow = self._compute_flow(self.prev_gray, gray)
        flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        valid_flow_mask = flow_magnitude > self.min_flow_magnitude

        # Estimate dominant flow
        if self.use_homography and self.prev_keypoints is not None:
            # Use homography for better ego-motion estimation
            dominant_flow_field = self._estimate_flow_from_homography(gray)
            if dominant_flow_field is not None:
                flow_deviation = flow - dominant_flow_field
                deviation_magnitude = np.sqrt(flow_deviation[..., 0]**2 +
                                             flow_deviation[..., 1]**2)
                static_mask = (deviation_magnitude < self.threshold).astype(np.uint8)
                static_mask[~valid_flow_mask] = 1

                # Update state
                self.prev_gray = gray
                kp, desc = self.detector.detectAndCompute(gray, None)
                self.prev_keypoints = kp
                self.prev_descriptors = desc

                return static_mask

        # Fallback to simple median-based approach
        dominant_flow = self._estimate_dominant_flow(flow, valid_flow_mask)
        flow_deviation = flow - dominant_flow[None, None, :]
        deviation_magnitude = np.sqrt(flow_deviation[..., 0]**2 + flow_deviation[..., 1]**2)
        static_mask = (deviation_magnitude < self.threshold).astype(np.uint8)
        static_mask[~valid_flow_mask] = 1

        # Update state
        self.prev_gray = gray
        if self.use_homography:
            kp, desc = self.detector.detectAndCompute(gray, None)
            self.prev_keypoints = kp
            self.prev_descriptors = desc

        return static_mask

    def _estimate_flow_from_homography(self, gray):
        """Estimate ego-motion flow field using homography.

        Args:
            gray: Current frame (grayscale)

        Returns:
            flow_field: Dense flow field (H, W, 2) representing ego-motion
                       Returns None if homography estimation fails
        """
        # Detect features in current frame
        kp_curr, desc_curr = self.detector.detectAndCompute(gray, None)

        if desc_curr is None or self.prev_descriptors is None:
            return None

        # Match features
        matches = self.matcher.match(self.prev_descriptors, desc_curr)

        if len(matches) < 10:
            return None

        # Extract matched keypoints
        pts_prev = np.float32([self.prev_keypoints[m.queryIdx].pt for m in matches])
        pts_curr = np.float32([kp_curr[m.trainIdx].pt for m in matches])

        # Estimate homography with RANSAC
        H, mask = cv2.findHomography(pts_prev, pts_curr, cv2.RANSAC, 3.0)

        if H is None:
            return None

        # Create dense flow field from homography
        h, w = gray.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        coords = np.stack([x_coords, y_coords, np.ones_like(x_coords)], axis=-1)

        # Warp coordinates using homography
        coords_flat = coords.reshape(-1, 3).T
        warped_flat = H @ coords_flat
        warped_flat = warped_flat / warped_flat[2:3, :]  # Normalize
        warped = warped_flat[:2, :].T.reshape(h, w, 2)

        # Compute flow
        original = np.stack([x_coords, y_coords], axis=-1)
        flow_field = warped - original

        return flow_field

    def reset(self):
        """Reset internal state."""
        super().reset()
        self.prev_keypoints = None
        self.prev_descriptors = None
