import cv2
import numpy as np


class KNNMatcher:
    def __init__(self, ratio=0.75):
        self.ratio = ratio
        # Will be set based on descriptor type
        self.matcher = None

    def _get_matcher(self, desc):
        """Determine matcher type based on descriptor dtype."""
        if desc is None or desc.size == 0:
            return None
        
        # Check if binary descriptor (uint8) or float descriptor
        if desc.dtype == np.uint8:
            # Binary descriptor (ORB) - use Hamming distance
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            # Float descriptor (SuperPoint, DISK) - use L2 distance
            return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def __call__(self, desc1, desc2):
        if desc1 is None or desc2 is None or desc1.size == 0 or desc2.size == 0:
            return []

        # Get appropriate matcher based on descriptor type
        matcher = self._get_matcher(desc1)
        if matcher is None:
            return []

        knn_matches = matcher.knnMatch(desc1, desc2, k=2)
        good = []

        for match_pair in knn_matches:
            if len(match_pair) < 2:
                continue
            m, n = match_pair[0], match_pair[1]
            if m.distance < self.ratio * n.distance:
                good.append(m)
        return good

