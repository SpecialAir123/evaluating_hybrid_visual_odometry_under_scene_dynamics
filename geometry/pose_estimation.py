import cv2
import numpy as np


class PoseEstimator:
    def __init__(self, K, ransac_threshold=0.999, ransac_confidence=0.999, min_matches=8):
        self.K = K
        self.ransac_threshold = ransac_threshold
        # Validate confidence: OpenCV requires 0 < confidence < 1
        if ransac_confidence >= 1.0 or ransac_confidence <= 0.0:
            raise ValueError(f"ransac_confidence must be between 0 and 1 (exclusive), got {ransac_confidence}")
        self.ransac_confidence = ransac_confidence
        self.min_matches = min_matches
        self.prev_R = np.eye(3)
        self.prev_t = np.zeros((3, 1))

    def estimate(self, kp1, kp2, matches):
        if len(matches) < self.min_matches:
            return None, None, None

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Debug: print values to verify
        # Ensure confidence is strictly between 0 and 1
        confidence = float(self.ransac_confidence)
        if confidence >= 1.0:
            confidence = 0.999
        elif confidence <= 0.0:
            confidence = 0.99
        
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, 
                                       threshold=self.ransac_threshold, 
                                       prob=confidence)

        if E is None:
            return None, None, None

        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K)

        return R, t, mask

