import cv2
import numpy as np


class KNNMatcher:
    def __init__(self, ratio=0.75):
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.ratio = ratio

    def __call__(self, desc1, desc2):
        if desc1 is None or desc2 is None:
            return []

        knn_matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good = []

        for m, n in knn_matches:
            if m.distance < self.ratio * n.distance:
                good.append(m)
        return good

