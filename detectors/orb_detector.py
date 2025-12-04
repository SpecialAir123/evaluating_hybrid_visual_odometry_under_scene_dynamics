import cv2


class ORBDetector:
    def __init__(self, nfeatures=2000):
        self.orb = cv2.ORB_create(nfeatures=nfeatures)

    def __call__(self, image):
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        return keypoints, descriptors

