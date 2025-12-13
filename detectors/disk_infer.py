import cv2
import numpy as np
import torch
import kornia.feature as KF
from kornia.geometry import resize

class DISKDetector:
    def __init__(self,
                 nfeatures=2000,
                 window_size=5,
                 desc_dim=128,
                 use_cuda=True):
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        # DISK.from_pretrained('depth') is the standard model
        self.net = KF.DISK.from_pretrained('depth').to(self.device)
        self.nfeatures = nfeatures
        
        # We can pass limits to the forward pass or filter afterwards.
        # Kornia DISK returns all detected features.
        
    def detect_and_compute(self, image):
        # Handle grayscale/RGB
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Prepare input: (B, C, H, W), float, [0, 1]
        img_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Kornia DISK forward returns a list of Features objects (one per batch item)
            # n=nfeatures limits the number of features kept
            features = self.net(img_tensor, n=self.nfeatures, window_size=5, score_threshold=0.0, pad_if_not_divisible=True)
            
        # Extract features from the first (and only) batch element
        feat = features[0]
        
        # Keypoints: (N, 2)
        kps_cpu = feat.keypoints.cpu().numpy()
        # Descriptors: (N, D)
        desc_cpu = feat.descriptors.cpu().numpy()
        # Scores: (N,)
        scores_cpu = feat.detection_scores.cpu().numpy()
        
        # Convert to cv2.KeyPoint
        keypoints = []
        for i in range(len(kps_cpu)):
            x, y = kps_cpu[i]
            score = scores_cpu[i]
            kp = cv2.KeyPoint(float(x), float(y), 1.0, response=float(score))
            keypoints.append(kp)
            
        return keypoints, desc_cpu

    def __call__(self, image):
        """Alias for detect_and_compute to match other detectors if needed"""
        return self.detect_and_compute(image)

