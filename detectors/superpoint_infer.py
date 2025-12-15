import cv2
import numpy as np
import torch

from third_party.SuperGluePretrainedNetwork.models.superpoint import SuperPoint

class SuperPointDetector:
    def __init__(self,
                 weights_path=None,
                 max_keypoints=1024,
                 keypoint_threshold=0.005,
                 nms_radius=4,
                 use_cuda=True):
        if weights_path is None:
            weights_path = "third_party/SuperGluePretrainedNetwork/models/weights/superpoint_v1.pth"

        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.net = SuperPoint({
            "descriptor_dim": 256,
            "nms_radius": nms_radius,
            "keypoint_threshold": keypoint_threshold,
            "max_keypoints": max_keypoints
        }).to(self.device)
        state_dict = torch.load(str(weights_path), map_location=self.device)
        self.net.load_state_dict(state_dict)
        self.net.eval()

    def detect_and_compute(self, image):
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        h, w = image.shape[:2]
        
        # [0,1] float, shape: (1,1,H,W)
        img = image.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img)[None, None].to(self.device)

        with torch.no_grad():
            pred = self.net({"image": img_tensor})

        kps = pred["keypoints"][0].cpu().numpy()           # (N,2) [x,y]
        desc = pred["descriptors"][0].cpu().numpy().T      # (N,D)
        scores = pred["scores"][0].cpu().numpy()           # (N,) - keypoint confidence scores

        # Handle empty keypoints case
        if len(kps) == 0:
            return [], np.array([], dtype=np.float32).reshape(0, 256), np.array([], dtype=np.float32)

        # Validate and filter keypoints within image bounds
        valid_mask = (kps[:, 0] >= 0) & (kps[:, 0] < w) & (kps[:, 1] >= 0) & (kps[:, 1] < h)
        kps = kps[valid_mask]
        desc = desc[valid_mask]
        scores = scores[valid_mask]
        
        # If no valid keypoints after filtering, return empty
        if len(kps) == 0:
            return [], np.array([], dtype=np.float32).reshape(0, 256), np.array([], dtype=np.float32)

        # Convert to OpenCV KeyPoint format
        # Use score as size (multiply by reasonable scale factor for visualization)
        # Size is typically 1-31 pixels, so scale score appropriately
        keypoints = [cv2.KeyPoint(float(x), float(y), float(score * 10.0 + 1.0)) 
                     for (x, y), score in zip(kps, scores)]
        
        desc = desc.astype(np.float32)
        scores = scores.astype(np.float32)
        
        # Explicitly L2-normalize descriptors for proper distance computation
        # (SuperPoint model already normalizes, but ensure it's done here for safety)
        # Only normalize if we have descriptors
        if len(desc) > 0:
            norms = np.linalg.norm(desc, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # Avoid division by zero
            desc = desc / norms

        return keypoints, desc, scores
    
    def __call__(self, image):
        """Interface to match ORBDetector API. Returns (keypoints, descriptors, scores)."""
        return self.detect_and_compute(image)