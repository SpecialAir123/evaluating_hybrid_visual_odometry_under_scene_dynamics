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

    def __call__(self, image):
        """image: 单通道 uint8 (H,W)，返回 cv2.KeyPoint 列表 + 描述子 np.ndarray (N,D)"""
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # [0,1] float, shape: (1,1,H,W)
        img = image.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img)[None, None].to(self.device)

        with torch.no_grad():
            pred = self.net({"image": img_tensor})

        # pred 通常包含 "keypoints" (B, N, 2) 和 "descriptors" (B, D, N)
        kps = pred["keypoints"][0].cpu().numpy()           # (N,2) [x,y]
        desc = pred["descriptors"][0].cpu().numpy().T      # (N,D)

        keypoints = [cv2.KeyPoint(float(x), float(y), 1.0) for (x, y) in kps]
        desc = desc.astype(np.float32)

        return keypoints, desc
