import numpy as np
import torch
import cv2

from third_party.SuperGluePretrainedNetwork.models.superglue import SuperGlue

class SuperGlueMatcher:
    def __init__(self,
                 weights_path=None,
                 sinkhorn_iterations=20,
                 match_threshold=0.2,
                 use_cuda=True):
        if weights_path is None:
            weights_path = "third_party/SuperGluePretrainedNetwork/models/weights/superglue_outdoor.pth"

        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.net = SuperGlue({
            "descriptor_dim": 256,
            "weights": "outdoor",
            "sinkhorn_iterations": sinkhorn_iterations,
            "match_threshold": match_threshold
        }).to(self.device)
        self.net.eval()
        self.match_threshold = match_threshold

    def __call__(self, kp1, desc1, kp2, desc2, image_shape=None):
        """
        kp1, kp2: list[cv2.KeyPoint]
        desc1, desc2: np.ndarray (N,D)
        返回: list[cv2.DMatch]
        """
        if desc1 is None or desc2 is None or len(kp1) == 0 or len(kp2) == 0:
            return []

        pts1 = np.array([[k.pt[0], k.pt[1]] for k in kp1], dtype=np.float32)
        pts2 = np.array([[k.pt[0], k.pt[1]] for k in kp2], dtype=np.float32)

        t_pts1 = torch.from_numpy(pts1)[None].to(self.device)        # (1,N1,2)
        t_pts2 = torch.from_numpy(pts2)[None].to(self.device)        # (1,N2,2)
        t_desc1 = torch.from_numpy(desc1.T)[None].to(self.device)    # (1,D,N1)
        t_desc2 = torch.from_numpy(desc2.T)[None].to(self.device)    # (1,D,N2)

        if image_shape is None:
            h = int(max(pts1[:,1].max(), pts2[:,1].max()) + 1)
            w = int(max(pts1[:,0].max(), pts2[:,0].max()) + 1)
            image_shape = (h, w)

        h, w = image_shape

        data = {
            "keypoints0": t_pts1,
            "keypoints1": t_pts2,
            "descriptors0": t_desc1,
            "descriptors1": t_desc2,
            "image0": torch.zeros(1, 1, h, w, device=self.device),  # 占位即可
            "image1": torch.zeros(1, 1, h, w, device=self.device)
        }

        with torch.no_grad():
            pred = self.net(data)

        matches0 = pred["matches0"][0].cpu().numpy()
        scores0 = pred["matching_scores0"][0].cpu().numpy()

        good_matches = []
        for i, j in enumerate(matches0):
            if j < 0:
                continue
            score = float(scores0[i])
            if score < self.match_threshold:
                continue
            m = cv2.DMatch(_queryIdx=i, _trainIdx=int(j),
                           _imgIdx=0, _distance=1.0 - score)
            good_matches.append(m)

        return good_matches
