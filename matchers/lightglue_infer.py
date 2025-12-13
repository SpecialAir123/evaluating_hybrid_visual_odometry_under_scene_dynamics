import torch
import cv2
import numpy as np
import kornia.feature as KF

class LightGlueMatcher:
    def __init__(self,
                 features='disk',
                 depth_confidence=-1,
                 width_confidence=-1,
                 filter_threshold=0.1,
                 use_cuda=True):
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        
        # features can be 'superpoint', 'disk', etc.
        self.net = KF.LightGlue(features=features, 
                                depth_confidence=depth_confidence, 
                                width_confidence=width_confidence,
                                filter_threshold=filter_threshold).to(self.device)
        self.net.eval()

    def __call__(self, kp1, desc1, kp2, desc2, image_shape=None):
        """
        kp1, kp2: list[cv2.KeyPoint]
        desc1, desc2: np.ndarray (N,D)
        image_shape: (h, w) tuple
        """
        if desc1 is None or desc2 is None or len(kp1) == 0 or len(kp2) == 0:
            return []
            
        # Convert KeyPoints to coordinates (N, 2)
        pts1 = np.array([[k.pt[0], k.pt[1]] for k in kp1], dtype=np.float32)
        pts2 = np.array([[k.pt[0], k.pt[1]] for k in kp2], dtype=np.float32)
        
        # Convert to tensor and move to device
        # LightGlue expects batch dimension (B, N, 2) and (B, N, D)
        t_pts1 = torch.from_numpy(pts1)[None].to(self.device)
        t_pts2 = torch.from_numpy(pts2)[None].to(self.device)
        t_desc1 = torch.from_numpy(desc1)[None].to(self.device)
        t_desc2 = torch.from_numpy(desc2)[None].to(self.device)
        
        # Prepare input dict matching Kornia LightGlue expectation
        # It expects data = {"image0": {...}, "image1": {...}}
        
        data = {
            "image0": {
                "keypoints": t_pts1,
                "descriptors": t_desc1,
            },
            "image1": {
                "keypoints": t_pts2,
                "descriptors": t_desc2,
            }
        }
        
        if image_shape is not None:
            # image_shape is (h, w). Convert to (w, h) for normalization of x,y
            h, w = image_shape
            size_tensor = torch.tensor([w, h], dtype=torch.float32).unsqueeze(0).to(self.device)
            data["image0"]["image_size"] = size_tensor
            data["image1"]["image_size"] = size_tensor

        with torch.no_grad():
            out = self.net(data)
            
        # Output structure: out['matches0'] is (B, M)
        matches = out['matches0'][0].cpu().numpy() # (N1,) containing index of match in image1 or -1
        scores = out['matching_scores0'][0].cpu().numpy() # (N1,)
        
        good_matches = []
        for i, j in enumerate(matches):
            if j > -1:
                idx = int(j)
                score = float(scores[i])
                m = cv2.DMatch(_queryIdx=i, _trainIdx=idx, 
                               _imgIdx=0, _distance=1.0 - score)
                good_matches.append(m)
                
        return good_matches
