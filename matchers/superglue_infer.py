import numpy as np
import torch
import cv2

from third_party.SuperGluePretrainedNetwork.models.superglue import SuperGlue

class SuperGlueMatcher:
    def __init__(self,
                 weights_path=None,
                 weights_type="indoor",  # "indoor" or "outdoor"
                 sinkhorn_iterations=20,
                 match_threshold=0.2,
                 use_cuda=True):
        # Determine weights path and type
        if weights_path is None:
            if weights_type == "indoor":
                weights_path = "third_party/SuperGluePretrainedNetwork/models/weights/superglue_indoor.pth"
                weights_name = "indoor"
            elif weights_type == "outdoor":
                weights_path = "third_party/SuperGluePretrainedNetwork/models/weights/superglue_outdoor.pth"
                weights_name = "outdoor"
            else:
                raise ValueError(f"weights_type must be 'indoor' or 'outdoor', got '{weights_type}'")
        else:
            # Infer weights type from path if not specified
            if "indoor" in weights_path.lower():
                weights_name = "indoor"
            elif "outdoor" in weights_path.lower():
                weights_name = "outdoor"
            else:
                # Default to indoor if can't determine
                weights_name = "indoor"

        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        # SuperGlue automatically loads weights based on weights_name
        self.net = SuperGlue({
            "descriptor_dim": 256,
            "weights": weights_name,  # "indoor" or "outdoor" - model auto-loads from weights/
            "sinkhorn_iterations": sinkhorn_iterations,
            "match_threshold": match_threshold
        }).to(self.device)
        self.net.eval()
        self.match_threshold = match_threshold

    def __call__(self, kp1, desc1, kp2, desc2, image_shape=None, scores1=None, scores2=None):
        """
        kp1, kp2: list[cv2.KeyPoint]
        desc1, desc2: np.ndarray (N,D)
        scores1, scores2: optional np.ndarray (N,) - keypoint scores
        返回: list[cv2.DMatch]
        """
        if desc1 is None or desc2 is None or len(kp1) == 0 or len(kp2) == 0:
            return []

        pts1 = np.array([[k.pt[0], k.pt[1]] for k in kp1], dtype=np.float32)
        pts2 = np.array([[k.pt[0], k.pt[1]] for k in kp2], dtype=np.float32)

        t_pts1 = torch.from_numpy(pts1)[None].to(self.device)        # (1,N1,2)
        t_pts2 = torch.from_numpy(pts2)[None].to(self.device)        # (1,N2,2)
        
        # Convert descriptors to float32 if they're binary (ORB uses uint8)
        # SuperGlue expects 256-dimensional float descriptors
        if desc1.dtype == np.uint8:
            # ORB descriptors are 32 bytes (256 bits) - unpack to 256 dimensions
            # Unpack each byte into 8 bits: [byte0_bits, byte1_bits, ..., byte31_bits]
            def unpack_binary_descriptor(desc):
                """Unpack binary descriptor (N, 32) uint8 -> (N, 256) float32"""
                N = desc.shape[0]
                # Unpack each byte into 8 bits
                unpacked = np.unpackbits(desc, axis=1).astype(np.float32)
                # Normalize to [0, 1] range (bits are already 0 or 1)
                return unpacked
            
            desc1 = unpack_binary_descriptor(desc1)
            desc2 = unpack_binary_descriptor(desc2)
            # L2-normalize for consistency with SuperPoint descriptors
            norms1 = np.linalg.norm(desc1, axis=1, keepdims=True)
            norms2 = np.linalg.norm(desc2, axis=1, keepdims=True)
            norms1[norms1 == 0] = 1.0
            norms2[norms2 == 0] = 1.0
            desc1 = desc1 / norms1
            desc2 = desc2 / norms2
        
        t_desc1 = torch.from_numpy(desc1.T)[None].to(self.device)    # (1,D,N1)
        t_desc2 = torch.from_numpy(desc2.T)[None].to(self.device)    # (1,D,N2)

        # Use provided scores or generate dummy scores (SuperGlue requires scores)
        if scores1 is not None:
            t_scores1 = torch.from_numpy(scores1)[None].to(self.device)  # (1,N1)
        else:
            t_scores1 = torch.ones(1, len(kp1), device=self.device)    # (1,N1)
        
        if scores2 is not None:
            t_scores2 = torch.from_numpy(scores2)[None].to(self.device)  # (1,N2)
        else:
            t_scores2 = torch.ones(1, len(kp2), device=self.device)    # (1,N2)

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
            "scores0": t_scores1,
            "scores1": t_scores2,
            "image0": torch.zeros(1, 1, h, w, device=self.device),
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
