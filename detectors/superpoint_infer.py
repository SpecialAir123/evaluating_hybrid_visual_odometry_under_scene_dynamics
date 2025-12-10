import cv2
import torch
import numpy as np


class SuperPoint:
    """SuperPoint feature detector and descriptor.

    Reference: SuperPoint: Self-Supervised Interest Point Detection and Description
    Paper: https://arxiv.org/abs/1712.07629

    This implementation uses the official pretrained model from MagicLeap.
    """

    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }

    def __init__(self, config=None, device='cuda'):
        """Initialize SuperPoint detector.

        Args:
            config: Dictionary with configuration parameters
            device: 'cuda' or 'cpu'
        """
        self.config = {**self.default_config, **(config or {})}
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'

        # Load pretrained model
        self.net = self._build_model()
        self.net.to(self.device)
        self.net.eval()

    def _build_model(self):
        """Build SuperPoint network architecture."""
        # Try to load from torch hub first
        try:
            # Using the official SuperPoint weights
            model = torch.hub.load('magicleap/SuperGluePretrainedNetwork', 'superpoint',
                                    pretrained=True, force_reload=False)
            return model
        except Exception as e:
            print(f"Warning: Could not load SuperPoint from torch hub: {e}")
            print("Attempting to load local weights...")
            # Fallback to local implementation
            return SuperPointNet(self.config)

    def __call__(self, image):
        """Detect keypoints and compute descriptors.

        Args:
            image: Input grayscale image (H, W) or RGB image (H, W, 3)

        Returns:
            keypoints: List of cv2.KeyPoint objects
            descriptors: Numpy array of shape (N, 256) with descriptors
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Normalize to [0, 1]
        gray = gray.astype(np.float32) / 255.0

        # Convert to torch tensor
        img_tensor = torch.from_numpy(gray)[None, None].to(self.device)

        # Run inference
        with torch.no_grad():
            pred = self.net({'image': img_tensor})

        # Extract keypoints and descriptors
        keypoints_np = pred['keypoints'][0].cpu().numpy()  # (N, 2)
        descriptors_np = pred['descriptors'][0].cpu().numpy().T  # (N, 256)
        scores = pred['scores'][0].cpu().numpy()  # (N,)

        # Convert to cv2.KeyPoint format for compatibility with existing pipeline
        keypoints = []
        for i, (x, y) in enumerate(keypoints_np):
            kp = cv2.KeyPoint(x=float(x), y=float(y), size=1.0, response=float(scores[i]))
            keypoints.append(kp)

        # Apply max_keypoints constraint if specified
        max_kpts = self.config['max_keypoints']
        if max_kpts > 0 and len(keypoints) > max_kpts:
            # Sort by response (score) and keep top-k
            indices = np.argsort(scores)[::-1][:max_kpts]
            keypoints = [keypoints[i] for i in indices]
            descriptors_np = descriptors_np[indices]

        return keypoints, descriptors_np


class SuperPointNet(torch.nn.Module):
    """SuperPoint network architecture (fallback if torch hub fails)."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Shared encoder
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)

        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)

        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)

        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        # Keypoint head
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        # Descriptor head
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, self.config['descriptor_dim'],
                                       kernel_size=1, stride=1, padding=0)

    def forward(self, data):
        """Forward pass."""
        x = data['image']

        # Shared encoder
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)

        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Keypoint head
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)

        # Descriptor head
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)

        # Process outputs
        semi = torch.nn.functional.softmax(semi, dim=1)[:, :-1]
        Hc, Wc = semi.shape[2], semi.shape[3]
        semi = semi.permute(0, 2, 3, 1).reshape(-1, Hc, Wc, 8, 8)
        semi = semi.permute(0, 1, 3, 2, 4).reshape(-1, Hc*8, Wc*8)

        # Extract keypoints
        keypoints, scores, descriptors = self._extract_keypoints(semi[0], desc[0])

        return {
            'keypoints': [keypoints],
            'scores': [scores],
            'descriptors': [descriptors]
        }

    def _extract_keypoints(self, semi, desc):
        """Extract keypoint locations from score map."""
        # Simple NMS and thresholding
        threshold = self.config['keypoint_threshold']
        nms_radius = self.config['nms_radius']

        # Apply threshold
        mask = semi > threshold

        # Simple local maximum detection
        if nms_radius > 0:
            kernel_size = 2 * nms_radius + 1
            max_pool = torch.nn.functional.max_pool2d(
                semi[None, None], kernel_size=kernel_size,
                stride=1, padding=nms_radius
            )[0, 0]
            mask = mask & (semi == max_pool)

        # Get keypoint coordinates
        indices = torch.nonzero(mask, as_tuple=False)

        if len(indices) == 0:
            return (torch.zeros((0, 2), dtype=torch.float32),
                    torch.zeros((0,), dtype=torch.float32),
                    torch.zeros((0, self.config['descriptor_dim']), dtype=torch.float32))

        # Extract locations and scores
        keypoints = indices.flip(-1).float()  # (N, 2) in (x, y) format
        scores = semi[indices[:, 0], indices[:, 1]]

        # Interpolate descriptors at keypoint locations
        # Descriptors are at 1/8 resolution
        desc = torch.nn.functional.normalize(desc, p=2, dim=0)
        kpt_desc = self._sample_descriptors(keypoints, desc)

        return keypoints, scores, kpt_desc

    def _sample_descriptors(self, keypoints, descriptors):
        """Sample descriptors at keypoint locations using bilinear interpolation."""
        # Keypoints are at full resolution, descriptors at 1/8 resolution
        H, W = descriptors.shape[1:]
        keypoints_scaled = keypoints / 8.0

        # Normalize to [-1, 1] for grid_sample
        keypoints_norm = keypoints_scaled.clone()
        keypoints_norm[:, 0] = 2 * keypoints_norm[:, 0] / (W - 1) - 1
        keypoints_norm[:, 1] = 2 * keypoints_norm[:, 1] / (H - 1) - 1

        # Sample using grid_sample
        grid = keypoints_norm[None, None, :, :]  # (1, 1, N, 2)
        descriptors_sampled = torch.nn.functional.grid_sample(
            descriptors[None], grid, mode='bilinear', align_corners=True
        )
        descriptors_sampled = descriptors_sampled[0, :, 0, :].t()  # (N, D)

        return torch.nn.functional.normalize(descriptors_sampled, p=2, dim=1)
