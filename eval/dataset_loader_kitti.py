"""
KITTI Odometry Dataset Loader

KITTI Odometry dataset structure:
    dataset/
    ├── sequences/
    │   ├── 00/
    │   │   ├── image_0/        # left grayscale camera
    │   │   │   ├── 000000.png
    │   │   │   ├── 000001.png
    │   │   │   └── ...
    │   │   ├── image_1/        # right grayscale camera
    │   │   ├── image_2/        # left color camera
    │   │   ├── image_3/        # right color camera
    │   │   ├── calib.txt       # calibration file
    │   │   └── times.txt       # timestamps
    │   ├── 01/
    │   └── ...
    └── poses/
        ├── 00.txt              # ground truth poses (only for 00-10)
        ├── 01.txt
        └── ...
"""

import os
import cv2
import numpy as np


class KITTIDataset:
    """KITTI Odometry Dataset loader."""
    
    def __init__(self, data_root, sequence, camera=0):
        """
        Initialize KITTI dataset loader.
        
        Args:
            data_root: Path to KITTI odometry dataset root (containing sequences/ and poses/)
            sequence: Sequence number (e.g., "00", "01", ..., "21")
            camera: Camera index (0=left gray, 1=right gray, 2=left color, 3=right color)
        """
        self.data_root = data_root
        self.sequence = str(sequence).zfill(2)  # Ensure 2-digit format
        self.camera = camera
        
        # Build paths
        # Try standard structure: data_root/sequences/{sequence}
        standard_path = os.path.join(data_root, "sequences", self.sequence)
        # Try flat structure: data_root/{sequence}
        flat_path = os.path.join(data_root, self.sequence)
        
        if os.path.exists(standard_path):
            self.sequence_path = standard_path
        elif os.path.exists(flat_path):
            self.sequence_path = flat_path
        else:
            raise ValueError(f"Sequence {self.sequence} not found in {data_root}. Expected 'sequences/{self.sequence}' or '{self.sequence}'")

        self.image_dir = os.path.join(self.sequence_path, f"image_{camera}")
        self.calib_path = os.path.join(self.sequence_path, "calib.txt")
        self.times_path = os.path.join(self.sequence_path, "times.txt")
        
        # Try to find poses
        # 1. Standard: data_root/poses/{sequence}.txt
        poses_std = os.path.join(data_root, "poses", f"{self.sequence}.txt")
        # 2. Flat/User: data_root/{sequence}.txt (unlikely but possible)
        poses_flat = os.path.join(data_root, f"{self.sequence}.txt")
        # 3. Inside sequence folder (sometimes people put it there)
        poses_in_seq = os.path.join(self.sequence_path, "groundtruth.txt")
        
        if os.path.exists(poses_std):
            self.poses_path = poses_std
        elif os.path.exists(poses_flat):
            self.poses_path = poses_flat
        elif os.path.exists(poses_in_seq):
            self.poses_path = poses_in_seq
        else:
            self.poses_path = os.path.join(data_root, "poses", f"{self.sequence}.txt") # Default fallback

        # Validate paths
        if not os.path.exists(self.image_dir):
            raise ValueError(f"Image directory not found: {self.image_dir}")
        
        # Load image list
        self.image_paths = self._load_image_list()
        
        # Load timestamps
        self.timestamps = self._load_timestamps()
        
        # Load calibration
        self.K = self._load_calibration()
        
        print(f"Loaded KITTI sequence {self.sequence} with {len(self.image_paths)} images")
        print(f"  Camera: {camera} ({'grayscale' if camera < 2 else 'color'})")
        print(f"  Calibration matrix K:\n{self.K}")
    
    def _load_image_list(self):
        """Load and sort image file paths."""
        image_files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.endswith('.png') or f.endswith('.jpg')
        ])
        
        if len(image_files) == 0:
            raise ValueError(f"No images found in: {self.image_dir}")
        
        return [os.path.join(self.image_dir, f) for f in image_files]
    
    def _load_timestamps(self):
        """Load timestamps from times.txt."""
        timestamps = []
        
        if os.path.exists(self.times_path):
            with open(self.times_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        timestamps.append(float(line))
        else:
            # If no timestamps file, use frame indices
            print(f"  Warning: times.txt not found, using frame indices as timestamps")
            timestamps = list(range(len(self.image_paths)))
        
        return timestamps
    
    def _load_calibration(self):
        """
        Load camera calibration from calib.txt.
        
        KITTI calib.txt format:
            P0: fx 0 cx 0 0 fy cy 0 0 0 1 0   (12 values, 3x4 projection matrix for cam0)
            P1: ...                            (cam1)
            P2: ...                            (cam2)
            P3: ...                            (cam3)
        
        We extract K (3x3) from the corresponding P matrix.
        """
        if not os.path.exists(self.calib_path):
            # Return default KITTI intrinsics if calib.txt not found
            print(f"  Warning: calib.txt not found, using default KITTI intrinsics")
            return self._get_default_intrinsics()
        
        with open(self.calib_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"P{self.camera}:"):
                    values = [float(x) for x in line.split()[1:]]
                    if len(values) >= 12:
                        # P is 3x4 projection matrix, extract K (3x3)
                        P = np.array(values[:12]).reshape(3, 4)
                        K = P[:3, :3].copy()
                        return K.astype(np.float32)
        
        # Fallback to default
        print(f"  Warning: P{self.camera} not found in calib.txt, using default intrinsics")
        return self._get_default_intrinsics()
    
    def _get_default_intrinsics(self):
        """Return default KITTI intrinsics (approximately correct for sequence 00-02)."""
        # These are typical values for KITTI grayscale camera
        fx, fy = 718.856, 718.856
        cx, cy = 607.1928, 185.2157
        
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        return K
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Load and return image at index idx (grayscale)."""
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {self.image_paths[idx]}")
        return img
    
    def get_color_image(self, idx):
        """Load and return color image at index idx."""
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image: {self.image_paths[idx]}")
        return img
    
    def get_timestamp(self, idx):
        """Get timestamp for frame at index idx."""
        if idx < len(self.timestamps):
            return self.timestamps[idx]
        return float(idx)
    
    def get_intrinsics(self):
        """Return camera intrinsic matrix K."""
        return self.K
    
    def get_poses_path(self):
        """Return path to ground truth poses file."""
        return self.poses_path
    
    def has_ground_truth(self):
        """Check if ground truth poses are available (only sequences 00-10)."""
        return os.path.exists(self.poses_path)


def load_kitti_groundtruth(poses_path):
    """
    Load KITTI format ground truth poses.
    
    KITTI poses format: Each line contains 12 values (3x4 transformation matrix in row-major)
        r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
    
    Args:
        poses_path: Path to poses file (e.g., poses/00.txt)
    
    Returns:
        timestamps: List of frame indices as timestamps
        poses: List of 4x4 transformation matrices
    """
    if not os.path.exists(poses_path):
        return None, None
    
    poses = []
    timestamps = []
    
    with open(poses_path, "r") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            values = [float(x) for x in line.split()]
            if len(values) != 12:
                print(f"Warning: Invalid pose at line {idx}, expected 12 values, got {len(values)}")
                continue
            
            # Build 4x4 transformation matrix
            pose = np.eye(4)
            pose[:3, :4] = np.array(values).reshape(3, 4)
            
            timestamps.append(float(idx))
            poses.append(pose)
    
    print(f"Loaded {len(poses)} ground truth poses from {poses_path}")
    return timestamps, poses


def save_trajectory_kitti(trajectory, output_path):
    """
    Save trajectory in KITTI format: 12 values per line (3x4 matrix row-major).
    
    Args:
        trajectory: List of 4x4 transformation matrices
        output_path: Output file path
    """
    with open(output_path, "w") as f:
        for pose in trajectory:
            # Extract 3x4 matrix and flatten to row-major
            values = pose[:3, :4].flatten()
            line = " ".join([f"{v:.6e}" for v in values])
            f.write(line + "\n")
    print(f"Saved {len(trajectory)} poses to {output_path}")

