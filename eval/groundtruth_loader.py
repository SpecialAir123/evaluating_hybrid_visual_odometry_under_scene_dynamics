import os
import numpy as np
from scipy.spatial.transform import Rotation as R


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


def load_tum_groundtruth(groundtruth_path):
    """
    Load TUM format ground truth trajectory.
    Format: timestamp tx ty tz qx qy qz qw
    Returns: list of (timestamp, pose_matrix) tuples
    """
    poses = []
    timestamps = []
    
    if not os.path.exists(groundtruth_path):
        return None, None
    
    with open(groundtruth_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            parts = line.split()
            if len(parts) >= 8:
                timestamp = float(parts[0])
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                
                # Convert quaternion to rotation matrix
                rotation = R.from_quat([qx, qy, qz, qw])
                R_mat = rotation.as_matrix()
                
                # Build 4x4 pose matrix
                pose = np.eye(4)
                pose[:3, :3] = R_mat
                pose[:3, 3] = [tx, ty, tz]
                
                timestamps.append(timestamp)
                poses.append(pose)
    
    return timestamps, poses


def sync_trajectories(timestamps_est, poses_est, timestamps_gt, poses_gt, max_time_diff=0.02):
    """
    Synchronize estimated and ground truth trajectories by matching timestamps.
    Returns synchronized poses_est and poses_gt (same length).
    """
    if len(timestamps_est) != len(poses_est) or len(timestamps_gt) != len(poses_gt):
        raise ValueError("Timestamps and poses must have same length")
    
    poses_est_sync = []
    poses_gt_sync = []
    
    timestamps_gt_array = np.array(timestamps_gt)
    
    for i, ts_est in enumerate(timestamps_est):
        # Find closest ground truth timestamp
        time_diffs = np.abs(timestamps_gt_array - ts_est)
        closest_idx = np.argmin(time_diffs)
        time_diff = time_diffs[closest_idx]
        
        # Only include if within max_time_diff
        if time_diff <= max_time_diff:
            poses_est_sync.append(poses_est[i])
            poses_gt_sync.append(poses_gt[closest_idx])
    
    if len(poses_est_sync) == 0:
        raise ValueError("No matching timestamps found. Check timestamp synchronization.")
    
    return poses_est_sync, poses_gt_sync


def align_trajectories(poses_est, poses_gt):
    """
    Align estimated trajectory to ground truth using Umeyama algorithm.
    Returns aligned poses_est.
    """
    if len(poses_est) != len(poses_gt):
        raise ValueError(f"Trajectories must have same length. Got {len(poses_est)} vs {len(poses_gt)}")
    
    # Extract positions
    positions_est = np.array([pose[:3, 3] for pose in poses_est])
    positions_gt = np.array([pose[:3, 3] for pose in poses_gt])
    
    # Center the trajectories
    mean_est = np.mean(positions_est, axis=0)
    mean_gt = np.mean(positions_gt, axis=0)
    
    positions_est_centered = positions_est - mean_est
    positions_gt_centered = positions_gt - mean_gt
    
    # Compute covariance
    H = positions_est_centered.T @ positions_gt_centered
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    R_align = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R_align) < 0:
        Vt[-1, :] *= -1
        R_align = Vt.T @ U.T
    
    # Compute scale
    scale = np.sum(S) / np.sum(positions_est_centered ** 2)
    
    # Compute translation
    t_align = mean_gt - scale * R_align @ mean_est
    
    # Apply alignment to all poses
    poses_aligned = []
    for pose in poses_est:
        pose_aligned = np.eye(4)
        pose_aligned[:3, :3] = R_align @ pose[:3, :3]
        pose_aligned[:3, 3] = scale * R_align @ pose[:3, 3] + t_align
        poses_aligned.append(pose_aligned)
    
    return poses_aligned, scale

