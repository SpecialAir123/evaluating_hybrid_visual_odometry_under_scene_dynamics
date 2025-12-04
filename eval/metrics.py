import numpy as np


def compute_ate(poses_est, poses_gt):
    """
    Compute Absolute Trajectory Error (ATE).
    Returns: RMSE, mean error, max error, errors array
    """
    if len(poses_est) != len(poses_gt):
        min_len = min(len(poses_est), len(poses_gt))
        poses_est = poses_est[:min_len]
        poses_gt = poses_gt[:min_len]
    
    positions_est = np.array([pose[:3, 3] for pose in poses_est])
    positions_gt = np.array([pose[:3, 3] for pose in poses_gt])
    
    errors = np.linalg.norm(positions_est - positions_gt, axis=1)
    
    rmse = np.sqrt(np.mean(errors ** 2))
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    return rmse, mean_error, max_error, errors


def compute_rpe(poses_est, poses_gt, delta=1):
    """
    Compute Relative Pose Error (RPE) for delta frames.
    Returns: translation RMSE, rotation RMSE (in degrees), errors arrays
    """
    if len(poses_est) != len(poses_gt) or len(poses_est) < delta + 1:
        return None, None, None, None
    
    trans_errors = []
    rot_errors = []
    
    for i in range(len(poses_est) - delta):
        # Relative motion in estimated trajectory
        T_est_i = poses_est[i]
        T_est_j = poses_est[i + delta]
        T_est_rel = np.linalg.inv(T_est_i) @ T_est_j
        
        # Relative motion in ground truth
        T_gt_i = poses_gt[i]
        T_gt_j = poses_gt[i + delta]
        T_gt_rel = np.linalg.inv(T_gt_i) @ T_gt_j
        
        # Error in relative motion
        T_error = np.linalg.inv(T_gt_rel) @ T_est_rel
        
        # Translation error
        trans_error = np.linalg.norm(T_error[:3, 3])
        trans_errors.append(trans_error)
        
        # Rotation error (angle in degrees)
        R_error = T_error[:3, :3]
        trace = np.trace(R_error)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        rot_error_deg = np.degrees(angle)
        rot_errors.append(rot_error_deg)
    
    trans_errors = np.array(trans_errors)
    rot_errors = np.array(rot_errors)
    
    trans_rmse = np.sqrt(np.mean(trans_errors ** 2))
    rot_rmse = np.sqrt(np.mean(rot_errors ** 2))
    
    return trans_rmse, rot_rmse, trans_errors, rot_errors

