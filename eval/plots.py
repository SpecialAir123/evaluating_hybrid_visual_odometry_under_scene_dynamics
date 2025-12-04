import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_trajectory(poses_est, poses_gt=None, title="Trajectory"):
    """
    Plot 3D trajectory.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract positions
    positions_est = np.array([pose[:3, 3] for pose in poses_est])
    
    # Plot estimated trajectory
    ax.plot(positions_est[:, 0], positions_est[:, 1], positions_est[:, 2],
            'b-', label='Estimated', linewidth=2)
    
    if poses_gt is not None:
        positions_gt = np.array([pose[:3, 3] for pose in poses_gt])
        ax.plot(positions_gt[:, 0], positions_gt[:, 1], positions_gt[:, 2],
                'r-', label='Ground Truth', linewidth=2)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    # Set equal aspect ratio
    max_range = np.array([positions_est[:, 0].max() - positions_est[:, 0].min(),
                          positions_est[:, 1].max() - positions_est[:, 1].min(),
                          positions_est[:, 2].max() - positions_est[:, 2].min()]).max() / 2.0
    mid_x = (positions_est[:, 0].max() + positions_est[:, 0].min()) * 0.5
    mid_y = (positions_est[:, 1].max() + positions_est[:, 1].min()) * 0.5
    mid_z = (positions_est[:, 2].max() + positions_est[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    return fig


def plot_errors(ate_errors, rpe_trans_errors=None, rpe_rot_errors=None):
    """
    Plot error distributions.
    """
    fig, axes = plt.subplots(1, 3 if rpe_trans_errors is not None else 1, figsize=(15, 4))
    if rpe_trans_errors is None:
        axes = [axes]
    
    # ATE errors
    axes[0].plot(ate_errors, 'b-', linewidth=1)
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('ATE (m)')
    axes[0].set_title(f'Absolute Trajectory Error\nRMSE: {np.sqrt(np.mean(ate_errors**2)):.4f} m')
    axes[0].grid(True)
    
    if rpe_trans_errors is not None:
        # RPE translation errors
        axes[1].plot(rpe_trans_errors, 'g-', linewidth=1)
        axes[1].set_xlabel('Frame')
        axes[1].set_ylabel('RPE Translation (m)')
        axes[1].set_title(f'Relative Pose Error (Translation)\nRMSE: {np.sqrt(np.mean(rpe_trans_errors**2)):.4f} m')
        axes[1].grid(True)
        
        # RPE rotation errors
        axes[2].plot(rpe_rot_errors, 'r-', linewidth=1)
        axes[2].set_xlabel('Frame')
        axes[2].set_ylabel('RPE Rotation (deg)')
        axes[2].set_title(f'Relative Pose Error (Rotation)\nRMSE: {np.sqrt(np.mean(rpe_rot_errors**2)):.2f} deg')
        axes[2].grid(True)
    
    plt.tight_layout()
    return fig

