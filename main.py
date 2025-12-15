import yaml
import cv2
import numpy as np
import argparse
import os
import time
from scipy.spatial.transform import Rotation as R

from detectors.orb_detector import ORBDetector
from matchers.knn_matcher import KNNMatcher
from geometry.pose_estimation import PoseEstimator
from eval.dataset_loader_tum import TUMDataset
from eval.groundtruth_loader import load_tum_groundtruth, align_trajectories, sync_trajectories
from eval.metrics import compute_ate, compute_rpe
from eval.plots import plot_trajectory, plot_errors


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_tum_intrinsics(sequence_name):
    """
    TUM RGB-D camera intrinsics.
    For freiburg1: fx=517.3, fy=516.5, cx=318.6, cy=255.3
    For freiburg2: fx=520.9, fy=521.0, cx=325.1, cy=249.7
    For freiburg3: fx=535.4, fy=539.2, cx=320.1, cy=247.6
    """
    if "freiburg1" in sequence_name or "fr1" in sequence_name:
        fx, fy, cx, cy = 517.3, 516.5, 318.6, 255.3
    elif "freiburg2" in sequence_name or "fr2" in sequence_name:
        fx, fy, cx, cy = 520.9, 521.0, 325.1, 249.7
    elif "freiburg3" in sequence_name or "fr3" in sequence_name:
        fx, fy, cx, cy = 535.4, 539.2, 320.1, 247.6
    else:
        # Default to freiburg1
        fx, fy, cx, cy = 517.3, 516.5, 318.6, 255.3
    
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float32)
    return K


def save_trajectory_tum(trajectory, timestamps, output_path):
    """Save trajectory in TUM format: timestamp tx ty tz qx qy qz qw"""
    with open(output_path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for i, pose in enumerate(trajectory):
            timestamp = timestamps[i] if i < len(timestamps) else float(i)
            tx, ty, tz = pose[:3, 3]
            R_mat = pose[:3, :3]
            rotation = R.from_matrix(R_mat)
            qx, qy, qz, qw = rotation.as_quat()  # Returns [x, y, z, w]
            f.write(f"{timestamp:.6f} {tx:.6f} {ty:.6f} {tz:.6f} "
                   f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")


def save_metrics(ate_rmse, ate_mean, ate_max, rpe_trans_rmse, rpe_rot_rmse, 
                 num_frames, num_matches_avg, inlier_ratio_avg, output_path, 
                 config_name, dataset, sequence, runtime_stats=None):
    """Save evaluation metrics to JSON file."""
    import json
    from datetime import datetime
    
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "pipeline": config_name,
        "dataset": dataset,
        "sequence": sequence,
        "num_frames": num_frames,
        "absolute_trajectory_error": {
            "rmse": float(ate_rmse),
            "mean": float(ate_mean),
            "max": float(ate_max),
            "unit": "meters"
        },
        "relative_pose_error": {
            "translation_rmse": float(rpe_trans_rmse) if rpe_trans_rmse is not None else None,
            "rotation_rmse": float(rpe_rot_rmse) if rpe_rot_rmse is not None else None,
            "translation_unit": "meters",
            "rotation_unit": "degrees"
        },
        "runtime_metrics": {
            "avg_matches_per_frame": float(num_matches_avg),
            "avg_inlier_ratio": float(inlier_ratio_avg)
        }
    }
    
    # Add runtime statistics if provided
    if runtime_stats is not None:
        metrics["runtime_metrics"].update({
            "total_time_seconds": runtime_stats.get("total_time_seconds", 0.0),
            "avg_time_per_frame_seconds": runtime_stats.get("avg_time_per_frame_seconds", 0.0),
            "fps": runtime_stats.get("fps", 0.0),
            "detection_time_ms": runtime_stats.get("detection_time_ms", 0.0),
            "matching_time_ms": runtime_stats.get("matching_time_ms", 0.0),
            "pose_estimation_time_ms": runtime_stats.get("pose_estimation_time_ms", 0.0)
        })
    
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Visual Odometry Pipeline")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file (defines method: detector/matcher/masking)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override dataset from config (TUM or KITTI)")
    parser.add_argument("--sequence", type=str, default=None,
                        help="Override sequence from config")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate against ground truth")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize trajectory")
    parser.add_argument("--save", type=str, default=None,
                        help="Save trajectory to file (TUM format)")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    # CLI args override config
    if args.dataset is not None:
        cfg["dataset"] = args.dataset
    if args.sequence is not None:
        cfg["sequence"] = args.sequence
    
    # Validate required fields
    if "dataset" not in cfg or "sequence" not in cfg:
        raise ValueError("Config must specify 'dataset' and 'sequence', or provide via --dataset and --sequence")
    
    # Get camera intrinsics based on sequence
    sequence_name = cfg.get("sequence", "")
    K = get_tum_intrinsics(sequence_name)
    
    # Dataset path
    dataset_type = cfg.get("dataset", "TUM")
    data_root = cfg.get("data_root", f"data/{dataset_type}")
    dataset_path = os.path.join(data_root, cfg['sequence'])
    dataset = TUMDataset(dataset_path)

    # Initialize detector
    detector_type = cfg.get("detector", "orb")
    detector_params = cfg.get("detector_params", {})

    if detector_type == "orb":
        detector = ORBDetector(**detector_params)
    elif detector_type == "superpoint":
        from detectors.superpoint_infer import SuperPointDetector
        detector = SuperPointDetector(**detector_params)
    elif detector_type == "disk":
        from detectors.disk_infer import DISKDetector
        detector = DISKDetector(**detector_params)
    else:
        raise ValueError(f"Unknown detector: {detector_type}")

    # Initialize matcher
    matcher_type = cfg.get("matcher", "knn")
    matcher_params = cfg.get("matcher_params", {})

    if matcher_type == "knn":
        matcher = KNNMatcher(**matcher_params)
    elif matcher_type == "superglue":
        from matchers.superglue_infer import SuperGlueMatcher
        matcher = SuperGlueMatcher(**matcher_params)
    elif matcher_type == "lightglue":
        from matchers.lightglue_infer import LightGlueMatcher
        matcher = LightGlueMatcher(**matcher_params)
    else:
        raise ValueError(f"Unknown matcher: {matcher_type}")

    # Initialize pose estimator
    pose_params = cfg.get("pose_estimation", {})
    pose_estimator = PoseEstimator(K, **pose_params)
    
    prev_img = None
    trajectory = []
    timestamps = []
    current_pose = np.eye(4)  # 4x4 transformation matrix
    match_counts = []
    inlier_ratios = []
    
    # Runtime tracking
    detection_times = []
    matching_times = []
    pose_estimation_times = []
    total_frame_times = []
    
    print(f"\nRunning VO Pipeline on {len(dataset)} frames...")
    print(f"  Method: {detector_type} + {matcher_type}")
    print(f"  Dataset: {dataset_type} / {cfg['sequence']}")
    print("-" * 50)
    
    pipeline_start_time = time.time()
    
    for i in range(len(dataset)):
        frame_start_time = time.time()
        img = dataset[i]
        timestamp = dataset.get_timestamp(i)
        
        if prev_img is None:
            prev_img = img
            trajectory.append(current_pose.copy())
            timestamps.append(timestamp)
            continue
        
        # Detect features
        detect_start = time.time()
        detector_output1 = detector(prev_img)
        detector_output2 = detector(img)
        detect_time = time.time() - detect_start
        detection_times.append(detect_time)
        
        # Handle detectors that return scores (SuperPoint) vs those that don't (ORB)
        if len(detector_output1) == 3:
            # SuperPoint returns (keypoints, descriptors, scores)
            kp1, desc1, scores1 = detector_output1
            kp2, desc2, scores2 = detector_output2
        else:
            # ORB returns (keypoints, descriptors)
            kp1, desc1 = detector_output1
            kp2, desc2 = detector_output2
            scores1, scores2 = None, None
        
        # Match features
        match_start = time.time()
        if matcher_type == "knn":
            matches = matcher(desc1, desc2)
        elif matcher_type in ["superglue", "lightglue"]:
            h, w = img.shape[:2]
            matches = matcher(kp1, desc1, kp2, desc2, image_shape=(h, w),
                            scores1=scores1, scores2=scores2)
        else:
            raise ValueError(f"Unknown matcher: {matcher_type}")
        match_time = time.time() - match_start
        matching_times.append(match_time)
        
        # Estimate pose
        pose_start = time.time()
        R_est, t_est, inliers = pose_estimator.estimate(kp1, kp2, matches)
        pose_time = time.time() - pose_start
        pose_estimation_times.append(pose_time)
        
        frame_time = time.time() - frame_start_time
        total_frame_times.append(frame_time)
        
        if R_est is not None and t_est is not None:
            # Update trajectory
            # Convert relative motion to absolute pose
            T = np.eye(4)
            T[:3, :3] = R_est
            T[:3, 3] = t_est.flatten()
            current_pose = current_pose @ T
            trajectory.append(current_pose.copy())
            timestamps.append(timestamp)
            
            inlier_count = np.sum(inliers) if inliers is not None else len(matches)
            inlier_ratio = inlier_count / len(matches) if len(matches) > 0 else 0
            
            # Track statistics
            match_counts.append(len(matches))
            inlier_ratios.append(inlier_ratio)
            
            # Note: translation is normalized (scale ambiguity in monocular VO)
            print(f"Frame {i:4d}: matches={len(matches):4d}, inliers={inlier_count:4d}, "
                  f"translation_norm={np.linalg.norm(t_est):.4f}")
        else:
            print(f"Frame {i:4d}: matches={len(matches):4d}, FAILED (insufficient matches)")
            # Keep previous pose if estimation fails
            trajectory.append(current_pose.copy())
            timestamps.append(timestamp)
            
            # Track statistics (failed frame)
            match_counts.append(len(matches))
            inlier_ratios.append(0.0)
        
        prev_img = img
    
    pipeline_total_time = time.time() - pipeline_start_time
    
    print("-" * 50)
    print(f"Completed! Processed {len(dataset)} frames.")
    print(f"Trajectory length: {len(trajectory)} poses")
    
    # Print runtime statistics
    print(f"\n⏱️  Runtime Statistics:")
    print(f"   Total pipeline time: {pipeline_total_time:.2f} s")
    print(f"   Average time per frame: {np.mean(total_frame_times):.4f} s ({1.0/np.mean(total_frame_times):.1f} FPS)")
    print(f"   Detection time: {np.mean(detection_times)*1000:.2f} ms/frame (avg)")
    print(f"   Matching time: {np.mean(matching_times)*1000:.2f} ms/frame (avg)")
    print(f"   Pose estimation time: {np.mean(pose_estimation_times)*1000:.2f} ms/frame (avg)")
    
    # Automatically save trajectory (use --save to override path)
    output_cfg = cfg.get("output", {})
    # Auto-generate config and sequence names for output organization
    config_name = os.path.basename(args.config).replace('.yaml', '')
    sequence_name = cfg['sequence'].replace('/', '_')
    
    if args.save:
        trajectory_file = args.save
    else:
        # Always auto-generate trajectory file path based on config and sequence
        trajectory_file = f"output/{config_name}_{sequence_name}/trajectory.txt"
    
    # Ensure output directory exists
    save_dir = os.path.dirname(trajectory_file) if os.path.dirname(trajectory_file) else "output"
    os.makedirs(save_dir, exist_ok=True)
    save_trajectory_tum(trajectory, timestamps, trajectory_file)
    print(f"💾 Saved trajectory to {trajectory_file}")
    
    # Automatically evaluate against ground truth (if available)
    groundtruth_path = os.path.join(dataset_path, "groundtruth.txt")
    gt_timestamps, poses_gt = load_tum_groundtruth(groundtruth_path)
    
    if poses_gt is None:
        print(f"\n⚠️  Ground truth not found at {groundtruth_path}")
        print("   Skipping evaluation.")
    else:
            print(f"\n📊 Evaluating against ground truth...")
            print(f"   Estimated: {len(trajectory)} poses")
            print(f"   Ground truth: {len(poses_gt)} poses")
            
            # Get evaluation parameters from config
            eval_params = cfg.get("evaluation", {})
            max_time_diff = eval_params.get("max_time_diff", 0.02)
            rpe_delta = eval_params.get("rpe_delta", 1)
            
            # Synchronize trajectories by timestamps
            trajectory_sync, poses_gt_sync = sync_trajectories(
                timestamps, trajectory, gt_timestamps, poses_gt, max_time_diff=max_time_diff
            )
            print(f"   After sync: {len(trajectory_sync)} matched poses")
            
            # Align trajectories (scale, rotation, translation)
            trajectory_aligned, scale = align_trajectories(trajectory_sync, poses_gt_sync)
            print(f"   Alignment scale: {scale:.4f}")
            
            # Compute ATE
            ate_rmse, ate_mean, ate_max, ate_errors = compute_ate(trajectory_aligned, poses_gt_sync)
            print(f"\n📈 Absolute Trajectory Error (ATE):")
            print(f"   RMSE: {ate_rmse:.4f} m")
            print(f"   Mean: {ate_mean:.4f} m")
            print(f"   Max:  {ate_max:.4f} m")
            
            # Compute RPE
            rpe_trans_rmse, rpe_rot_rmse, rpe_trans_errors, rpe_rot_errors = compute_rpe(
                trajectory_aligned, poses_gt_sync, delta=rpe_delta
            )
            if rpe_trans_rmse is not None:
                print(f"\n📈 Relative Pose Error (RPE):")
                print(f"   Translation RMSE: {rpe_trans_rmse:.4f} m")
                print(f"   Rotation RMSE:    {rpe_rot_rmse:.2f} deg")
            
            # Save metrics to file (always auto-generate path)
            metrics_file = f"output/{config_name}_{sequence_name}/metrics.json"
            # Ensure output directory exists
            metrics_dir = os.path.dirname(metrics_file) if os.path.dirname(metrics_file) else "output"
            os.makedirs(metrics_dir, exist_ok=True)
            
            num_matches_avg = np.mean(match_counts) if match_counts else 0
            inlier_ratio_avg = np.mean(inlier_ratios) if inlier_ratios else 0
            
            # Runtime statistics
            runtime_stats = {
                "total_time_seconds": pipeline_total_time,
                "avg_time_per_frame_seconds": float(np.mean(total_frame_times)),
                "fps": float(1.0 / np.mean(total_frame_times)) if np.mean(total_frame_times) > 0 else 0.0,
                "detection_time_ms": float(np.mean(detection_times) * 1000),
                "matching_time_ms": float(np.mean(matching_times) * 1000),
                "pose_estimation_time_ms": float(np.mean(pose_estimation_times) * 1000),
                "num_frames": len(total_frame_times)
            }
            
            save_metrics(
                ate_rmse, ate_mean, ate_max, 
                rpe_trans_rmse, rpe_rot_rmse,
                len(trajectory_sync),
                num_matches_avg, inlier_ratio_avg,
                metrics_file, config_name, dataset_type, cfg['sequence'],
                runtime_stats=runtime_stats
            )
            print(f"\n💾 Saved metrics to: {metrics_file}")
            
            # Automatically generate visualizations
            print("\n📊 Generating visualizations...")
            traj_plot = f"output/{config_name}_{sequence_name}/trajectory.png"
            errors_plot = f"output/{config_name}_{sequence_name}/errors.png"
            plot_dpi = output_cfg.get("plot_dpi", 150)
            
            # Ensure output directory exists
            traj_dir = os.path.dirname(traj_plot) if os.path.dirname(traj_plot) else "output"
            errors_dir = os.path.dirname(errors_plot) if os.path.dirname(errors_plot) else "output"
            os.makedirs(traj_dir, exist_ok=True)
            os.makedirs(errors_dir, exist_ok=True)
            
            fig1 = plot_trajectory(trajectory_aligned, poses_gt_sync, 
                                  title=f"Trajectory: {cfg['sequence']}")
            fig1.savefig(traj_plot, dpi=plot_dpi)
            print(f"   Saved: {traj_plot}")
            
            fig2 = plot_errors(ate_errors, rpe_trans_errors, rpe_rot_errors)
            fig2.savefig(errors_plot, dpi=plot_dpi)
            print(f"   Saved: {errors_plot}")
            
            try:
                import matplotlib.pyplot as plt
                plt.show()
            except:
                pass


if __name__ == "__main__":
    main()

