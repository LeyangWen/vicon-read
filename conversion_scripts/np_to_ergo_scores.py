"""
Compute ergonomic risk scores (posture, duration, frequency) from 3D pose .npy files.

Usage:
    python conversion_scripts/np_to_ergo_scores.py --config_file <yaml> --angle_limit_file <json>

Reads the estimated 3D pose, calculates joint angles via VEHSErgoSkeleton_angles,
classifies each frame into posture risk levels using angle thresholds from the JSON config,
then computes per-video per-joint scores.  Results are saved to CSV and visualized.
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Skeleton import VEHSErgoSkeleton_angles, JointAngles
from conversion_scripts.MB_np_to_visual import MB_output_pose_file_loader, MB_input_pose_file_loader
import yaml


# ---------------------------------------------------------------------------
# Mapping between angle-limit JSON joints and skeleton angle method names
# ---------------------------------------------------------------------------
# JSON joint name  -> skeleton angle_name
JOINT_MAP = {
    'neck':           'neck',
    'back':           'back',
    'rightShoulder':  'right_shoulder',
    'leftShoulder':   'left_shoulder',
    'rightElbow':     'right_elbow',
    'leftElbow':      'left_elbow',
    'rightWrist':     'right_wrist',
    'leftWrist':      'left_wrist',
    'rightKnee':      'right_knee',
    'leftKnee':       'left_knee',
}

# Reverse map: skeleton angle_name -> JSON joint name
SKELETON_TO_JSON = {v: k for k, v in JOINT_MAP.items()}

# Angle components in order
ANGLE_COMPONENTS = ['flexion', 'abduction', 'rotation']


# ---------------------------------------------------------------------------
# 1. Load angle-limit config
# ---------------------------------------------------------------------------
def load_angle_limits(json_path):
    """
    Parse joint_angle_limit_rick.json into a per-joint threshold dict.

    Returns
    -------
    thresholds : dict
        {json_joint_name: {dim_mode: {level: {component: [ranges]}}}}
        dim_mode is '2d', '3d', or 'seated'.
        level is 1 (cautious), 2 (dangerous), 3 (impossible).
    conditional_joints : dict
        joints with 'if' logic (e.g. elbow depends on shoulder angle)
    """
    with open(json_path) as f:
        cfg = json.load(f)

    thresholds = {}
    conditional_joints = {}

    level_map = {'cautious': 1, 'dangerous': 2, 'impossible': 3}

    for segment_name, segment_cfg in cfg['segments'].items():
        joint_name = segment_cfg['joint']
        if joint_name == 'none':
            continue
        # skip duplicate segments mapping to same joint (e.g. rightThigh & rightCalf both map to rightKnee)
        if joint_name in thresholds:
            continue

        if 'if' in segment_cfg:
            conditional_joints[joint_name] = segment_cfg
            continue

        if 'ranges' not in segment_cfg:
            continue

        thresholds[joint_name] = {}
        for dim_mode, levels in segment_cfg['ranges'].items():
            thresholds[joint_name][dim_mode] = {}
            for level_name, components in levels.items():
                level = level_map[level_name]
                thresholds[joint_name][dim_mode][level] = components

    return thresholds, conditional_joints


# ---------------------------------------------------------------------------
# 2. Classify frames into posture levels
# ---------------------------------------------------------------------------
def _in_range(value_deg, range_list):
    """Check if value falls in any of the ranges defined by range_list.

    range_list formats:
      [lo, hi]           -> single range
      [lo1, hi1, lo2, hi2] -> two ranges (positive and negative)
      [] or None         -> no threshold, never triggers
    """
    if range_list is None or len(range_list) == 0:
        return False
    if len(range_list) == 2:
        return range_list[0] <= value_deg <= range_list[1]
    if len(range_list) == 4:
        return (range_list[0] <= value_deg <= range_list[1]) or \
               (range_list[2] <= value_deg <= range_list[3])
    return False


def classify_frame(angles_deg, joint_thresholds, dim_mode='3d'):
    """
    Classify a single frame for one joint into posture level 0-3.

    Parameters
    ----------
    angles_deg : dict  {'flexion': float, 'abduction': float or None, 'rotation': float or None}
    joint_thresholds : dict  {dim_mode: {level: {component: [ranges]}}}
    dim_mode : str

    Returns
    -------
    int : posture level (0=safe, 1=cautious, 2=dangerous, 3=impossible)
    """
    if dim_mode not in joint_thresholds:
        available = list(joint_thresholds.keys())
        if len(available) == 0:
            return 0
        dim_mode = available[0]

    mode_thresholds = joint_thresholds[dim_mode]
    worst_level = 0

    for level in sorted(mode_thresholds.keys()):
        level_ranges = mode_thresholds[level]
        for component in ANGLE_COMPONENTS:
            val = angles_deg.get(component)
            if val is None:
                continue
            comp_range = level_ranges.get(component, [])
            if _in_range(val, comp_range):
                worst_level = max(worst_level, level)

    return worst_level


def classify_all_frames(angle_obj, joint_thresholds, dim_mode='3d'):
    """
    Classify all frames for a joint.

    Parameters
    ----------
    angle_obj : JointAngles with .flexion, .abduction, .rotation (radians, shape (T,))
    joint_thresholds : dict from load_angle_limits
    dim_mode : str

    Returns
    -------
    posture_levels : np.ndarray of int, shape (T,)
    """
    flex = angle_obj.flexion
    abd = angle_obj.abduction
    rot = angle_obj.rotation

    T = len(flex) if flex is not None else 0
    if T == 0:
        return np.array([], dtype=int)

    levels = np.zeros(T, dtype=int)
    for t in range(T):
        angles_deg = {}
        angles_deg['flexion'] = float(np.degrees(flex[t])) if flex is not None else None
        angles_deg['abduction'] = float(np.degrees(abd[t])) if abd is not None else None
        angles_deg['rotation'] = float(np.degrees(rot[t])) if rot is not None else None
        levels[t] = classify_frame(angles_deg, joint_thresholds, dim_mode)

    return levels


# ---------------------------------------------------------------------------
# 3. Compute ergo scores from posture levels
# ---------------------------------------------------------------------------
def count_occurrences(levels, window=5):
    """Count transitions into risk posture sustained for at least `window` consecutive frames."""
    count = 0
    i = 0
    in_risk = False
    while i < len(levels):
        if levels[i] > 0:
            if not in_risk:
                # check if sustained for window frames
                run_len = 0
                for j in range(i, min(i + window, len(levels))):
                    if levels[j] > 0:
                        run_len += 1
                    else:
                        break
                if run_len >= window:
                    count += 1
                    in_risk = True
            i += 1
        else:
            if in_risk:
                # check if the safe run is sustained for window frames
                run_len = 0
                for j in range(i, min(i + window, len(levels))):
                    if levels[j] == 0:
                        run_len += 1
                    else:
                        break
                if run_len >= window:
                    in_risk = False
            i += 1
    return count


def compute_ergo_scores(posture_levels, fps=20, is_wrist=False):
    """
    Compute duration, frequency, and posture scores from per-frame posture levels.

    Returns
    -------
    dict with keys: posture_score, duration_score, frequency_score,
                    pct_high_risk, freq_per_min, posture_levels
    """
    T = len(posture_levels)
    if T == 0:
        return {'posture_score': 0, 'duration_score': 0, 'frequency_score': 0,
                'pct_high_risk': 0.0, 'freq_per_min': 0.0, 'posture_levels': posture_levels}

    # Posture score: worst level observed (excluding impossible=3 mapped to 4 in Veeru code, here we keep 3)
    present_levels = [l for l in [1, 2, 3] if np.sum(posture_levels == l) > 0]
    posture_score = max(present_levels) if present_levels else 0

    # Duration: % of frames at level 2 or 3
    high_risk_frames = np.sum((posture_levels == 2) | (posture_levels == 3))
    pct_high_risk = high_risk_frames * 100.0 / T

    if posture_score > 0:
        if 10 <= pct_high_risk < 20:
            duration_score = 1
        elif 20 <= pct_high_risk < 30:
            duration_score = 2
        elif pct_high_risk >= 30:
            duration_score = 3
        else:
            duration_score = 0
    else:
        duration_score = 0

    # Frequency: occurrences per minute
    window = 3 if is_wrist else 5
    occ = count_occurrences(posture_levels, window=window)
    duration_sec = T / fps
    freq_per_min = (occ * 60.0 / duration_sec) if duration_sec > 0 else 0.0
    freq_threshold = 30 if is_wrist else 3
    frequency_score = 1 if freq_per_min > freq_threshold else 0

    return {
        'posture_score': posture_score,
        'duration_score': duration_score,
        'frequency_score': frequency_score,
        'pct_high_risk': pct_high_risk,
        'freq_per_min': freq_per_min,
        'posture_levels': posture_levels,
    }


# ---------------------------------------------------------------------------
# 4. Split concatenated frames into per-video segments
# ---------------------------------------------------------------------------
def split_by_video(total_frames, video_chunks_243, stride=243):
    """
    Given a list of chunk counts per video (each in units of stride),
    return list of (start, end) frame indices.
    """
    segments = []
    offset = 0
    for n_chunks in video_chunks_243:
        n_frames = n_chunks * stride
        end = min(offset + n_frames, total_frames)
        segments.append((offset, end))
        offset = end
    return segments


def split_by_source(source_array):
    """
    Given a per-frame source label array, return list of (start, end, label) per video.
    """
    segments = []
    if len(source_array) == 0:
        return segments
    start = 0
    current = source_array[0]
    for i in range(1, len(source_array)):
        if source_array[i] != current:
            segments.append((start, i, current))
            start = i
            current = source_array[i]
    segments.append((start, len(source_array), current))
    return segments


# ---------------------------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------------------------
def plot_video_ergo(video_results, video_label, fps, save_path=None):
    """
    Plot per-frame posture levels for all joints in one video.
    """
    joints = list(video_results.keys())
    n_joints = len(joints)
    if n_joints == 0:
        return

    fig, axes = plt.subplots(n_joints, 1, figsize=(14, 2.5 * n_joints), sharex=True)
    if n_joints == 1:
        axes = [axes]

    color_map = {0: 'green', 1: 'gold', 2: 'red', 3: 'black'}

    for ax, joint_name in zip(axes, joints):
        result = video_results[joint_name]
        levels = result['posture_levels']
        T = len(levels)
        time_sec = np.arange(T) / fps

        for level_val, color in color_map.items():
            mask = levels == level_val
            if np.any(mask):
                ax.fill_between(time_sec, 0, 1, where=mask, color=color, alpha=0.5, step='pre',
                                label=f'Level {level_val}')

        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(joint_name, fontsize=10, rotation=0, labelpad=80, ha='right')

        score_text = (f"P={result['posture_score']}  "
                      f"D={result['duration_score']}  "
                      f"F={result['frequency_score']}  "
                      f"({result['pct_high_risk']:.1f}% high risk, "
                      f"{result['freq_per_min']:.1f}/min)")
        ax.set_title(score_text, fontsize=9, loc='left')
        ax.grid(True, alpha=0.3, axis='x')

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'Ergonomic Risk — {video_label}', fontsize=13, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Compute ergonomic scores from 3D pose')
    parser.add_argument('--config_file', type=str,
                        default='config/experiment_config/37kpts/Inference-RTMPose-MB-20fps-industry_3.yaml')
    parser.add_argument('--skeleton_file', type=str,
                        default='config/VEHS_ErgoSkeleton_info/Ergo-Skeleton-37.yaml')
    parser.add_argument('--angle_limit_file', type=str,
                        default='config/experiment_config/37kpts/joint_angle_limit_rick.json')
    parser.add_argument('--angle_mode', type=str, default='paper')
    parser.add_argument('--try_wrist', type=bool, default=False)
    parser.add_argument('--clip_fill', type=bool, default=True)
    parser.add_argument('--MB_data_stride', type=int, default=243)
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--dim_mode', type=str, default='3d',
                        help='Which threshold dimension to use: 2d, 3d, or seated')
    parser.add_argument('--video_chunks', type=str, default=None,
                        help='Comma-separated list of chunk counts per video (in units of MB_data_stride). '
                             'If not provided, treats entire sequence as one video.')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        data = yaml.safe_load(f)
        args.name_list = data['name_list']
        args.GT_file = data.get('GT_file', 'None')
        args.eval_key = data['eval_key']
        args.estimate_file = data['estimate_file']
        if args.output_dir is None:
            if isinstance(args.estimate_file, str):
                args.output_dir = os.path.join(os.path.dirname(args.estimate_file), 'ergo_scores')
            else:
                args.output_dir = 'ergo_scores'

    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load 3D pose ---
    print(f"Loading estimated pose from: {args.estimate_file}")
    estimate_pose = MB_output_pose_file_loader(args)
    T_total = estimate_pose.shape[0]
    print(f"  Total frames: {T_total}, keypoints: {estimate_pose.shape[1]}")

    # --- Calculate angles ---
    print("Calculating joint angles...")
    skeleton = VEHSErgoSkeleton_angles(args.skeleton_file, mode=args.angle_mode, try_wrist=args.try_wrist)
    skeleton.load_name_list_and_np_points(args.name_list, estimate_pose)

    ergo_angles = {}
    for angle_name in skeleton.angle_names:
        method = f'{angle_name}_angles'
        try:
            ergo_angles[angle_name] = getattr(skeleton, method)()
        except Exception as e:
            print(f"  Warning: could not compute {angle_name}: {e}")
            ergo_angles[angle_name] = skeleton.empty_angles()

    # --- Load angle limits ---
    print(f"Loading angle limits from: {args.angle_limit_file}")
    thresholds, conditional_joints = load_angle_limits(args.angle_limit_file)

    # --- Determine which joints overlap ---
    active_joints = []
    for json_joint, skeleton_joint in JOINT_MAP.items():
        if json_joint in thresholds and skeleton_joint in ergo_angles:
            active_joints.append((json_joint, skeleton_joint))
    # conditional joints (elbows) — skip for now, can be added later
    for json_joint in conditional_joints:
        skeleton_joint = JOINT_MAP.get(json_joint)
        if skeleton_joint and skeleton_joint in ergo_angles:
            print(f"  Note: {json_joint} has conditional thresholds (skipped for now)")

    print(f"  Active joints: {[j[0] for j in active_joints]}")

    # --- Split into videos ---
    if args.video_chunks is not None:
        chunks = [int(x) for x in args.video_chunks.split(',')]
        video_segments = split_by_video(T_total, chunks, stride=args.MB_data_stride)
        video_labels = [f"video_{i}" for i in range(len(video_segments))]
    else:
        # try to get source from GT file
        try:
            _, _, _, source = MB_input_pose_file_loader(args, get_clip_id=True)
            src_segments = split_by_source(source)
            video_segments = [(s, e) for s, e, _ in src_segments]
            video_labels = [str(label) for _, _, label in src_segments]
            print(f"  Found {len(video_segments)} videos from GT source labels")
        except Exception:
            print("  No source info available, treating entire sequence as one video")
            video_segments = [(0, T_total)]
            video_labels = ['full_sequence']

    # --- Compute scores per video per joint ---
    all_rows = []
    for vid_idx, ((start, end), label) in enumerate(zip(video_segments, video_labels)):
        print(f"\nVideo {vid_idx}: {label}  frames [{start}, {end})  ({(end-start)/args.fps:.1f}s)")

        video_results = {}
        for json_joint, skeleton_joint in active_joints:
            angle_obj = ergo_angles[skeleton_joint]

            # slice this video's frames
            vid_angle = JointAngles()
            vid_angle.flexion = angle_obj.flexion[start:end] if angle_obj.flexion is not None else None
            vid_angle.abduction = angle_obj.abduction[start:end] if angle_obj.abduction is not None else None
            vid_angle.rotation = angle_obj.rotation[start:end] if angle_obj.rotation is not None else None

            # classify
            posture_levels = classify_all_frames(vid_angle, thresholds[json_joint], dim_mode=args.dim_mode)

            # compute scores
            is_wrist = 'Wrist' in json_joint or 'wrist' in json_joint
            scores = compute_ergo_scores(posture_levels, fps=args.fps, is_wrist=is_wrist)
            video_results[json_joint] = scores

            row = {
                'video_idx': vid_idx,
                'video_label': label,
                'joint': json_joint,
                'posture_score': scores['posture_score'],
                'duration_score': scores['duration_score'],
                'frequency_score': scores['frequency_score'],
                'pct_high_risk': scores['pct_high_risk'],
                'freq_per_min': scores['freq_per_min'],
                'n_frames': end - start,
                'duration_sec': (end - start) / args.fps,
            }
            all_rows.append(row)
            print(f"  {json_joint:20s}  P={scores['posture_score']}  D={scores['duration_score']}  "
                  f"F={scores['frequency_score']}  ({scores['pct_high_risk']:.1f}% high risk, "
                  f"{scores['freq_per_min']:.1f}/min)")

        # visualize
        plot_path = os.path.join(args.output_dir, f'ergo_video_{vid_idx}.png')
        plot_video_ergo(video_results, label, args.fps, save_path=plot_path)

    # --- Save CSV ---
    csv_path = os.path.join(args.output_dir, 'ergo_scores.csv')
    header = ['video_idx', 'video_label', 'joint', 'posture_score', 'duration_score',
              'frequency_score', 'pct_high_risk', 'freq_per_min', 'n_frames', 'duration_sec']
    with open(csv_path, 'w') as f:
        f.write(','.join(header) + '\n')
        for row in all_rows:
            f.write(','.join(str(row[h]) for h in header) + '\n')

    print(f"\nResults saved to: {csv_path}")
    print(f"Plots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
