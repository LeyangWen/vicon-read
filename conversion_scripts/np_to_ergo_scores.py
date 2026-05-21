"""
Compute ergonomic risk scores (posture, duration, frequency) from 3D pose .npy files.

Usage:
    python conversion_scripts/np_to_ergo_scores.py --config_file <yaml> --angle_limit_file <json>

Reads the estimated 3D pose, calculates joint angles via VEHSErgoSkeleton_angles,
classifies each frame into posture risk levels using angle thresholds from the JSON config,
then computes per-video per-joint scores.  Results are saved to CSV and visualized.


Column	Range	Definition
posture_score	0–3	Worst posture level observed in the video for that joint. 0=safe, 1=cautious (yellow), 2=dangerous (red), 3=impossible (black).
duration_score	0–3	How much % of time was spent at level 2 or 3. Scored as: <10%→0, 10–19%→1, 20–29%→2, ≥30%→3.
frequency_score	0 or 1	Binary flag: 1 if risk events happen more than 3×/min (or 30×/min for wrists), else 0. A "risk event" = transitioning into risk posture and staying there for ≥5 consecutive frames (≥3 for wrists).
pct_high_risk	0–100%	The raw percentage of frames at posture level 2 or 3. This is the underlying value that duration_score discretizes.
freq_per_min	≥0	The raw risk events per minute. This is the underlying value that frequency_score discretizes.
n_frames	int	Total number of frames in this video segment.
duration_sec	float	Video duration in seconds (n_frames / fps).
"""

import argparse
import json
import os
import pickle
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
# 1. Load angle-limit config (3D only)
# ---------------------------------------------------------------------------
def _pick_3d_ranges(ranges_dict):
    """Pick '3d' thresholds, fall back to '2d' if '3d' not present."""
    if '3d' in ranges_dict:
        return ranges_dict['3d']
    if '2d' in ranges_dict:
        return ranges_dict['2d']
    # return first available
    for k, v in ranges_dict.items():
        return v
    return {}


def load_angle_limits(json_path):
    """
    Parse joint_angle_limit_rick.json into a per-joint threshold dict (3D only).

    Returns
    -------
    thresholds : dict
        {json_joint_name: {level: {component: [ranges]}}}
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

        levels_3d = _pick_3d_ranges(segment_cfg['ranges'])
        thresholds[joint_name] = {}
        for level_name, components in levels_3d.items():
            level = level_map[level_name]
            thresholds[joint_name][level] = components

    return thresholds, conditional_joints


# ---------------------------------------------------------------------------
# 1b. Parse conditional joint config (e.g. elbows depend on shoulder angle)
# ---------------------------------------------------------------------------
def parse_conditional_thresholds(conditional_joints):
    """
    Parse conditional joint configs into a structured dict (3D only).

    Returns
    -------
    dict : {json_joint_name: {
        'condition_joint': str,       # e.g. 'rightShoulder'
        'condition_range': [lo, hi],  # shoulder flexion range that activates thresholds
        'then': {level: {component: [ranges]}},
        'else_default': [lo, hi] or None
    }}
    """
    level_map = {'cautious': 1, 'dangerous': 2, 'impossible': 3}
    parsed = {}

    for json_joint, segment_cfg in conditional_joints.items():
        if_cfg = segment_cfg['if']
        else_cfg = segment_cfg.get('else', {})

        levels_3d = _pick_3d_ranges(if_cfg['then'])
        then_thresholds = {}
        for level_name, components in levels_3d.items():
            level = level_map[level_name]
            then_thresholds[level] = components

        parsed[json_joint] = {
            'condition_joint': if_cfg['joint'],
            'condition_range': if_cfg['range'],
            'then': then_thresholds,
            'else_default': else_cfg.get('default', None),
        }

    return parsed


# ---------------------------------------------------------------------------
# 2. Classify frames into posture levels
# ---------------------------------------------------------------------------
NAN_LEVEL = -1  # sentinel for masked/NaN frames — excluded from scoring

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


def _to_deg_or_none(arr, t):
    """Convert radians to degrees at index t, returning None if the array is None or value is NaN."""
    if arr is None:
        return None
    val = float(np.degrees(arr[t]))
    if np.isnan(val):
        return None
    return val


def classify_frame(angles_deg, joint_thresholds):
    """
    Classify a single frame for one joint into posture level 0-3, or NAN_LEVEL if all angles are NaN.

    Parameters
    ----------
    angles_deg : dict  {'flexion': float or None, 'abduction': float or None, 'rotation': float or None}
    joint_thresholds : dict  {level: {component: [ranges]}}

    Returns
    -------
    int : posture level (0=safe, 1=cautious, 2=dangerous, 3=impossible, -1=NaN/masked)
    """
    # If all angle components are None (NaN), mark as masked
    if all(angles_deg.get(c) is None for c in ANGLE_COMPONENTS):
        return NAN_LEVEL

    worst_level = 0
    for level in sorted(joint_thresholds.keys()):
        level_ranges = joint_thresholds[level]
        for component in ANGLE_COMPONENTS:
            val = angles_deg.get(component)
            if val is None:
                continue
            comp_range = level_ranges.get(component, [])
            if _in_range(val, comp_range):
                worst_level = max(worst_level, level)

    return worst_level


def classify_all_frames(angle_obj, joint_thresholds):
    """
    Classify all frames for a joint (3D thresholds already selected).

    Parameters
    ----------
    angle_obj : JointAngles with .flexion, .abduction, .rotation (radians, shape (T,))
    joint_thresholds : dict  {level: {component: [ranges]}}

    Returns
    -------
    posture_levels : np.ndarray of int, shape (T,)
        Values: 0-3 for valid frames, NAN_LEVEL (-1) for masked frames.
    """
    flex = angle_obj.flexion
    abd = angle_obj.abduction
    rot = angle_obj.rotation

    T = len(flex) if flex is not None else 0
    if T == 0:
        return np.array([], dtype=int)

    levels = np.zeros(T, dtype=int)
    for t in range(T):
        angles_deg = {
            'flexion': _to_deg_or_none(flex, t),
            'abduction': _to_deg_or_none(abd, t),
            'rotation': _to_deg_or_none(rot, t),
        }
        levels[t] = classify_frame(angles_deg, joint_thresholds)

    return levels


def classify_all_frames_conditional(elbow_angle_obj, shoulder_angle_obj, cond_cfg):
    """
    Classify all frames for a conditional joint (e.g. elbow depends on shoulder flexion).

    Parameters
    ----------
    elbow_angle_obj : JointAngles for the elbow
    shoulder_angle_obj : JointAngles for the conditioning shoulder
    cond_cfg : dict from parse_conditional_thresholds for this joint

    Returns
    -------
    posture_levels : np.ndarray of int, shape (T,)
    """
    flex = elbow_angle_obj.flexion
    T = len(flex) if flex is not None else 0
    if T == 0:
        return np.array([], dtype=int)

    shoulder_flex = shoulder_angle_obj.flexion
    cond_lo, cond_hi = cond_cfg['condition_range']

    levels = np.zeros(T, dtype=int)
    for t in range(T):
        shoulder_deg = _to_deg_or_none(shoulder_flex, t)
        elbow_deg = _to_deg_or_none(flex, t)

        # if either joint is NaN, mark as masked
        if shoulder_deg is None or elbow_deg is None:
            levels[t] = NAN_LEVEL
            continue

        if cond_lo <= shoulder_deg <= cond_hi:
            angles_deg = {
                'flexion': elbow_deg,
                'abduction': _to_deg_or_none(elbow_angle_obj.abduction, t),
                'rotation': _to_deg_or_none(elbow_angle_obj.rotation, t),
            }
            levels[t] = classify_frame(angles_deg, cond_cfg['then'])
        else:
            # "else" branch: default range means safe (level 0)
            levels[t] = 0

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
    Frames with NAN_LEVEL (-1) are excluded from all calculations.

    Returns
    -------
    dict with keys: posture_score, duration_score, frequency_score,
                    pct_high_risk, freq_per_min, n_valid, n_masked, posture_levels
    """
    T = len(posture_levels)
    valid_mask = posture_levels != NAN_LEVEL
    valid_levels = posture_levels[valid_mask]
    n_valid = len(valid_levels)
    n_masked = T - n_valid

    if n_valid == 0:
        return {'posture_score': 0, 'duration_score': 0, 'frequency_score': 0,
                'pct_high_risk': 0.0, 'freq_per_min': 0.0,
                'n_valid': 0, 'n_masked': n_masked, 'posture_levels': posture_levels}

    # Posture score: worst level observed among valid frames
    present_levels = [l for l in [1, 2, 3] if np.sum(valid_levels == l) > 0]
    posture_score = max(present_levels) if present_levels else 0

    # Duration: % of valid frames at level 2 or 3
    high_risk_frames = int(np.sum((valid_levels == 2) | (valid_levels == 3)))
    pct_high_risk = high_risk_frames * 100.0 / n_valid

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

    # Frequency: occurrences per minute (NaN frames are skipped — treated as gaps)
    window = 3 if is_wrist else 5
    occ = count_occurrences(valid_levels, window=window)
    duration_sec = n_valid / fps
    freq_per_min = (occ * 60.0 / duration_sec) if duration_sec > 0 else 0.0
    freq_threshold = 30 if is_wrist else 3
    frequency_score = 1 if freq_per_min > freq_threshold else 0

    return {
        'posture_score': posture_score,
        'duration_score': duration_score,
        'frequency_score': frequency_score,
        'pct_high_risk': pct_high_risk,
        'freq_per_min': freq_per_min,
        'n_valid': n_valid,
        'n_masked': n_masked,
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
def plot_video_ergo(video_results, video_label, fps, save_path=None, mb_stride=243):
    """
    Plot per-frame posture levels for all joints in one video.
    """
    joints = list(video_results.keys())
    n_joints = len(joints)
    if n_joints == 0:
        return

    row_height = 0.8  # height per joint row in inches
    fig, axes = plt.subplots(n_joints, 1, figsize=(14, row_height * n_joints + 1.5), sharex=True)
    if n_joints == 1:
        axes = [axes]

    color_map = {0: 'green', 1: 'gold', 2: 'red', 3: 'black', NAN_LEVEL: 'lightgray'}
    label_map = {0: 'Safe', 1: 'Cautious', 2: 'Dangerous', 3: 'Impossible', NAN_LEVEL: 'Masked/NaN'}

    # Build color array per joint so we can use pcolorfast (no gaps)
    from matplotlib.colors import ListedColormap, BoundaryNorm
    # levels: -1=masked, 0=safe, 1=cautious, 2=dangerous, 3=impossible
    # map to indices 0-4 for colormap: masked→0, safe→1, cautious→2, dangerous→3, impossible→4
    cmap = ListedColormap(['lightgray', 'green', 'gold', 'red', 'black'])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    for ax, joint_name in zip(axes, joints):
        result = video_results[joint_name]
        levels = result['posture_levels']
        T = len(levels)
        max_time = T / fps

        # Draw as a 1-pixel-high image — no gaps
        level_row = levels.reshape(1, -1).astype(float)
        ax.imshow(level_row, aspect='auto', cmap=cmap, norm=norm,
                  extent=[0, max_time, 0, 1], interpolation='nearest')

        # vertical lines at every 243-frame boundary (MB clip boundaries)
        clip_interval_sec = mb_stride / fps
        for t_line in np.arange(clip_interval_sec, max_time, clip_interval_sec):
            ax.axvline(t_line, color='blue', linestyle='dotted', linewidth=0.6, alpha=0.4)

        ax.set_xlim(0, max_time)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(joint_name, fontsize=9, rotation=0, labelpad=80, ha='right')

        ps = result['posture_score']
        ds = result['duration_score']
        is_wrist = 'Wrist' in joint_name or 'wrist' in joint_name
        freq_thresh = 30 if is_wrist else 3
        score_text = (f"Posture score: {ps}  |  "
                      f"Duration score: {ds} ({result['pct_high_risk']:.1f}% high risk)  |  "
                      f"Frequency score: {result['frequency_score']} "
                      f"({result['freq_per_min']:.1f}/min, threshold {freq_thresh})")
        if result.get('n_masked', 0) > 0:
            score_text += f"  |  Masked: {result['n_masked']} frames"
        ax.set_title(score_text, fontsize=8, loc='left')

    axes[-1].set_xlabel('Time (s)')

    # shared legend at bottom
    import matplotlib.patches as mpatches
    legend_handles = [mpatches.Patch(color=c, alpha=0.5, label=label_map[lv])
                      for lv, c in color_map.items()]
    fig.legend(handles=legend_handles, loc='lower center', ncol=len(legend_handles),
               fontsize=9, frameon=False)

    fig.suptitle(f'Ergonomic Risk — {video_label}', fontsize=13, weight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

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
                        default=r'config/experiment_config/37kpts/Inference-RTMPose-MB-20fps-industry_3.yaml')
    parser.add_argument('--skeleton_file', type=str,
                        default=r'config/VEHS_ErgoSkeleton_info/Ergo-Skeleton-37.yaml')
    parser.add_argument('--angle_limit_file', type=str,
                        default=r'config/experiment_config/37kpts/joint_angle_limit_rick.json')
    parser.add_argument('--angle_mode', type=str, default='VEHS')
    parser.add_argument('--try_wrist', default=False, type=bool)
    parser.add_argument('--clip_fill', default=True, type=bool)
    parser.add_argument('--MB_data_stride', type=int, default=243)
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--video_chunks', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--debug_mode', default=False, type=bool)
    parser.add_argument('--filter_bad_support_kpts', default=True, type=bool,
                        help='Mask out frames with bad support keypoint estimates (from MB_np_detect_bad_est.py)')
    parser.add_argument('--filter_lowConf_legs', default=True, type=bool,
                        help='Mask out frames with low-confidence leg keypoints')
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

    # --- Filter bad keypoints (mask to NaN) ---
    if args.filter_bad_support_kpts or args.filter_lowConf_legs:
        bad_kpts_pkl_file = args.estimate_file.replace('.npy', '_support_kpt_score.pkl')
        print(f"Loading bad-kpt mask from: {bad_kpts_pkl_file}")
        with open(bad_kpts_pkl_file, "rb") as f:
            bad_kpts_data = pickle.load(f)

        if args.filter_bad_support_kpts:
            # mask shape: (frame_num, 6) for ['LSHOULDER', 'RSHOULDER', 'LELBOW', 'RELBOW', 'LHAND', 'RHAND']
            mask = bad_kpts_data['mask']
            assert mask.shape[0] == estimate_pose.shape[0], \
                f"mask frames ({mask.shape[0]}) != pose frames ({estimate_pose.shape[0]})"
            skeleton.poses['LAP_b'][mask[:, 0]] = np.nan
            skeleton.poses['LAP_f'][mask[:, 0]] = np.nan
            skeleton.poses['RAP_b'][mask[:, 1]] = np.nan
            skeleton.poses['RAP_f'][mask[:, 1]] = np.nan
            skeleton.poses['LLE'][mask[:, 2]] = np.nan
            skeleton.poses['LME'][mask[:, 2]] = np.nan
            skeleton.poses['RLE'][mask[:, 3]] = np.nan
            skeleton.poses['RME'][mask[:, 3]] = np.nan
            skeleton.poses['LMCP2'][mask[:, 4]] = np.nan
            skeleton.poses['LMCP5'][mask[:, 4]] = np.nan
            skeleton.poses['RMCP2'][mask[:, 5]] = np.nan
            skeleton.poses['RMCP5'][mask[:, 5]] = np.nan
            n_masked = int(mask.any(axis=1).sum())
            print(f"  Masked {n_masked}/{mask.shape[0]} frames for bad support kpts")

        if args.filter_lowConf_legs:
            kpt_names = bad_kpts_data['kpt_names']
            conf_2d = bad_kpts_data['confidence_2d']
            threshold = 5.8
            left_knee_mask = conf_2d[:, kpt_names.index('LKNEE')] < threshold
            right_knee_mask = conf_2d[:, kpt_names.index('RKNEE')] < threshold
            left_ankle_mask = conf_2d[:, kpt_names.index('LANKLE')] < threshold
            right_ankle_mask = conf_2d[:, kpt_names.index('RANKLE')] < threshold
            left_foot_mask = conf_2d[:, kpt_names.index('LFOOT')] < threshold
            right_foot_mask = conf_2d[:, kpt_names.index('RFOOT')] < threshold
            both_knee_mask = (left_knee_mask & right_knee_mask).reshape(-1)
            for kpt_name in ['LKNEE', 'RKNEE', 'LANKLE', 'RANKLE', 'LFOOT', 'RFOOT', 'LHIP', 'RHIP']:
                skeleton.poses[kpt_name][both_knee_mask] = np.nan
            both_foot_ankle_mask = (left_foot_mask & right_foot_mask & left_ankle_mask & right_ankle_mask).reshape(-1)
            for kpt_name in ['LANKLE', 'RANKLE', 'LFOOT', 'RFOOT']:
                skeleton.poses[kpt_name][both_foot_ankle_mask] = np.nan
            print(f"  Masked {int(both_knee_mask.sum())} frames for low-conf knees, "
                  f"{int(both_foot_ankle_mask.sum())} frames for low-conf ankles/feet")

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
    active_joints = []          # (json_joint, skeleton_joint) — simple threshold joints
    active_conditional = []     # (json_joint, skeleton_joint) — conditional joints (elbows)
    parsed_conditional = parse_conditional_thresholds(conditional_joints)

    for json_joint, skeleton_joint in JOINT_MAP.items():
        if json_joint in thresholds and skeleton_joint in ergo_angles:
            active_joints.append((json_joint, skeleton_joint))
        elif json_joint in parsed_conditional and skeleton_joint in ergo_angles:
            # check that the conditioning joint is also available
            cond_json = parsed_conditional[json_joint]['condition_joint']
            cond_skel = JOINT_MAP.get(cond_json)
            if cond_skel and cond_skel in ergo_angles:
                active_conditional.append((json_joint, skeleton_joint))
            else:
                print(f"  Warning: {json_joint} needs {cond_json} but it's not available, skipping")

    print(f"  Active joints: {[j[0] for j in active_joints]}")
    print(f"  Active conditional joints: {[j[0] for j in active_conditional]}")

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
            posture_levels = classify_all_frames(vid_angle, thresholds[json_joint])

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
                'n_valid': scores['n_valid'],
                'n_masked': scores['n_masked'],
                'duration_sec': (end - start) / args.fps,
            }
            all_rows.append(row)
            print(f"  {json_joint:20s}  P={scores['posture_score']}  D={scores['duration_score']}  "
                  f"F={scores['frequency_score']}  ({scores['pct_high_risk']:.1f}% high risk, "
                  f"{scores['freq_per_min']:.1f}/min, {scores['n_masked']} masked)")

        # conditional joints (e.g. elbows conditioned on shoulder flexion)
        for json_joint, skeleton_joint in active_conditional:
            cond_cfg = parsed_conditional[json_joint]
            cond_skel = JOINT_MAP[cond_cfg['condition_joint']]

            elbow_obj = ergo_angles[skeleton_joint]
            shoulder_obj = ergo_angles[cond_skel]

            # slice this video's frames
            vid_elbow = JointAngles()
            vid_elbow.flexion = elbow_obj.flexion[start:end] if elbow_obj.flexion is not None else None
            vid_elbow.abduction = elbow_obj.abduction[start:end] if elbow_obj.abduction is not None else None
            vid_elbow.rotation = elbow_obj.rotation[start:end] if elbow_obj.rotation is not None else None

            vid_shoulder = JointAngles()
            vid_shoulder.flexion = shoulder_obj.flexion[start:end] if shoulder_obj.flexion is not None else None

            posture_levels = classify_all_frames_conditional(
                vid_elbow, vid_shoulder, cond_cfg)

            scores = compute_ergo_scores(posture_levels, fps=args.fps, is_wrist=False)
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
                'n_valid': scores['n_valid'],
                'n_masked': scores['n_masked'],
                'duration_sec': (end - start) / args.fps,
            }
            all_rows.append(row)
            print(f"  {json_joint:20s}  P={scores['posture_score']}  D={scores['duration_score']}  "
                  f"F={scores['frequency_score']}  ({scores['pct_high_risk']:.1f}% high risk, "
                  f"{scores['freq_per_min']:.1f}/min, {scores['n_masked']} masked)  "
                  f"[conditional on {cond_cfg['condition_joint']}]")

        # visualize
        plot_path = os.path.join(args.output_dir, f'ergo_video_{vid_idx}.png')
        plot_video_ergo(video_results, label, args.fps, save_path=plot_path)

    # --- Save CSV ---
    csv_path = os.path.join(args.output_dir, 'ergo_scores.csv')
    header = ['video_idx', 'video_label', 'joint', 'posture_score', 'duration_score',
              'frequency_score', 'pct_high_risk', 'freq_per_min', 'n_frames', 'n_valid', 'n_masked', 'duration_sec']
    with open(csv_path, 'w') as f:
        f.write(','.join(header) + '\n')
        for row in all_rows:
            f.write(','.join(str(row[h]) for h in header) + '\n')

    print(f"\nResults saved to: {csv_path}")
    print(f"Plots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
