import argparse
import os.path
import pickle
import numpy as np
from Skeleton import *
import matplotlib
import copy
# matplotlib.use('Qt5Agg')
from MB_np_to_visual import MB_input_pose_file_loader, MB_output_pose_file_loader, flip_data
import matplotlib.ticker as ticker

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', type=str, default=r'config/experiment_config/37kpts/Inference-RTMPose-MB-20fps-Industry.yaml')
    parser.add_argument('--clip_fill', type=bool, default=False)
    parser.add_argument('--rescale_25d', type=bool, default=False)
    parser.add_argument('--debug_mode', default=False, type=bool)
    parser.add_argument('--MB_data_stride', type=int, default=243)
    args = parser.parse_args()
    with open(args.config_file, 'r') as stream:
        data = yaml.safe_load(stream)
        args.name_list = data['name_list']
        args.GT_file = data['GT_file']
        args.eval_key = data['eval_key']
        args.estimate_file = data['estimate_file']
    return args



def center_z_score(pose, args=False):
    """
    only for VEHS-37 kpts, enforce shoulder center to be in center of back & front of shoulder, same for elbow and wrist
    Input: (T, 37, 3)
    """
    if args.joint_format == 'RTM-37':
        # Indices in rtm_pose_37_keypoints_vicon_dataset_v1 (zero-based)
        # New order: LSHOULDER, RSHOULDER, LELBOW, RELBOW, LHAND, RHAND
        center_ids = [16, 15, 14, 13, 12, 11]

        # Supporting pairs in the same L,R order:
        # Shoulders: LAP_b/LAP_f, RAP_b/RAP_f
        # Elbows:    LLE/LME,    RLE/RME
        # Hands:     LMCP2/LMCP5, RMCP2/RMCP5
        support_pairs = [
            (27, 28),  # L shoulder supports
            (25, 26),  # R shoulder supports
            (31, 32),  # L elbow supports
            (29, 30),  # R elbow supports
            (35, 36),  # L hand supports
            (33, 34),  # R hand supports
        ]

        # Select predicted and GT points
        # Gather both support points for all pairs → (T, K, 2, 3)
        pred_support_pts = pose[:, support_pairs, 2]    # (T, K, 2)
        pred_mid_pts = pose[:, center_ids, 2]          # (T, K)

        # Compute midpoint along the "2" axis
        pred_support_mid = pred_support_pts.mean(axis=2) # (T, K)
        score = np.abs(pred_mid_pts - pred_support_mid)  # (T, K)
        return score
    else:
        raise NotImplementedError

def arm_bbox_diag_3d(pose, args=False):
    """
    only for VEHS-37 kpts,
    Input: (T, 37, 3)
    """
    if args.joint_format == 'RTM-37':
        # Indices in rtm_pose_37_keypoints_vicon_dataset_v1 (zero-based)
        # Centers: RSHOULDER, LSHOULDER, RELBOW, LELBOW, RHAND, LHAND
        left_arm_id = [16, 14]  # , 12]
        right_arm_id = [15, 13]  # , 11]
        # left arm 3D bbox, min
        left_arm_bbox = pose[:, left_arm_id, :]
        right_arm_bbox = pose[:, right_arm_id, :]
        left_arm_min = left_arm_bbox.min(axis=1)  # (T, 3)
        left_arm_max = left_arm_bbox.max(axis=1)
        right_arm_min = right_arm_bbox.min(axis=1)
        right_arm_max = right_arm_bbox.max(axis=1)
        left_arm_diag = np.linalg.norm(left_arm_max - left_arm_min, axis=1)
        right_arm_diag = np.linalg.norm(right_arm_max - right_arm_min, axis=1)

        return (right_arm_diag,left_arm_diag)
    else:
        raise NotImplementedError

def bbox_diag_3d(pose):
    """
    only for VEHS-37 kpts,
    Input: (T, 37, 3)
    """
    # Indices in rtm_pose_37_keypoints_vicon_dataset_v1 (zero-based)
    # left arm 3D bbox, min
    body_bbox = pose[:, :, :]
    body_min = body_bbox.min(axis=1)  # (T, 3)
    body_max = body_bbox.max(axis=1)
    body_diag = np.linalg.norm(body_max - body_min, axis=1)

    return body_diag

def frame_id_from_mask(mask, args):
    if args.joint_format == 'RTM-37':
        # Define the relevant keypoint names (add the full list as needed)
        joint_names = ["LSHOULDER", "RSHOULDER",
                       "LELBOW", "RELBOW",
                       "LHAND", "RHAND"]

        output = {}
        for i, name in enumerate(joint_names):
            # Find all frame indices where this keypoint is True
            frame_ids = np.where(mask[:, i])[0]
            output[name] = frame_ids

        return output


def consecutive_segments(indices, min_length=20):
    """Turn sorted frame indices into list of (start, end) segments with min length."""
    if len(indices) == 0:
        return []

    segments = []
    start = prev = indices[0]

    for x in indices[1:]:
        if x == prev + 1:
            prev = x
        else:
            if (prev - start + 1) >= min_length:
                segments.append((start, prev))
            start = prev = x

    # add the last segment if long enough
    if (prev - start + 1) >= min_length:
        segments.append((start, prev))

    return segments



def frames_to_times(segments, fps):
    """
    segments: list of (start_frame, end_frame) inclusive
    returns: list of (start_sec, end_sec, duration_sec)
    """
    out = []
    for s, e in segments:
        start_t = s / fps
        end_t   = (e + 1) / fps   # inclusive end frame -> end is next frame boundary
        dur     = (e - s + 1) / fps
        out.append((start_t, end_t, dur))
    return out

def fmt_hms(seconds):
    # 00:00:SSS.mmm formatting
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def moving_average_1d(x, w):
    """Centered moving average with window w (odd preferred)."""
    x = np.asarray(x, dtype=float)
    if w <= 1:
        return x

    kernel = np.ones(w, dtype=float) / w
    pad = w // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    y = np.convolve(xpad, kernel, mode="valid")

    # Fix off-by-one length if w is even
    if len(y) > len(x):
        y = y[:len(x)]
    return y


def smooth_scores(scores, window=10):
    """Apply moving average along time axis for each column."""
    if window <= 1:
        return scores
    out = np.empty_like(scores, dtype=float)
    for j in range(scores.shape[1]):
        out[:, j] = moving_average_1d(scores[:, j], window)
    return out


def plot_normalized_scores(
        normalized_score,
        thresholds,
        fps=20,
        smooth_window=40,
        min_segment_sec=0.0
):
    """
    normalized_score: (num_frames, 6) array
    thresholds: array-like of length 6
    bad_frame_dict: dict {joint_name: frame_indices}
    smooth_window: odd int for moving-average smoothing (display only)
    min_segment_sec: only highlight bad-frame segments >= this duration (seconds)
    """
    joint_names = ["LSHOULDER", "RSHOULDER", "LELBOW", "RELBOW", "LHAND", "RHAND"]  # plot left on left

    time = np.arange(normalized_score.shape[0]) / fps / 60

    # 1) Smooth for visualization
    normalized_score = smooth_scores(np.asarray(normalized_score, dtype=float), smooth_window)

    # 2) Build segment map (filtered by duration)
    segs_map = {}

    mask = normalized_score > thresholds[np.newaxis, :]  # shape (28431, 6)
    bad_frame_dict = frame_id_from_mask(mask, args)
    min_len_frames = max(1, int(np.ceil(min_segment_sec * fps)))
    for joint in joint_names:
        idxs = np.asarray(bad_frame_dict.get(joint, []), dtype=int)
        idxs.sort()
        segs = consecutive_segments(idxs, min_length=min_len_frames)
        segs_map[joint] = segs
    plot_end_time = 23.5*60 #s
    for joint in joint_names:
        segments = segs_map.get(joint, [])
        if not segments:
            continue
        print(f"{joint}: {len(segments)} segments (dur >= {min_segment_sec:.3f}s)")
        for s, e in segments:
            start_t = s / fps
            end_t = (e + 1) / fps  # inclusive end -> next frame boundary
            dur = (e - s + 1) / fps

            if start_t < plot_end_time:
                print(f"    {fmt_hms(start_t)} – {fmt_hms(end_t)}  "
                  f"(dur {dur:.3f}s)  | frames [{s}, {e}]")

    # 3) Plot
    plt.figure(figsize=(14, 8))
    for i, joint in enumerate(joint_names):
        ax = plt.subplot(3, 2, i + 1)

        # line: smoothed score
        ax.plot(time, normalized_score[:, i], label=f"{joint} score", alpha=0.60)

        # threshold
        thr = float(thresholds[i])
        ax.axhline(thr, linestyle="--", label="Threshold", color="red")

        # highlight segments
        for s, e in segs_map.get(joint, []):
            start_t = s / fps /60
            end_t = (e + 1) / fps / 60
            ax.axvspan(start_t, end_t, alpha=0.20, label="Segments to remove")

        # fixed y-limits as you had
        ax.set_ylim(0, 0.6)
        ax.set_xlim(0, plot_end_time/60) #1400/60


        ax.set_title(joint)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Normalized score")
        ax.grid(True, alpha=0.3)

        # dedupe legend
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), loc="upper right")


    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # read arguments
    args = parse_args()
    estimate_pose = MB_output_pose_file_loader(args)
    data_key = 'joint3d_image'

    args.joint_format = 'RTM-37'
    rtm_pose_37_keypoints_vicon_dataset_v1 = ['PELVIS', 'RWRIST', 'LWRIST', 'RHIP', 'LHIP', 'RKNEE', 'LKNEE', 'RANKLE', 'LANKLE', 'RFOOT', 'LFOOT', 'RHAND', 'LHAND', 'RELBOW', 'LELBOW', 'RSHOULDER',
                                              'LSHOULDER', 'HEAD', 'THORAX', 'HDTP', 'REAR', 'LEAR', 'C7', 'C7_d', 'SS', 'RAP_b', 'RAP_f', 'LAP_b', 'LAP_f', 'RLE', 'RME', 'LLE', 'LME', 'RMCP2',
                                              'RMCP5',
                                              'LMCP2', 'LMCP5']
    score = center_z_score(estimate_pose, args)

    normalized_score = score.copy()
    right_arm_diag, left_arm_diag = arm_bbox_diag_3d(estimate_pose, args)
    right_arm_diag = moving_average_1d(np.asarray(right_arm_diag, dtype=float), 40)
    left_arm_diag = moving_average_1d(np.asarray(left_arm_diag, dtype=float), 40)

    normalized_score[:,1::2] = score[:, 1::2] / (right_arm_diag[:, None]+1e-7)
    normalized_score[:,::2] = score[:, ::2] / (left_arm_diag[:, None]+1e-7)

    # body_diag = bbox_diag_3d(estimate_pose)
    # # body_diag = moving_average_1d(np.asarray(body_diag, dtype=float), 10)
    # normalized_score = score/body_diag[:, None]

    # thresholds by joint type:
    # column order: L-Shoulder, R-shoulder ...
    thresholds = np.array([0.3]*6, dtype=float)
    thresholds[-2:] = 0.3 # hand

    assert normalized_score.shape[1] == thresholds.size, \
        f"score has {normalized_score.shape[1]} cols but thresholds has {thresholds.size}"



    plot_normalized_scores(normalized_score, thresholds, fps=20)
