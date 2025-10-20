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
        args.output_dir = os.path.join(os.path.dirname(args.estimate_file), 'support_kpts_score')
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


# ---------- 1) Data preparation ----------
def prepare_score_data(
        normalized_score,
        thresholds,
        fps=20,
        smooth_window=40,
        min_segment_sec=0.0,
        args=None
):
    """
    Returns a dict with:
      - 'smoothed': (T,6) smoothed scores
      - 'mask': (T,6) boolean, True where score > threshold
      - 'bad_frame_dict': {joint_name: np.array(frame_ids)}
      - 'segs_map': {joint_name: [(start,end), ...]} with min length filter
    """
    joint_names = ["LSHOULDER", "RSHOULDER", "LELBOW", "RELBOW", "LHAND", "RHAND"]

    # Smooth for visualization
    smoothed = smooth_scores(np.asarray(normalized_score, dtype=float), smooth_window)

    # Over-threshold mask
    thresholds = np.asarray(thresholds, dtype=float)
    mask = smoothed > thresholds[np.newaxis, :]

    # Build segments per joint
    if args is None:
        raise ValueError("prepare_score_data requires args for frame_id_from_mask naming.")
    bad_frame_dict = frame_id_from_mask(mask, args)

    min_len_frames = max(1, int(np.ceil(min_segment_sec * fps)))
    segs_map = {}
    for joint in joint_names:
        idxs = np.asarray(bad_frame_dict.get(joint, []), dtype=int)
        idxs.sort()
        segs_map[joint] = consecutive_segments(idxs, min_length=min_len_frames)

    return {
        "smoothed": smoothed,
        "mask": mask,
        "bad_frame_dict": bad_frame_dict,
        "segs_map": segs_map,
        "joint_names": joint_names,
        "thresholds": thresholds,
        "fps": fps
    }

# ---------- 2) Full-sequence plot (minutes on x) ----------
def plot_normalized_scores(
        processed,
        plot_end_time_sec=23.5 * 60
):
    """
    Uses output from prepare_score_data to produce a full-sequence plot.
    - x in minutes (keeps your original)
    - red threshold line
    - light-blue fill where score > threshold
    - segment spans shown
    """
    smoothed = processed["smoothed"]
    thresholds = processed["thresholds"]
    joint_names = processed["joint_names"]
    fps = processed["fps"]
    segs_map = processed["segs_map"]

    # x in minutes for this overview plot
    T = smoothed.shape[0]
    time_min = (np.arange(T) / fps) / 60.0

    plt.figure(figsize=(14, 8))
    for i, joint in enumerate(joint_names):
        ax = plt.subplot(3, 2, i + 1)

        # line
        ax.plot(time_min, smoothed[:, i], label=f"{joint} score", alpha=0.60, color="blue")

        # threshold
        thr = float(thresholds[i])
        ax.axhline(thr, linestyle="--", linewidth=1.5, color="red", label="Threshold")

        # light-blue shading where over threshold
        ax.fill_between(time_min, 0, smoothed[:, i], where=(smoothed[:, i] > thr),
                        color="lightblue", alpha=0.30, step="pre")

        # highlight segments
        for s, e in segs_map.get(joint, []):
            start_t = s / fps / 60.0
            end_t = (e + 1) / fps / 60.0
            ax.axvspan(start_t, end_t, alpha=0.20, label="Segments to remove")

        ax.set_ylim(0, 0.6)
        ax.set_xlim(0, plot_end_time_sec / 60.0)
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

# ---------- 3) Per-frame render (seconds on x) ----------
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

def plot_normalized_scores_by_frame(
        processed,
        render_dir,
        frame_interval=243,
        frame_range=None,
        frame_range_max=None,
        skip_first=False,
        tick_targets_per_seg=6,
        title_fs=18,
        axis_label_fs=16
):
    """
    Per-frame sequence of images using processed data from prepare_score_data.
    - seconds on x
    - thicker red threshold
    - blue dotted vertical lines every frame_interval
    - blue score line (faint full, strong up-to-frame)
    - bigger black current-frame dot
    - black number label (hidden if too close to axis limits)
    - light-blue fill where score > threshold
    - no legends
    - Titles formatted like "<Joint Name>'s supporting keypoints"
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    smoothed = processed["smoothed"]
    thresholds = processed["thresholds"]
    joint_names = processed["joint_names"]
    fps = processed["fps"]

    if smoothed is None or len(smoothed) == 0:
        raise ValueError("processed['smoothed'] is empty")

    T = smoothed.shape[0]
    num_series = smoothed.shape[1]
    assert num_series == 6, f"Expected 6 columns, got {num_series}"

    # default frame range
    if frame_range is None:
        frame_range = [0, T]

    # build segment ranges from frame_range_max
    if frame_range_max is None:
        frame_ranges = [frame_range]
    elif isinstance(frame_range_max, int):
        frame_ranges = []
        total = frame_range[1] - frame_range[0]
        n_segs = int(np.floor(total / frame_range_max))
        for seg_id in range(n_segs):
            s = frame_range[0] + seg_id * frame_range_max
            e = frame_range[0] + (seg_id + 1) * frame_range_max
            frame_ranges.append([s, e])
    elif isinstance(frame_range_max, list):
        frame_ranges = []
        left = 0
        fr_list = frame_range_max[:]
        if skip_first and len(fr_list) > 0:
            left = fr_list[0]
            fr_list = fr_list[1:]
        for segment_frame_len in fr_list:
            right = left + segment_frame_len
            frame_ranges.append([left, right])
            left = right
    else:
        raise ValueError(f"frame_range_max must be None, int or list, not {type(frame_range_max)}")

    os.makedirs(render_dir, exist_ok=True)

    # seconds formatter for ticks
    def seconds_formatter(x, pos):
        return f"{x / fps:.0f}"

    y_lo, y_hi = 0.0, 0.6
    margin_frac = 0.02
    y_margin = (y_hi - y_lo) * margin_frac

    def pretty_title(name: str) -> str:
        base = (name or "").replace("_", " ").strip()
        return f"{base}'s supporting keypoints"

    for seg_id, fr in enumerate(frame_ranges):
        left, right = fr
        print(f"[{seg_id+1}/{len(frame_ranges)}] Saving normalized scores frames to {render_dir}")
        data_len = max(1, right - left)
        tick_interval = max(1, data_len // max(1, tick_targets_per_seg))

        for frame_id in range(left, right):
            if frame_id % 20 != 0:
                continue

            fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex=False)
            axes = axes.ravel()

            for i, ax in enumerate(axes):
                in_left_col = (i % 2 == 0)
                in_bottom_row = (i // 2 == 2)

                y = smoothed[:, i]
                thr = float(thresholds[i])

                # threshold line
                ax.axhline(thr, linestyle="--", linewidth=1.8, color="red")

                # 243-frame markers
                for xline in range(left, right, frame_interval):
                    ax.axvline(xline, linestyle="dotted", linewidth=0.8, alpha=0.6, color="blue")

                # full series (faint) and up-to-frame (strong)
                ax.plot(range(left, right), y[left:right], alpha=0.25, linewidth=1.0, color="blue")
                ax.plot(range(left, frame_id + 1), y[left:frame_id + 1], alpha=0.95, linewidth=1.5, color="blue")

                # light-blue shading
                ax.fill_between(
                    np.arange(left, right), y_lo, y[left:right],
                    where=(y[left:right] > thr),
                    color="lightblue", alpha=0.30, step="pre"
                )

                # current frame marker
                ax.axvline(frame_id, linestyle="--", linewidth=0.7, alpha=0.75, color="k")
                y_val = float(y[frame_id])
                ax.plot(frame_id, y_val, marker="o", markersize=4, color="black", zorder=5)

                # label only if comfortably inside
                if (y_lo + y_margin) <= y_val <= (y_hi - y_margin):
                    ax.text(frame_id, y_val, f"{y_val:.2f}",
                            fontsize=12, ha="left", va="bottom", clip_on=True)

                # cosmetics
                ax.set_xlim(left, right)
                ax.set_ylim(y_lo, y_hi)
                ax.set_title(pretty_title(joint_names[i]), fontsize=title_fs, weight="bold")

                if in_left_col:
                    ax.set_ylabel("Normalized score", fontsize=axis_label_fs)
                if in_bottom_row:
                    ax.set_xlabel("Time (s)", fontsize=axis_label_fs)

                ax.xaxis.set_major_formatter(FuncFormatter(seconds_formatter))
                ax.xaxis.set_major_locator(MultipleLocator(tick_interval))
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            out_path = os.path.join(render_dir, f"support_kpt_score_{frame_id:06d}.png")
            plt.savefig(out_path, dpi=150)
            plt.close()




if __name__ == '__main__':
    # read arguments
    args = parse_args()
    # args.estimate_file = '/Volumes/Z/RTMPose/37kpts_rtmw_v5/20fps/RTMW37kpts_v2_20fps-finetune-pitch-correct-5-angleLossV2-only/Industry_both/X3D.npy'
    estimate_pose = MB_output_pose_file_loader(args)

    # estimate_pose = estimate_pose_2[:243 * 21]
    # estimate_pose= estimate_pose_1[:243 * 21]


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

    # 1) Prepare once
    processed = prepare_score_data(
        normalized_score=normalized_score,
        thresholds=thresholds,
        fps=20,
        smooth_window=40,
        min_segment_sec=0.0,
        args=args  # needed for naming in frame_id_from_mask
    )

    # 2) Full-sequence overview (minutes on x)
    plot_normalized_scores(processed, plot_end_time_sec=23.5 * 60)

    # 3) Per-frame rendering (seconds on x), same processed data
    render_dir = os.path.join(args.output_dir)
    if "VEHS7M" in render_dir:
        frame_range_max = None
    elif "Industry/angles" in render_dir:
        frame_range_max = list(np.array([2, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2]) * args.MB_data_stride)
    elif "Industry_2" in render_dir:
        frame_range_max = list(np.array([11, 2, 9, 7, 7, 7, 3, 7, 22, 4, 17]) * args.MB_data_stride)
    elif "Industry_both" in render_dir:
        frame_range_max = list(np.array([2, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2, 11, 2, 9, 7, 7, 7, 3, 7, 22, 4, 17]) * args.MB_data_stride)

    # plot_normalized_scores_by_frame(
    #     processed=processed,
    #     render_dir=render_dir,
    #     frame_interval=args.MB_data_stride,  # 243 by default
    #     frame_range=[0, processed["smoothed"].shape[0]],
    #     frame_range_max=frame_range_max,
    #     skip_first=False,
    #     tick_targets_per_seg=6,  # fewer ticks
    #     title_fs=18,
    #     axis_label_fs=16
    # )
    fps = 1
    print(f"Copy command to merge frames into video at {fps} fps:")
    print(f"python conversion_scripts/video.py --imgs_dir {render_dir} --fps {fps}")  # --delete_imgs