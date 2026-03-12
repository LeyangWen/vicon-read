#!/usr/bin/env python3
import argparse
import pickle
import numpy as np
import csv
import os
from scipy.stats import ttest_rel


def phrase_args():
    parser = argparse.ArgumentParser(description="Video-level paired t-test for joint angle errors.")
    parser.add_argument("--file_AE_1", type=str, default=r"/Volumes/Z/RTMPose/37kpts_rtmw_v5/20fps/mesh_compare/SMPL_RTM37kpts_V1/results_compare/validate_paper_AE.pkl")
    parser.add_argument("--file_AE_2", type=str, default=r"/Volumes/Z/RTMPose/37kpts_rtmw_v5/20fps/RTMW37kpts_v2_20fps-finetune-pitch-correct-5-angleLossV2-only/VEHS7M-Validate/results_compare/validate_PlotMode-paper_MergeLR-True_TryWrist-False_AE.pkl")
    return parser.parse_args()


def segment_indices_by_source(source_list):
    segments = []
    start = 0
    for i in range(1, len(source_list)):
        if source_list[i] != source_list[i - 1]:
            segments.append((start, i))
            start = i
    segments.append((start, len(source_list)))
    return segments


if __name__ == "__main__":

    args = phrase_args()

    with open(args.file_AE_1, "rb") as f:
        errors_1 = pickle.load(f)

    with open(args.file_AE_2, "rb") as f:
        errors_2 = pickle.load(f)

    source = errors_1["source"]
    segments = segment_indices_by_source(source)

    angles = [k for k in errors_1.keys() if k != "source"]

    rows = []

    print("Angle | MAE_A | MAE_B | Delta | p-value")

    # ---- per-joint analysis ----
    for angle in angles:

        err1 = np.array(errors_1[angle])
        err2 = np.array(errors_2[angle])

        video_mae_1 = []
        video_mae_2 = []

        for start, end in segments:

            seg1 = err1[start:end]
            seg2 = err2[start:end]

            L = min(len(seg1), len(seg2))
            if L == 0:
                continue

            seg1 = seg1[:L]
            seg2 = seg2[:L]

            mae1 = np.nanmean(np.abs(seg1))
            mae2 = np.nanmean(np.abs(seg2))

            video_mae_1.append(mae1)
            video_mae_2.append(mae2)

        video_mae_1 = np.array(video_mae_1)
        video_mae_2 = np.array(video_mae_2)

        mae1_mean = np.mean(video_mae_1)
        mae2_mean = np.mean(video_mae_2)
        delta = mae1_mean - mae2_mean

        t_stat, p_val = ttest_rel(video_mae_1, video_mae_2, nan_policy="omit")

        print(f"{angle:20s} {mae1_mean:6.2f} {mae2_mean:6.2f} {delta:7.2f} {p_val:.3e}")

        rows.append([angle, mae1_mean, mae2_mean, delta, p_val])

    # ---- global analysis across all joints ----

    global_video_mae_1 = []
    global_video_mae_2 = []

    for start, end in segments:

        joint_errors_1 = []
        joint_errors_2 = []

        for angle in angles:

            err1 = np.array(errors_1[angle])[start:end]
            err2 = np.array(errors_2[angle])[start:end]

            L = min(len(err1), len(err2))
            if L == 0:
                continue

            joint_errors_1.append(np.abs(err1[:L]))
            joint_errors_2.append(np.abs(err2[:L]))

        if len(joint_errors_1) == 0:
            continue

        joint_errors_1 = np.concatenate(joint_errors_1)
        joint_errors_2 = np.concatenate(joint_errors_2)

        mae1 = np.nanmean(joint_errors_1)
        mae2 = np.nanmean(joint_errors_2)

        global_video_mae_1.append(mae1)
        global_video_mae_2.append(mae2)

    global_video_mae_1 = np.array(global_video_mae_1)
    global_video_mae_2 = np.array(global_video_mae_2)

    global_mae1 = np.mean(global_video_mae_1)
    global_mae2 = np.mean(global_video_mae_2)
    global_delta = global_mae1 - global_mae2

    t_stat_global, p_val_global = ttest_rel(global_video_mae_1, global_video_mae_2, nan_policy="omit")

    rows.append(["Average", global_mae1, global_mae2, global_delta, p_val_global])

    print(f"{'Average':20s} {global_mae1:6.2f} {global_mae2:6.2f} {global_delta:7.2f} {p_val_global:.3e}")

    # ---- determine common directory ----
    common_dir = os.path.commonpath([args.file_AE_1, args.file_AE_2])
    if os.path.isfile(common_dir):
        common_dir = os.path.dirname(common_dir)

    out_csv = os.path.join(common_dir, "angle_comparison.csv")

    # ---- write csv ----
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["Angle", "MAE_A", "MAE_B", "Delta", "p_value"])
        writer.writerows(rows)

        writer.writerow([])
        writer.writerow(["file_AE_1", args.file_AE_1])
        writer.writerow(["file_AE_2", args.file_AE_2])

    print(f"\nSaved results to: {out_csv}")