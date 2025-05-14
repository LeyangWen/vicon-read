
#!/usr/bin/env python3
import argparse
import pickle
import numpy as np
from numpy.ma.core import absolute
from scipy.stats import ttest_rel

from conversion_scripts.np_to_angles import parse_args


def phrase_args():
    parser = argparse.ArgumentParser(
        description="Compare per-frame angle errors between two models using a paired t‑test."
    )

    # more comprehensive file in 1
    parser.add_argument('--file_AE_1', type=str, default=r'/Users/leyangwen/Documents/Pose/paper/results/test_VEHS_AE.pkl')
    parser.add_argument('--fps_1', type=int, default=50)

    parser.add_argument('--file_AE_2', type=str, default=r'/Users/leyangwen/Downloads/scratch_pose/H36M-17kpts/FT_MB_release_MB_ft_h36m/VEHS7M-test-1920x1200/results/test_VEHS_AE.pkl')
    parser.add_argument('--fps_2', type=int, default=20)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = phrase_args()
    # Load the two error dictionaries
    with open(args.file_AE_1, 'rb') as f:
        errors_1 = pickle.load(f)
    with open(args.file_AE_2, 'rb') as f:
        errors_2 = pickle.load(f)


    for angle in errors_2.keys():
        if "Neck" in angle:
            continue
        err1 = np.array(errors_1[angle])
        err2 = np.array(errors_2[angle])

        # downsample by common denominator
        common_denominator = np.gcd(args.fps_1, args.fps_2)
        # print(f"downsample to {common_denominator} fps")
        err1 = err1[::args.fps_1 // common_denominator]
        err2 = err2[::args.fps_2 // common_denominator]

        frame_no_1 = len(err1)
        frame_no_2 = len(err2)

        # print(f"diff:{frame_no_1-frame_no_2}, frame_no_1: {frame_no_1}, frame_no_2: {frame_no_2}")

        # err2 = err2[:frame_no_1]
        err1 = err1[:frame_no_2]
        # paired t‑test
        absolute_err1 = np.absolute(err1)
        absolute_err2 = np.absolute(err2)

        t_stat, p_val = ttest_rel(absolute_err1, absolute_err2, nan_policy='omit')
        # 1 is mine,
        # 2 is others
        print(f"{angle} - t-statistic: {t_stat}, p-value: {p_val}")
