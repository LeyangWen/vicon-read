import json
import random

import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import ref
import copy
import argparse
from pycocotools.coco import COCO
import pandas as pd
import sys

from utility import *
from Skeleton import *
###############################################
#
# Used to evaluate RTM pose inference results
# PCK, MPJPE, MPJPE_norm
#
###############################################



def phrase_args():
    parser = argparse.ArgumentParser(description='RTM pose inference to 2D metric')

    parser.add_argument('--RTM_infer_folder', type=str, default='/Volumes/Z/RTMPose/37kpts_rtmw_v5/20fps/2D/exp_2b_industry_videos_20fps')
    parser.add_argument('--video_folder', type=str, default='/Volumes/Z/RTMPose/inference_vid/Industry')

    # parser.add_argument('--RTM_infer_folder', type=str, default='/Volumes/Z/RTMPose/37kpts_rtmw_v5/20fps/2D/exp_2b_industry_videos_20fps')
    # parser.add_argument('--video_folder', type=str, default='/Volumes/Z/RTMPose/inference_vid/Industry_2')

    # parser.add_argument('--RTM_infer_folder', type=str, default='/Volumes/Z/RTMPose/37kpts_rtmw_v5/20fps/2D/Industry_3')
    # parser.add_argument('--video_folder', type=str, default='/Volumes/Z/RTMPose/inference_vid/Industry_3')

    parser.add_argument('--infer_pose_type', type=str, choices=['rtm37_from_coco133', 'rtm37_from_37'], default='rtm37_from_37')
    parser.add_argument('--GT_ann_file', type=str, default=None)
    parser.add_argument('--config_file', type=str, default=r'config/experiment_config/37kpts/Inference-RTMPose-MB-20fps-VEHS7M.yaml')
    parser.add_argument('--skeleton_file', type=str, default=r'config/VEHS_ErgoSkeleton_info/Ergo-Skeleton-37.yaml')
    parser.add_argument('--type', type=str, default='body')

    parser.add_argument('--number_of_keypoints', type=int, default=37)
    parser.add_argument('--img_shape', type=tuple, default=(1200, 1920), help='height, width in px')
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--norm_mode', type=str, choices=['bbox_diag', 'part_diag'], default='part_diag')
    parser.add_argument('--filter_lowConf_legs', type=bool, default=True)
    args = parser.parse_args()
    with open(args.config_file, 'r') as stream:
        data = yaml.safe_load(stream)
        args.name_list = data['name_list']
    args.output_folder = os.path.join(args.RTM_infer_folder, 'filtered_2')
    return args

def match_video(npy_filename, video_folder):
    """
    Find a video in video_folder whose stem is contained in the npy filename.
    e.g. npy_filename='results_AAA_Rick_1.npy' matches 'Rick_1.mp4'
    Returns full video path or None.
    """
    npy_stem = os.path.splitext(npy_filename)[0]  # results_AAA_Rick_1
    best_match = None
    best_len = 0

    for vid_file in os.listdir(video_folder):
        if not vid_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue
        vid_stem = os.path.splitext(vid_file)[0]  # Rick_1
        if vid_stem in npy_stem and len(vid_stem) > best_len:
            best_match = vid_file
            best_len = len(vid_stem)

    if best_match:
        return os.path.join(video_folder, best_match)
    return None

def extrapolate_point(frames, index, part_index,filtered_keyps):
    """
    From Veeru
    """
    if index == 0 or index == len(frames) - 1:
        for dims in range(frames.shape[2]):
            filtered_keyps[index,part_index,dims] = frames[index,part_index,dims]
        return filtered_keyps
    if (np.any(frames[index-1])) and (np.any(frames[index+1])):
        for dims in range(frames.shape[2]):
            prev = frames[index - 1,part_index,dims]
            next = frames[index + 1,part_index,dims]
            if frames.shape[2] == 3:
                if dims == frames.shape[2] - 1:
                    filtered_keyps[index,part_index,dims] = frames[index,part_index,dims]
                else:
                    filtered_keyps[index,part_index,dims] = (prev + next) // 2
            if frames.shape[2] == 4:
                if dims == frames.shape[2] - 1:
                    filtered_keyps[index,part_index,dims] = frames[index,part_index,dims]
                else:
                    filtered_keyps[index,part_index,dims] = (prev + next) / 2
    return filtered_keyps


def smooth_point(frames, index, part_index, window=8):
    """
    From Veeru
    Smooth points out based on their adjacent points
    """
    start_index = max(0, index - window // 2)
    end_index = min(len(frames), index + window // 2 + 1)
    confidence_score = frames[index,part_index,-1]
    dims_sum = []
    for dims in range(frames.shape[2]):
        if dims != frames.shape[2] - 1:
            dims_sum.append(0.0)
    weight=0.0

    for i in range(start_index, end_index):
        point = frames[i,part_index]
        if np.any(frames[i]):
            # Weight each point by its distance from the target point
            multiplier = (abs(i - index) - window) ** 2 / (window ** 2)
            for dims in range(frames.shape[2]):
                if dims != frames.shape[2] - 1:
                    dims_sum[dims] += point[dims] * multiplier
            weight += multiplier

    if frames.shape[2] == 3:
        if np.any(frames[start_index:end_index,part_index,2] < 0.3):  # blip removal
            confidence_score = 0.0  # In case any frame blip w/low confidence, set all surrounding frame to zero, remove this
# 0.3 for lots, 0.45 for elbow, wrist, and finger/hand/MCPs, --> only for 2D visualization
    # for 3D, if both ankle or both knee is not visible, dont visualize leg, still visualize the hip

    # if np.count_nonzero(frames[start_index:end_index,part_index,-1] < 0.3) > 3:
    #     confidence_score = 0.0


    if weight == 0.0:
        if frames.shape[2] == 3:
            return frames[index,part_index,0], frames[index,part_index,1],confidence_score
        if frames.shape[2] == 4:
            return frames[index,part_index,0], frames[index,part_index,1],frames[index,part_index,2],confidence_score

    if frames.shape[2] == 3:
        return int(dims_sum[0] / weight), int(dims_sum[1]/ weight),confidence_score
    if frames.shape[2] == 4:
        return dims_sum[0] / weight, dims_sum[1]/ weight, dims_sum[2]/ weight,confidence_score

def extrapolate_and_smooth(all_keyps):
    """
    From Veeru
    """
    filtered_keyps = np.zeros(all_keyps.shape)
    final_filtered_keyps = np.zeros(all_keyps.shape)
    for i,frame in enumerate(all_keyps):
        if not np.any(frame): # Check if the frame is empty and has no prediction
                continue
        for j,part in enumerate(frame):
            filtered_keyps = extrapolate_point(all_keyps,i,j,filtered_keyps)
    for i,frame in enumerate(all_keyps):
        if not np.any(frame): # Check if the frame is empty and has no prediction
                continue
        for j,part in enumerate(frame):
            window = 8  # Default window size
            # if j in [ref.rtm_pose_keypoints_vicon_dataset.index('left_pinky'),ref.rtm_pose_keypoints_vicon_dataset.index('right_pinky'),ref.rtm_pose_keypoints_vicon_dataset.index('left_index'),ref.rtm_pose_keypoints_vicon_dataset.index('right_index'),ref.rtm_pose_keypoints_vicon_dataset.index('left_middle_mcp'),ref.rtm_pose_keypoints_vicon_dataset.index('right_middle_mcp')]:
            #     window = 4 # Smaller window size for hand keypoints
            if all_keyps.shape[2] == 3:
                final_filtered_keyps[i,j,0],final_filtered_keyps[i,j,1],final_filtered_keyps[i,j,2] = smooth_point(filtered_keyps,i,j,window=window)
            if all_keyps.shape[2] == 4:
                final_filtered_keyps[i,j,0],final_filtered_keyps[i,j,1],final_filtered_keyps[i,j,2],final_filtered_keyps[i,j,3] = smooth_point(filtered_keyps,i,j,window=window)
    return final_filtered_keyps


if __name__ == '__main__':
    args = phrase_args()

    if args.GT_ann_file is not None:
        coco = COCO(args.GT_ann_file)
        raise NotImplementedError

    os.makedirs(args.output_folder, exist_ok=True)

    for root, dirs, files in os.walk(args.RTM_infer_folder):
        dirs.sort()
        files.sort(key=str.lower)
        for file in files:
            if file.startswith('.'):
                continue

            if file.endswith('.json'):
                continue

            elif file.endswith('.npy'):
                video_path = match_video(file, args.video_folder)
                if video_path is None:
                    print(f"No matching video for {file}, skipping")
                    continue

                vid_stem = os.path.splitext(os.path.basename(video_path))[0]
                # make sure output and input folder is not the same using os
                assert os.path.dirname(video_path) != args.output_folder, "Output folder cannot be the same as video folder to avoid overwriting videos"
                output_path = os.path.join(args.output_folder, f"{vid_stem}.mp4")

                print(f"Processing {file} -> {os.path.basename(video_path)}")

                with open(os.path.join(root, file), 'rb') as f:
                    inference_pose = np.load(f)
                inference_pose[:,:,:3] = extrapolate_and_smooth(inference_pose[:,:,:3])
                estimate_skeleton = VEHSErgoSkeleton_angles(args.skeleton_file)
                estimate_skeleton.load_name_list_and_np_points(args.name_list, inference_pose)
                # estimate_skeleton.filter_lowpass(cutoff=6, keypoint_fps=20)
                # estimate_skeleton.filter_confidence(threshold=5.0)

                if args.filter_lowConf_legs:
                    kpt_names = args.name_list  # list of kpt names corresponding to the mask columns
                    conf_2d = inference_pose[:,:,2]  # shape: (frame_num, kpt_num)
                    threshold = 5.8
                    left_knee_mask = conf_2d[:, kpt_names.index('LKNEE')] < threshold
                    right_knee_mask = conf_2d[:, kpt_names.index('RKNEE')] < threshold
                    left_ankle_mask = conf_2d[:, kpt_names.index('LANKLE')] < threshold
                    right_ankle_mask = conf_2d[:, kpt_names.index('RANKLE')] < threshold
                    left_foot_mask = conf_2d[:, kpt_names.index('LFOOT')] < threshold
                    right_foot_mask = conf_2d[:, kpt_names.index('RFOOT')] < threshold
                    both_knee_mask = left_knee_mask & right_knee_mask
                    both_knee_mask = both_knee_mask.reshape(-1)  # ensure it's 1D
                    for kpt_name in ['LKNEE', 'RKNEE', 'LANKLE', 'RANKLE', 'LFOOT', 'RFOOT', 'LHIP', 'RHIP']:
                        estimate_skeleton.poses[kpt_name][both_knee_mask] = np.nan
                    both_foot_ankle_mask = left_foot_mask & right_foot_mask & left_ankle_mask & right_ankle_mask
                    both_foot_ankle_mask = both_foot_ankle_mask.reshape(-1)  # ensure it's 1D
                    for kpt_name in ['LANKLE', 'RANKLE', 'LFOOT', 'RFOOT']:
                        estimate_skeleton.poses[kpt_name][both_foot_ankle_mask] = np.nan




                estimate_skeleton.plot_2d_pose_cv(
                    video_path=video_path,
                    output_path=output_path
                )




