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
    parser.add_argument('--RTM_infer_folder', type=str, default='/media/leyang/My Book2/VEHS/VEHS data collection round 3/RTM2D/RTMWPose_VEHS7M_37kpts_v5_2-b')
    parser.add_argument('--base_frame_folder', type=str, default='/media/leyang/My Book2/VEHS/VEHS-7M/img/20fps')
    parser.add_argument('--infer_pose_type', type=str, choices=['rtm37_from_coco133', 'rtm37_from_37'], default='rtm37_from_37')
    parser.add_argument('--GT_ann_file', type=str, default=None)
    # parser.add_argument('--GT_ann_file', type=str, default='/Volumes/Z/RTMPose/37kpts_v1/GT/VEHS_6DCOCO_downsample20_keep1_validate.json')
    parser.add_argument('--config_file', type=str, default=r'config/experiment_config/37kpts/Inference-RTMPose-MB-20fps-VEHS7M.yaml')
    parser.add_argument('--skeleton_file', type=str, default=r'config/VEHS_ErgoSkeleton_info/Ergo-Skeleton-37.yaml')
    parser.add_argument('--type', type=str, default='body')

    parser.add_argument('--number_of_keypoints', type=int, default=37)
    parser.add_argument('--img_shape', type=tuple, default=(1200,1920), help='height, width in px')
    parser.add_argument('--verbose', type=bool, default=True)
    # select from "a", "b", "c"
    parser.add_argument('--norm_mode', type=str, choices=['bbox_diag', 'part_diag'], default='part_diag')


    args = parser.parse_args()
    with open(args.config_file, 'r') as stream:
        data = yaml.safe_load(stream)
        args.name_list = data['name_list']
    return args


if __name__ == '__main__':
    args = phrase_args()

    # read GT file
    if args.GT_ann_file is not None:
        coco = COCO(args.GT_ann_file)
        raise NotImplementedError

    for root, dirs, files in os.walk(args.RTM_infer_folder):
        dirs.sort()  # Sort directories in-place --> important, will change walk sequence
        files.sort(key=str.lower)  # Sort files in-place
        for file in files:
            if file.startswith('.'):  # ignore hidden files
                continue

            if file.endswith('.json'):
                continue
                # reference to `conversion_scripts/Veeru/rtm_infer_GT_to_2D_eval.py`
                raise NotImplementedError
            elif file.endswith('.npy'):
                with open(os.path.join(root, file), 'rb') as f:
                    inference_pose = np.load(f)
                print(f"Processing {root}/{file}") if args.verbose else None
                print(f"Inference pose shape: {inference_pose.shape}") if args.verbose else None
                estimate_skeleton = VEHSErgoSkeleton_angles(args.skeleton_file)
                estimate_skeleton.load_name_list_and_np_points(args.name_list, inference_pose)

                frame_no = 1200 #random.randint(0, estimate_skeleton.frame_number)
                # rtm pose v3 naming format by veeru
                # video_name = file[8:-14]
                # baseimage = os.path.join(args.base_frame_folder, video_name, f'{frame_no+1:05d}.png')
                # baseimage_folder = os.path.join(args.base_frame_folder, video_name)

                # rtmw pose v5 naming format by veeru
                basename = os.path.basename(file)  # 'results_Activity07_1_66920731.npy'
                parts = basename.replace('.npy', '').split('_')  # ['results', 'Activity07', '1', '66920731']
                activity_id = parts[1].lower().replace('activity', 'activity')  # 'activity07'
                cam_id = int(parts[3])  # 66920731

                # Extract subject ID from folder name like '3-1' -> S01
                folder_name = os.path.basename(root)  # '3-1'
                subject_id = int(folder_name.split('-')[0])  # 3
                subject_str = f'S{subject_id:02d}'  # 'S03'

                # Construct new path
                new_filename = f'{subject_str}-{activity_id}-{cam_id}-{frame_no+1:06d}.jpg'  # not working now, there is a mistake in 20fps images, the name of subject was replaced by FullCollection, need to regenerate images
                baseimage_folder = os.path.join(args.base_frame_folder, 'train')
                baseimage = os.path.join(baseimage_folder, new_filename)
                file_exists = os.path.exists(baseimage)
                print(f"baseimage exists {file_exists} - {baseimage}") if args.verbose else None

                estimate_skeleton.plot_2d_pose_frame(frame=frame_no) #, baseimage=baseimage)
                # break
                # output_frame_folder = os.path.join(args.RTM_infer_folder, video_name)
                # print(f"Output frame folder: {output_frame_folder}")
                # estimate_skeleton.plot_2d_pose(output_frame_folder, baseimage_folder=baseimage_folder)




"""
Snippet to convert mp4 to png
shopt -s nullglob; for f in *.mp4; do d="20fps/${f%.*}"; mkdir -p "$d"; ffmpeg -i "$f" -vf fps=20 "$d/%05d.png"; done

shopt -s nullglob; for f in *.mp4; do d="5fps/${f%.*}"; mkdir -p "$d"; ffmpeg -i "$f" -vf fps=5 "$d/%05d.png"; done

shopt -s nullglob; for f in *.mp4; do mkdir -p "50fps/${f%.*}" && ffmpeg -i "$f" -vf fps=50 "50fps/${f%.*}/%05d.png"; done
"""



