import json
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import ref
import copy
import argparse
from pycocotools.coco import COCO
import pandas as pd

from conversion_scripts.Veeru.rtm_pose_read_kps import *
from utility import *

###############################################
#
# Used to evaluate RTM pose inference results
# PCK, MPJPE, MPJPE_norm
#
###############################################



def phrase_args():
    parser = argparse.ArgumentParser(description='RTM pose inference to 2D metric')
    parser.add_argument('--RTM_infer_folder', type=str, default='/Volumes/Z/RTMPose/37kpts_v2/best_epoch/lab_videos_phase_2_correct_epoch_best_30')
    # parser.add_argument('--RTM_infer_folder', type=str, default='/Volumes/Z/RTMPose/37kpts_v1/best_epoch_40/outputs_epoch_best')
    # parser.add_argument('--RTM_infer_folder', type=str, default='/Volumes/Z/RTMPose/original_rtmpose/outputs_rtmpose_old')
    parser.add_argument('--infer_pose_type', type=str, choices=['rtm37_from_coco133', 'rtm37_from_37'], default='rtm37_from_37')
    parser.add_argument('--GT_ann_file', type=str, default='/Volumes/Z/RTMPose/37kpts_v1/GT/VEHS_6DCOCO_downsample20_keep1_validate.json')
    parser.add_argument('--number_of_keypoints', type=int, default=37)
    parser.add_argument('--img_shape', type=tuple, default=(1200,1920), help='height, width in px')
    parser.add_argument('--verbose', type=bool, default=False)
    # select from "a", "b", "c"
    parser.add_argument('--norm_mode', type=str, choices=['bbox_diag', 'part_diag'], default='part_diag')

    args = parser.parse_args()
    return args

def get_image_id_from_filename(coco, name_contains="S01-activity01-51470934"):
    '''
    Get all image id from coco object that filename contains the "name_contains"
    '''
    img_ids = []
    for img in coco.dataset['images']:
        if name_contains in img['file_name']:
            img_ids.append(img['id'])
    return img_ids

def coco_to_numpy(this_anns):
    '''
    Convert coco object to numpy array
    '''
    keypoints_list = []
    bbox_list = []
    bbox_diag_list = []
    for ann in this_anns:
        keypoints = np.array(ann['keypoints']).reshape(-1, 3)
        keypoints_list.append(keypoints)
        bbox_list.append(ann['bbox'])
        bbox_diag_list.append(np.sqrt(ann['bbox'][2]**2 + ann['bbox'][3]**2))  # for coco x_min, y_min, width, height

    return np.array(keypoints_list), np.array(bbox_list), np.array(bbox_diag_list)


def report_by_joint(MPE, keypoint_names, label="px"):
    if type(keypoint_names) == dict:
        keypoint_names = list(keypoint_names.values())
    data = {'Keypoint': keypoint_names, label: MPE}
    MPJPE = np.mean(MPE)

    df = pd.DataFrame(data)
    df.loc[len(df.index)] = ['Average', MPJPE]  # Add final row with MPJPE

    if "%" in label:
        df[label] = df[label].apply(lambda x: f"{x:.2%}")
    else:
        df[label] = df[label].apply(lambda x: f"{x:.2f}")
    print("#"*35)
    print(df.to_string(index=True))


def body_part_bbox_diag(pose):
    """
    Calculate the size of each body part bbox diag from 37kpts pose
    param pose: 37kpts pose, shape (n, 37, 3)
    """
    body_parts = {
        "head": [17, 19, 20, 21, 22],
        "torso": [0, 3, 4, 15, 16, 22, 23, 24, 25, 26, 27, 28],
        "left_arm": [1, 13, 15, 25, 26, 29, 30],
        "right_arm": [2, 14, 16, 27, 28, 31, 32],
        "left_hand": [1, 11, 33, 34],  # double hand size
        "right_hand": [2, 12, 35, 36],
        "left_leg": [3, 5, 7, 9],
        "right_leg": [4, 6, 8, 10],
    }

    sizes = {}
    for part, indices in body_parts.items():
        # Extract relevant keypoints for the body part
        keypoints = pose[:, indices, :]  # Shape: (n_frames, len(indices), 3)
        # Calculate min and max coordinates along each axis
        min_coords = np.min(keypoints, axis=1)  # Shape: (n_frames, 3)
        max_coords = np.max(keypoints, axis=1)  # Shape: (n_frames, 3)
        # Calculate diagonal using vectorized Euclidean distance
        diagonals = np.sqrt(np.sum((max_coords - min_coords) ** 2, axis=1))  # Shape: (n_frames,)
        # Store results
        if part == "left_hand" or part == "right_hand":
            diagonals *= 2
        sizes[part] = diagonals

    return sizes

def part_diag_by_joint(sizes):
    body_part_by_joint_idx = [
        "torso",  # 0: PELVIS
        "right_hand",  # 1: RWRIST
        "left_hand",  # 2: LWRIST
        "right_leg",  # 3: RHIP
        "left_leg",  # 4: LHIP
        "right_leg",  # 5: RKNEE
        "left_leg",  # 6: LKNEE
        "right_leg",  # 7: RANKLE
        "left_leg",  # 8: LANKLE
        "right_leg",  # 9: RFOOT
        "left_leg",  # 10: LFOOT
        "right_hand",  # 11: RHAND
        "left_hand",  # 12: LHAND
        "right_arm",  # 13: RELBOW
        "left_arm",  # 14: LELBOW
        "right_arm",  # 15: RSHOULDER
        "left_arm",  # 16: LSHOULDER
        "head",  # 17: HEAD
        "torso",  # 18: THORAX
        "head",  # 19: HDTP
        "head",  # 20: REAR
        "head",  # 21: LEAR
        "torso",  # 22: C7
        "torso",  # 23: C7_d
        "torso",  # 24: SS
        "right_arm",  # 25: RAP_b
        "right_arm",  # 26: RAP_f
        "left_arm",  # 27: LAP_b
        "left_arm",  # 28: LAP_f
        "right_arm",  # 29: RLE
        "right_arm",  # 30: RME
        "left_arm",  # 31: LLE
        "left_arm",  # 32: LME
        "right_hand",  # 33: RMCP2
        "right_hand",  # 34: RMCP5
        "left_hand",  # 35: LMCP2
        "left_hand"  # 36: LMCP5
    ]
    sizes_by_joint = []
    for idx, part in enumerate(body_part_by_joint_idx):
        sizes_by_joint.append(sizes[part])
    sizes_by_joint = np.array(sizes_by_joint)
    return sizes_by_joint.transpose()



if __name__ == '__main__':
    args = phrase_args()

    # read GT file
    coco = COCO(args.GT_ann_file)

    bbox_diags = []
    part_bbox_diags = []
    PE = []

    for root, dirs, files in os.walk(args.RTM_infer_folder):
        dirs.sort()  # Sort directories in-place --> important, will change walk sequence
        files.sort(key=str.lower)  # Sort files in-place
        for file in files:
            if not file.endswith('.json'):
                continue
            if file.startswith('.'):  # ignore hidden files
                continue
            print(f"Processing {file}") if args.verbose else None
            video_name = file.split('_')[1].split('.')[0]  # results_S01-activity00-51470934.json --> S01-activity00-51470934

            #### Get Inference Pose
            all_bboxes, all_kpts_bef, keypoint_names = load_json_arr(os.path.join(root, file))
            inference_pose = filter_subject_using_center_of_joints_with_disqualify(all_kpts_bef, img_shape=args.img_shape, window=4, num_keypoints=args.number_of_keypoints,
                                                                                   type="rtm37_from_37")
            if args.infer_pose_type == 'rtm37_from_coco133':
                inference_pose = coco133_to_VEHS7M_37(inference_pose)

            #### Get GT Pose
            # update keypoint names
            coco_keypoint_names = coco.cats[1]['keypoints']
            keypoint_names = {str(idx): name for idx, name in enumerate(coco_keypoint_names)}
            img_ids = get_image_id_from_filename(coco, video_name)
            ann_ids = coco.getAnnIds(imgIds=img_ids)  # todo: does this work in there are multiple instances in the image?
            assert img_ids==ann_ids, "img_ids and ann_ids are not the same, meaning there are multiple instances in the image, might cause bugs"
            this_anns = coco.loadAnns(ann_ids)
            gt_pose, gt_bbox, bbox_diag = coco_to_numpy(this_anns)
            part_bbox_diag = part_diag_by_joint(body_part_bbox_diag(gt_pose))
            diff = gt_pose.shape[0] - inference_pose.shape[0]
            print(f"Gt_shape: {gt_pose.shape}, Inference_shape: {inference_pose.shape}, Diff: {diff}") if args.verbose else None
            assert diff<10, "Fram difference between GT and inference > 10, check the file"
            if diff > 0:  # force to have the same number of frames
                gt_pose = gt_pose[:-diff]
                gt_bbox = gt_bbox[:-diff]
                bbox_diag = bbox_diag[:-diff]
                part_bbox_diag = part_bbox_diag[:-diff]
            elif diff < 0:
                inference_pose = inference_pose[:diff]

            # position error
            position_error = np.linalg.norm(inference_pose[:, :, :2] - gt_pose[:, :, :2], axis=2)
            PE.append(position_error)
            bbox_diags.append(bbox_diag)
            part_bbox_diags.append(part_bbox_diag)


    PE = np.concatenate(PE, axis=0)
    bbox_diags = np.concatenate(bbox_diags, axis=0)
    part_bbox_diags = np.concatenate(part_bbox_diags, axis=0)

    # MPJPE
    MPE = np.mean(PE, axis=0)
    report_by_joint(MPE, keypoint_names, label="MPJPE_(px)")

    if args.norm_mode == 'bbox_diag':
        # MPJPE_norm
        PE_norm = PE / bbox_diags[:, np.newaxis]
        MPE_norm = np.mean(PE_norm, axis=0)
    elif args.norm_mode == 'part_diag':
        # MPJPE_norm by body part
        PE_norm = PE / part_bbox_diags
        MPE_norm = np.mean(PE_norm, axis=0)

    report_by_joint(MPE_norm, keypoint_names, label=f"MPJPE_norm_({args.norm_mode}%)")

    # PCK
    for threshold in [0.2, 0.05]:
        correct_keypoints = (PE_norm < threshold)
        PCK = np.mean(correct_keypoints, axis=0)
        report_by_joint(PCK, keypoint_names, label=f"PCK@{threshold}_(%)")

        PE_outlier = PE * ~correct_keypoints
        PE_norm_outlier = PE_norm * ~correct_keypoints
        if args.verbose:
            print(f"Outlier mean: {non_zero_mean(PE_outlier):.2f} px, {non_zero_mean(PE_norm_outlier):.2%}")
            print(f"Outlier median: {non_zero_median(PE_outlier):.2f} px, {non_zero_median(PE_norm_outlier):.2%}")
            print(f"Outlier max: {np.max(PE_outlier):.2f} px, {np.max(PE_norm_outlier):.2%}")
            # Get the flat index of the maximum value
            flat_index = np.argmax(PE_outlier)
            # Convert the flat index to a tuple of indices (frame, joint)
            frame_index, joint_index = np.unravel_index(flat_index, PE_outlier.shape)
            print(f"Frame index: {frame_index}, Joint index: {joint_index}, Joint name: {keypoint_names[str(joint_index)]}")
            coco.imgs[list(coco.imgs.keys())[frame_index]]  # {'file_name': 'S05-activity06-66920758-000818.jpg', 'height': 1200, 'width': 1920, 'id': 164523, 'license': 99}
            coco.anns[list(coco.anns.keys())[frame_index]]['keypoints'][joint_index*3:joint_index*3+3]







