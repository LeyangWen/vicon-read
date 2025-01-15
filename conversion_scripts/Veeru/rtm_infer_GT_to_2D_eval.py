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
    parser.add_argument('--RTM_infer_folder', type=str, default='/Volumes/Z/RTMPose/37kpts_v1/last_epoch_270/Inference_results')
    parser.add_argument('--GT_ann_file', type=str, default='/Volumes/Z/RTMPose/37kpts_v1/GT/VEHS_6DCOCO_downsample20_keep1_validate.json')
    parser.add_argument('--number_of_keypoints', type=int, default=37)
    parser.add_argument('--img_shape', type=tuple, default=(1200,1920), help='height, width in px')
    parser.add_argument('--verbose', type=bool, default=False)

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


if __name__ == '__main__':
    args = phrase_args()

    # read GT file
    coco = COCO(args.GT_ann_file)

    scales = []
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
            inference_pose = filter_subject_using_center_of_joints_with_disqualify(all_kpts_bef, img_shape=args.img_shape, window=4, num_keypoints=args.number_of_keypoints, type='rtm37_from_37')

            #### Get GT Pose
            img_ids = get_image_id_from_filename(coco, video_name)
            ann_ids = coco.getAnnIds(imgIds=img_ids)  # todo: does this work in there are multiple instances in the image?
            assert img_ids==ann_ids, "img_ids and ann_ids are not the same, meaning there are multiple instances in the image, might cause bugs"
            this_anns = coco.loadAnns(ann_ids)
            gt_pose, gt_bbox, scale = coco_to_numpy(this_anns)

            diff = gt_pose.shape[0] - inference_pose.shape[0]
            print(f"Gt_shape: {gt_pose.shape}, Inference_shape: {inference_pose.shape}, Diff: {diff}") if args.verbose else None
            assert diff<10, "Fram difference between GT and inference > 10, check the file"
            if diff > 0:  # force to have the same number of frames
                gt_pose = gt_pose[:-diff]
                gt_bbox = gt_bbox[:-diff]
                scale = scale[:-diff]
            elif diff < 0:
                inference_pose = inference_pose[:diff]

            # position error
            position_error = np.linalg.norm(inference_pose[:, :, :2] - gt_pose[:, :, :2], axis=2)
            PE.append(position_error)
            scales.append(scale)

    PE = np.concatenate(PE, axis=0)
    scales = np.concatenate(scales, axis=0)

    # MPJPE
    MPE = np.mean(PE, axis=0)
    report_by_joint(MPE, keypoint_names, label="MPJPE_(px)")

    # MPJPE_norm
    PE_norm = PE / scales[:, np.newaxis]
    MPE_norm = np.mean(PE_norm, axis=0)
    report_by_joint(MPE_norm, keypoint_names, label="MPJPE_norm_(bbox_diag%)")

    # PCK
    for threshold in [0.05, 0.01]:
        correct_keypoints = (PE_norm < threshold)
        PCK = np.mean(correct_keypoints, axis=0)
        report_by_joint(PCK, keypoint_names, label=f"PCK@{threshold}_(%)")

        PE_outlier = PE * ~correct_keypoints
        PE_norm_outlier = PE_norm * ~correct_keypoints

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







