# for creating RTMPose inference file from json files

import argparse
import pickle
import os

import numpy as np

import rtm_pose_read_kps
import ref
from utility import *

def read_input(json_path):
    """
    Read the input from the json file
    :param json_path: path to the json file
    :return: all_keyps_rtm: (n,24,3), x, y, confidence
    """
    all_bboxes,all_keyps_bef = rtm_pose_read_kps.load_json_arr(json_path) # Load the json file and read the keypoints and bounding boxes
    all_keyps_rtm = rtm_pose_read_kps.filter_subject_using_center_of_joints_with_disqualify(all_keyps_bef,window=4) # Filter subject using the center of the joints
    angle_joint_order = [ref.rtm_pose_keypoints.index(joint) for joint in ref.rtm_pose_keypoints_vicon_dataset] # Get the order of the joints in the Vicon dataset
    all_keyps_rtm = all_keyps_rtm[:,angle_joint_order]
    all_keyps_rtm[:, ref.rtm_pose_keypoints_vicon_dataset.index('left_middle_mcp')] = (all_keyps_rtm[:,ref.rtm_pose_keypoints_vicon_dataset.index('left_pinky')]+all_keyps_rtm[:,ref.rtm_pose_keypoints_vicon_dataset.index('left_index')])/2
    all_keyps_rtm[:, ref.rtm_pose_keypoints_vicon_dataset.index('right_middle_mcp')] = (all_keyps_rtm[:,ref.rtm_pose_keypoints_vicon_dataset.index('right_pinky')]+all_keyps_rtm[:,ref.rtm_pose_keypoints_vicon_dataset.index('right_index')])/2
    print('RTMPOse keypoints shape:',all_keyps_rtm.shape)
    return all_keyps_rtm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_folder', type=str, default=r'W:\VEHS\Testing_Videos_and_rtmpose_results\OneDrive_2_9-4-2024\kps_133_fps_20')
    parser.add_argument('--output_file', type=str, default=r'rtmpose_industry_3_no3d_j24_f20_s1_RTM2D.pkl')
    parser.add_argument('--joint_num', type=int, default=24)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    output_MB_dataset = empty_MotionBert_dataset_dict(args.joint_num)
    cumulative_segments = [0,]
    name_list = []
    for root, dirs, files in os.walk(args.json_folder):
        dirs.sort()  # Sort directories in-place
        files.sort(key=str.lower)  # Sort files in-place
        for file in files:
            if not file.endswith('.json'):
                continue
            print("#"*50)
            json_path = os.path.join(root, file)
            print(f"file: {file}, root: {root}")

            # read from json file
            source = json_path.split('\\')[-1].split('.')[0]
            all_keyps_rtm = read_input(json_path)
            assert all_keyps_rtm.shape[1] == args.joint_num, f"all_keyps_rtm.shape[1]: {all_keyps_rtm.shape[1]}, args.joint_num: {args.joint_num}, they should be the same"
            frame_no = all_keyps_rtm.shape[0]
            print(f"frame_no: {frame_no}, %243: {frame_no%243}")
            this_segment = frame_no//243
            cum_segment = this_segment+cumulative_segments[-1]
            cumulative_segments.append(cum_segment)
            name_list.append(source)
            # write in MB format
            output = {}
            # Real data
            output['joint_2d'] = all_keyps_rtm[:, :, :2]
            output['confidence'] = all_keyps_rtm[:, :, 2:]
            # Fake placeholder data
            output['joint3d_image'] = np.ones_like(all_keyps_rtm)
            output['camera_name'] = np.ones(len(all_keyps_rtm))
            output['source'] = [source] * len(all_keyps_rtm)
            output['2.5d_factor'] = np.ones(len(all_keyps_rtm))
            output['joints_2.5d_image'] = np.ones_like(all_keyps_rtm)
            output['action'] = ['none'] * len(all_keyps_rtm)
            output['joint_3d_camera'] = np.ones_like(all_keyps_rtm)
            output['c3d_frame'] = ['none'] * len(all_keyps_rtm)

            output_MB_dataset = append_output_xD_dataset(output_MB_dataset, 'validate', output)
    output_filename = os.path.join(args.json_folder, args.output_file)
    with open(f'{output_filename}', 'wb') as f:
        pickle.dump(output_MB_dataset, f)
    print(f"output_filename: {output_filename}")

    cumulative_segments = np.array(cumulative_segments)
    cumulative_segments = cumulative_segments*243
    cumulative_segments = cumulative_segments[:-1]
    for na, seg in zip(name_list, cumulative_segments):
        print(f"{na}: {seg*5}")

#
# with open(json_path, 'r') as f:
#     json_data = json.load(f)











