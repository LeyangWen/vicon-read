# for creating MotionBert input file from RTMPose inference json files (e.g., for industry vids)
# Remember to rearrange file names first, see '/Volumes/Z/RTMPose/rename_rick.sh'


import argparse
import pickle
import os

import numpy as np

import rtm_pose_read_kps
import ref
from utility import *

def read_input(json_path, type='rtm24'):
    """
    Read the input from the json file
    :param json_path: path to the json file
    :return: all_keyps_rtm: (n,24,3), x, y, confidence
    """
    if type == 'rtm24':
        all_bboxes,all_keyps_bef = rtm_pose_read_kps.load_json_arr(json_path) # Load the json file and read the keypoints and bounding boxes
        all_keyps_rtm = rtm_pose_read_kps.filter_subject_using_center_of_joints_with_disqualify(all_keyps_bef,window=4) # Filter subject using the center of the joints
        angle_joint_order = [ref.rtm_pose_keypoints.index(joint) for joint in ref.rtm_pose_keypoints_vicon_dataset] # Get the order of the joints in the Vicon dataset
        all_keyps_rtm = all_keyps_rtm[:,angle_joint_order]
        all_keyps_rtm[:, ref.rtm_pose_keypoints_vicon_dataset.index('left_middle_mcp')] = (all_keyps_rtm[:,ref.rtm_pose_keypoints_vicon_dataset.index('left_pinky')]+all_keyps_rtm[:,ref.rtm_pose_keypoints_vicon_dataset.index('left_index')])/2
        all_keyps_rtm[:, ref.rtm_pose_keypoints_vicon_dataset.index('right_middle_mcp')] = (all_keyps_rtm[:,ref.rtm_pose_keypoints_vicon_dataset.index('right_pinky')]+all_keyps_rtm[:,ref.rtm_pose_keypoints_vicon_dataset.index('right_index')])/2
    elif type == 'rtm37_from_37':
        all_bboxes, all_keyps_bef, keypoint_names = rtm_pose_read_kps.load_json_arr(json_path)  # Load the json file and read the keypoints and bounding boxes
        all_keyps_rtm = rtm_pose_read_kps.filter_subject_using_center_of_joints_with_disqualify(all_keyps_bef, window=4, type=type ,num_keypoints = 37,)  # Filter subject using the center of the joints

    print('RTMPOse keypoints shape:',all_keyps_rtm.shape)
    return all_keyps_rtm


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--json_folder', type=str, default=r'W:\VEHS\Testing_Videos_and_rtmpose_results\OneDrive_2_9-4-2024\kps_133_fps_20')
    parser.add_argument('--json_folder', type=str, default=r'/Volumes/Z/RTMPose/37kpts_rtmw_v5/20fps/2D/Industry_3')
    parser.add_argument('--read_type', type=str, default='npy', help='json or npy')
    parser.add_argument('--output_file', type=str, default=r'rtmpose_v5-2b_20fps_industry_37kpts_v2.pkl')
    parser.add_argument('--joint_num', type=int, default=37)
    parser.add_argument('--type', type=str, default='rtm37_from_37')
    parser.add_argument('--score_2d', type=str, default='confidence', help='confidence or visibility')

    args = parser.parse_args()
    # args.output_file = args.output_file.replace('.pkl', f'_{args.score_2d}.pkl')
    return args

if __name__ == '__main__':
    args = parse_args()
    output_MB_dataset = empty_MotionBert_dataset_dict(args.joint_num)
    cumulative_segments = [0,]
    seg_length = []
    name_list = []
    for root, dirs, files in os.walk(args.json_folder):
        dirs.sort()  # Sort directories in-place
        files.sort(key=str.lower)  # Sort files in-place
        for file in files:
            if file.startswith('.'):
                continue
            json_path = os.path.join(root, file)
            source = json_path.split('/')[-1].split('.')[0]
            if file.endswith('.json') and args.read_type == 'json':
                print("#"*50)
                print(f"file: {file}, root: {root}")
                # read from json file
                all_keyps_rtm = read_input(json_path, type=args.type)
            elif file.endswith('.npy') and args.read_type == 'npy':
                print("#"*50)
                print(f"file: {file}, root: {root}")
                with open(os.path.join(root, file), 'rb') as f:
                    all_keyps_rtm = np.load(f)
            else:
                print(f"Skipping file: {file}, not a valid {args.read_type} file")
                continue

            assert all_keyps_rtm.shape[1] == args.joint_num, f"all_keyps_rtm.shape[1]: {all_keyps_rtm.shape[1]}, args.joint_num: {args.joint_num}, they should be the same"
            frame_no = all_keyps_rtm.shape[0]
            if args.score_2d == 'visibility':
                assert all_keyps_rtm.shape[2] == 4, f"all_keyps_rtm.shape[2]: {all_keyps_rtm.shape[2]}, should be 4 for x,y,conf,visibility"
                all_keyps_rtm[:, :, 2] = all_keyps_rtm[:, :, -1]
                all_keyps_rtm = all_keyps_rtm[:, :, :3]
            elif args.score_2d == 'confidence':
                all_keyps_rtm = all_keyps_rtm[:, :, :3]
            else:
                raise NotImplementedError


            print(f"frame_no: {frame_no}, %243: {frame_no%243}")
            this_segment = frame_no//243
            if this_segment ==0:
                # fill with last frame till 243
                pad_len = 243 - frame_no
                last_frame = all_keyps_rtm[-1:]
                pad_frames = np.repeat(last_frame, pad_len, axis=0)
                all_keyps_rtm = np.concatenate([all_keyps_rtm, pad_frames], axis=0)
                this_segment = 1
                source = source + '_padend'
            seg_length.append(this_segment)
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


    print(seg_length)
    cumulative_segments = np.array(cumulative_segments)
    cumulative_segment_frames = cumulative_segments*243
    cumulative_segment_frames = cumulative_segment_frames[:-1]
    for na, seg in zip(name_list, cumulative_segment_frames):
        print(f"{na}: {seg}")

#
# with open(json_path, 'r') as f:
#     json_data = json.load(f)











