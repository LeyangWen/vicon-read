# for creating RTMPose inference file from json files

import argparse
import pickle
import os

import numpy as np

import rtm_pose_read_kps
import ref
from utility import *

def read_input(json_path, type):
    """
    Read the input from the json file
    :param json_path: path to the json file
    :return: all_keyps_rtm: (n,24,3), x, y, confidence
    """
    all_bboxes,all_keyps_bef = rtm_pose_read_kps.load_json_arr_vid(json_path) # Load the json file and read the keypoints and bounding boxes
    all_keyps_rtm = rtm_pose_read_kps.filter_subject_using_center_of_joints_with_disqualify(all_keyps_bef,window=4, type=type) # Filter subject using the center of the joints
    print('RTMPOse keypoints shape:',all_keyps_rtm.shape)
    return all_keyps_rtm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_folder', type=str, default=r'/Users/leyangwen/Documents/Hand/barehand/') #/Volumes/Extreme SSD/Gloved Hands RTMPose/')  #'/Volumes/Extreme SSD/Gloved Hands/John/Rick/Sitting/Table - Upper/predictions/predictions/1_short.json'
    parser.add_argument('--output_file', type=str, default=r'lab_rtmpose_hand_for_MB.pkl')
    parser.add_argument('--joint_num', type=int, default=21)
    parser.add_argument('--scale', type=bool, default=True)  # goal: img_px * scale --> around 900 px canvas size*0.9

    parser.add_argument('--canvas_size', type=int, default=1000)  # only if scale is True
    parser.add_argument('--rootIdx', type=int, default=0)  # wrist

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    output_MB_dataset = empty_MotionBert_dataset_dict(args.joint_num)
    cumulative_segments = {'L':[0,], 'R':[0,]}
    name_list = {'L':[], 'R':[]}
    for root, dirs, files in os.walk(args.json_folder):
        dirs.sort()  # Sort directories in-place
        files.sort(key=str.lower)  # Sort files in-place
        for file in files:
            if not file.endswith('.json'):
                continue
            json_path = os.path.join(root, file)
            # read from json file
            source = json_path.split('\\')[-1].split('.')[0]

            for LR in ['R', 'L']:
                type = f"{LR}hand{args.joint_num}"
                try:
                    all_keyps_rtm = read_input(json_path, type)
                except:
                    continue

                assert all_keyps_rtm.shape[1] == args.joint_num, f"all_keyps_rtm.shape[1]: {all_keyps_rtm.shape[1]}, args.joint_num: {args.joint_num}, they should be the same"

                frame_no = all_keyps_rtm.shape[0]
                if frame_no < 243:
                    continue
                print("#"*50)
                print(f"file: {file}, root: {root}")

                this_segment = frame_no//243
                cum_segment = this_segment+cumulative_segments[LR][-1]
                cumulative_segments[LR].append(cum_segment)
                name_list[LR].append(source)
                # write in MB format
                output = {}
                confidence = all_keyps_rtm[:, :, 2:]
                joint_2d = all_keyps_rtm[:, :, :2]

                ## flip left hand to appear as right hand
                if LR == 'L':
                    joint_2d[:, :, 0] = -joint_2d[:, :, 0]

                ## scale hand to ~900px size to fit training data
                if args.scale:
                    scales = []
                    for test_frame in range(joint_2d.shape[0]):
                        hand_px_size = np.max(np.max(joint_2d[test_frame, :, :], axis=0) - np.min(joint_2d[test_frame, :, :], axis=0))
                        scales.append(args.canvas_size/2*0.9/hand_px_size)
                    scales = np.array(scales)
                    vid_scale = np.max(scales)  # goal: hand img_px * scale --> around 900 px
                    joint_2d = joint_2d * vid_scale
                    print(f"frame_no: {frame_no}, %243: {frame_no%243}, vid_scale: {vid_scale}")
                else:
                    print(f"frame_no: {frame_no}, %243: {frame_no % 243}")

                ## center around wrist
                joint_2d_centered = joint_2d - joint_2d[:, [args.rootIdx], :]
                joint_2d_on_canvas = joint_2d_centered + args.canvas_size / 2

                # Real data
                output['joint_2d'] = joint_2d_on_canvas
                output['confidence'] = confidence
                # Fake placeholder data
                output['joint3d_image'] = np.zeros_like(all_keyps_rtm)
                output['joint3d_image'][:, :, :2] = joint_2d_on_canvas
                output['camera_name'] = np.ones(len(all_keyps_rtm))
                output['source'] = [source] * len(all_keyps_rtm)
                output['2.5d_factor'] = np.ones(len(all_keyps_rtm))* vid_scale
                output['joints_2.5d_image'] = np.ones_like(all_keyps_rtm)
                output['action'] = ['none'] * len(all_keyps_rtm)
                output['joint_3d_camera'] = np.ones_like(all_keyps_rtm)
                output['c3d_frame'] = ['none'] * len(all_keyps_rtm)

                print("-" * 50)

                key_map = {'L': "validate", 'R': "test"}
                output_MB_dataset = append_output_xD_dataset(output_MB_dataset, key_map[LR], output)

    output_MB_dataset = append_output_xD_dataset(output_MB_dataset, "train", output)  # just so it is not empty
    output_filename = os.path.join(args.json_folder, args.output_file)
    with open(f'{output_filename}', 'wb') as f:
        pickle.dump(output_MB_dataset, f)

    for key, value in cumulative_segments.items():
        value = np.array(value)
        value = value*243
        value = value[:-1]
        for na, seg in zip(name_list[key], value):
            print(f"{na}: {seg}")
        break

    print(key_map)
    print(f"output_filename: {output_filename}")

#
# with open(json_path, 'r') as f:
#     json_data = json.load(f)











