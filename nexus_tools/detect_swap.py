from viconnexusapi import ViconNexus
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
import pickle

from utility import *
from ergo3d import *
import yaml
import datetime
import warnings
from utility import *
from Camera import *
from Skeleton import *
import argparse

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="invalid value encountered in arccos")

# helper functions
# vicon = ViconNexus.ViconNexus()
# dir(ViconNexus.ViconNexus)
# help(vicon.SaveTrial)


def get_left_right(pt_name):
    if pt_name[0] == 'R':
        left_right = 'Right'
    elif pt_name[0] == 'L':
        left_right = 'Left'
    else:
        left_right = 'General'
    return left_right


def trim_close_index(swap_index, threshold_frame=10, start_frame=0):
    last_recorded_index = -100
    for swap_id in swap_index:
        if abs(swap_id - last_recorded_index) < threshold_frame or swap_id < start_frame:
                swap_index = np.delete(swap_index, np.where(swap_index == swap_id))
        last_recorded_index = swap_id
    return swap_index



if __name__ == '__main__':
    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--swap', type=int, default=0, help='swap and fix marker pairs in list, mode 1: swap by comparing distance, mode 2: swap by checking speed')
    parser.add_argument('--swap_threshold', type=int, default=35, help='swap movement threshold for one frame in mm, not the main threshold')
    parser.add_argument('--mark', action='store_true', help='mark high speed markers with event')
    parser.add_argument('--mark_threshold', type=int, default=45, help='mark movement threshold for one frame in mm, smaller value is more sensitive')
    parser.add_argument('--mark_swap_threshold', type=int, default=25, help='mark swap movement threshold for one frame in mm, bigger value is more sensitive')
    parser.add_argument('--mark_frame_interval', type=int, default=1, help='mark frame interval')
    parser.add_argument('--clear', action='store_true', help='clear all events')
    parser.add_argument('--start_frame', type=int, default=0, help='dont swap before this frame')

    vicon = ViconNexus.ViconNexus()
    trial_name = vicon.GetTrialName()
    subject_names = vicon.GetSubjectNames()
    frame_count = vicon.GetFrameCount()
    frame_rate = vicon.GetFrameRate()
    joint_names = vicon.GetMarkerNames(subject_names[0])

    start_frame = parser.parse_args().start_frame
    swap_pairs = [
        # ['RRS', 'RUS'], ['LRS', 'LUS'],
        # ['RMCP2', 'RMCP5'], ['LMCP2', 'LMCP5'],
        ['RAP', 'RAP_f'],
        ['LAP', 'LAP_f'],
        ['RAP_f', 'RAP_b'],
        ['LAP_f', 'LAP_b'],
        ['RLE', 'RME'],
        ['LLE', 'LME'],
        ['LIC', 'LGT'], ['RIC', 'RGT'],
        ['LIC', 'LASIS'], ['RIC', 'RASIS'],
        ['SS', 'RAP_f'], ['SS', 'LAP_f'],
        ['SS', 'RAP'], ['SS', 'LAP'],
        ['LMFC', 'LLFC'], ['RMFC', 'RLFC'],
        ['LLM', 'LMM'], ['LMTP5', 'LMTP1'],
        ['RLM', 'RMM'], ['RMTP5', 'RMTP1'],
        ['LEAR', 'REAR'],  ['HDTP', 'MDFH'],
        ['LEAR', 'HDTP'], ['REAR', 'MDFH'],
        ['REAR', 'HDTP'], ['LEAR', 'MDFH'],
        ['C7_d', 'C7'], ['C7_d', 'SS'],
        ['RMFC', 'LMFC']
        # ['RMM', 'RLM']
    ]
    LR_pairs = [
        ['RRS', 'LRS'], ['RUS', 'LUS'],
        ['RMCP2', 'LMCP2'], ['RMCP5', 'LMCP5'],
        ['RAP', 'LAP'], ['RAP_f', 'LAP_f'], ['RAP_b', 'LAP_b'],
        ['RLE', 'LLE'], ['RME', 'LME'],
        ['RGT', 'LGT'], ['RIC', 'LIC'],
        ['RASIS', 'LASIS'], ['RPSIS', 'LPSIS'],
        ['LMFC', 'RMFC'], ['LLFC', 'RLFC'],
        ['LMM', 'RMM'], ['LMTP5', 'RMTP5'],
        ['LMTP1', 'RMTP1'],
        ['LEAR', 'REAR'],
    ]
    check_marker_pairs = swap_pairs
    name_tuple = ()
    for names in check_marker_pairs:
        for joint_name in names:
            name_tuple = name_tuple + (joint_name,)

    if parser.parse_args().clear:
        vicon.ClearAllEvents()

    if parser.parse_args().swap:
        joints_dict = {}
        # for joint_name in joint_names:
        for joint_name in name_tuple:
            joints_dict[joint_name] = MarkerPoint(vicon.GetTrajectory(subject_names[0], joint_name))
        for names in check_marker_pairs:
            for i in range(2):
                if i == 0:
                    pt1_name, pt2_name = names
                else:
                    pt2_name, pt1_name = names

                if parser.parse_args().swap == 1:
                    swap_index = Point.check_marker_swap(joints_dict[pt1_name], joints_dict[pt2_name], threshold=parser.parse_args().swap_threshold)
                elif parser.parse_args().swap == 2:
                    swap_index = Point.check_marker_swap_by_speed(joints_dict[pt1_name], joints_dict[pt2_name], threshold=parser.parse_args().mark_swap_threshold)
                else:
                    raise ValueError('swap mode not supported, need to be int 0, 1 or 2')

                swap_index = trim_close_index(swap_index, start_frame=start_frame)
                for swap_index_i in swap_index:
                    joints_dict[pt1_name], joints_dict[pt2_name] = Point.swap_trajectory(joints_dict[pt1_name], joints_dict[pt2_name], swap_index_i)
                if len(swap_index) != 0:
                    swap_index_after = Point.check_marker_swap(joints_dict[pt1_name], joints_dict[pt2_name], threshold=parser.parse_args().swap_threshold)
                    left_right = get_left_right(pt1_name)
                    for j in swap_index_after:
                        vicon.CreateAnEvent(subject_names[0],  left_right, 'General', int(j+1), 0)
                    print(f'{pt1_name} - {pt2_name}: {len(swap_index)} swaps detected, now {len(swap_index_after)} remaining')
                    print(f'------- swap index: {swap_index} -------')
                    print(f'------- remaining swap index: {swap_index_after} -------')
                    print()
        # output back to Vicon
        print("updating trajectories: ", end='r')
        for joint_name in name_tuple:
            vicon.SetTrajectory(subject_names[0], joint_name, joints_dict[joint_name].x, joints_dict[joint_name].y, joints_dict[joint_name].z, joints_dict[joint_name].exist)
            print(f'{joint_name}, ', end='r')

    if parser.parse_args().mark:
        check_marker_pairs_all = [
                                  ['RRS', 'RUS'], ['LRS', 'LUS'],
                                  ['RMCP2', 'RMCP5'], ['LMCP2', 'LMCP5'],
                                  ['RMCP2', 'RRS'], ['LMCP2', 'LRS'],
                                  ['RMCP5', 'RUS'], ['LMCP5', 'LUS'],
                                  ['RMCP5', 'RRS'], ['LMCP5', 'LRS'],
                                  ['RAP', 'RAP_f'],
                                  ['LAP', 'LAP_f'],
                                  ['RAP_f', 'RAP_b'], ['LAP_f', 'LAP_b'],

                                  ['RAP', 'RAP_b'], ['LAP', 'LAP_b'],
                                  ['RLE', 'RME'],
                                  ['LLE', 'LME'],
                                  ['LIC', 'LGT'], ['RIC', 'RGT'],
                                  ['LIC', 'LASIS'], ['RIC', 'RASIS'],
                                  ['LGT', 'LASIS'], ['RGT', 'RASIS'],
                                  ['SS', 'RAP_f'], ['SS', 'LAP_f'],
                                  ['SS', 'RAP'], ['SS', 'LAP'],
                                  ['LMFC', 'LLFC'], ['RMFC', 'RLFC'],
                                  ['LLM', 'LMM'], ['LMTP5', 'LMTP1'],
                                  ['RLM', 'RMM'], ['RMTP5', 'RMTP1'],
                                  ['LEAR', 'REAR'], ['HDTP', 'MDFH'],
                                  ['HDTP', 'REAR'], ['REAR', 'MDFH'],
                                  ['HDTP', 'LEAR'], ['LEAR', 'MDFH'],
                                  ['C7_d', 'C7'], ['C7_d', 'SS'],
                                  ['C7_d', 'RAP_b'], ['C7_d', 'LAP_b'], ['C7_d', 'RAP'], ['C7_d', 'LAP'], ['C7_d', 'RAP_f'], ['C7_d', 'LAP_f'],
                                  ['C7_d', 'RAP_b'], ['C7_d', 'LAP_b'], ['C7_d', 'RAP'], ['C7_d', 'LAP'], ['C7_d', 'RAP_f'], ['C7_d', 'LAP_f'],
                                  ['LMFC', 'LME'], ['RMFC', 'RME'],
                                  ['LMFC', 'LLE'], ['RMFC', 'RLE'],
                                  ['XP', 'LME'], ['XP', 'RME'],
                                  # ['RMFC', 'LMFC']
                                 ]
        check_marker_pairs = check_marker_pairs_all + swap_pairs
        joints_dict = {}
        finished = True
        count_speed_index = 0
        for joint_name in joint_names:
            joints_dict[joint_name] = MarkerPoint(vicon.GetTrajectory(subject_names[0], joint_name))
            speed_index = joints_dict[joint_name].check_marker_speed(threshold=parser.parse_args().mark_threshold)
                                                                     # interval_frames=parser.parse_args().mark_frame_interval)
            left_right = get_left_right(joint_name)
            store_j = -100
            count = 0
            count_speed_index = count_speed_index + len(speed_index)
            for frame_j in speed_index:
                if abs(frame_j - store_j) < 20:
                    continue
                store_j = frame_j
                vicon.CreateAnEvent(subject_names[0], left_right, 'Foot Off', int(frame_j + parser.parse_args().mark_frame_interval), 0)
                count += 1
                print(f'{joint_name} speed index: {frame_j}')
                if count > 10:
                    print(f'too many speed index for {joint_name}')
                    finished = False
                    break
        count_swap_index = 0
        for names in check_marker_pairs:
            for i in range(2):
                if i == 0:
                    pt1_name, pt2_name = names
                else:
                    pt2_name, pt1_name = names
                speed_swap_index = Point.check_marker_swap_by_speed(joints_dict[pt1_name], joints_dict[pt2_name],
                                                                    threshold=parser.parse_args().mark_swap_threshold,
                                                                    interval_frames=parser.parse_args().mark_frame_interval)
                left_right = get_left_right(pt1_name)
                store_j = -100
                count = 0
                count_swap_index = count_swap_index + len(speed_swap_index)
                for frame_j in speed_swap_index:
                    if abs(frame_j - store_j) < 20:
                        continue
                    store_j = frame_j
                    vicon.CreateAnEvent(subject_names[0], left_right, 'General', int(frame_j + parser.parse_args().mark_frame_interval), 0)
                    count += 1
                    print(f'{pt1_name} - {pt2_name} speed swap index: {frame_j}')
                    if count > 10:
                        print(f'too many pair index for {pt1_name}')
                        finished = False
                        break
        print(f'------- counting errors -------')
        print(f'Total: {count_speed_index+count_swap_index} - {count_speed_index} speed index, {count_swap_index} swap index')

        if finished:
            vicon.CreateAnEvent(subject_names[0], 'Right', 'General', frame_count, 0)
            vicon.CreateAnEvent(subject_names[0], 'Left', 'General', frame_count, 0)
            vicon.CreateAnEvent(subject_names[0], 'General', 'General', frame_count, 0)

    # print parser arguments
    print('------- parser arguments -------')
    for arg in vars(parser.parse_args()):
        print(f'{arg}: {getattr(parser.parse_args(), arg)}')
    print('------- ending -------')