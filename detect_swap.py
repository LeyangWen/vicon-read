from viconnexusapi import ViconNexus
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
import pickle

import Point
from utility import *
from Point import *
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
# help(vicon.ClearAllEvents)


def get_left_right(pt_name):
    if pt_name[0] == 'R':
        left_right = 'Right'
    elif pt_name[0] == 'L':
        left_right = 'Left'
    else:
        left_right = 'General'
    return left_right

if __name__ == '__main__':
    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--swap', type=bool, default=False, help='swap and fix marker pairs in list')
    parser.add_argument('--swap_threshold', type=int, default=35, help='swap movement threshold for one frame in mm')
    parser.add_argument('--mark', type=bool, default=False, help='mark high speed markers with event')
    parser.add_argument('--mark_threshold', type=int, default=35, help='mark movement threshold for one frame in mm')
    parser.add_argument('--mark_swap_threshold', type=int, default=35, help='mark swap movement threshold for one frame in mm')
    parser.add_argument('--clear', type=bool, default=True, help='clear all events')

    vicon = ViconNexus.ViconNexus()
    trial_name = vicon.GetTrialName()
    subject_names = vicon.GetSubjectNames()
    frame_count = vicon.GetFrameCount()
    frame_rate = vicon.GetFrameRate()
    joint_names = vicon.GetMarkerNames(subject_names[0])

    check_marker_pairs = [
        ['RRS', 'RUS'], ['RMCP2', 'RMCP5'],
        ['LRS', 'LUS'], ['LMCP2', 'LMCP5'],
        ['RAP', 'RAP_f'],
        ['LAP', 'LAP_f'],
        ['RAP_f', 'RAP_b'],
        ['LAP_f', 'LAP_b'],
        ['RLE', 'RME'],
        ['LLE', 'LME'],
        ['LIC', 'LGT'], ['RIC', 'RGT'],
        ['LIC', 'LASIS'], ['RIC', 'RASIS'],

        ['LMFC', 'LLFC'], ['RMFC', 'RLFC'],
        ['LLE', 'LMM'], ['LMTP5', 'LMTP1'],
        ['RLE', 'RMM'], ['RMTP5', 'RMTP1'],
        ['LEAR', 'REAR'],
        ['HDTP', 'MDFH'],
        # ['LEAR', 'HDTP'], ['REAR', 'HDTP'],
    ]
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
                swap_index = Point.check_marker_swap(joints_dict[pt1_name], joints_dict[pt2_name], threshold=parser.parse_args().swap_threshold)
                for i in swap_index:
                    joints_dict[pt1_name], joints_dict[pt2_name] = Point.swap_trajectory(joints_dict[pt1_name], joints_dict[pt2_name], i)
                swap_index_after = Point.check_marker_swap(joints_dict[pt1_name], joints_dict[pt2_name])
                left_right = get_left_right(pt1_name)
                for j in swap_index_after:
                    vicon.CreateAnEvent(subject_names[0],  left_right, 'General', int(j+1), 0)
                if len(swap_index) != 0:
                    print(f'{pt1_name} - {pt2_name}: {len(swap_index)} swaps detected, now {len(swap_index_after)} remaining')
                    print(f'------- swap index: {swap_index} -------')
                    print(f'------- remaining swap index: {swap_index_after} -------')
                    print()
        # output back to Vicon
        for joint_name in name_tuple:
            vicon.SetTrajectory(subject_names[0], joint_name, joints_dict[joint_name].x, joints_dict[joint_name].y, joints_dict[joint_name].z, joints_dict[joint_name].exist)
            print(f'{joint_name} trajectory updated')

    if parser.parse_args().mark:
        joints_dict = {}
        for joint_name in joint_names:
            joints_dict[joint_name] = MarkerPoint(vicon.GetTrajectory(subject_names[0], joint_name))
            speed_index = joints_dict[joint_name].check_marker_speed(threshold=parser.parse_args().mark_threshold)
            left_right = get_left_right(joint_name)
            store_j = -100
            for j in speed_index:
                if abs(j - store_j) < 10:
                    store_j = j
                    continue
                vicon.CreateAnEvent(subject_names[0], left_right, 'Foot Off', int(j + 1), 0)
                store_j = j
                if j > 50:
                    print(f'too many speed index for {joint_name}')
                    break
        for names in check_marker_pairs:
            for i in range(2):
                if i == 0:
                    pt1_name, pt2_name = names
                else:
                    pt2_name, pt1_name = names
                speed_swap_index = Point.check_marker_swap_by_speed(joints_dict[pt1_name], joints_dict[pt2_name], threshold=parser.parse_args().mark_swap_threshold)
                left_right = get_left_right(pt1_name)
                store_j = -100
                for j in speed_index:
                    if abs(j - store_j) < 10:
                        store_j = j
                        continue
                    vicon.CreateAnEvent(subject_names[0], left_right, 'Foot Off', int(j + 1), 0)
                    store_j = j
                    if j > 50:
                        print(f'too many speed index for {joint_name}')
                        break

