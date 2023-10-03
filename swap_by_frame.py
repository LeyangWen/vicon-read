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
# help(vicon.SaveTrial)


if __name__ == '__main__':
    # read arguments

    vicon = ViconNexus.ViconNexus()
    trial_name = vicon.GetTrialName()
    subject_names = vicon.GetSubjectNames()
    frame_count = vicon.GetFrameCount()
    frame_rate = vicon.GetFrameRate()
    joint_names = vicon.GetMarkerNames(subject_names[0])

    hand = ['MCP5', 'MCP2', 'US', 'RS']
    arm = ['MCP5', 'MCP2', 'US', 'RS', 'ME', 'LE', 'AP', 'AP_f', 'AP_b']
    leg = ['MFC', 'LFC', 'GT', 'IC']
    foot = ['MTP1', 'MTP5', 'MM', 'LM', 'HEEL']
    pelvis = ['ASIS', 'PSIS']

    frame = 14578
    target_markers = None

    joints_dict = {}
    for marker in target_markers:
        LR_markers = []
        for LR in ['L', 'R']:
            LR_marker = LR + marker
            joints_dict[LR_marker] = MarkerPoint(vicon.GetTrajectory(subject_names[0], LR_marker))
            LR_markers.append(LR_marker)
        L_marker, R_marker = LR_markers
        joints_dict[L_marker], joints_dict[R_marker] = Point.swap_trajectory(joints_dict[L_marker], joints_dict[R_marker], frame-1)

    # output back to Vicon
    for joint_name, value in joints_dict.items():
        vicon.SetTrajectory(subject_names[0], joint_name, joints_dict[joint_name].x, joints_dict[joint_name].y, joints_dict[joint_name].z, joints_dict[joint_name].exist)
        print(f'{joint_name} trajectory updated')

    if False:
        target_markers = ['RUS', 'RRS', 'RMCP2']
        joints_dict = {}
        for marker in target_markers:
            joints_dict[marker] = MarkerPoint(vicon.GetTrajectory(subject_names[0], marker))

        RRS_RUS = Point.vector(joints_dict['RRS'], joints_dict['RUS'])
        RMCP5 = Point.translate_point(joints_dict['RMCP2'], RRS_RUS, direction=0.75)
        vicon.SetTrajectory(subject_names[0], 'RMCP5', RMCP5.x, RMCP5.y, RMCP5.z, RMCP5.exist)

    if False:
        target_markers = ['LMFC', 'LLFC']
        joints_dict = {}
        for marker in target_markers:
            joints_dict[marker] = MarkerPoint(vicon.GetTrajectory(subject_names[0], marker))

        # # store in pkl
        # with open('LMFC_LLFC.pkl', 'wb') as f:
        #     pickle.dump(joints_dict, f)

        frame = 16946
        # load from pkl
        with open('LMFC_LLFC.pkl', 'rb') as f:
            joints_dict_knee = pickle.load(f)
        joints_dict['LMFC'] = joints_dict_knee['LMFC']
        joints_dict['LLFC'].xyz[:,frame:] = joints_dict_knee['LLFC'].xyz[:,frame:]
        joints_dict['LLFC'].exist[frame:] = joints_dict_knee['LLFC'].exist[frame:]
        vicon.SetTrajectory(subject_names[0], 'LLFC', joints_dict['LLFC'].x, joints_dict['LLFC'].y, joints_dict['LLFC'].z, joints_dict['LLFC'].exist)
        vicon.SetTrajectory(subject_names[0], 'LMFC', joints_dict['LMFC'].x, joints_dict['LMFC'].y, joints_dict['LMFC'].z, joints_dict['LMFC'].exist)

    if False:
        from spacepy import pycdf
        dir = r'W:\VEHS\VEHS data collection round 3\processed\WenZhou\FullCollection\cdf_output\2D_Pose'

        # find all files in the directory and subdirectory
        cdf_files = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith('.cdf'):
                    # append full root path
                    cdf_files.append(os.path.join(root, file))


        # read cdf file
        for cdf_file in cdf_files:
            with open(cdf_file, 'rb') as f:
                pycdf_file = pycdf.CDF(cdf_file)

            data_dict = {}
            for key in pycdf_file.keys():
                data_dict[key] = pycdf_file[key]
            for attri in pycdf_file.attrs:
                data_dict[attri] = pycdf_file.attrs[attri]

            pose_2d = np.array(data_dict['Pose'])
            points_2d_bbox_list = []
            for frame_idx, points_2d in enumerate(pose_2d):
                bbox_top_left, bbox_bottom_right = points_2d.min(axis=0) - 20, points_2d.max(axis=0) + 20
                points_2d_bbox_list.append([bbox_top_left, bbox_bottom_right])

            points_2d_bbox_list = np.array(points_2d_bbox_list)
            # close the file
            pycdf_file.close()
            cdf_file.replace('2D_Pose', '2D_Pose_bbox')
            store_cdf(cdf_file, pose_2d, TaskID=data_dict['TaskID'], CamID=data_dict['CamID'], kp_names=data_dict['KeypointNames'], bbox=points_2d_bbox_list)
            print(f'{cdf_file} stored')

    if False:
        target_markers = ['LME', 'RME']
        joints_dict = {}
        for marker in target_markers:
            joints_dict[marker] = MarkerPoint(vicon.GetTrajectory(subject_names[0], marker))

        frame_1 = 2
        frame_2 = 19360
        frame_number = vicon.GetFrameCount()
        for i in range(frame_1):
            # joints_dict['LME'].xyz[:,i] = joints_dict['LME'].xyz[:,frame_1]
            # joints_dict['LME'].exist[i] = joints_dict['LME'].exist[frame_1]
            joints_dict['RME'].xyz[:,i] = joints_dict['RME'].xyz[:,frame_1]
            joints_dict['RME'].exist[i] = joints_dict['RME'].exist[frame_1]
        for j in range(frame_number-frame_2):
        #     # joints_dict['RME'].xyz[:,j+frame_2] = joints_dict['RME'].xyz[:,frame_2]
        #     # joints_dict['RME'].exist[j+frame_2] = joints_dict['RME'].exist[frame_2]
            joints_dict['LME'].xyz[:,j+frame_2] = joints_dict['LME'].xyz[:,frame_2]
            joints_dict['LME'].exist[j+frame_2] = joints_dict['LME'].exist[frame_2]
        vicon.SetTrajectory(subject_names[0], 'LME', joints_dict['LME'].x, joints_dict['LME'].y, joints_dict['LME'].z, joints_dict['LME'].exist)
        vicon.SetTrajectory(subject_names[0], 'RME', joints_dict['RME'].x, joints_dict['RME'].y, joints_dict['RME'].z, joints_dict['RME'].exist)


