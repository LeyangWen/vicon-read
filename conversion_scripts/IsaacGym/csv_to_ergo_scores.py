import argparse
import os.path
import pickle

import numpy as np

from Skeleton import *
import matplotlib
from ergo3d import Point
matplotlib.use('Qt5Agg')
import csv


####################################################
# From isaac inference output, csv file
# Output NIOSH lifting index, REBA score, and output 3DSSPP batch txt file

####################################################


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=r'config/experiment_config/IssacGym/test.yaml')
    parser.add_argument('--skeleton_file', type=str, default=r'config/VEHS_ErgoSkeleton_info/IssacGym/15kpts-Skeleton.yaml')


    parser.add_argument('--output_frame_folder', type=str, default=None)
    parser.add_argument('--output_GT_frame_folder', type=str, default=None)
    parser.add_argument('--plot_mode', type=str, default='paper_view', help='mode: camera_view, camera_side_view, 0_135_view, normal_view, paper_view, global_view')
    parser.add_argument('--MB_data_stride', type=int, default=243)
    parser.add_argument('--debug_mode', default=True, type=bool)


    # parser.add_argument('--name_list', type=list, default=[])
    args = parser.parse_args()
    with open(args.config_file, 'r') as stream:
        data = yaml.safe_load(stream)
        args.name_list = data['name_list']
        args.estimate_file = data['estimate_file']
        args.fps = data['fps']
        args.dim = data['dim']
        args.density = data['density']
    print(args.plot_mode)
    print(args.estimate_file)
    args.mass = args.density * args.dim[0] * args.dim[1] * args.dim[2]  # kg
    args.output_frame_folder = os.path.dirname(args.estimate_file) if args.output_frame_folder is None else args.output_frame_folder
    args.output_frame_folder = os.path.join(args.output_frame_folder, args.plot_mode)
    return args


def extract_movement_segments(box_moved: np.ndarray,
                              min_segment_length: int = 15) -> np.ndarray:
    """
    Given a boolean mask `box_moved` of shape (frames,),
    return a 2×n array of [start_frame, stop_frame] for each movement segment
    with length >= min_segment_length.
    """
    # find rising (+1) and falling (–1) edges
    edges = np.diff(box_moved.astype(int))
    starts = np.where(edges ==  1)[0] + 1
    stops  = np.where(edges == -1)[0] + 1

    # handle case where movement is active at t=0 or t=frames-1
    if box_moved[0]:
        starts = np.r_[0, starts]
    if box_moved[-1]:
        stops = np.r_[stops, box_moved.size]
    assert len(starts) == len(stops), "Unequal number of starts and stops"

    # stack into 2×n and filter by minimum length
    segments = np.vstack([starts, stops])  # shape (2, n_total)
    lengths = segments[1] - segments[0]
    valid   = lengths >= min_segment_length
    return segments[:, valid]

if __name__ == '__main__':
    # read arguments
    args = parse_args()
    # read est file csv as a numpy array
    n = 18
    isaac_pose_all = np.loadtxt(args.estimate_file, delimiter=',', usecols=range(13 * n)) # (frame, 234)
    isaac_pose_all = isaac_pose_all.reshape(isaac_pose_all.shape[0], -1, 13)  # shape: (num_frames, 18, 13) 3 pos, 4 rot, 3 linear vel, 3 angular vel
    isaac_pose = isaac_pose_all[:, :n, 0:3]  # shape: (num_frames, 15 + extra, 3)
    issac_rot = isaac_pose_all[:, :n, 3:7]  # shape: (num_frames, 15 + extra, 4)  # quaternion

    ### 3D pose
    isaac_skeleton = IsaacSkeleton(args.skeleton_file)
    isaac_skeleton.load_name_list_and_np_points(args.name_list, isaac_pose)
    isaac_skeleton.load_rot_quat(issac_rot)
    frame_no = isaac_pose.shape[0]


    ######################## NIOSH lifting equation ########################

    # Step 1: detect key events in frame no: each lift start, lift end, lower start, lower end

    #################
    box_id = 15
    box_move_threshold = 0.01
    #################
    box_diff = isaac_pose[1:, box_id, :] - isaac_pose[:-1, box_id, :]
    box_moved = np.zeros(frame_no, dtype=bool)
    box_moved[1:] = np.sum(box_diff ** 2, axis=1) > box_move_threshold ** 2

    box_movement_segments = extract_movement_segments(box_moved, min_segment_length=15)  # shape: (2, no) [lift_start, lower_end]

    #################
    box_init_id = 16
    box_dest_id = 17
    box_near_xy_threshold = 0.4
    # box_near_z_threshold = 0.05
    #################
    box_init = isaac_pose[:, box_init_id]
    box_dest = isaac_pose[:, box_dest_id]
    box = isaac_pose[:, box_id]

    box2init_xy_diff = box[:,:2] - box_init[:,:2]
    box2dest_xy_diff = box[:,:2] - box_dest[:,:2]
    box2init_xy_near = np.sum(box2init_xy_diff ** 2, axis=1) < box_near_xy_threshold ** 2
    box2dest_xy_near = np.sum(box2dest_xy_diff ** 2, axis=1) < box_near_xy_threshold ** 2

    # far from box_init and box_dest
    box2_init_dest_xy_far = np.logical_not(box2init_xy_near) & np.logical_not(box2dest_xy_near)
    carry_height = np.median(box[box2_init_dest_xy_far, 2])

    # box2carry_z_diff = box[:, 2] - carry_height
    # box2carry_z_near = np.abs(box2carry_z_diff) < box_near_z_threshold

    box2init_xy_near_segments = extract_movement_segments(box2_init_dest_xy_far, min_segment_length=5)  # shape: (2, no) [lift_end, lower_start]
    # box2dest_xy_near_segments = extract_movement_segments(box2dest_xy_near, min_segment_length=5)

    ################# output #################
    lift_start, lower_end = box_movement_segments
    lift_end, lower_start = box2init_xy_near_segments

    lower_start = lower_start + 2

    assert len(lift_start) == len(lift_end), "Error: lift start and lower end not equal"
    assert len(lower_start) == len(lower_end), "Error: lift end and lower start not equal"
    for i in range(len(lift_start)):
        assert lift_start[i] < lift_end[i], "Error: lift start should be less than lift end"
    for i in range(len(lower_start)):
        assert lower_start[i] < lower_end[i], "Error: lower start should be less than lower end"

    low_id = np.concatenate((lift_start, lift_start, lower_end, lower_end))
    high_id = np.concatenate((lift_end, lift_end, lower_start, lower_start))
    all_id = np.concatenate((lift_start, lift_end, lower_start, lower_end))

    # Step 2: calculate NIOSH elements
    #################
    pelvis_id = 0
    foot_id = [11, 14]
    shoulder_id = [3, 6]
    #################
    foot_center = (isaac_pose[:, foot_id[0]] + isaac_pose[:, foot_id[1]]) / 2
    pelvis = isaac_pose[:, pelvis_id]

    # Horizontal location (box center to pelvis xy)
    H = (np.sum((box[all_id, :2] - foot_center[all_id, :2])**2, axis=1)**0.5)*100  # cm
    # Vertical location (box center to floor z)
    V = box[all_id, 2]*100  # cm
    # Vertical Travel Distance (D)
    D = (box[high_id, 2] - box[low_id, 2])*100  # cm shape: (2no,)
    # Asymmetric angle (A) in degrees
    PELVIS = isaac_skeleton.point_poses['PELVIS']
    LHIP = isaac_skeleton.point_poses['left_thigh']
    RHIP = isaac_skeleton.point_poses['right_thigh']
    sagittal_line = Point.orthogonal_vector(PELVIS, RHIP, LHIP)
    sagittal_line_xy = sagittal_line.xyz.T[all_id, :2]
    pelvis2box_xy = box[all_id, :2] - pelvis[all_id, :2]
    A = Point.angle(sagittal_line_xy.T, pelvis2box_xy.T)
    A = np.degrees(A)  # rad to degree

    # frequency
    total_time = (frame_no - 1) / args.fps /60 # s
    count = len(all_id)//2
    frequency = count / total_time  # Hz

    # Load Constant LC 23kg 51lb
    LC = 23.0  # kg
    # Horizontal Multiplier HM (25/H)
    HM = 25/H
    # Vertical Multiplier VM 1− (.003|V-75|)
    VM = 1 - (0.003 * np.abs(V - 75))  # 75cm
    # Distance Multiplier DM .82 + (4.5/D)
    DM = 0.82 + (4.5/D)  # 4.5cm
    # Asymmetric Multiplier AM 1− (.0032A)
    AM = 1 - (0.0032 * np.abs(A))  # degree
    # Frequency Multiplier FM From Table 5 -- From Table and assumption, assume  1-2 hr, based on frequency
    FM = 0.42
    # TODO: compound LI
    # Coupling Multiplier CM From Table 7 -- assume poor coupling because lifting box from side without handel
    CM = 0.9

    # RWL recommended weight limit
    RWL = LC * HM * VM * DM * AM * FM * CM

    LI = args.mass/RWL  # lifting index


    ####### reshape back to lift and lower
    LI = LI.reshape(4, -1)
    # keep max of row 1 and 2, keep max of row 3 and 4
    LI_lift = np.max(LI[:2], axis=0)
    LI_lower = np.max(LI[2:], axis=0)

    print(args.estimate_file)
    print(f"NIOSH LI \n"
          f"Lift, {','.join(map(str, LI_lift))}\n"
            f"Lower, {','.join(map(str, LI_lower))}")
    print(f"Frame No:")
    print(f"Lift Start, {','.join(map(str, lift_start))}")
    print(f"Lift End, {','.join(map(str, lift_end))}")
    print(f"Lift Duration, {','.join(map(str, lift_end - lift_start))}")
    print(f"Lower Start, {','.join(map(str, lower_start))}")
    print(f"Lower End, {','.join(map(str, lower_end))}")
    print(f"Lower Duration, {','.join(map(str, lower_end - lower_start))}")




    ######################## REBA ########################

    ######################## 3DSSPP batch file output ########################

    isaac_skeleton.c3d_file = args.estimate_file
    isaac_skeleton.calculate_joint_center()
    JOA = isaac_skeleton.calculate_3DSSPP_angles()


    lift_segments = np.vstack((lift_start, lift_end)).T
    lower_segments = np.vstack((lower_start, lower_end)).T

    all_segments = np.vstack((lift_segments, lower_segments))
    all_segments = np.hstack((all_segments, np.ones((all_segments.shape[0], 1))))  # add a column for frame number
    isaac_skeleton.output_3DSSPP_JOA(frame_range=all_segments, lift_mass=args.mass)







