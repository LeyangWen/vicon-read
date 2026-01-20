import argparse
import os.path
import pickle
from collections import defaultdict
import numpy as np

from Skeleton import *
import matplotlib
from ergo3d import Point
# matplotlib.use('Qt5Agg')
import csv


####################################################
# From isaac inference output, csv file
# Output NIOSH lifting index, REBA score, and output 3DSSPP batch txt file

####################################################


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=r'config/experiment_config/IssacGym/task_eval.yaml')
    parser.add_argument('--skeleton_file', type=str, default=r'config/VEHS_ErgoSkeleton_info/IssacGym/15kpts-Skeleton.yaml')


    parser.add_argument('--output_frame_folder', type=str, default=None)
    parser.add_argument('--output_GT_frame_folder', type=str, default=None)
    parser.add_argument('--plot_mode', type=str, default='paper_view', help='mode: camera_view, camera_side_view, 0_135_view, normal_view, paper_view, global_view')
    parser.add_argument('--debug_mode', default=True, type=bool)
    parser.add_argument('--output_type', type=list, default=[True, True, True], help='LI, plot, 3DSSPP')

    # parser.add_argument('--name_list', type=list, default=[])
    args = parser.parse_args()
    with open(args.config_file, 'r') as stream:
        data = yaml.safe_load(stream)
        args.name_list = data['name_list']
        args.estimate_file = data['estimate_file']
        args.fps = data['fps']
        args.dim = data['dim']
        args.density = data['density']
        # args.start_positions = data["start_positions"]
        # args.end_positions = data["end_positions"]
        args.start_heights = data.get("start_heights", [])
        args.start_offset = data.get("start_offset", 0)
        args.lifts = np.array(data.get("lifts", []))
        args.lowers = np.array(data.get("lowers", []))
    print(args.plot_mode)
    print(args.estimate_file)
    args.mass = args.density * args.dim[0] * args.dim[1] * args.dim[2]  # kg
    args.output_frame_folder = os.path.dirname(args.estimate_file) if args.output_frame_folder is None else args.output_frame_folder
    args.output_frame_folder = os.path.join(args.output_frame_folder, args.plot_mode)
    return args


def pop_by_id_list(input_list, id_list):
    for idx in sorted(id_list, reverse=True):
        input_list.pop(idx)
    return input_list


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

    # Step 1: specify lift and lower segments
    lift_start = args.lifts[:, 0].tolist()
    lift_end = args.lifts[:, 1].tolist()
    lower_start = args.lowers[:, 0].tolist()
    lower_end = args.lowers[:, 1].tolist()
    start_heights = args.start_heights

    low_id = np.concatenate((lift_start, lift_start, lower_end, lower_end))
    high_id = np.concatenate((lift_end, lift_end, lower_start, lower_start))
    all_id = np.concatenate((lift_start, lift_end, lower_start, lower_end))

    # Step 2: calculate NIOSH elements
    #################
    pelvis_id = 0
    foot_id = [11, 14]
    shoulder_id = [3, 6]
    box_id = 15
    #################
    foot_center = (isaac_pose[:, foot_id[0]] + isaac_pose[:, foot_id[1]]) / 2
    pelvis = isaac_pose[:, pelvis_id]
    box = isaac_pose[:, box_id]

    # Horizontal location (box center to pelvis xy)
    # H = (np.sum((box[all_id, :2] - pelvis[all_id, :2]) ** 2, axis=1) ** 0.5) * 100  # cm
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
    H_floor = 25
    H_new = np.where(H < H_floor, H_floor, H)  # if H < 25cm, set to 25cm
    HM = 25/H_new
    # Vertical Multiplier VM 1− (.003|V-75|)
    VM = 1 - (0.003 * np.abs(V - 75))  # 75cm
    # Distance Multiplier DM .82 + (4.5/D)
    D_floor = 25
    D_new = np.where(D < D_floor, D_floor, D)  # if D < 25cm, set to 25cm
    DM = 0.82 + (4.5/D_new)  # 4.5cm
    # Asymmetric Multiplier AM 1− (.0032A)
    AM = 1 - (0.0032 * np.abs(A))  # degree
    # Frequency Multiplier FM From Table 5 -- From Table and assumption, assume  1-2 hr, based on frequency
    FM = 0.94
    # TODO: compound LI
    # Coupling Multiplier CM From Table 7 -- assume poor coupling because lifting box from side without handel
    CM = 0.9 # -- assume poor coupling because lifting box from side without handel
    # CM = 1.0  # -- assume good coupling because lifting box  with handel
    # RWL recommended weight limit
    RWL = LC * HM * VM * DM * AM * FM * CM

    LI = args.mass/RWL  # lifting index


    ####### reshape back to lift and lower
    LI = LI.reshape(4, -1)
    # keep max of row 1 and 2, keep max of row 3 and 4
    # LI_lift = np.max(LI[:2], axis=0)
    # LI_lower = np.max(LI[2:], axis=0)
    LI_lift = LI[0]
    LI_lower = LI[3]

    bad_lift_id_LI = np.where(LI_lift>3)[0]
    bad_lower_id_LI = np.where(LI_lower>3)[0]

    # if > 3, replace w. nan
    LI_lift[LI_lift > 3] = np.nan
    LI_lower[LI_lower > 3] = np.nan

    lift_duration = np.array(lift_end) - np.array(lift_start)
    lower_duration = np.array(lower_end) - np.array(lower_start)

    bad_lift_id = np.where(lift_duration<=1)[0]
    bad_lower_id = np.where(lower_duration<=1)[0]
    print("#" * 40)
    print(f"Sanity check: bad lift id (0 dur): {bad_lift_id}, bad lower id (0 dur): {bad_lower_id}")
    print(f"Sanity check: bad lift id (LI>3):  {bad_lift_id_LI}, bad lower id (LI>3):  {bad_lower_id_LI}")
    print("#"*40)
    # print(f"NIOSH LI \n"
    #       f"Start height, {','.join(map(str, start_heights))}\n"
    #       f"Lift, {','.join(map(str, LI_lift))}\n"
    #         f"Lower, {','.join(map(str, LI_lower))}")
    # print(f"Frame No:")
    # print(f"Lift Start, {','.join(map(str, lift_start))}")
    # print(f"Lift End, {','.join(map(str, lift_end))}")
    # print(f"Lift Duration, {','.join(map(str, [e - s for e, s in zip(lift_end, lift_start)]))}")
    # print(f"Lower Start, {','.join(map(str, lower_start))}")
    # print(f"Lower End, {','.join(map(str, lower_end))}")
    # print(f"Lower Duration, {','.join(map(str, [e - s for e, s in zip(lower_end, lower_start)]))}")

    ######################## Back Angle ########################
    # angles from helpers
    to_deg = 180.0 / np.pi
    back_all        = isaac_skeleton.back_angles().flexion * to_deg
    left_knee_all   = isaac_skeleton.simple_knee_angle(side='left') * to_deg
    right_knee_all  = isaac_skeleton.simple_knee_angle(side='right') * to_deg
    left_elbow_all  = isaac_skeleton.simple_elbow_angle(side='left') * to_deg
    right_elbow_all = isaac_skeleton.simple_elbow_angle(side='right') * to_deg
    mean_elbow_all = 0.5 * (left_elbow_all + right_elbow_all)
    mean_knee_all = 0.5 * (left_knee_all + right_knee_all)

    pelvis_pos = isaac_skeleton.point_poses["PELVIS"].xyz.T
    torso_pos  = isaac_skeleton.point_poses["torso"].xyz.T
    # mean_body_pos = 0.5 * (pelvis_pos + torso_pos)
    mean_body_pos = pelvis_pos
    box_pos = isaac_skeleton.point_poses["1"].xyz.T
    box_to_torso_xy = box_pos[:, :2] - mean_body_pos[:, :2]
    box_to_torso_dist_all = np.linalg.norm(box_to_torso_xy, axis=1)



    ######################## CSV output ########################
    segment_ids = np.arange(len(start_heights))
    seperator = np.array([''] * len(start_heights))
    tasks = [['Lift'] * len(start_heights), ['Lower'] * len(start_heights)]

    compiled_LI_lift = np.vstack((tasks[0], start_heights, segment_ids, LI_lift, args.mass/LI_lift, lift_start, lift_end, lift_duration, seperator)).T
    compiled_LI_lower = np.vstack((tasks[1], start_heights, segment_ids, LI_lower, args.mass/LI_lower, lower_start, lower_end, lower_duration, seperator)).T
    compiled_LI = np.vstack((compiled_LI_lift, compiled_LI_lower))

    compiled_angles_lift = np.vstack((segment_ids, back_all[lift_start], mean_elbow_all[lift_start], mean_knee_all[lift_start], box_to_torso_dist_all[lift_start], seperator, left_elbow_all[lift_start], right_elbow_all[lift_start], left_knee_all[lift_start], right_knee_all[lift_start])).T
    compiled_angles_lower = np.vstack((segment_ids, back_all[lower_end], mean_elbow_all[lower_end], mean_knee_all[lower_end], box_to_torso_dist_all[lower_end], seperator, left_elbow_all[lower_end], right_elbow_all[lower_end], left_knee_all[lower_end], right_knee_all[lower_end])).T
    compiled_angles = np.vstack((compiled_angles_lift, compiled_angles_lower))

    compiled_array = np.hstack((compiled_LI, compiled_angles))
    # remove rows with id bad_lift_id and (bad_lower_id + len(start_heights))
    bad_lower_id_shifted = [i + len(start_heights) for i in bad_lower_id]
    # all_bad_id = list(set(bad_lift_id) | set(bad_lower_id_shifted) | set(bad_lift_id_LI) | set(bad_lower_id_LI))
    # # replace w. blank
    # for idx in sorted(all_bad_id, reverse=True):
    #     compiled_array[idx, 3:] = np.nan

    # rearrange based on start_heights
    # compiled_array = compiled_array[np.argsort(compiled_array[:, 0])]

    if args.output_type[0]:
        # write to csv
        csv_file = args.estimate_file.replace('.csv', '-NIOSH_LI.csv')
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Exp Name", csv_file.split('/')[-2]])
            writer.writerow(["File Name", csv_file])
            writer.writerow([
                "Task", "Lift or Lower Height (m)", "Segment Index", "LI", "RWL (kg)",
                "Start Frame", "End Frame", "Duration",
                "",  # separator
                "Segment Index",
                "Back Angle (deg)", "Mean Elbow Angle (deg)", "Mean Knee Angle (deg)",
                "Box to Body Horizontal Distance (m)",
                "",  # separator
                "Left Elbow Angle (deg)", "Right Elbow Angle (deg)",
                "Left Knee Angle (deg)", "Right Knee Angle (deg)"
            ])

            for row in compiled_array:
                writer.writerow(row)
        print(f"NIOSH LI and angle results saved to {csv_file}")

    ######################## Visualize ########################
    if args.output_type[1]:
        four_type = ['lift_start', 'lift_end', 'lower_start', 'lower_end']
        for info in zip(start_heights, segment_ids, lift_start, lift_end, lower_start, lower_end):
            start_height, segment_id = info[0], info[1]
            frame_4 = info[2:6]
            for j, frame in enumerate(frame_4):
                frame = int(frame)
                title = f"{start_height}m - {four_type[j]} - {segment_id}:{frame}"
                isaac_skeleton.plot_3d_pose_frame(frame=frame, coord_system="world-m", plot_range=2, mode=args.plot_mode, center_key='PELVIS', plot_rot=True, title=title)
            # raise NotImplementedError
    ######################## REBA ########################



    ######################## 3DSSPP batch file output ########################

    if args.output_type[2]:
        isaac_skeleton.c3d_file = args.estimate_file
        isaac_skeleton.calculate_joint_center()
        JOA = isaac_skeleton.calculate_3DSSPP_angles()
        output_type = "lowest_frame"

        # remove bad lift and lower
        lift_start = pop_by_id_list(lift_start, bad_lift_id)
        lower_start = pop_by_id_list(lower_start, bad_lower_id)
        lift_end = pop_by_id_list(lift_end, bad_lift_id)
        lower_end = pop_by_id_list(lower_end, bad_lower_id)
        if output_type == "lift_lower_motion":
            lift_segments = np.vstack((lift_start, lift_end)).T
            lower_segments = np.vstack((lower_start, lower_end)).T

            all_segments = np.vstack((lift_segments, lower_segments))
            all_segments = np.hstack((all_segments, np.ones((all_segments.shape[0], 1))))  # add a column for frame jump
        elif output_type == "lowest_frame":
            start = np.hstack((lift_start, lower_end)).T
            end = start + 1
            step = np.ones_like(start)
            all_segments = np.hstack((start[:, np.newaxis], end[:, np.newaxis], step[:, np.newaxis]))

        isaac_skeleton.output_3DSSPP_JOA(frame_range=all_segments, lift_mass=args.mass, start_offset=args.start_offset)



