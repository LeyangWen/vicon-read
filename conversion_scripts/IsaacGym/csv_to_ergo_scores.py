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
    parser.add_argument('--config_file', type=str, default=r'config/experiment_config/IssacGym/test.yaml')
    parser.add_argument('--skeleton_file', type=str, default=r'config/VEHS_ErgoSkeleton_info/IssacGym/15kpts-Skeleton.yaml')


    parser.add_argument('--output_frame_folder', type=str, default=None)
    parser.add_argument('--output_GT_frame_folder', type=str, default=None)
    parser.add_argument('--plot_mode', type=str, default='paper_view', help='mode: camera_view, camera_side_view, 0_135_view, normal_view, paper_view, global_view')
    parser.add_argument('--debug_mode', default=True, type=bool)
    parser.add_argument('--output_type', type=list, default=[True, False, True], help='LI, plot, 3DSSPP')

    # parser.add_argument('--name_list', type=list, default=[])
    args = parser.parse_args()
    with open(args.config_file, 'r') as stream:
        data = yaml.safe_load(stream)
        args.name_list = data['name_list']
        args.estimate_file = data['estimate_file']
        args.fps = data['fps']
        args.dim = data['dim']
        args.density = data['density']
        args.start_positions = data["start_positions"]
        args.end_positions = data["end_positions"]
    print(args.plot_mode)
    print(args.estimate_file)
    args.mass = args.density * args.dim[0] * args.dim[1] * args.dim[2]  # kg
    args.output_frame_folder = os.path.dirname(args.estimate_file) if args.output_frame_folder is None else args.output_frame_folder
    args.output_frame_folder = os.path.join(args.output_frame_folder, args.plot_mode)
    return args


def extract_movement_segments(box_moved: np.ndarray, min_segment_length: int = 15) -> np.ndarray:
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

def identify_start(args, isaac_pose):
    #################
    start_loc_id = 16
    box_id = 15
    #################
    i = 0
    start_ids = []
    start_heights = []
    last_z = -100
    for i in range(isaac_pose.shape[0]):
        start_loc_z = isaac_pose[i, start_loc_id, 2]
        if last_z != start_loc_z:
            # record the start id
            height = isaac_pose[i+5, box_id, 2] # center box height at segment start
            height = round(height, 2)  # round to 2 decimal places
            start_ids.append(i)
            start_heights.append(height)
            last_z = start_loc_z
    print(f"No_starts: {len(start_ids)}")
    print(f"Start ids: \n {start_ids}")
    print(f"Start heights: \n {start_heights}")
    return start_ids, start_heights


def get_MMH_start_end(args, segment):
    #################
    box_id = 15
    box_move_threshold = 0.01
    #################
    box_diff = segment[1:, box_id, :] - segment[:-1, box_id, :]
    box_moved = np.zeros(len(segment), dtype=bool)
    box_moved[1:] = np.sum(box_diff ** 2, axis=1) > box_move_threshold ** 2

    box_movement_segments = extract_movement_segments(box_moved, min_segment_length=15)  # shape: (2, no) [lift_start, lower_end]

    #################
    box_init_id = 16
    box_dest_id = 17
    box_near_xy_threshold = 0.4
    # box_near_z_threshold = 0.05
    #################
    box_init = segment[:, box_init_id]
    box_dest = segment[:, box_dest_id]
    box = segment[:, box_id]

    box2init_xy_diff = box[:, :2] - box_init[:, :2]
    box2dest_xy_diff = box[:, :2] - box_dest[:, :2]
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
    lift_start_seg, lower_end_seg = box_movement_segments
    lift_end_seg, lower_start_seg = box2init_xy_near_segments

    lower_start_seg = lower_start_seg + 2

    if len(lift_start_seg) == 0 or len(lift_end_seg) == 0 or len(lower_start_seg) == 0 or len(lower_end_seg) == 0:
        print("No lift or lower segments found, returning empty segments.")
        return False, False, False, False
    return lift_start_seg[0], lift_end_seg[0], lower_start_seg[-1], lower_end_seg[-1]
    # assert len(lift_start) == len(lift_end), "Error: lift start and lower end not equal"
    # assert len(lower_start) == len(lower_end), "Error: lift end and lower start not equal"
    # for i in range(len(lift_start)):
    #     assert lift_start[i] < lift_end[i], "Error: lift start should be less than lift end"
    # for i in range(len(lower_start)):
    #     assert lower_start[i] < lower_end[i], "Error: lower start should be less than lower end"


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

    start_ids, start_heights = identify_start(args, isaac_pose)
    end_ids = start_ids.copy()
    end_ids[:-1] = start_ids[1:]
    end_ids[-1] = frame_no - 1  # last end id is the last frame
    print(f"End ids: {end_ids}")
    bad_id = []
    lift_start, lift_end, lower_end, lower_start = [[], [], [], []]
    for i in range(len(start_ids)):
        print(f"Start: {start_ids[i]}, End: {end_ids[i]}")
        segment = isaac_pose[start_ids[i]:end_ids[i], :, :]

        a,b,c,d = get_MMH_start_end(args, segment)
        if not d:
            bad_id.append(i)
            print(f"Bad segment {i}, skipping")
        lift_start.append(a + start_ids[i])
        lift_end.append(b + start_ids[i])
        lower_start.append(c + start_ids[i])
        lower_end.append(d + start_ids[i])

    # if len(bad_id) > 0:
    #     bad_id.reverse()
    #     for i in bad_id:
    #         start_ids.pop(i)
    #         end_ids.pop(i)
    #         start_heights.pop(i)

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
    H = (np.sum((box[all_id, :2] - pelvis[all_id, :2]) ** 2, axis=1) ** 0.5) * 100  # cm
    # H = (np.sum((box[all_id, :2] - foot_center[all_id, :2])**2, axis=1)**0.5)*100  # cm
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
    CM = 0.9

    # RWL recommended weight limit
    RWL = LC * HM * VM * DM * AM * FM * CM

    LI = args.mass/RWL  # lifting index


    ####### reshape back to lift and lower
    LI = LI.reshape(4, -1)
    # keep max of row 1 and 2, keep max of row 3 and 4
    LI_lift = np.max(LI[:2], axis=0)
    LI_lower = np.max(LI[2:], axis=0)

    # if > 3, replace w. nan
    LI_lift[LI_lift > 3] = np.nan
    LI_lower[LI_lower > 3] = np.nan

    lift_duration = np.array(lift_end) - np.array(lift_start)
    lower_duration = np.array(lower_end) - np.array(lower_start)

    print(args.estimate_file)
    print(f"NIOSH LI \n"
          f"Start height, {','.join(map(str, start_heights))}\n"
          f"Lift, {','.join(map(str, LI_lift))}\n"
            f"Lower, {','.join(map(str, LI_lower))}")
    print(f"Frame No:")
    print(f"Lift Start, {','.join(map(str, lift_start))}")
    print(f"Lift End, {','.join(map(str, lift_end))}")
    print(f"Lift Duration, {','.join(map(str, [e - s for e, s in zip(lift_end, lift_start)]))}")
    print(f"Lower Start, {','.join(map(str, lower_start))}")
    print(f"Lower End, {','.join(map(str, lower_end))}")
    print(f"Lower Duration, {','.join(map(str, [e - s for e, s in zip(lower_end, lower_start)]))}")

    segment_ids = np.arange(len(start_heights))
    compiled_array = np.vstack((start_heights, segment_ids, LI_lift, LI_lower, lift_start, lift_end, lift_duration,
                                lower_start, lower_end, lower_duration)).T

    # rearrange based on start_heights
    compiled_array = compiled_array[np.argsort(compiled_array[:, 0])]

    if args.output_type[0]:
        # write to csv
        csv_file = args.estimate_file.replace('.csv', '-NIOSH_LI.csv')
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Exp Name", csv_file.split('/')[-2]])
            writer.writerow(["File Name", csv_file])
            writer.writerow(["Lift/lower Height (m)", "Segment Index", "Lift LI", "Lower LI", "Lift Start Frame", "Lift End Frame", "Lift Duration",
                             "Lower Start Frame", "Lower End Frame", "Lower Duration"])

            for row in compiled_array:
                writer.writerow(row)

    ######################## Visualize ########################
    if args.output_type[1]:
        four_type = ['lift_start', 'lift_end', 'lower_start', 'lower_end']
        for i, frame_complied in enumerate(compiled_array):
            start_height = frame_complied[0]
            frame_4 = [frame_complied[4], frame_complied[5], frame_complied[7], frame_complied[8]]
            segment_id = int(frame_complied[1])
            for j, frame in enumerate(frame_4):
                frame = int(frame)
                title = f"{start_height}m - {four_type[j]} - {segment_id}:{frame}"
                isaac_skeleton.plot_3d_pose_frame(frame=frame, coord_system="world-m", plot_range=2, mode=args.plot_mode, center_key='PELVIS', plot_rot=True, title=title)

    ######################## REBA ########################

    ######################## Back Angle ########################



    ######################## 3DSSPP batch file output ########################

    if args.output_type[2]:
        isaac_skeleton.c3d_file = args.estimate_file
        isaac_skeleton.calculate_joint_center()
        JOA = isaac_skeleton.calculate_3DSSPP_angles()
        output_type = "lowest_frame"
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

            start_offset = 80 * 2

        isaac_skeleton.output_3DSSPP_JOA(frame_range=all_segments, lift_mass=args.mass, start_offset=start_offset)



