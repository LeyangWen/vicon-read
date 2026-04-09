import argparse
import os.path
import pickle
from collections import defaultdict
import numpy as np
import copy

from Skeleton import *
import matplotlib
from ergo3d import Point
matplotlib.use('Qt5Agg')
import csv

# in vscode
# PYTHONPATH=$(pwd) python conversion_scripts/IsaacGym/csv_to_ergo_scores.py


####################################################
# From isaac inference output, csv file
# Output NIOSH lifting index, REBA score, and output 3DSSPP batch txt file


# Steps: csv-to-ergo-scores --> 3DSSPP run batch and adjust --> if adjusted, run export batch file to export --> run SSPP_txt_to_scores.py to read 3DSSPP --> CSV copy to excel --> excel sheet summerized results copy to claude to format --> boxplot_subject.py to plot
####################################################


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config_file', type=str, default=r'config/experiment_config/IssacGym/exp/exp_mocap.yaml')
    # parser.add_argument('--output_excel', type=str, default=r'/Volumes/Y/intervention_eval_data/worker_motion/issac/Mocap_results.csv')
    # #
    parser.add_argument('--config_file', type=str, default=r'config/experiment_config/IssacGym/exp/exp_flat_v1.yaml')
    parser.add_argument('--output_excel', type=str, default=r'/Volumes/Y/intervention_eval_data/recommended_motion/exp_flat_v1/Generated_flat_results.csv')

    # parser.add_argument('--config_file', type=str, default=r'config/experiment_config/IssacGym/exp/exp_terrain_step_v1.yaml')
    # parser.add_argument('--output_excel', type=str, default=r'/Volumes/Y/intervention_eval_data/recommended_motion/exp_terrain_step_v1/Generated_step_results.csv')

    parser.add_argument('--skeleton_file', type=str, default=r'config/VEHS_ErgoSkeleton_info/IssacGym/15kpts-Skeleton.yaml')
    parser.add_argument('--output_frame_folder', type=str, default=None)
    parser.add_argument('--output_GT_frame_folder', type=str, default=None)
    parser.add_argument('--plot_mode', type=str, default='camera_view', help='mode: camera_view, camera_side_view, 0_135_view, normal_view, paper_view, global_view')
    parser.add_argument('--debug_mode', default=True, type=bool)
    parser.add_argument('--output_type', type=list, default=[True, True, True], help='LI, plot, 3DSSPP')
    parser.add_argument('--mute_lowers', default=True, help='Exclude lower rows from output')

    # parser.add_argument('--name_list', type=list, default=[])
    args = parser.parse_args()

    with open(args.config_file, 'r') as stream:
        data = yaml.safe_load(stream)

    # Check if this is a batch config (has 'experiments' key)
    if 'experiments' in data:
        args.batch_mode = True
        args.batch_data = data
        args.start_offset = data.get("start_offset", 0)
    else:
        args.batch_mode = False
        args = _apply_single_config(args, data)

    return args


def _apply_single_config(args, data):
    """Apply a single-experiment YAML config dict to args. Used for both old-style configs and batch sub-experiments."""
    args.name_list = data['name_list']
    args.estimate_file = data['estimate_file']
    args.fps = data['fps']
    args.dim = data.get('dim', None)
    args.density = data.get('density', None)
    args.start_positions = data["start_positions"]
    args.end_positions = data["end_positions"]
    args.start_offset = data.get("start_offset", 0)
    args.keyframe_detect = data["keyframe_detect"]
    args.mute_lowers = data.get("mute_lowers", args.mute_lowers)
    args.lift_detect = data.get("lift_detect", False)

    # Support both old 'density'+'dim' and new 'weight' field
    if 'weight' in data:
        args.mass = data['weight']
    elif args.density is not None and args.dim is not None:
        args.mass = args.density * args.dim[0] * args.dim[1] * args.dim[2]
    else:
        raise ValueError("Config must specify either 'weight' or both 'density' and 'dim'")

    if not args.keyframe_detect:
        print("*"*10,"Warning: keyframe_detect is set to False, please specify lift start and end frames in the yaml file!","*"*10)
        keyframe_dict = {}
        no_experiments = data['no_experiments']
        keyframe_dict['lift_start'] = data.get("lift_start", [-1]*no_experiments)
        keyframe_dict['lift_end'] = data.get("lift_end", [-1]*no_experiments)
        keyframe_dict['lower_start'] = data.get("lower_start", [-1]*no_experiments)
        keyframe_dict['lower_end'] = data.get("lower_end", [-1]*no_experiments)
        keyframe_dict['start_heights'] = np.array(args.start_positions)[:, 2].T.tolist()
        if len(keyframe_dict['start_heights']) == 1:
            keyframe_dict['start_heights'] = keyframe_dict['start_heights'] * no_experiments
        args.keyframe_dict = keyframe_dict
        print(f"keyframe_dict: {keyframe_dict}")

    print(args.plot_mode)
    print(args.estimate_file)
    args.output_frame_folder = os.path.dirname(args.estimate_file) if args.output_frame_folder is None else args.output_frame_folder
    args.output_frame_folder = os.path.join(args.output_frame_folder, args.plot_mode)
    return args


def build_single_args(base_args, defaults, exp_dict):
    """Build an args namespace for one experiment by merging defaults with experiment-specific overrides."""
    merged = {}
    merged.update(defaults)
    merged.update(exp_dict)
    single_args = copy.deepcopy(base_args)
    single_args = _apply_single_config(single_args, merged)
    return single_args


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
    last_loc = hash((-1000, -1000, -1000))
    for i in range(isaac_pose.shape[0]):
        start_loc_z = isaac_pose[i, start_loc_id, 2]
        start_loc_x = isaac_pose[i, start_loc_id, 0]
        start_loc_y = isaac_pose[i, start_loc_id, 1]
        start_loc = hash((start_loc_x, start_loc_y, start_loc_z))
        if last_loc != start_loc:
            # record the start id
            height = isaac_pose[i+5, box_id, 2] # center box height at segment start
            height = round(height, 2)  # round to 2 decimal places
            start_ids.append(i)
            start_heights.append(height)
            last_loc = start_loc
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
    # carry_height = np.median(box[box2_init_dest_xy_far, 2])

    # box2carry_z_diff = box[:, 2] - carry_height
    # box2carry_z_near = np.abs(box2carry_z_diff) < box_near_z_threshold

    box2init_xy_near_segments = extract_movement_segments(box2_init_dest_xy_far, min_segment_length=5)  # shape: (2, no) [lift_end, lower_start]
    # box2dest_xy_near_segments = extract_movement_segments(box2dest_xy_near, min_segment_length=5)

    ################# output #################
    lift_start_seg, lower_end_seg = box_movement_segments
    lift_end_seg, lower_start_seg = box2init_xy_near_segments

    max_frame = segment.shape[0] - 1
    lower_start_seg = lower_start_seg + 2
    lower_start_seg = np.clip(lower_start_seg, 0, max_frame)

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


def pop_by_id_list(input_list, id_list):
    for idx in sorted(id_list, reverse=True):
        input_list.pop(idx)
    return input_list


def process_single_experiment(args, exp_name=None):
    """
    Process a single experiment. Returns (compiled_array, header, csv_file_path, locals_dict).
    compiled_array: np.ndarray of results, header: list of column names.
    If exp_name is provided, it is prepended to each row for batch mode.
    """
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

    # Auto-detect lift_start from box z-movement if lift_detect is enabled
    if getattr(args, 'lift_detect', False) and not args.keyframe_detect:
        box_id = 15
        box_z = isaac_pose[:, box_id, 2]
        box_z_diff = np.diff(box_z)
        kd = args.keyframe_dict

        # Detect lift_start if empty
        if len(kd['lift_start']) == 0:
            candidates = np.where(box_z_diff > 0.01)[0]
            if len(candidates) > 0:
                kd['lift_start'] = [int(candidates[0])+1]
            else:
                raise ValueError("lift_detect enabled but no lift_start detected from box movement. Please check the data or adjust the detection threshold.")
                kd['lift_start'] = [0]
            print(f"lift_detect: auto-detected lift_start = {kd['lift_start']}")

        # Replace -1 in lift_end with last frame
        kd['lift_end'] = [frame_no - 1 if v == -1 else v for v in kd['lift_end']]

        # Fill lower_start/lower_end with lift_end if still default (mute_lowers will skip them anyway)
        if len(kd['lower_start']) == 0 or kd['lower_start'] == [-1]:
            kd['lower_start'] = kd['lift_end']
        if len(kd['lower_end']) == 0 or kd['lower_end'] == [-1]:
            kd['lower_end'] = kd['lift_end']

        # Ensure start_heights matches
        if len(kd['start_heights']) != len(kd['lift_start']):
            kd['start_heights'] = kd['start_heights'] * len(kd['lift_start'])

        print(f"lift_detect: keyframe_dict after detection = {kd}")

    # Step 1: detect key events in frame no: each lift start, lift end, lower start, lower end
    if args.keyframe_detect:
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

    else:
        start_heights = args.keyframe_dict['start_heights']
        lift_start = args.keyframe_dict['lift_start']
        lift_end = args.keyframe_dict['lift_end']
        lower_start = args.keyframe_dict['lower_start']
        lower_end = args.keyframe_dict['lower_end']

    low_id = np.concatenate((lift_start, lift_start, lower_end, lower_end))
    high_id = np.concatenate((lift_end, lift_end, lower_start, lower_start))
    all_id = np.concatenate((lift_start, lift_end, lower_start, lower_end))

    # Step 2: calculate NIOSH elements
    #################
    pelvis_id = 0
    foot_id = [11, 14]
    shoulder_id = [3, 6]
    hand_id = [5, 8]
    box_id = 15
    #################
    foot_center = (isaac_pose[:, foot_id[0]] + isaac_pose[:, foot_id[1]]) / 2
    hand_center = (isaac_pose[:, hand_id[0]] + isaac_pose[:, hand_id[1]]) / 2
    pelvis = isaac_pose[:, pelvis_id]
    box = isaac_pose[:, box_id]

    # Horizontal location (box center to pelvis xy)
    H = (np.sum((hand_center[all_id, :2] - foot_center[all_id, :2])**2, axis=1)**0.5)*100  # cm
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
    # Frequency Multiplier FM From Table 5
    FM = 0.94
    # Coupling Multiplier CM From Table 7
    CM = 0.9
    # RWL recommended weight limit
    RWL = LC * HM * VM * DM * AM * FM * CM
    print("#"*40)
    print(f"H: {H} cm \nV: {V} cm\nD: {D} cm\nA: {A} deg\nFreq: {frequency} Hz\nCount: {count}\nTotal time: {total_time*60:.1f} min")
    print(f"LC: {LC} \nHM: {HM}\nVM: {VM}\nDM: {DM}\nAM: {AM}\nFM: {FM}\nCM: {CM}\nRWL: {RWL}") 
    LI = args.mass/RWL  # lifting index


    ####### reshape back to lift and lower
    
    print("LI", LI)
    print("all_id", all_id)
    print("#"*40)
    LI = LI.reshape(4, -1)
    LI_lift = LI[0]
    LI_lower = LI[3]

    # bad_lift_id_LI = np.where(LI_lift>3)[0]
    # bad_lower_id_LI = np.where(LI_lower>3)[0]

    # if > 3, replace w. nan
    # LI_lift[LI_lift > 3] = np.nan
    # LI_lower[LI_lower > 3] = np.nan

    lift_duration = np.array(lift_end) - np.array(lift_start)
    lower_duration = np.array(lower_end) - np.array(lower_start)

    bad_lift_id = np.where(lift_duration<=1)[0]
    bad_lower_id = np.where(lower_duration<=1)[0]
    print("#" * 40)
    print(f"Sanity check: bad lift id (0 dur): {bad_lift_id}, bad lower id (0 dur): {bad_lower_id}")
    # print(f"Sanity check: bad lift id (LI>3):  {bad_lift_id_LI}, bad lower id (LI>3):  {bad_lower_id_LI}")
    print("#"*40)

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
    mean_body_pos = pelvis_pos
    box_pos = isaac_skeleton.point_poses["1"].xyz.T
    box_to_torso_xy = box_pos[:, :2] - mean_body_pos[:, :2]
    box_to_torso_dist_all = np.linalg.norm(box_to_torso_xy, axis=1)



    ######################## Compile results ########################
    segment_ids = np.arange(len(start_heights))
    seperator = np.array([''] * len(start_heights))
    tasks = [['Lift'] * len(start_heights), ['Lower'] * len(start_heights)]

    weight_col = np.array([args.mass] * len(start_heights))
    sspp_placeholder = np.array([''] * len(start_heights))  # 3DSSPP Frame placeholder (filled in batch mode)
    compiled_LI_lift = np.vstack((tasks[0], weight_col, start_heights, segment_ids, LI_lift, args.mass/LI_lift, sspp_placeholder, lift_start, lift_end, lift_duration, seperator)).T
    compiled_LI_lower = np.vstack((tasks[1], weight_col, start_heights, segment_ids, LI_lower, args.mass/LI_lower, sspp_placeholder, lower_start, lower_end, lower_duration, seperator)).T
    compiled_LI = np.vstack((compiled_LI_lift, compiled_LI_lower))

    compiled_angles_lift = np.vstack((segment_ids, back_all[lift_start], mean_elbow_all[lift_start], mean_knee_all[lift_start], box_to_torso_dist_all[lift_start], seperator, left_elbow_all[lift_start], right_elbow_all[lift_start], left_knee_all[lift_start], right_knee_all[lift_start])).T
    compiled_angles_lower = np.vstack((segment_ids, back_all[lower_end], mean_elbow_all[lower_end], mean_knee_all[lower_end], box_to_torso_dist_all[lower_end], seperator, left_elbow_all[lower_end], right_elbow_all[lower_end], left_knee_all[lower_end], right_knee_all[lower_end])).T
    compiled_angles = np.vstack((compiled_angles_lift, compiled_angles_lower))

    compiled_array = np.hstack((compiled_LI, compiled_angles))
    # remove rows with id bad_lift_id and (bad_lower_id + len(start_heights))
    bad_lower_id_shifted = [i + len(start_heights) for i in bad_lower_id]
    all_bad_id = list(set(bad_lift_id) | set(bad_lower_id_shifted))
    # replace w. blank
    for idx in sorted(all_bad_id, reverse=True):
        compiled_array[idx, 4:] = np.nan

    # Optionally exclude Lower rows from output
    if args.mute_lowers:
        lift_mask = compiled_array[:, 0] == 'Lift'
        compiled_array = compiled_array[lift_mask]

    header = [
        "Task", "Weight (kg)", "Lift or Lower Height (m)", "Segment Index", "LI", "RWL (kg)",
        "3DSSPP Frame",
        "Start Frame", "End Frame", "Duration",
        "",  # separator
        "Segment Index",
        "Back Angle (deg)", "Mean Elbow Angle (deg)", "Mean Knee Angle (deg)",
        "Box to Body Horizontal Distance (m)",
        "",  # separator
        "Left Elbow Angle (deg)", "Right Elbow Angle (deg)",
        "Left Knee Angle (deg)", "Right Knee Angle (deg)"
    ]

    # Return everything needed for output
    result = {
        'compiled_array': compiled_array,
        'header': header,
        'start_heights': start_heights,
        'segment_ids': segment_ids,
        'lift_start': lift_start,
        'lift_end': lift_end,
        'lower_start': lower_start,
        'lower_end': lower_end,
        'bad_lift_id': bad_lift_id,
        'bad_lower_id': bad_lower_id,
        'isaac_skeleton': isaac_skeleton,
    }
    return result


def write_single_csv(args, result):
    """Write results for a single experiment to its own CSV file (original behavior)."""
    compiled_array = result['compiled_array']
    header = result['header']
    csv_file = args.estimate_file.replace('.csv', '-NIOSH_LI.csv')
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Exp Name", csv_file.split('/')[-2]])
        writer.writerow(["File Name", csv_file])
        writer.writerow(header)
        for row in compiled_array:
            writer.writerow(row)
    print(f"NIOSH LI and angle results saved to {csv_file}")


def visualize_experiment(args, result):
    """Visualize key frames for a single experiment (original behavior)."""
    isaac_skeleton = result['isaac_skeleton']
    start_heights = result['start_heights']
    segment_ids = result['segment_ids']
    lift_start = result['lift_start']
    lift_end = result['lift_end']
    lower_start = result['lower_start']
    lower_end = result['lower_end']

    os.makedirs(args.output_frame_folder, exist_ok=True)

    if args.mute_lowers:
        frame_types = ['lift_start', 'lift_end']
        for info in zip(start_heights, segment_ids, lift_start, lift_end):
            start_height, segment_id = info[0], info[1]
            frames = info[2:4]
            for j, frame in enumerate(frames):
                frame = int(frame)
                title = f"{start_height}m - {frame_types[j]} - {segment_id}:{frame}"
                filename = os.path.join(args.output_frame_folder, f'{frame}')
                # isaac_skeleton.plot_3d_pose_frame(frame=frame, coord_system="world-m", plot_range=2, mode=args.plot_mode, center_key='PELVIS', plot_rot=True, title=title)
                # raise NotImplementedError
                isaac_skeleton.plot_3d_pose_frame(filename=filename, frame=frame, coord_system="world-m", plot_range=2, mode=args.plot_mode, center_key='PELVIS', plot_rot=True, title=title)
    else:
        frame_types = ['lift_start', 'lift_end', 'lower_start', 'lower_end']
        for info in zip(start_heights, segment_ids, lift_start, lift_end, lower_start, lower_end):
            start_height, segment_id = info[0], info[1]
            frames = info[2:6]
            for j, frame in enumerate(frames):
                frame = int(frame)
                title = f"{start_height}m - {frame_types[j]} - {segment_id}:{frame}"
                filename = os.path.join(args.output_frame_folder, f'{frame}')
                isaac_skeleton.plot_3d_pose_frame(filename=filename, frame=frame, coord_system="world-m", plot_range=2, mode=args.plot_mode, center_key='PELVIS', plot_rot=True, title=title)


def _prepare_3dsspp_segments(args, result):
    """Compute the frame segments for 3DSSPP output. Returns (all_segments, isaac_skeleton with JOA computed)."""
    isaac_skeleton = result['isaac_skeleton']
    lift_start = list(result['lift_start'])  # copy to avoid mutating
    lift_end = list(result['lift_end'])
    lower_start = list(result['lower_start'])
    lower_end = list(result['lower_end'])
    bad_lift_id = result['bad_lift_id']
    bad_lower_id = result['bad_lower_id']

    isaac_skeleton.c3d_file = args.estimate_file
    isaac_skeleton.calculate_joint_center()
    JOA = isaac_skeleton.calculate_3DSSPP_angles()
    output_type = "lowest_frame"

    # remove bad lift and lower
    lift_start = pop_by_id_list(lift_start, bad_lift_id)
    lift_end = pop_by_id_list(lift_end, bad_lift_id)
    if not args.mute_lowers:
        lower_start = pop_by_id_list(lower_start, bad_lower_id)
        lower_end = pop_by_id_list(lower_end, bad_lower_id)

    if output_type == "lift_lower_motion":
        lift_segments = np.vstack((lift_start, lift_end)).T
        if args.mute_lowers:
            all_segments = lift_segments
        else:
            lower_segments = np.vstack((lower_start, lower_end)).T
            all_segments = np.vstack((lift_segments, lower_segments))
        all_segments = np.hstack((all_segments, np.ones((all_segments.shape[0], 1))))  # add a column for frame jump
    elif output_type == "lowest_frame":
        if args.mute_lowers:
            start = np.array(lift_start)
        else:
            start = np.hstack((lift_start, lower_end))
        end = start + 1
        step = np.ones_like(start)
        all_segments = np.hstack((start[:, np.newaxis], end[:, np.newaxis], step[:, np.newaxis]))

    return all_segments, isaac_skeleton


def output_3dsspp(args, result):
    """Output 3DSSPP batch file for a single experiment (original behavior)."""
    all_segments, isaac_skeleton = _prepare_3dsspp_segments(args, result)
    isaac_skeleton.output_3DSSPP_JOA(frame_range=all_segments, lift_mass=args.mass, start_offset=args.start_offset)


def generate_3dsspp_lines(args, result, frame_counter, export_only=False):
    """
    Generate 3DSSPP body lines for one experiment for batch accumulation.
    Returns (lines, frame_numbers, next_frame_counter).
    - lines: list of strings (FRM, SUP, JOA lines) without header/footer
    - frame_numbers: list of 3DSSPP frame numbers, one per compiled row
    - next_frame_counter: the frame counter after this experiment (including separator)
    """
    all_segments, isaac_skeleton = _prepare_3dsspp_segments(args, result)
    JOA = isaac_skeleton.JOA.copy()

    lines = []
    frame_numbers = []
    frame_range = np.array(all_segments, dtype=int)
    if frame_range.ndim == 1 and frame_range.shape[0] == 3:
        frame_range = [frame_range]

    support_feet_max_height = 0.15

    # Write experiment header: task name, hand load and description
    exp_label = getattr(args, "exp_name", "unknown")
    hand_load = args.mass * 9.8  # N
    lines.append(f'COM Experiment: {exp_label} #\n')
    lines.append(f'DES 1 "{exp_label}" "Analyst" "Comments" "Company" #\n')

    for f_range in frame_range:
        start_frame = f_range[0]
        end_frame = f_range[1]
        step = f_range[2]
        for k in np.arange(start_frame, end_frame, step):
            if not export_only:
                joint_rotations = np.array2string(JOA[k], separator=' ', max_line_width=1000000, precision=3, suppress_small=True)[1:-1].replace('0. ', '0 ')
                lines.append(f'FRM {frame_counter} #\n')
                lines.append(f'HAN {hand_load / 2} -90 0 {hand_load / 2} -90 0 #\n')
                # left_foot_supported = isaac_skeleton.poses['left_toe'][k, 2] < support_feet_max_height
                # right_foot_supported = isaac_skeleton.poses['right_toe'][k, 2] < support_feet_max_height
                # if left_foot_supported and right_foot_supported:
                #     foot_support_parameter = 0
                # elif left_foot_supported and not right_foot_supported:
                #     foot_support_parameter = 1
                # elif not left_foot_supported and right_foot_supported:
                #     foot_support_parameter = 2
                # else:
                #     foot_support_parameter = 0
                foot_support_parameter = 0
                pelvic_tilt = 0
                lines.append(f'SUP {foot_support_parameter} 0 0 0 20.0 {pelvic_tilt} #\n')
                lines.append(f'JOA {joint_rotations} #\n')
            lines.append("OUT #\n")
            frame_numbers.append(frame_counter)
            frame_counter += 1

    # Skip a frame number between experiments (gap in numbering, no output)
    frame_counter += 1

    return lines, frame_numbers, frame_counter


if __name__ == '__main__':
    # read arguments
    args = parse_args()

    if not args.batch_mode:
        ######################## Single experiment mode (backward compatible) ########################
        result = process_single_experiment(args)

        if args.output_type[0]:
            write_single_csv(args, result)

        if args.output_type[1]:
            visualize_experiment(args, result)

        if args.output_type[2]:
            output_3dsspp(args, result)

    else:
        ######################## Batch mode ########################
        data = args.batch_data
        experiments = data['experiments']

        # Collect default/shared keys (everything except 'experiments')
        defaults = {k: v for k, v in data.items() if k != 'experiments'}

        # Determine output CSV path
        if args.output_excel is not None:
            output_csv = args.output_excel
        else:
            output_csv = os.path.join(os.path.dirname(args.config_file),
                                      os.path.splitext(os.path.basename(args.config_file))[0] + '_results.csv')

        all_rows = []
        header = None
        all_3dsspp_lines = []   # accumulated 3DSSPP body lines
        export_command_lines = []  # for printing cp commands to terminal
        frame_counter = args.start_offset       # running 3DSSPP frame number across experiments

        for i, exp_dict in enumerate(experiments):
            exp_name = exp_dict.get('name', f'exp_{i}')
            print("=" * 60)
            print(f"Processing experiment {i+1}/{len(experiments)}: {exp_name}")
            print("=" * 60)

            try:
                single_args = build_single_args(args, defaults, exp_dict)
                single_args.exp_name = exp_name
                result = process_single_experiment(single_args, exp_name=exp_name)

                if header is None:
                    header = ["Exp Name"] + result['header']

                compiled_array = result['compiled_array']

                # Generate 3DSSPP lines and get frame number mapping
                if args.output_type[2]:
                    sspp_lines, frame_numbers, frame_counter = generate_3dsspp_lines(single_args, result, frame_counter)
                    all_3dsspp_lines.extend(sspp_lines)

                else:
                    frame_numbers = [''] * len(compiled_array)

                # Append rows with 3DSSPP frame numbers
                for row_idx, row in enumerate(compiled_array):
                    sspp_frame = frame_numbers[row_idx] if row_idx < len(frame_numbers) else ''
                    row_list = [exp_name] + list(row)
                    # Insert 3DSSPP frame at index 7 (after RWL, replacing the placeholder)
                    row_list[7] = sspp_frame
                    all_rows.append(row_list)

                # Optionally still write individual CSVs
                if args.output_type[0]:
                    write_single_csv(single_args, result)

                if args.output_type[1]:
                    visualize_experiment(single_args, result)

                # Individual 3DSSPP files still written in single mode only (batch writes one big file)

            except Exception as e:
                print(f"ERROR processing {exp_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Write combined CSV
        if len(all_rows) > 0 and header is not None:
            with open(output_csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
                for row in all_rows:
                    writer.writerow(row)
            print("=" * 60)
            print(f"Combined results saved to {output_csv}")
            print(f"Total experiments processed: {len(all_rows)} rows from {len(experiments)} experiments")
            print("=" * 60)

            # Write combined 3DSSPP batch file next to CSV
            if args.output_type[2] and len(all_3dsspp_lines) > 0:
                sspp_file = output_csv.replace('.csv', '-3DSSPP.txt')
                weight = 90   # default anthropometry
                height = 175
                gender_id = 0  # male
                hand_load = 0  # will vary per experiment, but header needs a default
                with open(sspp_file, 'w') as f:
                    f.write('3DSSPPBATCHFILE #\n')
                    f.write('COM Combined batch from all experiments #\n')
                    f.write(f'DES 1 "Batch" "Analyst" "Comments" "Company" #\n')
                    f.write(f'ANT {gender_id} 3 {height} {weight} #\n')
                    f.write('AUT 0 #\n')
                    for line in all_3dsspp_lines:
                        f.write(line)
                    f.write('COM Task done #')
                print(f"Combined 3DSSPP batch file saved to {sspp_file}")
                onedrive_dir ='/Users/leyangwen/Library/CloudStorage/OneDrive-Umich/isaac_3dsspp/intervention_eval_data/'
                # exp_dir = os.path.splitext(os.path.basename(args.config_file))[0] # 'exp_terrain_step_v1'
                file_name = sspp_file.split('/')[-1]
                print(r'cp "{}" "{}"'.format(sspp_file, os.path.join(onedrive_dir, f'{file_name}')))
        else:
            print("No results to write. All experiments may have failed.")

# # run in terminal:
# print(r'cp "/Volumes/Y/intervention_eval_data/recommended_motion/exp_flat_v1/Generated_results-3DSSPP.txt" "/Users/leyangwen/Library/CloudStorage/OneDrive-Umich/isaac_3dsspp/intervention_eval_data/" && cp "/Volumes/Y/intervention_eval_data/worker_motion/issac/Mocap_results-3DSSPP.txt" "/Users/leyangwen/Library/CloudStorage/OneDrive-Umich/isaac_3dsspp/intervention_eval_data/" ')
# print()
#
# print(r'cp "/Volumes/Y/intervention_eval_data/recommended_motion/exp_flat_v1/Generated_results-3DSSPP.txt" "/Users/leyangwen/Library/CloudStorage/OneDrive-Umich/isaac_3dsspp/intervention_eval_data/" && cp "/Volumes/Y/intervention_eval_data/worker_motion/issac/Mocap_results-3DSSPP.txt" "/Users/leyangwen/Library/CloudStorage/OneDrive-Umich/isaac_3dsspp/intervention_eval_data/" ')
