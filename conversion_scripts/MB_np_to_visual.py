import argparse
import os.path
import pickle
from Skeleton import *
import matplotlib
import copy
# matplotlib.use('Qt5Agg')

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config_file', type=str, default=r'config/experiment_config/Rokoko-Hand-21-MB.yaml')
    # parser.add_argument('--config_file', type=str, default=r'config/experiment_config/Inference-Hand-21-RTMPose-MB.yaml')
    # parser.add_argument('--skeleton_file', type=str, default=r'config/VEHS_ErgoSkeleton_info/Ergo-Hand-21.yaml')
    # parser.add_argument('--type', type=str, default='hand')

    # parser.add_argument('--config_file', type=str, default=r'config/experiment_config/H36M17kpts/H36M-MB.yaml')
    # parser.add_argument('--config_file', type=str, default=r'config/experiment_config/H36M17kpts/VEHS-3D-MB.yaml')
    # parser.add_argument('--skeleton_file', type=str, default=r'config/VEHS_ErgoSkeleton_info/H36M-17.yaml')

    parser.add_argument('--config_file', type=str, default=r'config/experiment_config/37kpts/Inference-RTMPose-MB-20fps-Industry.yaml')
    parser.add_argument('--skeleton_file', type=str, default=r'config/VEHS_ErgoSkeleton_info/Ergo-Skeleton-37.yaml')
    parser.add_argument('--type', type=str, default='body')
    parser.add_argument('--clip_fill', type=bool, default=False)
    parser.add_argument('--rescale_25d', type=bool, default=False)
    parser.add_argument('--dynamic_plot_range', type=bool, default=True)
    parser.add_argument('--debug_mode', default=False, type=bool)

    # parser.add_argument('--config_file', type=str, default=r'config/experiment_config/37kpts/Inference-RTMPose-MB-20fps-VEHS7M.yaml')
    # parser.add_argument('--skeleton_file', type=str, default=r'config/VEHS_ErgoSkeleton_info/Ergo-Skeleton-37.yaml')
    # parser.add_argument('--type', type=str, default='body')
    # parser.add_argument('--clip_fill', type=bool, default=True)
    # parser.add_argument('--rescale_25d', type=bool, default=True)
    # parser.add_argument('--dynamic_plot_range', type=bool, default=False)
    # parser.add_argument('--debug_mode', default=True, type=bool)

    parser.add_argument('--output_frame_folder', type=str, default=None)
    parser.add_argument('--output_GT_frame_folder', type=str, default=None)
    parser.add_argument('--plot_mode', type=str, default='camera_side_view', help='mode: paper_view, camera_side_view, camera_view, 0_135_view, normal_view')
    parser.add_argument('--MB_data_stride', type=int, default=243)



    # parser.add_argument('--name_list', type=list, default=[])
    args = parser.parse_args()
    with open(args.config_file, 'r') as stream:
        data = yaml.safe_load(stream)
        args.name_list = data['name_list']
        args.GT_file = data['GT_file']
        args.eval_key = data['eval_key']
        args.estimate_file = data['estimate_file']
    print(args.plot_mode)
    args.output_frame_folder = os.path.dirname(args.estimate_file) if args.output_frame_folder is None else args.output_frame_folder
    args.output_GT_frame_folder = os.path.dirname(args.GT_file) if args.output_GT_frame_folder is None else args.output_GT_frame_folder
    args.output_frame_folder = os.path.join(args.output_frame_folder, args.plot_mode)
    GT_base_folder = args.output_GT_frame_folder
    args.output_GT_frame_folder = os.path.join(GT_base_folder, args.plot_mode)
    args.output_2D_frame_folder = os.path.join(GT_base_folder, '2D')
    return args


def MB_output_pose_file_loader(args):
    if args.estimate_file=='None':
        return None
    with open(args.estimate_file, "rb") as f:
        output_np_pose = np.load(f)
    np_pose_shape = output_np_pose.shape
    output_np_pose = output_np_pose.reshape(-1, np_pose_shape[-2], np_pose_shape[-1])
    return output_np_pose


def MB_input_pose_file_loader(args, data_key='joint3d_image', get_confidence=False):
    if args.GT_file=='None':
        return None
    with open(args.GT_file, "rb") as f:
        data = pickle.load(f)

    print(f'2.5d_factor: {data[args.eval_key]["2.5d_factor"]}')

    if not args.clip_fill:
        if get_confidence:
            return data[args.eval_key][data_key], data[args.eval_key]["2.5d_factor"], data[args.eval_key]['confidence']
        return data[args.eval_key][data_key], data[args.eval_key]["2.5d_factor"]
    else:
        source = data[args.eval_key]['source']
        MB_clip_id = []
        k = 0
        for i in range(len(source)):  # MB clips each data into 243 frame segments, the last segment (<243) is discarded
            k += 1
            if k == args.MB_data_stride:
                k = 0
                good_id = list(range(i-args.MB_data_stride+1, i+1))
                MB_clip_id.extend(good_id)
            if i == len(source)-1:
                break
            if source[i] != source[i+1]:
                k = 0
        # dict_keys(['joint_2d', 'confidence', 'joint3d_image', 'joints_2.5d_image', '2.5d_factor', 'camera_name', 'action', 'source', 'c3d_frame'])
        np_pose = data[args.eval_key][data_key][MB_clip_id]
        factor_25d = data[args.eval_key]['2.5d_factor'][MB_clip_id]
        confidence_score = data[args.eval_key]['confidence'][MB_clip_id] if get_confidence else None
        # camera_name_store = ''
        # for n in range(100000):
        #     if data[args.eval_key]['camera_name'][n] != camera_name_store:
        #         print(n)
        #         print(data[args.eval_key]['action'][n])
        #         print(data[args.eval_key]['camera_name'][n])
        #         print()
        #         camera_name_store = data[args.eval_key]['camera_name'][n]
        if get_confidence:
            return np_pose, factor_25d, confidence_score
        return np_pose, factor_25d


def check_GT_file(args):
    with open(args.GT_file, "rb") as f:
        data = pickle.load(f)
    eval_keys = ['train', 'test', 'validate']
    args.eval_key = eval_keys[1]
    GT_pose = data[args.eval_key]['joint3d_image']

    # check for hands
    frame = 0
    print(f"Dist to middle base {np.linalg.norm(GT_pose[frame,5] - GT_pose[frame,0])} px")
    print(f"{(np.linalg.norm(GT_pose[frame, 5] - GT_pose[frame, 0])) * data[args.eval_key]['2.5d_factor'][frame]} mm")

    i = 0
    j = 0
    for frame in range(1, len(GT_pose)-1):
        frame_source = data[args.eval_key]['source'][frame]
        next_source = data[args.eval_key]['source'][frame+1]

        ## check 2: travel between frames
        travel = np.linalg.norm(GT_pose[frame,5] - GT_pose[frame + 1,5])
        if travel > 11.18*1000/20 and frame_source == next_source:  # 11.18 m/s boxer punch speed, 100 fps
            j += 1
            print(f"j-{j} - ave travel between frame {frame} &+1: {travel}")

        ## check 1: middle finger dist
        px_dist = np.linalg.norm(GT_pose[frame, 5] - GT_pose[frame, 0])
        mm_dist = px_dist * data[args.eval_key]['2.5d_factor'][frame]
        next_px_dist = np.linalg.norm(GT_pose[frame+1, 5] - GT_pose[frame+1, 0])
        next_mm_dist = next_px_dist * data[args.eval_key]['2.5d_factor'][frame+1]
        if abs(mm_dist-next_mm_dist)  > next_mm_dist*0.2 and frame_source == next_source:
            # print(f"Dist to middle base in frame {frame} &+1: {px_dist} - {next_px_dist} = {px_dist-next_px_dist} px")
            print(f"Dist to middle base in frame {frame} &+1: {mm_dist} - {next_mm_dist} = {mm_dist-next_mm_dist} mm")

        ## check 2.5
        if data[args.eval_key]['2.5d_factor'][frame]-data[args.eval_key]['2.5d_factor'][frame+1] > 3 and frame_source == next_source:
            i+=1
            print(f"i-{i} - 2.5d factor in frame {frame} &+1: {data[args.eval_key]['2.5d_factor'][frame]} - {data[args.eval_key]['2.5d_factor'][frame+1]} = {data[args.eval_key]['2.5d_factor'][frame]-data[args.eval_key]['2.5d_factor'][frame+1]}")


def find_joint_ids(full_list, select_list):
    output = []
    for i in range(len(full_list)):
        if full_list[i] in name_list:
            output.append(i)
    return output

    # frame = 120
    # # frame = 39747
    # # frame = 0
    # GT_skeleton.plot_3d_pose_frame(frame=frame, coord_system="camera-px", plot_range=2000, mode='normal_view', center_key='Wrist')
    # GT_skeleton.plot_3d_pose_frame(frame=frame+1, coord_system="camera-px", plot_range=2000, mode='normal_view', center_key='Wrist')
    # print(data[args.eval_key]['source'][frame])
    # print(data[args.eval_key]['source'][frame+1])
    #

def flip_data(data, args=False):
    """
    horizontal flip
        data: [N, F, 17, D] or [F, 17, D]. X (horizontal coordinate) is the first channel in D.
    Return
        result: same
    """
    if args == False or args.joint_format.upper() == 'H36M':
        left_joints = [4, 5, 6, 11, 12, 13] # Human 3.6
        right_joints = [1, 2, 3, 14, 15, 16]
    elif args.joint_format.upper() == 'RTM-24':
        # todo: update, missing 1, 2
        left_joints = [3, 5, 7, 9, 11, 13,16]
        right_joints = [4, 6, 8, 10, 12, 14,19]
    elif args.joint_format.upper() == 'RTM-37':
        left_joints = [2, 4, 6, 8, 10, 12, 14, 16, 21, 27, 28, 31, 32, 35, 36]
        right_joints = [1, 3, 5, 7, 9, 11, 13, 15, 20, 25, 26, 29, 30, 33, 34]
    elif args.joint_format.upper() == 'HAND-21':
        left_joints = []
        right_joints = []
    elif args.joint_format.upper() == 'UBHAND-48':
        left_joints = [0, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        right_joints = [1, 3, 5, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
    else:
        raise ValueError("args.joint_format not recognized")
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1                                               # flip x of all joints
    flipped_data[..., left_joints+right_joints, :] = flipped_data[..., right_joints+left_joints, :]
    return flipped_data

if __name__ == '__main__':
    # read arguments
    args = parse_args()
    estimate_pose = MB_output_pose_file_loader(args)
    # data_key = 'joint_2d'  # todo: only for 2D plot, maybe move in config
    data_key = 'joint3d_image'
    # GT_pose, factor_25d = MB_input_pose_file_loader(args, data_key=data_key)
    # if args.rescale_25d:
    #     if args.clip_fill:
    #         print(f'rescale by 2.5d factor in GT file')
    #     else:
    #         assert len(factor_25d) == len(estimate_pose), f"Can not rescale without clip_fill: len(factor_25d): {len(factor_25d)}, len(estimate_pose): {len(estimate_pose)}, they should be the same"
    #     estimate_pose = estimate_pose / factor_25d[:, None, None]
    # if args.debug_mode:
    #     small_sample = 11516
    #     # small_sample = 5103  # industry rick
    #     # small_sample = 16560
    #     estimate_pose = estimate_pose[:small_sample]
    #     GT_pose = GT_pose[:small_sample]

    if False:  # for upper body visualization
        # name_list = ['PELVIS',
        #             'RWRIST', 'LWRIST', 'RHIP', 'LHIP',
        #             'RHAND', 'LHAND', 'RELBOW', 'LELBOW', 'RSHOULDER', 'LSHOULDER',
        #             'HEAD', 'THORAX', 'HDTP', 'REAR', 'LEAR', 'C7']
        name_list = ['PELVIS',
                    'RWRIST', 'LWRIST', 'RHIP', 'LHIP',
                    'RHAND', 'LHAND', 'RELBOW', 'LELBOW', 'RSHOULDER', 'LSHOULDER',
                    'HEAD', 'THORAX', 'HDTP', 'REAR', 'LEAR', 'C7', 'C7_d', 'SS',
                    'RAP_b', 'RAP_f', 'LAP_b', 'LAP_f', 'RLE', 'RME', 'LLE', 'LME', 'RMCP2', 'RMCP5', 'LMCP2', 'LMCP5']
        select_list_id = find_joint_ids(args.name_list, name_list)
        estimate_pose = estimate_pose[:, select_list_id]
        args.name_list = name_list
        args.output_frame_folder = args.output_frame_folder + '_ub'

    args.joint_format = 'RTM-37'
    # estimate_pose = flip_data(estimate_pose, args)
    # estimate_pose = flip_data(estimate_pose, args)

    estimate_skeleton = VEHSErgoSkeleton_angles(args.skeleton_file)
    estimate_skeleton.load_name_list_and_np_points(args.name_list, estimate_pose)

    # GT_skeleton = VEHSErgoSkeleton_angles(args.skeleton_file)
    # GT_skeleton.load_name_list_and_np_points(args.name_list, GT_pose)

    if args.type == 'hand':
        # get legend
        # estimate_skeleton.plot_3d_pose(args.output_frame_folder, coord_system="camera-px", plot_range=1e20, mode=args.plot_mode, get_legend=True, center_key='Wrist')

        # estimate_skeleton.plot_3d_pose(args.output_frame_folder, coord_system="camera-px", mode=args.plot_mode, center_key='Middle_0', plot_range=1000)

        # frame = 1000
        # estimate_skeleton.plot_3d_pose_frame(frame=frame, coord_system="camera-px", mode="normal_view", center_key='Middle_0', plot_range=1000)

        # GT_skeleton.plot_2d_pose(args.output_2D_frame_folder, resolution=(1500, 1500), dpi=100)
        # GT_skeleton.plot_3d_pose(args.output_GT_frame_folder, coord_system="camera-px", mode=args.plot_mode, center_key='Middle_0', plot_range=1000)
        #
        frame = 100
        plot_range = 1000
        estimate_skeleton.plot_3d_pose_frame(frame=frame, coord_system="camera-px", mode="normal_view", center_key='Middle_0', plot_range=plot_range)
        GT_skeleton.plot_3d_pose_frame(frame=frame, coord_system="camera-px", mode="camera_view", center_key='Middle_0', plot_range=plot_range)
        # GT_skeleton.plot_3d_pose_frame(frame=frame+1, coord_system="camera-px", mode="normal_view", center_key='Middle_0', plot_range=plot_range)
        # GT_skeleton.plot_3d_pose_frame(frame=frame + 2, coord_system="camera-px", mode="normal_view", center_key='Middle_0', plot_range=plot_range)
    elif args.type == 'body':

        frame = 134*20
        frame = 18066

        estimate_skeleton.plot_3d_pose_frame(frame=frame, coord_system="camera-px", plot_range=700, mode='camera_view', center_key='PELVIS')

        ##### example of plotting 37 keypoints for industry and VEHS7M inference
        # if args.dynamic_plot_range:
        #     segment_count = estimate_skeleton.frame_number // 243
        #     for segment_id in range(segment_count):
        #         start_frame = segment_id * 243
        #         end_frame = start_frame + 243
        #         xyz_min = np.min(estimate_skeleton.points[start_frame:end_frame], axis=(0, 1))
        #         xyz_max = np.max(estimate_skeleton.points[start_frame:end_frame], axis=(0, 1))
        #         plot_range = np.max(xyz_max - xyz_min) * 1.0
        #         estimate_skeleton.plot_3d_pose(args.output_frame_folder, start_frame=start_frame, end_frame=end_frame,
        #                                        coord_system="camera-px", plot_range=plot_range, mode=args.plot_mode, center_key='PELVIS')
        # else:
        #     plot_range = 850  # for VEHS7M - camera_side_view
        #     # estimate_skeleton.plot_3d_pose(args.output_frame_folder, start_frame=0, coord_system="camera-px", plot_range=plot_range, mode=args.plot_mode, center_key='PELVIS')
        #
        #     GT_skeleton.plot_3d_pose(args.output_GT_frame_folder, start_frame=1190, coord_system="camera-px", plot_range=plot_range, mode=args.plot_mode, center_key='PELVIS')
        #
        # ###### example of plotting h36M 17 keypoints
        # # estimate_skeleton.plot_3d_pose(args.output_frame_folder, coord_system="camera-px", plot_range=1200, mode=args.plot_mode, center_key='HIP_c')
        # # estimate_skeleton.plot_3d_pose_frame(frame=frame, coord_system="camera-px", plot_range=700, mode='paper_view', center_key='PELVIS')
        # # GT_skeleton.plot_3d_pose_frame(frame=frame, coord_system="camera-px", plot_range=700, mode='paper_view', center_key='HIP_c')
        #
        # ##### example of overlaying 2D image
        # # baseimage = os.path.join('/Users/leyangwen/Downloads/S9/Videos/50fps/Directions 1.54138969', f'{frame + 1:05d}.png')
        # # GT_skeleton.plot_2d_pose_frame(frame=frame, baseimage=baseimage)
        #
        # ###### example of plotting 2D with transparent background
        # # GT_skeleton.plot_2d_pose_frame(frame=frame)
        # # GT_skeleton.plot_2d_pose(foldername=args.output_2D_frame_folder)
        #
        #
        # ###### example to get legend
        # # estimate_skeleton.plot_3d_pose(args.output_frame_folder, coord_system="camera-px", plot_range=1e20, mode=args.plot_mode, get_legend=True, center_key='PELVIS')
        # # GT_skeleton.plot_3d_pose(args.output_frame_folder, coord_system="camera-px", plot_range=1e20, mode=args.plot_mode, get_legend=True, center_key='HIP_c')



# store = ""
# frame_no = 0
# for i, s in enumerate(source):
#     frame_no += 1
#     if s != store:
#         print(i, s, frame_no//50)
#         store = s
#         frame_no = 0
#
# 2356/50


