import argparse
import os.path
import pickle
from Skeleton import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=r'config/experiment_config/VEHS-Hand-21-MB.yaml')
    parser.add_argument('--skeleton_file', type=str, default=r'config/VEHS_ErgoSkeleton_info/Ergo-Hand-21.yaml')
    # parser.add_argument('--config_file', type=str, default=r'config/experiment_config/Inference-RTMPose-MB.yaml')  #-pitch-corrected.yaml')
    # parser.add_argument('--skeleton_file', type=str, default=r'config/VEHS_ErgoSkeleton_info/RTMPose-Skeleton.yaml')
    parser.add_argument('--output_frame_folder', type=str, default=None)

    parser.add_argument('--output_GT_frame_folder', type=str, default=None)
    parser.add_argument('--plot_mode', type=str, default='0_135_view', help='mode: camera_view, camera_side_view, 0_135_view, normal_view')
    parser.add_argument('--MB_data_stride', type=int, default=243)
    parser.add_argument('--debug_mode', default=False, type=bool)


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
    args.output_GT_frame_folder = os.path.dirname(args.GT_file) if args.output_GT_frame_folder is None else args.output_GT
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


def MB_input_pose_file_loader(args):
    if args.GT_file=='None':
        return None
    with open(args.GT_file, "rb") as f:
        data = pickle.load(f)

    print(f'2.5d_factor: {data[args.eval_key]["2.5d_factor"]}')

    clip_fill = False
    if not clip_fill:
        return data[args.eval_key]['joint3d_image']
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
        np_pose = data[args.eval_key]['joint3d_image'][MB_clip_id]
        camera_name_store = ''
        for n in range(100000):
            if data[args.eval_key]['camera_name'][n] != camera_name_store:
                print(n)
                print(data[args.eval_key]['action'][n])
                print(data[args.eval_key]['camera_name'][n])
                print()
                camera_name_store = data[args.eval_key]['camera_name'][n]
        return np_pose


if __name__ == '__main__':
    # read arguments
    args = parse_args()
    estimate_pose = MB_output_pose_file_loader(args)
    GT_pose = MB_input_pose_file_loader(args)

    if args.debug_mode:
        small_sample = 7252
        # estimate_pose = estimate_pose[:small_sample]
        # GT_pose = GT_pose[:small_sample]

    frame = int(12263/5)
    estimate_skeleton = VEHSErgoSkeleton_angles(args.skeleton_file)
    estimate_skeleton.load_name_list_and_np_points(args.name_list, estimate_pose)
    # estimate_skeleton.plot_3d_pose(args.output_frame_folder, coord_system="camera-px", plot_range=1e20, mode=args.plot_mode, get_legend=True, center_key='Wrist')
    # estimate_skeleton.plot_3d_pose(args.output_frame_folder, coord_system="camera-px", plot_range=1000, mode=args.plot_mode, center_key='PELVIS')
    # estimate_skeleton.plot_3d_pose_frame(frame=frame, coord_system="camera-px", mode="normal_view", center_key='PELVIS', plot_range=2000)

    GT_skeleton = VEHSErgoSkeleton_angles(args.skeleton_file)
    GT_skeleton.load_name_list_and_np_points(args.name_list, GT_pose)
    # GT_skeleton.plot_3d_pose(args.output_GT_frame_folder, coord_system="camera-px", plot_range=1000, mode=args.plot_mode)
    #
    GT_skeleton.plot_3d_pose_frame(frame=frame, coord_system="camera-px", plot_range=2000, mode='normal_view', center_key='Wrist')
    # GT_skeleton.plot_3d_pose_frame(frame=frame, coord_system="camera-px", mode="normal_view", center_key='PELVIS', plot_range=1000)




