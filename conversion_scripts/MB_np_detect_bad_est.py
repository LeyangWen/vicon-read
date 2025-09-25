import argparse
import os.path
import pickle
import numpy as np
from Skeleton import *
import matplotlib
import copy
# matplotlib.use('Qt5Agg')
from MB_np_to_visual import MB_input_pose_file_loader, MB_output_pose_file_loader, flip_data

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



def center_z_score(pose, args=False):
    """
    only for VEHS-37 kpts, enforce shoulder center to be in center of back & front of shoulder, same for elbow and wrist
    Input: (T, 37, 3)
    """
    if args.joint_format == 'RTM-37':
        # Indices in rtm_pose_37_keypoints_vicon_dataset_v1 (zero-based)
        # Centers: RSHOULDER, LSHOULDER, RELBOW, LELBOW, RHAND, LHAND
        center_ids = [15, 16, 13, 14, 11, 12]
        # Supporting pairs:
        # Shoulders: RAP_b/RAP_f, LAP_b/LAP_f
        # Elbows: RLE/RME, LLE/LME
        # Hands: RMCP2/RMCP5, LMCP2/LMCP5
        support_pairs = [
            (25, 26),  # R shoulder supports
            (27, 28),  # L shoulder supports
            (29, 30),  # R elbow supports
            (31, 32),  # L elbow supports
            (33, 34),  # R hand supports
            (35, 36),  # L hand supports
        ]

        # Select predicted and GT points
        # Gather both support points for all pairs → (T, K, 2, 3)
        pred_support_pts = pose[:, support_pairs, 2]    # (T, K, 2)
        pred_mid_pts = pose[:, center_ids, 2]          # (T, K)

        # Compute midpoint along the "2" axis
        pred_support_mid = pred_support_pts.mean(axis=2) # (T, K)
        score = np.abs(pred_mid_pts - pred_support_mid)  # (T, K)
        return score
    else:
        raise NotImplementedError

def arm_bbox_diag_3d(pose, args=False):
    """
    only for VEHS-37 kpts,
    Input: (T, 37, 3)
    """
    if args.joint_format == 'RTM-37':
        # Indices in rtm_pose_37_keypoints_vicon_dataset_v1 (zero-based)
        # Centers: RSHOULDER, LSHOULDER, RELBOW, LELBOW, RHAND, LHAND
        left_arm_id = [16, 14, 12]
        right_arm_id = [15, 13, 11]
        # left arm 3D bbox, min
        left_arm_bbox = pose[:, left_arm_id, :]
        right_arm_bbox = pose[:, right_arm_id, :]
        left_arm_min = left_arm_bbox.min(axis=1)  # (T, 3)
        left_arm_max = left_arm_bbox.max(axis=1)
        right_arm_min = right_arm_bbox.min(axis=1)
        right_arm_max = right_arm_bbox.max(axis=1)
        left_arm_diag = np.linalg.norm(left_arm_max - left_arm_min, axis=1)
        right_arm_diag = np.linalg.norm(right_arm_max - right_arm_min, axis=1)

        return (left_arm_diag, right_arm_diag)
    else:
        raise NotImplementedError

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


    args.joint_format = 'RTM-37'
    # estimate_pose = flip_data(estimate_pose, args)
    # estimate_pose = flip_data(estimate_pose, args)

    estimate_skeleton = VEHSErgoSkeleton_angles(args.skeleton_file)
    estimate_skeleton.load_name_list_and_np_points(args.name_list, estimate_pose)

    # GT_skeleton = VEHSErgoSkeleton_angles(args.skeleton_file)
    # GT_skeleton.load_name_list_and_np_points(args.name_list, GT_pose)

    rtm_pose_37_keypoints_vicon_dataset_v1 = ['PELVIS', 'RWRIST', 'LWRIST', 'RHIP', 'LHIP', 'RKNEE', 'LKNEE', 'RANKLE', 'LANKLE', 'RFOOT', 'LFOOT', 'RHAND', 'LHAND', 'RELBOW', 'LELBOW', 'RSHOULDER',
                                              'LSHOULDER', 'HEAD', 'THORAX', 'HDTP', 'REAR', 'LEAR', 'C7', 'C7_d', 'SS', 'RAP_b', 'RAP_f', 'LAP_b', 'LAP_f', 'RLE', 'RME', 'LLE', 'LME', 'RMCP2',
                                              'RMCP5',
                                              'LMCP2', 'LMCP5']
    score = center_z_score(estimate_skeleton.np_points, args)
    threshold = 20  # px
    bad_frame_ids = np.where((score > threshold).any(axis=1))[0]

    left_arm_diag, right_arm_diag = arm_bbox_diag_3d(estimate_skeleton.np_points, args)
    normalized_score = score.copy()
    normalized_score[:,::2] = score[:, ::2] / (right_arm_diag[:, None]+1e-7)
    normalized_score[:,1::2] = score[:, 1::2] / (left_arm_diag[:, None]+1e-7)
    normalized_threshold = 0.3  # ratio
    normalized_bad_frame_ids = np.where((normalized_score > normalized_threshold).any(axis=1))[0]




