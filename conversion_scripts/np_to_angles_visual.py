import argparse
import os.path
import pickle

from Skeleton import *
from scipy.stats import f_oneway
from MB_np_to_visual import MB_input_pose_file_loader
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', type=str, default=r'config/experiment_config/37kpts/Inference-RTMPose-MB-20fps-industry.yaml')  # config/experiment_config/VEHS-6D-MB.yaml') #
    parser.add_argument('--overlay_GT', type=bool, default=False)
    parser.add_argument('--debug_mode', default=False)
    parser.add_argument('--clip_fill', type=bool, default=False)

    # parser.add_argument('--config_file', type=str, default=r'config/experiment_config/37kpts/Inference-RTMPose-MB-20fps-VEHS7M.yaml')
    # parser.add_argument('--overlay_GT', type=bool, default=True)
    # parser.add_argument('--debug_mode', default=True)
    # parser.add_argument('--clip_fill', type=bool, default=True)

    parser.add_argument('--skeleton_file', type=str, default=r'config/VEHS_ErgoSkeleton_info/Ergo-Skeleton-37.yaml')
    parser.add_argument('--MB_data_stride', type=int, default=243)
    parser.add_argument('--merge_lr', default=True)

    # parser.add_argument('--name_list', type=list, default=[])
    args = parser.parse_args()
    with open(args.config_file, 'r') as stream:
        data = yaml.safe_load(stream)
        args.name_list = data['name_list']
        args.GT_file = data['GT_file']
        args.eval_key = data['eval_key']
        args.estimate_file = data['estimate_file']
        args.output_dir = os.path.join(os.path.dirname(args.estimate_file), 'angles')
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    return args


def MB_output_pose_file_loader(args):
    with open(args.estimate_file, "rb") as f:
        output_np_pose = np.load(f)
    np_pose_shape = output_np_pose.shape
    output_np_pose = output_np_pose.reshape(-1, np_pose_shape[-2], np_pose_shape[-1])
    return output_np_pose


if __name__ == '__main__':
    # read arguments
    args = parse_args()
    now = datetime.now()
    print(f"Start Time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    if args.eval_key == 'dict':
        estimate_pose = []
        GT_pose = []
        for key, item in args.estimate_file.items():
            args.eval_key = key
            args.estimate_file = item
            this_estimate_pose = MB_output_pose_file_loader(args)
            estimate_pose.append(this_estimate_pose)

            this_gt_pose, _ = MB_input_pose_file_loader(args)
            GT_pose.append(this_gt_pose)
        estimate_pose = np.concatenate(estimate_pose, axis=0)
        GT_pose = np.concatenate(GT_pose, axis=0)
        print(estimate_pose.shape, GT_pose.shape)
        assert estimate_pose.shape == GT_pose.shape, f"GT_pose.shape: {GT_pose.shape}, estimate_pose.shape: {estimate_pose.shape}, they should be the same"
    else:
        estimate_pose = MB_output_pose_file_loader(args)
        GT_pose, _ = MB_input_pose_file_loader(args)
        if args.overlay_GT:
            assert GT_pose.shape == estimate_pose.shape, f"GT_pose.shape: {GT_pose.shape}, estimate_pose.shape: {estimate_pose.shape}, they should be the same"

    if args.debug_mode:
        small_sample = 1200 #243*6
        GT_pose = GT_pose[:small_sample] if GT_pose is not None else None
        estimate_pose = estimate_pose[:small_sample]

    if args.overlay_GT:
        # Step 1: calculate GT angles
        GT_skeleton = VEHSErgoSkeleton_angles(args.skeleton_file)
        GT_skeleton.load_name_list_and_np_points(args.name_list, GT_pose)
        GT_ergo_angles = {}
        for angle_name in GT_skeleton.angle_names:  # calling the angle calculation methods in skeleton class
            class_method_name = f'{angle_name}_angles'
            GT_ergo_angles[angle_name] = getattr(GT_skeleton, class_method_name)()

    # Step 2: calculate MB angles
    estimate_skeleton = VEHSErgoSkeleton_angles(args.skeleton_file)
    estimate_skeleton.load_name_list_and_np_points(args.name_list, estimate_pose)

    estimate_ergo_angles = {}
    for angle_name in estimate_skeleton.angle_names:  # calling the angle calculation methods in skeleton class
        class_method_name = f'{angle_name}_angles'
        # print(class_method_name)
        estimate_ergo_angles[angle_name] = getattr(estimate_skeleton, class_method_name)()
    # Step 3: visualize
    # frame = 10210
    # frame = 10180
    # GT_skeleton.plot_3d_pose_frame(frame)
    # estimate_skeleton.plot_3d_pose_frame(frame)

    frame_range = None #[0, 60*3*20]
    # target_angles = estimate_skeleton.angle_names  # ['neck', 'right_shoulder', 'left_shoulder', 'right_elbow', 'left_elbow', 'right_wrist', 'left_wrist', 'back', 'right_knee', 'left_knee']
    # target_angles = ['neck', 'right_shoulder', 'left_shoulder']
    # target_angles = ['right_elbow', 'left_elbow', 'right_wrist', 'left_wrist']

    target_angles = ['back', 'right_knee', 'left_knee']

    print(f"target angles: {target_angles}")
    # Single thread
    for angle_index, this_angle_name in enumerate(target_angles):
        # plot angles in single image
        # GT_fig, GT_ax = GT_ergo_angles[this_angle_name].plot_angles(joint_name=f"GT-{this_angle_name}", frame_range=frame_range, alpha=0.75, colors=['g', 'g', 'g'])
        # estimate_fig, _ = estimate_ergo_angles[this_angle_name].plot_angles(joint_name=f"Est-{this_angle_name}", frame_range=frame_range, alpha=0.75, colors=['r', 'r', 'r'], overlay=[GT_fig, GT_ax])
        # plt.show()
        # # GT_fig.savefig(f'frames/MB_angles/GT-{this_angle_name}.png')
        # estimate_fig.savefig(f'frames/MB_angles/Est-{this_angle_name}.png')

        # plot angles as video frames
        print_ergo_names = getattr(estimate_ergo_angles[this_angle_name], 'ergo_name')
        render_dir = os.path.join(args.output_dir, f'{this_angle_name}')
        if args.overlay_GT:
            frame_range_max = 1200  # 1 min, around 5 243 clips
            estimate_ergo_angles[this_angle_name].plot_angles_by_frame(render_dir, joint_name=f"{this_angle_name}", alpha=0.75, colors=['r', 'r', 'r'], frame_range=frame_range, frame_range_max=frame_range_max, label="Inference",
                                                                       angle_names=list(print_ergo_names.values()), overlay=GT_ergo_angles[this_angle_name], overlay_colors=['g', 'g', 'g'], fps=20, x_tick_s=15)
        else:
            skip_first = False
            # frame_range_max = 243
            if "VEHS7M" in render_dir:
                frame_range_max = None
            elif "Industry/angles" in render_dir:
                frame_range_max = list(np.array([2,2,1,2,1,2,2,2,1,2,2,2])*243)  # industry
            elif "Industry_2" in render_dir:
                frame_range_max = list(np.array([11, 2, 9, 7, 7, 7, 3, 7, 22, 4, 17]) * 243)  # industry #2
            elif "Industry_both" in render_dir:
                frame_range_max = list(np.array([2,2,1,2,1,2,2,2,1,2,2,2,11, 2, 9, 7, 7, 7, 3, 7, 22, 4, 17]) * 243)  # industry #2

                ## only 12 rick videos
                frame_range_max = list(np.array([2,2,1,2,1,2,2,2,1,2,2,2])*243)

                # ## only 11 long industry videos
                # frame_range_max = list(np.array([21,11, 2, 9, 7, 7, 7, 3, 7, 22, 4, 17]) * 243)  # industry #2
                # skip_first = True
            else:
                pass
                raise ValueError(f"Make sure this is the right dataset")
            estimate_ergo_angles[this_angle_name].plot_angles_by_frame(render_dir, joint_name=f"{this_angle_name}", alpha=0.75, colors=['r', 'r', 'r'], frame_range=frame_range, frame_range_max=frame_range_max, skip_first=skip_first,
                                                                       angle_names=list(print_ergo_names.values()), fps=20, x_tick_s=2)

