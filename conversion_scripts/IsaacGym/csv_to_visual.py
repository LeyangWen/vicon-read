import argparse
import os.path
import pickle
from Skeleton import *
import matplotlib
# matplotlib.use('Qt5Agg')
from mpl_toolkits.mplot3d import Axes3D
import csv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=r'config/experiment_config/IssacGym/task_eval.yaml')
    parser.add_argument('--skeleton_file', type=str, default=r'config/VEHS_ErgoSkeleton_info/IssacGym/15kpts-Skeleton.yaml')


    parser.add_argument('--output_frame_folder', type=str, default=None)
    parser.add_argument('--output_GT_frame_folder', type=str, default=None)
    parser.add_argument('--plot_mode', type=str, default='global_view', help='mode: camera_view, camera_side_view, 0_135_view, normal_view, paper_view, global_view')
    parser.add_argument('--MB_data_stride', type=int, default=243)
    parser.add_argument('--debug_mode', default=True, type=bool)


    # parser.add_argument('--name_list', type=list, default=[])
    args = parser.parse_args()
    with open(args.config_file, 'r') as stream:
        data = yaml.safe_load(stream)
        args.name_list = data['name_list']
        args.estimate_file = data['estimate_file']
    print(args.plot_mode)
    args.output_frame_folder = os.path.dirname(args.estimate_file) if args.output_frame_folder is None else args.output_frame_folder
    args.output_frame_folder = os.path.join(args.output_frame_folder, args.plot_mode)
    return args



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

    frames = list(range(4,	129, 3))

    for frame in frames:
        # frame -= 3
        # isaac_skeleton.plot_3d_pose_frame(frame=frame, coord_system="world-m", plot_range=10, mode=args.plot_mode, center_key='PELVIS', plot_rot=True, title='Frame {}'.format(frame))
        isaac_skeleton.plot_3d_pose_frame(frame=frame, coord_system="world-m", plot_range=2, mode="camera_view", center_key='PELVIS', plot_rot=True, title='Frame {}'.format(frame))

    # issac_skeleton.plot_3d_pose(args.output_frame_folder, coord_system="world-m", plot_range=2, mode=args.plot_mode, center_key='PELVIS')

    # get legend
    # issac_skeleton.plot_3d_pose(args.output_frame_folder, coord_system="camera-px", plot_range=1e20, mode=args.plot_mode, get_legend=True, center_key='PELVIS')


    frame = 29
    print(isaac_skeleton.point_poses.keys())
    print(isaac_skeleton.point_poses['head'].z[frame])
    print(isaac_skeleton.point_poses['left_foot'].z[frame])
    print(isaac_skeleton.point_poses['right_foot'].z[frame])

    print(isaac_skeleton.point_poses['right_upper_arm'].z[frame]-isaac_skeleton.point_poses['right_hand'].z[frame])
    print(isaac_skeleton.point_poses['left_upper_arm'].z[frame]-isaac_skeleton.point_poses['left_hand'].z[frame])

    if False: # quick visualization of copied rigid body pose frame
        rigid_body_pose = np.array(
            [
                [[-3.9588, -3.6682, 0.9103],
                 [-3.9055, -3.7026, 1.1378],
                 [-3.7298, -3.8079, 1.2280],
                 [-3.8232, -3.9591, 1.2821],
                 [-3.9857, -4.0467, 1.0786],
                 [-3.7779, -4.0445, 0.9240],
                 [-3.6184, -3.6594, 1.2332],
                 [-3.5769, -3.5418, 0.9884],
                 [-3.5704, -3.7893, 0.9126],
                 [-4.0238, -3.7061, 0.8259],
                 [-4.0882, -3.7831, 0.4546],
                 [-4.1992, -3.7253, 0.0737],
                 [-3.9422, -3.5959, 0.8225],
                 [-3.7871, -3.6508, 0.4835],
                 [-3.7059, -3.7623, 0.1074]]
            ])

        # # add 3 more points, (0,0,0), (0,0,1), (0,0,2)
        # rigid_body_pose = np.concatenate([rigid_body_pose, np.zeros((1, 3, 3))], axis=1)
        # rigid_body_pose[0, -2, -1] = 0.5
        # rigid_body_pose[0, -1, -1] = 1

        estimate_skeleton = VEHSErgoSkeleton_angles(args.skeleton_file)
        estimate_skeleton.load_name_list_and_np_points(args.name_list[:-3], rigid_body_pose)
        estimate_skeleton.plot_3d_pose_frame(frame=0, coord_system="world-m", plot_range=2, center_key='PELVIS')






