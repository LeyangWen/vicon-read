from ergo3d import *
from Skeleton import *
import pickle
import yaml
import argparse

# Not implemented yet

def append_output_xD_dataset(output_xD_dataset, this_train_val_test, append_outputxD_dict):
    for key in output_xD_dataset[this_train_val_test].keys():
        if key == 'source' or key == 'c3d_frame':
            output_xD_dataset[this_train_val_test][key] = output_xD_dataset[this_train_val_test][key] + append_outputxD_dict[key]
        else:
            output_xD_dataset[this_train_val_test][key] = np.append(output_xD_dataset[this_train_val_test][key], append_outputxD_dict[key], axis=0)
    return output_xD_dataset


if __name__ == '__main__':
    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_config_file', type=str, default=r'config/experiment_config/VEHS-R3-622-MotionBert.yaml')
    parser.add_argument('--skeleton_file', type=str, default=r'config\VEHS_ErgoSkeleton_info\Ergo-Skeleton-66.yaml')
    parser.add_argument('--downsample', type=int, default=5)
    parser.add_argument('--downsample_keep', type=int, default=1)
    parser.add_argument('--split_output', action='store_false')  # not implemented yet
    parser.add_argument('--output_type', type=list, default=[True, False, False, False], help='3D, 6D, SMPL, 3DSSPP')
    parser.add_argument('--distort', action='store_false', help='consider camera distortion in the output 2D pose')
    args = parser.parse_args()


    # base_folder = r'C:\Users\Public\Documents\Vicon\data\Vicon_F\Round3\LeyangWen'
    split_config_file = args.split_config_file
    with open(split_config_file, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
            if 'base_folder' in data:
                base_folders = [data['base_folder']]
            else:
                base_folders = data['base_folders']
                base_folders.sort()
            base_folder = base_folders[-1]
            print('base_folder', base_folder, 'from', base_folders)
            # base_folder = os.path.join(base_folder, 'LeyangWen')  # for testing
            val_keyword = data['val_keyword']
            test_keyword = data['test_keyword']
        except yaml.YAMLError as exc:
            print(split_config_file, exc)
            raise ValueError


    skeleton_file = args.skeleton_file


    # iterate through the folder to find all c3d
    # h36m_joint_names = ['Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine', 'Thorax', 'Neck/Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']  # h36m original names
    h36m_joint_names = ['PELVIS', 'RHIP', 'RKNEE', 'RANKLE', 'LHIP', 'LKNEE', 'LANKLE', 'T8', 'THORAX', 'C7', 'HEAD', 'LSHOULDER', 'LELBOW', 'LWRIST', 'RSHOULDER', 'RELBOW', 'RWRIST']
    output_3D_dataset = empty_MotionBert_dataset_dict(len(h36m_joint_names))  # 17
    custom_6D_joint_names = ['RPSIS', 'RASIS', 'LPSIS', 'LASIS', 'C7_d', 'SS', 'T8', 'XP', 'C7', 'HDTP', 'REAR', 'LEAR', 'RAP', 'RAP_f', 'RLE', 'RAP_b', 'RME', 'LAP', 'LAP_f', 'LLE', 'LAP_b', 'LME', 'LUS', 'LRS', 'RUS', 'RRS', 'RMCP5', 'RMCP2', 'LMCP5', 'LMCP2', 'LGT', 'LMFC', 'LLFC', 'RGT', 'RMFC', 'RLFC', 'RMM', 'RLM', 'LMM', 'LLM', 'LMTP1', 'LMTP5', 'LHEEL', 'RMTP1', 'RMTP5', 'RHEEL', 'HEAD', 'RSHOULDER', 'LSHOULDER', 'C7_m', 'THORAX', 'LELBOW', 'RELBOW', 'RWRIST', 'LWRIST', 'RHAND', 'LHAND', 'PELVIS', 'RHIP', 'RKNEE', 'RANKLE', 'RFOOT', 'LHIP', 'LKNEE', 'LANKLE', 'LFOOT']
    output_6D_dataset = empty_MotionBert_dataset_dict(len(custom_6D_joint_names))  # 66
    output_smpl_dataset = {}
    count = 0
    pkl_filenames = {'3D': [], '6D': [], 'SMPL': []}  # if split_output, save intermediate results
    dataset_statistics = {}
    total_frame_number = 0
    for base_folder in base_folders:
        for root, dirs, files in os.walk(base_folder):
            for file in files:
                if (
                        file.endswith('.c3d')
                        and not root.endswith('backup')
                        and not file.endswith('_bad.c3d')
                        and (file.startswith('Activity') or file.startswith('activity'))
                ):
                    # val_keyword is a list of string, if any of them is in the root, then it is val set
                    if any(keyword in root for keyword in val_keyword):
                        train_val_test = 'validate'
                    elif any(keyword in root for keyword in test_keyword):
                        train_val_test = 'test'
                    else:
                        train_val_test = 'train'

                    c3d_file = os.path.join(root, file)
                    count += 1
                    # if count > 1:  # give a very small file for testing
                    #     if train_val_test == 'train':
                    #         train_val_test = 'test'
                    #         count = 0
                    #     else:
                    #         break

                    print(f'{count}: Starting on {c3d_file} as {train_val_test} set')
                    frames = np.linspace(start_frame / fps_ratio, end_frame / fps_ratio, int((end_frame - start_frame) / fps_ratio), dtype=int)
                    world3D_filename = os.path.join(cdf_output_dir, '3D_Pose_World', activity_name, f'{activity_name}_{rep}.world.cdf')
                    store_cdf(world3D_filename, world3D, TaskID=activity_name, kp_names=kpt_names)
                    # world3D_skeleton = VEHSErgoSkeleton(r'config\VEHS_ErgoSkeleton_info\Ergo-Skeleton.yaml')
                    # world3D_skeleton.load_name_list_and_np_points(kpt_names, world3D)
                    # world3D_skeleton.plot_3d_pose(os.path.join(frame_output_dir, '3D_Pose_World'))

                    xcp_filename = c3d_file.replace('.c3d', '.xcp')
                    cameras = batch_load_from_xcp(xcp_filename)
                    for cam_idx, camera in enumerate(cameras):
                        print(f'Processing camera {cam_idx}: {camera.DEVICEID}')

                        points_2d_list = []
                        points_3d_camera_list = []
                        points_2d_bbox_list = []
                        for frame_idx, frame_no in enumerate(frames):
                            frame_idx = int(frame_idx * fps_ratio)  # todo: bug if fps_ratio is not an 1
                            print(f'Processing frame {frame_no}/{frames[-1]} of {activity_name}.{camera.DEVICEID}.timestamp.avi',
                                  end='\r')
                            points_3d = world3D[frame_idx, :, :].reshape(-1, 3) / 1000
                            points_3d_camera = camera.project_w_depth(points_3d)  # todo: test if transpose is still needed
                            points_2d = camera.project(points_3d)
                            points_2d = camera.distort(points_2d)
                            bbox_top_left, bbox_bottom_right = points_2d.min(axis=0) - 20, points_2d.max(axis=0) + 20
                            points_2d_list.append(points_2d)
                            points_3d_camera_list.append(points_3d_camera)
                            points_2d_bbox_list.append([bbox_top_left, bbox_bottom_right])

                        points_2d_list = np.array(points_2d_list)
                        # points_3d_camera_list = np.swapaxes(np.array(points_3d_camera_list), 1, 2)
                        world2D_skeleton = VEHSErgoSkeleton(r'config\VEHS_ErgoSkeleton_info\Ergo-Skeleton.yaml')
                        world2D_skeleton.load_name_list_and_np_points(kpt_names, points_2d_list)
                        world2D_skeleton.plot_2d_pose(os.path.join(frame_output_dir, f'2D_Pose_Camera{camera.DEVICEID}'))

                    raise NotImplementedError  # break for testing
                    # del this_skeleton
                    # del ergo_angles

