# c3d_to_MB.py

import c3d

from Skeleton import *
import pickle
import yaml
import argparse
import matplotlib
matplotlib.use('Qt5Agg')

# /W/VEHS/VEHS data collection Round 2/round 2/M-2022-09-26/Rokoko/scene-1/

if __name__ == '__main__':
    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_config_file', type=str, default=r'config/experiment_config/Rokoko-Hand-21-1433-MotionBert.yaml')
    parser.add_argument('--skeleton_file', type=str, default=r"config/VEHS_ErgoSkeleton_info/Ergo-Hand-21.yaml")
    parser.add_argument('--downsample', type=int, default=5)
    parser.add_argument('--downsample_keep', type=int, default=1)
    parser.add_argument('--output_file_name_end', type=str, default='')
    args = parser.parse_args()


    # base_folder = r'C:\Users\Public\Documents\Vicon\data\Vicon_F\Round3\LeyangWen'
    split_config_file = args.split_config_file
    with open(split_config_file, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
            base_folder = data['base_folder']
            # base_folder = os.path.join(base_folder, 'LeyangWen')  # for testing
            val_keyword = data['val_keyword']
            test_keyword = data['test_keyword']
        except yaml.YAMLError as exc:
            print(split_config_file, exc)


    skeleton_file = args.skeleton_file
    downsample = args.downsample
    downsample_keep = args.downsample_keep
    output_3d_filename = os.path.join(base_folder, f'Rokoko_hand_3D_downsample{downsample}_keep{downsample_keep}{args.output_file_name_end}.pkl')
    # iterate through the folder to find all c3d

    output_3D_dataset = empty_MotionBert_dataset_dict(21)
    count = 0
    dataset_statistics = {}
    total_frame_number = 0
    for root, dirs, files in os.walk(base_folder):
        dirs.sort()  # Sort directories in-place --> important, will change walk sequence
        files.sort(key=str.lower)  # Sort files in-place
        for file in files:
            if not file.endswith('.csv'):
                continue
            # val_keyword is a list of string, if any of them is in the root, then it is val set
            if any(keyword in root for keyword in val_keyword):
                train_val_test = 'validate'
            elif any(keyword in root for keyword in test_keyword):
                train_val_test = 'test'
            else:
                train_val_test = 'train'
            csv_file = os.path.join(root, file)
            count += 1
            print(f'{count}: Starting on {csv_file} as {train_val_test} set')
            if True:  # print out all found files first to debug
                continue
            for handiness in ['left', 'right']:
                this_skeleton = RokokoHandSkeleton(skeleton_file)
                this_skeleton.load_rokoko_csv(csv_file=csv_file, handiness=handiness, random_rotation=True, flip_left=True)
                this_frame_number = this_skeleton.frame_number
                dataset_statistics[csv_file] = this_frame_number
                total_frame_number += this_frame_number

                this_skeleton.calculate_isometric_projection(args, rootIdx=0)
                output3D = this_skeleton.output_MotionBert_pose(downsample=downsample, downsample_keep=downsample_keep)
                output_3D_dataset = append_output_xD_dataset(output_3D_dataset, train_val_test, output3D)

                if False:  # vis debug
                    frame = 1
                    this_skeleton.plot_3d_pose_frame(frame, plot_range=300, coord_system="world", center_key='Wrist', mode='normal_view')
                    this_skeleton.load_name_list_and_np_points(this_skeleton.point_labels, this_skeleton.pose_3d_camera['XY'])
                    this_skeleton.plot_3d_pose_frame(frame, plot_range=0.3, coord_system="world", center_key='Wrist', mode='normal_view')
                    this_skeleton.load_name_list_and_np_points(this_skeleton.point_labels, this_skeleton.pose_3d_image['XY'])
                    this_skeleton.plot_3d_pose_frame(frame, plot_range=1000, coord_system="world", center_key='Wrist', mode='normal_view')

                # del this_skeleton


    print(f'Saving final results in {output_3d_filename}')
    with open(f'{output_3d_filename}', 'wb') as f:
        pickle.dump(output_3D_dataset, f)

    # output statistics to json
    with open(f'{output_3d_filename}_dataset_statistics_total_{total_frame_number}.json', 'w') as f:
        json.dump(dataset_statistics, f)

    # dt_file -- .pkl
    # trainset = self.dt_dataset['train']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]
    # testset = self.dt_dataset['test']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]


    # snipet to read example pkl file

    # file_name = r'C:\Users\Public\Documents\Vicon\vicon_coding_projects\MotionBERT\data\motion3d\h36m_sh_conf_cam_source_final.pkl\h36m_sh_conf_cam_source_final.pkl'
    # # file_name = r'W:\VEHS\VEHS data collection round 3\processed\VEHS_6D_downsample5_keep1.pkl'
    # with open(file_name, 'rb') as f:
    #     data = pickle.load(f)
    # data['train']['camera_name']
    # data['test']['joints_2.5d_image'][0]/data['test']['joint3d_image'][0] == 2.5factor
    # data['test']['joints_2.5d_image'][0]
    # data['test']['joint3d_image'][-1]
    # data['test']['2.5d_factor'][0]
    # ['validate']
        # >>> data.keys()
        # dict_keys(['train', 'test'])
        # >>> data['train'].keys()
        # dict_keys(['joint_2d', 'confidence', 'joint3d_image', 'camera_name', 'source'])
        # >>> data['test'].keys()
        # dict_keys(['joint_2d', 'confidence', 'joint3d_image', 'joints_2.5d_image', '2.5d_factor', 'camera_name', 'action', 'source'])