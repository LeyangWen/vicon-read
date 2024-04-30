from ergo3d import *
from Skeleton import *
import pickle
import yaml
import argparse


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
    parser.add_argument('--skeleton_file', type=str, default=r'config\VEHS_ErgoSkeleton_info\Ergo-Skeleton.yaml')
    parser.add_argument('--downsample', type=int, default=2)
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
            base_folder = data['base_folder']
            val_keyword = data['val_keyword']
            test_keyword = data['test_keyword']
        except yaml.YAMLError as exc:
            print(split_config_file, exc)


    skeleton_file = args.skeleton_file
    downsample = args.downsample
    downsample_keep = args.downsample_keep
    output_3d_filename = os.path.join(base_folder, f'VEHS_3D_downsample{downsample}_keep{downsample_keep}.pkl')
    output_6d_filename = os.path.join(base_folder, f'VEHS_6D_downsample{downsample}_keep{downsample_keep}.pkl')
    output_smpl_filename = os.path.join(base_folder, f'VEHS_smpl_downsample{downsample}_keep{downsample_keep}.pkl')

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
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file.endswith('.c3d') and root[-6:] != 'backup' and (not file.startswith('ROM')) and (not file.endswith('_bad.c3d')):
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
                this_skeleton = VEHSErgoSkeleton_angles(skeleton_file)
                this_skeleton.load_c3d(c3d_file, analog_read=False)
                this_frame_number = this_skeleton.frame_number
                dataset_statistics[c3d_file] = this_frame_number
                total_frame_number += this_frame_number

                this_skeleton.calculate_joint_center()
                ergo_angles = {}
                for angle_name in this_skeleton.angle_names:  # calling the angle calculation methods in skeleton class
                    class_method_name = f'{angle_name}_angles'
                    ergo_angles[angle_name] = getattr(this_skeleton, class_method_name)()

                frame_range = [0, 3000]
                angle_index = 6
                this_angle_name = this_skeleton.angle_names[angle_index]
                print(f'Visualizing {c3d_file} - {this_angle_name}')
                ergo_angles[this_angle_name].plot_angles(joint_name=this_angle_name, frame_range=frame_range)

                raise NotImplementedError  # break for testing
                # del this_skeleton
                # del ergo_angles


    # # # export: h36M 17 joint center, 6D pose 49 keypoints, SMPL-related, GT-Vicon to 3DSSPP
    # if args.split_output:  #read and merge
    #     raise NotImplementedError  # need to merge the multilayer dict in the pkl files
    # else:  # one output, for smaller dataset
    #     print(f'Saving final results in {output_3d_filename}, {output_6d_filename}, {output_smpl_filename}')
    #     if args.output_type[0]:  # 3D pose
    #         with open(f'{output_3d_filename}', 'wb') as f:
    #             pickle.dump(output_3D_dataset, f)
    #     if args.output_type[1]:  # 6D pose
    #         with open(f'{output_6d_filename}', 'wb') as f:
    #             pickle.dump(output_6D_dataset, f)
    #     if args.output_type[2]:  # SMPL pose
    #         with open(output_smpl_filename, 'wb') as f:
    #             pickle.dump(output_smpl_dataset, f)
    #
    # # output statistics to json
    # with open(f'{output_3d_filename}_dataset_statistics_total_{total_frame_number}.json', 'w') as f:
    #     json.dump(dataset_statistics, f)

    # dt_file -- .pkl
    # trainset = self.dt_dataset['train']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]
    # testset = self.dt_dataset['test']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]


    # snipet to read example pkl file

    # file_name = r'C:\Users\Public\Documents\Vicon\vicon_coding_projects\MotionBERT\data\motion3d\h36m_sh_conf_cam_source_final.pkl\h36m_sh_conf_cam_source_final.pkl'
    # file_name = r'W:\VEHS\VEHS data collection round 3\processed\VEHS_3D_downsample2_keep1.pkl'
    # with open(file_name, 'rb') as f:
    #     data = pickle.load(f)
    # confidence = data['train']['confidence']
    # id = np.argmin(confidence)
    # print(confidence.flatten()[id])
    #
    # joint_2d = data['train']['joint_2d']
    # joint_2d_x = joint_2d[:, :, 0]
    # joint_2d_y = joint_2d[:, :, 1]
    # print(joint_2d_x.flatten()[id])
    # print(joint_2d_y.flatten()[id])
    #
    # data['test']['joints_2.5d_image'][0]/data['test']['joint3d_image'][0] == 2.5factor
    # data['test']['joints_2.5d_image'][0]
    # data['test']['joint3d_image'][-1]
    # data['test']['2.5d_factor'][0]
    #     >>> data.keys()
    #     dict_keys(['train', 'test'])
    #     >>> data['train'].keys()
    #     dict_keys(['joint_2d', 'confidence', 'joint3d_image', 'camera_name', 'source'])
    #     >>> data['test'].keys()
    #     dict_keys(['joint_2d', 'confidence', 'joint3d_image', 'joints_2.5d_image', '2.5d_factor', 'camera_name', 'action', 'source'])