import c3d
from utility import *
from spacepy import pycdf
from Point import *
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
    parser.add_argument('--split_config_file', type=str, default=r'config/experiment_config/VEHS-R3-811-MotionBert.yaml')
    parser.add_argument('--skeleton_file', type=str, default=r'config\VEHS_ErgoSkeleton_info\Ergo-Skeleton.yaml')
    parser.add_argument('--downsample', type=int, default=5)
    parser.add_argument('--downsample_keep', type=int, default=1)
    parser.add_argument('--split_output', action='store_true')
    parser.add_argument('--output_type', type=list, default=[True, False, False, False], help='3D, 6D, SMPL, 3DSSPP')
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
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file.endswith('.c3d') and root[-6:] != 'backup' and (not file.startswith('ROM')) and (not file.endswith('_bad.c3d')):
                if val_keyword in root:  # todo: only one subject for val set now, expand to multiple subjects
                    train_val_test = 'test'
                elif test_keyword in root:
                    train_val_test = 'skip'
                    continue
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
                this_skeleton = VEHSErgoSkeleton(skeleton_file)
                this_skeleton.load_c3d(c3d_file, analog_read=False)
                this_skeleton.calculate_joint_center()
                camera_xcp_file = c3d_file.replace('.c3d', '.xcp')

                if args.output_type[0]:  # calculate 3D pose first
                    this_skeleton.calculate_camera_projection(camera_xcp_file, kpts_of_interest_name=h36m_joint_names)
                    output3D = this_skeleton.output_MotionBert_pose(downsample=downsample, downsample_keep=downsample_keep)
                    output_3D_dataset = append_output_xD_dataset(output_3D_dataset, train_val_test, output3D)
                if args.output_type[1]:  # calculate 6D pose
                    this_skeleton.calculate_camera_projection(camera_xcp_file, kpts_of_interest_name=custom_6D_joint_names)
                    output6D = this_skeleton.output_MotionBert_pose(downsample=downsample, downsample_keep=downsample_keep)
                    output_6D_dataset = append_output_xD_dataset(output_6D_dataset, train_val_test, output6D)
                if args.output_type[2]:  # todo: calculate SMPL pose
                    pass
                    # output_smpl_dataset =
                if args.output_type[3]:  # GT 3DSSPP output
                    batch_3DSSPP_batch_filename = c3d_file.replace('.c3d', '-3DSSPP.txt')
                    this_skeleton.output_3DSSPP_loc(frame_range=[0,1500,10], loc_file=batch_3DSSPP_batch_filename)
                # break  # self = this_skeleton
                del this_skeleton

                if args.split_output:
                    if 'activity08' in root or 'Activity08' in file:  # save intermediate results
                        print(f'Saving intermediate results for {c3d_file}')
                        if args.output_type[0]:  # 3D pose
                            with open(f'{output_3d_filename}_segment{count}.pkl', 'wb') as f:
                                pickle.dump(output_3D_dataset, f)
                            pkl_filenames['3D'].append(f'{output_3d_filename}_segment{count}.pkl')
                        if args.output_type[1]:  # 6D pose
                            with open(f'{output_6d_filename}_segment{count}.pkl', 'wb') as f:
                                pickle.dump(output_6D_dataset, f)
                            pkl_filenames['6D'].append(f'{output_6d_filename}_segment{count}.pkl')
                        if args.output_type[2]:  # SMPL pose
                            with open(f'{output_smpl_filename}_segment{count}.pkl', 'wb') as f:
                                pickle.dump(output_smpl_dataset, f)
                            pkl_filenames['SMPL'].append(f'{output_smpl_filename}_segment{count}.pkl')
                        # empty the dataset, else it is too big for memory and make it slow
                        output_3D_dataset = empty_MotionBert_dataset_dict(len(h36m_joint_names))  # 17
                        output_6D_dataset = empty_MotionBert_dataset_dict(len(custom_6D_joint_names))  # 66
                        # output_smpl_dataset =

    # # export: h36M 17 joint center, 6D pose 49 keypoints, SMPL-related, GT-Vicon to 3DSSPP
    if args.split_output:  #read and merge
        raise NotImplementedError  # need to merge the multilayer dict in the pkl files

    else:  # one output, for smaller dataset
        print(f'Saving final results in {output_3d_filename}, {output_6d_filename}, {output_smpl_filename}')
        if args.output_type[0]:  # 3D pose
            with open(f'{output_3d_filename}', 'wb') as f:
                pickle.dump(output_3D_dataset, f)
            print(f'Train set length: {len(output_3D_dataset["train"]["source"])}, Test set length: {len(output_3D_dataset["test"]["source"])}')
        if args.output_type[1]:  # 6D pose
            with open(f'{output_6d_filename}', 'wb') as f:
                pickle.dump(output_6D_dataset, f)
        if args.output_type[2]:  # SMPL pose
            with open(output_smpl_filename, 'wb') as f:
                pickle.dump(output_smpl_dataset, f)



    # dt_file -- .pkl
    # trainset = self.dt_dataset['train']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]
    # testset = self.dt_dataset['test']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]


    # snipet to read example pkl file

    # file_name = r'C:\Users\Public\Documents\Vicon\vicon_coding_projects\MotionBERT\data\motion3d\h36m_sh_conf_cam_source_final.pkl\h36m_sh_conf_cam_source_final.pkl'
    # with open(file_name, 'rb') as f:
    #     data = pickle.load(f)
    # data['test']['action'].shape
    #
        # >>> data.keys()
        # dict_keys(['train', 'test'])
        # >>> data['train'].keys()
        # dict_keys(['joint_2d', 'confidence', 'joint3d_image', 'camera_name', 'source'])
        # >>> data['test'].keys()
        # dict_keys(['joint_2d', 'confidence', 'joint3d_image', 'joints_2.5d_image', '2.5d_factor', 'camera_name', 'action', 'source'])