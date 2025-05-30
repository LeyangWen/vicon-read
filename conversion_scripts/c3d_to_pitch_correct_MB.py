# c3d_to_MB.py

import c3d
from utility import *

from Skeleton import *
import pickle
import yaml
import argparse



if __name__ == '__main__':
    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_config_file', type=str, default=r'config/experiment_config/VEHS-R3-622-MotionBert.yaml') #default=r'config/experiment_config/VEHS-R3-721-MotionBert.yaml')
    parser.add_argument('--skeleton_file', type=str, default=r'config\VEHS_ErgoSkeleton_info\Ergo-Skeleton-66.yaml')
    parser.add_argument('--downsample', type=int, default=5)
    parser.add_argument('--downsample_keep', type=int, default=1)
    parser.add_argument('--split_output', action='store_true')  # not implemented yet
    parser.add_argument('--output_type', type=list, default=[False, True, False, False], help='3D, 6D, SMPL, 3DSSPP')
    parser.add_argument('--output_file_name_end', type=str, default='_37_oneCam_pitch_correct')  # v2
    parser.add_argument('--distort', action='store_false', help='consider camera distortion in the output 2D pose')
    parser.add_argument('--rootIdx', type=int, default=0, help='root index for 2D pose output')  # 21: pelvis for 66 kpts
    parser.add_argument('--MB_dict_version', type=str, default='normal', help='select from "normal", "diversity_metric"')
    parser.add_argument('--zero_camera_pitch', type=bool, default=True)
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
    output_3d_filename = os.path.join(base_folder, f'VEHS_3D_downsample{downsample}_keep{downsample_keep}{args.output_file_name_end}.pkl')
    output_6d_filename = os.path.join(base_folder, f'VEHS_6D_downsample{downsample}_keep{downsample_keep}{args.output_file_name_end}.pkl')
    output_smpl_filename = os.path.join(base_folder, f'VEHS_smpl_downsample{downsample}_keep{downsample_keep}{args.output_file_name_end}.pkl')

    # iterate through the folder to find all c3d
    # h36m_joint_names = ['Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine', 'Thorax', 'Neck/Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']  # h36m original names
    h36m_joint_names = ['HIP_c', 'RHIP', 'RKNEE', 'RANKLE', 'LHIP', 'LKNEE', 'LANKLE', 'T8', 'H36M_THORAX', 'H36M_NECK', 'H36M_HEAD', 'LSHOULDER', 'LELBOW', 'LWRIST', 'RSHOULDER', 'RELBOW', 'RWRIST']
    output_3D_dataset = empty_MotionBert_dataset_dict(len(h36m_joint_names))  # 17
    pkl_custom_6D_joint_names = ['RPSIS', 'RASIS', 'LPSIS', 'LASIS', 'C7_d', 'SS', 'T8', 'XP', 'C7', 'HDTP', 'REAR', 'LEAR', 'RAP', 'RAP_f', 'RLE', 'RAP_b', 'RME', 'LAP', 'LAP_f', 'LLE', 'LAP_b', 'LME',
                                 'LUS', 'LRS', 'RUS', 'RRS', 'RMCP5', 'RMCP2', 'LMCP5', 'LMCP2', 'LGT', 'LMFC', 'LLFC', 'RGT', 'RMFC', 'RLFC', 'RMM', 'RLM', 'LMM', 'LLM', 'LMTP1', 'LMTP5', 'LHEEL',
                                 'RMTP1', 'RMTP5', 'RHEEL', 'HEAD', 'RSHOULDER', 'LSHOULDER', 'C7_m', 'THORAX', 'LELBOW', 'RELBOW', 'RWRIST', 'LWRIST', 'RHAND', 'LHAND', 'PELVIS', 'RHIP', 'RKNEE',
                                 'RANKLE', 'RFOOT', 'LHIP', 'LKNEE', 'LANKLE', 'LFOOT']  # V1
    paper_custom_6D_joint_names = ['RPSIS', 'RASIS', 'LPSIS', 'LASIS', 'C7_d', 'SS', 'T8', 'XP', 'C7', 'HDTP', 'MDFH', 'REAR', 'LEAR', 'RAP', 'RAP_f', 'RLE', 'RAP_b', 'RME', 'LAP', 'LAP_f', 'LLE', 'LAP_b', 'LME',
                             'LUS', 'LRS', 'RUS', 'RRS', 'RMCP5', 'RMCP2', 'LMCP5', 'LMCP2', 'LGT', 'LMFC', 'LLFC', 'RGT', 'RMFC', 'RLFC', 'RMM', 'RLM', 'LMM', 'LLM', 'LMTP1', 'LMTP5', 'LHEEL',
                             'RMTP1', 'RMTP5', 'RHEEL', 'HEAD', 'RSHOULDER', 'LSHOULDER', 'THORAX', 'LELBOW', 'RELBOW', 'RWRIST', 'LWRIST', 'RHAND', 'LHAND', 'PELVIS', 'RHIP', 'RKNEE',
                             'RANKLE', 'RFOOT', 'LHIP', 'LKNEE', 'LANKLE', 'LFOOT']  # 66: drop c7_m, add MDFH, V2

    rtm_pose_keypoints_vicon_dataset_config6 = ['NOSE', 'LEAR', 'REAR', 'LSHOULDER', 'RSHOULDER', 'LELBOW', 'RELBOW', 'LWRIST', 'RWRIST', 'LHIP', 'RHIP',
                                        'LKNEE', 'RKNEE', 'LANKLE', 'RANKLE', 'LMCP2', 'LHAND', 'LMCP5', 'RMCP2', 'RHAND', 'RMCP5', 'HIP_c', 'SHOULDER_c', 'HEAD'] # RTMPose output (selected)
    rtm_pose_keypoints_vicon_dataset_config5 = ['NOSE', 'LEAR', 'REAR', 'LSHOULDER', 'RSHOULDER', 'LELBOW', 'RELBOW', 'LWRIST', 'RWRIST', 'LHIP', 'RHIP',
                                        'LKNEE', 'RKNEE', 'LANKLE', 'RANKLE', 'LMCP2', 'LHAND', 'LMCP5', 'RMCP2', 'RHAND', 'RMCP5', 'HIP_c', 'THORAX', 'HEAD']  # RTMPose output (selected)
    rtm_pose_keypoints_vicon_dataset_config2 = ['NOSE', 'LEAR', 'REAR', 'LSHOULDER', 'RSHOULDER', 'LELBOW', 'RELBOW', 'LWRIST', 'RWRIST', 'LHIP', 'RHIP',
                                                'LKNEE', 'RKNEE', 'LANKLE', 'RANKLE', 'LMCP2', 'LHAND', 'LMCP5', 'RMCP2', 'RHAND', 'RMCP5', 'PELVIS_b', 'SHOULDER_c', 'HEAD']

    rtm_pose_37_keypoints_vicon_dataset_v1 = ['PELVIS', 'RWRIST', 'LWRIST', 'RHIP', 'LHIP', 'RKNEE', 'LKNEE', 'RANKLE', 'LANKLE', 'RFOOT', 'LFOOT', 'RHAND', 'LHAND', 'RELBOW', 'LELBOW', 'RSHOULDER', 'LSHOULDER', 'HEAD', 'THORAX', 'HDTP', 'REAR', 'LEAR', 'C7', 'C7_d', 'SS', 'RAP_b', 'RAP_f', 'LAP_b', 'LAP_f', 'RLE', 'RME', 'LLE', 'LME', 'RMCP2', 'RMCP5', 'LMCP2', 'LMCP5']

    diversity_metric_keypointset = ['PELVIS', 'RWRIST', 'LWRIST', 'RHIP', 'LHIP', 'RKNEE', 'LKNEE', 'RANKLE', 'LANKLE', 'RFOOT', 'LFOOT', 'RHAND', 'LHAND', 'RELBOW', 'LELBOW', 'RSHOULDER',
                                              'LSHOULDER', 'HEAD', 'THORAX', 'HDTP', 'REAR', 'LEAR', 'C7', 'C7_d', 'RMCP2', 'RMCP5', 'LMCP2', 'LMCP5']

    ####### change output keypoints here
    custom_6D_joint_names = rtm_pose_37_keypoints_vicon_dataset_v1

    output_6D_dataset = empty_MotionBert_dataset_dict(len(custom_6D_joint_names), version=args.MB_dict_version)  # 66
    output_smpl_dataset = {}
    count = 0
    pkl_filenames = {'3D': [], '6D': [], 'SMPL': []}  # if split_output, save intermediate results
    dataset_statistics = {}
    total_frame_number = 0
    for root, dirs, files in os.walk(base_folder):
        dirs.sort()  # Sort directories in-place --> important, will change walk sequence
        files.sort(key=str.lower)  # Sort files in-place
        for file in files:
            if file.endswith('.c3d') and root[-6:] != 'backup' and (not file.startswith('ROM')) and (not file.endswith('_bad.c3d')):
                # val_keyword is a list of string, if any of them is in the root, then it is val set
                if any(keyword in root for keyword in val_keyword):
                    train_val_test = 'validate'
                elif any(keyword in root for keyword in test_keyword):
                    train_val_test = 'test'
                else:
                    train_val_test = 'train'

                print(f"file: {file}, root: {root}, train_val_test: {train_val_test}")

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
                this_frame_number = this_skeleton.frame_number
                dataset_statistics[c3d_file] = this_frame_number
                total_frame_number += this_frame_number

                this_skeleton.calculate_joint_center()
                camera_xcp_file = c3d_file.replace('.c3d', '.xcp')

                # this_skeleton.plot_3d_pose_frame(frame=0, coord_system="world")

                # S01-A01 camera pitch
                # correcting pitch for 51470934, pitch: 9.1 degrees
                # correcting pitch for 66920731, pitch: 29.2 degrees
                # correcting pitch for 66920734, pitch: 6.8 degrees
                # correcting pitch for 66920758, pitch: 16.7 degrees

                if args.output_type[0]:  # calculate 3D pose first
                    this_skeleton.calculate_camera_projection(args, camera_xcp_file, kpts_of_interest_name=h36m_joint_names, rootIdx=0)
                    output3D = this_skeleton.output_MotionBert_pose(downsample=downsample, downsample_keep=downsample_keep, pitch_correction=args.zero_camera_pitch)
                    output_3D_dataset = append_output_xD_dataset(output_3D_dataset, train_val_test, output3D)
                if args.output_type[1]:  # calculate 6D pose
                    this_skeleton.calculate_camera_projection(args, camera_xcp_file, kpts_of_interest_name=custom_6D_joint_names, rootIdx=args.rootIdx)  # Pelvis index
                    output6D = this_skeleton.output_MotionBert_pose(downsample=downsample, downsample_keep=downsample_keep, pitch_correction=args.zero_camera_pitch)
                    output_6D_dataset = append_output_xD_dataset(output_6D_dataset, train_val_test, output6D)
                if args.output_type[2]:
                    raise NotImplementedError('Implemented using moshpp+soma in separate repo')
                    pass
                    # output_smpl_dataset =
                if args.output_type[3]:  # GT 3DSSPP output
                    batch_3DSSPP_batch_filename = c3d_file.replace('.c3d', '-3DSSPP.txt')
                    this_skeleton.output_3DSSPP_loc(frame_range=[0, 3000, 10], loc_file=batch_3DSSPP_batch_filename)
                    break  # self = this_skeleton
                del this_skeleton
                if args.split_output:
                    if 'activity08' in root or 'Activity08' in file:  # save intermediate results, Not in use
                        print(f'Saving intermediate results for {c3d_file}')
                        if args.output_type[0]:  # 3D pose
                            with open(f'{output_3d_filename}_segment{count}.pkl', 'wb') as f:
                                pickle.dump(output_3D_dataset, f)
                            pkl_filenames['3D'].append(f'{output_3d_filename}_segment{count}.pkl')
                        if args.output_type[1]:  # 6D pose
                            with open(f'{output_6d_filename}_segment{count}.pkl', 'wb') as f:
                                pickle.dump(output_6D_dataset, f)
                            pkl_filenames['6D'].append(f'{output_6d_filename}_segment{count}.pkl')
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
        if args.output_type[1]:  # 6D pose
            with open(f'{output_6d_filename}', 'wb') as f:
                pickle.dump(output_6D_dataset, f)
        if args.output_type[3]:
            import subprocess
            import shutil
            ########################### Step 2: Run 3DSSPP ###########################
            # Get the initial modification time of the output file
            SSPP_CLI_folder = 'H:\\3DSSPP_all\Compiled\\3DSSPP 7.1.2 CLI'
            loc_file = batch_3DSSPP_batch_filename
            # export_file = os.path.join(SSPP_CLI_folder, 'export', 'batchinput_export.txt')  # constant if using wrapper
            # initial_mtime = os.stat(export_file).st_mtime
            loc_file = loc_file.replace('\\', '/')
            # copy the loc file to the 3DSSPP folder
            shutil.copy(loc_file, SSPP_CLI_folder)
            print(f"\n{'@' * 30} Subprocess start {'@' * 30}")
            # loc_file ="example_input_batch.txt"
            subprocess.call(['bash', '3DSSPP-script.sh', loc_file.split('/')[-1], '--avi', '4'], shell=True, cwd=SSPP_CLI_folder)  # '--autoclose'
            # careful to look for errors messages in terminal for the subprocess, will not stop code
            # wait_for_file_update(export_file, initial_mtime)  # Wait for the output file to be updated
            print(f"\n{'@' * 30} Subprocess end {'@' * 30}\n")

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
        # >>> data.keys()
        # dict_keys(['train', 'test'])
        # >>> data['train'].keys()
        # dict_keys(['joint_2d', 'confidence', 'joint3d_image', 'camera_name', 'source'])
        # >>> data['test'].keys()
        # dict_keys(['joint_2d', 'confidence', 'joint3d_image', 'joints_2.5d_image', '2.5d_factor', 'camera_name', 'action', 'source'])

