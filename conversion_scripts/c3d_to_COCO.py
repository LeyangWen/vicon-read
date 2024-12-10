# c3d_to_MB.py
import os.path

# import c3d
from utility import *

from Skeleton import *
import pickle
import yaml
import argparse



if __name__ == '__main__':
    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_config_file', type=str, default=r'config/experiment_config/VEHS-R3-721-MotionBert.yaml')
    parser.add_argument('--skeleton_file', type=str, default=r'config/VEHS_ErgoSkeleton_info/Ergo-Skeleton-66.yaml')
    parser.add_argument('--downsample', type=int, default=20)
    parser.add_argument('--downsample_keep', type=int, default=1)
    parser.add_argument('--split_output', action='store_true')  # not implemented yet
    parser.add_argument('--output_type', type=list, default=[False, True, False, False], help='3D, 6D, SMPL, 3DSSPP')
    parser.add_argument('--output_file_name_end', type=str, default='')
    parser.add_argument('--image_folder', default=r"/media/leyang/My Book/VEHS/VEHS-7M/img/5fps", help="split folder for images")
    parser.add_argument('--distort', action='store_false', help='consider camera distortion in the output 2D pose')
    parser.add_argument('--small_test', default=True, help='small test for debugging')
    args = parser.parse_args()

    if args.small_test:
        print("Small test mode: only 3 files will be processed for each set")
        args.output_file_name_end += '_small'

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
    output_3d_filename = os.path.join(base_folder, f'VEHS_3DCOCO_downsample{downsample}_keep{downsample_keep}{args.output_file_name_end}.pkl')
    output_6d_filename = os.path.join(base_folder, f'VEHS_6DCOCO_downsample{downsample}_keep{downsample_keep}{args.output_file_name_end}.pkl')

    # iterate through the folder to find all c3d
    # h36m_joint_names = ['Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine', 'Thorax', 'Neck/Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']  # h36m original names
    h36m_joint_names = ['PELVIS', 'RHIP', 'RKNEE', 'RANKLE', 'LHIP', 'LKNEE', 'LANKLE', 'T8', 'THORAX', 'C7', 'HEAD', 'LSHOULDER', 'LELBOW', 'LWRIST', 'RSHOULDER', 'RELBOW', 'RWRIST']
    output_3D_dataset = empty_MotionBert_dataset_dict(len(h36m_joint_names))  # 17
    pkl_custom_6D_joint_names = ['RPSIS', 'RASIS', 'LPSIS', 'LASIS', 'C7_d', 'SS', 'T8', 'XP', 'C7', 'HDTP', 'REAR', 'LEAR', 'RAP', 'RAP_f', 'RLE', 'RAP_b', 'RME', 'LAP', 'LAP_f', 'LLE', 'LAP_b', 'LME', 'LUS', 'LRS', 'RUS', 'RRS', 'RMCP5', 'RMCP2', 'LMCP5', 'LMCP2', 'LGT', 'LMFC', 'LLFC', 'RGT', 'RMFC', 'RLFC', 'RMM', 'RLM', 'LMM', 'LLM', 'LMTP1', 'LMTP5', 'LHEEL', 'RMTP1', 'RMTP5', 'RHEEL', 'HEAD', 'RSHOULDER', 'LSHOULDER', 'C7_m', 'THORAX', 'LELBOW', 'RELBOW', 'RWRIST', 'LWRIST', 'RHAND', 'LHAND', 'PELVIS', 'RHIP', 'RKNEE', 'RANKLE', 'RFOOT', 'LHIP', 'LKNEE', 'LANKLE', 'LFOOT']
    paper_custom_6D_joint_names = ['RPSIS', 'RASIS', 'LPSIS', 'LASIS', 'C7_d', 'SS', 'T8', 'XP', 'C7', 'HDTP', 'MDFH', 'REAR', 'LEAR', 'RAP', 'RAP_f', 'RLE', 'RAP_b', 'RME', 'LAP', 'LAP_f', 'LLE', 'LAP_b', 'LME',
                             'LUS', 'LRS', 'RUS', 'RRS', 'RMCP5', 'RMCP2', 'LMCP5', 'LMCP2', 'LGT', 'LMFC', 'LLFC', 'RGT', 'RMFC', 'RLFC', 'RMM', 'RLM', 'LMM', 'LLM', 'LMTP1', 'LMTP5', 'LHEEL',
                             'RMTP1', 'RMTP5', 'RHEEL', 'HEAD', 'RSHOULDER', 'LSHOULDER', 'THORAX', 'LELBOW', 'RELBOW', 'RWRIST', 'LWRIST', 'RHAND', 'LHAND', 'PELVIS', 'RHIP', 'RKNEE',
                             'RANKLE', 'RFOOT', 'LHIP', 'LKNEE', 'LANKLE', 'LFOOT']  # 66: drop c7_m, add MDFH

    rtm_pose_keypoints_vicon_dataset_config6 = ['NOSE', 'LEAR', 'REAR', 'LSHOULDER', 'RSHOULDER', 'LELBOW', 'RELBOW', 'LWRIST', 'RWRIST', 'LHIP', 'RHIP',
                                        'LKNEE', 'RKNEE', 'LANKLE', 'RANKLE', 'LMCP2', 'LHAND', 'LMCP5', 'RMCP2', 'RHAND', 'RMCP5', 'HIP_c', 'SHOULDER_c', 'HEAD'] # RTMPose output (selected)
    rtm_pose_keypoints_vicon_dataset_config5 = ['NOSE', 'LEAR', 'REAR', 'LSHOULDER', 'RSHOULDER', 'LELBOW', 'RELBOW', 'LWRIST', 'RWRIST', 'LHIP', 'RHIP',
                                        'LKNEE', 'RKNEE', 'LANKLE', 'RANKLE', 'LMCP2', 'LHAND', 'LMCP5', 'RMCP2', 'RHAND', 'RMCP5', 'HIP_c', 'THORAX', 'HEAD']  # RTMPose output (selected)
    rtm_pose_keypoints_vicon_dataset_config2 = ['NOSE', 'LEAR', 'REAR', 'LSHOULDER', 'RSHOULDER', 'LELBOW', 'RELBOW', 'LWRIST', 'RWRIST', 'LHIP', 'RHIP',
                                                'LKNEE', 'RKNEE', 'LANKLE', 'RANKLE', 'LMCP2', 'LHAND', 'LMCP5', 'RMCP2', 'RHAND', 'RMCP5', 'PELVIS_b', 'SHOULDER_c', 'HEAD']
    VEHS_ergo_37 = ['PELVIS', 'RWRIST', 'LWRIST', 'RHIP', 'LHIP', 'RKNEE', 'LKNEE', 'RANKLE', 'LANKLE', 'RFOOT', 'LFOOT', 'RHAND', 'LHAND', 'RELBOW', 'LELBOW',
                             'RSHOULDER', 'LSHOULDER', 'HEAD', 'THORAX', 'HDTP', 'REAR', 'LEAR', 'C7', 'C7_d', 'SS', 'RAP_b', 'RAP_f', 'LAP_b', 'LAP_f', 'RLE', 'RME',
                             'LLE', 'LME', 'RMCP2', 'RMCP5', 'LMCP2', 'LMCP5']


    ####### change output keypoints here
    custom_6D_joint_names = VEHS_ergo_37

    output_6D_dataset = empty_COCO_dataset_dict(len(custom_6D_joint_names))  # 66
    output_smpl_dataset = {}
    count = 0
    pkl_filenames = {'3D': [], '6D': [], 'SMPL': []}  # if split_output, save intermediate results
    dataset_statistics = {}
    total_frame_number = 0
    image_id_cum = 0
    pose_id_cum = 0
    small_test = {"train": 0, "validate": 0, "test": 0}
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
                if args.small_test:
                    small_test[train_val_test] += 1
                    cur_count = small_test[train_val_test]
                    print(f"{train_val_test}: cur_count: {cur_count}")
                    if cur_count > 1:
                        continue
                # if count > 1:  # give a very small file for testing
                #     break

                print(f'{count}: Starting on {c3d_file} as {train_val_test} set')
                this_skeleton = VEHSErgoSkeleton(skeleton_file)
                this_skeleton.load_c3d(c3d_file, analog_read=False)
                this_frame_number = this_skeleton.frame_number
                dataset_statistics[c3d_file] = this_frame_number
                total_frame_number += this_frame_number

                this_skeleton.calculate_joint_center()
                camera_xcp_file = c3d_file.replace('.c3d', '.xcp')

                # this_skeleton.plot_3d_pose_frame(frame=0, coord_system="world")

                if args.output_type[0]:  # calculate 3D pose first
                    raise NotImplementedError
                    this_skeleton.calculate_camera_projection(args, camera_xcp_file, kpts_of_interest_name=h36m_joint_names, rootIdx=0)
                    output3D = this_skeleton.output_MotionBert_pose(downsample=downsample, downsample_keep=downsample_keep)
                    output_3D_dataset = append_output_xD_dataset(output_3D_dataset, train_val_test, output3D)
                if args.output_type[1]:  # calculate 6D pose
                    this_skeleton.calculate_camera_projection(args, camera_xcp_file, kpts_of_interest_name=custom_6D_joint_names, rootIdx=0)  # Pelvis index
                    output6D = this_skeleton.output_COCO_2dPose(downsample=downsample, downsample_keep=downsample_keep, image_id_cum=image_id_cum, pose_id_cum=pose_id_cum, small_test=args.small_test)
                    image_id_cum = output6D["images"][-1]['id']
                    pose_id_cum = output6D["annotations"][-1]['id']

                    # visualization check
                    # frame = 500
                    # first_filename = output6D["images"][frame]['file_name']
                    # frame_idx = output6D["annotations"][frame]['id_100fps']
                    # skeleton_2d = VEHSErgoSkeleton(skeleton_file)
                    # skeleton_2d.load_name_list_and_np_points(custom_6D_joint_names, this_skeleton.pose_2d_camera['51470934'])
                    # skeleton_2d.plot_2d_pose_frame(frame=frame_idx, baseimage=os.path.join(args.image_folder, train_val_test, first_filename))




                    output_6D_dataset = append_COCO_xD_dataset(output_6D_dataset, train_val_test, output6D)
                del this_skeleton

                        # output_smpl_dataset =
    print(f'Saving final results in {output_3d_filename}, {output_6d_filename}')
    if args.output_type[0]:  # 3D pose
        with open(f'{output_3d_filename}', 'wb') as f:
            pickle.dump(output_3D_dataset, f)
        # save_COCO_json(output_3D_dataset, output_3d_filename)
    if args.output_type[1]:  # 6D pose
        with open(f'{output_6d_filename}', 'wb') as f:
            pickle.dump(output_6D_dataset, f)
        # output_6d_filename = os.path.join(r'C:\Users\wenleyan1\Downloads','VEHS_6DCOCO_downsample20_keep1.json')  # write to ssd instead of hhd for speed
        save_COCO_json(output_6D_dataset, output_6d_filename)

    # output statistics to json
    with open(f'{output_3d_filename}_dataset_statistics_total_{total_frame_number}.json', 'w') as f:
        json.dump(dataset_statistics, f)



    train_val_test = 'validate'
    json_filename = output_6d_filename.replace('.pkl', f'_{train_val_test}.json')
    with open(json_filename, 'r') as f:
        data = json.load(f)


    # dt_file -- .pkl
    # trainset = self.dt_dataset['train']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]
    # testset = self.dt_dataset['test']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]


    # snipet to read example pkl file

    # train_data = output_6D_dataset['train']
    #
    # train_data['images'][4000]['file_name']

    # cp '/media/leyang/My Book/VEHS/VEHS data collection round 3/processed/VEHS_6DCOCO_downsample20_keep1_train.json' '/media/leyang/My Book/VEHS/VEHS-7M/annotations/2d/VEHS_6DCOCO_downsample20_keep1_train.json'
    # cp '/media/leyang/My Book/VEHS/VEHS data collection round 3/processed/VEHS_6DCOCO_downsample20_keep1_validate.json' '/media/leyang/My Book/VEHS/VEHS-7M/annotations/2d/VEHS_6DCOCO_downsample20_keep1_validate.json'
    # cp '/media/leyang/My Book/VEHS/VEHS data collection round 3/processed/VEHS_6DCOCO_downsample20_keep1_test.json' '/media/leyang/My Book/VEHS/VEHS-7M/annotations/2d/VEHS_6DCOCO_downsample20_keep1_test.json'
