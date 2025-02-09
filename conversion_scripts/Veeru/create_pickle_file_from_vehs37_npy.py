import pickle
import numpy as np
import copy
import argparse

# File from Veeru to overwrite Vicon-MB pkl files with the RTMPose 2D pose and confidence score
####### For RMTpose - custom train v1 - 37kpts - sepearted npy files (2025 Jan)
# Step 1: c3d_to_MB.py to generate Vicon GT pkl file
# Step 2: create_pickle_file.py to overwrite with RTMPose 2D pose and confidence score

motionbert_pkl_file = r'W:\VEHS\VEHS data collection round 3\processed\VEHS_6D_downsample20_keep1_37_v1.pkl'
npy_dir = r'W:\VEHS\VEHS data collection round 3\RTM2D\RTMPose_VEHS7M_37kpts_v1\outputs_epoch_best_all'


new_pkl_file = motionbert_pkl_file.replace('.pkl', '_RTM2D.pkl')

def read_pkl(data_url):
    file = open(data_url,'rb')
    content = pickle.load(file)
    file.close()
    return content


motionbert_data_dict = read_pkl(motionbert_pkl_file)
new_motionbert_dict = copy.deepcopy(motionbert_data_dict)

for key in motionbert_data_dict.keys():
    print(f"===================={key}====================")
    sources = motionbert_data_dict[key]['source']
    sources.append('dummy_end')
    store_source = sources[0]
    old_clip_len = 0
    cum_start = 0
    sanity_check = np.ones(len(sources))
    for i, source in enumerate(sources):
        if source == store_source:
            old_clip_len += 1
        else:
            # print(f"Processing {store_source} with {old_clip_len} frames")
            subject_name = store_source.split('\\')[-3]
            action_name = store_source.split('\\')[-1].split('.')[0].lower()
            camera_id = store_source.split('\\')[-1].split('_')[-1]

            npy_file = f"results_{subject_name}-{action_name}-{camera_id}_keypoints.npy"
            with open(npy_dir + '/' + npy_file, 'rb') as f:
                det_2d_conf = np.load(f)
            length_diff = old_clip_len - det_2d_conf.shape[0]
            print( f"diff: {length_diff}, old_clip_len: {old_clip_len}, det_2d_conf.shape[0]: {det_2d_conf.shape[0]} for {subject_name}-{action_name}-{camera_id}")
            assert abs(length_diff) < 1.5, f"diff: {length_diff}, old_clip_len: {old_clip_len}, det_2d_conf.shape[0]: {det_2d_conf.shape[0]} for {subject_name}-{action_name}-{camera_id}"
            if length_diff > 0:
                ## concatenate the last frame to fill the gap, visualized and checked it is not big issues since last frame is stationary, frames before lines up perfectly
                det_2d_conf = np.concatenate([det_2d_conf, det_2d_conf[-1:]]*length_diff, axis=0)
            elif length_diff < 0:
                ## remove the last frames to fill the gap
                det_2d_conf = det_2d_conf[:-length_diff]
            new_motionbert_dict[key]['joint_2d'][cum_start:cum_start+old_clip_len] = det_2d_conf[:, :, :2]
            new_motionbert_dict[key]['confidence'][cum_start:cum_start+old_clip_len] = det_2d_conf[:, :, 2:]
            sanity_check[cum_start:cum_start+old_clip_len] = 0

            store_source = source
            cum_start += old_clip_len
            old_clip_len = 1

    assert np.sum(sanity_check) == 1, f"Some frames are not processed for set {key}"

pickle.dump(new_motionbert_dict, open(new_pkl_file, "wb"))

# test_dict = read_pkl(new_pkl_file)

print("Saved to:")
print(new_pkl_file)


## test visualize one frame to check fps alignment
######## Good snipet to visualize det and gt 2D pose
from Skeleton import *
det_2d = det_2d_conf[:, :, :2]
rtm_pose_37_keypoints_vicon_dataset_v1 = ['PELVIS', 'RWRIST', 'LWRIST', 'RHIP', 'LHIP', 'RKNEE', 'LKNEE', 'RANKLE', 'LANKLE', 'RFOOT', 'LFOOT', 'RHAND', 'LHAND', 'RELBOW', 'LELBOW', 'RSHOULDER',
                                          'LSHOULDER', 'HEAD', 'THORAX', 'HDTP', 'REAR', 'LEAR', 'C7', 'C7_d', 'SS', 'RAP_b', 'RAP_f', 'LAP_b', 'LAP_f', 'RLE', 'RME', 'LLE', 'LME', 'RMCP2', 'RMCP5',
                                          'LMCP2', 'LMCP5']
this_skeleton = VEHSErgoSkeleton(r'config\VEHS_ErgoSkeleton_info\Ergo-Skeleton-66.yaml')
this_skeleton.load_name_list_and_np_points(rtm_pose_37_keypoints_vicon_dataset_v1, det_2d)

frame_no = 400
base_image = f"W:\\VEHS\\VEHS-7M\\img\\5fps\\{key}\\{subject_name}-{action_name}-{camera_id}-{frame_no+1:06d}.jpg"
# W:\VEHS\VEHS-7M\img\5fps\test\S09-activity00-51470934-000001.jpg
this_skeleton.plot_2d_pose_frame(frame=frame_no, baseimage=base_image)  #, filename=r'C:\Users\wenleyan1\Downloads\test')





