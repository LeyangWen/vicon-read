import pickle
import numpy as np
import copy
import argparse
import matplotlib.pyplot as plt


def get_kpt_conf(conf, kpt_name):
    rtm_pose_37_keypoints_vicon_dataset_v1 = ['PELVIS', 'RWRIST', 'LWRIST', 'RHIP', 'LHIP', 'RKNEE', 'LKNEE', 'RANKLE', 'LANKLE', 'RFOOT', 'LFOOT', 'RHAND', 'LHAND', 'RELBOW', 'LELBOW', 'RSHOULDER',
                                              'LSHOULDER', 'HEAD', 'THORAX', 'HDTP', 'REAR', 'LEAR', 'C7', 'C7_d', 'SS', 'RAP_b', 'RAP_f', 'LAP_b', 'LAP_f', 'RLE', 'RME', 'LLE', 'LME', 'RMCP2',
                                              'RMCP5',
                                              'LMCP2', 'LMCP5']
    kpt_id = rtm_pose_37_keypoints_vicon_dataset_v1.index(kpt_name)
    return conf[:, kpt_id, -1]

def compare_center_support(center_support_names, conf, ):
    frame_no = conf.shape[0]
    center_name = center_support_names[0]
    support_name_1 = center_support_names[1]
    support_name_2 = center_support_names[2]
    center_conf = get_kpt_conf(conf, center_name)
    support_conf_1 = get_kpt_conf(conf, support_name_1)
    support_conf_2 = get_kpt_conf(conf, support_name_2)

    # occlusion criteria: bottom 20% of support conf
    occlusion_threshold = np.percentile(center_conf, 10)
    visible_threshold = np.percentile(center_conf, 90)
    occluded_idx = np.where(center_conf <= occlusion_threshold)[0]
    visible_idx = np.where(center_conf >= visible_threshold)[0]
    most_visible_score = np.max(conf[occluded_idx], axis=1)

    print(f"For {center_name}, occlusion threshold: {occlusion_threshold:.3f}, visible threshold: {visible_threshold:.3f}")
    print(f"Examples of occluded frames (bottom 20%): {occluded_idx[:10]}",
          f"\n{center_conf[occluded_idx][:10]}")
    print(f"Examples of visible frames (top 20%): {visible_idx[:10]}",
            f"\n{center_conf[visible_idx][:10]}")

    edge_id = np.concatenate([occluded_idx, visible_idx], axis=0)
    center_conf_edge = most_visible_score - center_conf[edge_id]
    support_conf_1_edge = most_visible_score - support_conf_1[edge_id]
    support_conf_2_edge = most_visible_score - support_conf_2[edge_id]
    support_conf_edge = np.concatenate([support_conf_1_edge, support_conf_2_edge], axis=0)

    # plot in histogram, reletive to total frame_no
    weights = np.ones_like(center_conf_edge) / frame_no
    weights_support = np.ones_like(support_conf_edge) / frame_no /2
    plt.figure(figsize=(8, 6))
    plt.hist(center_conf_edge, bins=30, alpha=0.5, label=center_name, weights=weights)
    plt.hist(support_conf_edge, bins=30, alpha=0.5, label=f"{support_name_1}&{support_name_2}", weights=weights_support)
    plt.xlabel('RTMW Inference Confidence Score', fontsize=14)
    plt.ylabel('Relative Frequency', fontsize=14)
    plt.title(f'Confidence Distribution of {center_name} and Supports', fontsize=16)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()

# pkl_file = "/Volumes/Z/RTMPose/37kpts_rtmw_v5/20fps/GT3D/VEHS_6D_downsample5_keep1_37_v2_pitch_correct_modified_RTM2D.pkl"
# pkl_file_industry = "/Volumes/Z/RTMPose/37kpts_rtmw_v5/20fps/2D/exp_2b_industry_videos_20fps/rtmpose_v5-2b_20fps_industry_37kpts_v2.pkl"


# visiblity
pkl_file = "/Users/leyangwen/Downloads/From_Veeru/exp_2b_industry_videos_20fps_with_visibility/rtmpose_v5-2b_20fps_industry_37kpts_v2_confidence.pkl"
pkl_file_industry = "/Users/leyangwen/Downloads/From_Veeru/exp_2b_industry_videos_20fps_with_visibility/rtmpose_v5-2b_20fps_industry_37kpts_v2_visibility.pkl"



rtm_pose_37_keypoints_vicon_dataset_v1 = ['PELVIS', 'RWRIST', 'LWRIST', 'RHIP', 'LHIP', 'RKNEE', 'LKNEE', 'RANKLE', 'LANKLE', 'RFOOT', 'LFOOT', 'RHAND', 'LHAND', 'RELBOW', 'LELBOW', 'RSHOULDER',
                                          'LSHOULDER', 'HEAD', 'THORAX', 'HDTP', 'REAR', 'LEAR', 'C7', 'C7_d', 'SS', 'RAP_b', 'RAP_f', 'LAP_b', 'LAP_f', 'RLE', 'RME', 'LLE', 'LME', 'RMCP2', 'RMCP5',
                                          'LMCP2', 'LMCP5']
rtm_pose_37_keypoints_vicon_dataset_v1 = np.array(rtm_pose_37_keypoints_vicon_dataset_v1)



with open(pkl_file, "rb") as f:
    data = pickle.load(f)

with open(pkl_file_industry, "rb") as f:
    data_industry = pickle.load(f)

# # confidence score of supporint keypoints (e.g., on elbow and shoulder side)
if False:
    confidence = data['validate']['confidence'][:,25:33]
    confidence_train = data['train']['confidence'][:,25:33]
    confidence_industry = data_industry['validate']['confidence'][:,25:33]
else:
    confidence = data['validate']['confidence']
    confidence_train = data['train']['confidence']
    confidence_industry = data_industry['validate']['confidence']


# show histogram of confidence distribution
confidence_flat = confidence.reshape(-1, confidence.shape[-1])
confidence_train_flat = confidence_train.reshape(-1, confidence_train.shape[-1])
confidence_industry_flat = confidence_industry.reshape(-1, confidence_industry.shape[-1])

if True:  # histogram plot
    # Calculate weights for relative frequency
    weights_vehs = np.ones_like(confidence_flat.flatten()) / len(confidence_flat.flatten())
    weights_vehs_train = np.ones_like(confidence_train_flat.flatten()) / len(confidence_train_flat.flatten())
    weights_industry = np.ones_like(confidence_industry_flat.flatten()) / len(confidence_industry_flat.flatten())

    # Show histogram of confidence distribution with relative frequency
    plt.figure(figsize=(8, 6))
    plt.hist(confidence_train_flat.flatten(), bins=50, alpha=0.5, color='green', label='VEHS6.8M Training Videos', weights=weights_vehs_train)
    plt.hist(confidence_flat.flatten(), bins=50, alpha=0.7, color='blue', label='VEHS6.8M Validation Videos', weights=weights_vehs)
    plt.hist(confidence_industry_flat.flatten(), bins=50, alpha=0.7, color='orange', label='Industry Videos', weights=weights_industry)
    plt.xlabel('RTMW Inference Confidence Score', fontsize=14)
    plt.ylabel('Relative Frequency', fontsize=14)
    plt.legend(fontsize=14)

    # set xy limit
    plt.xlim(0, 14)
    plt.ylim(0, 0.14)
    plt.tight_layout()
    plt.show()

if True:  # corrlation plot
    confidence_flat = confidence_flat[::100]
    confidence_industry_flat = confidence_industry_flat[::100]
    plt.figure(figsize=(8, 6))
    plt.scatter(confidence_flat.flatten(), confidence_industry_flat.flatten(), alpha=0.5, marker='.')
    plt.xlabel('Industry Videos Confidence Score', fontsize=14)
    plt.ylabel('Industry Videos Visibility Score', fontsize=14)
    # plt.title('Confidence Score Correlation', fontsize=16)
    plt.tight_layout()
    plt.show()
compare = np.concatenate([confidence, confidence_industry], axis=2)

from Skeleton import *
frame = 20145
estimate_skeleton = VEHSErgoSkeleton('config/VEHS_ErgoSkeleton_info/Ergo-Skeleton-37.yaml')
estimate_skeleton.load_name_list_and_np_points(rtm_pose_37_keypoints_vicon_dataset_v1, data_industry['validate']['joint_2d'])
estimate_skeleton.plot_2d_pose_frame(frame=frame) #, baseimage=baseimage)





# compare_center_support(["RELBOW", "RLE", "RME"], confidence)
# compare_center_support(["RELBOW", "RLE", "RME"], confidence_industry)
