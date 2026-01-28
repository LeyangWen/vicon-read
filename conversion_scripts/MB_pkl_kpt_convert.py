import pickle
import numpy as np

# 37kpts --> 17kpts

# file = r"/media/leyang/My Book3/VEHS/VEHS data collection round 3/processed/VEHS_6D_downsample5_keep1_37_v2_pitch_correct_modified_RTM2D_Pitched_SMPL.pkl"
# file = r"/media/leyang/My Book3/VEHS/VEHS data collection round 3/processed/VEHS_6D_downsample5_keep1_37_v2_modified_RTM2D.pkl"
file = r"/media/leyang/My Book3/VEHS/VEHS data collection round 3/processed/VEHS_6D_downsample5_keep1_37_v2_pitch_correct_modified_RTM2D.pkl"


with open(file, "rb") as f:
    MB_data = pickle.load(f)
    
h36m_joint_names = ['HIP_c', 'RHIP', 'RKNEE', 'RANKLE', 'LHIP', 'LKNEE', 'LANKLE', 'T8', 'H36M_THORAX', 'H36M_NECK', 'H36M_HEAD', 'LSHOULDER', 'LELBOW', 'LWRIST', 'RSHOULDER', 'RELBOW', 'RWRIST']
rtm_pose_37_keypoints_vicon_dataset_v1 = ['PELVIS', 'RWRIST', 'LWRIST', 'RHIP', 'LHIP', 'RKNEE', 'LKNEE', 'RANKLE', 'LANKLE', 'RFOOT', 'LFOOT', 'RHAND', 'LHAND', 'RELBOW', 'LELBOW', 'RSHOULDER', 'LSHOULDER', 'HEAD', 'THORAX', 'HDTP', 'REAR', 'LEAR', 'C7', 'C7_d', 'SS', 'RAP_b', 'RAP_f', 'LAP_b', 'LAP_f', 'RLE', 'RME', 'LLE', 'LME', 'RMCP2', 'RMCP5', 'LMCP2', 'LMCP5']


RTM2D_37_TO_17_IDX = [
    0,   # HIP_c        -> PELVIS
    3,   # RHIP         -> RHIP
    5,   # RKNEE        -> RKNEE
    7,   # RANKLE       -> RANKLE
    4,   # LHIP         -> LHIP
    6,   # LKNEE        -> LKNEE
    8,   # LANKLE       -> LANKLE
    23,  # T8           -> C7_d (cloest one we have)
    18,  # H36M_THORAX  -> THORAX
    22,  # H36M_NECK    -> C7
    17,  # H36M_HEAD    -> HEAD
    16,  # LSHOULDER    -> LSHOULDER
    14,  # LELBOW       -> LELBOW
    2,   # LWRIST       -> LWRIST
    15,  # RSHOULDER    -> RSHOULDER
    13,  # RELBOW       -> RELBOW
    1    # RWRIST       -> RWRIST
]

for split in MB_data.keys():
    for data_key in MB_data[split].keys():
        if type(MB_data[split][data_key]) == np.ndarray:
            
            shape = MB_data[split][data_key].shape
            print(f"Processing {split} set, key: {data_key}, original shape: {shape}")
            if len(shape) < 2 or shape[1] != 37:
                continue
            
            MB_data[split][data_key] = MB_data[split][data_key][:, RTM2D_37_TO_17_IDX, ...]
            print(f"New shape: {MB_data[split][data_key].shape}")
# Save the new pkl file
output_file = file.replace('37_v2', '17_v2')
with open(output_file, "wb") as f:
    pickle.dump(MB_data, f)
print(f"Output file saved to {output_file}")