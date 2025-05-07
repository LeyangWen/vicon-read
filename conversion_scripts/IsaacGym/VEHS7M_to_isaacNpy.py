
import os
import numpy as np

# AMASS dataset format

example_AMASS_file = r"/Users/leyangwen/Documents/Isaac/carry_data/AMASS_ACCAD/ACCAD/Female1General_c3d/A5_-_pick_up_box_stageii.npz"

with open(example_AMASS_file, 'rb') as f:
    data = np.load(f, allow_pickle=True)

    print(f"data.files: {data.files}")
    # ['gender', 'surface_model_type', 'mocap_frame_rate', 'mocap_time_length', 'markers_latent', 'latent_labels', 'markers_latent_vids', 'trans', 'poses', 'betas', 'num_betas', 'root_orient', 'pose_body', 'pose_hand', 'pose_jaw', 'pose_eye']
    for key in data.files:
        print(f"key: {key}, shape: {data[key].shape}, data: {data[key]}")

# TokenHSI input format
example_dir = r"/Users/leyangwen/Documents/Isaac/carry_data/dataset_carry/motions/carry/ACCAD+__+Female1General_c3d+__+A5_-_pick_up_box_stageii/"
example_box_motion = r"phys_humanoid_v3/box_motion.npy"
example_ref_motion = r"phys_humanoid_v3/ref_motion.npy"
example_smpl_params = r"smpl_params.npy"

with open(os.path.join(example_dir, example_box_motion), 'rb') as f:
    box_motion = np.load(f, allow_pickle=True)

print(box_motion.shape)

###############################################
with open(os.path.join(example_dir, example_ref_motion), 'rb') as f:
    ref_motion = np.load(f, allow_pickle=True)

ref_motion_dict = ref_motion.item()
print(ref_motion_dict.keys())
# odict_keys(['rotation', 'root_translation', 'global_velocity', 'global_angular_velocity', 'skeleton_tree', 'is_local', 'fps', '__name__'])

print(f"ref_motion_dict['rotation']['arr'] shape == {ref_motion_dict['rotation']['arr'].shape}")
print(f"ref_motion_dict['root_translation']['arr'] shape == {ref_motion_dict['root_translation']['arr'].shape}")
print(f"ref_motion_dict['global_velocity']['arr'] shape == {ref_motion_dict['global_velocity']['arr'].shape}")
print(f"ref_motion_dict['global_angular_velocity']['arr'] shape == {ref_motion_dict['global_angular_velocity']['arr'].shape}")


print(f"ref_motion_dict['skeleton_tree']['arr'] shape == {ref_motion_dict['skeleton_tree']['arr'].shape}")
print(f"ref_motion_dict['is_local']['arr'] shape == {ref_motion_dict['is_local']['arr'].shape}")




###############################################
with open(os.path.join(example_dir, example_smpl_params), 'rb') as f:
    smpl_params = np.load(f, allow_pickle=True)
# smpl_params is a 0‑d array containing one dict
params_dict = smpl_params.item()

# Now you can inspect its keys
print(params_dict.keys())
# dict_keys(['poses', 'trans', 'fps'])

# And access each entry
poses = params_dict['poses']    # (N_frames, 72)-shaped array of axis‑angle joint rotations
trans = params_dict['trans']    # (N_frames, 3)  translations
fps   = params_dict['fps']      # e.g. 30

print(poses.shape)
print(trans.shape)
print(fps)

import pickle
VEHS_file = r"/Users/leyangwen/Documents/Isaac/carry_data/VEHS7M/S01/Activity04_stageii.pkl"

with open(VEHS_file, 'rb') as f:
    VEHS_data = pickle.load(f)
