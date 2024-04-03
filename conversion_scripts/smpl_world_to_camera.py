from ergo3d import *
from Skeleton import *
import pickle
# import smplx

smpl_dir = r"/W/VEHS/VEHS-7M/SMPL/"

# Step 1: read SMPL pkl inputs
file = r"/home/leyang/Downloads/Activity03_stageii.pkl"
file = r"/home/leyang/Downloads/mesh_det_h36m.pkl"
# file = r"C:\Users\Public\Documents\Vicon\vicon_coding_projects\MotionBERT\data\motion3d\MB3D_VEHS_R3_small\3DPose\VEHS_3D_downsample_4.pkl_small.pkl"
# file = r"C:\Users\Public\Documents\Vicon\vicon_coding_projects\MotionBERT\data\motion3d\h36m_sh_conf_cam_source_final.pkl\h36m_sh_conf_cam_source_final.pkl"
with open(file, "rb") as f:
    data = pickle.load(f)

data.keys()
pose = data['fullpose']
#pose is the SMPL fullpose in axis angle representation
global_orient = pose[:, :3]
body_pose = pose[:, 3:66]
jaw_pose = pose[:, 66:69]
leye_pose = pose[:, 69:72]
reye_pose = pose[:, 72:75]
left_hand_pose = pose[:, 75:120]
right_hand_pose = pose[:, 120:]
# body_params = {"global orient": global_orient, "body pose": body_pose, "jaw_pose": jaw_pose, "leye pose": leye pose,
#     "reye_pose": reye_pose,"left_hand_pose": left_hand_pose, "right_hand_pose": right_hand_pose, "transl": trans}
# smplx_model.forward(**body_params)

# Step 2: read camera orientation & translation

# Step 3: project SMPLX pose from world coord to camera coord

# Step 3.5: save SMPLX pose as pkl file

# Step 4: convert SMPLX pose to SMPL pose

# Step 6: convert SMPL pose to MotionBERT format

