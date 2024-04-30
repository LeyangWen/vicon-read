from ergo3d import *
from Skeleton import *
import pickle
# import smplx


# Step 1: read SMPL pkl inputs
file = r"W:/VEHS/VEHS-7M/SMPL/S01/Activity03_stageii.pkl"
with open(file, "rb") as f:
    data = pickle.load(f)

file = r"C:\Users\Public\Documents\Vicon\vicon_coding_projects\MotionBERT\data\mesh\mesh_det_h36m.pkl"
with open(file, "rb") as f:
    data_MB = pickle.load(f)

data_MB['test'].keys()  # dict_keys(['joint_2d', 'confidence', 'joint_cam', 'smpl_pose', 'smpl_shape', 'camera_name', 'action', 'source'])
data_MB['test']['joint_2d'].shape  # (102280, 17, 2)
data_MB['test']['confidence'].shape  # (102280, 17, 1)
data_MB['test']['joint_cam'].shape  # (102280, 17, 3)
data_MB['test']['smpl_pose'].shape  # (102280, 72)
data_MB['test']['smpl_shape'].shape  # (102280, 10)
data_MB['test']['camera_name']  # [102280]
data_MB['test']['action']  # [102280]
data_MB['test']['source']  # [102280]

data_MB['train'].keys()

store_cam_name = ""
for frame_id, camera_name in enumerate(data_MB['test']['camera_name']):
    if camera_name != store_cam_name:
        store_cam_name = camera_name
        activity = data_MB['test']['action'][frame_id]
        print(f"frame_id: {frame_id}, camera_name: {camera_name}, activity: {activity}")

frame_ids = [97858, 98764, 99667]
for frame_id in frame_ids:
    # print(f"frame_id: {frame_id}, camera_name: {data_MB['test']['camera_name'][frame_id]}, activity: {data_MB['test']['action'][frame_id]} #, SMPL_pose: {data_MB['test']['smpl_pose'][frame_id][0:3]}")
    print(f"joint_cam: {data_MB['test']['joint_cam'][frame_id]}")



data.keys()
pose = data['fullpose']
trans = data['trans']
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

