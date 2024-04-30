# Stuff from Hourglass
accIdxs = [0, 1, 2, 3, 4, 5, 10, 11, 14, 15]

edges = [
    [0, 1],
    [1, 2],
    [2, 6],
    [6, 3],
    [3, 4],
    [4, 5],
    [10, 11],
    [11, 12],
    [12, 8],
    [8, 13],
    [13, 14],
    [14, 15],
    [6, 8],
    [8, 9],
]

h36mImgDir = "datasets/h36m/images/"
expDir = "../exp"

bbox_padding = 1.5

skeletonRef = [
    [[0, 1], [1, 2], [3, 4], [4, 5]],
    [[10, 11], [11, 12], [13, 14], [14, 15]],
    [[2, 6], [3, 6]],
    [[12, 8], [13, 8]],
]

skeletonWeight = [
    [1.0085885098415446, 1, 1, 1.0085885098415446],
    [1.1375361376887123, 1, 1, 1.1375361376887123],
    [1, 1],
    [1, 1],
]

# order of joints in original H36M dataset
old_h36m_joints = [
    "pelvis",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_foot",
    "right_toe",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot",
    "left_toe",
    "pelvis2",
    "spine",
    "thorax",
    "neck",
    "head",
    "thorax2",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_wrist2",
    "left_thumb",
    "left_hand",
    "left_hand2",
    "thorax3",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_wrist2",
    "right_thumb",
    "right_hand",
    "right_hand2",
]

# order of 3D keypoints in processed dataset
new_h36m_joints = [
    "right_toe",
    "right_foot",
    "right_ankle",
    "right_knee",
    "right_hip",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot",
    "left_toe",
    "pelvis",
    "spine",
    "thorax",
    "neck",
    "head",
    "right_thumb",
    "right_hand",
    "right_wrist",
    "right_elbow",
    "right_shoulder",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hand",
    "left_thumb",
]

new_synthetic_data_keypoints = ['RightToe_End','RightToeBase','RightFoot','RightLeg','RightUpLeg','LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase','LeftToe_End', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head','RightHandThumb1','RightHandMiddle1','RightHand','RightForeArm','RightArm', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandMiddle1', 'LeftHandThumb1']


# order of joints in raw detectron output
old_detectron_joints = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# order of 2D keypoints in processed dataset
new_detectron_joints = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

old_ft_joints = [
    "pelvis",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "spine",
    "thorax",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
]

new_ft_joints = [
    "right_ankle",
    "right_knee",
    "right_hip",
    "left_hip",
    "left_knee",
    "left_ankle",
    "pelvis",
    "spine",
    "thorax",
    "neck",
    "head",
    "right_wrist",
    "right_elbow",
    "right_shoulder",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
]

# This represents the order of data points after post_processing the keypoints at the output of the detectron model in production.
# The format has been changed from snake format to lowercase with an underscore.
body_matrix = ["left_eye", "right_eye", "left_ear", "right_ear","right_foot", "right_knee", "right_hip", "left_hip", "left_knee", "left_foot", "back",
               "neck", "bottom_head", "top_head", "right_hand", "right_elbow", "right_shoulder", "left_shoulder",
               "left_elbow", "left_hand","lower_head","upper_head","left_palm","right_palm","left_heel","right_heel","spine"]

# This are the keypoint order for the model in production
model_prod_keypoints  = [ "right_foot", "right_knee", "right_hip", "left_hip", "left_knee", "left_foot", "back", "spine",
               "neck", "top_head", "right_hand", "right_elbow", "right_shoulder", "left_shoulder",
               "left_elbow", "left_hand"]

model_prod_keypoints_filter_2D_output = ["left_eye","right_eye","left_ear","right_ear","right_foot", "right_knee", "right_hip", "left_hip", "left_knee", "left_foot", "back", "spine",
               "neck", "top_head", "right_hand", "right_elbow", "right_shoulder", "left_shoulder",
               "left_elbow", "left_hand"]


# This are the keypoint order for the model in production
mediapipe_keypoints  = [ "right_foot", "right_knee", "right_hip", "left_hip", "left_knee", "left_foot", "back", "spine",
               "neck", "right_hand", "right_elbow", "right_shoulder", "left_shoulder",
               "left_elbow", "left_hand"]

mediapipe_full_keypoints = ['nose','left_eye_inner','left_eye','left_eye_outer','right_eye_inner','right_eye','right_eye_outer','left_ear','right_ear',
                            'mouth_left','mouth_right','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist',
                            'left_pinky','right_pinky','left_index','right_index','left_thumb','right_thumb','left_hip','right_hip','left_knee','right_knee',
                            'left_ankle','right_ankle','left_heel','right_heel','left_foot_index','right_foot_index','left_middle_mcp','right_middle_mcp']

mediapipe_edges_set = [(15, 21), (16, 20), (18, 20), (3, 7), (14, 16), (23, 25), (28, 30), (11, 23), (27, 31), (6, 8), (15, 17), (24, 26), (16, 22), (4, 5), (5, 6), (29, 31), (12, 24), (23, 24), (0, 1), (9, 10), (1, 2), (0, 4), (11, 13), (30, 32), (28, 32), (15, 19), (16, 18), (25, 27), (26, 28), (12, 14), (17, 19), (2, 3), (11, 12), (27, 29), (13, 15)]

mediapipe_edges = [[mediapipe_full_keypoints[i[0]],mediapipe_full_keypoints[i[1]]] for i in mediapipe_edges_set]

mediapipe_angle_keypoints = ['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist',
                            'left_pinky','right_pinky','left_index','right_index','left_hip','right_hip','left_knee','right_knee',
                            'left_ankle','right_ankle','left_middle_mcp','right_middle_mcp','pelvis','thorax','head']

mediapipe_angle_keypoints_old = ['nose','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist',
                            'left_pinky','right_pinky','left_index','right_index','left_hip','right_hip','left_knee','right_knee',
                            'left_ankle','right_ankle','pelvis','thorax']

mediapipe_angle_edges_set = [(24,5), (24,6), (6, 8), (5, 7),(7, 9), (8, 10),(9,21),(10,22), (23, 15), (23,16), (15, 17), (16, 18), (17, 19), (18, 20),(23,24),(24,25)]


# mediapipe_angle_edges_set = [(22,5), (22,6), (6, 8), (5, 7),(7, 9), (8, 10), (9, 11), (10, 12), (9, 13), (10, 14), (11, 13), (12, 14), (21, 15), (21,16), (15, 17), (16, 18), (17, 19), (18, 20),(21,22),(22,23)]

# mediapipe_angle_edges_set = [(0, 3), (0, 4), (22,5), (22,6), (6, 8), (5, 7),(7, 9), (8, 10), (9, 11), (10, 12), (9, 13), (10, 14), (11, 13), (12, 14), (21, 15), (21,16), (15, 17), (16, 18), (17, 19), (18, 20),(21,22),(22,23)]

mediapipe_angle_edges = [[mediapipe_angle_keypoints[i[0]],mediapipe_angle_keypoints[i[1]]] for i in mediapipe_angle_edges_set]

# This is a set of 37 original vicon keypoints (includes surface markers and joints) for the vicon dataset.
orig_vicon_joints = ["head_top","right_ear","left_ear",'C7','C7_d','SS','RAP_b','RAP_f','LAP_b','LAP_f','RLE','RME','LLE','LME',"right_index","right_pinky","left_index","left_pinky","pelvis","right_wrist","left_wrist","right_hip","left_hip","right_knee","left_knee","right_ankle","left_ankle","right_foot","left_foot","right_hand","left_hand","right_elbow","left_elbow","right_shoulder","left_shoulder","head","thorax"]
vicon_surface_markers = ['HDTP','REAR','LEAR','C7','C7_d','SS','RAP_b','RAP_f','LAP_b','LAP_f','RLE','RME','LLE','LME','RMCP2','RMCP5','LMCP2','LMCP5']

# This is a set of new 38 keypoints (includes surface markers and joints) for the vicon dataset to be used for multistride and motionbert training with rtmpose pretrained input. The name right/left hand has been replaced by right/left middle_mcp for convenience to use with rtmpose. Nose keypoint has been added
vicon_joints_for_rtmpose = ["head_top","right_ear","left_ear",'C7','C7_d','SS','RAP_b','RAP_f','LAP_b','LAP_f','RLE','RME','LLE','LME',"right_index","right_pinky","left_index","left_pinky","pelvis","right_wrist","left_wrist","right_hip","left_hip","right_knee","left_knee","right_ankle","left_ankle","right_foot","left_foot","right_middle_mcp","left_middle_mcp","right_elbow","left_elbow","right_shoulder","left_shoulder","head","thorax",'nose']


rtm_pose_keypoints = ['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_index','left_middle_mcp','left_pinky','right_index','right_middle_mcp','right_pinky','pelvis','thorax','head'] # This is a set of rtmpose keypoints for the mediapipe package
rtm_pose_keypoints_vicon_dataset = ['nose','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_index','left_middle_mcp','left_pinky','right_index','right_middle_mcp','right_pinky','pelvis','thorax','head'] # This is a set of rtmpose keypoints for the new training of the multistride and the MotionBert package

rtmpose_vicon_angle_edges_set = [(22,3), (22,4), (4, 6), (3, 5),(5, 7), (6, 8),(7,16),(8,19), (21, 9), (21,10), (9, 11), (10, 12), (11, 13), (12, 14),(21,22),(22,23)]

rtmpose_vicon_angle_edges = [[rtm_pose_keypoints_vicon_dataset[i[0]],rtm_pose_keypoints_vicon_dataset[i[1]]] for i in rtmpose_vicon_angle_edges_set]


angle_thresholds_pos_1 = {}
angle_thresholds_pos_2 = {}
angle_thresholds_pos_3 = {}

angle_thresholds_pos_1['spine'] = [120, 149]
angle_thresholds_pos_1['neck'] = [140, 159]
angle_thresholds_pos_1['left_shoulder'] = [45, 89]
angle_thresholds_pos_1['right_shoulder'] = [45, 89]
angle_thresholds_pos_1['left_elbow'] = [90, 119]
angle_thresholds_pos_1['right_elbow'] = [90, 119]
angle_thresholds_pos_1['left_knee'] = [120, 149]
angle_thresholds_pos_1['right_knee'] = [120, 149]
angle_thresholds_pos_2['spine'] = [90, 119]
angle_thresholds_pos_2['neck'] = [120, 139]
angle_thresholds_pos_2['left_shoulder'] = [90, 119]
angle_thresholds_pos_2['right_shoulder'] = [90, 119]
angle_thresholds_pos_2['left_elbow'] = [120, 149]
angle_thresholds_pos_2['right_elbow'] = [120, 149]
angle_thresholds_pos_2['left_knee'] = [90, 119]
angle_thresholds_pos_2['right_knee'] = [90, 119]
angle_thresholds_pos_3['spine'] = [0, 89]
angle_thresholds_pos_3['neck'] = [0, 119]
angle_thresholds_pos_3['left_shoulder'] = [120, 360]
angle_thresholds_pos_3['right_shoulder'] = [120, 360]
angle_thresholds_pos_3['left_elbow'] = [150, 180]
angle_thresholds_pos_3['right_elbow'] = [150, 180]
angle_thresholds_pos_3['left_knee'] = [0, 89]
angle_thresholds_pos_3['right_knee'] = [0, 89]

length_factor = 1
ears_factor = 1 * length_factor
eyes_factor = 1.05 * length_factor
ear_eye_factor = 0.8 * length_factor


# joint pairs for drawing 2D skeleton
detectron_edges = [
    ["nose", "left_eye"],
    ["nose", "right_eye"],
    ["left_eye", "left_ear"],
    ["right_eye", "right_ear"],
    ["left_shoulder", "left_elbow"],
    ["right_shoulder", "right_elbow"],
    ["left_elbow", "left_wrist"],
    ["right_elbow", "right_wrist"],
    ["left_hip", "left_knee"],
    ["right_hip", "right_knee"],
    ["left_knee", "left_ankle"],
    ["right_knee", "right_ankle"],
    ["left_hip", "left_shoulder"],
    ["right_hip", "right_shoulder"],
    ["left_hip", "right_hip"],
    ["left_shoulder", "right_shoulder"],
]

# joint pairs for drawing 3D skeleton
h36m_edges = [
    ["pelvis", "spine"],
    ["spine", "thorax"],
    ["thorax", "neck"],
    ["neck", "head"],
    ["thorax", "left_shoulder"],
    ["thorax", "right_shoulder"],
    ["left_shoulder", "left_elbow"],
    ["right_shoulder", "right_elbow"],
    ["left_elbow", "left_wrist"],
    ["right_elbow", "right_wrist"],
    ["pelvis", "left_hip"],
    ["pelvis", "right_hip"],
    ["left_hip", "left_knee"],
    ["right_hip", "right_knee"],
    ["left_knee", "left_ankle"],
    ["right_knee", "right_ankle"],
    ["left_wrist", "left_thumb"],
    ["right_wrist", "right_thumb"],
    ["left_wrist", "left_hand"],
    ["right_wrist", "right_hand"],
    ["left_ankle", "left_foot"],
    ["left_ankle", "left_toe"],
    ["right_ankle", "right_foot"],
    ["right_ankle", "right_toe"],
]

# joint pairs for drawing 3D skeleton for blender 3d data
new_synthetic_data_edges = [
    ["Spine", "Spine1"],
    ["Spine1", "Spine2"],
    ["Spine2", "Neck"],
    ["Neck", "Head"],
    ["Spine2", "LeftArm"],
    ["Spine2", "RightArm"],
    ["LeftArm", "LeftForeArm"],
    ["RightArm", "RightForeArm"],
    ["LeftForeArm", "LeftHand"],
    ["RightForeArm", "RightHand"],
    ["Spine", "LeftUpLeg"],
    ["Spine", "RightUpLeg"],
    ["LeftUpLeg", "LeftLeg"],
    ["RightUpLeg", "RightLeg"],
    ["LeftLeg", "LeftFoot"],
    ["RightLeg", "RightFoot"],
    ["LeftHand", "LeftHandThumb1"],
    ["RightHand", "RightHandThumb1"],
    ["LeftHand", "LeftHandMiddle1"],
    ["RightHand", "RightHandMiddle1"],
    ["LeftFoot", "LeftToeBase"],
    ["LeftFoot", "LeftToe_End"],
    ["RightFoot", "RightToeBase"],
    ["RightFoot", "RightToe_End"],
]

# angles in h36m skeleton
# in list [a, b, c], measured angle is between vector from joint b to joint a, and vector from joint b to joint c
h36m_angles = [
    ["pelvis", "spine", "thorax"],
    ["spine", "thorax", "neck"],
    ["thorax", "neck", "head"],
    ["thorax", "left_shoulder","left_elbow"],
    ["thorax", "right_shoulder", "right_elbow"],
    ["left_shoulder", "left_elbow", "left_wrist"],
    ["right_shoulder", "right_elbow", "right_wrist"],
    ["pelvis", "left_hip", "left_knee"],
    ["pelvis", "right_hip", "right_knee"],
    ["left_hip", "left_knee", "left_ankle"],
    ["right_hip", "right_knee", "right_ankle"],
    ["left_elbow", "left_wrist", "left_hand"],
    ["right_elbow", "right_wrist", "right_hand"],
    ["left_knee", "left_ankle", "left_foot"],
    ["right_knee", "right_ankle", "right_foot"],
]
synthetic_data_angles = [
    ["Spine", "Spine1", "Spine2"],
    ["Spine1", "Spine2", "Neck"],
    ["Spine2", "Neck", "Head"],
    ["Spine2", "LeftArm","LeftForeArm"],
    ["Spine2", "RightArm","RightForeArm"],
    ["LeftArm","LeftForeArm",'LeftHand'],
    ["RightArm","RightForeArm",'RightHand'],
    ["Spine", "LeftUpLeg", "LeftLeg"],
    ["Spine", "RightUpLeg", "RightLeg"],
    ["LeftUpLeg", "LeftLeg","LeftFoot"],
    ["RightUpLeg", "RightLeg","RightFoot"],
    ["LeftForeArm", "LeftHand", "LeftHandMiddle1"],
    ["RightForeArm", "RightHand", "RightHandMiddle1"],
    ["LeftLeg", "LeftFoot", "LeftToeBase"],
    ["RightLeg", "RightFoot", "RightToeBase"],
]
model_prod_angles = [
    ["back", "spine", "neck"],
    ["spine", "neck", "top_head"],
    ["neck", "left_shoulder","left_elbow"],
    ["neck", "right_shoulder", "right_elbow"],
    ["left_shoulder", "left_elbow", "left_hand"],
    ["right_shoulder", "right_elbow", "right_hand"],
    ["back", "left_hip", "left_knee"],
    ["back", "right_hip", "right_knee"],
    ["left_hip", "left_knee", "left_foot"],
    ["right_hip", "right_knee", "right_foot"]
]
mediapipe_angles = [
    ["back", "spine", "neck"],
    ["neck", "left_shoulder","left_elbow"],
    ["neck", "right_shoulder", "right_elbow"],
    ["left_shoulder", "left_elbow", "left_hand"],
    ["right_shoulder", "right_elbow", "right_hand"],
    ["back", "left_hip", "left_knee"],
    ["back", "right_hip", "right_knee"],
    ["left_hip", "left_knee", "left_foot"],
    ["right_hip", "right_knee", "right_foot"]
]

h36m_model_prod_map = {}

h36m_model_prod_map['right_foot'] = 'right_ankle'
h36m_model_prod_map['right_knee'] = 'right_knee'
h36m_model_prod_map['right_hip'] = 'right_hip'
h36m_model_prod_map['left_hip'] = 'left_hip'
h36m_model_prod_map['left_knee'] = 'left_knee'
h36m_model_prod_map['left_foot'] = 'left_ankle'
h36m_model_prod_map['back'] = 'pelvis'
h36m_model_prod_map['spine'] = 'spine'
h36m_model_prod_map['neck'] = 'thorax'
h36m_model_prod_map['top_head'] = 'head'
h36m_model_prod_map['right_hand'] = 'right_wrist'
h36m_model_prod_map['right_elbow'] = 'right_elbow'
h36m_model_prod_map['right_shoulder'] = 'right_shoulder'
h36m_model_prod_map['left_hand'] = 'left_wrist'
h36m_model_prod_map['left_elbow'] = 'left_elbow'
h36m_model_prod_map['left_shoulder'] = 'left_shoulder'

required_joints = ['spine', 'neck', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_knee', 'right_knee']

required_mediapipe_joints = ['spine', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_knee', 'right_knee']


# average lengths of h36m_edges in millimeters based on H36M mocap data
h36m_lengths = [
    487.60711027315,
    116.263543144122,
    115.000130036391,
    155.811487642304,
    155.81067558865,
    282.534007378305,
    282.536410218625,
    249.334844982907,
    249.334491465808,
    132.97660448403,
    132.977064821147,
    453.831582557215,
    453.832602746845,
    450.630201778088,
    450.630511503536,
]

Camera_cali = \
{
'A': [1,1,1,1,1,1,1,1,11,10],
'B': [1,1,1,1,1,1,1,1,9,12],
'C':[1,1,1,1,1,1,1,1],
'D':[1,1,1,1,1,1,1,1,9,12],
'E':[1,1,1,1,1,1,1,1,9,9],
'F': [1,1,1,1,1,1,1,1,9,9],
'G':[1,1,1,1,1,1,1,1,10],
'H':[1,1,1,1,1,1,1,1,10],
'I':[1,1,1,1,1,1,1,1,13,9],
'J':[1,1,1,1,1,1,1,1,13,9]
}

# the camera params need to be updated as they are from human 3.6m and not from velocity dataset
velocity_camera_params = [{
        "index": 0,
        "id": "Cam1",
        "res_w": 3840,
        "res_h": 2160,
        "center": [1921.2128681350878, 1072.0539237886987],
        "focal_length": [2916.9304199218750, 2975.1408691406250],
    },
    {
        "index": 1,
        "id": "Cam2",
        "res_w": 3840,
        "res_h": 2160,
        "center": [1921.2128681350878, 1072.0539237886987],
        "focal_length": [2916.9304199218750, 2975.1408691406250],
    },
    {
        "index": 2,
        "id": "Cam3",
        "res_w": 3840,
        "res_h": 2160,
        "center": [1921.2128681350878, 1072.0539237886987],
        "focal_length": [2916.9304199218750, 2975.1408691406250],
    },
    {
        "index": 3,
        "id": "Cam4",
        "res_w": 3840,
        "res_h": 2160,
        "center": [1921.2128681350878, 1072.0539237886987],
        "focal_length": [2916.9304199218750, 2975.1408691406250],
    },
{
        "index": 4,
        "id": "Cam5",
        "res_w": 3840,
        "res_h": 2160,
        "center": [1921.2128681350878, 1072.0539237886987],
        "focal_length": [2916.9304199218750, 2975.1408691406250],
    },
    {
        "index": 5,
        "id": "Cam6",
        "res_w": 3840,
        "res_h": 2160,
        "center": [1921.2128681350878, 1072.0539237886987],
        "focal_length": [2916.9304199218750, 2975.1408691406250],
    },
    {
        "index": 6,
        "id": "Cam7",
        "res_w": 3840,
        "res_h": 2160,
        "center": [1921.2128681350878, 1072.0539237886987],
        "focal_length": [2916.9304199218750, 2975.1408691406250],
    },
    {
        "index": 7,
        "id": "Cam8",
        "res_w": 3840,
        "res_h": 2160,
        "center": [1921.2128681350878, 1072.0539237886987],
        "focal_length": [2916.9304199218750, 2975.1408691406250],
    },
    {
        "index": 8,
        "id": "Cam9A",
        "res_w": 1080,
        "res_h": 1920,
        "center": [315.56665516851353, 233.59612678746544],
        "focal_length": [563.65844726562500, 485.42465209960938],
    },

    {
        "index": 9,
        "id": "Cam10A",
        "res_w": 1080,
        "res_h": 1920,
        "center": [1463.8655624949024, 2021.8136473281775],
        "focal_length": [1605.9387207031250, 1584.8847656250000],
    },


    {
        "index": 10,
        "id": "Cam9B",
        "res_w": 1080,
        "res_h": 1920,
        "center": [550.13520699142828, 940.21047748764977],
        "focal_length": [1684.5032958984375, 1688.2584228515625],
    },
    {
        "index": 11,
        "id": "Cam10B",
        "res_w": 1080,
        "res_h": 1920,
        "center": [548.87169682847889, 958.25498621280713],
        "focal_length": [892.06292724609375, 895.37463378906250],
    } ,
     {
        "index": 12,
        "id": "Cam9D",
        "res_w": 1080,
        "res_h": 1920,
        "center": [550.13520699142828, 940.21047748764977],
        "focal_length": [1684.5032958984375, 1688.2584228515625],
    },
    {
        "index": 13,
        "id": "Cam10D",
        "res_w": 1080,
        "res_h": 1920,
        "center": [548.87169682847889, 958.25498621280713],
        "focal_length": [892.06292724609375, 895.37463378906250],
    } ,
    {
        "index": 14,
        "id": "Cam9E",
        "res_w": 1080,
        "res_h": 1920,
        "center": [550.13520699142828, 940.21047748764977],
        "focal_length": [1684.5032958984375, 1688.2584228515625],
    },
    {
        "index": 15,
        "id": "Cam10E",
        "res_w": 1080,
        "res_h": 1920,
        "center": [550.13520699142828, 940.21047748764977],
        "focal_length": [1684.5032958984375, 1688.2584228515625],
    },

    {
        "index": 16,
        "id": "Cam9F",
        "res_w": 1080,
        "res_h": 1920,
        "center": [550.13520699142828, 940.21047748764977],
        "focal_length": [1684.5032958984375, 1688.2584228515625],
    },
    {
        "index": 17,
        "id": "Cam10F",
        "res_w": 1080,
        "res_h": 1920,
        "center": [550.13520699142828, 940.21047748764977],
        "focal_length": [1684.5032958984375, 1688.2584228515625],
    },
    {
        "index": 18,
        "id": "Cam10G",
        "res_w": 1080,
        "res_h": 1920,
        "center": [1463.8655624949024, 2021.8136473281775],
        "focal_length": [1605.9387207031250, 1584.8847656250000],
    },
    {
        "index": 19,
        "id": "Cam10H",
        "res_w": 1080,
        "res_h": 1920,
        "center": [1463.8655624949024, 2021.8136473281775],
        "focal_length": [1605.9387207031250, 1584.8847656250000],
    },

    {
        "index": 20,
        "id": "Cam9I",
        "res_w": 1080,
        "res_h": 1920,
        "center": [522.36395893767622, 927.01250215668915],
        "focal_length": [915.45989990234375, 943.77825927734375],
    },
    {
        "index": 21,
        "id": "Cam10I",
        "res_w": 1080,
        "res_h": 1920,
        "center": [550.13520699142828, 940.21047748764977],
        "focal_length": [1684.5032958984375, 1688.2584228515625],
    },


    {
        "index": 22,
        "id": "Cam9J",
        "res_w": 1080,
        "res_h": 1920,
        "center": [522.36395893767622, 927.01250215668915],
        "focal_length": [915.45989990234375, 943.77825927734375],
    },
    {
        "index": 23,
        "id": "Cam10J",
        "res_w": 1080,
        "res_h": 1920,
        "center": [550.13520699142828, 940.21047748764977],
        "focal_length": [1684.5032958984375, 1688.2584228515625],
    }

]

# the camera params need to be updated as they are from human 3.6m and not from velocity dataset
velocity_round_2_camera_params = [{
        "index": 0,
        "id": "Cam1",
        "res_w": 3840,
        "res_h": 2160,
        "center": [1921.2128681350878, 1072.0539237886987],
        "focal_length": [2916.9304199218750, 2975.1408691406250],
    },
    {
        "index": 1,
        "id": "Cam2",
        "res_w": 3840,
        "res_h": 2160,
        "center": [1921.2128681350878, 1072.0539237886987],
        "focal_length": [2916.9304199218750, 2975.1408691406250],
    },
    {
        "index": 2,
        "id": "Cam3",
        "res_w": 3840,
        "res_h": 2160,
        "center": [1921.2128681350878, 1072.0539237886987],
        "focal_length": [2916.9304199218750, 2975.1408691406250],
    },
    {
        "index": 3,
        "id": "Cam4",
        "res_w": 3840,
        "res_h": 2160,
        "center": [1921.2128681350878, 1072.0539237886987],
        "focal_length": [2916.9304199218750, 2975.1408691406250],
    },
{
        "index": 4,
        "id": "Cam5",
        "res_w": 3840,
        "res_h": 2160,
        "center": [1921.2128681350878, 1072.0539237886987],
        "focal_length": [2916.9304199218750, 2975.1408691406250],
    },
    {
        "index": 5,
        "id": "Cam6",
        "res_w": 3840,
        "res_h": 2160,
        "center": [1921.2128681350878, 1072.0539237886987],
        "focal_length": [2916.9304199218750, 2975.1408691406250],
    },
    {
        "index": 6,
        "id": "Cam7",
        "res_w": 3840,
        "res_h": 2160,
        "center": [1921.2128681350878, 1072.0539237886987],
        "focal_length": [2916.9304199218750, 2975.1408691406250],
    },
    {
        "index": 7,
        "id": "Cam8",
        "res_w": 3840,
        "res_h": 2160,
        "center": [1921.2128681350878, 1072.0539237886987],
        "focal_length": [2916.9304199218750, 2975.1408691406250],
    },
    {
        "index": 8,
        "id": "Cam9",
        "res_w": 3840,
        "res_h": 2160,
        "center": [1921.2128681350878, 1072.0539237886987],
        "focal_length": [2916.9304199218750, 2975.1408691406250],
    },
    {
        "index": 9,
        "id": "Cam10",
        "res_w": 3840,
        "res_h": 2160,
        "center": [1921.2128681350878, 1072.0539237886987],
        "focal_length": [2916.9304199218750, 2975.1408691406250],
    },
    {
        "index": 10,
        "id": "Cam11",
        "res_w": 3840,
        "res_h": 2160,
        "center": [1921.2128681350878, 1072.0539237886987],
        "focal_length": [2916.9304199218750, 2975.1408691406250],
    },
    {
        "index": 11,
        "id": "Cam12",
        "res_w": 3840,
        "res_h": 2160,
        "center": [1921.2128681350878, 1072.0539237886987],
        "focal_length": [2916.9304199218750, 2975.1408691406250],
    },
{
        "index": 12,
        "id": "Cam13",
        "res_w": 3840,
        "res_h": 2160,
        "center": [1921.2128681350878, 1072.0539237886987],
        "focal_length": [2916.9304199218750, 2975.1408691406250],
    },
    {
        "index": 13,
        "id": "Cam14",
        "res_w": 3840,
        "res_h": 2160,
        "center": [1921.2128681350878, 1072.0539237886987],
        "focal_length": [2916.9304199218750, 2975.1408691406250],
    },
    {
        "index": 14,
        "id": "Cam15",
        "res_w": 3840,
        "res_h": 2160,
        "center": [1921.2128681350878, 1072.0539237886987],
        "focal_length": [2916.9304199218750, 2975.1408691406250],
    },
    {
        "index": 15,
        "id": "Cam16",
        "res_w": 3840,
        "res_h": 2160,
        "center": [1921.2128681350878, 1072.0539237886987],
        "focal_length": [2916.9304199218750, 2975.1408691406250],
    },
    {
        "index": 16,
        "id": "Cam17",
        "res_w": 3840,
        "res_h": 2160,
        "center": [1921.2128681350878, 1072.0539237886987],
        "focal_length": [2916.9304199218750, 2975.1408691406250],
    },
    {
        "index": 17,
        "id": "Cam18",
        "res_w": 3840,
        "res_h": 2160,
        "center": [1921.2128681350878, 1072.0539237886987],
        "focal_length": [2916.9304199218750, 2975.1408691406250],
    },
    {
        "index": 18,
        "id": "Cam19",
        "res_w": 3840,
        "res_h": 2160,
        "center": [1921.2128681350878, 1072.0539237886987],
        "focal_length": [2916.9304199218750, 2975.1408691406250],
    },
    {
        "index": 19,
        "id": "Cam20",
        "res_w": 3840,
        "res_h": 2160,
        "center": [1921.2128681350878, 1072.0539237886987],
        "focal_length": [2916.9304199218750, 2975.1408691406250],
    },
]

# metadata for the four cameras in H36M
h36m_camera_params = [
    {
        "index": 0,
        "id": "54138969",
        "res_w": 1000,
        "res_h": 1002,
        "center": [512.54150390625, 515.4514770507812],
        "focal_length": [1145.0494384765625, 1143.7811279296875],
    },
    {
        "index": 1,
        "id": "55011271",
        "res_h": 1000,
        "res_w": 1000,
        "center": [508.8486328125, 508.0649108886719],
        "focal_length": [1149.6756591796875, 1147.5916748046875],
    },
    {
        "index": 2,
        "id": "58860488",
        "res_h": 1000,
        "res_w": 1000,
        "center": [519.8158569335938, 501.40264892578125],
        "focal_length": [1149.1407470703125, 1148.7989501953125],
    },
    {
        "index": 3,
        "id": "60457274",
        "res_h": 1002,
        "res_w": 1000,
        "center": [514.9682006835938, 501.88201904296875],
        "focal_length": [1145.5113525390625, 1144.77392578125],
    },
]


# metadata for the five cameras in H36M
blender_3d_camera_params = [
    {
        "index": 0,
        "id": "1",
        "res_w": 1080,
        "res_h": 1080,
        "center": [512.54150390625, 515.4514770507812],
        "focal_length": [1145.0494384765625, 1143.7811279296875],
    },
    {
        "index": 1,
        "id": "2",
        "res_h": 1080,
        "res_w": 1080,
        "center": [508.8486328125, 508.0649108886719],
        "focal_length": [1149.6756591796875, 1147.5916748046875],
    },
    {
        "index": 2,
        "id": "3",
        "res_h": 1080,
        "res_w": 1080,
        "center": [519.8158569335938, 501.40264892578125],
        "focal_length": [1149.1407470703125, 1148.7989501953125],
    },
    {
        "index": 3,
        "id": "4",
        "res_h": 1080,
        "res_w": 1080,
        "center": [514.9682006835938, 501.88201904296875],
        "focal_length": [1145.5113525390625, 1144.77392578125],
    },
    {
        "index": 4,
        "id": "5",
        "res_h": 1080,
        "res_w": 1080,
        "center": [514.9682006835938, 501.88201904296875],
        "focal_length": [1145.5113525390625, 1144.77392578125],
    },
]
