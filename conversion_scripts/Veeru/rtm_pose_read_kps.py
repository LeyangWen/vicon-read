import copy
import json
import matplotlib.pyplot as plt
import os
import numpy as np



######################################
#
# Assortment of functions given by Veeru and Francis
#
######################################


# def filter_subject_using_conf(all_keyps_bef):
#     # all_keyps_bef = np.array(all_keyps_bef)
#     all_keyps = []
#     for frame in all_keyps_bef:
#         if len(frame) > 0:
#             frame_keyps = [[0,0,0]]*frame.shape[1]
#             for instance in frame:
#                 for idx, keyp in enumerate(instance):
#                     if keyp[2] > frame_keyps[idx][2]:
#                         frame_keyps[idx] = keyp
#
#             all_keyps.append(frame_keyps)
#         else:
#             all_keyps.append([[0,0,0]]*133)
#     return np.array(all_keyps)

def filter_subject_across_bodies(all_keyps_bef):
    # all_keyps_bef = np.array(all_keyps_bef)
    all_keyps = []
    for frame in all_keyps_bef:
        if len(frame) > 0:
            frame_keyps = [[0,0,0]]*23
            confidence = 0
            for instance in frame:
                instance_expanded = np.expand_dims(instance, axis=0)
                rtm_keyps = get_rtm_keyps(instance_expanded)
                average_conf = np.mean(rtm_keyps[0,:,2])
                if average_conf > confidence:
                    confidence = average_conf
                    frame_keyps = rtm_keyps[0]
            all_keyps.append(frame_keyps)
        else:
            all_keyps.append([[0,0,0]]*23)
    return np.array(all_keyps)


def find_minimum_values(distance):
    min_values = [min(distance[key][i] for key in distance) for i in range(len(next(iter(distance.values()))))]
    min_idxs_count = {}
    for i in range(len(min_values)):
        for k in distance.keys():
            if distance[k][i] == min_values[i]:
                if k not in min_idxs_count:
                    min_idxs_count[k] = 1
                else:
                    min_idxs_count[k] += 1
                break
    for k in min_idxs_count.keys():
        if min_idxs_count[k] == max(min_idxs_count.values()):
            return k



def filter_subject_using_center_of_joints(all_keyps_bef,window = 8):
    # all_keyps_bef = np.array(all_keyps_bef)
    all_keyps = []
    center_of_joints = [0]
    for frame_idx, frame in enumerate(all_keyps_bef):
        if len(frame) > 0:
            frame_keyps = [[0,0,0]]*23
            confidence = 0
            distance = {} # Disctionary to store the distances of each instance from the center of joints
            instance_dict = {} # Dictionary to store the instances keypoints
            current_center_of_joints = {}
            for instance_idx, instance in enumerate(frame):
                instance_expanded = np.expand_dims(instance, axis=0)
                rtm_keyps = get_rtm_keyps(instance_expanded)
                if frame_idx == 0:
                    average_conf = np.mean(rtm_keyps[0,:,2])
                    if average_conf > confidence:
                        confidence = average_conf
                        frame_keyps = rtm_keyps[0]
                        center_of_joints[0] = np.mean(rtm_keyps[0,:,:2], axis=0)
                else:
                    instance_dict[instance_idx] = rtm_keyps[0]
                    current_center_of_joints[instance_idx] = np.mean(rtm_keyps[0,:,:2], axis=0)
                    distance[instance_idx] = [np.linalg.norm(i - current_center_of_joints[instance_idx]) for i in center_of_joints]
            if frame_idx != 0:
                instance_with_maximum_matches = find_minimum_values(distance)
                frame_keyps = instance_dict[instance_with_maximum_matches]
                center_of_joints.append(current_center_of_joints[instance_with_maximum_matches])
                if len(center_of_joints) > 8:
                    center_of_joints.pop(0)
            all_keyps.append(frame_keyps)
        else:
            all_keyps.append([[0,0,0]]*23)
    return np.array(all_keyps)


def filter_subject_using_center_of_joints_with_disqualify(all_keyps_bef,window = 4, type='rtm24' ,num_keypoints = 37, img_shape = (1200,1920)):
    # all_keyps_bef = np.array(all_keyps_bef)
    all_keyps = []
    prev_frames = [0]
    disqualify_val = 15/100 * (min(img_shape[0],img_shape[1]))
    for frame_idx, frame in enumerate(all_keyps_bef):
        if len(frame) > 0:
            frame_keyps = [[0,0,0]]*num_keypoints
            confidence = 0
            distance = [] # List to store the distances of each instance from the center of joints
            instance_dict = {} # Dictionary to store the instances keypoints
            if (frame_idx == 0) : # for frame 0, we chose average confidence as we do not have any previous frames
                for instance_idx, instance in enumerate(frame):
                    instance_expanded = np.expand_dims(instance, axis=0)
                    rtm_keyps = get_rtm_keyps(instance_expanded, type=type)
                    average_conf = np.mean(rtm_keyps[0,:,2])
                    if average_conf > confidence:
                        confidence = average_conf
                        frame_keyps = rtm_keyps[0]
                        prev_frames[0] = frame_keyps
            else:
                if not np.array(prev_frames).any(): # If all the previous frames are empty, we chose the subject with the highest average confidence
                    for instance_idx, instance in enumerate(frame):
                        instance_expanded = np.expand_dims(instance, axis=0)
                        rtm_keyps = get_rtm_keyps(instance_expanded, type=type)
                        average_conf = np.mean(rtm_keyps[0,:,2])
                        if average_conf > confidence:
                            confidence = average_conf
                            frame_keyps = rtm_keyps[0]
                    prev_frames.append(frame_keyps)
                    if len(prev_frames) > window:
                        prev_frames.pop(0)
                else:
                    for instance_idx, instance in enumerate(frame):
                        instance_expanded = np.expand_dims(instance, axis=0)
                        rtm_keyps = get_rtm_keyps(instance_expanded, type=type)
                        instance_dict[instance_idx] = rtm_keyps[0]
                        frames_matrix = np.concatenate((np.array(prev_frames),rtm_keyps),axis=0)
                        # common_joints = []
                        # for joint_column in range(frames_matrix.shape[1]):
                        #     if np.all((frames_matrix[:, joint_column, 0] > 0) & (frames_matrix[:, joint_column, 1] > 0) & (frames_matrix[:, joint_column, 2] > 0.3)):
                        #         common_joints.append(joint_column)
                        average_x_previous = np.average(frames_matrix[:-1, :, 0])
                        average_y_previous = np.average(frames_matrix[:-1, :, 1])
                        average_x_subject = np.average(frames_matrix[-1, :, 0])
                        average_y_subject = np.average(frames_matrix[-1, :, 1])
                        distance.append(np.linalg.norm(np.array([average_x_previous, average_y_previous]) - np.array([average_x_subject, average_y_subject])))
                    min_index, min_value = distance.index(min(distance)), min(distance)
                    frame_keyps = instance_dict[min_index]
                    # if min_value < disqualify_val:
                    #     frame_keyps = instance_dict[min_index]
                    # else:
                    #     frame_keyps = [[0,0,0]]*num_keypoints
                    prev_frames.append(frame_keyps)
                    if len(prev_frames) > window:
                            prev_frames.pop(0)
            all_keyps.append(frame_keyps)
        else:
            if isinstance(prev_frames[0],int):
                prev_frames[0] = [[0,0,0]]*num_keypoints
            else:
                prev_frames.append([[0,0,0]]*num_keypoints)
            all_keyps.append([[0,0,0]]*num_keypoints)
    return np.array(all_keyps)

def load_json_arr(json_path):
    """
    For RTM-img output, given by Veeru
    """

    lines = []
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        all_bboxes, all_keyps = [],[]
        for frame_idx in range(len(json_data['instance_info'])):
            bboxes = []
            keyps = []
            for instance in range(len(json_data['instance_info'][frame_idx]['instances'])):
                bbox_value = json_data['instance_info'][frame_idx]['instances'][instance]['bbox'][0]
                bbox_value.append(json_data['instance_info'][frame_idx]['instances'][instance]['bbox_score'])
                bboxes.append(bbox_value)
                keyp_values = json_data['instance_info'][frame_idx]['instances'][instance]['keypoints']
                keyp_scores = json_data['instance_info'][frame_idx]['instances'][instance]['keypoint_scores']
                [keyp_values[i].append(keyp_scores[i]) for i in range(len(keyp_values))]
                keyps.append(keyp_values)
            all_bboxes.append(np.array(bboxes))
            all_keyps.append(np.array(keyps))
    keypoint_names = json_data['meta_info']['keypoint_id2name']
    # print('done')
    return all_bboxes,all_keyps, keypoint_names  # todo: find all usage and add a, b, _

def load_json_arr_vid(json_path):
    """
    for rtm-vid output, given by Francis
    """

    lines = []
    with open(json_path, 'r', encoding='Windows-1252') as f:
        json_data = json.load(f)
        all_bboxes, all_keyps = [],[]
        for frame_idx in range(len(json_data)):
            bboxes = []
            keyps = []
            for instance in range(len(json_data[frame_idx]['instances'])):
                bbox_value = json_data[frame_idx]['instances'][instance]['bbox'][0]
                bbox_value.append(json_data[frame_idx]['instances'][instance]['bbox_score'])
                bboxes.append(bbox_value)
                keyp_values = json_data[frame_idx]['instances'][instance]['keypoints']
                keyp_scores = json_data[frame_idx]['instances'][instance]['keypoint_scores']
                [keyp_values[i].append(keyp_scores[i]) for i in range(len(keyp_values))]
                keyps.append(keyp_values)
            all_bboxes.append(np.array(bboxes))
            all_keyps.append(np.array(keyps))
    # print('done')
    return all_bboxes,all_keyps

# def get_rtm_keyps_vicon_dataset(all_keyps):
#     rtm_keyps = np.zeros((len(all_keyps),26,3))
#     rtm_keyps[:,0,:3] = all_keyps[:,0,:3]
#     rtm_keyps[:,1:15,:3] = all_keyps[:,3:17,:3]
#     rtm_keyps[:,15,:3] = all_keyps[:,96,:3]
#     rtm_keyps[:,16,:3] = (all_keyps[:,96,:3] + all_keyps[:,108,:3])/2
#     rtm_keyps[:,17,:3] = all_keyps[:,108,:3]
#     rtm_keyps[:,18,:3] = all_keyps[:,117,:3]
#     rtm_keyps[:,19,:3] = (all_keyps[:,117,:3] + all_keyps[:,129,:3])/2
#     rtm_keyps[:,20,:3] = all_keyps[:,129,:3]
#     rtm_keyps[:,21,:3] = (all_keyps[:,11,:3] + all_keyps[:,12,:3])/2 # Pelvis
#     rtm_keyps[:,22,:3] = (all_keyps[:,5,:3] + all_keyps[:,6,:3])/2 # Thorax
#     rtm_keyps[:,23,:3] = (all_keyps[:,3,:3] + all_keyps[:,4,:3])/2 # Head
#     return rtm_keyps


def get_rtm_keyps(all_keyps_before, type='rtm24'):
    all_keyps = copy.deepcopy(all_keyps_before)
    # print(f'type: {type}')
    if type == 'rtm24':
        rtm_keyps = np.zeros((len(all_keyps),26,3))
        rtm_keyps[:,:17,:3] = all_keyps[:,:17,:3]
        rtm_keyps[:,17,:3] = all_keyps[:,96,:3]
        rtm_keyps[:,18,:3] = all_keyps[:,100,:3]
        rtm_keyps[:,19,:3] = all_keyps[:,108,:3]
        rtm_keyps[:,20,:3] = all_keyps[:,117,:3]
        rtm_keyps[:,21,:3] = all_keyps[:,121,:3]
        rtm_keyps[:,22,:3] = all_keyps[:,129,:3]
        rtm_keyps[:,23,:3] = (rtm_keyps[:,11,:3] + rtm_keyps[:,12,:3])/2 # Pelvis
        rtm_keyps[:,24,:3] = (rtm_keyps[:,5,:3] + rtm_keyps[:,6,:3])/2 # Thorax or shoulder?
        rtm_keyps[:,25,:3] = (rtm_keyps[:,3,:3] + rtm_keyps[:,4,:3])/2 # Head
    elif type == 'rtm37_from_37':
        rtm_keyps = all_keyps
    elif type == 'rtm37_from_coco133':
        rtm_keyps = coco133_to_VEHS7M_37(all_keyps)
        raise NotImplementedError('Bug in other places')
    elif type == 'Lhand21':
        rtm_keyps = all_keyps[:, 91:112, :]
    elif type == 'Rhand21':
        rtm_keyps = all_keyps[:, 112:, :]
    return rtm_keyps

# experiment_folder = '/Users/vtalreja/Documents/Video_Pose_3D/RTMPose/output/kps_133_fps_15'
# json_file = "results_Limpador_Para_Brisas.json"
# img_shape = cv2.imread('/Users/vtalreja/Documents/Video_Pose_3D/mediapipe/inputs/Limpador_Para_Brisas/img00001.jpg').shape
# all_bboxes,all_keyps_bef = load_json_arr(os.path.join(experiment_folder,json_file))
# all_keyps = filter_subject_using_center_of_joints_with_disqualify(all_keyps_bef,img_shape=img_shape, window=4)
# all_keyps = filter_subject_using_center_of_joints(all_keyps_bef, window=8)

# all_keyps = filter_subject_across_bodies(all_keyps_bef)
# all_bboxes, all_keyps = filter_subject(all_bboxes, all_keyps_bef, img_shape, fps=30)
# rtm_keyps = get_rtm_keyps(all_keyps)
#
# print(rtm_keyps.shape)

# plt.plot(
#     [x['iteration'] for x in experiment_metrics],
#     [x['total_loss'] for x in experiment_metrics])
# plt.plot(
#     [x['iteration'] for x in experiment_metrics if 'validation_loss' in x],
#     [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])
# plt.legend(['total_loss', 'validation_loss'], loc='upper left')
# plt.savefig(os.path.join(experiment_folder,'loss_plot.png'))
# # plt.show()

def coco133_to_VEHS7M_37(coco133):
    """
    Convert coco133 to VEHS7M_37 for RTMPose eval
    coco133: (n, 133, 2)
    output: (n, 37, 2), fill with nan
    """
    coco133_VEHS7M = [(11-1, 1),
                        (10-1, 2),
                        (13-1, 3),
                        (12-1, 4),
                        (15-1, 5),
                        (14-1, 6),
                        (17-1, 7),
                        (16-1, 8),
                        (21-1, 9),
                        (18-1, 10),
                        (122-1, 11),
                        (101-1, 12),
                        (9-1, 13),
                        (8-1, 14),
                        (7-1, 15),
                        (6-1, 16),
                        (51-1, 17),
                        (5-1, 20),
                        (4-1, 21),
                        (118-1, 33),
                        (130-1, 34),
                        (97-1, 35),
                        (109-1, 36)]
    n, _, _ = coco133.shape
    VEHS7M_37 = np.full((n, 37, 3), np.nan)
    for i, j in coco133_VEHS7M:
        VEHS7M_37[:, j] = coco133[:, i]
    return VEHS7M_37
