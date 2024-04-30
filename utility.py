import c3d
from spacepy import pycdf
import os
import matplotlib.pyplot as plt
import numpy as np
import datetime
from mpl_toolkits.mplot3d import Axes3D


def plot_joint_axis(joint_axis_pts,label=None):
    # example
    # idx = [75,79] # head
    # idx = [75,79] # head
    # joint_axis_pts = points[idx[0]:idx[1]]
    # plot_joint_axis(joint_axis_pts,label = reader.point_labels[idx[0]])
    if label:
        print(label)
    # sgmentNameO segment Origin
    # segmentNameA Anterior axis
    # segmentNameP Proximal axis
    # segmentNameL Lateral axis
    axis_name = ['Anterior axis',
                'Lateral axis',
                'Proximal axis']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(joint_axis_pts[:, 0], joint_axis_pts[:, 1], joint_axis_pts[:, 2], c='r', marker='o')
    ax.set_xlabel('X (mm))')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    # connect points to joint_axis_pts[0]
    for i in range(1,4):
        ax.plot([joint_axis_pts[0, 0], joint_axis_pts[i, 0]],
                [joint_axis_pts[0, 1], joint_axis_pts[i, 1]],
                [joint_axis_pts[0, 2], joint_axis_pts[i, 2]])
        #print the length of each line
        #show axis name on the plot
        ax.text(joint_axis_pts[i, 0], joint_axis_pts[i, 1], joint_axis_pts[i, 2], axis_name[i - 1])
        print('length of {}: {}'.format(axis_name[i - 1], np.linalg.norm(joint_axis_pts[0] - joint_axis_pts[i])))
    ax.text(joint_axis_pts[0, 0], joint_axis_pts[0, 1], joint_axis_pts[0, 2], label)
    plt.show()


def create_dir(directory, is_base_dir=True):
    if is_base_dir:
        if not os.path.exists(directory):
            os.makedirs(directory)
    else:
        base_dir = os.path.dirname(directory)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)


def BMI_classUS(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Healthy weight'
    elif bmi < 30:
        return 'Overweight'
    elif bmi == 0:
        return 'Not available'
    else:
        return 'Obesity'


def BMI_caculate(weight, height):
    try:
        bmi = weight / (height / 100) ** 2
    except ZeroDivisionError:
        bmi = 0
    return bmi


def dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


def dist_array(p1s, p2s):
    return np.sqrt((p1s[:, 0] - p2s[:, 0]) ** 2 + (p1s[:, 1] - p2s[:, 1]) ** 2 + (p1s[:, 2] - p2s[:, 2]) ** 2)


def store_cdf(file_name, data, date='', kp_names='', subjectID='', TaskID='', CamID = '', jointName='', bbox=np.array([])):
    create_dir(os.path.dirname(file_name))
    if os.path.exists(file_name):
        os.remove(file_name)
    cdf = pycdf.CDF(file_name, '')
    cdf['Pose'] = data
    cdf.attrs['SubjectID'] = subjectID
    cdf.attrs['TaskID'] = TaskID
    cdf.attrs['CamID'] = CamID
    cdf.attrs['UpdateDate'] = datetime.datetime.now()
    cdf.attrs['CaptureDate'] = os.path.basename(date)
    cdf.attrs['KeypointNames'] = kp_names
    cdf['bbox'] = bbox
    cdf.close()

def empty_MotionBert_dataset_dict(joint_number):
    '''
    usage example:
    h36m_joint_names = ['PELVIS', 'RHIP', 'RKNEE', 'RANKLE', 'LHIP', 'LKNEE', 'LANKLE', 'T8', 'THORAX', 'C7', 'HEAD', 'LSHOULDER', 'LELBOW', 'LWRIST', 'RSHOULDER', 'RELBOW', 'RWRIST']
    output_3D_dataset = empty_MotionBert_dataset_dict(len(h36m_joint_names))  # 17
    custom_6D_joint_names = ['RPSIS', 'RASIS', 'LPSIS', 'LASIS', 'C7_d', 'SS', 'T8', 'XP', 'C7', 'HDTP', 'REAR', 'LEAR', 'RAP', 'RAP_f', 'RLE', 'RAP_b', 'RME', 'LAP', 'LAP_f', 'LLE', 'LAP_b', 'LME', 'LUS', 'LRS', 'RUS', 'RRS', 'RMCP5', 'RMCP2', 'LMCP5', 'LMCP2', 'LGT', 'LMFC', 'LLFC', 'RGT', 'RMFC', 'RLFC', 'RMM', 'RLM', 'LMM', 'LLM', 'LMTP1', 'LMTP5', 'LHEEL', 'RMTP1', 'RMTP5', 'RHEEL', 'HEAD', 'RSHOULDER', 'LSHOULDER', 'C7_m', 'THORAX', 'LELBOW', 'RELBOW', 'RWRIST', 'LWRIST', 'RHAND', 'LHAND', 'PELVIS', 'RHIP', 'RKNEE', 'RANKLE', 'RFOOT', 'LHIP', 'LKNEE', 'LANKLE', 'LFOOT']
    output_6D_dataset = empty_MotionBert_dataset_dict(len(custom_6D_joint_names))  # 66
    '''
    return {
        'train': {
            'joint_2d': np.empty((0, joint_number, 2)),
            'confidence': np.empty((0, joint_number, 1)),
            'joint3d_image': np.empty((0, joint_number, 3)),
            'camera_name': np.empty((0,)),
            'source': [],
            'c3d_frame': []
        },
        'validate': {
            'joint_2d': np.empty((0, joint_number, 2)),
            'confidence': np.empty((0, joint_number, 1)),
            'joint3d_image': np.empty((0, joint_number, 3)),
            'joints_2.5d_image': np.empty((0, joint_number, 3)),
            '2.5d_factor': np.empty((0,)),
            'camera_name': np.empty((0,)),
            'action': [],
            'source': [],
            'c3d_frame': []
        },
        'test': {
            'joint_2d': np.empty((0, joint_number, 2)),
            'confidence': np.empty((0, joint_number, 1)),
            'joint3d_image': np.empty((0, joint_number, 3)),
            'joints_2.5d_image': np.empty((0, joint_number, 3)),
            '2.5d_factor': np.empty((0,)),
            'camera_name': np.empty((0,)),
            'action': [],
            'source': [],
            'c3d_frame': []
        }
    }


def append_output_xD_dataset(output_xD_dataset, this_train_val_test, append_outputxD_dict):
    for key in output_xD_dataset[this_train_val_test].keys():
        if key == 'source' or key == 'c3d_frame':
            output_xD_dataset[this_train_val_test][key] = output_xD_dataset[this_train_val_test][key] + append_outputxD_dict[key]
        else:
            output_xD_dataset[this_train_val_test][key] = np.append(output_xD_dataset[this_train_val_test][key], append_outputxD_dict[key], axis=0)
    return output_xD_dataset

