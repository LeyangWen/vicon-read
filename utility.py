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


def store_cdf(file_name, data, date='', kp_names='', subjectID='', TaskID='', CamID = '', jointName=''):
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
    cdf.close()