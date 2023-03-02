import c3d
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D



def plot_joint_axis(joint_axis_pts,label=None):
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


