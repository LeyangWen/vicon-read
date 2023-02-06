import c3d
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D




lower_joint_acronym = {
'PELO':'Pelvis Origin',
'PELP':'Pelvis Proximal',
'PELA':'Pelvis Anterior',
'PELL':'Pelvis Lateral',
'RFEO':'Right Femur Origin',
'RFEP':'Right Femur Proximal',
'RFEA':'Right Femur Anterior',
'RFEL':'Right Femur Lateral',
'LFEO':'Left Femur Origin',
'LFEP':'Left Femur Proximal',
'LFEA':'Left Femur Anterior',
'LFEL':'Left Femur Lateral',
'RTIO':'Right Tibia Origin',
'RTIP':'Right Tibia Proximal',
'RTIA':'Right Tibia Anterior',
'RTIL':'Right Tibia Lateral',
'LTIO':'Left Tibia Origin',
'LTIP':'Left Tibia Proximal',
'LTIA':'Left Tibia Anterior',
'LTIL':'Left Tibia Lateral',
'RFOO':'Right Foot Origin',
'RFOP':'Right Foot Proximal',
'RFOA':'Right Foot Anterior',
'RFOL':'Right Foot Lateral',
'LFOO':'Left Foot Origin',
'LFOP':'Left Foot Proximal',
'LFOA':'Left Foot Anterior',
'LFOL':'Left Foot Lateral',
'RTOO':'Right Toe Origin',
'RTOP':'Right Toe Proximal',
'RTOA':'Right Toe Anterior',
'RTOL':'Right Toe Lateral',
'LTOO':'Left Toe Origin',
'LTOP':'Left Toe Proximal',
'LTOA':'Left Toe Anterior',
'LTOL':'Left Toe Lateral'
}

upper_joint_acronym = {
'HEDO':	'Head Origin',
'HEDP':	'Head Proximal',
'HEDA':	'Head Anterior',
'HEDL':	'Head Lateral',
'TRXO':	'Thorax Origin',
'TRXP':	'Thorax Proximal',
'TRXA':	'Thorax Anterior',
'TRXL':	'Thorax Lateral',
'CSPO':	'C Spine Origin',
'CSPP':	'C Spine Proximal',
'CSPA':	'C Spine Anterior',
'CSPL':	'C Spine Lateral',
'SACO':	'Sacrum Origin',
'SACP':	'Sacrum Proximal',
'SACA':	'Sacrum Anterior',
'SACL':	'Sacrum Lateral',
'RCLO':	'Right Clavicle Origin',
'RCLP':	'Right Clavicle Proximal',
'RCLA':	'Right Clavicle Anterior',
'RCLL':	'Right Clavicle Lateral',
'LCLO':	'Left Clavicle Origin',
'LCLP':	'Left Clavicle Proximal',
'LCLA':	'Left Clavicle Anterior',
'LCLL':	'Left Clavicle Lateral',
'RHUO':	'Right Humerus Origin',
'RHUP':	'Right Humerus Proximal',
'RHUA':	'Right Humerus Anterior',
'RHUL':	'Right Humerus Lateral',
'LHUO':	'Left Humerus Origin',
'LHUP':	'Left Humerus Proximal',
'LHUA':	'Left Humerus Anterior',
'LHUL':	'Left Humerus Lateral',
'RRAO':	'Right Radius Origin',
'RRAP':	'Right Radius Proximal',
'RRAA':	'Right Radius Anterior',
'RRAL':	'Right Radius Lateral',
'LRAO':	'Left Radius Origin',
'LRAP':	'Left Radius Proximal',
'LRAA':	'Left Radius Anterior',
'LRAL':	'Left Radius Lateral',
'RHNO':	'Right Hand Origin',
'RHNP':	'Right Hand Proximal',
'RHNA':	'Right Hand Anterior',
'RHNL':	'Right Hand Lateral',
'LHNO':	'Left Hand Origin',
'LHNP':	'Left Hand Proximal',
'LHNA':	'Left Hand Anterior',
'LHNL':	'Left Hand Lateral',
'RFIO':	'Right Finger Origin',
'RFIP':	'Right Finger Proximal',
'RFIA':	'Right Finger Anterior',
'RFIL':	'Right Finger Lateral',
'LFIO':	'Left Finger Origin',
'LFIP':	'Left Finger Proximal',
'LFIA':	'Left Finger Anterior',
'LFIL':	'Left Finger Lateral',
'RTBO':	'Right Thumb Origin',
'RTBP':	'Right Thumb Proximal',
'RTBA':	'Right Thumb Anterior',
'RTBL':	'Right Thumb Lateral',
'LTBO':	'Left Thumb Origin',
'LTBP':	'Left Thumb Proximal',
'LTBA':	'Left Thumb Anterior',
'LTBL':	'Left Thumb Lateral',
}


def description(joint_acronym, acronym_dict_list = [upper_joint_acronym,lower_joint_acronym]):
    for acronym_dict in acronym_dict_list:
        joint_acronym = joint_acronym.strip()
        if joint_acronym in acronym_dict.keys():
            return acronym_dict[joint_acronym]
    return False


def plot_joint_axis(joint_axis_pts,label=None):
    if label:
        print(label,description(label))
    # sgmentNameO segment Origin
    # segmentNameA Anterior axis
    # segmentNameP Proximal axis
    # segmentNameL Lateral axis
    axis_name = ['Anterior axis',
                'Proximal axis',
                'Lateral axis']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(joint_axis_pts[:, 0], joint_axis_pts[:, 1], joint_axis_pts[:, 2], c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # connect points to joint_axis_pts[0]
    for i in range(1,4):
        ax.plot([joint_axis_pts[0, 0], joint_axis_pts[i, 0]],
                [joint_axis_pts[0, 1], joint_axis_pts[i, 1]],
                [joint_axis_pts[0, 2], joint_axis_pts[i, 2]])
        #print the length of each line
        #show axis name on the plot
        ax.text(joint_axis_pts[i, 0], joint_axis_pts[i, 1], joint_axis_pts[i, 2], axis_name[i - 1])
        print('length of {}: {}'.format(axis_name[i - 1], np.linalg.norm(joint_axis_pts[0] - joint_axis_pts[i])))
    plt.show()