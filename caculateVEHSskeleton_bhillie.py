from viconnexusapi import ViconNexus
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
import pickle
from utility import *
from Point import *
import yaml
import datetime
import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="invalid value encountered in arccos")

# helper functions
# vicon = ViconNexus.ViconNexus()
# dir(ViconNexus.ViconNexus)
# help(vicon.GetSubjectParam)

if __name__ == '__main__':

    ######################################## START UP ########################################
    # specify big or small marker
    marker_height = 14/2+2  # 14mm marker
    # marker_height = 9.5/2+2  # 9.5mm marker

    vicon = ViconNexus.ViconNexus()
    trial_name = vicon.GetTrialName()
    subject_names = vicon.GetSubjectNames()
    frame_count = vicon.GetFrameCount()
    frame_rate = vicon.GetFrameRate()
    subject_info = vicon.GetSubjectInfo()
    weight = vicon.GetSubjectParam(subject_names[0], 'Bodymass')[0]
    height = vicon.GetSubjectParam(subject_names[0], 'Height')[0]
    BMI = BMI_caculate(weight, height/1000)
    BMI_class = BMI_classUS(BMI)

    # write to yaml file
    trial_yaml_file = os.path.join(trial_name[0],trial_name[1]+ '.yaml')
    c3d_file = os.path.join(trial_name[0],trial_name[1]+ '.c3d')
    trial_info = {'processed_time': datetime.datetime.now(), 'trial_dir': trial_name[0], 'trial_name': trial_name[1], 'description':'',
                  'c3d_file': c3d_file,
                  'subject_name': subject_names[0], 'frame_count': frame_count, 'marker_height': marker_height,
                  'frame_rate': frame_rate, 'weight': weight, 'height': height,
                  'BMI': BMI, 'BMI_class': BMI_class}
    with open(trial_yaml_file, 'w') as f:
        f.write(yaml.dump(trial_info, default_flow_style=False, sort_keys=False))


    # if more than one subject, report not implemented error
    if len(subject_names) > 1:
        print(subject_names)
        raise NotImplementedError(f"More than one subject not implemented --> {subject_names}")

    ######################################## Read marker data ########################################
    # upper body
    C7_d = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'C7_d'))
    SS = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'SS'))
    RSHO_f = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RSHO_f'))
    LSHO_f = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LSHO_f'))
    RSHO_b = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RSHO_b'))
    LSHO_b = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LSHO_b'))
    RME = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RME'))
    RLE = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RLE'))
    # LME = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LME'))
    # LLE = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LLE'))


    ######################################## Create virtual markers ########################################
    RSHOULDER = Point.mid_point(RSHO_f, RSHO_b)
    LSHOULDER = Point.mid_point(LSHO_f, LSHO_b)
    C7_m = Point.mid_point(C7_d, SS)
    # LELBOW = Point.mid_point(LME, LLE)
    RELBOW = Point.mid_point(RME, RLE)



    ######################################## Calculate angles ########################################
    zero_frame = 941
    # RShoulder angles
    RSHOULDER_plane = Plane(RSHOULDER, RSHO_f, RSHO_b)
    RSHOULDER_coord = CoordinateSystem3D()
    RSHOULDER_coord.set_by_plane(RSHOULDER_plane, RSHOULDER, C7_m, sequence='zyx', axis_positive=False)
    RSHOULDER_angles = JointAngles()
    RSHOULDER_angles.set_zero_frame(zero_frame)
    RSHOULDER_angles.get_flex_abd(RSHOULDER_coord, RELBOW, plane_seq=['xy', 'yz'])
    RSHOULDER_angles.get_rot(RSHO_b, RSHO_f, RME, RLE)
    print(f'RSHOULDER_angles:\n Flexion: {RSHOULDER_angles.flexion}, \n Abduction: {RSHOULDER_angles.abduction},\n Rotation: {RSHOULDER_angles.rotation}')

    # RHip angles

    # Back angles

    # Head angles



    ######################################## Visual for debugging ########################################
    frame = 1000
    # Point.plot_points([RSHOULDER, RELBOW, LSHOULDER],frame=frame)
    RSHOULDER_angles.plot_angles(joint_name='Right Shoulder')
    plt.show()

