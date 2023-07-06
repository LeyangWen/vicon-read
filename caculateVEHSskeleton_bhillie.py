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
    ##### upper body #####
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

    ##### upper body #####
    RASIS = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RASIS'))
    LASIS = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LASIS'))
    RPSIS = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RPSIS'))
    LPSIS = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LPSIS'))
    LGT = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LGT'))
    # RGT = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RGT'))
    RLFC = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RLFC'))
    RMFC = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RMFC'))
    # LLFC = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LLFC'))
    # LMFC = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LMFC'))

    ######################################## Create virtual markers ########################################
    ##### upper body #####
    RSHOULDER = Point.mid_point(RSHO_f, RSHO_b)
    LSHOULDER = Point.mid_point(LSHO_f, LSHO_b)
    C7_m = Point.mid_point(C7_d, SS)
    # LELBOW = Point.mid_point(LME, LLE)
    RELBOW = Point.mid_point(RME, RLE)

    ##### upper body #####
    PELVIS_f = Point.mid_point(RASIS, LASIS)
    PELVIS_b = Point.mid_point(RPSIS, LPSIS)
    RHIP = Point.translate_point(LGT, Point.vector(RASIS, LASIS, normalize=2*25.4))  # offset 2 inches
    RKNEE = Point.mid_point(RLFC, RMFC)



    ######################################## Calculate angles ########################################
    # RShoulder angles
    try:
        zero_frame = 941
        # RSHOULDER_plane = Plane(RSHO_b, RSHO_f, C7_m)
        RSHOULDER_plane = Plane(RSHOULDER, SS, C7_m)
        RSHOULDER_coord = CoordinateSystem3D()
        RSHOULDER_coord.set_by_plane(RSHOULDER_plane, RSHOULDER, C7_m, sequence='zyx', axis_positive=False)
        RSHOULDER_angles = JointAngles()
        RSHOULDER_angles.set_zero_frame(zero_frame)
        RSHOULDER_angles.get_flex_abd(RSHOULDER_coord, RELBOW, plane_seq=['xy', 'yz'])
        RSHOULDER_angles.get_rot(RSHO_b, RSHO_f, RME, RLE)

        ##### Visual for debugging #####
        frame = 1000
        print(f'RSHOULDER_angles:\n Flexion: {RSHOULDER_angles.flexion[frame]}, \n Abduction: {RSHOULDER_angles.abduction[frame]},\n Rotation: {RSHOULDER_angles.rotation[frame]}')
        # Point.plot_points([RSHOULDER_coord.origin, RSHOULDER_coord.x_axis_end, RSHOULDER_coord.y_axis_end, RSHOULDER_coord.z_axis_end], frame=frame)
        # RSHOULDER_angles.plot_angles(joint_name='Right Shoulder', frame_range=[1000, 1021])
        render_dir = os.path.join(trial_name[0], 'render', trial_name[1], 'RSHOULDER')
        RSHOULDER_angles.plot_angles_by_frame(render_dir, joint_name='Right Shoulder', frame_range=[941, 5756])
    except:
        print('RSHOULDER_angles failed')

    # RHip angles
    try:
        zero_frame = 941
        HIP_plane = Plane(RASIS, LASIS, PELVIS_b)
        HIP_coord = CoordinateSystem3D()
        HIP_coord.set_by_plane(HIP_plane, PELVIS_f, RASIS, sequence='zyx', axis_positive=True)
        RHIP_angles = JointAngles()
        RHIP_angles.set_zero_frame(zero_frame)
        RKNEE_equivalentPt = Point.translate_point(PELVIS_f, Point.vector(RHIP, RKNEE), 100)
        RHIP_angles.get_flex_abd(HIP_coord, RKNEE_equivalentPt, plane_seq=['xy', 'yz'])

        ##### Visual for debugging #####
        frame = 1000
        print(f'RSHOULDER_angles:\n Flexion: {RSHOULDER_angles.flexion[frame]}, \n Abduction: {RSHOULDER_angles.abduction[frame]},\n Rotation: {RSHOULDER_angles.rotation[frame]}')
        # Point.plot_points([RSHOULDER_coord.origin, RSHOULDER_coord.x_axis_end, RSHOULDER_coord.y_axis_end, RSHOULDER_coord.z_axis_end], frame=frame)
        # RSHOULDER_angles.plot_angles(joint_name='Right Shoulder', frame_range=[1000, 1021])
        render_dir = os.path.join(trial_name[0], 'render', trial_name[1], 'RSHOULDER')
        RSHOULDER_angles.plot_angles_by_frame(render_dir, joint_name='Right Shoulder', frame_range=[941, 5756])

    except:
        print('RHip_angles failed')

    # Back angles



    # Head angles

