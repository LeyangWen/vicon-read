from viconnexusapi import ViconNexus
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
import pickle

from utility import *
from ergo3d import *
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
    HDTP = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'HDTP'))
    LEAR = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LEAR'))
    REAR = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'REAR'))
    MDFH = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'MDFH'))

    C7 = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'C7'))
    C7_d = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'C7_d'))
    SS = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'SS'))
    RSHO_f = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RSHO_f'))
    LSHO_f = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LSHO_f'))
    RSHO_b = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RSHO_b'))
    LSHO_b = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LSHO_b'))
    RAP = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RAP'))
    LAP = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LAP'))
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
    RGT = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RGT'))
    RLFC = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RLFC'))
    RMFC = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RMFC'))
    # LLFC = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LLFC'))
    # LMFC = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LMFC'))
    RMM = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RMM'))
    RLM = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RLM'))
    # LMM = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LMM'))
    # LLM = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LLM'))
    RUS = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RUS'))
    LUS = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LUS'))
    RRS = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RRS'))
    LRS = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LRS'))
    RMCP2 = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RMCP2'))
    LMCP2 = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LMCP2'))
    RMCP5 = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RMCP5'))
    LMCP5 = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LMCP5'))

    ######################################## Create virtual markers ########################################
    ##### upper body #####
    EAR = Point.mid_point(LEAR, REAR)
    RSHOULDER = Point.mid_point(RSHO_f, RSHO_b)
    LSHOULDER = Point.mid_point(LSHO_f, LSHO_b)
    C7_m = Point.mid_point(C7_d, SS)
    # LELBOW = Point.mid_point(LME, LLE)
    RELBOW = Point.mid_point(RME, RLE)
    RWRIST = Point.mid_point(RRS, RUS)
    # LWRIST = Point.mid_point(LRS, LUS)
    RHAND = Point.mid_point(RMCP2, RMCP5)
    # LHAND =

    ##### lower body #####
    PELVIS_f = Point.mid_point(RASIS, LASIS)
    PELVIS_b = Point.mid_point(RPSIS, LPSIS)
    try:
        RHIP = Point.translate_point(RGT, Point.vector(RASIS, LASIS, normalize=2*25.4))  # offset 2 inches
    except:
        print("RHIP error")
    RKNEE = Point.mid_point(RLFC, RMFC)
    RANKLE = Point.mid_point(RMM, RLM)



    ######################################## Calculate angles ########################################

    try:  # RShoulder angles
    # if True:
    #     #  Upper_lim_angles01
    #     PELVIS_b = Point.translate_point(C7, Point.create_const_vector(0,0,-1000,examplePt=C7))
    #     # RSHOULDER = Point.translate_point(RAP, Point.create_const_vector(0,0,-50,examplePt=RAP))
    #     zero_frame = [941, 1320, 941]
    #     frame_range = [941, 5756]
        # shoulder02
        zero_frame = [803, 1019, 803]
        frame_range = [803, 3231]

        RSHOULDER_plane = Plane()
        RSHOULDER_plane.set_by_vector(RSHOULDER, Point.vector(C7_d, PELVIS_b), direction=-1)
        RSHOULDER_C7_m_project = RSHOULDER_plane.project_point(C7_m)
        RSHOULDER_SS_project = RSHOULDER_plane.project_point(SS)
        RSHOULDER_coord = CoordinateSystem3D()
        RSHOULDER_coord.set_by_plane(RSHOULDER_plane, C7_d, RSHOULDER_SS_project, sequence='xyz', axis_positive=True)  # new: use back to chest vector
        # RSHOULDER_coord.set_by_plane(RSHOULDER_plane, RSHOULDER, RSHOULDER_C7_m_project, sequence='zyx', axis_positive=False)  # old: use shoulder to chest vector
        RSHOULDER_angles = JointAngles()
        RSHOULDER_angles.set_zero_frame(zero_frame)
        RSHOULDER_angles.get_flex_abd(RSHOULDER_coord, Point.vector(RSHOULDER, RELBOW), plane_seq=['xy', 'xz'])
        RSHOULDER_angles.get_rot(RSHO_b, RSHO_f, RME, RLE)
        RSHOULDER_angles.flexion = Point.angle(Point.vector(RSHOULDER, RELBOW).xyz, Point.vector(C7, PELVIS_b).xyz)
        RSHOULDER_angles.flexion = RSHOULDER_angles.zero_by_idx(0)  # zero by zero frame after setting flexion without function


        shoulder_threshold = 10/180*np.pi  # the H-abduction is not well defined when the flexion is small or near 180 degrees
        shoulder_filter = np.logical_and(np.abs(RSHOULDER_angles.flexion) > shoulder_threshold, np.abs(RSHOULDER_angles.flexion) < (np.pi - shoulder_threshold))
        RSHOULDER_angles.abduction = np.array([np.where(shoulder_filter[i], RSHOULDER_angles.abduction[i], 0) for i in range(len(shoulder_filter))])  # set abduction to nan if shoulder filter is false

        ##### Visual for debugging #####
        frame = 1000
        # print(f'RSHOULDER_angles:\n Flexion: {RSHOULDER_angles.flexion[frame]}, \n Abduction: {RSHOULDER_angles.abduction[frame]},\n Rotation: {RSHOULDER_angles.rotation[frame]}')
        # Point.plot_points([
        #                    RSHOULDER_coord.origin, RSHOULDER_coord.x_axis_end, RSHOULDER_coord.y_axis_end, RSHOULDER_coord.z_axis_end,
        #                    RSHOULDER, C7_m, RSHOULDER_C7_m_project, RELBOW, RSHO_f, RSHO_b,
        #                    ], frame=frame)
        # RSHOULDER_angles.plot_angles(joint_name='Right Shoulder', frame_range=frame_range)
        render_dir = os.path.join(trial_name[0], 'render', trial_name[1], 'RSHOULDER_1_all_filtered')
        RSHOULDER_angles.plot_angles_by_frame(render_dir, joint_name='Right Shoulder', frame_range=frame_range, angle_names=['Flexion', 'H-Abduction', 'Rotation'])
        print('**** RSHOULDER_angles done ****')
    except:
        print('RSHOULDER_angles failed')

    # try:  # Head angles
    # # if True:
    #     zero_frame = [1093, 1093, 1093]
    #     frame_range = [1093, 4007]
    #     HEAD_plane = Plane()
    #     HEAD_plane.set_by_pts(REAR, LEAR, HDTP)
    #     HEAD_coord = CoordinateSystem3D()
    #     HEAD_coord.set_by_plane(HEAD_plane, EAR, HDTP, sequence='yxz', axis_positive=True)
    #     HEAD_angles = JointAngles()
    #     HEAD_angles.set_zero(zero_frame)
    #     HEAD_angles.get_flex_abd(HEAD_coord, Point.vector(C7, Point.mid_point(RPSIS, LPSIS)), plane_seq=['xy', 'yz'], flip_sign=[1, -1])
    #     HEAD_angles.get_rot(LEAR, REAR, LAP, RAP)
    #
    #     frame = 1286
    #     # Point.plot_points([
    #     #                    HEAD_coord.origin, HEAD_coord.x_axis_end, HEAD_coord.y_axis_end, HEAD_coord.z_axis_end,
    #     #                    C7, HDTP, LAP, RAP, MDFH, LEAR, REAR,
    #     #                    ], frame=frame)
    #     HEAD_angles.plot_angles(joint_name='Right Head', frame_range=frame_range)
    #     # render_dir = os.path.join(trial_name[0], 'render', trial_name[1], 'HEAD')
    #     # HEAD_angles.plot_angles_by_frame(render_dir, joint_name='Right Head', frame_range=frame_range, angle_names=['Flexion', 'Lateral Bend', 'Rotation'])
    #     print('**** HEAD_angles done ****')
    # except:
    #     print('HEAD_angles failed')
    #
    # try:  # RKnee angles
    # # if True:
    #     zero_frame = [1215, None, None]
    #     frame_range = [1215, 3414]
    #     RKNEE_angles = JointAngles()
    #     RKNEE_angles.set_zero(zero_frame)
    #     RKNEE_angles.flexion = Point.angle(Point.vector(RKNEE, RHIP).xyz, Point.vector(RKNEE, RANKLE).xyz)
    #     RKNEE_angles.flexion = RKNEE_angles.zero_by_idx(0)  # zero by zero frame
    #     RKNEE_angles.is_empty = False
    #     RKNEE_angles.abduction = None
    #     RKNEE_angles.rotation = None
    #     RKNEE_angles.plot_angles(joint_name='Right Knee', frame_range=frame_range)
    #     # render_dir = os.path.join(trial_name[0], 'render', trial_name[1], 'RKNEE')
    #     # RKNEE_angles.plot_angles_by_frame(render_dir, joint_name='Right Knee', frame_range=frame_range)
    #     print('**** RKnee_angles done ****')
    # except:
    #     print('RKnee_angles failed')
    #
    # try:  # RElbow angles
    # # if True:
    #     zero_frame = [889, None, None]
    #     frame_range = [889, 1837]
    #     RELBOW_angles = JointAngles()
    #     RELBOW_angles.set_zero(zero_frame)
    #     RELBOW_angles.flexion = Point.angle(Point.vector(RELBOW, RSHOULDER).xyz, Point.vector(RELBOW, RWRIST).xyz)
    #     RELBOW_angles.flexion = RELBOW_angles.zero_by_idx(0)  # zero by zero frame
    #     RELBOW_angles.is_empty = False
    #     RELBOW_angles.abduction = None
    #     RELBOW_angles.rotation = None
    #     RELBOW_angles.plot_angles(joint_name='Right Elbow', frame_range=frame_range)
    #     # render_dir = os.path.join(trial_name[0], 'render', trial_name[1], 'RELBOW')
    #     # RELBOW_angles.plot_angles_by_frame(render_dir, joint_name='Right Elbow', frame_range=frame_range)
    #     print('**** RElbow_angles done ****')
    # except:
    #     print('RElbow_angles failed')
    #
    try:  # RWrist angles
    # if True:
        zero_frame = [1369, 1369, None]  #01
        frame_range = [1369, 3115]
        zero_frame = [1165, 1165, None]  #02
        frame_range = [1165, 2747]
        zero_frame = [1318, 1318, 2882]  #03
        frame_range = [1318, 3386]

        # # set by elbow
        # RWRIST_plane = Plane()
        # RWRIST_plane.set_by_pts(RELBOW, RRS, RUS)
        # RWRIST_coord = CoordinateSystem3D()
        # RWRIST_coord.set_by_plane(RWRIST_plane, RWRIST, RELBOW, sequence='yxz', axis_positive=True)
        # RWRIST_angles = JointAngles()
        # RWRIST_angles.set_zero(zero_frame)
        # RWRIST_angles.get_flex_abd(RWRIST_coord, Point.vector(RWRIST, RHAND), plane_seq=['xy', 'yz'])

        # set by hand
        RWRIST_plane = Plane()
        RWRIST_plane.set_by_pts(RMCP2, RWRIST, RMCP5)
        RWRIST_coord = CoordinateSystem3D()
        RWRIST_coord.set_by_plane(RWRIST_plane, RWRIST, RHAND, sequence='yxz', axis_positive=True)
        RWRIST_angles = JointAngles()
        RWRIST_angles.set_zero_frame(zero_frame)
        RWRIST_angles.get_flex_abd(RWRIST_coord, Point.vector(RWRIST, RELBOW), plane_seq=['xy', 'yz'])
        RWRIST_angles.get_rot(RRS, RUS, RLE, RME)

        # RWRIST_angles.rotation = None
        # frame = 3000
        # Point.plot_points([
        #                     RWRIST_coord.origin, RWRIST_coord.x_axis_end, RWRIST_coord.y_axis_end, RWRIST_coord.z_axis_end,
        #                     RWRIST, RELBOW, RRS, RUS, RHAND
        #                     ], frame=frame)
        RWRIST_angles.plot_angles(joint_name='Right Wrist', frame_range=frame_range)
        # render_dir = os.path.join(trial_name[0], 'render', trial_name[1], 'RWRIST')
        # RWRIST_angles.plot_angles_by_frame(render_dir, joint_name='Right Wrist', frame_range=frame_range, angle_names=['Flexion', 'Deviation', 'Rotation'])
        print('**** RWrist_angles done ****')
    except:
        print('RWRIST_angles failed')
    #
    # try:  # Back angles: Back02
    # # if True:
    #     zero_frame = [756, 756, 756]
    #     frame_range = [756, 3421]
    #     PELVIS_b = Point.mid_point(RPSIS, LPSIS)
    #     BACK_plane = Plane()
    #     BACK_plane.set_by_vector(PELVIS_b, Point.create_const_vector(0, 0, 1000, examplePt=PELVIS_b),direction=1)
    #     BACK_coord = CoordinateSystem3D()
    #     BACK_coord.set_by_plane(BACK_plane, PELVIS_b, RPSIS, sequence='zyx', axis_positive=True)
    #     BACK_angles = JointAngles()
    #     BACK_angles.set_zero(zero_frame)
    #     BACK_angles.get_flex_abd(BACK_coord, Point.vector(PELVIS_b, C7), plane_seq=['xy', 'yz'])
    #     BACK_angles.get_rot(RAP, LAP, RPSIS, LPSIS, flip_sign=1)
    #     frame = 3000
    #     # Point.plot_points([
    #     #                     RWRIST_coord.origin, RWRIST_coord.x_axis_end, RWRIST_coord.y_axis_end, RWRIST_coord.z_axis_end,
    #     #                     RWRIST, RELBOW, RRS, RUS, RHAND
    #     #                     ], frame=frame)
    #     BACK_angles.plot_angles(joint_name='Back', frame_range=frame_range)
    #     # render_dir = os.path.join(trial_name[0], 'render', trial_name[1], 'BACK')
    #     # BACK_angles.plot_angles_by_frame(render_dir, joint_name='Back', frame_range=frame_range, angle_names=['Flexion', 'L-Flexion', 'Rotation'])
    #     print('**** Back_angles done ****')
    # except:
    #     print('Back_angles failed')