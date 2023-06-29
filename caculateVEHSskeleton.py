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


# helper functions
# vicon = ViconNexus.ViconNexus()
# dir(ViconNexus.ViconNexus)
# help(vicon.GetSubjectParam)

if __name__ == '__main__':
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

    # create markers
    vicon.CreateModeledMarker(subject_names[0], 'LEYE')
    vicon.CreateModeledMarker(subject_names[0], 'REYE')
    vicon.CreateModeledMarker(subject_names[0], 'LPKO')
    vicon.CreateModeledMarker(subject_names[0], 'RPKO')
    vicon.CreateModeledMarker(subject_names[0], 'LMFO')
    vicon.CreateModeledMarker(subject_names[0], 'RMFO')
    vicon.CreateModeledMarker(subject_names[0], 'NKTP')
    # vicon.CreateModeledMarker(subject_names[0], 'NKBT')
    vicon.CreateModeledMarker(subject_names[0], 'FHEC')
    vicon.CreateModeledMarker(subject_names[0], 'BHEC')
    vicon.CreateModeledMarker(subject_names[0], 'RHEC')
    vicon.CreateModeledMarker(subject_names[0], 'LHEC')
    vicon.CreateModeledMarker(subject_names[0], 'HTPO')
    vicon.CreateModeledMarker(subject_names[0], 'HDCC')

    # get reference value
    RFHD = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RFHD'))
    LFHD = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LFHD'))
    RBHD = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RBHD'))
    LBHD = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LBHD'))
    HDEY = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'HDEY'))
    HDTP = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'HDTP'))
    RNKT = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RNKT'))
    LNKT = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LNKT'))
    # C7 = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'C7'))

    LPIK = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LPIK'))
    LFIN = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LFIN'))
    LWRA = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LWRA'))
    LWRB = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LWRB'))

    RPIK = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RPIK'))
    RFIN = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RFIN'))
    RWRA = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RWRA'))
    RWRB = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RWRB'))

    LeftPinkyWidth = vicon.GetSubjectParam(subject_names[0],'LeftPinkyWidth')[0]
    RightPinkyWidth = vicon.GetSubjectParam(subject_names[0],'RightPinkyWidth')[0]
    EyeWidth = vicon.GetSubjectParam(subject_names[0],'EyeWidth')[0]
    RightHandThickness = vicon.GetSubjectParam(subject_names[0],'RightHandThickness')[0]
    LeftHandThickness = vicon.GetSubjectParam(subject_names[0],'LeftHandThickness')[0]

    # CALCULATIONS
    # head centers
    FHEC = Point.mid_point(RFHD, LFHD)  # FHDC is midpoint of RFHD and LFHD
    BHEC = Point.mid_point(RBHD, LBHD)  # BHDC is midpoint of RBHD and LBHD
    RHDC = Point.mid_point(RFHD, RBHD)  # RHDC is midpoint of RFHD and RBHD
    LHDC = Point.mid_point(LFHD, LBHD)  # LHDC is midpoint of LFHD and LBHD
    HDCC = Point.mid_point(FHEC, BHEC)  # HDCC is midpoint of all four head markers

    HDCC_HDTP = Point.vector(HDCC, HDTP, normalize=marker_height)
    HTPO = Point.translate_point(HDTP, HDCC_HDTP, -1)  # HTPC is HDTP moving downwards of marker height

    # eyes
    LFHD_RFHD = Point.vector(LFHD, RFHD, normalize=EyeWidth/2)
    try:
        LEYE = Point.translate_point(HDEY, LFHD_RFHD, -1)  # left eye is HEEY moving left of eye width/2 in the direction of RFHD-->LFHD
        REYE = Point.translate_point(HDEY, LFHD_RFHD, 1)  # right eye is HEEY moving right of eye width/2 in the direction of RFHD-->LFHD
    except:
        LEYE = LFHD
        REYE = RFHD
        HDEY = Point.mid_point(LEYE, REYE)

    # neck
    NKTP = Point.mid_point(LNKT, RNKT) # neck top is midpoint of LNKT and RNKT

    # fingers
    RPIK_RPKO = Point.orthogonal_vector(RPIK, RWRB, RWRA, normalize=RightPinkyWidth/2+marker_height)
    LPIK_LPKO = Point.orthogonal_vector(LPIK, LWRA, LWRB, normalize=LeftPinkyWidth/2+marker_height)
    RPKO = Point.translate_point(RPIK, RPIK_RPKO)  # right pinky is RPIK moved inward with marker height and pinky width
    LPKO = Point.translate_point(LPIK, LPIK_LPKO)  # left pinky is LPIK moved inward with marker height and pinky width

    RFIN_RMFO = Point.orthogonal_vector(RFIN, RWRB, RWRA, normalize=RightHandThickness/2+marker_height)
    LFIN_LMFO = Point.orthogonal_vector(LFIN, LWRA, LWRB, normalize=LeftHandThickness/2+marker_height)
    RMFO = Point.translate_point(RFIN, RFIN_RMFO)  # right mid finger is RPIK moved inward with marker height and hand width
    LMFO = Point.translate_point(LFIN, LFIN_LMFO)  # left mid finger is LPIK moved inward with marker height and hand width

    # output to vicon nexus
    vicon.SetModelOutput(subject_names[0], 'LEYE', LEYE.xyz, LEYE.exist)
    vicon.SetModelOutput(subject_names[0], 'REYE', REYE.xyz, REYE.exist)
    vicon.SetModelOutput(subject_names[0], 'LPKO', LPKO.xyz, LPKO.exist)
    vicon.SetModelOutput(subject_names[0], 'RPKO', RPKO.xyz, RPKO.exist)
    vicon.SetModelOutput(subject_names[0], 'LMFO', LMFO.xyz, LMFO.exist)
    vicon.SetModelOutput(subject_names[0], 'RMFO', RMFO.xyz, RMFO.exist)
    vicon.SetModelOutput(subject_names[0], 'NKTP', NKTP.xyz, NKTP.exist)
    # vicon.SetModelOutput(subject_names[0], 'NKBT', NKBT.xyz, NKBT.exist)
    vicon.SetModelOutput(subject_names[0], 'FHEC', FHEC.xyz, FHEC.exist)
    vicon.SetModelOutput(subject_names[0], 'BHEC', BHEC.xyz, BHEC.exist)
    vicon.SetModelOutput(subject_names[0], 'RHEC', RHDC.xyz, RHDC.exist)
    vicon.SetModelOutput(subject_names[0], 'LHEC', LHDC.xyz, LHDC.exist)
    vicon.SetModelOutput(subject_names[0], 'HDCC', HDCC.xyz, HDCC.exist)
    vicon.SetModelOutput(subject_names[0], 'HTPO', HTPO.xyz, HTPO.exist)


    # visual for debugging
    # frame = 0
    # Point.plot_points([LWRB,LWRA,LPIK,LFIN,LPKO,LMFO],frame=frame)
    # Point.plot_points([RWRB,RWRA,RPIK,RFIN,RPKO,RMFO],frame=frame)

