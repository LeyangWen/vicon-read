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
    # vicon.CreateModeledMarker(subject_names[0], 'LEYE')
    #
    # get reference value
    # upper body
    LMCP2 = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LMCP2'))
    LMCP5 = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LMCP5'))
    RMCP2 = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RMCP2'))
    RMCP5 = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RMCP5'))
    LRS = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LRS'))
    LUS = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LUS'))
    RRS = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RRS'))
    RUS = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RUS'))
    LME = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LME'))
    LLE = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LLE'))
    RME = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RME'))
    RLE = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RLE'))

    # lower body
    LMTP1 = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LMTP1'))
    LMTP5 = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LMTP5'))
    RMTP1 = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RMTP1'))
    RMTP5 = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RMTP5'))
    LMM = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LMM'))
    LLM = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LLM'))
    RMM = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RMM'))
    RLM = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RLM'))
    LLTC = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LLTC'))
    LLFC = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LLFC'))
    LMTC = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LMTC'))
    LMFC = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LMFC'))
    RLTC = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RLTC'))
    RLFC = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RLFC'))
    RMTC = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RMTC'))
    RMFC = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RMFC'))





    # LeftPinkyWidth = vicon.GetSubjectParam(subject_names[0],'LeftPinkyWidth')[0]
    # RightPinkyWidth = vicon.GetSubjectParam(subject_names[0],'RightPinkyWidth')[0]
    # EyeWidth = vicon.GetSubjectParam(subject_names[0],'EyeWidth')[0]
    # RightHandThickness = vicon.GetSubjectParam(subject_names[0],'RightHandThickness')[0]
    # LeftHandThickness = vicon.GetSubjectParam(subject_names[0],'LeftHandThickness')[0]

    # CALCULATIONS
    # upper body
    LHAND = Point.mid_point(LMCP2, LMCP5)  # LHAND is midpoint of LMCP2 and LMCP5
    RHAND = Point.mid_point(RMCP2, RMCP5)  # RHAND is midpoint of RMCP2 and RMCP5
    LWRIST = Point.mid_point(LRS, LUS)  # LWRIST is midpoint of LRS and LUS
    RWRIST = Point.mid_point(RRS, RUS)  # RWRIST is midpoint of RRS and RUS
    LELBOW = Point.mid_point(LME, LLE)  # LELBOW is midpoint of LME and LLE
    RELBOW = Point.mid_point(RME, RLE)  # RELBOW is midpoint of RME and RLE

    # lower body
    LFOOT = Point.mid_point(LMTP1, LMTP5)  # LFOOT is midpoint of LMTP1 and LMTP5
    RFOOT = Point.mid_point(RMTP1, RMTP5)  # RFOOT is midpoint of RMTP1 and RMTP5
    LANKLE = Point.mid_point(LMM, LLM)  # LANKLE is midpoint of LMM and LLM
    RANKLE = Point.mid_point(RMM, RLM)  # RANKLE is midpoint of RMM and RLM



    #
    # HDCC_HDTP = Point.vector(HDCC, HDTP, normalize=marker_height)
    # HTPO = Point.translate_point(HDTP, HDCC_HDTP, -1)  # HTPC is HDTP moving downwards of marker height
    #
    # # eyes
    # LFHD_RFHD = Point.vector(LFHD, RFHD, normalize=EyeWidth/2)
    # try:
    #     LEYE = Point.translate_point(HDEY, LFHD_RFHD, -1)  # left eye is HEEY moving left of eye width/2 in the direction of RFHD-->LFHD
    #     REYE = Point.translate_point(HDEY, LFHD_RFHD, 1)  # right eye is HEEY moving right of eye width/2 in the direction of RFHD-->LFHD
    # except:
    #     LEYE = LFHD
    #     REYE = RFHD
    #     HDEY = Point.mid_point(LEYE, REYE)
    #
    # # neck
    # NKTP = Point.mid_point(LNKT, RNKT) # neck top is midpoint of LNKT and RNKT
    #
    # # fingers
    # RPIK_RPKO = Point.orthogonal_vector(RPIK, RWRB, RWRA, normalize=RightPinkyWidth/2+marker_height)
    # LPIK_LPKO = Point.orthogonal_vector(LPIK, LWRA, LWRB, normalize=LeftPinkyWidth/2+marker_height)
    # RPKO = Point.translate_point(RPIK, RPIK_RPKO)  # right pinky is RPIK moved inward with marker height and pinky width
    # LPKO = Point.translate_point(LPIK, LPIK_LPKO)  # left pinky is LPIK moved inward with marker height and pinky width
    #
    # RFIN_RMFO = Point.orthogonal_vector(RFIN, RWRB, RWRA, normalize=RightHandThickness/2+marker_height)
    # LFIN_LMFO = Point.orthogonal_vector(LFIN, LWRA, LWRB, normalize=LeftHandThickness/2+marker_height)
    # RMFO = Point.translate_point(RFIN, RFIN_RMFO)  # right mid finger is RPIK moved inward with marker height and hand width
    # LMFO = Point.translate_point(LFIN, LFIN_LMFO)  # left mid finger is LPIK moved inward with marker height and hand width
    #
    # # output to vicon nexus
    # vicon.SetModelOutput(subject_names[0], 'HTPO', HTPO.xyz, HTPO.exist)
    #
    #
    # # visual for debugging
    # # frame = 0
    # # Point.plot_points([LWRB,LWRA,LPIK,LFIN,LPKO,LMFO],frame=frame)
    # # Point.plot_points([RWRB,RWRA,RPIK,RFIN,RPKO,RMFO],frame=frame)

