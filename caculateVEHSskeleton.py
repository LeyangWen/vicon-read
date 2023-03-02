from viconnexusapi import ViconNexus
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
import pickle
from utility import *
from Point import *

vicon = ViconNexus.ViconNexus()


# read vicon marker values
# dir(ViconNexus.ViconNexus)
# help(vicon.SetSubjectParam)

trial_name = vicon.GetTrialName()
subject_names = vicon.GetSubjectNames()
frame_count = vicon.GetFrameCount()
frame_rate = vicon.GetFrameRate()
subject_info = vicon.GetSubjectInfo()

# if more than one subject, report not implemented error
if len(subject_names) > 1:
    print(subject_names)
    raise NotImplementedError(f"More than one subject not implemented --> {subject_names}")

# create markers
vicon.CreateModeledMarker(subject_names[0], 'LEYE')
vicon.CreateModeledMarker(subject_names[0], 'REYE')
vicon.CreateModeledMarker(subject_names[0], 'LPKO')
vicon.CreateModeledMarker(subject_names[0], 'RPKO')
vicon.CreateModeledMarker(subject_names[0], 'NKTP')
vicon.CreateModeledMarker(subject_names[0], 'NKBT')
vicon.CreateModeledMarker(subject_names[0], 'FHEC')
vicon.CreateModeledMarker(subject_names[0], 'BHEC')
vicon.CreateModeledMarker(subject_names[0], 'RHEC')
vicon.CreateModeledMarker(subject_names[0], 'LHEC')
vicon.CreateModeledMarker(subject_names[0], 'HDEC')

# get reference value
RFHD = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RFHD'))
LFHD = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LFHD'))
RBHD = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RBHD'))
LBHD = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LBHD'))
HDEY = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'HDEY'))
HDTP = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'HDTP'))
RNKT = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RNKT'))
LNKT = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LNKT'))

LPIK = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LPIK'))
LWRA = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LWRA'))
LFIN = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'LFIN'))

RPIK = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RPIK'))
RWRA = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RWRA'))
RWRB = MarkerPoint(vicon.GetTrajectory(subject_names[0], 'RWRB'))

LeftPinkyWidth = vicon.GetSubjectParam(subject_names[0],'LeftPinkyWidth')
RightPinkyWidth = vicon.GetSubjectParam(subject_names[0],'RightPinkyWidth')
EyeWidth = vicon.GetSubjectParam(subject_names[0],'EyeWidth')




# head centers

FHDC = Point.mid_point(RFHD, LFHD)  # FHDC is midpoint of RFHD and LFHD
BHDC = Point.mid_point(RBHD, LBHD)  # BHDC is midpoint of RBHD and LBHD
RHDC = Point.mid_point(RFHD, RBHD)  # RHDC is midpoint of RFHD and RBHD
LHDC = Point.mid_point(LFHD, LBHD)  # LHDC is midpoint of LFHD and LBHD
HDCC = Point.mid_point(FHDC, BHDC)  # HDCC is midpoint of all four head markers
# todo: HTPC is HDTP moving downwards of marker height
# HTPC = HDTP

# eyes
LFHD_RFHD = Point.vector(LFHD, RFHD, normalize=EyeWidth[0]/2)
LEYE = Point.translate_point(HDEY, LFHD_RFHD, -1)  # left eye is HEEY moving left of eye width/2 in the direction of RFHD-->LFHD
REYE = Point.translate_point(HDEY, LFHD_RFHD, 1)  # right eye is HEEY moving right of eye width/2 in the direction of RFHD-->LFHD

# neck
NKTP = Point.mid_point(LNKT, RNKT) # neck top is midpoint of LNKT and RNKT
# todo: neck bottom is c7 moved inward
# NKBT = c7

# finguers
# todo: left pinky is LPIK moved inward with marker width
# todo: right pinky is RPIK moved inward with marker width


vicon.SetModelOutput(subject_names[0], 'LEYE', LEYE.xyz, LEYE.exist)
vicon.SetModelOutput(subject_names[0], 'REYE', REYE.xyz, REYE.exist)
# vicon.SetModelOutput(subject_names[0], 'LPKO', LPKO.xyz, LPKO.exist)
# vicon.SetModelOutput(subject_names[0], 'RPKO', RPKO.xyz, RPKO.exist)
vicon.SetModelOutput(subject_names[0], 'NKTP', NKTP.xyz, NKTP.exist)
# vicon.SetModelOutput(subject_names[0], 'NKBT', NKBT.xyz, NKBT.exist)
# vicon.SetModelOutput(subject_names[0], 'FHDC', FHDC.xyz, FHDC.exist)
# vicon.SetModelOutput(subject_names[0], 'BHDC', BHDC.xyz, BHDC.exist)
# vicon.SetModelOutput(subject_names[0], 'RHDC', RHDC.xyz, RHDC.exist)
# vicon.SetModelOutput(subject_names[0], 'LHDC', LHDC.xyz, LHDC.exist)
# vicon.SetModelOutput(subject_names[0], 'HDCC', HDCC.xyz, HDCC.exist)






