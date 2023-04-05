from ForceTorqueTransducer import ATIMini45
from viconnexusapi import ViconNexus
import numpy as np

vicon = ViconNexus.ViconNexus()

# read vicon marker values
# dir(ViconNexus.ViconNexus)
# help(vicon.GetDeviceOutputDetails)
# name, device_type, rate, deviceOutputIDs, forceplate, eyetracker = vicon.GetDeviceDetails(4)
# name, output_type, unit, ready, channelNames, channelIDs = vicon.GetDeviceOutputDetails(4,1)

trial_name = vicon.GetTrialName()
subject_names = vicon.GetSubjectNames()
frame_count = vicon.GetFrameCount()
frame_rate = vicon.GetFrameRate()
subject_info = vicon.GetSubjectInfo()

forceplate_fz = np.array(vicon.GetDeviceChannel(2, 1, 3)[0])
gain_voltage = []
for i in range(1,8):
    gain_voltage.append(vicon.GetDeviceChannel(4, 1, i)[0])
gain_voltage = np.array(gain_voltage).T
zero_frame = 10
transducer = ATIMini45(gain_voltage, zero_frame)
# transducer.conversion_math_test()


