from ForceTorqueTransducer import ATIMini45
from viconnexusapi import ViconNexus
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

vicon = ViconNexus.ViconNexus()

# read vicon marker values
# dir(ViconNexus.ViconNexus)
# help(vicon.GetDeviceIDs)
# vicon.GetDeviceIDs()
# name, device_type, rate, deviceOutputIDs, forceplate, eyetracker = vicon.GetDeviceDetails(4)
# name, output_type, unit, ready, channelNames, channelIDs = vicon.GetDeviceOutputDetails(3,1)

trial_name = vicon.GetTrialName()
subject_names = vicon.GetSubjectNames()
frame_count = vicon.GetFrameCount()
frame_rate = vicon.GetFrameRate()
subject_info = vicon.GetSubjectInfo()

device_ID = 6
print(vicon.GetDeviceDetails(device_ID))
forceplate_Fz = np.array(vicon.GetDeviceChannel(device_ID, 1, 3)[0])
forceplate_Fy = -np.array(vicon.GetDeviceChannel(device_ID, 1, 2)[0])
forceplate_Fx = np.array(vicon.GetDeviceChannel(device_ID, 1, 1)[0])
forceplate_Tz = np.array(vicon.GetDeviceChannel(device_ID, 2, 3)[0])
forceplate_Ty = -np.array(vicon.GetDeviceChannel(device_ID, 2, 2)[0])
forceplate_Tx = np.array(vicon.GetDeviceChannel(device_ID, 2, 1)[0])
gain_voltage = []
for i in range(1,8):
    gain_voltage.append(vicon.GetDeviceChannel(4, 1, i)[0])
gain_voltage = np.array(gain_voltage).T
transducer = ATIMini45(gain_voltage)
# transducer.conversion_math_test()


# plot forceplate_fz and transducer.Fz in the same figure
figureX = plt.figure()
figureY = plt.figure()
figureZ = plt.figure()
axX = figureX.add_subplot(111)
axY = figureY.add_subplot(111)
axZ = figureZ.add_subplot(111)
axZ.plot(forceplate_Fz, label='forceplate_Fz')
axY.plot(forceplate_Fy, label='forceplate_Fy')
axX.plot(forceplate_Fx, label='forceplate_Fx')
axZ.plot(transducer.Fz, label='transducer_Fz')
axX.plot(transducer.Fx, label ='transducer_Fx')
axY.plot(transducer.Fy, label ='transducer_Fy')
axX.legend()
axY.legend()
axZ.legend()
# x label frames, y label Force (N)
axX.set_xlabel('Frame (100 fps)')
axX.set_ylabel('Force (N)')
axY.set_xlabel('Frame (100 fps)')
axY.set_ylabel('Force (N)')
axZ.set_xlabel('Frame (100 fps)')
axZ.set_ylabel('Force (N)')
# title
axX.set_title('Force X')
axY.set_title('Force Y')
axZ.set_title('Force Z')
figureX.savefig('output/Force_X.png')
figureY.savefig('output/Force_Y.pdf')
figureZ.savefig('output/Force_Z.pdf')

figureTX = plt.figure()
figureTY = plt.figure()
figureTZ = plt.figure()
axTX = figureTX.add_subplot(111)
axTY = figureTY.add_subplot(111)
axTZ = figureTZ.add_subplot(111)
axTZ.plot(forceplate_Tz, label='forceplate_Tz')
axTY.plot(forceplate_Ty, label='forceplate_Ty')
axTX.plot(forceplate_Tx, label='forceplate_Tx')
axTZ.plot(transducer.Tz, label='transducer_Tz')
axTX.plot(transducer.Tx, label ='transducer_Tx')
axTY.plot(transducer.Ty, label ='transducer_Ty')
axTX.legend()
axTY.legend()
axTZ.legend()
# x label frames, y label Force (N)
axTX.set_xlabel('Frame (100 fps)')
axTX.set_ylabel('Force (N)')
axTY.set_xlabel('Frame (100 fps)')
axTY.set_ylabel('Force (N)')
axTZ.set_xlabel('Frame (100 fps)')
axTZ.set_ylabel('Force (N)')
# title
axTX.set_title('Torque X')
axTY.set_title('Torque Y')
axTZ.set_title('Torque Z')
figureTX.savefig('output/Torque_X.png')
figureTY.savefig('output/Torque_Y.pdf')
figureTZ.savefig('output/Torque_Z.pdf')

plt.show()

# export all data to one excel

force_data = {'forceplate_Fx': forceplate_Fx, 'forceplate_Fy': forceplate_Fy, 'forceplate_Fz': forceplate_Fz,
              'transducer_Fx': transducer.Fx, 'transducer_Fy': transducer.Fy, 'transducer_Fz': transducer.Fz}

torque_data = {'forceplate_Mx': forceplate_Tx, 'forceplate_My': forceplate_Ty, 'forceplate_Mz': forceplate_Tz,
               'transducer_Tx': transducer.Tx, 'transducer_Ty': transducer.Ty, 'transducer_Tz': transducer.Tz}

force_df = pd.DataFrame(force_data)
torque_df = pd.DataFrame(torque_data)
# write to excel
writer = pd.ExcelWriter('output/force_torque_data.xlsx', engine='xlsxwriter')
force_df.to_excel(writer, sheet_name='force')
torque_df.to_excel(writer, sheet_name='torque')
writer.save()


