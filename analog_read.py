from ForceTorqueTransducer import ATIMini45
from viconnexusapi import ViconNexus
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def peak_compare(forceplate_data, transducer_data, unit='N'):
    try:
        # get the index max abs value transducer
        max_idx = np.argmax(np.abs(transducer_data))
        # find abs diff between forceplate and transducer at that index
        diff = np.abs(forceplate_data[max_idx] - transducer_data[max_idx])
        # find % diff between forceplate and transducer at that index
        percent_diff = abs(diff / transducer_data[max_idx])

        msg = f'Peak diff: {diff:.1f}{unit} or {percent_diff*100:.1f}%'

        return diff, percent_diff, max_idx, msg
    except:
        return None, None, None, ''




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

print(vicon.GetDeviceIDs())
# device_ID = 2 # 08
# device_ID = 3
# device_ID = 6 # 14 17 19*
device_ID = 7 # 09* 18 20
print(vicon.GetDeviceDetails(device_ID))
trail_info = {'trial_name': trial_name, 'frame_count': frame_count, 'frame_rate': frame_rate,
              'forceplate name': vicon.GetDeviceDetails(device_ID)[0]}


forceplate_Fz = np.array(vicon.GetDeviceChannel(device_ID, 1, 3)[0])
forceplate_Fy = -np.array(vicon.GetDeviceChannel(device_ID, 1, 2)[0])
forceplate_Fx = np.array(vicon.GetDeviceChannel(device_ID, 1, 1)[0])
forceplate_Tz = np.array(vicon.GetDeviceChannel(device_ID, 2, 3)[0])/1000
forceplate_Ty = -np.array(vicon.GetDeviceChannel(device_ID, 2, 2)[0])/1000
forceplate_Tx = np.array(vicon.GetDeviceChannel(device_ID, 2, 1)[0])/1000
gain_voltage = []
for i in range(1,8):
    gain_voltage.append(vicon.GetDeviceChannel(4, 1, i)[0])
gain_voltage = np.array(gain_voltage).T
transducer = ATIMini45(gain_voltage)
# transducer.conversion_math_test()

if forceplate_Fz.shape[0] == 0:
    print('No forceplate data found')
    forceplate_collected = False
else:
    forceplate_collected = True
# plot forceplate_fz and transducer.Fz in the same figure
figureX = plt.figure()
figureY = plt.figure()
figureZ = plt.figure()
axX = figureX.add_subplot(111)
axY = figureY.add_subplot(111)
axZ = figureZ.add_subplot(111)
if forceplate_collected:
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
unit = 'N'
axX.set_title(f'Force X {peak_compare(forceplate_Fx,transducer.Fx,unit)[-1]}')
axY.set_title(f'Force Y {peak_compare(forceplate_Fy,transducer.Fy,unit)[-1]}')
axZ.set_title(f'Force Z {peak_compare(forceplate_Fz,transducer.Fz,unit)[-1]}')
figureX.savefig('output/Force_X.png')
figureY.savefig('output/Force_Y.png')
figureZ.savefig('output/Force_Z.png')

figureTX = plt.figure()
figureTY = plt.figure()
figureTZ = plt.figure()
axTX = figureTX.add_subplot(111)
axTY = figureTY.add_subplot(111)
axTZ = figureTZ.add_subplot(111)
if forceplate_collected:
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


#

axTX.set_xlabel('Frame (100 fps)')
axTX.set_ylabel('Torque (Nm)')
axTY.set_xlabel('Frame (100 fps)')
axTY.set_ylabel('Torque (Nm)')
axTZ.set_xlabel('Frame (100 fps)')
axTZ.set_ylabel('Torque (Nm)')
# title
unit = 'Nm'
axTX.set_title(f'Torque X {peak_compare(forceplate_Tx,transducer.Tx,unit=unit)[-1]}')
axTY.set_title(f'Torque Y {peak_compare(forceplate_Ty,transducer.Ty,unit=unit)[-1]}')
axTZ.set_title(f'Torque Z {peak_compare(forceplate_Tz,transducer.Tz,unit=unit)[-1]}')
figureTX.savefig('output/Torque_X.png')
figureTY.savefig('output/Torque_Y.png')
figureTZ.savefig('output/Torque_Z.png')



# export all data to one excel
if forceplate_collected:
    force_data = {'forceplate_Fx': forceplate_Fx, 'forceplate_Fy': forceplate_Fy, 'forceplate_Fz': forceplate_Fz,
                  'transducer_Fx': transducer.Fx, 'transducer_Fy': transducer.Fy, 'transducer_Fz': transducer.Fz}

    torque_data = {'forceplate_Mx': forceplate_Tx, 'forceplate_My': forceplate_Ty, 'forceplate_Mz': forceplate_Tz,
                   'transducer_Tx': transducer.Tx, 'transducer_Ty': transducer.Ty, 'transducer_Tz': transducer.Tz}
else:
    force_data = {'transducer_Fx': transducer.Fx, 'transducer_Fy': transducer.Fy, 'transducer_Fz': transducer.Fz}
    torque_data = {'transducer_Tx': transducer.Tx, 'transducer_Ty': transducer.Ty, 'transducer_Tz': transducer.Tz}

force_df = pd.DataFrame.from_dict(force_data, orient='index').transpose()
torque_df = pd.DataFrame.from_dict(torque_data, orient='index').transpose()
trail_info_df = pd.DataFrame.from_dict(trail_info)
# write to excel
writer = pd.ExcelWriter('output/force_torque_data.xlsx', engine='xlsxwriter')
force_df.to_excel(writer, sheet_name='force')
torque_df.to_excel(writer, sheet_name='torque')
trail_info_df.to_excel(writer, sheet_name='trail_info')

plt.show()
writer._save()
print('Data saved to output/force_torque_data.xlsx')


