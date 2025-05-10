from ForceTorqueTransducer import *
from viconnexusapi import ViconNexus
import numpy as np
import pandas as pd
import os

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
device_IDs = vicon.GetDeviceIDs()

stacked_df = pd.DataFrame()

for ID in device_IDs:
    device_details = vicon.GetDeviceDetails(ID)
    if device_details[1] == 'ForcePlate':
        sensor = AMTIForcePlate()
    elif device_details[1] == 'Other':
        sensor = ATIMini45()
    else:
        print(f'Unknown device type: {device_details}')
        continue
    sensor.load_nexus_channels(vicon, ID)
    sensor.load_nexus_device_detail(vicon, ID)
    name = f'{sensor.device_type}{sensor.side}'
    df = pd.DataFrame(sensor.force_torque_values)
    print(f"total frames: {len(df)}")
    # column name = name-sensor.channel_names-sensor.units
    df.columns = [f'{name}-{channel}-{unit}' for channel, unit in zip(sensor.channel_names, sensor.units)]
    stacked_df = pd.concat([stacked_df, df], axis=1)



# write pandas to csv of stacked df
output_file_name = os.path.join(trial_name[0], f'{trial_name[1]}.FT.csv')
stacked_df.to_csv(output_file_name, index=False)
print(f'Wrote {output_file_name}')

