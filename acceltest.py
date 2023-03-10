from viconnexusapi import ViconNexus
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
import pickle
import os

vicon = ViconNexus.ViconNexus()


# read vicon marker values
# dir(ViconNexus.ViconNexus)
# help(vicon.SetSubjectParam)

trial_name = vicon.GetTrialName()
subject_names = vicon.GetSubjectNames()
frame_count = vicon.GetFrameCount()
frame_rate = vicon.GetFrameRate()
subject_info = vicon.GetSubjectInfo()



figure2CDF = plt.figure()
figure3CDF = plt.figure()
figure3CDF2nd = plt.figure()
ax2CDF = figure2CDF.add_subplot(111)
ax3CDF = figure3CDF.add_subplot(111)
ax3CDF2nd = figure3CDF2nd.add_subplot(111)
# title
ax2CDF.set_title('Speed CDF')
ax3CDF.set_title('Acceleration CDF')
ax3CDF2nd.set_title('Acceleration CDF 2nd')
# label
ax2CDF.set_xlabel('Frame (100 fps)')
ax3CDF.set_xlabel('Frame (100 fps)')
ax3CDF2nd.set_xlabel('Frame (100 fps)')
ax2CDF.set_ylabel('Speed (m/s)')
ax3CDF.set_ylabel('Acceleration (m/s^2)')
ax3CDF2nd.set_ylabel('Acceleration (m/s^2)')


figure1 = plt.figure()
figure2 = plt.figure()
figure3 = plt.figure()
ax1 = figure1.add_subplot(111)
ax2 = figure2.add_subplot(111)
ax3 = figure3.add_subplot(111)
# title
ax1.set_title('Position')
ax2.set_title('Speed')
ax3.set_title('Acceleration')
# label
ax1.set_xlabel('Frame (100 fps)')
ax2.set_xlabel('Frame (100 fps)')
ax3.set_xlabel('Frame (100 fps)')
ax1.set_ylabel('Position (m)')
ax2.set_ylabel('Speed (m/s)')
ax3.set_ylabel('Acceleration (m/s^2)')
start_idxs = []
drop_caculations = []
# plot the 3D trajectory of the unlabeled markers
for i in range(vicon.GetUnlabeledCount()):
    drop_caculation = {}
    trajectory = vicon.GetUnlabeled(i)
    trajectory = np.array(trajectory)
    trajectory = trajectory/1000 # convert to meter
    drops = []
    try:
        start_idx = np.where(trajectory[2] > 1.65)[0][-1]-50
    except:
        print(i)
        continue

    end_idx = start_idx + 100
    start_idxs.append(start_idx)
    drop = trajectory[2][start_idx:end_idx]
    # drop = np.convolve(drop, np.ones((8,)) / 8, mode='valid')
    speed = np.diff(drop, n=1)*frame_rate
    speed_CDF = (drop[2:]-drop[:-2])/2*frame_rate
    acceleration = np.diff(speed, n=1)*frame_rate
    acceleration_CDF = (speed_CDF[2:]-speed_CDF[:-2])/2*frame_rate
    acceleration_CDF2nd = (drop[2:]-2*drop[1:-1]+drop[:-2])*frame_rate**2
    # smooth with moving average filter of 8 frames front and back
    # acceleration = np.convolve(acceleration, np.ones((8,))/8, mode='valid')
    if trajectory[2,start_idx] < 1.45 or max(abs(speed)) > 50 or max(acceleration[:40])>5 or min(acceleration[:80])<-13:
        continue
    ax1.plot(drop, label='drop {}'.format(i))
    ax2.plot(speed)
    ax3.plot(acceleration)
    ax2CDF.plot(speed_CDF)
    ax3CDF.plot(acceleration_CDF)
    ax3CDF2nd.plot(acceleration_CDF2nd)
    # set figure3 ylimit to -12,3
    ax2.set_ylim([-8,1])
    ax3.set_ylim([-12,3])
    ax2CDF.set_ylim([-8,1])
    ax3CDF.set_ylim([-12,3])
    ax3CDF2nd.set_ylim([-12,3])
    # ax1.legend()
    drop_caculation['position'] = drop
    drop_caculation['speed'] = speed
    drop_caculation['acceleration'] = acceleration
    # plot a dotted line at -9.81 m/s^2
    ax3.plot([0, len(acceleration)], [-9.81, -9.81], 'k--', lw=1)
    ax3CDF.plot([0, len(acceleration_CDF)], [-9.81, -9.81], 'k--', lw=1)
    ax3CDF2nd.plot([0, len(acceleration_CDF2nd)], [-9.81, -9.81], 'k--', lw=1)
    drop_caculations.append(drop_caculation)

#dump to pickle
vicon_output = {'start_idxs':start_idxs, 'frame_rate':frame_rate, 'drop_caculations':drop_caculations}
with open(f'vicon{trial_name[-1]}.pkl', 'wb') as f:
    pickle.dump(vicon_output, f)

figure1.savefig(f'vicon{trial_name[-1]}_position.png')
figure2.savefig(f'vicon{trial_name[-1]}_speed.png')
figure3.savefig(f'vicon{trial_name[-1]}_acceleration.png')
figure2CDF.savefig(f'vicon{trial_name[-1]}_speed_CDF.png')
figure3CDF.savefig(f'vicon{trial_name[-1]}_acceleration_CDF.png')
figure3CDF2nd.savefig(f'vicon{trial_name[-1]}_acceleration_CDF_2nd.png')


IMU_file = r'IMU drop 02.IMU.csv'

# # read csv file
with open(os.path.join(trial_name[0],IMU_file), 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    header = data[0]
    data = data[1:]

t = np.array(data)[:,2].astype(float)
x = np.array(data)[:,3].astype(float)
y = np.array(data)[:,4].astype(float)
z = np.array(data)[:,5].astype(float)

z_threshold = [-2,2]
end_idxs = []
last_i = 0
for i in range(len(z)):
    if z[i] > z_threshold[1] or z[i] < z_threshold[0]:

        if i - last_i < 2.5:
            continue
        end_idxs.append(i)
        last_i = i


figureIMU = plt.figure()
axIMU = figureIMU.add_subplot(111)
axIMU.set_title('IMU Acceleration (8 fps)')
# label
axIMU.set_xlabel('Vicon Frame (100 fps)')
axIMU.set_ylabel('Acceleration (m/s^2)')
time = 12/8
fps = 8
for ii, end_idx in enumerate(end_idxs):
    start_idx = int(end_idx - time*fps)
    if z[start_idx+8]*9.81-9.81>-4 or z[start_idx+7]*9.81-9.81>-4:
        # print(z[7]*9.81-9.81)
        continue
    x = np.linspace(0, time*100, end_idx-start_idx)
    axIMU.plot(x, z[start_idx:end_idx] * 9.81 - 9.81)
    axIMU.set_ylim([-12, 3])
    print(f'{ii} IMU idx: {start_idx} to {end_idx}')
axIMU.plot([0, time * 100], [-9.81, -9.81], 'k--', lw=1)

figureIMU.savefig(f'IMU{trial_name[-1]}.png')
plt.show()
