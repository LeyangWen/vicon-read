from viconnexusapi import ViconNexus
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
import pickle

vicon = ViconNexus.ViconNexus()


# read vicon marker values
# dir(ViconNexus.ViconNexus)
# help(vicon.SetSubjectParam)

trial_name = vicon.GetTrialName()
subject_names = vicon.GetSubjectNames()
frame_count = vicon.GetFrameCount()
frame_rate = vicon.GetFrameRate()
subject_info = vicon.GetSubjectInfo()
trajectory = vicon.GetTrajectory(subject_names[0], 'LTOE')
len(trajectory[0])
subject_param = vicon.GetSubjectParam(subject_names[0],'LeftPinkyWidth')




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
    acceleration = np.diff(speed, n=1)*frame_rate
    # smooth with moving average filter of 8 frames front and back
    # acceleration = np.convolve(acceleration, np.ones((8,))/8, mode='valid')
    if trajectory[2,start_idx] < 1.45 or max(abs(speed)) > 50 or max(acceleration[:40])>5 or min(acceleration[:80])<-13:
        continue
    ax1.plot(drop, label='drop {}'.format(i))
    ax2.plot(speed)
    ax3.plot(acceleration)
    # set figure3 ylimit to -12,3
    ax2.set_ylim([-8,1])
    ax3.set_ylim([-12,3])
    # ax1.legend()
    drop_caculation['position'] = drop
    drop_caculation['speed'] = speed
    drop_caculation['acceleration'] = acceleration
    # plot a dotted line at -9.81 m/s^2
    ax3.plot([0, len(acceleration)], [-9.81, -9.81], 'k--', lw=1)
    drop_caculations.append(drop_caculation)

#dump to pickle
vicon_output = {'start_idxs':start_idxs, 'frame_rate':frame_rate, 'drop_caculations':drop_caculations}
with open(f'vicon{trial_name[-1]}.pkl', 'wb') as f:
    pickle.dump(vicon_output, f)

figure1.savefig(f'vicon{trial_name[-1]}_position.png')
figure2.savefig(f'vicon{trial_name[-1]}_speed.png')
figure3.savefig(f'vicon{trial_name[-1]}_acceleration.png')


# IMU_file = r'C:\Users\wenleyan1\Downloads\W2_2023-03-01T15.37.28.692_E703F686FEDD_Accelerometer.csv'
# # read csv file
# with open(IMU_file, 'r') as f:
#     reader = csv.reader(f)
#     data = list(reader)
#     header = data[0]
#     data = data[1:]
# data = np.array(data)
# shift = 200
# length = 1000
# start_idxs = np.round(np.array(start_idxs)/8)
#
# figureIMU = plt.figure()
# axIMU = figureIMU.add_subplot(111)
# for start_idx in start_idxs:
#     start_idx = start_idx.astype(int)
#     start_idx = start_idx+shift
#     end_idx = start_idx + length
#     axIMU.plot(data[start_idx:end_idx,5].astype(float))
#     break

plt.show()
