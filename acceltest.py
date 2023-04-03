from viconnexusapi import ViconNexus
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
import pickle
from utility import *
import os


# helper functions
def centered_difference(x, h):
    return (x[2:] - x[:-2]) / (2 * h)


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
ax1.set_title('Y')
ax2.set_title('Speed')
ax3.set_title('Acceleration')
# label
ax1.set_xlabel('Frame (100 fps)')
ax2.set_xlabel('Frame (100 fps)')
ax3.set_xlabel('Frame (100 fps)')
ax1.set_ylabel('Position (m)')
ax2.set_ylabel('Speed (m/s)')
ax3.set_ylabel('Acceleration (m/s^2)')

# plot the 3D trajectory of the unlabeled markers
start_idx = 642
end_idx = start_idx+1800
trajectory = vicon.GetUnlabeled(0)
trajectory = np.array(trajectory).T
trajectory = trajectory[start_idx:end_idx]/1000 # convert to meter

fig = plt.figure()

# calculate speed from 3D point position measurements
speed_CDF = centered_difference(trajectory, 1/frame_rate)
acceleration_CDF = centered_difference(speed_CDF, 1/frame_rate)
# acceleration_CDF = np.linalg.norm(acceleration_CDF, axis=1)


drop = trajectory[:,1]
ax1.plot(drop, label='Vicon')
ax2CDF.plot(speed_CDF[:,1], label='Vicon')
ax3CDF.plot(acceleration_CDF[:,1], label='Vicon')





# plot ideal acceleration of pendulum over time
L = 1.33
g = 9.80305 # GGB1440b gravity
theta_0 = 0.193178717  # Initial angle of displacement
t = np.linspace(0, int((end_idx - start_idx)/100),1800)  # Time range

# Calculate the ideal acceleration of the pendulum over time
theta = theta_0 * np.cos(np.sqrt(g / L) * t)
a = -g * np.cos(theta)*np.sin(theta)

# Plot the ideal acceleration of the pendulum over time
figure = plt.figure()
ax = figure.add_subplot(111)
ax.plot(t*100, a)
ax.set_xlabel('Frame (100 fps)')
ax.set_ylabel('Acceleration (m/s^2)')
ax.set_title('Ideal Acceleration of a Pendulum')

ax3CDF.plot(t*100, a, label='Theoretical')
ax3CDF.legend()

# Save the plot
figure.savefig('output/ideal_acceleration.png')
figure1.savefig(f'output/vicon{trial_name[-1]}_position.png')
figure2.savefig(f'output/vicon{trial_name[-1]}_speed.png')
figure3.savefig(f'output/vicon{trial_name[-1]}_acceleration.png')
figure2CDF.savefig(f'output/vicon{trial_name[-1]}_speed_CDF.png')
figure3CDF.savefig(f'output/vicon{trial_name[-1]}_acceleration_CDF.png')
figure3CDF2nd.savefig(f'output/vicon{trial_name[-1]}_acceleration_CDF_2nd.png')
plt.show()





