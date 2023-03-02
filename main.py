import c3d
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from utility import *
import Skeleton
fileName = r'F:\delete\Full Body Plug-in Gait\Gunwoo Cal 03.c3d'
fileName = r'G:\vicon\test\Full Body Plug-in Gait\Gunwoo Cal 03.c3d'
fileName = r'C:\Users\Public\Documents\Vicon\data\VEHS_ske\Test\Gunwoo\Test1\Gunwoo movement 02.c3d'

skeleton = Skeleton.PulginGaitSkeleton(fileName)
skeleton.point_labels
frame = 0
fig, ax = skeleton.plot_pose_frame(frame)


# skeleton.point_labels
point_acronym = ['LEYE',
                 'RPIK',
                 'LPIK',
                 'HDTP']
skeleton.add_point_to_plot(point_acronym, ax, fig, frame=frame)
# point_acronym = 'NKTP'
# skeleton.add_point_to_plot(point_acronym, ax, fig, frame=frame)
# point_acronym = 'HDEY'
# skeleton.add_point_to_plot(point_acronym, ax, fig, frame=frame)
plt.show()
# a = skeleton.output_3DSSPP_loc(frame_range=[0,10])


# reader = c3d.Reader(open(os.path.join(fileDir,fileName), 'rb'))
# for i, points, analog in reader.read_frames():
#     print('frame {}: point {}, analog {}'.format(
#         i, points.shape, analog.shape))
#     break
#
#
#
# reader.analog_labels
#
#
# idx = [75,79] # head
# idx = [75,79] # head
# joint_axis_pts = points[idx[0]:idx[1]]
# plot_joint_axis(joint_axis_pts,label = reader.point_labels[idx[0]])
#
#
# # make a list from 39 to 114 with step 4
# joint_org_idx = list(range(39,114,4))
