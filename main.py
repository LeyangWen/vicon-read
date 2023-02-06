import c3d
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from utility import *
import Skeleton
fileDir = r'H:\vicon\test\Full Body Plug-in Gait'
fileName = r'Gunwoo Cal 03.c3d'

skeleton = Skeleton.PulginGaitSkeleton(os.path.join(fileDir,fileName))
skeleton.plot_pose_frame(100)


# reader = c3d.Reader(open(os.path.join(fileDir,fileName), 'rb'))
# for i, points, analog in reader.read_frames():
#     print('frame {}: point {}, analog {}'.format(
#         i, points.shape, analog.shape))
#     break
#
#
# for label in reader.point_labels:
#     desc = description(label)
#     if desc:
#         print(desc)
#     else:
#         print(label)
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
