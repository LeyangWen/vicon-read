import c3d
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from utility import *
import Skeleton
import yaml


config_file = r'config/Gunwoo_test.yaml'

with open(config_file, 'r') as stream:
    try:
        data = yaml.safe_load(stream)
        c3d_files = data['c3d_files']
        skeleton_file = data['skeleton_file']
    except yaml.YAMLError as exc:
        print(config_file, exc)

for c3d_file in c3d_files:
    skeleton = Skeleton.PulginGaitSkeleton(c3d_file, skeleton_file)
    break
skeleton.output_3DSSPP_loc(frame_range=[1700,1900,2])
# skeleton.point_labels
# frame = 0
# for frame in range(1700,1900,2):
#     fig, ax = skeleton.plot_pose_frame(frame)
#     fig.savefig('frames/{}.png'.format(frame))



# skeleton.point_labels
# point_acronym = ['LEYE',
#                  'RPIK',
#                  'LPIK',
#                  'HDTP']
# skeleton.add_point_to_plot(point_acronym, ax, fig, frame=frame)
# point_acronym = 'NKTP'
# skeleton.add_point_to_plot(point_acronym, ax, fig, frame=frame)
# point_acronym = 'HDEY'
# skeleton.add_point_to_plot(point_acronym, ax, fig, frame=frame)
# plt.show()
# a = skeleton.output_3DSSPP_loc(frame_range=[0,10])



