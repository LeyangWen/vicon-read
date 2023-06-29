import c3d
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from utility import *
import Skeleton
import yaml


config_file = r'F:\wen_storage\test\VEHS_ske\Test\Gunwoo\Test1\Gunwoo movement 02.yaml'

with open(config_file, 'r') as stream:
    try:
        data = yaml.safe_load(stream)
        c3d_file = data['c3d_file']
    except yaml.YAMLError as exc:
        print(config_file, exc)


skeleton = Skeleton.PulginGaitSkeleton(c3d_file)


# skeleton.point_labels
# frame = 0
# for frame in range(0,1000,1):
#     fig, ax = skeleton.plot_pose_frame(frame)
#     # format frame number to 5 digits with filler 0
#     frame = format(frame, '05')
#     fig.savefig('frames/Gunwoo02_0-1000/{}.png'.format(frame))
#     plt.close(fig)




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

bone = 'S'
for LR in ['L','R']:
    for pos in ['TL','TR','BL','BR']:
        point_acronym = LR + bone + pos
        print(point_acronym, end=', ')
    print()



