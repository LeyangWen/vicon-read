import c3d
import os

fileDir = r'H:\vicon\test\Full Body Plug-in Gait'
fileName = r'Gunwoo Cal 03.c3d'

reader = c3d.Reader(open(os.path.join(fileDir,fileName), 'rb'))
for i, points, analog in reader.read_frames():
    print('frame {}: point {}, analog {}'.format(
        i, points.shape, analog.shape))


reader.point_labels

