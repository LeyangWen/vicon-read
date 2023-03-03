
import numpy as np
import matplotlib.pyplot as plt


class Point():
    def __init__(self):
        pass

    @staticmethod
    def mid_point(p1, p2):
        xyz = (p1.xyz + p2.xyz) / 2
        exist = p1.exist and p2.exist
        p_out = virtualPoint((xyz, exist))
        return p_out

    @staticmethod
    def distance(p1, p2):
        return np.linalg.norm(p1.xyz - p2.xyz)

    @staticmethod
    def vector(p1, p2, normalize=None):
        '''
        return the vector from p1 to p2
        normalize: None->return the vector, 1->return the unit vector, other->return the vector with length normalize
        '''
        if normalize is None:
            normalize = Point.distance(p1, p2)

        xyz = (p2.xyz - p1.xyz)/Point.distance(p1, p2) * normalize
        exist = p1.exist and p2.exist
        return virtualPoint((xyz, exist))

    @staticmethod
    def translate_point(p, vector, direction=1):
        '''
        move p in the direction of vector with length of distance
        '''
        xyz = p.xyz + direction * vector.xyz
        exist = p.exist and vector.exist
        return virtualPoint((xyz, exist))

    @staticmethod
    def plot_points(point_list, ax=None, fig=None, frame=0):
        '''
        plot a list of points
        '''
        if ax is None or fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        x = []
        y = []
        z = []
        for p in point_list:
            x.append(p.x[frame])
            y.append(p.y[frame])
            z.append(p.z[frame])
        ax.scatter(x, y, z)
        return ax, fig

class MarkerPoint(Point):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.type = 'marker'
        self.x, self.y, self.z, self.exist = data
        self.xyz = np.array([self.x, self.y, self.z])
        self.x = self.xyz[0]
        self.y = self.xyz[1]
        self.z = self.xyz[2]
        self.frames = len(self.exist)


class virtualPoint(Point):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.type = 'virtual'
        self.xyz, self.exist = data
        self.xyz = np.array(self.xyz)
        self.x = self.xyz[0]
        self.y = self.xyz[1]
        self.z = self.xyz[2]
        self.frames = len(self.exist)

    def output_format(self):
        return (self.xyz, self.exist)