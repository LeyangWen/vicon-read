
import numpy as np
import matplotlib.pyplot as plt


class Point():
    def __init__(self):
        self.random_id = np.random.randint(0, 100000000)
        pass

    @staticmethod
    def mid_point(p1, p2):
        xyz = (p1.xyz + p2.xyz) / 2
        exist = p1.exist and p2.exist
        p_out = virtualPoint((xyz, exist))
        return p_out

    @staticmethod
    def distance(p1, p2 = None):
        if p2 is None:
            return np.linalg.norm(p1.xyz, axis=0)
        else:
            return np.linalg.norm(p1.xyz - p2.xyz,axis=0)

    @staticmethod
    def vector(p1, p2, normalize=None):
        """
        return the vector from p1 to p2
        normalize: None->return the vector, 1->return the unit vector, other->return the vector with length normalize
        """
        xyz = (p2.xyz - p1.xyz)
        if normalize is not None:
            xyz = xyz/Point.distance(p1, p2) * normalize
        exist = p1.exist and p2.exist
        return virtualPoint((xyz, exist))

    @staticmethod
    def orthogonal_vector(p1, p2, p3, normalize=None):
        """
        return the vector orthogonal to the plane defined by p1, p2, p3
        direction is determined by the right hand rule based on vector p1->p2 and then p1->p3
        normalize: None->return the vector, 1->return the unit vector, other->return the vector with length normalize
        """
        v1 = Point.vector(p1, p2, normalize=1)
        v2 = Point.vector(p1, p3, normalize=1)
        xyz = np.cross(v1.xyz.T, v2.xyz.T).T
        if normalize is not None:
            xyz = xyz / np.linalg.norm(xyz, axis=0) * normalize
        exist = p1.exist and p2.exist and p3.exist
        return virtualPoint((xyz, exist))

    @staticmethod
    def translate_point(p, vector, direction=1):
        """
        move p in the direction of vector with length of distance
        """
        xyz = p.xyz + direction * vector.xyz
        exist = p.exist and vector.exist
        return virtualPoint((xyz, exist))

    @staticmethod
    def plot_points(point_list, ax=None, fig=None, frame=0):
        """
        plot a list of points
        """
        if ax is None or fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        x = []
        y = []
        z = []
        # make legend with point sequence
        for i, p in enumerate(point_list):
            ax.scatter(p.x[frame], p.y[frame], p.z[frame], label=str(i))
        # labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # legend
        ax.legend()
        plt.show()
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
        self.frame_no = len(self.exist)


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
        self.frame_no = len(self.exist)

    def output_format(self):
        return (self.xyz, self.exist)


class Plane():
    def __init__(self):
        self.id = str(np.random.randint(0, 100000000))
        self.is_empty = True

    def set_by_pts(self, pt1, pt2, pt3):
        self.pt1 = pt1
        self.pt2 = pt2
        self.pt3 = pt3
        self.is_empty = False

    def find_orthogonal_vector(self, direction=1):
        self.vector = Point.orthogonal_vector(self.pt1, self.pt2, self.pt3, normalize=1)
        return self.vector