
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
        p_out = VirtualPoint((xyz, exist))
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
        return VirtualPoint((xyz, exist))

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
        return VirtualPoint((xyz, exist))

    @staticmethod
    def translate_point(p, vector, direction=1):
        """
        move p in the direction of vector with length of distance
        """
        xyz = p.xyz + direction * vector.xyz
        exist = p.exist and vector.exist
        return VirtualPoint((xyz, exist))

    @staticmethod
    def angle(p1, p2):
        """
        return the angle between p1 and p2
        """
        return np.arccos(np.dot(p1.xyz, p2.xyz) / (np.linalg.norm(p1.xyz) * np.linalg.norm(p2.xyz)))

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


class VirtualPoint(Point):
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

    def find_frame(self, frame):
        if self.exist[frame]:



class Plane():
    def __init__(self, pt1=None, pt2=None, pt3=None):
        self.random_id = np.random.randint(0, 100000000)
        self.is_empty = True
        if pt1 is not None and pt2 is not None and pt3 is not None:
            self.set_by_pts(pt1, pt2, pt3)
    def set_by_pts(self, pt1, pt2, pt3):
        # rhr for vec: pt1->pt2, pt1->pt3
        self.pt1 = pt1
        self.pt2 = pt2
        self.pt3 = pt3
        self.is_empty = False
        self.orthogonal_vector = Point.orthogonal_vector(pt1, pt2, pt3, normalize=1)

    def project_vector(self, vector):
        """
        project a vector onto the plane
        vector as xyz
        """
        plane_normal = self.orthogonal_vector.xyz
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        projection = vector - np.dot(vector, plane_normal) * plane_normal
        return projection

class Coordinate_System_3D():
    def __init__(self):
        self.random_id = np.random.randint(0, 100000000)
        self.is_empty = True

    def set_by_plane(self, plane, origin_pt, x_axis_pt):
        self.plane = plane
        self.is_empty = False
        self.origin = origin_pt
        self.x_axis_end = Point.vector(origin_pt, x_axis_pt)
        self.y_axis_end = Point.translate_point(origin_pt, plane.orthogonal_vector)
        self.z_axis_end = Point.translate_point(Point.orthogonal_vector(self.origin, self.x_axis, self.y_axis, normalize=1), self.origin)

        self.xy_plane = Plane(self.origin, self.x_axis_end, self.y_axis_end)
        self.xz_plane = Plane(self.origin, self.x_axis_end, self.z_axis_end)
        self.yz_plane = Plane(self.origin, self.y_axis_end, self.z_axis_end)

    def projection_angles(self, pt):
        vector = pt.xyz
        # xy plane
        xy_projection = self.xy_plane.project_vector(vector)
        xy_angle = np.arctan2(xy_projection[1], xy_projection[0])
        # xz plane
        xz_projection = self.xz_plane.project_vector(vector)
        xz_angle = np.arctan2(xz_projection[2], xz_projection[0])
        # yz plane
        yz_projection = self.yz_plane.project_vector(vector)
        yz_angle = np.arctan2(yz_projection[2], yz_projection[1])
        return xy_angle, xz_angle, yz_angle

class Joint_Angles():
    def __init__(self):
        self.random_id = np.random.randint(0, 100000000)
        self.is_empty = True
        self.zero_frame = 0

    def set_zero_frame(self, frame):
        self.zero_frame = frame

    def get_flex_abd(self, coordinate_system, target_pt, plane_seq=['xy', 'xz']):
        """
        get flexion and abduction angles of a point in a coordinate system
        plane_seq: ['xy', None]
        """
        if len(plane_seq) != 2:
            raise ValueError('plane_seq must be a list of length 2, with flexion plane first and abduction plane second, fill None if not needed')
        xy_angle, xz_angle, yz_angle = coordinate_system.projection_angles(target_pt)
        output_angles = []
        zero_angles = []
        for plane_name in plane_seq:
            if plane_name is not None:
                if plane_name == 'xy':
                    output_angle = xy_angle
                elif plane_name == 'xz':
                    output_angle = xz_angle
                elif plane_name == 'yz':
                    output_angle = yz_angle
                else:
                    raise ValueError('plane_name must be one of "xy", "xz", "yz", or None')
                zero_angles.append(output_angle[self.zero_frame])
                output_angle = output_angle - zero_angles[-1]
                output_angles.append(output_angle)

            else:
                output_angles.append(None)


        self.flexion = output_angles[0]
        self.flexion_info = {'plane': plane_seq[0], 'zero_angle': zero_angles[0], 'zero_frame': self.zero_frame}
        self.abduction = output_angles[1]
        self.abduction_info = {'plane': plane_seq[1], 'zero_angle': zero_angles[1], 'zero_frame': self.zero_frame}
        return output_angles

    def get_rot(self, pt_vec1, pt_vec2):
        '''
        get rotation angle between two vectors
        Example:
        rot_angle = Joint_Angles.get_rot(Point.vector(pt1, pt2), Point.vector(pt3, pt4))
        '''
        rotation_angle = Point.angle(pt_vec1, pt_vec2)
        rotation_zero = rotation_angle[self.zero_frame]
        rotation_angle = rotation_angle - rotation_zero
        self.rotation = rotation_angle
        self.rotation_info = {'plane': None, 'zero_angle': rotation_zero, 'zero_frame': self.zero_frame}
        return self.rotation




