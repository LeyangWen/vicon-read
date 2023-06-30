
import numpy as np
import matplotlib.pyplot as plt


class Point():
    def __init__(self):
        self.random_id = np.random.randint(0, 100000000)
        pass

    @staticmethod
    def mid_point(p1, p2):
        try:
            xyz = (p1.xyz + p2.xyz) / 2
            exist = p1.exist and p2.exist
            p_out = VirtualPoint((xyz, exist))
            return p_out
        except:
            print('Point not defined')
            return None
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



class Plane:
    def __init__(self, pt1=None, pt2=None, pt3=None):
        self.random_id = np.random.randint(0, 100000000)
        self.is_empty = True
        if pt1 is not None and pt2 is not None and pt3 is not None:
            self.set_by_pts(pt1, pt2, pt3)
    def set_by_pts(self, pt1, pt2, pt3):
        # rhr for vec: pt1->pt2, pt1->pt3
        # try to make orthogonal vector positive for one axis
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


class CoordinateSystem3D:
    def __init__(self):
        self.random_id = np.random.randint(0, 100000000)
        self.is_empty = True

    def set_by_plane(self, plane, origin_pt, axis_pt, sequence='xyz', axis_positive=True):
        '''
        first axis is the in-plane axis
        second axis is the orthogonal axis
        third axis is orthogonal to the first two
        '''
        self.plane = plane
        self.is_empty = False
        self.origin = origin_pt
        if x_direction:
            inplane_end = axis_pt
        else:
            inplace_end = Point.translate_point(origin_pt, Point.vector(origin_pt, axis_pt), direction=-1)
        orthogonal_end = Point.translate_point(origin_pt, plane.orthogonal_vector)
        if sequence[0] == 'x':
            self.x_axis_end = inplane_end
        elif sequence[0] == 'y':
            self.y_axis_end = inplane_end
        elif sequence[0] == 'z':
            self.z_axis_end = inplane_end
        else:
            raise ValueError(f'sequence must be xyz, xzy, yxz, yzx, zxy, zyx, but {sequence} is given')
        if sequence[1] == 'x':
            self.x_axis_end = orthogonal_end
        elif sequence[1] == 'y':
            self.y_axis_end = orthogonal_end
        elif sequence[1] == 'z':
            self.z_axis_end = orthogonal_end
        else:
            raise ValueError(f'sequence must be xyz, xzy, yxz, yzx, zxy, zyx, but {sequence} is given')
        self.set_third_axis(sequence)
        self.set_plane_from_axis_end()


    def set_third_axis(self, sequence='xyz'):
        if sequence[-1] == 'x':
            self.x_axis_end = Point.translate_point(Point.orthogonal_vector(self.origin, self.y_axis_end, self.z_axis_end, normalize=1), self.origin)
        elif sequence[-1] == 'y':
            self.y_axis_end = Point.translate_point(Point.orthogonal_vector(self.origin, self.z_axis_end, self.x_axis_end, normalize=1), self.origin)
        elif sequence[-1] == 'z':
            self.z_axis_end = Point.translate_point(Point.orthogonal_vector(self.origin, self.x_axis_end, self.y_axis_end, normalize=1), self.origin)
        else:
            raise ValueError(f'sequence must be xyz, xzy, yxz, yzx, zxy, zyx, but got {sequence}')

    def set_plane_from_axis_end(self):
        self.xy_plane = Plane(self.origin, self.x_axis_end, self.y_axis_end)
        self.xz_plane = Plane(self.origin, self.x_axis_end, self.z_axis_end)
        self.yz_plane = Plane(self.origin, self.y_axis_end, self.z_axis_end)

    def projection_angles(self, pt):
        vector = Point.vector(self.origin, pt).xyz
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


class JointAngles:
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
        self.is_empty = False
        return output_angles

    def get_rot(self, pt1a, pt1b, pt2a, pt2b):
        '''
        get rotation angle between two vectors
        Example:

        '''
        pt1mid = Point.midpoint(pt1a, pt1b)
        pt2mid = Point.midpoint(pt2a, pt2b)
        plane1 = Plane(pt1a, pt1b, pt2mid)
        plane2 = Plane(pt2a, pt2b, pt1mid)
        rotation_angle = Point.angle(plane1.normal_vector, plane2.normal_vector)
        rotation_zero = rotation_angle[self.zero_frame]
        rotation_angle = rotation_angle - rotation_zero
        self.rotation = rotation_angle
        self.rotation_info = {'plane': None, 'zero_angle': rotation_zero, 'zero_frame': self.zero_frame}
        self.is_empty = False
        return self.rotation

    def plot_angles(self, joint_name='', alpha=1, linewidth=1, linestyle='-', label=None):
        if self.is_empty:
            raise ValueError('JointAngles is empty, please set angles first')
        # a verticaly stacked plot of flexion, abduction, and rotation in one figure if they are not None
        fig, ax = plt.subplots(3, 1, sharex=True)
        if self.flexion is not None:
            ax[0].plot(self.flexion, color='r', alpha=alpha, linewidth=linewidth, linestyle=linestyle, label=label)
            ax[0].set_ylabel(f'{joint_name} Flexion')
            ax[0].legend()
        if self.abduction is not None:
            ax[1].plot(self.abduction, color='g', alpha=alpha, linewidth=linewidth, linestyle=linestyle, label=label)
            ax[1].set_ylabel(f'{joint_name} Abduction')
            ax[1].legend()
        if self.rotation is not None:
            ax[2].plot(self.rotation, color='b', alpha=alpha, linewidth=linewidth, linestyle=linestyle, label=label)
            ax[2].set_ylabel(f'{joint_name} Rotation')
            ax[2].legend()
        return fig, ax

