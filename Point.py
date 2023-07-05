
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os


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
        p1.xyz, p2.xyz, p3.xyz are 3xn np arrays
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
    def angle(v1, v2):
        """
        return the angle between v1 and v2
        v1 and v2 are vectors with shape (3, n)
        """
        return np.arccos(np.sum(v1 * v2, axis=0) / (np.linalg.norm(v1, axis=0) * np.linalg.norm(v2, axis=0)))


    @staticmethod
    def angle_w_direction(target_vector, main_axis_vector, secondary_axis_vector):
        """
        return the angle between main_axis_pt and target_pt using right hand rule
        secondary_axis_pt is used to determine the direction of the angle
        use arctan2 to get a range of [-pi, pi]
        """
        angle_abs = Point.angle(target_vector, main_axis_vector)
        angle_sign = Point.angle(target_vector, secondary_axis_vector)
        angle = np.where(angle_sign > np.pi / 2, -angle_abs, angle_abs)
        return angle

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
        # make sure the axis are equal
        ax.set_aspect('equal')
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

    # def find_frame(self, frame):
    #     if self.exist[frame]:



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
        self.normal_vector = Point.orthogonal_vector(pt1, pt2, pt3, normalize=1)
        self.normal_vector_end = Point.translate_point(pt1, self.normal_vector, direction=1)

    def project_vector(self, vector):
        """
        project a vector onto the plane
        vector as xyz
        """
        plane_normal = self.normal_vector.xyz
        plane_normal = plane_normal / np.linalg.norm(plane_normal, axis=0)
        projection = vector - np.diagonal(np.dot(vector.T, plane_normal)) * plane_normal
        return projection

    @staticmethod
    def angle_w_direction(plane1, plane2):
        '''
        return the angle between plane1 and plane2 in range of [-pi, pi]
        '''
        angle = Point.angle(plane1.normal_vector, plane2.normal_vector)
        angle_sign = Point.angle(plane1.normal_vector, plane2.normal_vector_end)
        angle = np.where(angle_sign > np.pi / 2, -angle, angle)
        return angle



class CoordinateSystem3D:
    def __init__(self):
        self.random_id = np.random.randint(0, 100000000)
        self.is_empty = True

    def set_by_plane(self, plane, origin_pt, axis_pt, sequence='xyz', axis_positive=True):
        '''
        first axis is the in-plane axis
        second axis is the orthogonal axis
        third axis is orthogonal to the first two (also should be in the plane)
        '''
        self.plane = plane
        self.is_empty = False
        self.origin = origin_pt
        axis_vec = Point.vector(origin_pt, axis_pt, normalize=1)
        if axis_positive:
            inplane_end = Point.translate_point(origin_pt, axis_vec, direction=1)
        else:
            inplane_end = Point.translate_point(origin_pt, axis_vec, direction=-1)
        orthogonal_end = Point.translate_point(origin_pt, plane.normal_vector)
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
        # Point.plot_points([self.origin, self.x_axis_end, self.y_axis_end, self.z_axis_end], frame=1000)


    def set_third_axis(self, sequence='xyz'):
        if sequence[-1] == 'x':
            self.x_axis_end = Point.translate_point(self.origin, Point.orthogonal_vector(self.origin, self.y_axis_end, self.z_axis_end, normalize=1))
        elif sequence[-1] == 'y':
            self.y_axis_end = Point.translate_point(self.origin, Point.orthogonal_vector(self.origin, self.z_axis_end, self.x_axis_end, normalize=1))
        elif sequence[-1] == 'z':
            self.z_axis_end = Point.translate_point(self.origin, Point.orthogonal_vector(self.origin, self.x_axis_end, self.y_axis_end, normalize=1))
        else:
            raise ValueError(f'sequence must be xyz, xzy, yxz, yzx, zxy, zyx, but got {sequence}')

    def set_plane_from_axis_end(self):
        self.xy_plane = Plane(self.origin, self.x_axis_end, self.y_axis_end)
        self.zx_plane = Plane(self.origin, self.z_axis_end, self.x_axis_end)
        self.yz_plane = Plane(self.origin, self.y_axis_end, self.z_axis_end)

    def projection_angles(self, pt):
        vector = Point.vector(self.origin, pt).xyz
        x_vector = Point.vector(self.origin, self.x_axis_end).xyz
        y_vector = Point.vector(self.origin, self.y_axis_end).xyz
        z_vector = Point.vector(self.origin, self.z_axis_end).xyz
        # xy plane
        xy_projection = self.xy_plane.project_vector(vector)
        xy_angle = Point.angle_w_direction(xy_projection, x_vector, y_vector)
        # xz plane
        xz_projection = self.zx_plane.project_vector(vector)
        xz_angle = Point.angle_w_direction(xz_projection, x_vector, z_vector)
        # yz plane
        yz_projection = self.yz_plane.project_vector(vector)
        yz_angle = Point.angle_w_direction(yz_projection, y_vector, z_vector)
        return xy_angle, xz_angle, yz_angle


class JointAngles:
    def __init__(self):
        self.random_id = np.random.randint(0, 100000000)
        self.is_empty = True
        self.zero_frame = [0, 0, 0]

    def set_zero_frame(self, frame):
        '''
        set to None if you don't want to zero the angles
        '''
        if frame is None:
            self.zero_frame = [None, None, None]
        if type(frame) == list:
            if len(frame) != 3:
                raise ValueError('zero frame must be a list of length 3 or a single int')
            self.zero_frame = frame
        elif type(frame) == int:
            self.zero_frame = [frame, frame, frame]

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
        for plane_id, plane_name in enumerate(plane_seq):
            if plane_name is not None:
                if plane_name == 'xy':
                    output_angle = xy_angle
                elif plane_name == 'xz':
                    output_angle = xz_angle
                elif plane_name == 'yz':
                    output_angle = yz_angle
                else:
                    raise ValueError('plane_name must be one of "xy", "xz", "yz", or None')
                # output_angle = np.abs(output_angle)
                if self.zero_frame[plane_id] is not None:
                    zero_frame_id = self.zero_frame[plane_id]
                    zero_angles.append(output_angle[zero_frame_id])
                    output_angle = output_angle - zero_angles[-1]
                else:
                    zero_angles.append(None)
                output_angles.append(output_angle)

            else:
                output_angles.append(None)


        self.flexion = output_angles[0]
        self.flexion_info = {'plane': plane_seq[0], 'zero_angle': zero_angles[0], 'zero_frame': self.zero_frame[0]}
        self.abduction = output_angles[1]
        self.abduction_info = {'plane': plane_seq[1], 'zero_angle': zero_angles[1], 'zero_frame': self.zero_frame[1]}
        self.is_empty = False
        return output_angles

    def get_rot(self, pt1a, pt1b, pt2a, pt2b):
        '''
        get rotation angle between two vectors
        Example:

        '''
        pt1mid = Point.mid_point(pt1a, pt1b)
        pt2mid = Point.mid_point(pt2a, pt2b)
        plane1 = Plane(pt1a, pt1b, pt2mid)
        plane2 = Plane(pt2a, pt2b, pt1mid)
        rotation_angle = Point.angle(plane1.normal_vector.xyz, plane2.normal_vector.xyz)
        # todo: rot angle should be in range of -pi to pi
        if self.zero_frame[2] is not None:
            rotation_zero = rotation_angle[self.zero_frame[2]]
            rotation_angle = rotation_angle - rotation_zero
        else:
            rotation_zero = None
        self.rotation = rotation_angle
        self.rotation_info = {'plane': None, 'zero_angle': rotation_zero, 'zero_frame': self.zero_frame[2]}
        self.is_empty = False
        return self.rotation

    def plot_angles(self, joint_name='', alpha=1, linewidth=1, linestyle='-', label=None, frame_range=None):
        if self.is_empty:
            raise ValueError('JointAngles is empty, please set angles first')
        if frame_range is None:
            if self.flexion is not None:
                frame_range = [0, len(self.flexion)]
            elif self.abduction is not None:
                frame_range = [0, len(self.abduction)]
            elif self.rotation is not None:
                frame_range = [0, len(self.rotation)]
            else:
                raise ValueError('all three angles are None, cannot plot')
        fig, ax = plt.subplots(3, 1, sharex=True)
        angle_names = ['Flexion', 'Abduction', 'Rotation']
        colors = ['r', 'g', 'b']
        for angle_id, angle in enumerate([self.flexion, self.abduction, self.rotation]):
            # horizontal line at zero, pi, and -pi
            ax[angle_id].axhline(0, color='k', linestyle='--', alpha=0.5, linewidth=0.25)
            ax[angle_id].axhline(180, color='k', linestyle='--', alpha=0.5, linewidth=0.25)
            ax[angle_id].axhline(-180, color='k', linestyle='--', alpha=0.5, linewidth=0.25)
            ax[angle_id].yaxis.set_ticks(np.arange(-180, 181, 90))
            ax[angle_id].set_ylabel(f'{angle_names[angle_id]}')
            ax[angle_id].margins(x=0)
            if angle is not None:
                ax[angle_id].plot(angle[frame_range[0]:frame_range[1]]/np.pi*180, color=colors[angle_id], alpha=alpha, linewidth=linewidth, linestyle=linestyle, label=label)
            else:
                # plot diagonal line crossing through the chart
                ax[angle_id].plot([frame_range[0], frame_range[1]], [-180, 180], color='black', linewidth=4)

        ax[0].set_title(f'{joint_name} (deg)')
        plt.show()
        return fig, ax

    def plot_angles_by_frame(self, render_dir, joint_name='', alpha=1, linewidth=1, linestyle='-', label=None, frame_range=None):
        if self.is_empty:
            raise ValueError('JointAngles is empty, please set angles first')
        if frame_range is None:
            if self.flexion is not None:
                frame_range = [0, len(self.flexion)]
            elif self.abduction is not None:
                frame_range = [0, len(self.abduction)]
            elif self.rotation is not None:
                frame_range = [0, len(self.rotation)]
            else:
                raise ValueError('all three angles are None, cannot plot')

        for frame_id in range(frame_range[0], frame_range[1]):
            fig, ax = plt.subplots(3, 1, sharex=True)
            angle_names = ['Flexion', 'Abduction', 'Rotation']
            colors = ['r', 'g', 'b']
            for angle_id, angle in enumerate([self.flexion, self.abduction, self.rotation]):
                print(f'frame {frame_id}/{frame_range[1]}', end='\r')
                # horizontal line at zero, pi, and -pi
                ax[angle_id].axhline(0, color='k', linestyle='--', alpha=0.5, linewidth=0.25)
                ax[angle_id].axhline(180, color='k', linestyle='--', alpha=0.5, linewidth=0.25)
                ax[angle_id].axhline(-180, color='k', linestyle='--', alpha=0.5, linewidth=0.25)
                ax[angle_id].yaxis.set_ticks(np.arange(-180, 181, 90))
                ax[angle_id].set_xlim(frame_range[0], frame_range[1])  # set xlim
                ax[angle_id].axvline(frame_id, color='k', linestyle='--', alpha=0.5, linewidth=0.25)  # vertical line at current frame
                # a dot with value at current frame
                ax[angle_id].set_ylabel(f'{angle_names[angle_id]}')
                ax[angle_id].margins(x=0)
                if angle is not None:
                    ax[angle_id].plot(angle[0:frame_id+1]/np.pi*180, color=colors[angle_id], alpha=alpha, linewidth=linewidth, linestyle=linestyle, label=label)
                    ax[angle_id].plot(frame_id, angle[frame_id]/np.pi*180, color=colors[angle_id], marker='o', markersize=5)  # a dot with value at current frame
                    ax[angle_id].text(frame_id, angle[frame_id]/np.pi*180, f'{angle[frame_id]/np.pi*180:.1f}', fontsize=12, horizontalalignment='left', verticalalignment='bottom')  # add text of current angle value
                else:
                    ax[angle_id].plot([frame_id, frame_id], [-180, 180], color='black', linewidth=4)  # plot diagonal line crossing through the chart

            ax[0].set_title(f'{joint_name} (deg)')

            ax[2].xaxis.set_major_locator(MaxNLocator(integer=True))  # set x ticks to integer only
            if not os.path.exists(render_dir):
                os.makedirs(render_dir)
            plt.savefig(os.path.join(render_dir, f'{joint_name}_{frame_id:06d}.png'))
            plt.close()



