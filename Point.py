
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os


class Point():
    def __init__(self):
        self.random_id = np.random.randint(0, 100000000)
        self.name = None
        pass

    def copy(self):
        copied_point = MarkerPoint([self.x, self.y, self.z, self.exist])
        return copied_point

    @staticmethod
    def mid_point(p1, p2, precentage=0.5):
        '''
        return the midpoint of p1 and p2, if precentage is 0.5, return the mid point, if 0.25, return the point 1/4 way from p1 to p2
        '''
        try:
            xyz = p1.xyz * precentage + p2.xyz * (1 - precentage)
            exist = p1.exist and p2.exist  # exist need to be in pyton list, not np array
            p_out = VirtualPoint((xyz, exist))
            return p_out
        except:
            print('Error in "mid_point", Point not defined or exist not in python list')
            raise ValueError

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
        if type(vector) is np.ndarray:
            vector = Point.point_from_nparray(vector)
        xyz = p.xyz + direction * vector.xyz
        exist = p.exist and vector.exist
        return VirtualPoint((xyz, exist))

    @staticmethod
    def create_const_vector(x, y, z, frame=100, examplePt=None):
        '''
        x, y, z are float
        frame is ignored if examplePt is not None
        '''
        if examplePt:
            frame = examplePt.frame_no
        xyz = np.vstack((np.ones(frame) * x, np.ones(frame) * y, np.ones(frame) * z))
        exist = np.ones(frame, dtype=bool).tolist()
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

    @staticmethod
    def point_from_nparray(xyz):
        exist = np.ones(xyz.shape[0], dtype=bool).tolist()
        return VirtualPoint((xyz, exist))

    @staticmethod
    def batch_export_to_nparray(point_list):
        '''
        point_list is a list of Point objects
        return a np array with shape (frame, keypoints, 3)
        '''
        xyz = np.array([p.xyz.T for p in point_list])
        xyz = np.swapaxes(xyz, 0, 1)
        return xyz

    @staticmethod
    def swap_trajectory(p1, p2, index):
        '''
        swap p1 and p2 trajectory from index forward (including index)
        '''
        out1 = p1.copy()
        out2 = p2.copy()
        index = index
        out1.xyz[:, index:], out2.xyz[:, index:] = p2.xyz[:, index:], p1.xyz[:, index:]
        out1.exist[index:], out2.exist[index:] = p2.exist[index:], p1.exist[index:]
        return out1, out2

    @staticmethod
    def check_marker_swap(p1, p2, threshold=35):
        '''
        check if p1 and p2 are swapped, one way, need to check both directions
        this does not work well when the marker pairs are moving fast in the direction of each other
        '''
        p1_xyz = p1.xyz[:, :-1]
        p2_xyz = p2.xyz[:, :-1]
        p2_xyz_shift = p2.xyz[:, 1:]
        criteria = np.linalg.norm(p1_xyz - p2_xyz_shift, axis=0) < np.linalg.norm(p2_xyz - p2_xyz_shift, axis=0)  # check for swaps when both markers are present
        swap_index = criteria.nonzero()[0]+1
        p2_missing = (~np.array(p2.exist)).nonzero()[0]
        for swap_id in swap_index:
            if swap_id in p2_missing or swap_id - 1 in p2_missing:
                if p1.exist[swap_id-1]:  # if p1 is present at swap_id-1 and p2 is missing, then it is not a swap
                    check = True
                else:
                    criteria_2 = np.linalg.norm(p1_xyz[:, swap_id - 1] - p2_xyz_shift[:, swap_id - 1])
                    check = criteria_2 < threshold and criteria_2 > 0  # check for swaps when p2 is actually missing (p1 is incorrectly swapped and labeled as missing instead)
                    check = not check
                if check:
                    swap_index = np.delete(swap_index, np.where(swap_index == swap_id))
                else:
                    print(f'Caught by check: swap_id: ', swap_id, 'check: ', check)
        # todo: can not detect when p1 is missing before and p2 is present
        return swap_index

    @staticmethod
    def check_marker_swap_by_speed(p1, p2, threshold=15, interval_frames=1):
        '''
        check if p1 and p2 are swapped by speed threshold only, one way, need to check both directions
        '''
        p1_xyz = p1.xyz[:, :-interval_frames]
        p2_xyz_shift = p2.xyz[:, interval_frames:]
        criteria_value = np.linalg.norm(p1_xyz - p2_xyz_shift, axis=0)
        criteria = np.logical_and(criteria_value < (threshold), criteria_value > 0)
        swap_index = criteria.nonzero()[0]+1
        return swap_index

    def check_marker_speed(self, threshold=35, interval_frames=1):
        '''
        check if marker speed is too high
        '''
        xyz = self.xyz[:, :-interval_frames]
        xyz_shift = self.xyz[:, interval_frames:]
        criteria = np.linalg.norm(xyz - xyz_shift, axis=0) > (threshold)  # * interval_frames)
        swap_index = criteria.nonzero()[0]+1
        for swap_id in swap_index:
            if (not self.exist[swap_id-1]) or (not self.exist[swap_id]):
                swap_index = np.delete(swap_index, np.where(swap_index == swap_id))
        return swap_index


class MarkerPoint(Point):
    def __init__(self, data, name=None):
        super().__init__()
        self.data = data
        self.type = 'marker'
        self.x, self.y, self.z, self.exist = data
        self.xyz = np.array([self.x, self.y, self.z])
        self.x = self.xyz[0]
        self.y = self.xyz[1]
        self.z = self.xyz[2]
        self.frame_no = len(self.exist)
        self.name = name


class VirtualPoint(Point):
    def __init__(self, data, name=None):
        super().__init__()
        self.data = data
        self.type = 'virtual'
        self.xyz, self.exist = data
        self.xyz = np.array(self.xyz)
        self.x = self.xyz[0]
        self.y = self.xyz[1]
        self.z = self.xyz[2]
        self.frame_no = len(self.exist)
        self.name = name

    def output_format(self):
        return (self.xyz, self.exist)


class Plane:
    def __init__(self, pt1=None, pt2=None, pt3=None):
        self.random_id = np.random.randint(0, 100000000)
        self.is_empty = True
        if pt1 is not None and pt2 is not None and pt3 is not None:
            self.set_by_pts(pt1, pt2, pt3)

    def set_by_pts(self, pt1, pt2, pt3):
        """
        rhr for vec: pt1->pt2, pt1->pt3
        try to make orthogonal vector positive for one axis
        """
        self.pt1 = pt1
        self.pt2 = pt2
        self.pt3 = pt3
        self.is_empty = False
        self.normal_vector = Point.orthogonal_vector(pt1, pt2, pt3, normalize=1)
        self.normal_vector_end = Point.translate_point(pt1, self.normal_vector, direction=1)

    def set_by_vector(self, pt1, vector, direction=1):
        """
        vector as virtual point
        """
        self.pt1 = pt1
        self.pt2 = None
        self.pt3 = None
        self.is_empty = False
        vector_xyz = vector.xyz
        normal_vector_xyz = vector_xyz / np.linalg.norm(vector_xyz, axis=0) * direction  # normalize vector
        self.normal_vector = Point.point_from_nparray(normal_vector_xyz)
        self.normal_vector_end = Point.translate_point(pt1, vector, direction=direction)

    def project_vector(self, vector):
        """
        project a vector onto the plane
        vector as xyz
        """
        plane_normal = self.normal_vector.xyz
        plane_normal = plane_normal / np.linalg.norm(plane_normal, axis=0)

        # vector = np.array([[1,1,1],[1,1,2]]).T
        # plane_normal = np.array([[0,1,0],[0,1,0]]).T
        angle = Point.angle(vector, plane_normal)
        projection = vector - np.diagonal(np.dot(vector.T, plane_normal)) * plane_normal
        return projection

    def project_point(self, point):
        """
        project a point onto the plane
        """
        vector = point.xyz - self.pt1.xyz
        projection = self.project_vector(vector)  # todo: find a less entangled way in the future
        return Point.translate_point(self.pt1, projection)

    def above_or_below(self, point):
        """
        return 1 if point is above the plane, -1 if below
        """
        vector = point.xyz - self.pt1.xyz
        normal_vector = self.normal_vector.xyz
        return np.sign(np.sum(vector * normal_vector, axis=0))


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
        requirement: need axis_pt and origin_pt to be in the specified plane
        sequence [first, second, thrid axis] meaning:
        first axis is the in-plane axis, if axis_positive is True, the direction is from origin_pt to axis_pt
        second axis is the orthogonal axis to plane
        third axis is orthogonal to the first two (also should be in the plane)
        '''
        '''
        Usage:
        # RShoulder angles
        try:
            PELVIS_b = Point.translate_point(C7_m, Point.create_const_vector(0,0,-1000,examplePt=C7))  # todo: this is temp for this shoulder trial, change to real marker in the future
    
            zero_frame = [941, 941, None]
            # RSHOULDER_plane = Plane(RSHO_b, RSHO_f, C7_m)
            RSHOULDER_plane = Plane(RSHOULDER, PELVIS_b, C7_m)
            RSHOULDER_coord = CoordinateSystem3D()
            RSHOULDER_coord.set_by_plane(RSHOULDER_plane, RSHOULDER, C7_m, sequence='zxy', axis_positive=False)
            RSHOULDER_angles = JointAngles()
            RSHOULDER_angles.set_zero_frame(zero_frame)
            RSHOULDER_angles.get_flex_abd(RSHOULDER_coord, RELBOW, plane_seq=['xy', 'xz'])
            # RSHOULDER_angles.get_rot(RSHO_b, RSHO_f, RME, RLE)
            RSHOULDER_angles.flexion = Point.angle(Point.vector(RSHOULDER, RELBOW).xyz, Point.vector(C7, PELVIS_b).xyz)
            RSHOULDER_angles.rotation = None
    
            ##### Visual for debugging #####
            # frame = 1000
            # print(f'RSHOULDER_angles:\n Flexion: {RSHOULDER_angles.flexion[frame]}, \n Abduction: {RSHOULDER_angles.abduction[frame]},\n Rotation: {RSHOULDER_angles.rotation[frame]}')
            # Point.plot_points([RSHOULDER_coord.origin, RSHOULDER_coord.x_axis_end, RSHOULDER_coord.y_axis_end, RSHOULDER_coord.z_axis_end], frame=frame)
            # RSHOULDER_angles.plot_angles(joint_name='Right Shoulder', frame_range=[941, 5756])
            render_dir = os.path.join(trial_name[0], 'render', trial_name[1], 'RSHOULDER')
            RSHOULDER_angles.plot_angles_by_frame(render_dir, joint_name='Right Shoulder', frame_range=[941, 5756])
        except:
            print('RSHOULDER_angles failed')
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

    def projection_angles(self, target_vector, threshold=1):
        vector = target_vector.xyz
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

    def get_flex_abd(self, coordinate_system, target_vector, plane_seq=['xy', 'xz'], flip_sign=[1, 1]):
        """
        get flexion and abduction angles of a vector in a coordinate system
        plane_seq: ['xy', None]
        """
        if len(plane_seq) != 2:
            raise ValueError('plane_seq must be a list of length 2, with flexion plane first and abduction plane second, fill None if not needed')
        xy_angle, xz_angle, yz_angle = coordinate_system.projection_angles(target_vector)
        output_angles = []
        zero_angles = []
        for plane_id, plane_name in enumerate(plane_seq):
            if plane_name is not None:
                if plane_name == 'xy':
                    output_angle = xy_angle
                elif plane_name == 'xz' or plane_name == 'zx':
                    output_angle = xz_angle
                elif plane_name == 'yz':
                    output_angle = yz_angle
                else:
                    raise ValueError('plane_name must be one of "xy", "xz", "zx", "yz", or None')
                # output_angle = np.abs(output_angle)
                if self.zero_frame[plane_id] is not None:
                    zero_frame_id = self.zero_frame[plane_id]
                    zero_angles.append(output_angle[zero_frame_id])
                    output_angle = output_angle - zero_angles[-1]
                    # deal with output -pi pi range issue
                    output_angle = np.where(output_angle > np.pi, output_angle - 2 * np.pi, output_angle)
                    output_angle = np.where(output_angle < -np.pi, output_angle + 2 * np.pi, output_angle)
                else:
                    zero_angles.append(None)
                output_angles.append(output_angle)

            else:
                output_angles.append(None)

        self.flexion = output_angles[0] * flip_sign[0]
        self.flexion_info = {'plane': plane_seq[0], 'zero_angle': zero_angles[0], 'zero_frame': self.zero_frame[0], 'flip_sign': flip_sign[0]}
        self.abduction = output_angles[1] * flip_sign[1]
        self.abduction_info = {'plane': plane_seq[1], 'zero_angle': zero_angles[1], 'zero_frame': self.zero_frame[1], 'flip_sign': flip_sign[1]}
        self.is_empty = False
        return output_angles

    def get_rot(self, pt1a, pt1b, pt2a, pt2b, flip_sign=1):
        '''
        get rotation angle between two vectors
        flip_sign: 1 or -1, if the rotation is in the opposite direction
        Example:

        '''
        pt1mid = Point.mid_point(pt1a, pt1b)
        pt2mid = Point.mid_point(pt2a, pt2b)
        plane1 = Plane(pt1a, pt1b, pt2mid)
        plane2 = Plane(pt2a, pt2b, pt1mid)
        rotation_angle = Point.angle(plane1.normal_vector.xyz, plane2.normal_vector.xyz)

        rotation_sign = plane2.above_or_below(pt1a)
        rotation_angle = rotation_angle * rotation_sign * flip_sign

        if self.zero_frame[2] is not None:
            rotation_zero = rotation_angle[self.zero_frame[2]]
            rotation_angle = rotation_angle - rotation_zero
        else:
            rotation_zero = None
            
        # make plot in -pi to pi range
        rotation_angle = np.where(rotation_angle > np.pi, rotation_angle - 2 * np.pi, rotation_angle)
        rotation_angle = np.where(rotation_angle < -np.pi, rotation_angle + 2 * np.pi, rotation_angle)
        
        self.rotation = rotation_angle
        self.rotation_info = {'plane': None, 'zero_angle': rotation_zero, 'zero_frame': self.zero_frame[2]}
        self.is_empty = False
        return self.rotation

    def zero_by_idx(self, idx):
        """
        idx is 0, 1, 2, corresponding to flexion, abduction, rotation
        usage:
        RKNEE_angles.flexion = RKNEE_angles.zero_by_idx(0)
        """
        angle = self.flexion if idx == 0 else self.abduction if idx == 1 else self.rotation
        this_zero_frame = self.zero_frame[idx]
        if this_zero_frame is not None:
                zero_angle = angle[this_zero_frame]
                output_angle = angle - zero_angle
                output_angle = np.where(output_angle > np.pi, output_angle - 2 * np.pi, output_angle)
                output_angle = np.where(output_angle < -np.pi, output_angle + 2 * np.pi, output_angle)
        else:
            output_angle = angle
        return output_angle

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
        angle_names = ['Flexion', 'H-Abduction', 'Rotation']
        colors = ['r', 'g', 'b']
        for angle_id, angle in enumerate([self.flexion, self.abduction, self.rotation]):
            # horizontal line at zero, pi, and -pi
            ax[angle_id].axhline(0, color='k', linestyle='--', alpha=0.5, linewidth=0.25)
            ax[angle_id].axhline(90, color='k', linestyle='dotted', alpha=0.5, linewidth=0.25)
            ax[angle_id].axhline(180, color='k', linestyle='--', alpha=0.5, linewidth=0.25)
            ax[angle_id].axhline(-90, color='k', linestyle='dotted', alpha=0.5, linewidth=0.25)
            ax[angle_id].axhline(-180, color='k', linestyle='--', alpha=0.5, linewidth=0.25)
            ax[angle_id].yaxis.set_ticks(np.arange(-180, 181, 90))
            ax[angle_id].set_ylabel(f'{angle_names[angle_id]}')
            ax[angle_id].set_xlim(frame_range[0], frame_range[1])  # set xlim
            ax[angle_id].margins(x=0)
            if angle is not None:
                ax[angle_id].plot(angle[0:frame_range[1]]/np.pi*180, color=colors[angle_id], alpha=alpha, linewidth=linewidth, linestyle=linestyle, label=label)
            else:
                # plot diagonal line crossing through the chart
                ax[angle_id].plot([frame_range[0], frame_range[1]], [-180, 180], color='black', linewidth=4)

        ax[0].set_title(f'{joint_name} (deg)')
        plt.show()
        return fig, ax

    def plot_angles_by_frame(self, render_dir, joint_name='', alpha=1, linewidth=1, linestyle='-', label=None, frame_range=None, angle_names = ['Flexion', 'H-Abduction', 'Rotation']):
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
        print(f'Saving {joint_name} angle frames to {render_dir}')
        for frame_id in range(frame_range[0], frame_range[1]):
            fig, ax = plt.subplots(3, 1, sharex=True)

            colors = ['r', 'g', 'b']
            for angle_id, angle in enumerate([self.flexion, self.abduction, self.rotation]):
                print(f'frame {frame_id}/{frame_range[1]}', end='\r')
                # horizontal line at zero, pi, and -pi
                ax[angle_id].axhline(0, color='k', linestyle='--', alpha=0.5, linewidth=0.25)
                ax[angle_id].axhline(90, color='k', linestyle='dotted', alpha=0.5, linewidth=0.25)
                ax[angle_id].axhline(180, color='k', linestyle='--', alpha=0.5, linewidth=0.25)
                ax[angle_id].axhline(-90, color='k', linestyle='dotted', alpha=0.5, linewidth=0.25)
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
                    ax[angle_id].plot([frame_range[0], frame_range[1]], [-180, 180], color='gray', linewidth=1)  # plot diagonal line crossing through the chart

            ax[0].set_title(f'{joint_name} (deg)')

            ax[2].xaxis.set_major_locator(MaxNLocator(integer=True))  # set x ticks to integer only
            if not os.path.exists(render_dir):
                os.makedirs(render_dir)
            plt.savefig(os.path.join(render_dir, f'{joint_name}_{frame_id:06d}.png'))
            plt.close()



