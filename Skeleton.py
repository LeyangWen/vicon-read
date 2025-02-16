# a class for vicon skeleton
from unicodedata import category

import c3d
import numpy as np
import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# from PyQt5.QtWidgets.QWidget import width
from scipy.io import savemat
import yaml
from spacepy.data_assimilation import output

# from backup_c3d import file_name
from utility import *
# from spacepy import pycdf
import cv2
from ergo3d import *
from ergo3d.camera.FLIR_camera import batch_load_from_xcp
from scipy.spatial.transform import Rotation as R
# todo: self.frame_no vs self.frame_number be consistent

class Skeleton:
    def __init__(self, skeleton_file):
        self.__load_key_joints(skeleton_file)

    def load_name_list(self, name_list):
        self.point_labels = name_list
        self.point_number = len(name_list)

    def load_custom_points(self, custom_points):
        pt_np = Point.batch_export_to_nparray(custom_points)
        try:
            self.point_labels
        except(AttributeError):
            print('point_labels is empty, need to load point names first')
            raise AttributeError
        self.points = pt_np
        self.poses = {}
        self.point_poses = {}
        self.frame_number = np.shape(pt_np)[0]
        self.points_dimension = np.shape(pt_np)[-1]
        for i in range(self.point_number):
            self.poses[self.point_labels[i]] = pt_np[:, i, :]
            self.point_poses[self.point_labels[i]] = custom_points[i]

    def load_np_points(self, pt_np):
        try:
            self.point_labels
        except(AttributeError):
            print('point_labels is empty, need to load point names first')
            raise AttributeError
        self.points = pt_np
        self.poses = {}
        self.frame_number = np.shape(pt_np)[0]
        self.points_dimension = np.shape(pt_np)[-1]
        self.point_poses = {}
        for i in range(self.point_number):
            self.poses[self.point_labels[i]] = pt_np[:, i, :]
            if self.points_dimension == 3:
                exist = pt_np[:, i, 0] != np.nan  # todo: check if missing points is expressed as np.nan in c3d
                xyz_exist = [pt_np[:, i, 0], pt_np[:, i, 1], pt_np[:, i, 2], exist.tolist()]
                self.point_poses[self.point_labels[i]] = MarkerPoint(xyz_exist, name=self.point_labels[i])

    def load_name_list_and_np_points(self, name_list, pt_np):
        self.load_name_list(name_list)
        self.load_np_points(pt_np)

    def load_mesh(self, vertices, format='smpl'):
        """
        load vertices from mesh as 49 surface markers using vert ids
        vertices: np array of vertices (n, 6890, 3) for smpl
        """
        try:
            self.marker_vids
        except AttributeError:
            raise AttributeError('marker_vids is empty, need to load marker_vids in skeleton config first')
        vid = self.marker_vids[format]
        name_list = list(vid.keys())
        idx_list = list(vid.values())
        marker_vertices = vertices[:, idx_list, :]
        self.load_name_list_and_np_points(name_list, marker_vertices)


    def load_c3d(self, c3d_file, analog_read=True):
        self.c3d_file = c3d_file
        reader = c3d.Reader(open(c3d_file, 'rb'))
        if analog_read:
            try:
                self.analog_labels = reader.analog_labels
                self.analog_labels = [label.strip() for label in self.analog_labels]  # strip whitespace from analog labels
            except AttributeError:
                self.analog_labels = None
        else:
            self.analog_labels = None
        self.point_labels = [label.strip() for label in reader.point_labels]  # strip whitespace from point labels
        points = []
        analog = []
        for i, this_points, this_analog in reader.read_frames():
            print('frame {}: point {}, analog {}'.format(
                i, this_points.shape, this_analog.shape), end='\r')
            points.append(this_points)
            analog.append(this_analog)
        pt_np = np.array(points)
        self.analog = np.array(analog)
        self.frame_number = pt_np.shape[0]
        self.point_number = pt_np.shape[1]
        self.points = pt_np
        self.poses = {}
        self.point_poses = {}
        self.frame_number = np.shape(pt_np)[0]
        self.points_dimension = np.shape(pt_np)[-1]
        for i in range(self.point_number):
            self.poses[self.point_labels[i]] = pt_np[:, i, :3]
            exist = pt_np[:, i, 0] != np.nan  # todo: check if missing points is expressed as np.nan in c3d
            xyz_exist = [pt_np[:, i, 0], pt_np[:, i, 1], pt_np[:, i, 2], exist.tolist()]
            self.point_poses[self.point_labels[i]] = MarkerPoint(xyz_exist, name=self.point_labels[i])

    def __load_key_joints(self, filename):  # read xml
        with open(filename, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
                self.joint_name_mid = data['joints']['mid']
                self.joint_name_botL = data['joints']['botL']
                self.joint_name_topL = data['joints']['topL']
                self.joint_name_botR = data['joints']['botR']
                self.joint_name_topR = data['joints']['topR']
                self.key_joint_name = data['joints']['mid'] + data['joints']['botL'] + data['joints']['topL'] + data['joints']['botR'] + data['joints']['topR']
                self.key_joint_parent = data['parent']['mid'] + data['parent']['botL'] + data['parent']['topL'] + data['parent']['botR'] + data['parent']['topR']
                self.surface_marker_list = []
                self.joint_center_list = []
            except yaml.YAMLError as exc:
                print(filename, exc)
                raise exc
            try:
                self.surface_marker_list = data['type']['surface_markers']
                self.joint_center_list = data['type']['joint_centers']
            except KeyError:
                pass
            try:
                self.marker_vids = data['marker_vids']
            except KeyError:
                pass

    def update_pose_from_point_pose(self):
        for point_key, point_pose in self.point_poses.items():
            self.poses[point_key] = point_pose.xyz.T

    def get_parent(self, joint_name):
        parent_idx = self.key_joint_name.index(joint_name)
        # catch IndexError: list index out of range and return None
        try:
            return self.key_joint_parent[parent_idx]
        except IndexError:
            return None

    def get_plot_property(self, joint_name, size=[4,8]):
        '''
        return point_type and point_size for plot
        '''
        if joint_name in self.joint_name_mid:
            point_type = 's'
            point_size = size[0]
        elif joint_name in self.joint_name_botL or joint_name in self.joint_name_topL:
            point_type = '<'
            point_size = size[0]
        elif joint_name in self.joint_name_botR or joint_name in self.joint_name_topR:
            point_type = '>'
            point_size = size[0]
        else:
            point_type = 'o'
            point_size = size[1]
        if joint_name in self.joint_center_list:
            point_type = 'o'
            point_size = size[1]
        return point_type, point_size

    def plot_3d_pose_frame(self, frame=0, filename=False, plot_range=1800, coord_system="world", center_key='PELVIS', mode='normal_view', get_legend=False):
        """
        plot 3d pose in 3d space
        coord_system: camera-px or world
        mode: camera_view, camera_side_view, 0_135_view, normal_view
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if coord_system == "camera-px":
            pose_sequence = [0, 2, 1]
            xyz_label = ['X (px)', 'Y (px)', 'Z (px)']
        elif coord_system == "camera-mm":
            pose_sequence = [0, 2, 1]
            xyz_label = ['X (mm)', 'Y (mm)', 'Z (mm)']
        elif coord_system == "world":
            pose_sequence = [0, 1, 2]
            xyz_label = ['X (mm)', 'Y (mm)', 'Z (mm)']
        else:
            raise ValueError(f"coord_system {coord_system} not recognized, set to camera or world")

        for joint_name in self.key_joint_name:
            if joint_name in self.poses:
                point_type, point_size = self.get_plot_property(joint_name, size=[10, 20])
                ax.scatter(self.poses[joint_name][frame, pose_sequence[0]],
                           self.poses[joint_name][frame, pose_sequence[1]],
                           self.poses[joint_name][frame, pose_sequence[2]], label=joint_name, marker=point_type, s=point_size)
                # connect points to parent
                parent_name = self.get_parent(joint_name)
                if parent_name is not None and parent_name != 'None' and parent_name in self.poses:
                    ax.plot([self.poses[joint_name][frame, pose_sequence[0]], self.poses[parent_name][frame, pose_sequence[0]]],
                            [self.poses[joint_name][frame, pose_sequence[1]], self.poses[parent_name][frame, pose_sequence[1]]],
                            [self.poses[joint_name][frame, pose_sequence[2]], self.poses[parent_name][frame, pose_sequence[2]]], 'k-')
        # ax.legend(bbox_to_anchor=(0.95, 1), loc=2, borderaxespad=0.)
        # uniform scale based on pelvis location and 1800mm

        # mode = 'camera_view'
        # mode = 'camera_side_view'
        # mode = '0_135_view'
        # mode = 'normal_view'
        if mode == 'camera_view':
            # camera view in px
            ax.view_init(elev=0, azim=270)
        elif mode == 'camera_side_view':
            # camera side view in px
            ax.view_init(elev=0, azim=0)
        elif mode == '0_135_view':
            # azimuth 135 elev 0 - side view
            ax.view_init(elev=0, azim=135)
        elif mode == 'normal_view':
            pass
        else:
            raise ValueError(f"mode {mode} not recognized, set to camera_view, camera_side_view, 0_135_view, normal_view")

        pelvis_loc = self.poses[center_key][frame, :]
        ax.set_xlim(pelvis_loc[pose_sequence[0]] - plot_range / 2, pelvis_loc[pose_sequence[0]] + plot_range / 2)
        ax.set_ylim(pelvis_loc[pose_sequence[1]] - plot_range / 2, pelvis_loc[pose_sequence[1]] + plot_range / 2)
        ax.set_zlim(pelvis_loc[pose_sequence[2]] - plot_range / 2, pelvis_loc[pose_sequence[2]] + plot_range / 2)
        if coord_system[:6] == "camera":  # invert y axis in camera coord
            ax.invert_zaxis()
        ax.set_xlabel(xyz_label[pose_sequence[0]])
        ax.set_ylabel(xyz_label[pose_sequence[1]])
        ax.set_zlabel(xyz_label[pose_sequence[2]])
        fig.tight_layout()

        if get_legend:  # use this to get a legend screenshot
            # also set range to big value
            ax.legend(loc='upper center', fontsize=5, ncol=6)
            plt.gca().set_axis_off()
            plt.savefig(r'legend_new.png', dpi=250)
            raise NameError(r"Intentional break: legend.png saved to legend_new.png")  # break here

        if True:  # no legend
            pass
        elif False:  # normal legends
            fig.subplots_adjust(right=0.65)
            ax.legend(loc='center left', bbox_to_anchor=(1.08, 0.5), fontsize=7, ncol=2)

        if filename:
            plt.savefig(filename, dpi=250)
            plt.close(fig)
            return None
        else:
            plt.show()
            return fig, ax

    def plot_2d_pose_frame(self, frame=0, baseimage=False, filename=False, resolution=(1920, 1200), dpi=100):
        # return a transparent image
        img_width, img_height = resolution
        inches_width, inches_height = img_width / dpi, img_height / dpi
        fig = plt.figure(figsize=(inches_width, inches_height), dpi=dpi)  # not quite working
        ax = fig.add_subplot(111)
        if baseimage:
            import matplotlib.image as mpimg
            img = mpimg.imread(baseimage)
            plt.imshow(img)
            # raise NotImplementedError
        for joint_name in self.key_joint_name:
            if joint_name in self.poses:
                point_type, point_size = self.get_plot_property(joint_name, size=[50, 60])
                ax.scatter(self.poses[joint_name][frame, 0],
                           self.poses[joint_name][frame, 1], label=joint_name, marker=point_type, s=point_size, zorder=2)
        # connect points to parent
                parent_name = self.get_parent(joint_name)
                if parent_name in self.poses and parent_name is not None and parent_name != 'None':
                    ax.plot([self.poses[joint_name][frame, 0], self.poses[parent_name][frame, 0]],
                            [self.poses[joint_name][frame, 1], self.poses[parent_name][frame, 1]], 'gray', zorder=1, linewidth=2)
        ax.set_xlim(0, img_width)
        ax.set_ylim(0, img_height)
        # ax.set_xlim(-500, 1500)
        # ax.set_ylim(-500, 1500)
        # plot edge lines
        # ax.plot([0, img_width], [0, 0], 'gray', zorder=0)
        # ax.plot([0, 0], [0, img_height], 'gray', zorder=0)
        # ax.plot([img_width, img_width], [0, img_height], 'gray', zorder=0)
        # ax.plot([0, img_width], [img_height, img_height], 'gray', zorder=0)

        ax.set_aspect('equal', adjustable='box')
        ax.invert_yaxis()  # flip y axis
        fig.tight_layout()
        # ax.axis('off')  # Hide axis
        if filename:
            plt.savefig(filename, transparent=True, bbox_inches='tight')
            plt.close(fig)
            return None
        else:
            plt.show()
            return fig, ax

    def plot_3d_pose(self, foldername=False, **kwargs):
        if foldername:
            create_dir(foldername)
        for i in range(self.frame_number):
            print(f'plotting frame {i}/{self.frame_number} in {foldername}...', end='\r')
            filename = foldername if not foldername else os.path.join(foldername, f'{i:05d}.png')
            self.plot_3d_pose_frame(frame=i, filename=filename, **kwargs)
            # break

    def plot_2d_pose(self, foldername=False, **kwargs):
        if foldername:
            create_dir(foldername)
        for i in range(self.frame_number):
            print(f'plotting frame {i}/{self.frame_number} in {foldername}...', end='\r')
            filename = foldername if not foldername else os.path.join(foldername, f'{i:05d}.png')
            self.plot_2d_pose_frame(frame=i, filename=filename, **kwargs)


class VEHSErgoSkeleton(Skeleton):
    def __init__(self, skeleton_file):
        super().__init__(skeleton_file)
        self.marker_height = 14/2+2  # 14mm marker
        # marker_height = 9.5/2+2  # 9.5mm marker

    def calculate_joint_center(self):
        self.point_poses['HEAD'] = Point.mid_point(self.point_poses['LEAR'], self.point_poses['REAR'])

        ear_vector = Point.vector(self.point_poses['REAR'], self.point_poses['LEAR'], normalize=1)
        self.point_poses['REAR'] = Point.translate_point(self.point_poses['REAR'], ear_vector, direction=9)  # todo: need to define unit, this is in meters now
        self.point_poses['LEAR'] = Point.translate_point(self.point_poses['LEAR'], ear_vector, direction=-9)

        head_plane = Plane()
        head_plane.set_by_pts(self.point_poses['REAR'], self.point_poses['LEAR'], self.point_poses['HDTP'])
        head = self.point_poses['HEAD']
        distance_between_ears = Point.distance(self.point_poses['REAR'], self.point_poses['LEAR'])
        self.point_poses['NOSE'] = Point.translate_point(head, head_plane.normal_vector, direction=distance_between_ears * 0.78)

        self.point_poses['RSHOULDER'] = Point.mid_point(self.point_poses['RAP_f'], self.point_poses['RAP_b'])
        self.point_poses['LSHOULDER'] = Point.mid_point(self.point_poses['LAP_f'], self.point_poses['LAP_b'])
        self.point_poses['C7_m'] = Point.mid_point(self.point_poses['C7_d'], self.point_poses['SS'])
        self.point_poses['THORAX'] = Point.translate_point(self.point_poses['SS'], Point.vector(self.point_poses['SS'], self.point_poses['C7_m'], normalize=self.marker_height))  # offset by marker height
        self.point_poses['LELBOW'] = Point.mid_point(self.point_poses['LME'], self.point_poses['LLE'])
        self.point_poses['RELBOW'] = Point.mid_point(self.point_poses['RME'], self.point_poses['RLE'])
        self.point_poses['RWRIST'] = Point.mid_point(self.point_poses['RRS'], self.point_poses['RUS'])
        self.point_poses['LWRIST'] = Point.mid_point(self.point_poses['LRS'], self.point_poses['LUS'])

        right_wrist_plane = Plane()
        right_wrist_plane.set_by_pts(self.point_poses['RMCP2'], self.point_poses['RMCP5'], self.point_poses['RWRIST'])
        self.point_poses['RMCP5'] = Point.translate_point(self.point_poses['RMCP5'], right_wrist_plane.normal_vector, direction=9)
        self.point_poses['RMCP2'] = Point.translate_point(self.point_poses['RMCP2'], right_wrist_plane.normal_vector, direction=9)
        left_wrist_plane = Plane()
        left_wrist_plane.set_by_pts(self.point_poses['LMCP2'], self.point_poses['LWRIST'], self.point_poses['LMCP5'])
        self.point_poses['LMCP5'] = Point.translate_point(self.point_poses['LMCP5'], left_wrist_plane.normal_vector, direction=9)
        self.point_poses['LMCP2'] = Point.translate_point(self.point_poses['LMCP2'], left_wrist_plane.normal_vector, direction=9)

        self.point_poses['RHAND'] = Point.mid_point(self.point_poses['RMCP2'], self.point_poses['RMCP5'])
        self.point_poses['LHAND'] = Point.mid_point(self.point_poses['LMCP2'], self.point_poses['LMCP5'])
        self.point_poses['LGRIP'] = Point.translate_point(self.point_poses['LHAND'],
                                                          Point.orthogonal_vector(self.point_poses['LMCP5'], self.point_poses['LMCP2'], self.point_poses['LWRIST'], normalize=40))
        self.point_poses['RGRIP'] = Point.translate_point(self.point_poses['RHAND'],
                                                            Point.orthogonal_vector(self.point_poses['RMCP2'], self.point_poses['RMCP5'], self.point_poses['RWRIST'], normalize=40))

        ##### lower body #####
        self.point_poses['PELVIS_f'] = Point.mid_point(self.point_poses['RASIS'], self.point_poses['LASIS'])
        self.point_poses['PELVIS_b'] = Point.mid_point(self.point_poses['RPSIS'], self.point_poses['LPSIS'])
        self.point_poses['PELVIS'] = Point.mid_point(self.point_poses['PELVIS_f'], self.point_poses['PELVIS_b'])
        self.point_poses['RHIP'] = Point.translate_point(self.point_poses['RGT'], Point.vector(self.point_poses['RASIS'], self.point_poses['LASIS'], normalize=2 * 25.4))  # offset 2 inches
        self.point_poses['RKNEE'] = Point.mid_point(self.point_poses['RLFC'], self.point_poses['RMFC'])
        self.point_poses['RANKLE'] = Point.mid_point(self.point_poses['RMM'], self.point_poses['RLM'])
        self.point_poses['RFOOT'] = Point.mid_point(self.point_poses['RMTP1'], self.point_poses['RMTP5'])
        self.point_poses['LHIP'] = Point.translate_point(self.point_poses['LGT'], Point.vector(self.point_poses['LASIS'], self.point_poses['RASIS'], normalize=2 * 25.4))  # offset 2 inches
        self.point_poses['LKNEE'] = Point.mid_point(self.point_poses['LLFC'], self.point_poses['LMFC'])
        self.point_poses['LANKLE'] = Point.mid_point(self.point_poses['LMM'], self.point_poses['LLM'])
        self.point_poses['LFOOT'] = Point.mid_point(self.point_poses['LMTP1'], self.point_poses['LMTP5'])
        self.point_poses['HIP_c'] = Point.mid_point(self.point_poses['RHIP'], self.point_poses['LHIP'])
        self.point_poses['SHOULDER_c'] = Point.mid_point(self.point_poses['RSHOULDER'], self.point_poses['LSHOULDER'])

        self.update_pose_from_point_pose()

    def calculate_camera_projection(self, args, camera_xcp_file, kpts_of_interest_name='all', rootIdx=0):
        if kpts_of_interest_name == 'all':  # get all points
            kpts_of_interest = self.point_poses.values()
        else:
            kpts_of_interest = [self.point_poses[kpt] for kpt in kpts_of_interest_name]
        self.current_kpts_of_interest_name = kpts_of_interest_name
        self.current_kpts_of_interest = kpts_of_interest
        world3D = Point.batch_export_to_nparray(kpts_of_interest)
        self.pose_3d_world = world3D

        cameras = batch_load_from_xcp(camera_xcp_file)
        start_frame = 0
        end_frame = self.frame_number
        rgb_frame_rate = 100
        fps_ratio = 100 / rgb_frame_rate
        frames = np.linspace(start_frame / fps_ratio, end_frame / fps_ratio, int((end_frame - start_frame) / fps_ratio), dtype=int)
        set_vis = 2
        self.pose_3d_camera = {}
        self.pose_2d_camera = {}
        self.pose_2d_bbox = {}
        self.pose_depth_px = {}
        self.pose_depth_ratio = {}
        self.pose_2d_vis_camera = {}
        for cam_idx, camera in enumerate(cameras):
            print(f'Processing camera {cam_idx}: {camera.DEVICEID}')
            points_2d_list = []
            points_2d_vis_list = []
            points_3d_camera_list = []
            points_2d_bbox_list = []
            points_depth_px_list = []
            depth_ratio_list = []
            for frame_idx, frame_no in enumerate(frames):
                frame_idx = int(frame_idx * fps_ratio)  # todo: bug if fps_ratio is not an 1
                print(f'Processing frame {frame_no}/{frames[-1]} of {self.c3d_file}', end='\r')
                points_3d = world3D[frame_idx, :, :].reshape(-1, 3) / 1000  # convert to meters, (n_joints, 3)
                points_2d = camera.project(points_3d)
                if args.distort:
                    points_2d = camera.distort(points_2d)
                points_3d_camera = camera.project_w_depth(points_3d)  # (n_joints, 3)
                points_depth_px, ratio = self.get_norm_depth_ratio(points_3d_camera, camera, rootIdx=rootIdx)
                bbox_top_left, bbox_bottom_right = points_2d.min(axis=0) - 20, points_2d.max(axis=0) + 20

                # points_2d_vis = np.ones((points_2d.shape[0], 1)) * set_vis
                # points_2d_vis = np.concatenate([points_2d, points_2d_vis], axis=1)
                # points_2d_vis_list.append(points_2d_vis.tolist())
                points_2d_list.append(points_2d)
                points_3d_camera_list.append(points_3d_camera)
                points_2d_bbox_list.append([bbox_top_left, bbox_bottom_right])
                points_depth_px_list.append(points_depth_px)
                depth_ratio_list.append(ratio)


            self.pose_3d_camera[camera.DEVICEID] = np.array(points_3d_camera_list)
            self.pose_2d_camera[camera.DEVICEID] = np.array(points_2d_list)
            self.pose_2d_camera[camera.DEVICEID] = np.array(points_2d_list)
            # self.pose_2d_vis_camera[camera.DEVICEID] = points_2d_vis_list  # for coco, need to be list for json, --> too slow for memory
            self.pose_2d_bbox[camera.DEVICEID] = np.array(points_2d_bbox_list)
            self.pose_depth_px[camera.DEVICEID] = np.array(points_depth_px_list)
            self.pose_depth_ratio[camera.DEVICEID] = np.array(depth_ratio_list)
        self.cameras = cameras

    def zero_frame_pitch(self, points, angle):
        """
        zero camera pitch angle, rotate camera around x axis for angle
        :param points: JX3 np array
        :param angle: in radians
        """
        # angle = -angle  # negate angle
        rotated_points = points.copy()
        rot_M = np.array([[1, 0, 0],
                            [0, np.cos(angle), -np.sin(angle)],
                            [0, np.sin(angle), np.cos(angle)]])
        rotated_points = np.dot(rotated_points, rot_M)
        return rotated_points

    def get_norm_depth_ratio(self, pose3d, camera, rootIdx=0):
        """
        LCN style, convert to px, center by pelvis
        in here meters * ratio = px
        ratio px/meters
        """
        rectangle_3d_size = 2.0  # 2000mm
        box = self._infer_box(pose3d, camera, rootIdx, size=rectangle_3d_size)
        ratio = (box[2] - box[0] + 1) / rectangle_3d_size
        pose3d_depth = ratio * (pose3d[:, 2] - pose3d[rootIdx, 2])
        # print(f"box: {box}")
        # print(f"ratio: {ratio}, lcm_ratio {1000/ratio}")
        return pose3d_depth, ratio

    def _infer_box(self, pose3d, camera, rootIdx, size):
        """
        a 2d box 2000mm x 2000mm in px, centered at rootIdx (pelvis if 0)
        """
        root_joint = pose3d[rootIdx, :]
        tl_joint = root_joint.copy()
        tl_joint[:2] -= size/2
        br_joint = root_joint.copy()
        br_joint[:2] += size/2
        tl_joint = np.reshape(tl_joint, (1, 3))
        br_joint = np.reshape(br_joint, (1, 3))
        tl_br_joint = np.concatenate([tl_joint, br_joint], axis=0)
        tl_br = camera._weak_project(tl_br_joint).flatten()
        return tl_br

    def output_MotionBert_SMPL(self):
        '''
        MotionBert Style
        '''
        raise NotImplementedError

    def output_COCO_2dPose(self, downsample=5, downsample_keep=1, image_id_cum=0, pose_id_cum=0, small_test=False):
        # append data at end
        annotations = []
        images = []

        ################ Preset Params
        set_vis = 2
        set_crowd = 0
        set_category_id = 1
        set_license = 99
        res_w = 1920
        res_h = 1200
        file_elements = self.c3d_file.split('/')
        ### subject to change based on your filename structure
        subject = file_elements[-3]
        activity = file_elements[-1].split('.')[0].lower()
        ################
        # print(f"Check: subject: {subject}, activity: {activity}")
        assert subject[0] == "S"
        assert int(subject[1:]) < 11
        assert activity[:8] == "activity"
        for this_camera in self.cameras:
            frame_count = 0
            # camera_pitch_angle = this_camera.get_camera_pitch()
            for downsample_idx in range(downsample):
                if downsample_idx != downsample_keep-1:
                    continue
                for frame_idx in range(0, self.frame_number, downsample):
                    real_frame_idx = frame_idx + downsample_idx + int(downsample/2)  # middle frame for coco image alignment
                    if real_frame_idx >= self.frame_number:
                        break
                    frame_count += 1
                    image_id_cum += 1
                    pose_id_cum += 1  # since we only have one pose per image
                    this_joint_2d = self.pose_2d_camera[this_camera.DEVICEID][real_frame_idx]
                    this_joint_vis = np.ones(this_joint_2d.shape[0]) * set_vis
                    this_joint_2d_vis = np.concatenate([this_joint_2d, this_joint_vis[:, None]], axis=1)
                    num_keypoints = len(this_joint_2d_vis[0])

                    annotation_dict = {}
                    annotation_dict['keypoints'] = this_joint_2d_vis.reshape(-1)
                    annotation_dict['num_keypoints'] =num_keypoints
                    annotation_dict['iscrowd'] = set_crowd
                    annotation_dict['bbox'] = bbox_tlbr2tlwh(self.pose_2d_bbox[this_camera.DEVICEID][real_frame_idx])
                    annotation_dict['category_id'] = set_category_id
                    annotation_dict['id'] = pose_id_cum
                    annotation_dict['image_id'] = image_id_cum
                    annotation_dict['id_100fps'] = real_frame_idx
                    # annotation_dict['area'] = float(annotation_dict['bbox'][2] * annotation_dict['bbox'][3])  # should be segmentation area, just set cocoeval use_area=False

                    # COCO image name format need to match conversion_scripts/vid_to_img.py
                    # test/S09-activity00-51470934-000001jpg
                    image_file = f"{subject}-{activity}-{this_camera.DEVICEID}-{frame_count:06d}.jpg"

                    image_dict = {}
                    image_dict['file_name'] = image_file
                    image_dict['height'] = res_h
                    image_dict['width'] = res_w
                    image_dict['id'] = image_id_cum
                    image_dict['license'] = set_license

                    annotations.append(annotation_dict)
                    images.append(image_dict)
                    if small_test and real_frame_idx > 80:
                        break

        output = {"annotations": annotations, "images": images}
        return output

    def output_MotionBert_pose(self, downsample=5, downsample_keep=1, pitch_correction=False):
        # append data at end
        output = {}
        joint_2d = []
        confidence = []
        joint_3d_image = []  # px coordinate, px
        joint_3d_camera = []  # camera coordinate, mm
        joint_25d_image = []  # px coordinate, mm
        factor_25d = []  # ratio, ~4.xx, 2.5d_image/3d_image
        camera_name = []
        source = []
        c3d_frame = []
        for this_camera in self.cameras:
            camera_pitch_angle = this_camera.get_camera_pitch()
            print(f"correcting pitch for {this_camera.DEVICEID}, pitch: {camera_pitch_angle / np.pi * 180:.01f} degrees") if pitch_correction else None
            for downsample_idx in range(downsample):
                if downsample_idx != downsample_keep-1:
                    continue
                # print(f"downsample {downsample_idx}/{downsample}")
                for frame_idx in range(0, self.frame_number, downsample):
                    real_frame_idx = frame_idx + downsample_idx
                    if real_frame_idx >= self.frame_number:
                        break
                    joint_2d.append(self.pose_2d_camera[this_camera.DEVICEID][real_frame_idx, :, :])
                    confidence.append(np.ones((len(self.current_kpts_of_interest_name), 1)))
                    joint_3d_camera.append(self.pose_3d_camera[this_camera.DEVICEID][real_frame_idx, :, :])

                    joint_3d_image_frame = self.pose_3d_camera[this_camera.DEVICEID][real_frame_idx, :, :].copy()  # need the shape
                    # todo: maybe joint_3d_image_frame = ratio * (pose3d[:] - pose3d[rootIdx]) instead of merging the two results
                    joint_3d_image_frame[:, :2] = self.pose_2d_camera[this_camera.DEVICEID][real_frame_idx, :, :]  # overwrite xy
                    joint_3d_image_frame[:, 2] = self.pose_depth_px[this_camera.DEVICEID][real_frame_idx, :]  # overwrite z

                    if pitch_correction:  # use root_rel = True when training MotionBert, or do additional correction
                        # rotate joint_3d_image_frame around camera x axis for pitch correction angle
                        joint_3d_image_frame = self.zero_frame_pitch(joint_3d_image_frame, camera_pitch_angle)

                    joint_3d_image.append(joint_3d_image_frame)
                    factor_25d_frame = self.pose_depth_ratio[this_camera.DEVICEID][real_frame_idx]
                    factor_25d_frame = 1000/factor_25d_frame  # in motionbert 2.5d factor, px * factor = mm
                    factor_25d.append(factor_25d_frame)
                    joint_25d_image_frame = joint_3d_image_frame.copy() * factor_25d_frame
                    joint_25d_image.append(joint_25d_image_frame)

                    camera_name.append(this_camera.DEVICEID)
                    source.append(f"{self.c3d_file} - cam_{this_camera.DEVICEID}")
                    c3d_frame.append(real_frame_idx)

        output['joint_2d'] = np.array(joint_2d)  # this is gt, but should be detection
        output['confidence'] = np.array(confidence)
        output['joint3d_image'] = np.array(joint_3d_image)  # px coordinate, px
        output['camera_name'] = np.array(camera_name)
        output['source'] = source
        output['2.5d_factor'] = np.array(factor_25d)
        output['joints_2.5d_image'] = np.array(joint_25d_image)  # mm
        output['action'] = [self.c3d_file[-14:-4]] * len(source)

        ################## additional ##################
        # LCN format https://github.com/CHUNYUWANG/lcn-pose/blob/master/tools/gendb.py#L85
        output['joint_3d_camera'] = np.array(joint_3d_camera) * 1000  # convert to mm
        # c3d info
        output['c3d_frame'] = c3d_frame
        return output

    def get_pelvic_tilt(self, loc):
        # 3DSSPP manual page 54
        RHIP = Point(loc[:, 96:99])
        LHIP = Point(loc[:, 75:78])
        HIP_center = Point.mid_point(RHIP, LHIP)
        L5S1 = Point(loc[:, 27:30])
        up_axis = [0,0,1000]  # todo: world coord only
        zero_frame = [90, 180, 180]
        BACK_plane = Plane()
        BACK_plane.set_by_vector(HIP_center, Point.create_const_vector(*up_axis, examplePt=HIP_center), direction=1)
        BACK_coord = CoordinateSystem3D()
        BACK_RHIP_PROJECT = BACK_plane.project_point(RHIP)
        BACK_coord.set_by_plane(BACK_plane, HIP_center, BACK_RHIP_PROJECT, sequence='zyx', axis_positive=True)
        BACK_angles = JointAngles()
        BACK_angles.ergo_name = {'flexion': 'flexion', 'abduction': 'L-flexion', 'rotation': 'rotation'}  #lateral flexion
        BACK_angles.set_zero(zero_frame)
        BACK_angles.get_flex_abd(BACK_coord, Point.vector(HIP_center, L5S1), plane_seq=['xy', 'yz'], flip_sign=[-1, 1])

        angles = BACK_angles.flexion
        angles = (angles/np.pi*180).astype(int)
        # print(f'pelvic tilt: {angles} degree')
        return angles

    def output_3DSSPP_loc(self, frame_range=None, loc_file=None):
        """
        # 3DSSPP format:
        # LOC File filename.loc
        # Value Anatomical Location Attribute
        # 1 - 3 Top Head Skin Surface
        # 4 - 6 L. Head Skin Surface
        # 7 - 9 R. Head Skin Surface
        # 10 - 12 Head origin Virtual point
        # 13 - 15 Nasion Skin Surface
        # 16 - 18 Sight end Virtual point
        # 19 - 21 C7/T1 Joint Center
        # 22 - 24 Sternoclavicular Joint Joint Center
        # 25 - 27 Suprasternale Skin Surface
        # 28 - 30 L5/S1 Joint Center
        # 31 - 33 PSIS Joint Center
        # 34 - 36 L. Shoulder Joint Center
        # 37 - 39 L. Acromion Skin Surface
        # 40 - 42 L. Elbow Joint Center
        # 43 - 45 L. Lat. Epicon. of Humer. Skin Surface
        # 46 - 48 L. Wrist Joint Center
        # 49 - 51 L. Grip Center Virtual point
        # 52 - 54 L. Hand Skin Surface
        # 55 - 57 R. Shoulder Joint Center
        # 58 - 60 R. Acromion Skin Surface
        # 61 - 63 R. Elbow Joint Center
        # 64 - 66 R. Lat. Epicon. of Humer. Skin Surface
        # 67 - 69 R. Wrist Joint Center
        # 70 - 72 R. Grip Center Virtual point
        # 73 - 75 R. Hand Skin Surface
        # 76 - 79 L. Hip Joint Center
        # 79 - 81 L. Knee Joint Center
        # 82 - 84 L. Lat. Epicon. of Femur Skin Surface
        # 85 - 87 L. Ankle Joint Center
        # 88 - 90 L. Lateral Malleolus Skin Surface
        # 91 - 93 L. Ball of Foot Virtual point
        # 94 - 96 L. Metatarsalphalangeal Skin Surface
        # 97 - 99 R. Hip Joint Center
        # 100 - 102 R. Knee Joint Center
        # 103 - 105 R. Lat. Epicon. of Femur Skin Surface
        # 106 - 108 R. Ankle Joint Center
        # 109 - 111 R. Lateral Malleolus Skin Surface
        # 112 - 114 R. Ball of Foot Virtual point
        # 115 - 117 R. Metatarsalphalangeal Skin Surface
        """
        weight = getattr(self, 'weight', 75)
        height = getattr(self, 'height', 180)
        gender = getattr(self, 'gender', 'male')
        gender_id = 0 if gender=='male' else 1  # male 0, female 1
        if frame_range is not None:
            start_frame = frame_range[0]
            end_frame = frame_range[1]
            step = frame_range[2]
        else:
            start_frame = 0
            end_frame = self.frame_number
            step = 1
        if True:
            loc = np.zeros((self.frame_number, 117))
            loc[:, 0:3] = self.poses['HDTP']  # 1 - 3 Top Head Skin Surface
            loc[:, 3:6] = self.poses['LEAR']  # 4 - 6 L. Head Skin Surface
            loc[:, 6:9] = self.poses['REAR']  # 7 - 9 R. Head Skin Surface
            loc[:, 9:12] = self.point_poses['HEAD'].xyz.T  # 10 - 12 Head origin Virtual point
            head_plane = Plane(self.point_poses['HDTP'], self.point_poses['REAR'], self.point_poses['LEAR'])
            loc[:, 12:15] = Point.translate_point(self.point_poses['HEAD'], head_plane.normal_vector, direction=100).xyz.T  # 13 - 15 Nasion Skin Surface
            loc[:, 15:18] = Point.translate_point(self.point_poses['HEAD'], head_plane.normal_vector, direction=200).xyz.T  # 16 - 18 Sight end Virtual point
            loc[:, 18:21] = Point.mid_point(self.point_poses['THORAX'], self.point_poses['C7'], 0.85).xyz.T  # 19 - 21 C7/T1 Joint Center
            loc[:, 21:24] = self.point_poses['THORAX'].xyz.T  # 22 - 24 Sternoclavicular Joint Joint Center
            loc[:, 24:27] = self.point_poses['SS'].xyz.T  # 25 - 27 Suprasternale Skin Surface
            loc[:, 27:30] = Point.mid_point(self.point_poses['T8'], self.point_poses['PELVIS_b'], 0.5).xyz.T  # 28 - 30 L5/S1 Joint Center, todo: maybe use PSIS?
            loc[:, 30:33] = self.point_poses['PELVIS_b'].xyz.T  # 31 - 33 PSIS Joint Center
            loc[:, 33:36] = self.point_poses['LSHOULDER'].xyz.T  # 34 - 36 L. Shoulder Joint Center
            loc[:, 36:39] = self.point_poses['LAP'].xyz.T  # 37 - 39 L. Acromion Skin Surface
            loc[:, 39:42] = self.point_poses['LELBOW'].xyz.T  # 40 - 42 L. Elbow Joint Center
            loc[:, 42:45] = self.point_poses['LLE'].xyz.T  # 43 - 45 L. Lat. Epicon. of Humer. Skin Surface
            loc[:, 45:48] = self.point_poses['LWRIST'].xyz.T  # 46 - 48 L. Wrist Joint Center
            loc[:, 48:51] = self.point_poses['LGRIP'].xyz.T  # 49 - 51 L. Grip Center Virtual point
            loc[:, 51:54] = self.point_poses['LHAND'].xyz.T  # 52 - 54 L. Hand Skin Surface
            loc[:, 54:57] = self.point_poses['RSHOULDER'].xyz.T  # 55 - 57 R. Shoulder Joint Center
            loc[:, 57:60] = self.point_poses['RAP'].xyz.T  # 58 - 60 R. Acromion Skin Surface
            loc[:, 60:63] = self.point_poses['RELBOW'].xyz.T  # 61 - 63 R. Elbow Joint Center
            loc[:, 63:66] = self.point_poses['RLE'].xyz.T  # 64 - 66 R. Lat. Epicon. of Humer. Skin Surface
            loc[:, 66:69] = self.point_poses['RWRIST'].xyz.T  # 67 - 69 R. Wrist Joint Center
            loc[:, 69:72] = self.point_poses['RGRIP'].xyz.T  # 70 - 72 R. Grip Center Virtual point
            loc[:, 72:75] = self.point_poses['RHAND'].xyz.T  # 73 - 75 R. Hand Skin Surface
            loc[:, 75:78] = Point.mid_point(self.point_poses['LHIP'], self.point_poses['LKNEE'], 0.7).xyz.T  # 76 - 79 L. Hip Joint Center
                # self.point_poses['LHIP'].xyz.T  # 76 - 79 L. Hip Joint Center
            loc[:, 78:81] = self.point_poses['LKNEE'].xyz.T  # 79 - 81 L. Knee Joint Center
            loc[:, 81:84] = self.point_poses['LLFC'].xyz.T  # 82 - 84 L. Lat. Epicon. of Femur Skin Surface
            loc[:, 84:87] = self.point_poses['LANKLE'].xyz.T  # 85 - 87 L. Ankle Joint Center
            loc[:, 87:90] = self.point_poses['LLM'].xyz.T  # 88 - 90 L. Lateral Malleolus Skin Surface
            loc[:, 90:93] = Point.mid_point(self.point_poses['LFOOT'], self.point_poses['LHEEL'], 0.4).xyz.T  # 91 - 93 L. Ball of Foot Virtual point
            loc[:, 93:96] = self.point_poses['LFOOT'].xyz.T  # 94 - 96 L. Metatarsalphalangeal Skin Surface
            loc[:, 96:99] = Point.mid_point(self.point_poses['RHIP'], self.point_poses['RKNEE'], 0.7).xyz.T
                # self.point_poses['RHIP'].xyz.T  # 97 - 99 R. Hip Joint Center
            loc[:, 99:102] = self.point_poses['RKNEE'].xyz.T  # 100 - 102 R. Knee Joint Center
            loc[:, 102:105] = self.point_poses['RLFC'].xyz.T  # 103 - 105 R. Lat. Epicon. of Femur Skin Surface
            loc[:, 105:108] = self.point_poses['RANKLE'].xyz.T  # 106 - 108 R. Ankle Joint Center
            loc[:, 108:111] = self.point_poses['RLM'].xyz.T  # 109 - 111 R. Lateral Malleolus Skin Surface
            loc[:, 111:114] = Point.mid_point(self.point_poses['RFOOT'], self.point_poses['RHEEL'], 0.4).xyz.T  # 112 - 114 R. Ball of Foot Virtual point
            loc[:, 114:117] = self.point_poses['RFOOT'].xyz.T  # 115 - 117 R. Metatarsalphalangeal Skin Surface
        loc = loc / 1000  # convert to meters
        # convert np list to space separated string
        if loc_file is None:
            loc_file = self.c3d_file.replace('.c3d', '.txt')
        pelvic_tilt_angles = self.get_pelvic_tilt(loc)
        # write as txt file
        with open(loc_file, 'w') as f:
            f.write('3DSSPPBATCHFILE #\n')
            f.write('COM #\n')
            f.write('DES 1 "Task Name" "Analyst Name" "Comments" "Company" #\n')  # English is 0 and metric is 1
            f.write(f'ANT {gender_id} 3 {height} {weight} #\n')  # male 0, female 1, self-set-height-weight-values (not population percentile) 3, height  , weight
            f.write('COM Enabling auto output #\n')  # comment
            f.write('AUT 1 #\n')   # auto output when ANT, HAN, JOA, JOI, and PPR commands are called
            for i, k in enumerate(np.arange(start_frame, end_frame, step)):
                joint_locations = np.array2string(loc[k], separator=' ', max_line_width=1000000, precision=3, suppress_small=True)[1:-1].replace('0. ', '0 ')
                f.write('FRM ' + str(i) + ' #\n')
                f.write(f'LOC {joint_locations} #\n')
                support_feet_max_height = 0.15  # m
                left_foot_supported = True if self.poses['LFOOT'][k, 2] < support_feet_max_height else False  # todo: this only works in world coord
                right_foot_supported = True if self.poses['RFOOT'][k, 2] < support_feet_max_height else False
                if left_foot_supported and right_foot_supported:
                    foot_support_parameter = 0
                elif left_foot_supported and (not right_foot_supported):
                    foot_support_parameter = 1
                elif (not left_foot_supported) and right_foot_supported:
                    foot_support_parameter = 2
                else:
                    foot_support_parameter = 0  # 3
                pelvic_tilt = 0 #pelvic_tilt_angles[k]  # -15
                f.write(f'SUP {foot_support_parameter} 0 0 0 20.0 {pelvic_tilt} #\n')  # set support: Feet Support (0=both, 1=left, 2=right, 3=none), Position (0=stand, 1=seated), Front Seat Pan Support (0=no, 1=yes), Seat Has Back Rest (0=no, 1=yes), and Back Rest Center Height (real number in cm, greater than 19.05).
                hand_load = 0  # N
                f.write(f'HAN {hand_load / 2} -90 0 {hand_load / 2} -90 0 #\n')  # HAN can trigger output write line
            f.write(f'COM Task done #')
        return loc


class PulginGaitSkeleton(Skeleton):
    """A class for plugin gait skeleton"""
    def __init__(self, c3d_file, skeleton_file='config/Plugingait_info/plugingait_VEHS.yaml', acronym_file='config/Plugingait_info/acronym.yaml'):
        super().__init__()
        self.skeleton_file = skeleton_file
        self.__load_acronym(acronym_file)
        self.__load_key_joints(skeleton_file)
        self.joint_number = len(self.key_joint_name)
        self.load_c3d(c3d_file)
        self.c3d_file = c3d_file
        # exclude .c3d
        base_dir = c3d_file[:-4]
        cdf_file = os.path.join(base_dir, 'Pose3D.cdf')
        self.output_cdf(cdf_file)

    def set_weight_height(self, weight=0, height=0):
        """
        set weight in kg and height in m
        """
        if weight == 0 or height == 0:
            self.exp_yaml = self.c3d_file.replace('.c3d', '.yaml')
            with open (self.exp_yaml, 'r') as stream:
                try:
                    data = yaml.safe_load(stream)
                    self.weight = data['weight']
                    self.height = data['height']/1000
                except yaml.YAMLError as exc:
                    print(self.exp_yaml, exc)
        else:
            self.weight = weight
            self.height = height

    def __load_acronym(self, filename):
        # read yaml
        with open(filename, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
                self.lower_bone_acronym = data['lower_bone_acronym']
                self.upper_bone_acronym = data['upper_bone_acronym']
                self.joint_center_acronym = data['joint_center_acronym']
                self.com_acronym = data['com_acronym']
                self.plugingait_marker_acronym = data['plugingait_marker_acronym']
                self.VEHS_marker_acronym = data['VEHS_marker_acronym']
                self.VEHS_virtualPt_acronym = data['VEHS_virtualPt_acronym']
                self.joint_acronym = {**self.lower_bone_acronym, **self.upper_bone_acronym, **self.joint_center_acronym, **self.com_acronym, **self.plugingait_marker_acronym, **self.VEHS_marker_acronym, **self.VEHS_virtualPt_acronym}
            except yaml.YAMLError as exc:
                print(filename, exc)

    def __load_key_joints(self, filename):
        # read xml
        with open (filename, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
                self.joint_name_mid = data['joints']['mid']
                self.joint_name_botL = data['joints']['botL']
                self.joint_name_topL = data['joints']['topL']
                self.joint_name_botR = data['joints']['botR']
                self.joint_name_topR = data['joints']['topR']
                self.key_joint_name = data['joints']['mid'] + data['joints']['botL'] + data['joints']['topL'] + data['joints']['botR'] + data['joints']['topR']
                self.key_joint_parent = data['parent']['mid'] + data['parent']['botL'] + data['parent']['topL'] + data['parent']['botR'] + data['parent']['topR']
                self.key_joint_acronym = []
                for joint_name in self.key_joint_name:
                    self.key_joint_acronym.append(self.acronym(joint_name))
            except yaml.YAMLError as exc:
                print(filename, exc)

    def get_parent(self, joint_name):
        parent_idx = self.key_joint_name.index(joint_name)
        # catch IndexError: list index out of range and return None
        try:
            return self.key_joint_parent[parent_idx]
        except IndexError:
            return None

    def acronym(self, joint_name):
        joint_description = joint_name
        if joint_description in self.joint_acronym.values():
            return list(self.joint_acronym.keys())[list(self.joint_acronym.values()).index(joint_description)]
        else:
            # warning message joint_name not found
            print('Joint name not found: {}'.format(joint_name))
            return None

    def description(self, joint_acronym, origin=True):
        joint_acronym = joint_acronym.strip()
        if joint_acronym in self.joint_acronym.keys():
            if origin:
                return self.joint_acronym[joint_acronym].replace(' Origin', '')
            else:
                return self.joint_acronym[joint_acronym]
        else:
            return None

    def get_pose_idx_from_acronym(self, input_name_list, extract_pt='O'):
        """extract_pt: O, P, A, L, all"""
        pose_idx = {}
        if type(input_name_list) == str:
            input_name_list = [input_name_list]
        else:
            input_name_list = list(input_name_list)
        for joint_wspace in input_name_list:
            joint = joint_wspace.strip()
            if extract_pt in ['O', 'P', 'A', 'L']:
                if joint in self.key_joint_acronym:
                    pose_idx[joint] = self.point_labels.index(joint_wspace)
            elif extract_pt == 'all':
                print(joint)
                if joint in self.point_labels:
                    pose_idx[joint] = self.point_labels.index(joint_wspace)
        return pose_idx

    def get_pose_idx_from_description(self, input_name_list, extract_pt='O'):
        """extract_pt: O, all"""
        pose_idx = {}
        if type(input_name_list) == str:
            input_name_list = [input_name_list]
        else:
            input_name_list = list(input_name_list)
        for joint_description in input_name_list:
            joint_acronym = self.acronym(joint_description)
            if extract_pt in ['O']:
                if joint_acronym in self.joint_acronym.keys():
                    pose_idx[joint_acronym] = self.point_labels.index(joint_acronym)
            elif extract_pt == 'all':
                # replace last character of joint_acronym with O, P, A, L
                joint_base = joint_acronym[:-1]
                for pt in ['O', 'P', 'A', 'L']:
                    joint_acronym = joint_base + pt
                    if joint_acronym in self.joint_acronym.keys():
                        pose_idx[joint_acronym] = self.point_labels.index(joint_acronym)
        return pose_idx

    def get_poses(self, input_data, pose_idx, output_type='dict'):
        if output_type == 'dict':
            poses = {}
            for joint in pose_idx:
                poses[joint] = input_data[:, pose_idx[joint]]
        elif output_type == 'list':
            poses = []
            for joint in pose_idx:
                poses.append(input_data[:, pose_idx[joint]])
        elif output_type == 'list_last':
            print(pose_idx)
            for joint in pose_idx:
                poses = input_data[:, pose_idx[joint],:3]
        else:
            raise ValueError('output_type must be dict, list or list_last')
            poses = None
        return poses

    def get_pose_from_description(self, input_name_list, extract_pt='all'):
        pose_idx = self.get_pose_idx_from_description(input_name_list, extract_pt=extract_pt)
        return self.get_poses(self.points, pose_idx)

    def get_pose_from_acronym(self, input_name_list, extract_pt='all', output_type='dict'):
        pose_idx = self.get_pose_idx_from_acronym(input_name_list, extract_pt=extract_pt)
        return self.get_poses(self.points, pose_idx, output_type=output_type)

    def load_c3d(self, c3d_file):
        reader = c3d.Reader(open(c3d_file, 'rb'))
        try:
            self.analog_labels = reader.analog_labels
            self.analog_labels = [label.strip() for label in self.analog_labels]  # strip whitespace from analog labels
        except AttributeError:
            self.analog_labels = None
        self.point_labels = reader.point_labels
        self.point_labels = [label.strip() for label in self.point_labels]  # strip whitespace from point labels
        self.pose_idx = self.get_pose_idx_from_acronym(self.point_labels)
        points = []
        analog = []
        for i, this_points, this_analog in reader.read_frames():
            print('frame {}: point {}, analog {}'.format(
                i, this_points.shape, this_analog.shape), end='\r')
            points.append(this_points)
            analog.append(this_analog)
        self.points = np.array(points)
        self.analog = np.array(analog)
        self.poses = self.get_poses(self.points, self.pose_idx)
        self.frame_no = self.points.shape[0]

    def plot_3d_pose_frame(self, frame=0):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for joint_name in self.key_joint_name:
            joint = self.acronym(joint_name)
            if joint_name in self.joint_name_mid:
                point_type = 's'
                point_size = 15
            else:
                point_type = 'o'
                point_size = 10
            ax.scatter(self.poses[joint][frame, 0],
                       self.poses[joint][frame, 1],
                       self.poses[joint][frame, 2], label=joint_name, marker=point_type,s=point_size)
        # connect points to parent
        for joint_name in self.key_joint_name:
            joint = self.acronym(joint_name)
            parent_name = self.get_parent(joint_name)
            if parent_name is not None and parent_name != 'None':
                parent = self.acronym(parent_name)
                ax.plot([self.poses[joint][frame, 0], self.poses[parent][frame, 0]],
                        [self.poses[joint][frame, 1], self.poses[parent][frame, 1]],
                        [self.poses[joint][frame, 2], self.poses[parent][frame, 2]], 'k-')

        # ax.legend(bbox_to_anchor=(0.95, 1), loc=2, borderaxespad=0.)
        # uniform scale based on pelvis location and 1800mm
        range = 1800
        pelvis_loc = self.poses['PELO'][frame, :]
        ax.set_xlim(pelvis_loc[0] - range / 2, pelvis_loc[0] + range / 2)
        ax.set_ylim(pelvis_loc[1] - range / 2, pelvis_loc[1] + range / 2)
        ax.set_zlim(pelvis_loc[2] - range / 2, pelvis_loc[2] + range / 2)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        fig.tight_layout()
        fig.subplots_adjust(right=0.65)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7)
        # plt.show()
        return fig, ax

    def add_point_to_plot(self, point_acronym, ax, fig, frame=0):
        if type(point_acronym) == str:
            point_acronym_list = [point_acronym]
        else:
            point_acronym_list = list(point_acronym)
        for point_acronym in point_acronym_list:
            poses = self.get_pose_from_acronym(point_acronym, extract_pt='all')
            ax.scatter(poses[point_acronym][frame, 0],
                       poses[point_acronym][frame, 1],
                       poses[point_acronym][frame, 2], label=point_acronym, s=25)
        return fig, ax

    def output_cdf(self, file_name):
        pose_3D = []
        pose_sequence = []
        for joint_name in self.key_joint_name:
            joint = self.acronym(joint_name)
            pose_3D.append(self.poses[joint])
            pose_sequence.append(joint)
        pose_3D = np.array(pose_3D)
        pose_3D = np.transpose(pose_3D, (1, 0, 2))

        create_dir(os.path.dirname(file_name))
        if os.path.exists(file_name):
            os.remove(file_name)
        cdf = pycdf.CDF(file_name, '')
        cdf['Pose'] = pose_3D
        cdf.attrs['PoseSequence'] = pose_sequence
        # cdf.attrs['SubjectID'] = subjectID
        # cdf.attrs['TaskID'] = TaskID
        # cdf.attrs['CamID'] = CamID
        # cdf.attrs['UpdateDate'] = datetime.datetime.now()
        # cdf.attrs['CaptureDate'] = os.path.basename(date)
        # cdf.attrs['KeypointNames'] = kp_names
        cdf.close()

    def output_3DSSPP_loc(self, frame_range=None,loc_file=None):
        # 3DSSPP format:
        #LOC File filename.loc
        # Value Anatomical Location Attribute
        # 1 - 3 Top Head Skin Surface
        # 4 - 6 L. Head Skin Surface
        # 7 - 9 R. Head Skin Surface
        # 10 - 12 Head origin Virtual point
        # 13 - 15 Nasion Skin Surface
        # 16 - 18 Sight end Virtual point
        # 19 - 21 C7/T1 Joint Center
        # 22 - 24 Sternoclavicular Joint Joint Center
        # 25 - 27 Suprasternale Skin Surface
        # 28 - 30 L5/S1 Joint Center
        # 31 - 33 PSIS Joint Center
        # 34 - 36 L. Shoulder Joint Center
        # 37 - 39 L. Acromion Skin Surface
        # 40 - 42 L. Elbow Joint Center
        # 43 - 45 L. Lat. Epicon. of Humer. Skin Surface
        # 46 - 48 L. Wrist Joint Center
        # 49 - 51 L. Grip Center Virtual point
        # 52 - 54 L. Hand Skin Surface
        # 55 - 57 R. Shoulder Joint Center
        # 58 - 60 R. Acromion Skin Surface
        # 61 - 63 R. Elbow Joint Center
        # 64 - 66 R. Lat. Epicon. of Humer. Skin Surface
        # 67 - 69 R. Wrist Joint Center
        # 70 - 72 R. Grip Center Virtual point
        # 73 - 75 R. Hand Skin Surface
        # 76 - 79 L. Hip Joint Center
        # 79 - 81 L. Knee Joint Center
        # 82 - 84 L. Lat. Epicon. of Femur Skin Surface
        # 85 - 87 L. Ankle Joint Center
        # 88 - 90 L. Lateral Malleolus Skin Surface
        # 91 - 93 L. Ball of Foot Virtual point
        # 94 - 96 L. Metatarsalphalangeal Skin Surface
        # 97 - 99 R. Hip Joint Center
        # 100 - 102 R. Knee Joint Center
        # 103 - 105 R. Lat. Epicon. of Femur Skin Surface
        # 106 - 108 R. Ankle Joint Center
        # 109 - 111 R. Lateral Malleolus Skin Surface
        # 112 - 114 R. Ball of Foot Virtual point
        # 115 - 117 R. Metatarsalphalangeal Skin Surface
        try:
            weight = self.weight
        except:
            weight = 70
        try:
            height = self.height
        except:
            height = 180
        if frame_range is not None:
            start_frame = frame_range[0]
            end_frame = frame_range[1]
            step = frame_range[2]
        else:
            start_frame = 0
            end_frame = self.frame_no
            step = 1
        if True:
            fill = 0
            loc = np.zeros((self.frame_no, 117))
            # loc[:,0:3] = self.get_pose_from_acronym('HDTP', extract_pt='all', output_type='list_last')  # 1 - 3 Top Head Skin Surface
            # loc[:,3:6] = self.get_pose_from_acronym('LHEC', extract_pt='all', output_type='list_last')  # 4 - 6 L. Head Skin Surface
            # loc[:,6:9] = self.get_pose_from_acronym('RHEC', extract_pt='all', output_type='list_last')  # 7 - 9 R. Head Skin Surface
            loc[:,9:12] = self.get_pose_from_acronym('BHEC', extract_pt='all', output_type='list_last')  # 10 - 12 Head origin Virtual point
            loc[:,12:15] = self.get_pose_from_acronym('HDEY', extract_pt='all', output_type='list_last')  # 13 - 15 Nasion Skin Surface
            # loc[:,15:18] =  self.get_pose_from_acronym('HDTP', extract_pt='all', output_type='list_last')  # 16 - 18 Sight end Virtual point
            loc[:,18:21] = (self.get_pose_from_acronym('C7', extract_pt='all', output_type='list_last')*3+self.get_pose_from_acronym('T10', extract_pt='all', output_type='list_last')*1)/4  # 19 - 21 C7/T1 Joint Center
            loc[:,21:24] = (self.get_pose_from_acronym('TRXO', extract_pt='all', output_type='list_last')*3 +self.get_pose_from_acronym('T10', extract_pt='all', output_type='list_last')*1)/4 # 22 - 24 Sternoclavicular Joint Joint Center
            # loc[:,24:27] = self.get_pose_from_acronym('CLAV', extract_pt='all', output_type='list_last')  # 25 - 27 Suprasternale Skin Surface
            loc[:,27:30] = (self.get_pose_from_acronym('RPSI', extract_pt='all', output_type='list_last')+self.get_pose_from_acronym('LPSI', extract_pt='all', output_type='list_last')
                             + self.get_pose_from_acronym('RASI', extract_pt='all', output_type='list_last') + self.get_pose_from_acronym('LASI', extract_pt='all', output_type='list_last')
                            )/4  # 28 - 30 L5/S1 Joint Center
            loc[:,30:33] = fill  # 31 - 33 PSIS Joint Center
            loc[:,33:36] = self.get_pose_from_acronym('LSHO', extract_pt='all', output_type='list_last')  # 34 - 36 L. Shoulder Joint Center
            # loc[:,36:39] = self.get_pose_from_acronym('LSHO', extract_pt='all', output_type='list_last')  # 37 - 39 L. Acromion Skin Surface
            loc[:,39:42] = self.get_pose_from_acronym('LEJC', extract_pt='all', output_type='list_last')  # 40 - 42 L. Elbow Joint Center
            loc[:,42:45] = fill  # 43 - 45 L. Lat. Epicon. of Humer. Skin Surface
            loc[:,45:48] = self.get_pose_from_acronym('LWJC', extract_pt='all', output_type='list_last')  # 46 - 48 L. Wrist Joint Center
            loc[:,48:51] = self.get_pose_from_acronym('LMFO', extract_pt='all', output_type='list_last')  # 49 - 51 L. Grip Center Virtual point
            loc[:,51:54] = self.get_pose_from_acronym('LFIN', extract_pt='all', output_type='list_last')  # 52 - 54 L. Hand Skin Surface
            loc[:,54:57] = self.get_pose_from_acronym('RSHO', extract_pt='all', output_type='list_last')  # 55 - 57 R. Shoulder Joint Center
            # loc[:,57:60] = self.get_pose_from_acronym('RSHO', extract_pt='all', output_type='list_last')  # 58 - 60 R. Acromion Skin Surface
            loc[:,60:63] = self.get_pose_from_acronym('REJC', extract_pt='all', output_type='list_last')  # 61 - 63 R. Elbow Joint Center
            loc[:,63:66] = fill  # 64 - 66 R. Lat. Epicon. of Humer. Skin Surface
            loc[:,66:69] = self.get_pose_from_acronym('RWJC', extract_pt='all', output_type='list_last')  # 67 - 69 R. Wrist Joint Center
            loc[:,69:72] = self.get_pose_from_acronym('RMFO', extract_pt='all', output_type='list_last')  # 70 - 72 R. Grip Center Virtual point
            loc[:,72:75] = self.get_pose_from_acronym('RFIN', extract_pt='all', output_type='list_last')  # 73 - 75 R. Hand Skin Surface
            loc[:,75:78] = self.get_pose_from_acronym('LHJC', extract_pt='all', output_type='list_last')  # 76 - 78 L. Hip Joint Center
            loc[:,78:81] = self.get_pose_from_acronym('LKJC', extract_pt='all', output_type='list_last')  # 79 - 81 L. Knee Joint Center
            loc[:,81:84] = fill  # 82 - 84 L. Lat. Epicon. of Femur Skin Surface
            loc[:,84:87] = self.get_pose_from_acronym('LAJC', extract_pt='all', output_type='list_last')  # 85 - 87 L. Ankle Joint Center
            loc[:,87:90] = self.get_pose_from_acronym('RANK', extract_pt='all', output_type='list_last')  # 88 - 90 L. Lateral Malleolus Skin Surface
            loc[:,90:93] = self.get_pose_from_acronym('LFOO', extract_pt='all', output_type='list_last')  # 91 - 93 L. Ball of Foot Virtual point
            loc[:,93:96] = fill #self.get_pose_from_acronym('LTOE', extract_pt='all', output_type='list_last')  # 94 - 96 L. Metatarsalphalangeal Skin Surface
            loc[:,96:99] = self.get_pose_from_acronym('RHJC', extract_pt='all', output_type='list_last')  # 97 - 99 R. Hip Joint Center
            loc[:,99:102] = self.get_pose_from_acronym('RKJC', extract_pt='all', output_type='list_last')  # 100 - 102 R. Knee Joint Center
            loc[:,102:105] = fill  # 103 - 105 R. Lat. Epicon. of Femur Skin Surface
            loc[:,105:108] = self.get_pose_from_acronym('RAJC', extract_pt='all', output_type='list_last')  # 106 - 108 R. Ankle Joint Center
            loc[:,108:111] = self.get_pose_from_acronym('RANK', extract_pt='all', output_type='list_last')  # 109 - 111 R. Lateral Malleolus Skin Surface
            loc[:,111:114] = self.get_pose_from_acronym('RFOO', extract_pt='all', output_type='list_last')  # 112 - 114 R. Ball of Foot Virtual point
            loc[:,114:117] = fill #self.get_pose_from_acronym('RTOE', extract_pt='all', output_type='list_last')  # 115 - 117 R. Metatarsalphalangeal Skin Surface
        loc = loc/1000  # convert to meters
        # convert np list to space separated string
        if loc_file is None:
            loc_file = self.c3d_file.replace('.c3d', '.txt')
        # write as txt file
        with open(loc_file, 'w') as f:
            f.write('3DSSPPBATCHFILE #\n')
            f.write('COM #\n')
            f.write('DES 1 "Task Name" "Analyst Name" "Comments" "Company" #\n') # English is 0 and metric is 1
            for i,k in enumerate(np.arange(start_frame, end_frame, step)):
                joint_locations = np.array2string(loc[k], separator=' ', max_line_width=1000000, precision=3, suppress_small=True)[1:-1].replace('0. ', '0 ')
                # f.write('AUT 1 #\n')
                f.write('FRM ' + str(i + 1) + ' #\n')
                f.write(f'ANT 0 3 {height} {weight} #\n')  # male 0, female 1, self-set 3, height  , weight

                f.write(f'LOC {joint_locations} #\n')
                # f.write('HAN 15 -20 85 15 -15 80 #\n')
                # f.write('EXP #\n')
            # f.write('AUT 1 #\n')

        return loc


class VEHSErgoSkeleton_angles(VEHSErgoSkeleton):
    def __init__(self, skeleton_file='config\VEHS_ErgoSkeleton_info\Ergo-Skeleton.yaml'):
        super().__init__(skeleton_file)
        self.angle_names = ['neck', 'right_shoulder', 'left_shoulder', 'right_elbow', 'left_elbow', 'right_wrist', 'left_wrist', 'back', 'right_knee', 'left_knee']

    def neck_angles(self):
        zero_frame = [-90, -180, -180]
        REAR = self.point_poses['REAR']
        LEAR = self.point_poses['LEAR']
        HDTP = self.point_poses['HDTP']
        EAR = Point.mid_point(REAR, LEAR)
        RSHOULDER = self.point_poses['RSHOULDER']
        LSHOULDER = self.point_poses['LSHOULDER']
        C7 = self.point_poses['C7']
        RPSIS = self.point_poses['RPSIS']
        LPSIS = self.point_poses['LPSIS']

        HEAD_plane = Plane()
        HEAD_plane.set_by_pts(REAR, LEAR, HDTP)
        HEAD_coord = CoordinateSystem3D()
        HEAD_coord.set_by_plane(HEAD_plane, EAR, HDTP, sequence='yxz', axis_positive=True)
        NECK_angles = JointAngles()
        NECK_angles.ergo_name = {'flexion': 'flexion', 'abduction': 'L-bend', 'rotation': 'rotation'}  # lateral bend
        NECK_angles.set_zero(zero_frame, by_frame=False)
        NECK_angles.get_flex_abd(HEAD_coord, Point.vector(C7, Point.mid_point(RPSIS, LPSIS)), plane_seq=['xy', 'yz'], flip_sign=[1, -1])
        NECK_angles.get_rot(LEAR, REAR, LSHOULDER, RSHOULDER)
        return NECK_angles

    def right_shoulder_angles(self):
        zero_frame = [0, 90, 90]
        RSHOULDER = self.point_poses['RSHOULDER']
        C7 = self.point_poses['C7']
        C7_d = self.point_poses['C7_d']
        PELVIS_b = Point.mid_point(self.point_poses['RPSIS'], self.point_poses['LPSIS'])
        # C7_m = self.point_poses['C7_m']
        SS = self.point_poses['SS']
        RELBOW = self.point_poses['RELBOW']
        RAP_b = self.point_poses['RAP_b']
        RAP_f = self.point_poses['RAP_f']
        RME = self.point_poses['RME']
        RLE = self.point_poses['RLE']

        RSHOULDER_plane = Plane()
        RSHOULDER_plane.set_by_vector(RSHOULDER, Point.vector(C7_d, PELVIS_b), direction=-1)
        # RSHOULDER_C7_m_project = RSHOULDER_plane.project_point(C7_m)
        RSHOULDER_SS_project = RSHOULDER_plane.project_point(SS)
        RSHOULDER_coord = CoordinateSystem3D()
        RSHOULDER_coord.set_by_plane(RSHOULDER_plane, C7_d, RSHOULDER_SS_project, sequence='xyz', axis_positive=True)  # new: use back to chest vector
        # RSHOULDER_coord.set_by_plane(RSHOULDER_plane, RSHOULDER, RSHOULDER_C7_m_project, sequence='zyx', axis_positive=False)  # old: use shoulder to chest vector
        RSHOULDER_angles = JointAngles()
        RSHOULDER_angles.ergo_name = {'flexion': 'elevation', 'abduction': 'H-abduction', 'rotation': 'rotation'}
        RSHOULDER_angles.set_zero(zero_frame, by_frame=False)
        RSHOULDER_angles.get_flex_abd(RSHOULDER_coord, Point.vector(RSHOULDER, RELBOW), plane_seq=['xy', 'xz'])
        RSHOULDER_angles.get_rot(RAP_b, RAP_f, RME, RLE)
        RSHOULDER_angles.flexion = Point.angle(Point.vector(RSHOULDER, RELBOW).xyz, Point.vector(C7, PELVIS_b).xyz)
        RSHOULDER_angles.flexion = RSHOULDER_angles.zero_by_idx(0)  # zero by zero frame after setting flexion without function

        # todo: add this filter as a function; set undefined angles to zero
        shoulder_threshold = 10/180*np.pi  # the H-abduction is not well defined when the flexion is small or near 180 degrees
        shoulder_filter = np.logical_and(np.abs(RSHOULDER_angles.flexion) > shoulder_threshold, np.abs(RSHOULDER_angles.flexion) < (np.pi - shoulder_threshold))
        RSHOULDER_angles.abduction = np.array([np.where(shoulder_filter[i], RSHOULDER_angles.abduction[i], 0) for i in range(len(shoulder_filter))])  # set abduction to nan if shoulder filter is false
        return RSHOULDER_angles

    def left_shoulder_angles(self):     # not checked
        zero_frame = [0, 0, -90]
        LSHOULDER = self.point_poses['LSHOULDER']
        C7 = self.point_poses['C7']
        C7_d = self.point_poses['C7_d']
        PELVIS_b = Point.mid_point(self.point_poses['RPSIS'], self.point_poses['LPSIS'])
        # C7_m = self.point_poses['C7_m']
        SS = self.point_poses['SS']
        LELBOW = self.point_poses['LELBOW']
        LAP_b = self.point_poses['LAP_b']
        LAP_f = self.point_poses['LAP_f']
        LME = self.point_poses['LME']
        LLE = self.point_poses['LLE']

        LSHOULDER_plane = Plane()
        LSHOULDER_plane.set_by_vector(LSHOULDER, Point.vector(C7_d, PELVIS_b), direction=-1)
        # LSHOULDER_C7_m_project = LSHOULDER_plane.project_point(C7_m)
        LSHOULDER_SS_project = LSHOULDER_plane.project_point(SS)
        LSHOULDER_coord = CoordinateSystem3D()
        LSHOULDER_coord.set_by_plane(LSHOULDER_plane, C7_d, LSHOULDER_SS_project, sequence='zyx', axis_positive=True)
        LSHOULDER_angles = JointAngles()
        LSHOULDER_angles.ergo_name = {'flexion': 'elevation', 'abduction': 'H-abduction', 'rotation': 'rotation'}  # horizontal abduction
        LSHOULDER_angles.set_zero(zero_frame, by_frame=False)
        LSHOULDER_angles.get_flex_abd(LSHOULDER_coord, Point.vector(LSHOULDER, LELBOW), plane_seq=['xy', 'xz'], flip_sign=[1, -1])
        LSHOULDER_angles.get_rot(LAP_b, LAP_f, LME, LLE)
        LSHOULDER_angles.flexion = Point.angle(Point.vector(LSHOULDER, LELBOW).xyz, Point.vector(C7, PELVIS_b).xyz)
        LSHOULDER_angles.flexion = LSHOULDER_angles.zero_by_idx(0)
        shoulder_threshold = 10/180*np.pi
        shoulder_filter = np.logical_and(np.abs(LSHOULDER_angles.flexion) > shoulder_threshold, np.abs(LSHOULDER_angles.flexion) < (np.pi - shoulder_threshold))
        LSHOULDER_angles.abduction = np.array([np.where(shoulder_filter[i], LSHOULDER_angles.abduction[i], 0) for i in range(len(shoulder_filter))])
        return LSHOULDER_angles

    def right_elbow_angles(self):
        zero_frame = [-180, 0, 0]
        RELBOW = self.point_poses['RELBOW']
        RSHOULDER = self.point_poses['RSHOULDER']
        RWRIST = self.point_poses['RWRIST']

        RELBOW_angles = JointAngles()
        RELBOW_angles.ergo_name = {'flexion': 'flexion', 'abduction': 'na', 'rotation': 'na'}
        RELBOW_angles.set_zero(zero_frame, by_frame=False)
        RELBOW_angles.flexion = -Point.angle(Point.vector(RELBOW, RSHOULDER).xyz, Point.vector(RELBOW, RWRIST).xyz)
        RELBOW_angles.flexion = RELBOW_angles.zero_by_idx(0)  # zero by zero frame
        RELBOW_angles.is_empty = False
        RELBOW_angles.abduction = None
        RELBOW_angles.rotation = None
        return RELBOW_angles

    def left_elbow_angles(self):  # not checked
        zero_frame = [-180, 0, 0]
        LELBOW = self.point_poses['LELBOW']
        LSHOULDER = self.point_poses['LSHOULDER']
        LWRIST = self.point_poses['LWRIST']

        LELBOW_angles = JointAngles()
        LELBOW_angles.ergo_name = {'flexion': 'flexion', 'abduction': 'na', 'rotation': 'na'}
        LELBOW_angles.set_zero(zero_frame, by_frame=False)
        LELBOW_angles.flexion = -Point.angle(Point.vector(LELBOW, LSHOULDER).xyz, Point.vector(LELBOW, LWRIST).xyz)
        LELBOW_angles.flexion = LELBOW_angles.zero_by_idx(0)
        LELBOW_angles.is_empty = False
        LELBOW_angles.abduction = None
        LELBOW_angles.rotation = None
        return LELBOW_angles

    def right_wrist_angles(self):
        zero_frame = [-90, -180, -90]
        RWRIST = self.point_poses['RWRIST']
        RMCP2 = self.point_poses['RMCP2']
        RMCP5 = self.point_poses['RMCP5']
        RHAND = self.point_poses['RHAND']
        RRS = self.point_poses['RRS']
        RUS = self.point_poses['RUS']
        RLE = self.point_poses['RLE']
        RME = self.point_poses['RME']
        RELBOW = self.point_poses['RELBOW']

        RWRIST_plane = Plane()
        RWRIST_plane.set_by_pts(RMCP2, RWRIST, RMCP5)
        RWRIST_coord = CoordinateSystem3D()
        RWRIST_coord.set_by_plane(RWRIST_plane, RWRIST, RHAND, sequence='yxz', axis_positive=True)
        RWRIST_angles = JointAngles()
        RWRIST_angles.ergo_name = {'flexion': 'flexion', 'abduction': 'deviation', 'rotation': 'pronation'}
        RWRIST_angles.set_zero(zero_frame, by_frame=False)
        RWRIST_angles.get_flex_abd(RWRIST_coord, Point.vector(RWRIST, RELBOW), plane_seq=['xy', 'yz'])
        RWRIST_angles.get_rot(RRS, RUS, RLE, RME)
        # RWRIST_angles.get_rot(RMCP2, RMCP5, RLE, RME)
        return RWRIST_angles

    def left_wrist_angles(self):  # not checked
        zero_frame = [-90, -180, 90]
        LWrist = self.point_poses['LWRIST']
        LMCP2 = self.point_poses['LMCP2']
        LMCP5 = self.point_poses['LMCP5']
        LHand = self.point_poses['LHAND']
        LRS = self.point_poses['LRS']
        LUS = self.point_poses['LUS']
        LLE = self.point_poses['LLE']
        LME = self.point_poses['LME']
        LELBOW = self.point_poses['LELBOW']

        LWrist_plane = Plane()
        LWrist_plane.set_by_pts(LMCP2, LWrist, LMCP5)
        LWrist_coord = CoordinateSystem3D()
        LWrist_coord.set_by_plane(LWrist_plane, LWrist, LHand, sequence='yxz', axis_positive=True)
        LWrist_angles = JointAngles()
        LWrist_angles.ergo_name = {'flexion': 'flexion', 'abduction': 'deviation ', 'rotation': 'pronation'}
        LWrist_angles.set_zero(zero_frame, by_frame=False)
        LWrist_angles.get_flex_abd(LWrist_coord, Point.vector(LWrist, LELBOW), plane_seq=['xy', 'yz'])
        LWrist_angles.get_rot(LRS, LUS, LLE, LME)
        # LWrist_angles.get_rot(LMCP2, LMCP5, LLE, LME)
        return LWrist_angles

    def back_angles(self, up_axis=[0, 1000, 0], zero_frame = [-90, 180, 180]):
        # todo: back correction
        C7 = self.point_poses['C7']
        RPSIS = self.point_poses['RPSIS']
        LPSIS = self.point_poses['LPSIS']
        RHIP = self.point_poses['RHIP']
        LHIP = self.point_poses['LHIP']
        RSHOULDER = self.point_poses['RSHOULDER']
        LSHOULDER = self.point_poses['LSHOULDER']
        PELVIS_b = Point.mid_point(RPSIS, LPSIS)

        BACK_plane = Plane()
        BACK_plane.set_by_vector(PELVIS_b, Point.create_const_vector(*up_axis, examplePt=PELVIS_b), direction=1)
        BACK_coord = CoordinateSystem3D()
        # BACK_RPSIS_PROJECT = BACK_plane.project_point(RPSIS)
        BACK_RHIP_PROJECT = BACK_plane.project_point(RHIP)
        BACK_coord.set_by_plane(BACK_plane, PELVIS_b, BACK_RHIP_PROJECT, sequence='zyx', axis_positive=True)
        BACK_angles = JointAngles()
        BACK_angles.ergo_name = {'flexion': 'flexion', 'abduction': 'L-flexion', 'rotation': 'rotation'}  #lateral flexion
        BACK_angles.set_zero(zero_frame)
        BACK_angles.get_flex_abd(BACK_coord, Point.vector(PELVIS_b, C7), plane_seq=['xy', 'yz'])
        # BACK_angles.get_rot(RSHOULDER, LSHOULDER, RPSIS, LPSIS, flip_sign=1)
        BACK_angles.get_rot(RSHOULDER, LSHOULDER, RHIP, LHIP, flip_sign=1)

        return BACK_angles

    def right_knee_angles(self):
        zero_frame = -180
        RKNEE = self.point_poses['RKNEE']
        RHIP = self.point_poses['RHIP']
        RANKLE = self.point_poses['RANKLE']

        RKNEE_angles = JointAngles()
        RKNEE_angles.ergo_name = {'flexion': 'flexion', 'abduction': 'na', 'rotation': 'na'}
        RKNEE_angles.set_zero(zero_frame, by_frame=False)
        RKNEE_angles.flexion = -Point.angle(Point.vector(RKNEE, RHIP).xyz, Point.vector(RKNEE, RANKLE).xyz)
        RKNEE_angles.flexion = RKNEE_angles.zero_by_idx(0)  # zero by zero frame
        RKNEE_angles.is_empty = False
        RKNEE_angles.abduction = None
        RKNEE_angles.rotation = None
        return RKNEE_angles

    def left_knee_angles(self):  # not checked
        zero_frame = -180
        LKNEE = self.point_poses['LKNEE']
        LHIP = self.point_poses['LHIP']
        LANKLE = self.point_poses['LANKLE']

        LKNEE_angles = JointAngles()
        LKNEE_angles.ergo_name = {'flexion': 'flexion', 'abduction': 'na', 'rotation': 'na'}
        LKNEE_angles.set_zero(zero_frame, by_frame=False)
        LKNEE_angles.flexion = -Point.angle(Point.vector(LKNEE, LHIP).xyz, Point.vector(LKNEE, LANKLE).xyz)
        LKNEE_angles.flexion = LKNEE_angles.zero_by_idx(0)
        LKNEE_angles.is_empty = False
        LKNEE_angles.abduction = None
        LKNEE_angles.rotation = None
        return LKNEE_angles


class H36MSkeleton_angles(VEHSErgoSkeleton_angles):
    def __init__(self, skeleton_file='config\VEHS_ErgoSkeleton_info\Ergo-Skeleton.yaml'):
        super().__init__(skeleton_file)
        self.angle_names = ['right_shoulder', 'left_shoulder', 'right_elbow', 'left_elbow', 'back', 'right_knee', 'left_knee']

    def right_shoulder_angles(self):
        zero_frame = [0, 90, 90]
        RSHOULDER = self.point_poses['RSHOULDER']
        LSHOULDER = self.point_poses['LSHOULDER']
        THORAX = self.point_poses['THORAX']
        PELVIS = self.point_poses['PELVIS']
        RELBOW = self.point_poses['RELBOW']

        RSHOULDER_plane = Plane()
        RSHOULDER_plane.set_by_pts(PELVIS, LSHOULDER, RSHOULDER)
        RSHOULDER_coord = CoordinateSystem3D()
        RSHOULDER_coord.set_by_plane(RSHOULDER_plane, RSHOULDER, LSHOULDER, sequence='zxy', axis_positive=False)
        RSHOULDER_angles = JointAngles()

        RSHOULDER_angles.ergo_name = {'flexion': 'elevation', 'abduction': 'H-abduction', 'rotation': 'na'}
        RSHOULDER_angles.set_zero(zero_frame, by_frame=False)
        RSHOULDER_angles.get_flex_abd(RSHOULDER_coord, Point.vector(RSHOULDER, RELBOW), plane_seq=['xy', 'xz'])
        RSHOULDER_angles.flexion = Point.angle(Point.vector(RSHOULDER, RELBOW).xyz, Point.vector(THORAX, PELVIS).xyz)
        RSHOULDER_angles.flexion = RSHOULDER_angles.zero_by_idx(0)  # zero by zero frame after setting flexion without function

        shoulder_threshold = 10/180*np.pi  # the H-abduction is not well defined when the flexion is small or near 180 degrees
        shoulder_filter = np.logical_and(np.abs(RSHOULDER_angles.flexion) > shoulder_threshold, np.abs(RSHOULDER_angles.flexion) < (np.pi - shoulder_threshold))
        RSHOULDER_angles.abduction = np.array([np.where(shoulder_filter[i], RSHOULDER_angles.abduction[i], 0) for i in range(len(shoulder_filter))])  # set abduction to nan if shoulder filter is false
        RSHOULDER_angles.rotation = None
        return RSHOULDER_angles

    def left_shoulder_angles(self):
        zero_frame = [0, -90, 90]
        RSHOULDER = self.point_poses['RSHOULDER']
        LSHOULDER = self.point_poses['LSHOULDER']
        THORAX = self.point_poses['THORAX']
        PELVIS = self.point_poses['PELVIS']
        LELBOW = self.point_poses['LELBOW']

        LSHOULDER_plane = Plane()
        LSHOULDER_plane.set_by_pts(PELVIS, LSHOULDER, RSHOULDER)
        LSHOULDER_coord = CoordinateSystem3D()
        LSHOULDER_coord.set_by_plane(LSHOULDER_plane, LSHOULDER, RSHOULDER, sequence='zxy', axis_positive=True)
        LSHOULDER_angles = JointAngles()
        LSHOULDER_angles.ergo_name = {'flexion': 'elevation', 'abduction': 'H-abduction', 'rotation': 'na'}
        LSHOULDER_angles.set_zero(zero_frame, by_frame=False)
        LSHOULDER_angles.get_flex_abd(LSHOULDER_coord, Point.vector(LSHOULDER, LELBOW), plane_seq=['xy', 'xz'], flip_sign=[1, -1])
        LSHOULDER_angles.flexion = Point.angle(Point.vector(LSHOULDER, LELBOW).xyz, Point.vector(THORAX, PELVIS).xyz)
        LSHOULDER_angles.flexion = LSHOULDER_angles.zero_by_idx(0)  # zero by zero frame after setting flexion without function

        shoulder_threshold = 10 / 180 * np.pi  # the H-abduction is not well defined when the flexion is small or near 180 degrees
        shoulder_filter = np.logical_and(np.abs(LSHOULDER_angles.flexion) > shoulder_threshold, np.abs(LSHOULDER_angles.flexion) < (np.pi - shoulder_threshold))
        LSHOULDER_angles.abduction = np.array(
            [np.where(shoulder_filter[i], LSHOULDER_angles.abduction[i], 0) for i in range(len(shoulder_filter))])  # set abduction to nan if shoulder filter is false

        LSHOULDER_angles.rotation = None
        return LSHOULDER_angles

    def back_angles(self, up_axis=[0, 1000, 0], zero_frame = [-90, 180, 180]):
        # todo: back correction
        THORAX = self.point_poses['THORAX']
        RHIP = self.point_poses['RHIP']
        LHIP = self.point_poses['LHIP']
        RSHOULDER = self.point_poses['RSHOULDER']
        LSHOULDER = self.point_poses['LSHOULDER']
        PELVIS = self.point_poses['PELVIS']

        BACK_plane = Plane()
        BACK_plane.set_by_vector(PELVIS, Point.create_const_vector(*up_axis, examplePt=PELVIS), direction=1)
        BACK_coord = CoordinateSystem3D()
        BACK_RHIP_PROJECT = BACK_plane.project_point(RHIP)
        BACK_coord.set_by_plane(BACK_plane, PELVIS, BACK_RHIP_PROJECT, sequence='zyx', axis_positive=True)
        BACK_angles = JointAngles()
        BACK_angles.ergo_name = {'flexion': 'flexion', 'abduction': 'L-flexion', 'rotation': 'rotation'}  #lateral flexion
        BACK_angles.set_zero(zero_frame)
        BACK_angles.get_flex_abd(BACK_coord, Point.vector(PELVIS, THORAX), plane_seq=['xy', 'yz'])
        BACK_angles.get_rot(RSHOULDER, LSHOULDER, RHIP, LHIP, flip_sign=1)

        return BACK_angles


class RokokoHandSkeleton(Skeleton):
    def __init__(self, skeleton_file=r"config/VEHS_ErgoSkeleton_info/Ergo-Hand-21.yaml"):
        super().__init__(skeleton_file)

    def load_rokoko_csv(self, csv_file=r"/Users/leyangwen/Documents/Hand/blender_joints_take1.csv",
                        handiness='LeftHand', random_rotation=False, flip_left=False, ignore_first_n_frame=5):
        """
        load rokoko csv file
        :param handiness: 'l' or 'r'
        """
        if handiness[0].lower() == 'l':
            handiness = 'LeftHand'
        elif handiness[0].lower() == 'r':
            handiness = 'RightHand'
        else:
            raise ValueError(f"handiness should be 'LeftHand' or 'RightHand', got {handiness}")
        self.c3d_file = csv_file
        # csv_file = r"/Users/leyangwen/Documents/Hand/blender_joints_take1.csv"
        df = pd.read_csv(csv_file)
        name_list = ['Wrist', 'Thumb_0', 'Thumb_1', 'Thumb_2', 'Thumb_3',
                    'Index_0', 'Index_1', 'Index_2', 'Index_3',
                    'Middle_0', 'Middle_1', 'Middle_2', 'Middle_3',
                    'Ring_0', 'Ring_1', 'Ring_2', 'Ring_3',
                    'Pinky_0', 'Pinky_1', 'Pinky_2', 'Pinky_3']
        # get all hand columns
        hand_columns = [col for col in df.columns if handiness in col]
        hand_columns = [col for col in hand_columns if ('1_tail' not in col and '2_tail' not in col)]
        hand_21_columns = hand_columns[:3]+hand_columns[6:]
        assert len(hand_21_columns) == 21*3
        # rearrange
        start_idx = [3+12*4, 3, 3+12*1, 3+12*3, 3+12*2]  # order to wrist, thumb, index, middle, ring, pinky
        hand_21_columns_order = hand_21_columns[0:3] + hand_21_columns[start_idx[0]:start_idx[0]+12] + hand_21_columns[start_idx[1]:start_idx[1]+12] + hand_21_columns[start_idx[2]:start_idx[2]+12] + hand_21_columns[start_idx[3]:start_idx[3]+12] + hand_21_columns[start_idx[4]:start_idx[4]+12]
        # print(hand_21_columns_order)
        hand_pose = np.array(df[hand_21_columns_order]).reshape((-1, 21, 3))*1000 # convert to mm
        hand_pose = hand_pose[ignore_first_n_frame:]  # first few (2 at least) frame is bad sometimes
        if random_rotation:
            hand_pose = self.random_rotation(hand_pose)
        if flip_left and handiness == 'LeftHand':
            hand_pose = self.flip_hand(hand_pose)
        self.load_name_list_and_np_points(name_list, hand_pose)

    def random_rotation(self, pose):
        """
        Add a random 3D rotation to the hand pose
        :param pose: nx21x3 np array
        :return: nx21x3 np array after rotation
        """
        # Generate a random rotation using quaternions (random axis-angle representation)
        random_rotation = R.random()  # Create a random rotation object using scipy

        # Apply the rotation to each 3D point in the pose
        pose_rotated = random_rotation.apply(pose.reshape(-1, 3)).reshape(pose.shape)
        return pose_rotated

    import numpy as np

    def flip_hand(self, pose, ref_idx=[0, 9]):
        """
        Flip the hand pose along the ref vector.
        :param pose: nx21x3 np array
        :param ref_idx: list of two indices to define the reference vector
        :return: nx21x3 np array after flipping
        """
        # Calculate the reference vector
        ref_vector = pose[:, ref_idx[1], :] - pose[:, ref_idx[0], :]

        # Normalize the reference vector
        ref_vector = ref_vector / np.linalg.norm(ref_vector, axis=1, keepdims=True)

        # Flip each point in the pose by reflecting across the reference vector plane
        pose_flipped = pose.copy()
        for i in range(pose.shape[1]):
            vec_to_flip = pose[:, i, :] - pose[:, ref_idx[0], :]  # Vector from ref point to each point in the pose
            projection_on_ref = np.sum(vec_to_flip * ref_vector, axis=1, keepdims=True) * ref_vector  # Projection onto ref_vector
            reflected_vec = vec_to_flip - 2 * projection_on_ref  # Reflection formula
            pose_flipped[:, i, :] = reflected_vec + pose[:, ref_idx[0], :]  # Update the flipped position

        return pose_flipped

    def calculate_isometric_projection(self, args, kpts_of_interest_name='all', rootIdx=0, canvas_size=1000, ratio_noise=0.05):
        if kpts_of_interest_name == 'all':  # get all points
            kpts_of_interest = self.point_poses.values()
            self.current_kpts_of_interest_name = self.point_poses.keys()
        else:
            kpts_of_interest = [self.point_poses[kpt] for kpt in kpts_of_interest_name]
            self.current_kpts_of_interest_name = kpts_of_interest_name
        self.current_kpts_of_interest = kpts_of_interest
        world3D = Point.batch_export_to_nparray(kpts_of_interest)
        self.pose_3d_world = world3D
        start_frame = 0
        end_frame = self.frame_number
        rgb_frame_rate = 100
        fps_ratio = 100 / rgb_frame_rate
        frames = np.linspace(start_frame / fps_ratio, end_frame / fps_ratio, int((end_frame - start_frame) / fps_ratio), dtype=int)
        self.pose_3d_camera = {}
        self.pose_2d_camera = {}
        self.pose_2d_bbox = {}
        self.pose_depth_px = {}
        self.pose_depth_ratio = {}
        self.pose_3d_image = {}
        cameras = [Camera()]
        cameras[0].DEVICEID = 'XY'
        for cam_idx, camera in enumerate(cameras):
            print(f'Processing camera {cam_idx} - {camera.DEVICEID}')
            points_2d_list = []
            points_3d_camera_list = []
            points_2d_bbox_list = []
            points_depth_px_list = []
            depth_ratio_list = []
            points_3d_image_list = []
            world3D_centered = world3D - world3D[:, [rootIdx], :]  # center around root joint
            # find max and min to fit within canvas
            max_xyz = world3D_centered.max(axis=0).max(axis=0)
            min_xyz = world3D_centered.min(axis=0).min(axis=0)
            max_range = max(max_xyz - min_xyz)
            ratio_base = canvas_size * 0.9 / max_range * 1000 # px/m
            ratio = ratio_base
            print(f'Processing camera {cam_idx} - {camera.DEVICEID} - ratio_base: {ratio_base:.2f}')
            for frame_idx, frame_no in enumerate(frames):
                frame_idx = int(frame_idx * fps_ratio)  # todo: bug if fps_ratio is not an 1
                points_3d = world3D_centered[frame_idx, :, :].reshape(-1, 3)/1000 # m
                if ratio_noise:
                    ratio = ratio_base * np.random.uniform(1 - ratio_noise, 1 + ratio_noise)  # random noise to simulate zoom/moving camera
                points_3d_camera = points_3d # in m, but should be in camera coordinate (assume it is already projected into camera coord, since we performed random rotation before, it does not matter)
                points_3d_image = points_3d_camera * ratio  # in px
                points_2d = points_3d_image[:, :2] + canvas_size / 2  # xy
                points_3d_image[:, :2] = points_2d

                points_depth_px = points_3d_image[:, 2]  # z
                bbox_top_left, bbox_bottom_right = points_2d.min(axis=0) - 20, points_2d.max(axis=0) + 20

                points_2d_list.append(points_2d)
                points_3d_camera_list.append(points_3d_camera)
                points_2d_bbox_list.append([bbox_top_left, bbox_bottom_right])
                points_depth_px_list.append(points_depth_px)
                depth_ratio_list.append(ratio)
                points_3d_image_list.append(points_3d_image)


            self.pose_3d_camera[camera.DEVICEID] = np.array(points_3d_camera_list)
            self.pose_2d_camera[camera.DEVICEID] = np.array(points_2d_list)
            self.pose_2d_bbox[camera.DEVICEID] = np.array(points_2d_bbox_list)
            self.pose_depth_px[camera.DEVICEID] = np.array(points_depth_px_list)
            self.pose_depth_ratio[camera.DEVICEID] = np.array(depth_ratio_list)
            self.pose_3d_image[camera.DEVICEID] = np.array(points_3d_image_list)

        self.cameras = cameras

    def output_MotionBert_pose(self, downsample=5, downsample_keep=1, handiness='left'):
        # append data at end
        output = {}
        joint_2d = []
        confidence = []
        joint_3d_image = []  # px coordinate, px
        joint_3d_camera = []  # camera coordinate, mm
        joint_25d_image = []  # px coordinate, mm
        factor_25d = []  # ratio, ~4.xx, 2.5d_image/3d_image
        camera_name = []
        source = []
        c3d_frame = []
        for this_camera in self.cameras:
            for downsample_idx in range(downsample):
                if downsample_idx != downsample_keep-1:
                    continue
                for frame_idx in range(0, self.frame_number, downsample):
                    real_frame_idx = frame_idx + downsample_idx
                    if real_frame_idx >= self.frame_number:
                        break
                    joint_2d.append(self.pose_2d_camera[this_camera.DEVICEID][real_frame_idx, :, :])
                    confidence.append(np.ones((len(self.current_kpts_of_interest_name), 1)))
                    joint_3d_camera.append(self.pose_3d_camera[this_camera.DEVICEID][real_frame_idx, :, :])

                    joint_3d_image_frame = self.pose_3d_camera[this_camera.DEVICEID][real_frame_idx, :, :].copy()  # need the shape
                    joint_3d_image_frame[:, :2] = self.pose_2d_camera[this_camera.DEVICEID][real_frame_idx, :, :]  # overwrite xy
                    joint_3d_image_frame[:, 2] = self.pose_depth_px[this_camera.DEVICEID][real_frame_idx, :]  # overwrite z

                    joint_3d_image.append(joint_3d_image_frame)
                    factor_25d_frame = self.pose_depth_ratio[this_camera.DEVICEID][real_frame_idx]
                    factor_25d_frame = 1000/factor_25d_frame  # in motionbert 2.5d factor, px * factor = mm
                    factor_25d.append(factor_25d_frame)
                    joint_25d_image_frame = joint_3d_image_frame.copy() * factor_25d_frame
                    joint_25d_image.append(joint_25d_image_frame)

                    camera_name.append(this_camera.DEVICEID)
                    source.append(self.c3d_file + handiness)
                    c3d_frame.append(real_frame_idx)

        output['joint_2d'] = np.array(joint_2d)  # this is gt, but should be detection
        output['confidence'] = np.array(confidence)
        output['joint3d_image'] = np.array(joint_3d_image)  # px coordinate, px
        output['camera_name'] = np.array(camera_name)
        output['source'] = source
        output['2.5d_factor'] = np.array(factor_25d)
        output['joints_2.5d_image'] = np.array(joint_25d_image)  # mm
        output['action'] = [self.c3d_file[-14:-4]] * len(source)

        ################## additional ##################
        # LCN format https://github.com/CHUNYUWANG/lcn-pose/blob/master/tools/gendb.py#L85
        output['joint_3d_camera'] = np.array(joint_3d_camera) * 1000  # convert to mm
        # c3d info
        output['c3d_frame'] = c3d_frame
        return output


# RMCP2,RMCP5, RWRIST
# marker_height = 9  # mm
# R_wrist_plane = Plane()
# R_wrist_plane.set_by_pts(RMCP5, RWRIST, RMCP2)
# RMCP2 = Point.translate_point(RMCP2, R_wrist_plane.normal_vector, marker_height)
# RMCP5 = Point.translate_point(RMCP5, R_wrist_plane.normal_vector, marker_height)
#
# LMCP2,LMCP5, LWRIST
# L_wrist_plane = Plane()
# L_wrist_plane.set_by_pts(LMCP2, LWRIST, LMCP5)
# LMCP2 = Point.translate_point(LMCP2, L_wrist_plane.normal_vector, marker_height)
# LMCP5 = Point.translate_point(LMCP5, L_wrist_plane.normal_vector, marker_height)



