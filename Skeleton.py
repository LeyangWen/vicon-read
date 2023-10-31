# a class for vicon skeleton
import c3d
import numpy as np
import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from scipy.io import savemat
import yaml
from utility import *
from spacepy import pycdf
import cv2
from Point import *
from Camera import *

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
        for i in range(self.point_number):
            self.poses[self.point_labels[i]] = pt_np[:, i, :]

    def load_name_list_and_np_points(self, name_list, pt_np):
        self.load_name_list(name_list)
        self.load_np_points(pt_np)

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
            except yaml.YAMLError as exc:
                print(filename, exc)
                raise exc

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

    def get_plot_property(self, joint_name):
        '''
        return point_type and point_size for plot
        '''
        if joint_name in self.joint_name_mid:
            point_type = 's'
            point_size = 10
        elif joint_name in self.joint_name_botL or joint_name in self.joint_name_topL:
            point_type = '<'
            point_size = 4
        elif joint_name in self.joint_name_botR or joint_name in self.joint_name_topR:
            point_type = '>'
            point_size = 4
        else:
            point_type = 'o'
            point_size = 4
        return point_type, point_size

    def plot_3d_pose_frame(self, frame=0, filename=False):
        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        for joint_name in self.key_joint_name:
            point_type, point_size = self.get_plot_property(joint_name)
            ax.scatter(self.poses[joint_name][frame, 0],
                       self.poses[joint_name][frame, 1],
                       self.poses[joint_name][frame, 2], label=joint_name, marker=point_type, s=point_size)
        # connect points to parent
        for joint_name in self.key_joint_name:
            parent_name = self.get_parent(joint_name)
            if parent_name is not None and parent_name != 'None':
                ax.plot([self.poses[joint_name][frame, 0], self.poses[parent_name][frame, 0]],
                        [self.poses[joint_name][frame, 1], self.poses[parent_name][frame, 1]],
                        [self.poses[joint_name][frame, 2], self.poses[parent_name][frame, 2]], 'k-')

        # ax.legend(bbox_to_anchor=(0.95, 1), loc=2, borderaxespad=0.)
        # uniform scale based on pelvis location and 1800mm
        range = 1800
        pelvis_loc = self.poses['PELVIS'][frame, :]
        ax.set_xlim(pelvis_loc[0] - range / 2, pelvis_loc[0] + range / 2)
        ax.set_ylim(pelvis_loc[1] - range / 2, pelvis_loc[1] + range / 2)
        ax.set_zlim(pelvis_loc[2] - range / 2, pelvis_loc[2] + range / 2)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        fig.tight_layout()
        if True:
            fig.subplots_adjust(right=0.65)
            ax.legend(loc='center left', bbox_to_anchor=(1.08, 0.5), fontsize=7, ncol=2)
        else:  # use this to get a legend screenshot
            ax.legend(loc='upper center', fontsize=7, ncol=5)
            plt.gca().set_axis_off()
            plt.savefig(r'C:\Users\Public\Documents\Vicon\data\Vicon_F\Round3\LeyangWen\FullCollection\render\frame_output\legend.png', dpi=250)
            raise NameError  # break here
        if filename:
            plt.savefig(filename, dpi=250)
            plt.close(fig)
            return None
        else:
            plt.show()
            return fig, ax

    def plot_2d_pose_frame(self, frame=0, baseimage=False, filename=False):
        if baseimage:
            raise NotImplementedError
        else:  # return a transparent image
            img_width = 1920
            img_height = 1200
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for joint_name in self.key_joint_name:
                point_type, point_size = self.get_plot_property(joint_name)
                ax.scatter(self.poses[joint_name][frame, 0],
                           self.poses[joint_name][frame, 1], label=joint_name, marker=point_type, s=point_size, zorder=2)
            # connect points to parent
            for joint_name in self.key_joint_name:
                parent_name = self.get_parent(joint_name)
                if parent_name is not None and parent_name != 'None':
                    ax.plot([self.poses[joint_name][frame, 0], self.poses[parent_name][frame, 0]],
                            [self.poses[joint_name][frame, 1], self.poses[parent_name][frame, 1]], 'k-', zorder=1)
            ax.set_xlim(0, img_width)
            ax.set_ylim(0, img_height)
            ax.set_aspect('equal', adjustable='box')
            ax.invert_yaxis()  # flip y axis
            fig.tight_layout()
            ax.set_axis_off()
            if filename:
                plt.savefig(filename, dpi=300, transparent=True, bbox_inches='tight')
                plt.close(fig)
                return None
            else:
                plt.show()
                return fig, ax

    def plot_3d_pose(self, foldername=False):
        if foldername:
            create_dir(foldername)
        for i in range(self.frame_number):
            print(f'plotting frame {i}/{self.frame_number} in {foldername}...', end='\r')
            filename = foldername if not foldername else os.path.join(foldername, f'{i:05d}.png')
            self.plot_3d_pose_frame(frame=i, filename=filename)

    def plot_2d_pose(self, foldername=False):
        if foldername:
            create_dir(foldername)
        for i in range(self.frame_number):
            print(f'plotting frame {i}/{self.frame_number} in {foldername}...', end='\r')
            filename = foldername if not foldername else os.path.join(foldername, f'{i:05d}.png')
            self.plot_2d_pose_frame(frame=i, filename=filename)



class VEHSErgoSkeleton(Skeleton):
    def __init__(self, skeleton_file):
        super().__init__(skeleton_file)
        self.marker_height = 14/2+2  # 14mm marker
        # marker_height = 9.5/2+2  # 9.5mm marker

    def calculate_joint_center(self):
        self.point_poses['HEAD'] = Point.mid_point(self.point_poses['LEAR'], self.point_poses['REAR'])
        self.point_poses['RSHOULDER'] = Point.mid_point(self.point_poses['RAP_f'], self.point_poses['RAP_b'])
        self.point_poses['LSHOULDER'] = Point.mid_point(self.point_poses['LAP_f'], self.point_poses['LAP_b'])
        self.point_poses['C7_m'] = Point.mid_point(self.point_poses['C7_d'], self.point_poses['SS'])
        self.point_poses['THORAX'] = Point.translate_point(self.point_poses['SS'], Point.vector(self.point_poses['SS'], self.point_poses['C7_m'], normalize=self.marker_height))  # offset by marker height
        self.point_poses['LELBOW'] = Point.mid_point(self.point_poses['LME'], self.point_poses['LLE'])
        self.point_poses['RELBOW'] = Point.mid_point(self.point_poses['RME'], self.point_poses['RLE'])
        self.point_poses['RWRIST'] = Point.mid_point(self.point_poses['RRS'], self.point_poses['RUS'])
        self.point_poses['LWRIST'] = Point.mid_point(self.point_poses['LRS'], self.point_poses['LUS'])
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

        self.update_pose_from_point_pose()

    def calculate_camera_projection(self, camera_xcp_file, kpts_of_interest_name='all'):
        # todo: currently real world only, pelvis center option
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
        self.cameras = cameras
        self.pose_3d_camera = {}
        self.pose_2d_camera = {}
        for cam_idx, camera in enumerate(cameras):
            print(f'Processing camera {cam_idx}: {camera.DEVICEID}')

            points_2d_list = []
            points_3d_camera_list = []
            points_2d_bbox_list = []
            for frame_idx, frame_no in enumerate(frames):
                frame_idx = int(frame_idx * fps_ratio)  # todo: bug if fps_ratio is not an 1
                print(f'Processing frame {frame_no}/{frames[-1]} of {self.c3d_file}',
                      end='\r')
                points_3d = world3D[frame_idx, :, :].reshape(-1, 3) / 1000
                points_3d_camera = camera.project_w_depth(points_3d)
                points_2d = camera.project(points_3d)
                points_2d = camera.distort(points_2d)
                bbox_top_left, bbox_bottom_right = points_2d.min(axis=0) - 20, points_2d.max(axis=0) + 20
                points_2d_list.append(points_2d)
                points_3d_camera_list.append(points_3d_camera)
                points_2d_bbox_list.append([bbox_top_left, bbox_bottom_right])

            points_2d_list = np.array(points_2d_list)
            points_3d_camera_list = np.swapaxes(np.array(points_3d_camera_list), 1, 2)
            points_2d_bbox_list = np.array(points_2d_bbox_list)
            self.pose_3d_camera[camera.DEVICEID] = points_3d_camera_list
            self.pose_2d_camera[camera.DEVICEID] = points_2d_list

    def output_MotionBert_SMPL(self):
        '''
        MotionBert Style
        '''
        pass

    def output_MotionBert_pose(self, downsample=5, downsample_keep=1):
        # append data at end
        output = {}
        joint_2d = []
        confidence = []
        joint3d_image = []
        camera_name = []
        source = []
        c3d_frame = []
        for this_camera in self.cameras:
            for downsample_idx in range(downsample):
                if downsample_idx == downsample_keep:
                    break
                for frame_idx in range(0, self.frame_number, downsample):
                    real_frame_idx = frame_idx + downsample_idx
                    if real_frame_idx >= self.frame_number:
                        break
                    joint_2d.append(self.pose_2d_camera[this_camera.DEVICEID][real_frame_idx, :, :])
                    confidence.append(np.ones((len(self.current_kpts_of_interest_name), 1)))
                    joint3d_image.append(self.pose_3d_camera[this_camera.DEVICEID][real_frame_idx, :, :])
                    camera_name.append(this_camera.DEVICEID)
                    source.append(self.c3d_file)
                    c3d_frame.append(real_frame_idx)

        output['joint_2d'] = np.array(joint_2d)
        output['confidence'] = np.array(confidence)
        output['joint3d_image'] = np.array(joint3d_image)*1000  # convert to mm
        output['camera_name'] = np.array(camera_name)
        output['source'] = source
        output['c3d_frame'] = c3d_frame
        output['2.5d_factor'] = np.ones((output['joint_2d'].shape[0],))  # set to 1 for now, it seems to only affect motionbert eval
        output['joints_2.5d_image'] = output['joint3d_image']  # * output['2.5d_factor']  # * or /?, figure it out from example pkl file if we change 2.5d_factor
        output['action'] = [self.c3d_file[-14:-4]] * len(source)
        return output

    def output_3DSSPP_loc(self, frame_range=None, loc_file=None):
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
            loc[:, 27:30] = self.point_poses['PELVIS'].xyz.T  # 28 - 30 L5/S1 Joint Center
                #Point.mid_point(self.point_poses['PELVIS'], Point.mid_point(self.point_poses['RHIP'], self.point_poses['LHIP'], 0.5), 0.5).xyz.T  # 28 - 30 L5/S1 Joint Center
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
            loc[:, 75:78] = self.point_poses['LHIP'].xyz.T  # 76 - 79 L. Hip Joint Center
            loc[:, 78:81] = self.point_poses['LKNEE'].xyz.T  # 79 - 81 L. Knee Joint Center
            loc[:, 81:84] = self.point_poses['LLFC'].xyz.T  # 82 - 84 L. Lat. Epicon. of Femur Skin Surface
            loc[:, 84:87] = self.point_poses['LANKLE'].xyz.T  # 85 - 87 L. Ankle Joint Center
            loc[:, 87:90] = self.point_poses['LLM'].xyz.T  # 88 - 90 L. Lateral Malleolus Skin Surface
            loc[:, 90:93] = Point.mid_point(self.point_poses['LFOOT'], self.point_poses['LHEEL'], 0.4).xyz.T  # 91 - 93 L. Ball of Foot Virtual point
            loc[:, 93:96] = self.point_poses['LFOOT'].xyz.T  # 94 - 96 L. Metatarsalphalangeal Skin Surface
            loc[:, 96:99] = self.point_poses['RHIP'].xyz.T  # 97 - 99 R. Hip Joint Center
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
        # write as txt file
        with open(loc_file, 'w') as f:
            f.write('3DSSPPBATCHFILE #\n')
            f.write('COM #\n')
            f.write('DES 1 "Task Name" "Analyst Name" "Comments" "Company" #\n')  # English is 0 and metric is 1
            for i, k in enumerate(np.arange(start_frame, end_frame, step)):
                joint_locations = np.array2string(loc[k], separator=' ', max_line_width=1000000, precision=3, suppress_small=True)[1:-1].replace('0. ', '0 ')
                f.write('FRM ' + str(i) + ' #\n')
                f.write(f'ANT 0 3 {height} {weight} #\n')  # male 0, female 1, self-set-height-weight-values (not population percentile) 3, height  , weight
                support_feet_max_height = 0.15
                left_foot_supported = True if self.poses['LFOOT'][k, 2] < support_feet_max_height else False
                right_foot_supported = True if self.poses['RFOOT'][k, 2] < support_feet_max_height else False
                if left_foot_supported and right_foot_supported:
                    foot_support_parameter = 0
                elif left_foot_supported and (not right_foot_supported):
                    foot_support_parameter = 1
                elif (not left_foot_supported) and right_foot_supported:
                    foot_support_parameter = 2
                else:
                    foot_support_parameter = 3
                f.write(f'SUP {foot_support_parameter} 0 0 0 20.0 –15 #\n')  # set support: Feet Support (0=both, 1=left, 2=right, 3=none), Position (0=stand, 1=seated), Front Seat Pan Support (0=no, 1=yes), Seat Has Back Rest (0=no, 1=yes), and Back Rest Center Height (real number in cm, greater than 19.05).
                f.write(f'LOC {joint_locations} #\n')
                # f.write('HAN 15 -20 85 15 -15 80 #\n')
                # f.write('EXP #\n')
                f.write('AUT 1 #\n')
            # f.write('OUT #\n')

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












































