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

class Skeleton:
    def __init__(self):
        pass


class PulginGaitSkeleton(Skeleton):
    """A class for plugin gait skeleton"""
    def __init__(self, c3d_file, skeleton_file='config/Plugingait_info/Plugingait.xml'):
        super().__init__()
        self.skeleton_file = skeleton_file
        self.__load_acronym('config/Plugingait_info/acronym.yaml')
        self.__load_key_joints(skeleton_file)
        self.joint_number = len(self.key_joint_name)
        self.__load_c3d(c3d_file)
        self.c3d_file = c3d_file
        # exclude .c3d
        base_dir = c3d_file[:-4]
        cdf_file = os.path.join(base_dir, 'Pose3D.cdf')
        self.output_cdf(cdf_file)




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
            for joint in pose_idx:
                poses = input_data[:, pose_idx[joint],:3]
        else:
            raise ValueError('output_type must be dict, list or list_last')
        return poses

    def get_pose_from_description(self, input_name_list, extract_pt='all'):
        pose_idx = self.get_pose_idx_from_description(input_name_list, extract_pt=extract_pt)
        return self.get_poses(self.points, pose_idx)

    def get_pose_from_acronym(self, input_name_list, extract_pt='all', output_type='dict'):
        pose_idx = self.get_pose_idx_from_acronym(input_name_list, extract_pt=extract_pt)
        return self.get_poses(self.points, pose_idx, output_type=output_type)

    def __load_c3d(self, c3d_file):
        reader = c3d.Reader(open(c3d_file, 'rb'))
        self.analog_labels = reader.analog_labels
        self.analog_labels = [label.strip() for label in self.analog_labels]  # strip whitespace from analog labels
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

    def plot_pose_frame(self, frame=0):
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
            loc[:,9:12] = (self.get_pose_from_acronym('HDCC', extract_pt='all', output_type='list_last')*4+self.get_pose_from_acronym('NKTP', extract_pt='all', output_type='list_last')*0)/4  # 10 - 12 Head origin Virtual point
            loc[:,12:15] = self.get_pose_from_acronym('HDEY', extract_pt='all', output_type='list_last')  # 13 - 15 Nasion Skin Surface
            loc[:,15:18] = fill  # 16 - 18 Sight end Virtual point
            loc[:,18:21] = (self.get_pose_from_acronym('C7', extract_pt='all', output_type='list_last')*4+self.get_pose_from_acronym('T10', extract_pt='all', output_type='list_last')*0)/4  # 19 - 21 C7/T1 Joint Center
            loc[:,21:24] = (self.get_pose_from_acronym('TRXO', extract_pt='all', output_type='list_last')*2 +self.get_pose_from_acronym('C7', extract_pt='all', output_type='list_last')*2)/4 # 22 - 24 Sternoclavicular Joint Joint Center
            loc[:,24:27] = self.get_pose_from_acronym('CLAV', extract_pt='all', output_type='list_last')  # 25 - 27 Suprasternale Skin Surface
            loc[:,27:30] = (self.get_pose_from_acronym('RPSI', extract_pt='all', output_type='list_last')+self.get_pose_from_acronym('LPSI', extract_pt='all', output_type='list_last')
                             + self.get_pose_from_acronym('RASI', extract_pt='all', output_type='list_last') + self.get_pose_from_acronym('LASI', extract_pt='all', output_type='list_last')
                            )/4  # 28 - 30 L5/S1 Joint Center
            loc[:,30:33] = fill  # 31 - 33 PSIS Joint Center
            loc[:,33:36] = self.get_pose_from_acronym('LSJC', extract_pt='all', output_type='list_last')  # 34 - 36 L. Shoulder Joint Center
            loc[:,36:39] = self.get_pose_from_acronym('LSHO', extract_pt='all', output_type='list_last')  # 37 - 39 L. Acromion Skin Surface
            loc[:,39:42] = self.get_pose_from_acronym('LEJC', extract_pt='all', output_type='list_last')  # 40 - 42 L. Elbow Joint Center
            loc[:,42:45] = fill  # 43 - 45 L. Lat. Epicon. of Humer. Skin Surface
            loc[:,45:48] = self.get_pose_from_acronym('LWJC', extract_pt='all', output_type='list_last')  # 46 - 48 L. Wrist Joint Center
            loc[:,48:51] = self.get_pose_from_acronym('LMFO', extract_pt='all', output_type='list_last')  # 49 - 51 L. Grip Center Virtual point
            loc[:,51:54] = self.get_pose_from_acronym('LFIN', extract_pt='all', output_type='list_last')  # 52 - 54 L. Hand Skin Surface
            loc[:,54:57] = self.get_pose_from_acronym('RSJC', extract_pt='all', output_type='list_last')  # 55 - 57 R. Shoulder Joint Center
            loc[:,57:60] = self.get_pose_from_acronym('RSHO', extract_pt='all', output_type='list_last')  # 58 - 60 R. Acromion Skin Surface
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
                f.write(f'ANT 0 3 1.844 80 #\n')  # male 0, female 1, self-set 3, height  , weight

                f.write(f'LOC {joint_locations} #\n')
                # f.write('HAN 15 -20 85 15 -15 80 #\n')
                # f.write('EXP #\n')
            # f.write('AUT 1 #\n')

        return loc












































