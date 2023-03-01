# a class for vicon skeleton
import c3d
import numpy as np
import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from scipy.io import savemat

class Skeleton:
    def __init__(self):
        pass


class PulginGaitSkeleton(Skeleton):
    '''A class for plugin gait skeleton'''

    def __init__(self, c3d_file):
        super().__init__()
        self.bone_name_mid = ['Pelvis', 'Head', 'Thorax']
        self.bone_name_botL = ['Left Femur', 'Left Tibia', 'Left Foot', 'Left Toe']
        self.bone_name_topL = ['Left Clavicle', 'Left Humerus', 'Left Radius', 'Left Hand']
        self.bone_name_botR = ['Right Femur', 'Right Tibia', 'Right Foot', 'Right Toe']
        self.bone_name_topR = ['Right Clavicle', 'Right Humerus', 'Right Radius', 'Right Hand']
        self.joint_center_name_botL = ['Left Hip Joint Centre', 'Left Knee Joint Centre', 'Left Ankle Joint Centre']
        self.joint_center_name_topL = ['Left Shoulder Joint Centre', 'Left Elbow Joint Centre', 'Left Wrist Joint Centre']
        self.joint_center_name_botR = ['Right Hip Joint Centre', 'Right Knee Joint Centre', 'Right Ankle Joint Centre']
        self.joint_center_name_topR = ['Right Shoulder Joint Centre', 'Right Elbow Joint Centre', 'Right Wrist Joint '
                                                                                                  'Centre']
        self.joint_center_name = self.joint_center_name_botL + self.joint_center_name_topL + self.joint_center_name_botR + self.joint_center_name_topR
        self.com_name = ['Pelvis COM', 'Left Femur COM', 'Left Tibia COM', 'Left Foot COM', 'Right Femur COM',
                         'Right Tibia COM', 'Right Foot COM', 'Thorax COM', 'Head COM',
                         'Left Humerus COM', 'Left Radius COM', 'Left Hand COM', 'Right Humerus COM',
                         'Right Radius COM', 'Right Hand COM', 'Centre Of Mass', 'Centre Of Mass Floor']

        self.bone_parent_mid = [None, 'Thorax', 'Pelvis']
        if False:
            # this is the original one
            self.bone_parent_botL = ['Pelvis', 'Left Femur', 'Left Tibia', 'Left Foot']
            self.bone_parent_topL = ['Thorax', 'Left Clavicle', 'Left Humerus', 'Left Radius']
            self.bone_parent_botR = ['Pelvis', 'Right Femur', 'Right Tibia', 'Right Foot']
            self.bone_parent_topR = ['Thorax', 'Right Clavicle', 'Right Humerus', 'Right Radius']
        else:
            # with joint centers
            self.bone_parent_botL = ['Pelvis', 'Left Femur', 'Left Ankle Joint Centre', 'Left Foot']
            self.bone_parent_topL = ['Thorax', 'Left Clavicle', 'Left Humerus', 'Left Wrist Joint Centre']
            self.bone_parent_botR = ['Pelvis', 'Right Femur', 'Right Ankle Joint Centre', 'Right Foot']
            self.bone_parent_topR = ['Thorax', 'Right Clavicle', 'Right Humerus', 'Right Wrist Joint Centre']
        self.joint_center_parent_botL = ['Pelvis', 'Left Hip Joint Centre', 'Left Knee Joint Centre']
        self.joint_center_parent_topL = ['Thorax', 'Left Shoulder Joint Centre', 'Left Elbow Joint Centre']
        self.joint_center_parent_botR = ['Pelvis', 'Right Hip Joint Centre', 'Right Knee Joint Centre']
        self.joint_center_parent_topR = ['Thorax', 'Right Shoulder Joint Centre', 'Right Elbow Joint Centre']
        self.joint_center_parent = self.joint_center_parent_botL + self.joint_center_parent_topL + self.joint_center_parent_botR + self.joint_center_parent_topR

        # select desired joints
        self.joint_name = self.bone_name_mid + self.joint_center_name + self.bone_name_botL[2:] + self.bone_name_topL[3:] + self.bone_name_botR[2:] + self.bone_name_topR[3:]
        self.joint_parent = self.bone_parent_mid + self.joint_center_parent + self.bone_parent_botL[2:] + self.bone_parent_topL[3:] + self.bone_parent_botR[2:] + self.bone_parent_topR[3:]

        self.joint_number = len(self.joint_name)
        self.lower_bone_acronym = {
            'PELO': 'Pelvis Origin',
            'PELP': 'Pelvis Proximal',
            'PELA': 'Pelvis Anterior',
            'PELL': 'Pelvis Lateral',
            'RFEO': 'Right Femur Origin',
            'RFEP': 'Right Femur Proximal',
            'RFEA': 'Right Femur Anterior',
            'RFEL': 'Right Femur Lateral',
            'LFEO': 'Left Femur Origin',
            'LFEP': 'Left Femur Proximal',
            'LFEA': 'Left Femur Anterior',
            'LFEL': 'Left Femur Lateral',
            'RTIO': 'Right Tibia Origin',
            'RTIP': 'Right Tibia Proximal',
            'RTIA': 'Right Tibia Anterior',
            'RTIL': 'Right Tibia Lateral',
            'LTIO': 'Left Tibia Origin',
            'LTIP': 'Left Tibia Proximal',
            'LTIA': 'Left Tibia Anterior',
            'LTIL': 'Left Tibia Lateral',
            'RFOO': 'Right Foot Origin',
            'RFOP': 'Right Foot Proximal',
            'RFOA': 'Right Foot Anterior',
            'RFOL': 'Right Foot Lateral',
            'LFOO': 'Left Foot Origin',
            'LFOP': 'Left Foot Proximal',
            'LFOA': 'Left Foot Anterior',
            'LFOL': 'Left Foot Lateral',
            'RTOO': 'Right Toe Origin',
            'RTOP': 'Right Toe Proximal',
            'RTOA': 'Right Toe Anterior',
            'RTOL': 'Right Toe Lateral',
            'LTOO': 'Left Toe Origin',
            'LTOP': 'Left Toe Proximal',
            'LTOA': 'Left Toe Anterior',
            'LTOL': 'Left Toe Lateral'
        }
        self.upper_bone_acronym = {
            'HEDO': 'Head Origin',
            'HEDP': 'Head Proximal',
            'HEDA': 'Head Anterior',
            'HEDL': 'Head Lateral',
            'TRXO': 'Thorax Origin',
            'TRXP': 'Thorax Proximal',
            'TRXA': 'Thorax Anterior',
            'TRXL': 'Thorax Lateral',
            'CSPO': 'C Spine Origin',
            'CSPP': 'C Spine Proximal',
            'CSPA': 'C Spine Anterior',
            'CSPL': 'C Spine Lateral',
            'SACO': 'Sacrum Origin',
            'SACP': 'Sacrum Proximal',
            'SACA': 'Sacrum Anterior',
            'SACL': 'Sacrum Lateral',
            'RCLO': 'Right Clavicle Origin',
            'RCLP': 'Right Clavicle Proximal',
            'RCLA': 'Right Clavicle Anterior',
            'RCLL': 'Right Clavicle Lateral',
            'LCLO': 'Left Clavicle Origin',
            'LCLP': 'Left Clavicle Proximal',
            'LCLA': 'Left Clavicle Anterior',
            'LCLL': 'Left Clavicle Lateral',
            'RHUO': 'Right Humerus Origin',
            'RHUP': 'Right Humerus Proximal',
            'RHUA': 'Right Humerus Anterior',
            'RHUL': 'Right Humerus Lateral',
            'LHUO': 'Left Humerus Origin',
            'LHUP': 'Left Humerus Proximal',
            'LHUA': 'Left Humerus Anterior',
            'LHUL': 'Left Humerus Lateral',
            'RRAO': 'Right Radius Origin',
            'RRAP': 'Right Radius Proximal',
            'RRAA': 'Right Radius Anterior',
            'RRAL': 'Right Radius Lateral',
            'LRAO': 'Left Radius Origin',
            'LRAP': 'Left Radius Proximal',
            'LRAA': 'Left Radius Anterior',
            'LRAL': 'Left Radius Lateral',
            'RHNO': 'Right Hand Origin',
            'RHNP': 'Right Hand Proximal',
            'RHNA': 'Right Hand Anterior',
            'RHNL': 'Right Hand Lateral',
            'LHNO': 'Left Hand Origin',
            'LHNP': 'Left Hand Proximal',
            'LHNA': 'Left Hand Anterior',
            'LHNL': 'Left Hand Lateral',
            'RFIO': 'Right Finger Origin',
            'RFIP': 'Right Finger Proximal',
            'RFIA': 'Right Finger Anterior',
            'RFIL': 'Right Finger Lateral',
            'LFIO': 'Left Finger Origin',
            'LFIP': 'Left Finger Proximal',
            'LFIA': 'Left Finger Anterior',
            'LFIL': 'Left Finger Lateral',
            'RTBO': 'Right Thumb Origin',
            'RTBP': 'Right Thumb Proximal',
            'RTBA': 'Right Thumb Anterior',
            'RTBL': 'Right Thumb Lateral',
            'LTBO': 'Left Thumb Origin',
            'LTBP': 'Left Thumb Proximal',
            'LTBA': 'Left Thumb Anterior',
            'LTBL': 'Left Thumb Lateral',
        }
        self.joint_center_acronym = {'LHJC': 'Left Hip Joint Centre',
                                     'RHJC': 'Right Hip Joint Centre',
                                     'LKJC': 'Left Knee Joint Centre',
                                     'RKJC': 'Right Knee Joint Centre',
                                     'LAJC': 'Left Ankle Joint Centre',
                                     'RAJC': 'Right Ankle Joint Centre',
                                     'LSJC': 'Left Shoulder Joint Centre',
                                     'RSJC': 'Right Shoulder Joint Centre',
                                     'LEJC': 'Left Elbow Joint Centre',
                                     'REJC': 'Right Elbow Joint Centre',
                                     'LWJC': 'Left Wrist Joint Centre',
                                     'RWJC': 'Right Wrist Joint Centre'}
        self.com_acronym = {'PelvisCOM': 'Pelvis COM',
                            'LeftFemurCOM': 'Left Femur COM',
                            'LeftTibiaCOM': 'Left Tibia COM',
                            'LeftFootCOM': 'Left Foot COM',
                            'RightFemurCOM': 'Right Femur COM',
                            'RightTibiaCOM': 'Right Tibia COM',
                            'RightFootCOM': 'Right Foot COM',
                            'ThoraxCOM': 'Thorax COM',
                            'HeadCOM': 'Head COM',
                            'LeftHumerusCOM': 'Left Humerus COM',
                            'LeftRadiusCOM': 'Left Radius COM',
                            'LeftHandCOM': 'Left Hand COM',
                            'RightHumerusCOM': 'Right Humerus COM',
                            'RightRadiusCOM': 'Right Radius COM',
                            'RightHandCOM': 'Right Hand COM',
                            'CentreOfMass': 'Centre Of Mass',
                            'CentreOfMassFloor': 'Centre Of Mass Floor'}
        # todo: marker acronym
        self.marker_acronym = {}
        self.joint_acronym = {**self.lower_bone_acronym, **self.upper_bone_acronym, **self.joint_center_acronym}

        self.read_c3d(c3d_file)
        self.c3d_file = c3d_file

    def get_parent(self, joint_name):
        parent_idx = self.joint_name.index(joint_name)
        # catch IndexError: list index out of range and return None
        try:
            return self.joint_parent[parent_idx]
        except IndexError:
            return None

    def acronym(self, joint_name):
        if joint_name in self.joint_center_acronym.values():
            joint_description = joint_name
        else:
            joint_description = joint_name + ' Origin'
        if joint_description in self.joint_acronym.values():
            return list(self.joint_acronym.keys())[list(self.joint_acronym.values()).index(joint_description)]
        else:
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
        '''extract_pt: O, P, A, L, all'''
        pose_idx = {}
        if type(input_name_list) == str:
            input_name_list = [input_name_list]
        else:
            input_name_list = list(input_name_list)
        for joint_wspace in input_name_list:
            joint = joint_wspace.strip()
            if extract_pt in ['O', 'P', 'A', 'L']:
                if joint in self.joint_acronym.keys() and joint[-1] == extract_pt:
                    pose_idx[joint] = self.point_labels.index(joint_wspace)
                elif joint in self.joint_center_acronym.keys():
                    pose_idx[joint] = self.point_labels.index(joint_wspace)
            elif extract_pt == 'all':
                if joint in self.point_labels:
                    pose_idx[joint] = self.point_labels.index(joint_wspace)
        return pose_idx

    def get_pose_idx_from_description(self, input_name_list, extract_pt='O'):
        '''extract_pt: O, all'''
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

    def read_c3d(self, c3d_file):
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
        self.frames = self.points.shape[0]

    def plot_pose_frame(self, frame=0):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for joint_name in self.joint_name:
            joint = self.acronym(joint_name)
            if joint_name in self.bone_name_mid:
                point_type = 's'
                point_size = 15
            else:
                point_type = 'o'
                point_size = 10
            ax.scatter(self.poses[joint][frame, 0],
                       self.poses[joint][frame, 1],
                       self.poses[joint][frame, 2], label=joint_name, marker=point_type,s=point_size)
        # connect points to parent
        for joint_name in self.joint_name:
            joint = self.acronym(joint_name)
            parent_name = self.get_parent(joint_name)
            if parent_name is not None:
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
        if loc_file is not None:
            start_frame = frame_range[0]
            end_frame = frame_range[1]
        else:
            start_frame = 0
            end_frame = self.frames

        loc = np.empty((end_frame-start_frame,117))
        loc[start_frame:end_frame,0:3] = np.nan  # 1 - 3 Top Head Skin Surface
        loc[start_frame:end_frame,3:6] = np.nan  # 4 - 6 L. Head Skin Surface
        loc[start_frame:end_frame,6:9] = np.nan  # 7 - 9 R. Head Skin Surface
        loc[start_frame:end_frame,9:12] = self.get_pose_from_acronym('HEDO', extract_pt='all', output_type='list_last')*2-self.get_pose_from_acronym('HEDP', extract_pt='all', output_type='list_last')  # 10 - 12 Head origin Virtual point
        loc[start_frame:end_frame,12:15] = self.get_pose_from_acronym('HEDO', extract_pt='all', output_type='list_last')  # 13 - 15 Nasion Skin Surface
        loc[start_frame:end_frame,15:18] = self.get_pose_from_acronym('HEDA', extract_pt='all', output_type='list_last')  # 16 - 18 Sight end Virtual point
        loc[start_frame:end_frame,18:21] = self.get_pose_from_acronym('TRXO', extract_pt='all', output_type='list_last')*2-self.get_pose_from_acronym('TRXA', extract_pt='all', output_type='list_last')  # 19 - 21 C7/T1 Joint Center # todo: replace with neck bot (C7/T1)
        loc[start_frame:end_frame,21:24] = self.get_pose_from_acronym('TRXO', extract_pt='all', output_type='list_last')  # 22 - 24 Sternoclavicular Joint Joint Center
        loc[start_frame:end_frame,21:24] = self.get_pose_from_acronym('HEDO', extract_pt='all', output_type='list_last')  # 25 - 27 Suprasternale Skin Surface
        loc[start_frame:end_frame,27:30] = self.get_pose_from_acronym('PELO', extract_pt='all', output_type='list_last')  # 28 - 30 L5/S1 Joint Center
        loc[start_frame:end_frame,30:33] = np.nan  # 31 - 33 PSIS Joint Center
        loc[start_frame:end_frame,33:36] = self.get_pose_from_acronym('LSJC', extract_pt='all', output_type='list_last')  # 34 - 36 L. Shoulder Joint Center
        loc[start_frame:end_frame,36:39] = np.nan  # 37 - 39 L. Acromion Skin Surface
        loc[start_frame:end_frame,39:42] = self.get_pose_from_acronym('LEJC', extract_pt='all', output_type='list_last')  # 40 - 42 L. Elbow Joint Center
        loc[start_frame:end_frame,42:45] = np.nan  # 43 - 45 L. Lat. Epicon. of Humer. Skin Surface
        loc[start_frame:end_frame,45:48] = self.get_pose_from_acronym('LWJC', extract_pt='all', output_type='list_last')  # 46 - 48 L. Wrist Joint Center
        loc[start_frame:end_frame,48:51] = self.get_pose_from_acronym('LHNO', extract_pt='all', output_type='list_last')  # 49 - 51 L. Grip Center Virtual point
        loc[start_frame:end_frame,51:54] = self.get_pose_from_acronym('LFIN', extract_pt='all', output_type='list_last')  # 52 - 54 L. Hand Skin Surface
        loc[start_frame:end_frame,54:57] = self.get_pose_from_acronym('RSJC', extract_pt='all', output_type='list_last')  # 55 - 57 R. Shoulder Joint Center
        loc[start_frame:end_frame,57:60] = np.nan  # 58 - 60 R. Acromion Skin Surface
        loc[start_frame:end_frame,60:63] = self.get_pose_from_acronym('REJC', extract_pt='all', output_type='list_last')  # 61 - 63 R. Elbow Joint Center
        loc[start_frame:end_frame,63:66] = np.nan  # 64 - 66 R. Lat. Epicon. of Humer. Skin Surface
        loc[start_frame:end_frame,66:69] = self.get_pose_from_acronym('RWJC', extract_pt='all', output_type='list_last')  # 67 - 69 R. Wrist Joint Center
        loc[start_frame:end_frame,69:72] = self.get_pose_from_acronym('RHNO', extract_pt='all', output_type='list_last')  # 70 - 72 R. Grip Center Virtual point
        loc[start_frame:end_frame,72:75] = self.get_pose_from_acronym('RFIN', extract_pt='all', output_type='list_last')  # 73 - 75 R. Hand Skin Surface
        loc[start_frame:end_frame,75:78] = self.get_pose_from_acronym('LHJC', extract_pt='all', output_type='list_last')  # 76 - 78 L. Hip Joint Center
        loc[start_frame:end_frame,78:81] = self.get_pose_from_acronym('LKJC', extract_pt='all', output_type='list_last')  # 79 - 81 L. Knee Joint Center
        loc[start_frame:end_frame,81:84] = np.nan  # 82 - 84 L. Lat. Epicon. of Femur Skin Surface
        loc[start_frame:end_frame,84:87] = self.get_pose_from_acronym('LAJC', extract_pt='all', output_type='list_last')  # 85 - 87 L. Ankle Joint Center
        loc[start_frame:end_frame,87:90] = np.nan  # 88 - 90 L. Lateral Malleolus Skin Surface
        loc[start_frame:end_frame,90:93] = self.get_pose_from_acronym('LFOO', extract_pt='all', output_type='list_last')  # 91 - 93 L. Ball of Foot Virtual point
        loc[start_frame:end_frame,93:96] = np.nan  # 94 - 96 L. Metatarsalphalangeal Skin Surface
        loc[start_frame:end_frame,96:99] = self.get_pose_from_acronym('RHJC', extract_pt='all', output_type='list_last')  # 97 - 99 R. Hip Joint Center
        loc[start_frame:end_frame,99:102] = self.get_pose_from_acronym('RKJC', extract_pt='all', output_type='list_last')  # 100 - 102 R. Knee Joint Center
        loc[start_frame:end_frame,102:105] = np.nan  # 103 - 105 R. Lat. Epicon. of Femur Skin Surface
        loc[start_frame:end_frame,105:108] = self.get_pose_from_acronym('RAJC', extract_pt='all', output_type='list_last')  # 106 - 108 R. Ankle Joint Center
        loc[start_frame:end_frame,108:111] = np.nan  # 109 - 111 R. Lateral Malleolus Skin Surface
        loc[start_frame:end_frame,111:114] = self.get_pose_from_acronym('RFOO', extract_pt='all', output_type='list_last')  # 112 - 114 R. Ball of Foot Virtual point
        loc[start_frame:end_frame,114:117] = np.nan  # 115 - 117 R. Metatarsalphalangeal Skin Surface

        if loc_file is None:
            loc_file = self.c3d_file.replace('.c3d', '.loc')

        # write as mat file
        save_mat = {}
        for i in range (loc.shape[0]):
            # frame i
            save_mat['frame_' + str(i)] = loc[i, :]
        savemat(loc_file, save_mat)


        # write to csv file
        # np.savetxt(loc_file, loc, delimiter=',')
        # data = ET.Element('loc')
        # for i in range(loc.shape[0]):
        #     frame = ET.SubElement(data, 'frame')
        #     frame.set('frame', str(i))
        #     # set as one array
        #     frame.text = ' '.join([str(x) for x in loc[i, :]])
        #
        # b_xml = ET.tostring(data)
        # with open(loc_file, "wb") as f:
        #     f.write(b_xml)

        return loc












































