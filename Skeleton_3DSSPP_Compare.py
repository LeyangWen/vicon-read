from Skeleton import *
from ergo3d.camera import *

class Skeleton3D(VEHSErgoSkeleton):
    def output_3DSSPP_loc(self, frame_range=None, loc_file=None):
        """
        # 3DSSPP format from P87 of manual
        17 joint from h36M:
        h36m_joint_names = ['PELVIS', 'RHIP', 'RKNEE', 'RANKLE', 'LHIP', 'LKNEE', 'LANKLE', 'T8', 'THORAX', 'C7', 'HEAD', 'LSHOULDER', 'LELBOW', 'LWRIST', 'RSHOULDER', 'RELBOW', 'RWRIST']

        """
        weight = getattr(self, 'weight', 75)
        height = getattr(self, 'height', 180)
        gender = getattr(self, 'gender', 'male')
        gender_id = 0 if gender == 'male' else 1  # male 0, female 1
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
            # loc[:, 0:3] = self.poses['HDTP']  # 1 - 3 Top Head Skin Surface
            # loc[:, 3:6] = self.poses['LEAR']  # 4 - 6 L. Head Skin Surface
            # loc[:, 6:9] = self.poses['REAR']  # 7 - 9 R. Head Skin Surface
            loc[:, 9:12] = self.point_poses['HEAD'].xyz.T  # 10 - 12 Head origin Virtual point
            head_plane = Plane(self.point_poses['HEAD'], self.point_poses['RSHOULDER'], self.point_poses['LSHOULDER'])
            loc[:, 12:15] = Point.translate_point(self.point_poses['HEAD'], head_plane.normal_vector,
                                                  direction=100).xyz.T  # 13 - 15 Nasion Skin Surface
            # loc[:, 15:18] = Point.translate_point(self.point_poses['HEAD'], head_plane.normal_vector,
            #                                       direction=200).xyz.T  # 16 - 18 Sight end Virtual point
            loc[:, 18:21] = Point.mid_point(self.point_poses['THORAX'], self.point_poses['C7'],
                                            0.85).xyz.T  # 19 - 21 C7/T1 Joint Center
            loc[:, 21:24] = self.point_poses['THORAX'].xyz.T  # 22 - 24 Sternoclavicular Joint Joint Center
            # loc[:, 24:27] = self.point_poses['SS'].xyz.T  # 25 - 27 Suprasternale Skin Surface
            loc[:, 27:30] = Point.mid_point(self.point_poses['T8'], self.point_poses['PELVIS'],
                                            0.5).xyz.T  # 28 - 30 L5/S1 Joint Center
            # loc[:, 30:33] = self.point_poses['PELVIS_b'].xyz.T  # 31 - 33 PSIS Joint Center
            loc[:, 33:36] = self.point_poses['LSHOULDER'].xyz.T  # 34 - 36 L. Shoulder Joint Center
            # loc[:, 36:39] = self.point_poses['LAP'].xyz.T  # 37 - 39 L. Acromion Skin Surface
            loc[:, 39:42] = self.point_poses['LELBOW'].xyz.T  # 40 - 42 L. Elbow Joint Center
            # loc[:, 42:45] = self.point_poses['LLE'].xyz.T  # 43 - 45 L. Lat. Epicon. of Humer. Skin Surface
            loc[:, 45:48] = self.point_poses['LWRIST'].xyz.T  # 46 - 48 L. Wrist Joint Center
            loc[:, 48:51] = self.point_poses['LGRIP'].xyz.T  # 49 - 51 L. Grip Center Virtual point
            loc[:, 51:54] = self.point_poses['LHAND'].xyz.T  # 52 - 54 L. Hand Skin Surface
            loc[:, 54:57] = self.point_poses['RSHOULDER'].xyz.T  # 55 - 57 R. Shoulder Joint Center
            # loc[:, 57:60] = self.point_poses['RAP'].xyz.T  # 58 - 60 R. Acromion Skin Surface
            loc[:, 60:63] = self.point_poses['RELBOW'].xyz.T  # 61 - 63 R. Elbow Joint Center
            # loc[:, 63:66] = self.point_poses['RLE'].xyz.T  # 64 - 66 R. Lat. Epicon. of Humer. Skin Surface
            loc[:, 66:69] = self.point_poses['RWRIST'].xyz.T  # 67 - 69 R. Wrist Joint Center
            loc[:, 69:72] = self.point_poses['RGRIP'].xyz.T  # 70 - 72 R. Grip Center Virtual point
            loc[:, 72:75] = self.point_poses['RHAND'].xyz.T  # 73 - 75 R. Hand Skin Surface
            loc[:, 75:78] = Point.mid_point(self.point_poses['LHIP'], self.point_poses['LKNEE'],
                                            0.7).xyz.T  # 76 - 79 L. Hip Joint Center
            # self.point_poses['LHIP'].xyz.T  # 76 - 79 L. Hip Joint Center
            loc[:, 78:81] = self.point_poses['LKNEE'].xyz.T  # 79 - 81 L. Knee Joint Center
            # loc[:, 81:84] = self.point_poses['LLFC'].xyz.T  # 82 - 84 L. Lat. Epicon. of Femur Skin Surface
            loc[:, 84:87] = self.point_poses['LANKLE'].xyz.T  # 85 - 87 L. Ankle Joint Center
            # loc[:, 87:90] = self.point_poses['LLM'].xyz.T  # 88 - 90 L. Lateral Malleolus Skin Surface
            loc[:, 90:93] = Point.mid_point(self.point_poses['LFOOT'], self.point_poses['LHEEL'],
                                            0.4).xyz.T  # 91 - 93 L. Ball of Foot Virtual point
            # loc[:, 93:96] = self.point_poses['LFOOT'].xyz.T  # 94 - 96 L. Metatarsalphalangeal Skin Surface
            loc[:, 96:99] = Point.mid_point(self.point_poses['RHIP'], self.point_poses['RKNEE'], 0.7).xyz.T
            # self.point_poses['RHIP'].xyz.T  # 97 - 99 R. Hip Joint Center
            loc[:, 99:102] = self.point_poses['RKNEE'].xyz.T  # 100 - 102 R. Knee Joint Center
            # loc[:, 102:105] = self.point_poses['RLFC'].xyz.T  # 103 - 105 R. Lat. Epicon. of Femur Skin Surface
            loc[:, 105:108] = self.point_poses['RANKLE'].xyz.T  # 106 - 108 R. Ankle Joint Center
            # loc[:, 108:111] = self.point_poses['RLM'].xyz.T  # 109 - 111 R. Lateral Malleolus Skin Surface
            loc[:, 111:114] = Point.mid_point(self.point_poses['RFOOT'], self.point_poses['RHEEL'],
                                              0.4).xyz.T  # 112 - 114 R. Ball of Foot Virtual point
            # loc[:, 114:117] = self.point_poses['RFOOT'].xyz.T  # 115 - 117 R. Metatarsalphalangeal Skin Surface
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
            f.write(
                f'ANT {gender_id} 3 {height} {weight} #\n')  # male 0, female 1, self-set-height-weight-values (not population percentile) 3, height  , weight
            f.write('COM Enabling auto output #\n')  # comment
            f.write('AUT 1 #\n')  # auto output when ANT, HAN, JOA, JOI, and PPR commands are called
            for i, k in enumerate(np.arange(start_frame, end_frame, step)):
                joint_locations = np.array2string(loc[k], separator=' ', max_line_width=1000000, precision=3,
                                                  suppress_small=True)[1:-1].replace('0. ', '0 ')
                f.write('FRM ' + str(i) + ' #\n')
                f.write(f'LOC {joint_locations} #\n')
                support_feet_max_height = 0.15  # m
                left_foot_supported = True if self.poses['LFOOT'][
                                                  k, 2] < support_feet_max_height else False  # todo: this only works in world coord
                right_foot_supported = True if self.poses['RFOOT'][k, 2] < support_feet_max_height else False
                if left_foot_supported and right_foot_supported:
                    foot_support_parameter = 0
                elif left_foot_supported and (not right_foot_supported):
                    foot_support_parameter = 1
                elif (not left_foot_supported) and right_foot_supported:
                    foot_support_parameter = 2
                else:
                    foot_support_parameter = 0  # 3
                pelvic_tilt = 0  # pelvic_tilt_angles[k]  # -15
                f.write(
                    f'SUP {foot_support_parameter} 0 0 0 20.0 {pelvic_tilt} #\n')  # set support: Feet Support (0=both, 1=left, 2=right, 3=none), Position (0=stand, 1=seated), Front Seat Pan Support (0=no, 1=yes), Seat Has Back Rest (0=no, 1=yes), and Back Rest Center Height (real number in cm, greater than 19.05).
                hand_load = 0  # N
                f.write(f'HAN {hand_load / 2} -90 0 {hand_load / 2} -90 0 #\n')  # HAN can trigger output write line
            f.write(f'COM Task done #')
        return loc

class Skeleton6D(VEHSErgoSkeleton):
    """
    This is for 6D pose and Vicon GT pose
    """
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
        gender_id = 0 if gender == 'male' else 1  # male 0, female 1
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
            loc[:, 12:15] = Point.translate_point(self.point_poses['HEAD'], head_plane.normal_vector,
                                                  direction=100).xyz.T  # 13 - 15 Nasion Skin Surface
            loc[:, 15:18] = Point.translate_point(self.point_poses['HEAD'], head_plane.normal_vector,
                                                  direction=200).xyz.T  # 16 - 18 Sight end Virtual point
            loc[:, 18:21] = Point.mid_point(self.point_poses['THORAX'], self.point_poses['C7'],
                                            0.85).xyz.T  # 19 - 21 C7/T1 Joint Center
            loc[:, 21:24] = self.point_poses['THORAX'].xyz.T  # 22 - 24 Sternoclavicular Joint Joint Center
            loc[:, 24:27] = self.point_poses['SS'].xyz.T  # 25 - 27 Suprasternale Skin Surface
            loc[:, 27:30] = Point.mid_point(self.point_poses['T8'], self.point_poses['PELVIS_b'],
                                            0.5).xyz.T  # 28 - 30 L5/S1 Joint Center
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
            loc[:, 75:78] = Point.mid_point(self.point_poses['LHIP'], self.point_poses['LKNEE'],
                                            0.7).xyz.T  # 76 - 79 L. Hip Joint Center
            # self.point_poses['LHIP'].xyz.T  # 76 - 79 L. Hip Joint Center
            loc[:, 78:81] = self.point_poses['LKNEE'].xyz.T  # 79 - 81 L. Knee Joint Center
            loc[:, 81:84] = self.point_poses['LLFC'].xyz.T  # 82 - 84 L. Lat. Epicon. of Femur Skin Surface
            loc[:, 84:87] = self.point_poses['LANKLE'].xyz.T  # 85 - 87 L. Ankle Joint Center
            loc[:, 87:90] = self.point_poses['LLM'].xyz.T  # 88 - 90 L. Lateral Malleolus Skin Surface
            loc[:, 90:93] = Point.mid_point(self.point_poses['LFOOT'], self.point_poses['LHEEL'],
                                            0.4).xyz.T  # 91 - 93 L. Ball of Foot Virtual point
            loc[:, 93:96] = self.point_poses['LFOOT'].xyz.T  # 94 - 96 L. Metatarsalphalangeal Skin Surface
            loc[:, 96:99] = Point.mid_point(self.point_poses['RHIP'], self.point_poses['RKNEE'], 0.7).xyz.T
            # self.point_poses['RHIP'].xyz.T  # 97 - 99 R. Hip Joint Center
            loc[:, 99:102] = self.point_poses['RKNEE'].xyz.T  # 100 - 102 R. Knee Joint Center
            loc[:, 102:105] = self.point_poses['RLFC'].xyz.T  # 103 - 105 R. Lat. Epicon. of Femur Skin Surface
            loc[:, 105:108] = self.point_poses['RANKLE'].xyz.T  # 106 - 108 R. Ankle Joint Center
            loc[:, 108:111] = self.point_poses['RLM'].xyz.T  # 109 - 111 R. Lateral Malleolus Skin Surface
            loc[:, 111:114] = Point.mid_point(self.point_poses['RFOOT'], self.point_poses['RHEEL'],
                                              0.4).xyz.T  # 112 - 114 R. Ball of Foot Virtual point
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
            f.write(
                f'ANT {gender_id} 3 {height} {weight} #\n')  # male 0, female 1, self-set-height-weight-values (not population percentile) 3, height  , weight
            f.write('COM Enabling auto output #\n')  # comment
            f.write('AUT 1 #\n')  # auto output when ANT, HAN, JOA, JOI, and PPR commands are called
            for i, k in enumerate(np.arange(start_frame, end_frame, step)):
                joint_locations = np.array2string(loc[k], separator=' ', max_line_width=1000000, precision=3,
                                                  suppress_small=True)[1:-1].replace('0. ', '0 ')
                f.write('FRM ' + str(i) + ' #\n')
                f.write(f'LOC {joint_locations} #\n')
                support_feet_max_height = 0.15  # m
                left_foot_supported = True if self.poses['LFOOT'][
                                                  k, 2] < support_feet_max_height else False  # todo: this only works in world coord
                right_foot_supported = True if self.poses['RFOOT'][k, 2] < support_feet_max_height else False
                if left_foot_supported and right_foot_supported:
                    foot_support_parameter = 0
                elif left_foot_supported and (not right_foot_supported):
                    foot_support_parameter = 1
                elif (not left_foot_supported) and right_foot_supported:
                    foot_support_parameter = 2
                else:
                    foot_support_parameter = 0  # 3
                pelvic_tilt = 0  # pelvic_tilt_angles[k]  # -15
                f.write(
                    f'SUP {foot_support_parameter} 0 0 0 20.0 {pelvic_tilt} #\n')  # set support: Feet Support (0=both, 1=left, 2=right, 3=none), Position (0=stand, 1=seated), Front Seat Pan Support (0=no, 1=yes), Seat Has Back Rest (0=no, 1=yes), and Back Rest Center Height (real number in cm, greater than 19.05).
                hand_load = 0  # N
                f.write(f'HAN {hand_load / 2} -90 0 {hand_load / 2} -90 0 #\n')  # HAN can trigger output write line
            f.write(f'COM Task done #')
        return loc

class SkeletonSMPL(VEHSErgoSkeleton):
    # Use SMPLX repo SMPLPose class instead
    pass
