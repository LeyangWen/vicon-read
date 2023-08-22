import xml.etree.ElementTree as ET
import os
import numpy as np
import cv2
import pandas as pd
import json

def batch_load_from_xcp(xcp_filename):
    # Read the xcp file as xml
    with open(xcp_filename, 'r') as f:
        xml_string = f.read()
    root = ET.fromstring(xml_string)
    cameras = []
    for child in root:
        if child.attrib['DISPLAY_TYPE'] == 'VideoInputDevice:Blackfly S BFS-U3-23S3C':
            camera = FLIR_Camera()
            camera.load_vicon_xcp(child)
            cameras.append(camera)
    return cameras


def load_csv(csv_filename):
    df = pd.read_csv(csv_filename, skiprows=[0, 1, 2,3, 4], header=None)
    header_df = pd.read_csv(csv_filename, skiprows=[0, 1], nrows=1, header=None)
    # remove nan in the header
    # iterate over header_df
    header = ['Frame', 'Sub Frame']
    axis = ['X', 'Y', 'Z']
    joint_names = []
    for x in header_df.iloc[0]:
        # print(x)
        if type(x) == str:
            joint_name = x.split(':')[-1]
            joint_names.append(joint_name)
            for ax in axis:
                header.append(f'{joint_name}-{ax}')
    # add header to df
    df.columns = header
    return df, joint_names


class Camera():
    def __int__(self):
        pass

    @staticmethod
    def string_to_float(string, delimiter=' ', suppress_warning=False):
        output = []
        for x in string.split(delimiter):
            try:
                output.append(float(x))
            except ValueError:
                if not suppress_warning:
                    print(f'Warning: \'{x}\' in string \'{string} \'is not a float.')
        if len(output) == 1:
            return output[0]
        else:
            return np.array(output)

    @staticmethod
    def rot3x3_from_quaternion(quaternion):
        # quaternion = [w, x, y, z]
        x, y, z, w = quaternion
        rot_matrix = np.array([[1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                               [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w],
                               [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2]])
        return rot_matrix


    @staticmethod
    def rot3x4_from_rot3x3_and_position(rot3x3, position):
        rot3x4 = np.hstack([rot3x3, -rot3x3.dot(position.reshape(-1, 1))])
        return rot3x4

    def get_camera_intrinsic_matrix(self):
        K = np.array([[self.FOCAL_LENGTH, self.SKEW, self.PRINCIPAL_POINT[0]],
                      [0, self.FOCAL_LENGTH / self.PIXEL_ASPECT_RATIO, self.PRINCIPAL_POINT[1]],
                      [0, 0, 1]])
        return K

    @staticmethod
    def projection_matrix_from_intrinsic_matrix_and_rot3x4(K, rot3x4):
        P = K.dot(rot3x4)
        return P

    def draw_2d(self, image, points_2d, color=(0, 0, 255)):
        for point_idx, point_2d in enumerate(points_2d):
            cv2.circle(image, (int(point_2d[0]), int(point_2d[1])), 3, color, -1)
            cv2.putText(image, str(point_idx), (int(point_2d[0]), int(point_2d[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return image


class FLIR_Camera(Camera):
    def __init__(self):
        super().__init__()

    def load_vicon_xcp(self, xcp_element):
        # Read the xcp file as xml
        self.DEVICEID = xcp_element.attrib['DEVICEID']
        self.SENSOR_SIZE = self.string_to_float(xcp_element.attrib['SENSOR_SIZE'])
        self.DISPLAY_TYPE = xcp_element.attrib['DISPLAY_TYPE']
        self.USERID = xcp_element.attrib['USERID']
        self.PIXEL_ASPECT_RATIO = float(xcp_element.attrib['PIXEL_ASPECT_RATIO'])
        for grandchild in xcp_element:
            if grandchild.tag == 'ThresholdGrid':
                pass
            elif grandchild.tag == 'Calibration':
                pass
            elif grandchild.tag == 'IntrinsicsCalibration':
                pass
            elif grandchild.tag == 'Capture':
                self.CAPTURE_START_TIME = grandchild.attrib['START_TIME']
                self.CAPTURE_END_TIME = grandchild.attrib['END_TIME']
            elif grandchild.tag == 'KeyFrames':
                for grandchild2 in grandchild:
                    self.FOCAL_LENGTH = self.string_to_float(grandchild2.attrib['FOCAL_LENGTH'])
                    self.IMAGE_ERROR = self.string_to_float(grandchild2.attrib['IMAGE_ERROR'])
                    self.ORIENTATION = self.string_to_float(grandchild2.attrib['ORIENTATION'])
                    self.POSITION = self.string_to_float(grandchild2.attrib['POSITION'])/1000
                    self.PRINCIPAL_POINT = self.string_to_float(grandchild2.attrib['PRINCIPAL_POINT'])
                    self.VICON_RADIAL2 = self.string_to_float(grandchild2.attrib['VICON_RADIAL2'], suppress_warning=True)
                    self.DISTORTION_CENTER = self.VICON_RADIAL2[0:2]
                    self.DISTORTION_SCALE = np.hstack([np.array([1]), self.VICON_RADIAL2[2:]])
                    self.WORLD_ERROR = self.string_to_float(grandchild2.attrib['WORLD_ERROR'])
                    self.frame = self.string_to_float(grandchild2.attrib['FRAME'])
                    break
                if len(grandchild) > 1:
                    # raise warning
                    print('Warning: more than one keyframe found, is this a moving camera? Only first keyframe will be used.')
                    # todo: accommodate moving camera in the future
        self.SKEW = 0  # not exist in xcp file, assuming no skew

    def get_matrix(self):
        pass

    def undistort(self, points_2d):
        # undistort radial distortion, formula provided by Vicon
        points_2d = np.array(points_2d)
        dp = (points_2d - self.DISTORTION_CENTER) * (np.array([1, self.PIXEL_ASPECT_RATIO]))
        radius = np.linalg.norm(dp, axis=1)
        scale = np.vstack([np.ones_like(radius), radius ** 2, radius ** 4, radius ** 6]).T.dot(self.DISTORTION_SCALE).reshape(-1, 1)
        point_2d_corrected = (scale*dp + self.DISTORTION_CENTER)*(np.array([1, 1/self.PIXEL_ASPECT_RATIO]))  # assuming distortion center is the same as principal point
        return point_2d_corrected

    def distort(self, points_2d):
        # approximate distort radial distortion, scale calculation should use distorted points, but here we use undistorted points
        points_2d = np.array(points_2d)
        dp = points_2d * (np.array([1, self.PIXEL_ASPECT_RATIO])) - self.DISTORTION_CENTER
        radius = np.linalg.norm(dp, axis=1)
        scale = np.vstack([np.ones_like(radius), radius ** 2, radius ** 4, radius ** 6]).T.dot(self.DISTORTION_SCALE).reshape(-1, 1)
        point_2d_distorted = dp/scale*(np.array([1, 1/self.PIXEL_ASPECT_RATIO])) + self.DISTORTION_CENTER # assuming distortion center is the same as principal point
        return point_2d_distorted

    def get_projection_matrix(self):
        self.rot3x3 = self.rot3x3_from_quaternion(self.ORIENTATION)
        self.rot3x4 = self.rot3x4_from_rot3x3_and_position(self.rot3x3, self.POSITION)
        self.intrinsic_matrix = self.get_camera_intrinsic_matrix()
        self.projection_matrix = self.projection_matrix_from_intrinsic_matrix_and_rot3x4(self.intrinsic_matrix, self.rot3x4)
        return self.projection_matrix

    def project(self, points_3d):
        '''
        :param points_3d: 3d points in world coordinate
        :return: 2d points in camera coordinate, in pixel
        '''
        self.get_projection_matrix()
        points_3d = np.array(points_3d)
        points_2d = self.projection_matrix.dot(np.vstack([points_3d.T, np.ones([1, points_3d.shape[0]])]))
        points_2d = points_2d[0:2, :] / points_2d[2, :]
        points_2d = points_2d.T
        return points_2d

    def project_w_depth(self, points_3d):  # todo: test
        '''
        :param points_3d: 3d points in world coordinate
        :return: 3d points in camera coordinate, with depth, in mm/m
        '''
        self.get_projection_matrix()
        points_3d = np.array(points_3d)
        points_3d_cam = self.rot3x4.dot(np.vstack([points_3d.T, np.ones([1, points_3d.shape[0]])]))
        # translation_vector = self.POSITION.reshape([3, 1])
        # rotM = self.rot3x3
        # points_3d_cam = (rotM.dot(points_3d.T) + translation_vector).T
        return points_3d_cam

    @staticmethod
    def plugingait_to_H36M_converter(points_2d, frame_no, joint_names):
#         [{"image_id": "0.jpg",
#         "category_id": 1,
#         "keypoints": [371.8258361816406, 184.70167541503906, 0.9479594826698303, 383.28143310546875, 173.24606323242188, 0.9629446268081665, 366.0980224609375, 173.24606323242188, 0.9532919526100159, 411.92047119140625, 178.97386169433594, 0.9926742315292358, 360.3702087402344, 178.97386169433594, 0.8256455659866333, 440.55950927734375, 241.979736328125, 0.9333735704421997, 348.91461181640625, 241.979736328125, 0.9294862151145935, 474.92633056640625, 322.1690368652344, 0.9467807412147522, 343.1867980957031, 327.8968200683594, 0.9887729287147522, 434.8316955566406, 293.5299987792969, 0.9161380529403687, 331.7311706542969, 304.985595703125, 0.931166410446167, 423.3760681152344, 385.1748962402344, 0.9257044196128845, 371.8258361816406, 385.1748962402344, 0.8760479688644409, 434.8316955566406, 528.3700561523438, 0.8742569088935852, 348.91461181640625, 528.3700561523438, 0.9642834067344666, 469.1985168457031, 631.4705810546875, 0.901423990726471, 360.3702087402344, 637.1983642578125, 0.9199998378753662, 383.28143310546875, 144.60702514648438, 0.9656839966773987, 389.0092468261719, 219.06851196289062, 0.9374449849128723, 394.737060546875, 379.44708251953125, 0.9205844402313232, 452.0151062011719, 683.0208129882812, 0.9289825558662415, 348.91461181640625, 683.0208129882812, 0.9448415040969849, 469.1985168457031, 683.0208129882812, 0.9240862727165222, 337.458984375, 677.2930297851562, 0.9301682114601135, 474.92633056640625, 642.9262084960938, 0.8143528699874878, 371.8258361816406, 648.6539916992188, 0.8643205165863037],
#         "score": 3.155129909515381,
#         "box": [304.7809753417969, 129.14193725585938, 185.63992309570312, 586.5273742675781],
#         "idx": [0.0]},]
        # 17 joints of Human3.6M:
        # 'root', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'torso', 'neck', 'nose', 'head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist'
        # 17 joints of Plugingait model output:
        # BHEC, CentreOfMass, CentreOfMassFloor, FHEC, HDCC, HECC, HTPO, HeadCOM, LAJC, LEJC, LEYE, LHEC, LHJC, LKJC, LMFO, LPKO, LSJC, LWJC, LeftFemurCOM, LeftFootCOM, LeftHandCOM, LeftHumerusCOM, LeftRadiusCOM, LeftTibiaCOM, NKTP, PelvisCOM, RAJC, REJC, REYE, RHEC, RHJC, RKJC, RMFO, RPKO, RSJC, RWJC, RightFemurCOM, RightFootCOM, RightHandCOM, RightHumerusCOM, RightRadiusCOM, RightTibiaCOM, ThoraxCOM
        # w. index: 0. BHEC, 1. CentreOfMass, 2. CentreOfMassFloor, 3. FHEC, 4. HDCC, 5. HECC, 6. HTPO, 7. HeadCOM, 8. LAJC, 9. LEJC, 10. LEYE, 11. LHEC, 12. LHJC, 13. LKJC, 14. LMFO, 15. LPKO, 16. LSJC, 17. LWJC, 18. LeftFemurCOM, 19. LeftFootCOM, 20. LeftHandCOM, 21. LeftHumerusCOM, 22. LeftRadiusCOM, 23. LeftTibiaCOM, 24. NKTP, 25. PelvisCOM, 26. RAJC, 27. REJC, 28. REYE, 29. RHEC, 30. RHJC, 31. RKJC, 32. RMFO, 33. RPKO, 34. RSJC, 35. RWJC, 36. RightFemurCOM, 37. RightFootCOM, 38. RightHandCOM, 39. RightHumerusCOM, 40. RightRadiusCOM, 41. RightTibiaCOM, 42. ThoraxCOM
        H36M_frame = {}
        H36M_frame['image_id'] = f'{frame_no}.jpg'
        H36M_frame['category_id'] = 1
        H36M_frame['keypoints'] = []
        H36M_frame['score'] = 3.0
        H36M_frame['box'] = []
        H36M_frame['idx'] = [0.0]
        # selected plugingait joints in sequence:
        # PelvisCOM, RHJC, RKJC, RAJC, LHJC, LKJC, LACJ, 0.5*(HeadCOM+ThoraxCOM), ThoraxCOM, 0.5*(LEYE+REYE), HeadCOM, LSJC, LEJC, LWJC, RSJC, REJC, RWJC
        plugingait_joints = ['PelvisCOM', 'RHJC', 'RKJC', 'RAJC', 'LHJC', 'LKJC', 'LAJC', 'temp', 'ThoraxCOM', 'temp', 'HeadCOM', 'LSJC', 'LEJC', 'LWJC', 'RSJC', 'REJC', 'RWJC']
        keypoints = np.ones((17, 3))
        for i, joint_name in enumerate(plugingait_joints):
            if joint_name == 'temp':
                continue
            keypoints[i, :2] = points_2d[joint_names.index(joint_name)]
        keypoints[7, :2]    = 0.5*(points_2d[joint_names.index('HeadCOM')] + points_2d[joint_names.index('ThoraxCOM')])
        keypoints[9, :2]    = 0.5*(points_2d[joint_names.index('LEYE')] + points_2d[joint_names.index('REYE')])
        H36M_frame['keypoints'] = keypoints.reshape(-1).tolist()
        # find a bbox containing all keypoints
        H36M_frame['box'] = [np.min(keypoints[:, 0]), np.min(keypoints[:, 1]), np.max(keypoints[:, 0])-np.min(keypoints[:, 0]), np.max(keypoints[:, 1])-np.min(keypoints[:, 1])]
        return H36M_frame


if __name__ == '__main__':
    example_case = 2
    if example_case == 1:  # visualize rgb with overlay 2d skeleton
        base_dir_name = r'C:\Users\Public\Documents\Vicon\data\VEHS_ske\Round3\Leyang Wen\Subject 3-1'
        activity_no = 1
        frames = np.linspace(0, 6000, int(6000/5+1), dtype=int)
        time = '20230427162751'
        activity_name = f'Activity {activity_no:02d}'
        basename = os.path.join(base_dir_name, activity_name)
        output_dir = 'output/frames'
        c3d_filename = basename + '.c3d'
        xcp_filename = basename + '.xcp'
        csv_filename = basename + '.trajectory.csv'
        df, joint_names = load_csv(csv_filename)

        cameras = batch_load_from_xcp(xcp_filename)
        camera = cameras[1]
        for cam_idx, camera in enumerate(cameras):
            if cam_idx == 2:
                continue
            video_filename = f'{basename}.{camera.DEVICEID}.{time}.avi'
            # get first frame of video
            for frame_no in frames:
                print(f'Processing frame {frame_no}/{frames[-1]} of {activity_name}.{camera.DEVICEID}.{time}.avi', end='\r')
                points_3d = df.iloc[frame_no, 2:].values.reshape(-1, 3) / 1000
                cap = cv2.VideoCapture(video_filename)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no-1)
                ret, frame = cap.read()
                points_2d = camera.project(points_3d)
                # frame = camera.draw_2d(frame, points_2d)
                points_2d = camera.distort(points_2d)
                # points_2d = camera.undistort(points_2d)
                frame = camera.draw_2d(frame, points_2d, color=(0, 255, 0))
                # cv2.imshow('frame', frame)
                # cv2.waitKey(0)
                dir_name = os.path.join(output_dir, camera.DEVICEID)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                output_file = os.path.join(dir_name, f'{activity_name}.{camera.DEVICEID}.{time}.frame{frame_no:06d}.jpg')
                cv2.imwrite(output_file, frame)

    if example_case == 2:  # output H36M json file for motionBert
        base_dir_name = r'C:\Users\Public\Documents\Vicon\vicon_coding_projects\MotionBERT\experiment\Vicon_Gunwoo_Test_movement02'
        activity_no = 2
        time = '20230224152348'  #'20230224152907'
        activity_name = f'Gunwoo movement {activity_no:02d}'
        basename = os.path.join(base_dir_name, activity_name)
        output_dir = f'{base_dir_name}'
        sep_frames = False
        xcp_filename = basename + '.xcp'
        csv_filename = basename + '.joint_loc.csv'
        df, joint_names = load_csv(csv_filename)
        fps = 50
        start_frame = 0
        end_frame = 7497
        fps_ratio = 100/fps
        frames = np.linspace(start_frame/fps_ratio, end_frame/fps_ratio, int((end_frame-start_frame)/fps_ratio), dtype=int)
        cameras = batch_load_from_xcp(xcp_filename)
        camera = cameras[1]
        for cam_idx, camera in enumerate(cameras):
            print(f'Processing camera {camera.DEVICEID}')
            if camera.DEVICEID == '66920731' or camera.DEVICEID == '66920758':
                pass
            else:
                continue
            json_list = []
            video_filename = f'{basename}.{camera.DEVICEID}.{time}.avi'
            for frame_idx, frame_no in enumerate(frames):
                frame_idx = int(frame_idx * fps_ratio)
                print(f'Processing frame {frame_no}/{frames[-1]} of {activity_name}.{camera.DEVICEID}.{time}.avi',
                      end='\r')
                points_3d = df.iloc[frame_idx, 2:].values.reshape(-1, 3) / 1000
                points_2d = camera.project(points_3d)
                points_2d = camera.distort(points_2d)
                json_line = camera.plugingait_to_H36M_converter(points_2d, frame_idx, joint_names)
                json_list.append(json_line)
                camera_points_2d = np.array(json_line['keypoints']).reshape(-1, 3)[:, :2]
                if sep_frames:
                    cap = cv2.VideoCapture(video_filename)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
                    ret, frame = cap.read()
                    if True:
                        frame = camera.draw_2d(frame, camera_points_2d, color=(0, 255, 0))
                        dir_name = os.path.join(output_dir, f'{camera.DEVICEID}_wJoints')
                    else:
                        dir_name = os.path.join(output_dir, camera.DEVICEID)
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name)
                    output_file = os.path.join(dir_name,
                                               f'{activity_name}.{camera.DEVICEID}.{time}.frame{int(frame_idx/2):06d}.jpg')
                    cv2.imwrite(output_file, frame)

            json_filename = os.path.join(output_dir, f'{activity_name}.{camera.DEVICEID}.json')
            with open(json_filename, 'w') as f:
                json.dump(json_list, f)

