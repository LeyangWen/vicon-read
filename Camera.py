import xml.etree.ElementTree as ET
import os
import numpy as np
import cv2
import pandas as pd


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
    df = pd.read_csv(csv_filename, skiprows=[0, 1, 2, 4], header=[0])
    # header_df = pd.read_csv(csv_filename, skiprows=[0, 1], nrows=2, header=None)
    return df


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

    def camera_intrinsic_matrix(self):
        K = np.array([[self.FOCAL_LENGTH, self.SKEW, self.PRINCIPAL_POINT[0]],
                      [0, self.FOCAL_LENGTH / self.PIXEL_ASPECT_RATIO, self.PRINCIPAL_POINT[1]],
                      [0, 0, 1]])
        return K

    @staticmethod
    def projection_matrix_from_intrinsic_matrix_and_rot3x4(K, rot3x4):
        P = K.dot(rot3x4)
        return P

    def draw_2d(self, image, points_2d, color=(0, 0, 255)):
        print(points_2d)
        for point_2d in points_2d:
            cv2.circle(image, (int(point_2d[0]), int(point_2d[1])), 3, color, -1)
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
        self.intrinsic_matrix = self.camera_intrinsic_matrix()
        self.projection_matrix = self.projection_matrix_from_intrinsic_matrix_and_rot3x4(self.intrinsic_matrix, self.rot3x4)
        return self.projection_matrix

    def project(self, points_3d):
        self.get_projection_matrix()
        points_3d = np.array(points_3d)
        points_2d = self.projection_matrix.dot(np.vstack([points_3d.T, np.ones([1, points_3d.shape[0]])]))
        points_2d = points_2d[0:2, :] / points_2d[2, :]
        points_2d = points_2d.T
        return points_2d

if __name__ == '__main__':
    base_dir_name = 'Z:\project 2d\skeleton template'
    activity_no = 1
    frame_no = 600
    activity_name = f'Activity {activity_no:02d}'
    basename = os.path.join(base_dir_name, activity_name)
    c3d_filename = basename + '.c3d'
    xcp_filename = basename + '.xcp'
    csv_filename = basename + '.trajectory.csv'
    df = load_csv(csv_filename)
    points_3d = df.iloc[frame_no, 2:].values.reshape(-1, 3) / 1000
    time = '20230407161015'
    cameras = batch_load_from_xcp(xcp_filename)
    camera = cameras[1]
    for camera in cameras:
        video_filename = f'{basename}.{camera.DEVICEID}.{time}.avi'
        # get first frame of video
        cap = cv2.VideoCapture(video_filename)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no-1)
        ret, frame = cap.read()
        points_2d = camera.project(points_3d)
        # frame = camera.draw_2d(frame, points_2d)
        points_2d = camera.distort(points_2d)
        # points_2d = camera.undistort(points_2d)
        frame = camera.draw_2d(frame, points_2d, color=(0, 255, 0))
        cv2.imshow('frame', frame)
        cv2.waitKey(0)