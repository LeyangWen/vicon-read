import copy
import cv2
import os

import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import shutil
import multiprocessing as mproc

import kineticalib_mediapipe_local
import ref
from kineticalib_mediapipe_local import *
from calculate_angles_mediapipe_full import *
from angle_utils import *

# Set paths
absFilePath = os.path.dirname(os.path.abspath(__file__))
gen_keypoints_list = []


def parse_args(call_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", dest="dataset", help="The dataset to process")
    parser.add_argument("-c", "--config", dest="config",
                        help="Relative path to config file use for thresholds and colors",
                        default="thresholds/kinetica_config.json")
    parser.add_argument("-f", "--fps", dest="fps", type=int, help="Frames per second analyzed", default=15)
    parser.add_argument("--imageOutput", dest="image_output",
                        help="relative directory to where the processed images will be written")
    parser.add_argument("--poseInput", dest="pose_input",
                        help="relative directory to where pose estimation data is stored", default="pose/exp/default/")
    parser.add_argument("--keypoint2D", dest="keypoint_2d",
                        help="relative directory to where the 2D keypoint output file is stored",
                        default="keypoint_2d.txt")
    parser.add_argument("--keypoint3D", dest="keypoint_3d",
                        help="relative directory to where the 3D keypoint output text file is stored",
                        default="")
    parser.add_argument("--keypoint3D_MB", dest="keypoint_3d_MB",
                        help="relative directory to where the MotionBert 3D keypoint output text file is stored",
                        default="")
    parser.add_argument("--keypoint3D_MB_untilted", dest="keypoint_3d_MB_untilted",
                        help="relative directory to where the MotionBert 3D untilted keypoint output text file is stored",
                        default="")
    parser.add_argument("--data3D_MB_untilted", dest="data_3d_MB_untilted",
                        help="MotionBert 3D untilted keypoint data",
                        default="")
    parser.add_argument("--data3D_MB", dest="data_3d_MB",
                        help="MotionBert 3D keypoint data",
                        default="")
    parser.add_argument("--data3D", dest="data_3d",
                        help="3D keypoint data",
                        default="")
    parser.add_argument("--data2D", dest="data_2d",
                        help="2D keypoint data",
                        default="")
    parser.add_argument("--jsonOutput", dest="json_output",
                        help="relative directory to where json output will be written", default="data")
    parser.add_argument("-o", "--blockSubject", dest="block_subject", help="Block all detected faces",
                        action="store_true")
    parser.add_argument("--blurAll", dest="blur_all", help="Apply blur level 3", action="store_true")
    parser.add_argument("-b", "--blurLevel", dest="blur_level", type=int, help="Blur Level (0-3)", default=0)
    parser.add_argument("--blackBackground", dest="black_background", help="Draw the skeleton over a black background",
                        action="store_true")
    parser.add_argument("--noLegs", dest="no_legs", help="Do not draw leg segments", action="store_true")
    parser.add_argument("--seatedJob", dest="seated_job", help="Seated Job", action="store_true")

    parser.add_argument("--faceFile", dest="face_file", help="relative path to face data file",
                        default="face_data.json")
    parser.add_argument("--genKpsList", dest="gen_kps_list", help="List to store the keypoints being used",
                        default="gen_keypoints_list")
    parser.add_argument("--imgShape", dest="img_shape", help="Image shape (H ,W)",
                        default=(1200, 1920))
    parser.add_argument("--jsonPath2D", dest="json_path_2d", help="Full file location where the 2d json file is stored",
                        default='/home/ubuntu/mmpose/output/2d.json')

    if call_args:
        return parser.parse_args(args=[], namespace=copy.deepcopy(call_args))
    return parser.parse_args()


def configure(args):
    configuration["fps"] = args.fps
    configuration["configFile"] = os.path.join(absFilePath, args.config)
    configuration["dataset"] = args.dataset
    configuration["poseInput"] = os.path.join(absFilePath, args.pose_input)
    configuration["jsonOutput"] = os.path.join(absFilePath, args.json_output)
    configuration["blurLevel"] = args.blur_level
    configuration["keypoint2D"] = args.keypoint_2d
    configuration["keypoint3D"] = args.keypoint_3d
    configuration["keypoint3D_MB"] = args.keypoint_3d_MB
    configuration["keypoint3D_MB_untilted"] = args.keypoint_3d_MB_untilted
    configuration["data3D_MB_untilted"] = args.data_3d_MB_untilted
    configuration["data3D_MB"] = args.data_3d_MB
    configuration["data3D"] = args.data_3d
    configuration["data2D"] = args.data_2d
    configuration["faceFile"] = args.face_file
    configuration["noLegs"] = args.no_legs
    configuration["seated"] = args.seated_job
    configuration["genKpsList"] = args.gen_kps_list
    configuration["imgShape"] = args.img_shape
    configuration["jsonPath2D"] = args.json_path_2d

    if args.image_output is None:
        configuration["imageOutput"] = \
            os.path.join(absFilePath, "final_plot_out", "final_skeleton_plot_" + configuration["dataset"])
    else:
        configuration["imageOutput"] = os.path.join(absFilePath, args.image_output)

    processing_options = []
    if args.block_subject:
        processing_options.append("blockSubject")
    if args.blur_all:
        processing_options.append("blurAll")
    if args.black_background:
        processing_options.append("blackBackground")

    configuration["processingOptions"] = processing_options


def init_files():
    dataset = configuration["dataset"]

    if not os.path.isfile(configuration["keypoint2D"]):
        raise FileNotFoundError("2D keypoint file input not found: " + configuration["keypoint2D"])
    # if configuration["keypoint3D"] != "" and not os.path.isfile(configuration["keypoint3D"]):
    #     raise FileNotFoundError("3D keypoint file input not found: " + configuration["keypoint3D"])
    if configuration["keypoint3D_MB"] != "" and not os.path.isfile(configuration["keypoint3D_MB"]):
        raise FileNotFoundError("MotionBert 3D keypoint file input not found: " + configuration["keypoint3D_MB"])
    if configuration["keypoint3D_MB_untilted"] != "" and not os.path.isfile(configuration["keypoint3D_MB_untilted"]):
        raise FileNotFoundError("MotionBert 3D untilted keypoint file input not found: " + configuration["keypoint3D_MB_untilted"])

    # Setup outputs
    image_output_directory = configuration["imageOutput"]
    if os.path.isdir(image_output_directory):
        shutil.rmtree(image_output_directory)

    os.makedirs(image_output_directory)

    json_output_file = os.path.join(configuration["jsonOutput"], dataset + "_processed.json")
    if os.path.isfile(json_output_file):
        os.remove(json_output_file)


def ingest_config_file():
    config_file_name = configuration["configFile"]
    with open(config_file_name) as json_in:
        json_data = json.load(json_in)

    if "kinetica" in config_file_name:
        configuration["client"] = "kinetica"
    else:
        configuration["client"] = ""

    if "colors" in json_data:
        for color, rgb in json_data["colors"].items():
            skeleton_colors[color] = (rgb[2], rgb[1], rgb[0])

    for segment_name, segment in json_data["segments"].items():
        segments[segment_name]["thresholds"] = segment

    if "options" in json_data:
        configuration["options"] = json_data["options"]


# def calculate_angle_right_wrist(frame, joint):
#     """ Calculate the angle of a joint in degrees """
#     metadata = joint_metadata_2d_mediapipe[joint]
#     if "segments" not in metadata:
#         return None
#
#     # for adjacent_joint in get_segment_joints(metadata["segments"]):
#     #     if not check_point_confidence(frame, adjacent_joint):
#     #         return -1
#
#     invert = False
#     segment1, segment2 = metadata["segments"]
#     if "invertAngle" in metadata:
#         invert = bool(metadata["invertAngle"])
#
#     u_vector = frame[2,:2] - frame[0,:2] #(First vector : Elbow - Wrist)
#     v_vector = frame[0,:2] - frame[1,:2] #(First vector : Wrist - Middle Knuckle)
#
#     #     get_vector_2d_mediapipe(frame, segment1)
#     # v_vector = get_vector_2d_mediapipe(frame, segment2)
#
#     norm_u = norm(u_vector)
#     norm_v = norm(v_vector)
#     if norm_u == 0 or norm_v == 0:
#         return 0
#
#     # This is the cosine formula for vectors
#     angle = to_degrees(np.arccos(np.clip(np.dot(u_vector, v_vector) / (norm_u * norm_v), -1, 1)))
#     if math.isnan(angle):
#         angle = 0.0
#
#     if invert:
#         return 180.0 - angle
#     else:
#         return angle


def calculate_angle(frame, joint):
    """ Calculate the angle of a joint in degrees """
    metadata = joint_metadata[joint]
    if "segments" not in metadata:
        return None

    for adjacent_joint in get_segment_joints(metadata["segments"]):
        if not check_point_confidence(frame, adjacent_joint):
            return -1

    invert = False
    segment1, segment2 = metadata["segments"]
    if "invertAngle" in metadata:
        invert = bool(metadata["invertAngle"])

    u_vector = get_vector(frame, segment1)
    v_vector = get_vector(frame, segment2)

    norm_u = norm(u_vector)
    norm_v = norm(v_vector)
    if norm_u == 0 or norm_v == 0:
        return 0

    # This is the cosine formula for vectors
    angle = to_degrees(np.arccos(np.clip(np.dot(u_vector, v_vector) / (norm_u * norm_v), -1, 1)))
    if math.isnan(angle):
        angle = 0.0

    if invert:
        return 180.0 - angle
    else:
        return angle


def calculate_back_angle_2d(frame, joint):
    """ Calculate the angle of a joint in degrees """
    metadata = joint_metadata[joint]
    if "segments" not in metadata:
        return None

    # for adjacent_joint in get_segment_joints(metadata["segments"]):
    #     if not check_point_confidence(frame, adjacent_joint):
    #         return -1

    invert = False
    segment1, segment2 = metadata["segments"]
    if "invertAngle" in metadata:
        invert = bool(metadata["invertAngle"])

    u_vector = get_vector(frame, segment1)
    v_vector = get_vector(frame, segment2)

    norm_u = norm(u_vector)
    norm_v = norm(v_vector)
    if norm_u == 0 or norm_v == 0:
        return 0

    # This is the cosine formula for vectors
    angle = to_degrees(np.arccos(np.clip(np.dot(u_vector, v_vector) / (norm_u * norm_v), -1, 1)))
    if math.isnan(angle):
        angle = 0.0

    if invert:
        return 180.0 - angle
    else:
        return angle


def get_trunk_length(frame):
    trunk_length = get_segment_length(frame, "trunk")
    if trunk_length < 0:
        trunk_length = estimate_back_length(frame)
    if trunk_length < 0:
        return -1
    return trunk_length


def dict_to_array(coord_dict):
    """ Turns a dictionary into an np array of just values, no keys """
    coord_list = [coord_dict["x"], coord_dict["y"]]
    if "z" in coord_dict:
        coord_list.append(coord_dict["z"])
    if "confidence" in coord_dict:
        coord_list.append(coord_dict["confidence"])
    return np.array(coord_list)


def array_to_dict(coord_arr):
    """ Turns an np array of values into a dict with keys """
    if len(coord_arr) == 2:
        return {"x": coord_arr[0], "y": coord_arr[1]}
    elif len(coord_arr) == 3:
        return {"x": coord_arr[0], "y": coord_arr[1], "confidence": coord_arr[2]}
    elif len(coord_arr) == 4:
        return {"x": coord_arr[0], "y": coord_arr[1], "z": coord_arr[2], "confidence": coord_arr[3]}


def get_joint_midpoint(frame, joint1, joint2):
    """ Calculates midpoint between two given joints """
    joint1_data = dict_to_array(frame['points'][joint1])
    joint2_data = dict_to_array(frame['points'][joint2])

    return (joint1_data + joint2_data) / 2


def get_joint_difference(frame, joint1, joint2):
    """ Calculates vector difference between two given joints """
    joint1_data = dict_to_array(frame['points'][joint1])
    joint2_data = dict_to_array(frame['points'][joint2])

    return joint1_data - joint2_data


def get_angle(vector_a, vector_b):
    mag_a = np.linalg.norm(vector_a)
    mag_b = np.linalg.norm(vector_b)
    return np.degrees(np.arccos(np.clip(np.dot(vector_a, vector_b) / mag_a / mag_b, -1, 1)))


def get_neck_angle_3d(points):
    """ Calculates the neck angle and tilt direction based on 3D keypoints """
    thorax_to_neck = dict_to_array(points["lowerHead"])[:3] - dict_to_array(points["neck"])[:3]
    pelvis_to_thorax = dict_to_array(points["neck"])[:3] - dict_to_array(points["back"])[:3]
    neck_angle = get_angle(pelvis_to_thorax, thorax_to_neck)
    tilt_direction = "backward" if neck_angle < 45 else "forward"
    # tune to match desired magnitude
    neck_angle = abs(neck_angle - 45)
    return neck_angle, tilt_direction


def get_tilt_direction(frame):
    """ Returns tilt direction of head """

    mid_hip = get_joint_midpoint(frame, 'leftHip', 'rightHip')
    mid_shoulder = get_joint_midpoint(frame, 'leftShoulder', 'rightShoulder')
    lower_head = dict_to_array(frame['points']['lowerHead'])
    upper_head = dict_to_array(frame['points']['upperHead'])
    head = upper_head[:2] - lower_head[:2]
    backbone = mid_shoulder[:2] - mid_hip[:2]

    # Create axis perpendicular to backbone to project onto
    horizontal_axis = np.array([-backbone[1], backbone[0]])

    left_ear_to_eye = get_joint_difference(frame, 'leftEye', 'leftEar')[:2]
    right_ear_to_eye = get_joint_difference(frame, 'rightEye', 'rightEar')[:2]

    # If axis is really small we cannot determine (projection would be giant)
    axis_len_sq = horizontal_axis.dot(horizontal_axis)
    if axis_len_sq < 1e-5:
        return "cannot-determine"

    # Project ear-to-eye vectors on axis
    left_proj = left_ear_to_eye.dot(horizontal_axis) / axis_len_sq * horizontal_axis
    right_proj = right_ear_to_eye.dot(horizontal_axis) / axis_len_sq * horizontal_axis

    # Check if projections are big enough
    left_right_sum = left_proj + right_proj
    if left_right_sum.dot(left_right_sum) < 0.01:
        return "cannot-determine"

    # Determine pose direction
    left_proj_len = left_proj.dot(left_proj)
    right_proj_len = right_proj.dot(right_proj)
    pos_dir = left_proj if left_proj_len > right_proj_len else right_proj

    return "forward" if pos_dir.dot(head) > 0 else "backward"


def change_knee_angles(final_frame, angles, i):
    if (final_frame["points"]['leftHip']['confidence'] < 0.3 or final_frame["points"]['leftKnee']['confidence'] < 0.3 or final_frame["points"]['leftAnkle']['confidence'] < 0.3) and (
            final_frame["points"]['rightHip']['confidence'] < 0.3 or final_frame["points"]['rightKnee']['confidence'] < 0.3 or final_frame["points"]['rightAnkle']['confidence'] < 0.3):
        final_frame["points"]['leftKnee']['flexion'] = -1
        final_frame["points"]['rightKnee']['flexion'] = -1
        angles['leftKnee'].flexion[i] = -1 * (np.pi / 180)
        angles['rightKnee'].flexion[i] = -1 * (np.pi / 180)
    return final_frame, angles


def same_angle_for_knees(final_frame, angles, i):
    '''This function assignes lowest of the two angles to both knees'''
    left_knee_angle = final_frame["points"]['leftKnee']['flexion']
    right_knee_angle = final_frame["points"]['rightKnee']['flexion']
    if left_knee_angle == -1:
        if not right_knee_angle == -1:
            final_frame["points"]['leftKnee']['flexion'] = right_knee_angle
            final_frame["points"]['leftKnee']['dim'] = final_frame["points"]['rightKnee']['dim']
            angles['leftKnee'].flexion[i] = angles['rightKnee'].flexion[i]
    elif right_knee_angle == -1:
        final_frame["points"]['rightKnee']['flexion'] = left_knee_angle
        final_frame["points"]['rightKnee']['dim'] = final_frame["points"]['leftKnee']['dim']
        angles['rightKnee'].flexion[i] = angles['leftKnee'].flexion[i]
    elif left_knee_angle <= right_knee_angle:
        final_frame["points"]['rightKnee']['flexion'] = left_knee_angle
        final_frame["points"]['rightKnee']['dim'] = final_frame["points"]['leftKnee']['dim']
        angles['rightKnee'].flexion[i] = angles['leftKnee'].flexion[i]
    else:
        final_frame["points"]['leftKnee']['flexion'] = right_knee_angle
        final_frame["points"]['leftKnee']['dim'] = final_frame["points"]['rightKnee']['dim']
        angles['leftKnee'].flexion[i] = angles['rightKnee'].flexion[i]
    return final_frame, angles

    # if (not left_knee_angle == -1) and not (right_knee_angle == -1):
    #     if left_knee_angle <= right_knee_angle:
    #         final_frame["points"]['rightKnee']['flexion'] = left_knee_angle
    #         final_frame["points"]['rightKnee']['dim'] = final_frame["points"]['leftKnee']['dim']
    #         angles['rightKnee'].flexion[i] = angles['leftKnee'].flexion[i]
    #     else:
    #         final_frame["points"]['leftKnee']['flexion'] = right_knee_angle
    #         final_frame["points"]['leftKnee']['dim'] = final_frame["points"]['rightKnee']['dim']
    #         angles['leftKnee'].flexion[i] = angles['rightKnee'].flexion[i]
    # return final_frame,angles


def get_tanh_parameter(final_frame, img_vertical, image_shape=(1200, 1920)):
    '''This function returns the parameter for tanh function based on the image orientation'''
    if img_vertical:
        if (final_frame['points']['head']['confidence'] > 0.3) and (final_frame['points']['leftAnkle']['confidence'] > 0.3):
            height_person = abs(final_frame['points']['head']['y'] - final_frame['points']['leftAnkle']['y'])
            height_ratio = height_person / image_shape[0]
        elif (final_frame['points']['head']['confidence'] > 0.3) and (final_frame['points']['rightAnkle']['confidence'] > 0.3):
            height_person = abs(final_frame['points']['head']['y'] - final_frame['points']['rightAnkle']['y'])
            height_ratio = height_person / image_shape[0]
        elif (final_frame['points']['head']['confidence'] > 0.3) and (final_frame['points']['pelvis']['confidence'] > 0.3):
            height_person = abs(final_frame['points']['head']['y'] - final_frame['points']['pelvis']['y'])
            height_ratio = height_person * 2 / image_shape[0]
        else:
            height_ratio = None

        if height_ratio:
            tanh_param = max(0.1, (90 * height_ratio - 45))
        else:
            tanh_param = 25
    else:
        if (final_frame['points']['head']['confidence'] > 0.3) and (final_frame['points']['leftAnkle']['confidence'] > 0.3):
            tanh_param = 0.1
        elif (final_frame['points']['head']['confidence'] > 0.3) and (final_frame['points']['rightAnkle']['confidence'] > 0.3):
            tanh_param = 0.1
        else:
            tanh_param = 25
    return tanh_param


def assign_untilted_angles_to_tilted_angles(angles, angles_untilted, part):
    angles[part].flexion = angles_untilted[part].flexion
    angles[part].abduction = angles_untilted[part].abduction
    angles[part].rotation = angles_untilted[part].rotation


def get_output_data(input_data, input_data_3d_MB=None, data_3d_MB=None, input_data_3d_MB_untilted=None, data_3d_MB_untilted=None, img_shape=(1200, 1920)):
    """ Perform transformation on points and enrich with calculated information, e.g. joint angles """
    res_h, res_w = img_shape
    if res_h > res_w:
        img_vertical = True
    else:
        img_vertical = False

    output_frames = []

    pixel_trunk = []
    initial_pixel_heights = []

    # Calculate new head points and inject into input data
    # for i, frame in enumerate(input_data):
    #     lower_head, upper_head = get_head_segment(frame)
    #     input_data[i]["points"]["lowerHead"] = lower_head
    #     input_data[i]["points"]["upperHead"] = upper_head
    angles = calculate_angles_mediapipe(data_3d_MB[:, :, :3])
    angles_untilted = calculate_angles_mediapipe(data_3d_MB_untilted[:, :, :3])
    assign_untilted_angles_to_tilted_angles(angles, angles_untilted, "neck")
    assign_untilted_angles_to_tilted_angles(angles, angles_untilted, "back")

    back_angles_3d = {}
    back_angles_3d["flexion"] = angles["back"].flexion * (180 / np.pi)
    back_angles_3d["abduction"] = angles["back"].abduction * (180 / np.pi)
    back_angles_3d["rotation"] = angles["back"].rotation * (180 / np.pi)

    # for sub_parts in ['neck','leftWrist','rightWrist']: # For neck and wrists we are using 2d angles, so we create the Joint angles and populate them with actual values in lines 345-349
    #     angles[sub_parts] = JointAngles()
    #     angles[sub_parts].flexion = np.zeros(len(input_data))
    #     angles[sub_parts].abduction = None
    #     angles[sub_parts].rotation = None
    back_angles = []
    for i, frame in enumerate(input_data):
        i_str = str(i + 1)
        input_data[i]["headDirection"] = None
        final_frame = {
            "headDirection": input_data[i]["headDirection"],
            "points": {},
            "filename": "img" + "0" * (5 - len(i_str)) + i_str + ".jpg",
            "filepath": frame["filename"]
        }

        # if input_data_3d is not None:
        #     for part, point in input_data_3d[i]["points"].items():
        #         extrapolate_point(input_data_3d, i, part)
        #         point['x'], point['y'], point['z'] = smooth_point(input_data_3d, i, part)
        #         point["confidence"] = smooth_point_confidence(input_data_3d, i, part)

        for part, point in frame["points"].items():
            final_frame["points"][part] = {}
            final_frame["points"][part]["x"] = point['x']
            final_frame["points"][part]["y"] = point['y']
            final_frame["points"][part]["confidence"] = point['confidence']

        for part, point in final_frame["points"].items():
            if has_adjacent_segments(part):
                if ('back' in part):
                    back_angles.append(calculate_back_angle_2d(final_frame, part))

                if ("Knee" in part):
                    final_frame["points"][part]['flexion'] = calculate_angle(final_frame, part)
                    final_frame["points"][part]['abduction'] = None
                    final_frame["points"][part]['rotation'] = None
                    if final_frame["points"][part]['flexion'] == -1:
                        final_frame["points"][part]['flexion'] = (angles[part].flexion[i]) * (180 / np.pi)
                        final_frame["points"][part]['dim'] = '3d'
                    else:
                        final_frame["points"][part]['dim'] = '2d'
                        angles[part].flexion[i] = (final_frame["points"][part]['flexion']) * (np.pi / 180)
                    if configuration["seated"] and not configuration["noLegs"]:
                        final_frame["points"][part]['dim'] = 'seated'
                elif ("Wrist" in part):
                    final_frame["points"][part]['flexion'] = calculate_angle(final_frame, part)
                    final_frame["points"][part]['abduction'] = None
                    final_frame["points"][part]['rotation'] = None
                    final_frame["points"][part]['dim'] = '2d'
                    angles[part].flexion[i] = (final_frame["points"][part]['flexion']) * (
                                np.pi / 180)  # We are doing this because the 3d angles calculated are in radians, so we are converting our 2d angles to radians to store them in angles dictionary
                elif ("back" in part):
                    back_angle_frame = calculate_back_angle_2d(final_frame, part)
                    # if back_angle_frame > 10:
                    #     tanh_param = 25
                    # else:
                    #     tanh_param = 45
