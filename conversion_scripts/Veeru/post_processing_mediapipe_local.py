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
                        default=(1200,1920))
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
def change_knee_angles(final_frame,angles,i):
    if (final_frame["points"]['leftHip']['confidence'] < 0.3 or final_frame["points"]['leftKnee']['confidence'] < 0.3 or final_frame["points"]['leftAnkle']['confidence'] < 0.3) and (final_frame["points"]['rightHip']['confidence'] < 0.3 or final_frame["points"]['rightKnee']['confidence'] < 0.3 or final_frame["points"]['rightAnkle']['confidence'] < 0.3):
        final_frame["points"]['leftKnee']['flexion'] = -1
        final_frame["points"]['rightKnee']['flexion'] = -1
        angles['leftKnee'].flexion[i] = -1 * (np.pi/180)
        angles['rightKnee'].flexion[i] = -1 * (np.pi/180)
    return final_frame,angles



def same_angle_for_knees(final_frame,angles,i):
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
    return final_frame,angles

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

def get_tanh_parameter(final_frame,img_vertical,image_shape=(1200,1920)):
    '''This function returns the parameter for tanh function based on the image orientation'''
    if img_vertical:
        if (final_frame['points']['head']['confidence'] > 0.3) and (final_frame['points']['leftAnkle']['confidence'] > 0.3 ):
            height_person = abs(final_frame['points']['head']['y'] - final_frame['points']['leftAnkle']['y'])
            height_ratio = height_person / image_shape[0]
        elif (final_frame['points']['head']['confidence'] > 0.3) and (final_frame['points']['rightAnkle']['confidence'] > 0.3):
            height_person = abs(final_frame['points']['head']['y'] - final_frame['points']['rightAnkle']['y'])
            height_ratio = height_person / image_shape[0]
        elif (final_frame['points']['head']['confidence'] > 0.3) and (final_frame['points']['pelvis']['confidence'] > 0.3):
            height_person = abs(final_frame['points']['head']['y'] - final_frame['points']['pelvis']['y'])
            height_ratio = height_person*2 / image_shape[0]
        else:
            height_ratio = None

        if height_ratio:
            tanh_param = max(0.1,(90*height_ratio - 45))
        else:
            tanh_param = 25
    else:
        if (final_frame['points']['head']['confidence'] > 0.3) and (final_frame['points']['leftAnkle']['confidence'] > 0.3 ):
            tanh_param = 0.1
        elif (final_frame['points']['head']['confidence'] > 0.3) and (final_frame['points']['rightAnkle']['confidence'] > 0.3):
            tanh_param = 0.1
        else:
            tanh_param = 25
    return tanh_param




def assign_untilted_angles_to_tilted_angles(angles,angles_untilted,part):
    angles[part].flexion = angles_untilted[part].flexion
    angles[part].abduction = angles_untilted[part].abduction
    angles[part].rotation = angles_untilted[part].rotation

def get_output_data(input_data,input_data_3d_MB=None,data_3d_MB = None,input_data_3d_MB_untilted=None,data_3d_MB_untilted = None,img_shape=(1200,1920)):
    """ Perform transformation on points and enrich with calculated information, e.g. joint angles """
    res_h,res_w = img_shape
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
    angles = calculate_angles_mediapipe(data_3d_MB[:,:,:3])
    angles_untilted = calculate_angles_mediapipe(data_3d_MB_untilted[:,:,:3])
    assign_untilted_angles_to_tilted_angles(angles,angles_untilted,"neck")
    assign_untilted_angles_to_tilted_angles(angles,angles_untilted,"back")



    back_angles_3d = {}
    back_angles_3d["flexion"] = angles["back"].flexion *(180/np.pi)
    back_angles_3d["abduction"] = angles["back"].abduction *(180/np.pi)
    back_angles_3d["rotation"] = angles["back"].rotation *(180/np.pi)


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
                    back_angles.append(calculate_back_angle_2d(final_frame,part))

                if ("Knee" in part):
                    final_frame["points"][part]['flexion'] = calculate_angle(final_frame,part)
                    final_frame["points"][part]['abduction'] = None
                    final_frame["points"][part]['rotation'] = None
                    if final_frame["points"][part]['flexion'] == -1:
                        final_frame["points"][part]['flexion'] = (angles[part].flexion[i]) * (180/np.pi)
                        final_frame["points"][part]['dim'] = '3d'
                    else:
                        final_frame["points"][part]['dim'] = '2d'
                        angles[part].flexion[i] = (final_frame["points"][part]['flexion'])*(np.pi/180)
                    if configuration["seated"] and not configuration["noLegs"]:
                        final_frame["points"][part]['dim'] = 'seated'
                elif ("Wrist" in part):
                     final_frame["points"][part]['flexion'] = calculate_angle(final_frame,part)
                     final_frame["points"][part]['abduction'] = None
                     final_frame["points"][part]['rotation'] = None
                     final_frame["points"][part]['dim'] = '2d'
                     angles[part].flexion[i] = (final_frame["points"][part]['flexion'])*(np.pi/180) # We are doing this because the 3d angles calculated are in radians, so we are converting our 2d angles to radians to store them in angles dictionary
                elif ("back" in part):
                    back_angle_frame = calculate_back_angle_2d(final_frame,part)
                    # if back_angle_frame > 10:
                    #     tanh_param = 25
                    # else:
                    #     tanh_param = 45
                    # tanh_param = get_tanh_parameter(final_frame,img_vertical,image_shape=(res_h,res_w))
                    # tanh_param = 25
                    angle = angles[part]
                    if np.any(angle.flexion):
                            final_frame["points"][part]['flexion'] = angle.flexion[i]*(180/np.pi)
                            # if not configuration["seated"]:
                            if final_frame["points"][part]['flexion']:
                                back_angle_flex = copy.deepcopy(final_frame["points"][part]['flexion'])
                                if back_angle_frame <= 20:
                                    final_frame["points"][part]['flexion'] = 20
                                else:
                                    if back_angle_flex < 0:
                            #             tanh_param = 25
                            #         else:
                                        tanh_param = 45
                                        final_frame["points"][part]['flexion'] *= abs(np.tanh(back_angle_flex/tanh_param))
                                angles[part].flexion[i] = (final_frame["points"][part]['flexion'])*(np.pi/180)
                                angle.flexion[i] = angles[part].flexion[i]
                    else:
                            final_frame["points"][part]['flexion'] = None
                    if np.any(angle.abduction):
                            final_frame["points"][part]['abduction'] = angle.abduction[i]*(180/np.pi)
                            # if final_frame["points"][part]['abduction']:
                            #     back_angle = copy.deepcopy(final_frame["points"][part]['abduction'])
                            #     if back_angle_frame <= 20:
                            #         final_frame["points"][part]['abduction'] = 10
                            #     else:
                            #         tanh_param = 25
                            #         final_frame["points"][part]['abduction'] *= abs(np.tanh(back_angle/tanh_param))
                            #     angles[part].abduction[i] = (final_frame["points"][part]['abduction'])*(np.pi/180)
                            #     angle.abduction[i] = angles[part].abduction[i]
                    else:
                            final_frame["points"][part]['abduction'] = None
                    if np.any(angle.rotation):
                        final_frame["points"][part]['rotation'] = angle.rotation[i]*(180/np.pi)
                    else:
                        final_frame["points"][part]['rotation'] = None
                    final_frame["points"][part]['dim'] = '3d'
                    if configuration["seated"]:
                        final_frame["points"][part]['dim'] = 'seated'
                    # if np.all((input_data_3d[i]['points'][part]['x'] == 0)&(input_data_3d[i]['points'][part]['y'] == 0)&(input_data_3d[i]['points'][part]['z'] == 0)): # If the 3d point is not detected, we use the 2d angles
                    #     final_frame["points"][part]['flexion'] = calculate_angle(final_frame,part)
                    #     final_frame["points"][part]['abduction'] = None
                    #     final_frame["points"][part]['rotation'] = None
                    #     final_frame["points"][part]['dim'] = '2d'
                    #     angles[part].flexion[i] = (final_frame["points"][part]['flexion'])*(np.pi/180)
                    #     angles[part].rotation[i] = None # This is because the twist is coming from the motionbert model
                    # else:
                    #     angle = angles[part]
                    #     if np.any(angle.flexion):
                    #         final_frame["points"][part]['flexion'] = angle.flexion[i]*(180/np.pi)
                    #         if (final_frame["points"][part]['flexion']): # This is the idea to reduce the back angle backward bend only
                    #             back_angle_flex = final_frame["points"][part]['flexion']
                    #             if back_angle_flex < 0:
                    #                 divisor = -25
                    #                 final_frame["points"][part]['flexion'] *= np.tanh(back_angle_flex/divisor)
                    #                 angles[part].flexion[i] = (final_frame["points"][part]['flexion'])*(np.pi/180)
                    #                 angle.flexion[i] = angles[part].flexion[i]
                    #     else:
                    #         final_frame["points"][part]['flexion'] = None
                    #     if np.any(angle.abduction):
                    #         final_frame["points"][part]['abduction'] = angle.abduction[i]*(180/np.pi)
                    #         if (final_frame["points"][part]['abduction']): # This is the idea to reduce the back angle lateral bend
                    #             back_angle = final_frame["points"][part]['abduction']
                    #             divisor = 25
                    #             final_frame["points"][part]['abduction'] *= abs(np.tanh(back_angle/divisor))
                    #             angles[part].abduction[i] = (final_frame["points"][part]['abduction'])*(np.pi/180)
                    #             angle.abduction[i] = angles[part].abduction[i]
                    #     else:
                    #         final_frame["points"][part]['abduction'] = None
                    #     if np.any(angle.rotation):
                    #         final_frame["points"][part]['rotation'] = angle.rotation[i]*(180/np.pi)
                    #         # if ('back' in part) and (final_frame["points"][part]['rotation']): # This is the idea to reduce the back angle lateral bend
                    #         #     back_angle_rot = final_frame["points"][part]['rotation']
                    #         #     divisor = 25
                    #         #     if back_angle_rot < 0:
                    #         #         divisor = -25
                    #         #     final_frame["points"][part]['rotation'] *= np.tanh(back_angle_rot/divisor)
                    #         #     angles[part].rotation[i] = (final_frame["points"][part]['rotation'])*(np.pi/180)
                    #         #     angle.rotation[i] = angles[part].rotation[i]
                    #     else:
                    #         final_frame["points"][part]['rotation'] = None
                    #     final_frame["points"][part]['dim'] = '3d'

                elif ("neck" in part):
                    angle = angles[part]
                    if np.any(angle.flexion):
                            final_frame["points"][part]['flexion'] = angle.flexion[i]*(180/np.pi)
                            if final_frame["points"][part]['flexion']:
                                final_frame["points"][part]['flexion'] -= 20
                                angles[part].flexion[i] = (final_frame["points"][part]['flexion'])*(np.pi/180)
                                angle.flexion[i] = angles[part].flexion[i]
                    else:
                        final_frame["points"][part]['flexion'] = None
                    if np.any(angle.abduction):
                        final_frame["points"][part]['abduction'] = angle.abduction[i]*(180/np.pi)
                    else:
                        final_frame["points"][part]['abduction'] = None
                    if np.any(angle.rotation):
                        final_frame["points"][part]['rotation'] = angle.rotation[i]*(180/np.pi)
                    else:
                        final_frame["points"][part]['rotation'] = None
                    final_frame["points"][part]['dim'] = '2d'
                else:
                    angle = angles[part]
                    if np.any(angle.flexion):
                        final_frame["points"][part]['flexion'] = angle.flexion[i]*(180/np.pi)
                    else:
                        final_frame["points"][part]['flexion'] = None
                    if np.any(angle.abduction):
                        final_frame["points"][part]['abduction'] = angle.abduction[i]*(180/np.pi)
                    else:
                        final_frame["points"][part]['abduction'] = None
                    if np.any(angle.rotation):
                        final_frame["points"][part]['rotation'] = angle.rotation[i]*(180/np.pi)
                    else:
                        final_frame["points"][part]['rotation'] = None
                    # if 'neck' in part:
                    #     final_frame["points"][part]['dim'] = '2d'
                    # else:
                    final_frame["points"][part]['dim'] = '3d'

                # final_frame["points"][part]['abduction'] = angle.abduction[i]*(180/np.pi) if angle.abduction[i] else None
                # final_frame["points"][part]['rotation'] = angle.rotation[i]*(180/np.pi) if angle.rotation[i] else None


                # use special function for calculating neck in 3D
            #     if part == "neck":
            #         angle, tilt_direction = get_neck_angle_3d(input_data_3d[i]["points"])
            #         final_frame["headDirection"] = tilt_direction
            #     else:
            #         angle = calculate_angle(input_data_3d[i], part)
            # else:
            #     angle = calculate_angle(final_frame, part)
            # if angle is not None:
            #     final_frame["points"][part]["angle"] = angle

        # if input_data_3d is not None:
        #     final_frame["hand_angles"] = {}
        #     for side in ["left", "right"]:
        #         final_frame["hand_angles"][side] = get_hand_angles(input_data_3d[i], side)
        #         final_frame["points"][side + "Palm"]["angle"] = abs(final_frame["hand_angles"][side]["flexion"])

        # Fixing the legs issue
        final_frame,angles = change_knee_angles(final_frame,angles,i)
        final_frame,angles = same_angle_for_knees(final_frame,angles,i)
        initial_pixel_heights.append(int(estimate_height(final_frame)))
        pixel_trunk.append(get_trunk_length(final_frame))
        output_frames.append(final_frame)

    # plot_back_angle_2d(back_angles,configuration["dataset"])

    # plot_back_angle_3d(back_angles_3d,back_angles,configuration["dataset"])

    # corrected 2D trunk length is 95th percentile of all measured lengths
    max_pixel_trunk = np.percentile(pixel_trunk, 95)
    for frame in output_frames:

        calculate_frame_data(frame, max_pixel_trunk)

    final_pixel_heights = get_local_max_pixel_heights(initial_pixel_heights)
    return {"frames": output_frames, "metadata": {"pixel_heights": final_pixel_heights}, "angles": angles}


def draw_individual_skeleton_mp(par_values):
    start_idx,end_idx,frames,segments,faces,configuration,skeleton_colors = par_values
    segment_mapper = {}
    for index in range(start_idx, end_idx):
        frame = frames[index]
        file_name = frame['filename']
        image_path = frame['filepath']
        image_path_final = os.path.join(configuration["imageOutput"], str(file_name))

        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path + " doesn't exist")
        image_file = cv2.imread(str(image_path))

        if "blackBackground" in configuration["processingOptions"]:
            image_file = black_background(image_file)
        else:
            if "blockSubject" in configuration["processingOptions"]:
                image_file = block_all_faces(image_file, frames, index, faces)
            if configuration["blurLevel"] > 0:
                image_file = blur_image(image_file, configuration["blurLevel"])
            elif "blurAll" in configuration["processingOptions"]:
                image_file = blur_image(image_file, 3)

        for segment_name, segment_config in segments.items():
            image_file,risk_color_2d = draw_segment(image_file, frames[index], segment_name, segment_config,configuration=configuration,skeleton_colors=skeleton_colors)
            if "imaginary" not in segment_config:
                if segment_name not in segment_mapper:
                    segment_mapper[segment_name] = {}
                    segment_mapper[segment_name]["adjacent_joints"] = []
                    segment_mapper[segment_name]["risk_color"] = []
                part1 = vicon_gt_matrix.index(segment_config["adjacent_joints"][0])
                part2 = vicon_gt_matrix.index(segment_config["adjacent_joints"][1])
                segment_mapper[segment_name]["adjacent_joints"].append((part1,part2))
                segment_mapper[segment_name]["risk_color"].append(risk_color_2d)

        # Write notes to frame
        if 'notes' in frame:
            draw_text(image_file, frame['notes'])  # , font_color=(255, 255, 255), font_thickness = 2, font_scale = 1

        cv2.imwrite(image_path_final, image_file)

    return segment_mapper


def draw_skeleton_mp(data):
    """ Draw's lines connecting joints on all images """
    frames = data["frames"]
    segment_mapper_final = {}

    if "blockSubject" in configuration["processingOptions"]:
        faces = load_faces_from_file()
    else:
        faces = {}

    total_frames_count = len(frames)
    core_used = mproc.cpu_count() - 1
    pool = mproc.Pool(core_used)
    core_share = int(math.ceil(total_frames_count / core_used)) #how many frames would each core process
    # results = pool.map(function_to_process_a_frame, [(frame[core_ID * core_share: min((core_ID + 1) * core_share, total_frames_count, keypoints[core_ID * core_share: min((core_ID + 1) * core_share, total_frames_count)]) for core_ID in range(core_used)])
    parameter_list = []
    for core_idx in range(core_used):
        # define subset of frames that current core will process
        start_idx = core_idx * core_share
        end_idx = min(
            (core_idx + 1) * core_share,total_frames_count)
        parameter_list.append((start_idx,end_idx,frames,segments,faces,configuration,skeleton_colors))

    results_segment = pool.map(draw_individual_skeleton_mp,parameter_list)

    pool.close()
    pool.join()

    for result in results_segment:
        for segment_name_out, segment_data_out in result.items():
            if segment_name_out not in segment_mapper_final:
                segment_mapper_final[segment_name_out] = {}
                segment_mapper_final[segment_name_out]["adjacent_joints"] = []
                segment_mapper_final[segment_name_out]["risk_color"] = []
            segment_mapper_final[segment_name_out]["adjacent_joints"].extend(segment_data_out["adjacent_joints"])
            segment_mapper_final[segment_name_out]["risk_color"].extend(segment_data_out["risk_color"])

    return segment_mapper_final





def draw_skeleton(data):
    """ Draw's lines connecting joints on all images """

    segment_mapper = {}

    frames = data["frames"]

    if "blockSubject" in configuration["processingOptions"]:
        faces = load_faces_from_file()
    else:
        faces = {}

    for index, frame in enumerate(frames):
        file_name = frame['filename']
        image_path = frame['filepath']
        image_path_final = os.path.join(configuration["imageOutput"], str(file_name))

        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path + " doesn't exist")
        image_file = cv2.imread(str(image_path))

        if "blackBackground" in configuration["processingOptions"]:
            image_file = black_background(image_file)
        else:
            if "blockSubject" in configuration["processingOptions"]:
                image_file = block_all_faces(image_file, frames, index, faces)
            if configuration["blurLevel"] > 0:
                image_file = blur_image(image_file, configuration["blurLevel"])
            elif "blurAll" in configuration["processingOptions"]:
                image_file = blur_image(image_file, 3)

        for segment_name, segment_config in segments.items():

            # if 'rightUpperArm' in segment_name and index > 49:
            #     print('here')
            image_file,risk_color_2d = draw_segment(image_file, frames[index], segment_name, segment_config,configuration=configuration,skeleton_colors=skeleton_colors)
            if "imaginary" not in segment_config:
                if segment_name not in segment_mapper:
                    segment_mapper[segment_name] = {}
                    segment_mapper[segment_name]["adjacent_joints"] = []
                    segment_mapper[segment_name]["risk_color"] = []
                part1 = vicon_gt_matrix.index(segment_config["adjacent_joints"][0])
                part2 = vicon_gt_matrix.index(segment_config["adjacent_joints"][1])
                segment_mapper[segment_name]["adjacent_joints"].append((part1,part2))
                segment_mapper[segment_name]["risk_color"].append(risk_color_2d)






        # Write notes to frame
        if 'notes' in frame:
            draw_text(image_file, frame['notes'])  # , font_color=(255, 255, 255), font_thickness = 2, font_scale = 1

        cv2.imwrite(image_path_final, image_file)
    return segment_mapper


def write_json(data):
    """ Write the output data into a json file """
    json_data = get_json_data(data)
    with open(configuration["jsonPath2D"], 'r') as f:
        json_data_2d = json.load(f)
    json_data.update(json_data_2d)

    with open(os.path.join(configuration["jsonOutput"], configuration["dataset"] + "_processed.json"), 'w') \
            as json_file:
        json.dump(json_data, json_file)


def extrapolate_point(frames, index, part):
    if index == 0 or index == len(frames) - 1 or check_point_confidence(frames[index], part):
        return

    if check_point_confidence(frames[index - 1], part) and check_point_confidence(frames[index + 1], part):
        coord_prev = np.array(get_coords(frames[index - 1], part))
        coord_next = np.array(get_coords(frames[index + 1], part))
        coord_avg = (coord_prev + coord_next) // 2
        frames[index]["points"][part]["x"] = int(coord_avg[0])
        frames[index]["points"][part]["y"] = int(coord_avg[1])
        if "z" in frames[index]["points"][part]:
            frames[index]["points"][part]["z"] = int(coord_avg[2])
        frames[index]["points"][part]["confidence"] = joint_metadata[part]["tolerance"]


def smooth_point(frames, index, part, window=8):
    """ Smooth points out based on their adjacent points """
    start_index = max(0, index - window // 2)
    end_index = min(len(frames), index + window // 2 + 1)

    arr_sum = np.zeros(3 if "z" in frames[index]["points"][part] else 2)
    weight = 0.0
    for i in range(start_index, end_index):
        point = frames[i]["points"][part]
        if check_point_confidence(frames[i], part):
            # Weight each point by the algorithm's confidence in it as well as its distance from the target point
            multiplier = point["confidence"] * (abs(i - index) - window) ** 2 / (window ** 2)
            arr_sum += np.array(get_coords(frames[i], part)) * multiplier
            weight += multiplier

    if weight == 0.0:
        return get_coords(frames[index], part)

    return (arr_sum/weight).astype(np.int).tolist()


def smooth_point_confidence(frames, index, part, window=8):
    """ Smooth point confidence out based on their adjacent frame's confidence values (only maxima) """
    start_index = max(0, index - window // 2)
    end_index = min(len(frames), index + window // 2 + 1)
    num_points = end_index - start_index

    sum_neighbor_confidence = 0.0
    for i in range(start_index, end_index):
        sum_neighbor_confidence += frames[i]["points"][part]['confidence']

    average_adjacent_confidence = sum_neighbor_confidence / num_points
    return average_adjacent_confidence


def check_point_confidence(frame, point):
    if "verticalSpine" == point:
        return check_point_confidence(frame, "back")

    return frame["points"][point]["confidence"] >= joint_metadata[point]["tolerance"]


def has_adjacent_segments(joint):
    metadata = joint_metadata[joint]
    if "segments" not in metadata:
        return False
    return True
    # return all(joint in frame["points"] for joint in get_segment_joints(metadata["segments"]))


# def calculate_angle(frame, joint):
#     """ Calculate the angle of a joint in degrees """
#     metadata = joint_metadata[joint]
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
#     u_vector = get_vector(frame, segment1)
#     v_vector = get_vector(frame, segment2)
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


def calculate_frame_data(frame, max_pixel_trunk):
    # calculate_extension(frame, "Elbow", "Wrist", "Palm")
    # calculate_extension(frame, "Knee", "Ankle", "Heel")
    # hands_x, hands_y = estimate_midpoint(frame, "Palm")
    # feet_x, feet_y = estimate_midpoint(frame, "Heel")
    wrist_x, wrist_y = estimate_midpoint(frame, "Wrist")

    frame["wrists"] = {"x": wrist_x, "y": wrist_y}

    # if hands_y > 0 and feet_y > 0:
    #     hand_horizontal = abs(hands_x - feet_x)
    #     # this seems backwards but y pixels are counted starting at the top of the image
    #     hand_vertical = feet_y - hands_y
    #
    #     trunk_length = get_trunk_length(frame)
    #     # strictly greater to avoid divide by zero
    #     if trunk_length > 0:
    #         # divide initial calculation by cos(theta) which is ratio between current and max trunk length
    #         hand_horizontal /= trunk_length / max_pixel_trunk
    # else:
    #     hand_horizontal = -1
    #     hand_vertical = -1
    #
    # # Discard hand data if we aren't confident in both palms
    # if not check_point_confidence(frame, "leftPalm") or not check_point_confidence(frame, "rightPalm"):
    #     hand_horizontal = -1
    #     hand_vertical = -1

    frame["hands"] = {"horizontal": -1, "vertical": -1}


def calculate_extension(frame, start, end, part):
    """ Calculate a body point that extends from the end of an existing segment, scaled by some constant factor """
    calculate_single_extension(frame, start, end, part, "left")
    calculate_single_extension(frame, start, end, part, "right")


def calculate_single_extension(frame, start, end, part, side):
    # defines how far the new part will extend past the end point
    scale_factor = configuration["options"]["{}ScaleFactor".format(part.lower())]

    start = side + start
    end = side + end
    part = side + part

    start_x, start_y = get_coords(frame, start)
    end_x, end_y = get_coords(frame, end)

    x_diff = end_x - start_x
    y_diff = end_y - start_y

    new_x_diff = x_diff * scale_factor
    new_y_diff = y_diff * scale_factor

    part_x = start_x + new_x_diff
    part_y = start_y + new_y_diff

    frame["points"][part] = {
        "x": int(part_x),
        "y": int(part_y),
        "confidence": min(frame["points"][start]["confidence"], frame["points"][end]["confidence"])
    }


def estimate_height(frame):
    # ratio of head length to body height
    head_ratio = configuration["options"]["headScaleFactor"]

    calf_length = max(get_segment_length(frame, "rightCalf"), get_segment_length(frame, "leftCalf"))
    if calf_length < 0:
        return -1

    thigh_length = max(get_segment_length(frame, "rightThigh"), get_segment_length(frame, "leftThigh"))
    if thigh_length < 0:
        return -1

    back_length = get_segment_length(frame, "trunk")
    if back_length < 0:
        back_length = estimate_back_length(frame)
    if back_length < 0:
        return -1

    return (calf_length + thigh_length + back_length) / (1 - head_ratio)


def get_local_max_pixel_heights(initial_heights, window=3):
    adjusted_heights = []
    for index, height in enumerate(initial_heights):
        local_max = -1
        for window_index in range(max(0, index - window), min(len(initial_heights), index + window + 1)):
            local_max = max(initial_heights[window_index], local_max)
        adjusted_heights.append(local_max)
    return adjusted_heights


def block_all_faces(image_file, frames, frame_index, faces):
    """ Draw a black circle over each face detected in the frame. """
    for person in faces:
        face_x_coords, face_y_coords = get_face_coords(frames, frame_index, person)
        center_x, center_y, min_radius = get_smoothed_face_data(frames, frame_index, person)

        image_file = draw_circle_over_points(image_file, face_x_coords, face_y_coords, center_x, center_y, min_radius)

    return image_file


def get_smoothed_face_data(frames, index, person, window=4):
    """
    Determine where and how big to draw a circle to conceal a detected face in a way that is smoothed with adjacent
    frames. Look ahead and behind a number of frames determined by the window parameter and calculate a weighted average
    midpoint, with frames that are further away from the target frame weighted lower.
    """
    start_index = max(0, index - window)
    end_index = min(len(frames), index + window + 1)

    min_radius = 0
    x_sum = 0
    y_sum = 0
    weight_sum = 0
    for i in range(start_index, end_index):
        weight = window ** (window - abs(index - i))
        x_coords, y_coords = get_face_coords(frames, i, person)

        x_sum += sum(x_coords) * weight
        y_sum += sum(y_coords) * weight
        weight_sum += weight * len(x_coords)

        radius = get_radius(x_coords, y_coords)
        if radius > min_radius:
            min_radius = radius

    # This prevents divide by zero error if there were no face keypoints found.
    if weight_sum == 0:
        weight_sum = 1

    center_x = x_sum / weight_sum
    center_y = y_sum / weight_sum

    return int(center_x), int(center_y), int(min_radius)


def get_face_coords(frames, frame_index, person, confidence_cutoff=.25):
    """
    Retrieve the X and Y locations of facial keypoints that were detected with a confidence above the provided logit
    value.
    """
    face_x_coords = []
    face_y_coords = []

    frame = frames[frame_index]
    current_image_path = frame["filepath"]

    if current_image_path not in person:
        # This indicates the person was not detected in this image
        return face_x_coords, face_y_coords

    face = person[current_image_path]

    for part, point in face.items():
        if point["confidence"] > confidence_cutoff:
            face_x_coords.append(point["x"])
            face_y_coords.append(point["y"])
            if "eye" in part:  # Weight eye points higher
                face_x_coords.append(point["x"])
                face_y_coords.append(point["y"])

    return face_x_coords, face_y_coords


def draw_circle_over_points(image_file, x_coords, y_coords, center_point_x, center_point_y, min_radius,
                            buffer_factor=1.0):
    # Dont draw a circle if we dont have points
    if len(x_coords) == 0:
        return image_file

    # Find the distance to the farthest coordinate
    longest_distance = int(max(get_radius(x_coords, y_coords), min_radius) * buffer_factor)

    return cv2.circle(image_file, (center_point_x, center_point_y), longest_distance, (0, 0, 0), thickness=-1)


def get_radius(x_coords, y_coords):
    """ Returns the radius of a circle covering all and centered on the input coordinates """
    center_point_x, center_point_y = get_center(x_coords, y_coords)

    longest_distance = 0
    for i in range(len(x_coords)):
        distance = math.sqrt(
            abs((center_point_x - x_coords[i]) ** 2 + (center_point_y - y_coords[i]) ** 2))
        if distance > longest_distance:
            longest_distance = distance

    return longest_distance


def get_center(x_coords, y_coords):
    """ Returns the average center of the points """
    if len(x_coords) == 0 or len(y_coords) == 0:
        return 0, 0
    center_point_x = sum(x_coords) / len(x_coords)
    center_point_y = sum(y_coords) / len(y_coords)

    return center_point_x, center_point_y


def blur_image(image_file, blur_level):
    blur_factor = pow(blur_level, 1.6) * configuration["options"]["blurFactor"]
    image_dimensions = np.shape(image_file)
    kernel_size = int(blur_factor * (image_dimensions[0] + image_dimensions[1]))
    return cv2.blur(image_file, (kernel_size, kernel_size))


def black_background(image_file):
    image_dimensions = np.shape(image_file)
    return np.zeros((image_dimensions[0], image_dimensions[1], 3))


def sum_likelihoods(frame, joint1, joint2):
    """ Sums likelihoods of two joints together. If distance between points is less than 10, returns -10 """
    a = dict_to_array(frame[joint1])
    b = dict_to_array(frame[joint2])
    distance = np.linalg.norm(a[:2] - b[:2])

    # If distance between two vectors is too small, then resulting vector isn't
    # too helpful in 2D, so we shouldn't use it
    if distance < 10:
        return -10

    return float(a[2]) + float(b[2])


def get_face_points(frame):
    """ Finds face point to use"""
    both_ear_likelihood = sum_likelihoods(frame, "leftEar", "rightEar")
    both_eyes_likelihood = sum_likelihoods(frame, "leftEye", "rightEye")
    left_ear_eye_likelihood = sum_likelihoods(frame, "leftEye", "leftEar")
    right_ear_eye_likelihood = sum_likelihoods(frame, "rightEye", "rightEar")
    likelihoods = np.array([both_ear_likelihood, both_eyes_likelihood,
                            left_ear_eye_likelihood, right_ear_eye_likelihood])
    points = ["ears", "eyes", "left-ear-eye", "right-ear-eye"]

    return points[int(np.argmax(likelihoods))]


def rotate_vector(vector, angle):
    """ Rotates vector counterclockwise in pixel coordinates by angle in degrees. """
    cos = math.cos(angle * math.pi / 180)
    sin = math.sin(angle * math.pi / 180)
    rotation_matrix = np.array([[cos, -sin], [sin, cos]])
    return np.dot(vector, rotation_matrix)


def get_head_segment(frame):
    points = frame["points"]

    left_eye = (points["leftEye"]['x'], points["leftEye"]['y'])
    right_eye = (points["rightEye"]['x'], points["rightEye"]['y'])
    left_ear = (points["leftEar"]['x'], points["leftEar"]['y'])
    right_ear = (points["rightEar"]['x'], points["rightEar"]['y'])
    neck = (points["neck"]['x'], points["neck"]['y'])

    # Determine head height
    ears = np.array([right_ear[0] - left_ear[0], right_ear[1] - left_ear[1]])
    face_diagonal_1 = np.array([right_eye[0] - left_ear[0], right_eye[1] - left_ear[1]])
    face_diagonal_2 = np.array([left_eye[0] - right_ear[0], left_eye[1] - right_ear[1]])
    head_height = max(np.linalg.norm(face_diagonal_1) / 2, np.linalg.norm(face_diagonal_2) / 2,
                      np.linalg.norm(ears) / 2)

    face_point = get_face_points(frame["points"])

    correction_angle = 15
    # Determine Anchor point
    if face_point == "left-ear-eye":
        head_anchor = get_coords(frame, "leftEar")
        confidence = points["leftEye"]["confidence"] * points["leftEar"]["confidence"]

        # vector from left ear to left eye
        ear_to_eye = [left_eye[0] - left_ear[0], left_eye[1] - left_ear[1]]
        # vector perpendicular to ear_to_eye, then rotated 15 degrees counterclockwise
        head_pose_direction = rotate_vector(ear_to_eye, correction_angle - 90)

    elif face_point == "right-ear-eye":
        head_anchor = get_coords(frame, "rightEar")
        confidence = points["rightEye"]["confidence"] * points["rightEar"]["confidence"]

        # vector from right ear to right eye
        ear_to_eye = [right_eye[0] - right_ear[0], right_eye[1] - right_ear[1]]
        # vector perpendicular to ear_to_eye, then rotated 15 degrees clockwise
        head_pose_direction = rotate_vector(ear_to_eye, 90 - correction_angle)

    elif face_point == "eyes":
        # Even though we used the eyes to determine direction, we still use the ears' midpoint
        head_anchor = estimate_midpoint(frame, "Ear")
        confidence = points["leftEye"]["confidence"] * points["rightEye"]["confidence"]
        # vector perpendicular to vector between eyes
        head_pose_direction = np.array([left_eye[1] - right_eye[1], right_eye[0] - left_eye[0]])

    else:
        head_anchor = estimate_midpoint(frame, "Ear")
        confidence = points["leftEar"]["confidence"] * points["rightEar"]["confidence"]
        # vector perpendicular to vector between ears
        head_pose_direction = np.array([left_ear[1] - right_ear[1], right_ear[0] - left_ear[0]])

    # ensure upperHead is in direction away from neck
    neck_to_anchor = np.array([head_anchor[0] - neck[0], head_anchor[1] - neck[1]])
    if np.dot(neck_to_anchor, head_pose_direction) < 0:
        head_pose_direction = -head_pose_direction

    head_pose_norm = np.linalg.norm(head_pose_direction)
    if head_pose_norm == 0:
        head_pose_norm = 1
        head_pose_direction = np.array([0, -1])
    head_pose_unit_vector = head_pose_direction / head_pose_norm

    # Get adjusted head segment
    lower_head = {
        'x': int(round(head_anchor[0] - head_height * head_pose_unit_vector[0])),
        'y': int(round(head_anchor[1] - head_height * head_pose_unit_vector[1])),
        "confidence": confidence
    }
    upper_head = {
        'x': int(round(head_anchor[0] + head_height * head_pose_unit_vector[0])),
        'y': int(round(head_anchor[1] + head_height * head_pose_unit_vector[1])),
        "confidence": confidence
    }

    return lower_head, upper_head


def draw_segment(image_file, frame, segment_name, segment_config,configuration=None,skeleton_colors=None):
    """ Draw a segment with corresponding risk color determined by the joint's risk angle thresholds """

    # Scale segment width to video height in pixels
    segment_thickness = int(max(np.shape(image_file)[0] / configuration["options"]["segmentWidthFactor"], 2))

    if "imaginary" in segment_config:
        return image_file,None

    risk_color = get_risk_color(frame, segment_config,skeleton_colors=skeleton_colors)


    # Do not draw leg segments if configured as such
    if configuration["noLegs"]:
        if "Calf" in segment_name or "Thigh" in segment_name:
            return image_file,risk_color

    if (not check_segment_confidence(frame, segment_config)):
        return image_file,risk_color

    if (not valid_angle(frame,segment_config,segment_name)):
        return image_file,risk_color



    # risk_color = get_risk_color(frame, segment_config)




    # if risk_color == [0,0,0]:
    #     return image_file

    return draw_line_from_joints(image_file, frame, *segment_config["adjacent_joints"],
                                 color=risk_color, thickness=segment_thickness),risk_color


def draw_text(img_file, text, position=(30, 30), font_scale=0.6, font_thickness=1, font_color=(255, 0, 0)):
    """ Draw text somewhere on an image """
    return cv2.putText(img_file, str(text), position, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale,
                       thickness=font_thickness, color=font_color)

def get_worst_angle_values(angle_list,joint):
    if 'Elbow' in joint:
        worst_angle = min([i for i in angle_list if not i == -1] )
    else:
        worst_angle = max(angle_list)
    worst_angle_idx = angle_list.index(worst_angle)
    worst_angle_time_stamp = (worst_angle_idx+1)/configuration["fps"]
    i_str = str(worst_angle_idx+1)
    file_name = "img" + "0" * (5 - len(i_str)) + i_str + ".jpg"
    image_path_to_be_copied = os.path.join(configuration["imageOutput"], str(file_name))
    if "Wrist" in joint:
        new_joint_name = joint.replace("Wrist","Hand")
        shutil.copy(image_path_to_be_copied,"intermediate/{}/{}_{}.jpg".format(configuration["dataset"],configuration["dataset"],new_joint_name))
    else:
        shutil.copy(image_path_to_be_copied,"intermediate/{}/{}_{}.jpg".format(configuration["dataset"],configuration["dataset"],joint))

    return worst_angle,worst_angle_idx,worst_angle_time_stamp


def countInRange_single(arr, n, x, y):
    # initialize result
    count = 0

    for i in range(n):

        # check if element is in range
        if (arr[i] >= x and arr[i] <= y):
            count += 1
    return count

def add_legs_to_json(full_data,left_knee_angles,right_knee_angles,json_data):
    leg_data ={'leg_angles':[],'knee_name':[]}
    # leg_angles = []
    # knee_name = []
    for angle_index,left_knee_angle in enumerate(left_knee_angles):
        right_knee_angle = right_knee_angles[angle_index]
        if left_knee_angle == -1:
            if right_knee_angle == -1:
                leg_data['leg_angles'].append(left_knee_angle)
                leg_data['knee_name'].append("left")
            else:
                leg_data['leg_angles'].append(right_knee_angle)
                leg_data['knee_name'].append("right")
        elif right_knee_angle == -1:
            leg_data['leg_angles'].append(left_knee_angle)
            leg_data['knee_name'].append("left")
        elif left_knee_angle <= right_knee_angle:
            leg_data['leg_angles'].append(left_knee_angle)
            leg_data['knee_name'].append("left")
        else:
            leg_data['leg_angles'].append(right_knee_angle)
            leg_data['knee_name'].append("right")
    worst_leg_angle,worst_leg_angle_idx,worst_leg_angle_time_stamp = get_worst_angle_values(leg_data['leg_angles'],"Legs")
    freq_score,dur_score,pos_score,all_frames_pos_score = cal_duration_freq(full_data,leg_data,"Legs",angle_thresholds_pos_1_2d,angle_thresholds_pos_2_2d,angle_thresholds_pos_3_2d,angle_thresholds_pos_imp_2d,
                                                                                     angle_thresholds_pos_1_3d,angle_thresholds_pos_2_3d,angle_thresholds_pos_3_3d,angle_thresholds_pos_imp_3d,
                                                                            fps=configuration["fps"],seated=configuration["seated"],no_legs=configuration["noLegs"],data2D=configuration["data2D"])
    # for segment_name, segment in segments.items():
    #     if segment_name == "leftThigh":
    leg_joint = {"name": "Legs",
                 "worstFrame": {"timestamp":str(worst_leg_angle_time_stamp)+' secs', "angle": worst_leg_angle},
                 "scores": {"duration": dur_score, "frequency": freq_score, "posture": pos_score},
                 "all_frames_posture_scores": all_frames_pos_score,
                 "thresholds": joint_metadata["leftKnee"]["legacyThresholds"],
                 "fps": configuration["fps"],
                 "angles": leg_data['leg_angles'],
                 "thresholdConfig": segments["leftThigh"]["thresholds"]}
    leg_joint["thresholdConfig"]["joint"] = "Legs"
    json_data["processedData"]["joints"].append(leg_joint)
    return all_frames_pos_score
    # return json_data

# function to count elements within given range
def countInRange(arr, array_threshold, target_joint):
    count_list = []
    for k,v in array_threshold.items():
        if v:
            if 'Wrist' in target_joint:
                for i in range(len(arr[k])):
                    if (abs(arr[k][i]) >= v[0] and abs(arr[k][i]) <= v[1]):
                        count_list.append(i)
            else:
                if len(v) == 4:
                    for i in range(len(arr[k])):
                        if (arr[k][i] >= v[0] and arr[k][i] <= v[1]) or (arr[k][i] >= v[2] and arr[k][i] <= v[3]) :
                            count_list.append(i)
                if len(v) == 2:
                    for i in range(len(arr[k])):
                        if (arr[k][i] >= v[0] and arr[k][i] <= v[1]) :
                            count_list.append(i)


    count = len(list(set(count_list)))
    return count

# def freq_count(arr, array_threshold_pos, target_joint):
#     '''This function calculates the posture scores for individual thresholds and then consolidates a list of posture scores for each of the movement (flexion,abduction,rotation)'''
#     count_list = {}
#     for pos_score,pos_score_val in array_threshold_pos.items():
#         for k,v in pos_score_val.items():
#             if v:
#                 if k not in count_list:
#                     count_list[k] = {}
#                 count_list[k][pos_score] = []
#                 if 'Wrist' in target_joint:
#                     for i in range(len(arr[k])):
#                         if (abs(arr[k][i]) >= v[0] and abs(arr[k][i]) <= v[1]):
#                             count_list[k][pos_score].append(i)
#                 elif 'Legs' in target_joint:
#                     for i in range(len(arr)):
#                             if (arr[i] >= v[0] and arr[i] <= v[1]) :
#                                 count_list[k][pos_score].append(i)
#                 else:
#                     if len(v) == 4:
#                         for i in range(len(arr[k])):
#                             if (arr[k][i] >= v[0] and arr[k][i] <= v[1]) or (arr[k][i] >= v[2] and arr[k][i] <= v[3]) :
#                                 count_list[k][pos_score].append(i)
#                     if len(v) == 2:
#                         for i in range(len(arr[k])):
#                             if (arr[k][i] >= v[0] and arr[k][i] <= v[1]) :
#                                 count_list[k][pos_score].append(i)
#
#
#     posture_scores = {}
#     for k in count_list.keys():
#         posture_scores[k] = []
#
#     if 'Legs' in target_joint:
#         end_idx = len(arr)
#     else:
#         end_idx = len(arr['flexion'])
#
#     for i in range(end_idx):
#         for k in count_list.keys():
#             posture_scores[k].append(0)
#             for pos_score,pos_score_val in count_list[k].items():
#                 if i in pos_score_val:
#                     posture_scores[k][i]=int(pos_score)
#
#     return posture_scores

def freq_count(full_data, arr, array_threshold_pos, target_joint,angles,data2D):
    '''This function calculates the posture scores for individual thresholds and then consolidates a list of posture scores for each of the movement (flexion,abduction,rotation)'''
    count_list = {}
    for i in range(len(full_data['frames'])):
        if target_joint == 'Legs':
            dimension = full_data['frames'][i]['points'][angles['knee_name'][i]+'Knee']['dim']
        else:
            dimension = full_data['frames'][i]['points'][target_joint]['dim']
        thresholds = array_threshold_pos[dimension]
        for pos_score,pos_score_val in thresholds.items():
            for k,v in pos_score_val.items():
                if target_joint == 'back':
                    if dimension == '2d':
                        if k != 'flexion':
                            if k not in count_list:
                                count_list[k] = {}
                            if pos_score not in count_list[k]:
                                count_list[k][pos_score] = []
                            if pos_score == 4:
                                count_list[k][pos_score].append(i)
                if v:
                    if k not in count_list:
                        count_list[k] = {}
                    if pos_score not in count_list[k]:
                        count_list[k][pos_score] = []
                    if 'Legs' in target_joint:
                        target_value = arr[i]
                    else:
                        target_value = arr[k][i]
                    if 'Wrist' in target_joint:
                        # for i in range(len(arr[k])):
                        if (abs(arr[k][i]) >= v[0] and abs(arr[k][i]) <= v[1]):
                            count_list[k][pos_score].append(i)
                    elif 'Legs' in target_joint: # We have it separately here for Legs due to the way the data is structured. Legs uses arr[i] instead of arr[k][i]
                        if len(v) == 4:
                            # for i in range(len(arr)):
                            if (arr[i] >= v[0] and arr[i] <= v[1]) or (arr[i] >= v[2] and arr[i] <= v[3]) :
                                count_list[k][pos_score].append(i)
                        if len(v) == 2:
                            # for i in range(len(arr)):
                            if (arr[i] >= v[0] and arr[i] <= v[1]) :
                                count_list[k][pos_score].append(i)
                    else:
                        if len(v) == 4:
                            # for i in range(len(arr[k])):
                            if (arr[k][i] >= v[0] and arr[k][i] <= v[1]) or (arr[k][i] >= v[2] and arr[k][i] <= v[3]) :
                                if 'Elbow' in target_joint:
                                    side = target_joint.split('Elbow')[0]
                                    if full_data['frames'][i]['points'][side+'Shoulder']['flexion'] > 45:
                                        count_list[k][pos_score].append(i)
                                else:
                                    count_list[k][pos_score].append(i)


                        if len(v) == 2:
                            # for i in range(len(arr[k])):
                            if (arr[k][i] >= v[0] and arr[k][i] <= v[1]) :
                                if 'Elbow' in target_joint:
                                    side = target_joint.split('Elbow')[0]
                                    if full_data['frames'][i]['points'][side+'Shoulder']['flexion'] > 45:
                                        count_list[k][pos_score].append(i)
                                else:
                                    count_list[k][pos_score].append(i)



    for i in range(len(full_data['frames'])): # Skip frames where the person is not detected or the confidence score of all the keypoints is less than 0.3
        if np.all(data2D[i,:,2] < 0.3):
            for k in count_list.keys():
                for pos_score,pos_score_val in count_list[k].items():
                    if i in pos_score_val:
                        count_list[k][pos_score].remove(i)

    # if count_list == {}:    ## This is to handle the case when there are no humans detected and confidence score is less than 0.3
    #     count_list = {'flexion':{1:[]},'abduction':{1:[]},'rotation':{1:[]}}


    posture_scores = {}
    for k in count_list.keys():
        posture_scores[k] = []

    if 'Legs' in target_joint:
        end_idx = len(arr)
    else:
        end_idx = len(arr['flexion'])

    for i in range(end_idx):
        for k in count_list.keys():
            posture_scores[k].append(0)
            for pos_score,pos_score_val in count_list[k].items():
                if i in pos_score_val:
                    posture_scores[k][i]=int(pos_score)

    return posture_scores

def cal_awkward_postures(posture_scores,joint):
    awkward_postures = {}
    final_posture_score = []
    for i in range(len(posture_scores['flexion'])):
        temp_list = []
        for k,v in posture_scores.items():
            temp_list.append(v[i])
        temp_list.sort()
        if len(temp_list) > 1 and temp_list[-1] == 4:
            if 'Shoulder' in joint:
                final_posture_score.append(temp_list[-1])
            else:
                final_posture_score.append(temp_list[-2])
        else:
            final_posture_score.append(temp_list[-1])
    awkward_postures = {x:final_posture_score.count(x) for x in [1,2,3]}
    return awkward_postures,final_posture_score



def count_occurences(posture_scores,window):
    occurences = {}
    for k,v in posture_scores.items():
        v = [i for i in v if i != 4]
        occurences[k] = 0
        index = 0
        occurence_flag = False

        while (index < len(v)-1):
                if v[index] > 0:
                    for i in range(1,len(v)-index):
                        index += 1
                        if i == window and occurence_flag == False:
                            occurences[k] += 1
                            occurence_flag = True
                        if v[index] > 0:
                            continue
                        else:
                            break
                else:
                    for i in range(1,len(v)-index):
                        index += 1
                        if i == window and occurence_flag == True:
                            occurence_flag = False
                        if v[index] == 0:
                            continue
                        else:
                            break

    return occurences

def cal_duration_freq(full_data, angles, joint, angle_thresholds_pos_1_2d, angle_thresholds_pos_2_2d, angle_thresholds_pos_3_2d,angle_thresholds_pos_imp_2d,
                      angle_thresholds_pos_1_3d, angle_thresholds_pos_2_3d, angle_thresholds_pos_3_3d,angle_thresholds_pos_imp_3d,fps=50,seated=False,no_legs=False,data2D=None):
    if joint == "Legs":
        angle_of_interest = np.array(angles['leg_angles'])
        num_frames = len(angle_of_interest)

        # awkward_posture = {1: countInRange_single(angle_of_interest, len(angle_of_interest), angle_thresholds_pos_1[joint]["flexion"][0],
        #                                angle_thresholds_pos_1[joint]["flexion"][1]),
        #                2: countInRange_single(angle_of_interest, len(angle_of_interest), angle_thresholds_pos_2[joint]["flexion"][0],
        #                                angle_thresholds_pos_2[joint]["flexion"][1]),
        #                3: countInRange_single(angle_of_interest, len(angle_of_interest), angle_thresholds_pos_3[joint]["flexion"][0],
        #                                angle_thresholds_pos_3[joint]["flexion"][1])}
    else:
        angle_of_interest ={}
        angle_of_interest['flexion'] = angles.flexion*(180/np.pi) if np.any(angles.flexion) else None
        angle_of_interest['abduction'] = angles.abduction*(180/np.pi) if np.any(angles.abduction) else None
        angle_of_interest['rotation'] = angles.rotation*(180/np.pi) if np.any(angles.rotation) else None

        num_frames = (angle_of_interest['flexion'].shape[0])

        # awkward_posture = {1: countInRange(angle_of_interest, angle_thresholds_pos_1[joint],joint),
        #                    2: countInRange(angle_of_interest, angle_thresholds_pos_2[joint],joint),
        #                    3: countInRange(angle_of_interest, angle_thresholds_pos_3[joint],joint)}
    angle_thresholds_pos = {
        '2d':{1:angle_thresholds_pos_1_2d[joint],
                         2:angle_thresholds_pos_2_2d[joint],
                         3: angle_thresholds_pos_3_2d[joint],
                        4: angle_thresholds_pos_imp_2d[joint]},
        '3d':{1:angle_thresholds_pos_1_3d[joint],
                         2:angle_thresholds_pos_2_3d[joint],
                         3: angle_thresholds_pos_3_3d[joint],
                        4: angle_thresholds_pos_imp_3d[joint]}
    }
    if seated:
        if joint == 'back':
            angle_thresholds_pos['seated'] = {1:{"flexion":[30,59,-39,-15],"abduction":[15,29,-29,-15],"rotation":[30,44,-44,-30]},
                                              2:{"flexion":[60,89,-59,-40],"abduction":[30,44,-44,-30],"rotation":[45,89,-89,-45]},
                                              3:{"flexion":[90,119,-180,-60],"abduction":[45,59,-59,-45],"rotation":[90,109,-109,-90]},
                                              4:{"flexion":[120,180],"abduction":[60,180,-180,-60],"rotation":[110,180,-180,-110]}}

        if not no_legs:
            if joint == 'Legs':
                angle_thresholds_pos['seated'] = {1:{"flexion":[95,105],"abduction":None,"rotation":None},
                                                  2:{"flexion":[105,180],"abduction":None,"rotation":None},
                                                  3:{"flexion":None,"abduction":None,"rotation":None},
                                                  4:{"flexion":[-180,-10],"abduction":None,"rotation":None}}

    posture_scores = freq_count(full_data, angle_of_interest,angle_thresholds_pos,joint,angles,data2D)
    awkward_posture,all_frames_posture_scores = cal_awkward_postures(posture_scores,joint)
    window = 5
    if 'Wrist' in joint:
        window = 3
    final_occurences = count_occurences(posture_scores,window=window)
    Total_frequency = sum(final_occurences.values())
    freq = int((Total_frequency * 60) / (num_frames / fps))
    if 'Wrist' not in joint:
        freq_score = 1 if freq > 3 else 0
    else:
        freq_score = 1 if freq > 30 else 0

    temp_post_list = []
    duration_tot = 0
    for k, v in awkward_posture.items():
        if v != 0:
            temp_post_list.append(k)
    posture_score = max(temp_post_list) if temp_post_list else 0
    duration_tot = awkward_posture[2] + awkward_posture[3]
    if posture_score != 0:
        # if len(temp_post_list) == 3:
        #     duration_tot = awkward_posture[2] + awkward_posture[3]
        # else:
        #     for k in temp_post_list:
        #         duration_tot += awkward_posture[k]
        if 10 <= duration_tot * 100 / num_frames <= 19:
            duration_score = 1
        elif 20 <= duration_tot * 100 / num_frames <= 29:
            duration_score = 2
        elif duration_tot * 100 / num_frames >= 30:
            duration_score = 3
        else:
            duration_score = 0

    else:
        duration_score = 0

    return freq_score, duration_score, posture_score,all_frames_posture_scores  # individual for each body joint (e.g., back), but for the same video, only 3d, forget about seated

def add_posture_scores(all_frames_posture_scores,posture_scores):
    result = []
    for x, y in zip(all_frames_posture_scores, posture_scores):
        if y != 4:
            result.append(x + y)
        else:
            result.append(x)
    return result

# noinspection PyTypeChecker
def get_json_data(data):
    """ Transform data into an object representing the correct JSON output object """
    json_data = {"processedData": {"id": configuration["dataset"],
                                   "joints": [],
                                   "headDirections": [],
                                   "colors": skeleton_colors,"PostureRiskScore": []}
                 }

    all_joint_angles = data["angles"]
    hand_vertical = []
    hand_horizontal = []
    hand_angles = {}
    wrist_locations = []
    pixel_heights = data["metadata"]["pixel_heights"]
    pixel_height = 1
    if len(data["metadata"]["pixel_heights"]) > 0:
        pixel_height = max(max(data["metadata"]["pixel_heights"]), 1)

    joint_angles = {}
    for joint in vicon_gt_matrix:
        joint_angles[joint] = []


    joints_three_angles = {}
    joints_three_angles["back"] = {}
    # joints_three_angles["neck"] = {}

    for k in joints_three_angles.keys():
        joints_three_angles[k]["flexion"] = []
        joints_three_angles[k]["abduction"] = []
        joints_three_angles[k]["rotation"] = []

    joints_two_angles = {}
    joints_two_angles["leftShoulder"] = {}
    joints_two_angles["rightShoulder"] = {}
    # joints_two_angles["leftWrist"] = {}
    # joints_two_angles["rightWrist"] = {}

    for k in joints_two_angles.keys():
        joints_two_angles[k]["flexion"] = []
        joints_two_angles[k]["abduction"] = []



    for index, frame in enumerate(data["frames"]):
        json_data["processedData"]["headDirections"].append(frame["headDirection"])

        if frame["hands"]["vertical"] != -1 and pixel_heights[index] != -1:
            hand_vertical.append(frame["hands"]["vertical"] / pixel_heights[index])
        else:
            hand_vertical.append(-1)
        if frame["hands"]["horizontal"] != -1 and pixel_heights[index] != -1:
            hand_horizontal.append(frame["hands"]["horizontal"] / pixel_heights[index])
        else:
            hand_horizontal.append(-1)

        wrist_locations.append(frame["wrists"])

        for joint in vicon_gt_matrix:
            joint_data = frame["points"][joint]
            if ("flexion" not in joint_data) and ("abduction" not in joint_data) and ("rotation" not in joint_data) :
                continue
            # if joint in ['back','neck']:
            if joint in ['back']:
                joints_three_angles[joint]["flexion"].append(joint_data["flexion"])
                joints_three_angles[joint]["abduction"].append(joint_data["abduction"])
                joints_three_angles[joint]["rotation"].append(joint_data["rotation"])
            # elif 'Shoulder' in joint or 'Wrist' in joint:
            elif 'Shoulder' in joint:
                joints_two_angles[joint]["flexion"].append(joint_data["flexion"])
                joints_two_angles[joint]["abduction"].append(joint_data["abduction"])
            else:
                joint_angles[joint].append(joint_data["flexion"])

    subjoints ={}
    subjoints['back'] = []
    # subjoints['neck'] = []
    subjoints['leftShoulder'] = []
    subjoints['rightShoulder'] = []
    # subjoints['leftWrist'] = []
    # subjoints['rightWrist'] = []

    for subjoint_keys in subjoints.keys():
        subjoint_mtdata = subjoint_metadata[subjoint_keys]
        for k,v in subjoint_mtdata["name"].items():
            # if subjoint_keys in ["back","neck"]:
            if subjoint_keys in ["back"]:
                subjoints[subjoint_keys].append({"name":v,"angles":joints_three_angles[subjoint_keys][k]})
            else:
                subjoints[subjoint_keys].append({"name":v,"angles":joints_two_angles[subjoint_keys][k]})


    written_joints = []
    PostureRiskScore = [0]* len(data["frames"])
    for segment_name, segment in segments.items():
        if "thresholds" not in segment:
            continue
        target_joint = segment["thresholds"]["joint"]
        if target_joint in ["none","rightKnee","leftKnee"] or target_joint in written_joints:
            continue
        written_joints.append(target_joint)
        angle_list = joint_angles[target_joint]
        if target_joint in subjoints.keys():
            angle_list = subjoints[target_joint][0]["angles"]
        worst_angle,worst_angle_idx,worst_angle_time_stamp = get_worst_angle_values(angle_list,target_joint)
        freq_score,dur_score,pos_score,all_frames_posture_scores = cal_duration_freq(data,all_joint_angles[target_joint],target_joint,angle_thresholds_pos_1_2d,angle_thresholds_pos_2_2d,angle_thresholds_pos_3_2d,angle_thresholds_pos_imp_2d,
                                                                                     angle_thresholds_pos_1_3d,angle_thresholds_pos_2_3d,angle_thresholds_pos_3_3d,angle_thresholds_pos_imp_3d,fps=configuration["fps"],seated=configuration["seated"],no_legs=configuration["noLegs"],data2D=configuration["data2D"])
        PostureRiskScore = add_posture_scores(PostureRiskScore,all_frames_posture_scores)
        # "img" + "0" * (5 - len(i_str)) + i_str + ".jpg",
        # print(target_joint)
        if target_joint in subjoints.keys():
            joint = {"name": joint_metadata[target_joint]["name"], "subjoints": subjoints[target_joint],
                 "worstFrame": {"timestamp":str(worst_angle_time_stamp)+' secs', "angle": worst_angle},
                 "scores": {"duration": dur_score, "frequency": freq_score, "posture": pos_score},
                 "all_frames_posture_scores": all_frames_posture_scores,
                 "thresholds": joint_metadata[target_joint]["legacyThresholds"],
                 "fps": configuration["fps"],
                 "angles": joint_angles[target_joint],
                 "thresholdConfig": segment["thresholds"]}
            json_data["processedData"]["joints"].append(joint)
        else:

            joint = {"name": joint_metadata[target_joint]["name"],
                     "worstFrame": {"timestamp":str(worst_angle_time_stamp)+' secs', "angle": worst_angle},
                     "scores": {"duration": dur_score, "frequency": freq_score, "posture": pos_score},
                     "all_frames_posture_scores": all_frames_posture_scores,
                     "thresholds": joint_metadata[target_joint]["legacyThresholds"],
                     "fps": configuration["fps"],
                     "angles": joint_angles[target_joint],
                     "thresholdConfig": segment["thresholds"]}
            json_data["processedData"]["joints"].append(joint)

        pass
        # MAE and MSE, and mediumn absolute error to get score difference
        # freq_score --> % of frame
            # in vicon test dataset --> each video have 1 posture, 1 duration and 1 frequency score --> also do it from GT, --> calcualte error, 1 error per video per metric --> MAE and MSE, and mediumn absolute error
            # historgram of all errors, x axis is video

    left_knee_angles  = joint_angles["leftKnee"]
    right_knee_angles = joint_angles["rightKnee"]
    legs_all_frames_pos_score = add_legs_to_json(data,left_knee_angles,right_knee_angles,json_data)
    PostureRiskScore = add_posture_scores(PostureRiskScore,legs_all_frames_pos_score)

    # json_data["processedData"]["hands"] = {
    #     "horizontal": hand_horizontal,
    #     "vertical": hand_vertical
    # }
    # json_data["processedData"]["hands"].update(hand_angles)
    json_data["processedData"]["PostureRiskScore"] = PostureRiskScore
    if configuration["client"] == "kinetica":
        # Experimental data goes here to avoid polluting HT data file
        json_data["processedData"]["pixelSubjectHeight"] = pixel_height
        json_data["processedData"]["pixelSubjectHeights"] = pixel_heights
        json_data["processedData"]["wristLocations"] = wrist_locations

    return json_data


def get_coords(frame, point):
    """ Get the x and y coordinates for a particular point """
    # spine_vertical is a special imaginary point above the spine's base
    if point == "verticalSpine":
        coord = get_coords(frame, "back")
        coord[1] = 0
        return coord
    else:
        joint = frame["points"][point]
        if "z" in joint:
            return [joint["x"], joint["y"], joint["z"]]
        return [joint["x"], joint["y"]]


def get_segment_joints(joint_segments):
    joint_list = []
    for segment in joint_segments:
        for joint in segments[segment]["adjacent_joints"]:
            joint_list.append(joint)

    return joint_list


def get_vector(frame, segment):
    """ For a particular segment, create a vector from one point to the other """
    if segment not in segments:
        return None

    point1, point2 = segments[segment]["adjacent_joints"]

    arr1 = np.array(get_coords(frame, point1))
    arr2 = np.array(get_coords(frame, point2))

    return arr1 - arr2


def to_degrees(radians):
    """ Convert radians... to degrees """
    return radians * radians_to_degrees


def estimate_midpoint(frame, partial_part):
    part1 = "left" + partial_part
    part2 = "right" + partial_part

    x1, y1 = get_coords(frame, part1)
    x2, y2 = get_coords(frame, part2)

    conf_1 = check_point_confidence(frame, part1)
    conf_2 = check_point_confidence(frame, part2)

    if conf_1:
        if not conf_2:
            x2, y2 = x1, y1
    elif conf_2:
        x1, y1 = x2, y2
    else:
        # neither has confidence, midpoint cannot be estimated
        x1, y1, x2, y2 = -1, -1, -1, -1

    return (x1 + x2) // 2, (y1 + y2) // 2


def get_segment_length(frame, segment):
    joint1, joint2 = segments[segment]["adjacent_joints"]

    if not check_point_confidence(frame, joint1) or not check_point_confidence(frame, joint2):
        return -1

    x1, y1 = get_coords(frame, joint1)
    x2, y2 = get_coords(frame, joint2)

    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_color_from_ranges(angle, ranges,joint,skeleton_colors):
    final_color = ['default']
    for color, angle_range in ranges.items():
        if isinstance(angle_range, list):
            if evaluate_range_single(angle['flexion'],angle_range):
                final_color.append(color)
        else:
            if evaluate_range(angle, angle_range,joint):
                final_color.append(color)
    # if 'Shoulder' in joint:
    #     joint_color = final_color[-1]
    if 'back' in joint:
        for i in range(len(final_color)):
            if final_color[-(i+1)] != "impossible":
                joint_color = final_color[-(i+1)]
                break
            else:
                joint_color = "impossible"
    else:
        joint_color = final_color[-1]

    return skeleton_colors[joint_color]

    # return skeleton_colors["default"]


def estimate_back_length(frame):
    right = get_segment_length(frame, "rightSide")
    if right < 0:
        return -1
    left = get_segment_length(frame, "leftSide")
    if left < 0:
        return -1

    return (right + left) / 2


def get_risk_color(frame, segment,skeleton_colors):
    """ Determine the color a segment should be based on the corresponding angle and threshold"""
    if "thresholds" in segment:
        thresholds = segment["thresholds"]
    else:
        return skeleton_colors["default"]

    if "none" == thresholds["joint"]:
        return skeleton_colors["default"]

    target_joint = thresholds["joint"]
    target_angle ={}
    target_angle["flexion"] = frame["points"][target_joint]["flexion"]
    target_angle["abduction"] = frame["points"][target_joint]["abduction"]
    target_angle["rotation"] = frame["points"][target_joint]["rotation"]
    angle_dim = frame["points"][target_joint]["dim"]

    if "if" in thresholds:
        if_statement = thresholds["if"]
        else_statement = thresholds["else"]
        if_angle = frame["points"][if_statement["joint"]]["flexion"]
        if evaluate_range_single(if_angle, if_statement["range"]):
            return get_color_from_ranges(target_angle, if_statement["then"][angle_dim],joint=target_joint,skeleton_colors=skeleton_colors)
        else:
            return get_color_from_ranges(target_angle, else_statement,joint=target_joint,skeleton_colors=skeleton_colors)
    else:
        return get_color_from_ranges(target_angle, thresholds["ranges"][angle_dim],joint=target_joint,skeleton_colors=skeleton_colors)


def draw_line(img_file, coord1, coord2, color=(0, 255, 0), thickness=10):
    """
    Draw a line between two body parts on an image

    Args:
        img_file: cv2 image to be modified
        coord1: start coordinate (x,y)
        coord2: end coordinate (x,y)
        color: RGB color of line
        thickness: thickness of line in pixels
    """
    return cv2.line(img_file, tuple(coord1), tuple(coord2), color, thickness)


def draw_line_from_joints(img_file, frame, joint1, joint2, color=(0, 255, 0), thickness=10):
    return draw_line(img_file, get_coords(frame, joint1), get_coords(frame, joint2), color, thickness)


def check_segment_confidence(frame, segment):
    for part in segment["adjacent_joints"]:
        if not check_point_confidence(frame, part):
            return False

    return True

def check_confidence_scores_knee(joint,frame):
    knee_segments = joint_metadata[joint]["segments"]
    for segment in knee_segments:
        segment_config = segments[segment]
        if not check_segment_confidence(frame, segment_config):
            return False
    return True




def valid_angle(frame,segment_config,segment):
    if "thresholds" in segment_config:
        thresholds = segment_config["thresholds"]
    else:
        return True

    if "none" == thresholds["joint"]:
        return True

    target_joint = thresholds["joint"]

    if 'Knee' in target_joint:
        if not check_confidence_scores_knee(target_joint,frame):
            return False


    if (frame['points'][target_joint]['dim'] == '2d') and (frame['points'][target_joint]['flexion'] == -1): # This should work only for the wrist
        return False

    return True




def evaluate_range_single(angle, angle_range):
    return angle_range[0] <= angle < angle_range[1]

def evaluate_range(angle, angle_range,joint):
    for movement,range in angle_range.items():
        if range and angle[movement]:
            if "Wrist" in joint:
                if (range[0] <= abs(angle[movement]) < range[1]):
                    return True
            else:
                if len(range) == 4:
                    if (range[0] <= angle[movement] < range[1]) or (range[2] <= angle[movement] < range[3]):
                        return True
                if len(range) == 2:
                    if (range[0] <= angle[movement] < range[1]):
                        return True
    return False

    # return angle_range[0] <= angle < angle_range[1]


def perpendicular_normalized(a, b):
    c = b - a
    distance = np.linalg.norm(c)
    if distance == 0:
        return np.array([0, 1])
    if c[0] < 0:
        return np.array([-c[1] / distance, c[0] / distance])
    return np.array([c[1] / distance, -c[0] / distance])


def find_closest_angle_normalized(vector1, vector2, vector3):
    """ Determine the vector with the smallest angle with respect to a third """
    if vector1 is None or vector2 is None or vector3 is None:
        return None
    unit_vector_1 = vector1 / np.linalg.norm(vector1)
    unit_vector_2 = vector2 / np.linalg.norm(vector2)
    unit_vector_compare = vector3 / np.linalg.norm(vector3)
    distance_1 = np.dot(unit_vector_1, unit_vector_compare)
    distance_2 = np.dot(unit_vector_2, unit_vector_compare)
    angle1 = np.arccos(distance_1)
    angle2 = np.arccos(distance_2)
    if angle1 > angle2:
        return unit_vector_2
    return unit_vector_1


def correct_neck_angle(eye_to_ear_vector, head_pose_vector):
    """ Rotate neck angle to be more vertical """
    correction_angle = 15
    rotation_direction = np.linalg.det(np.transpose(np.array([eye_to_ear_vector, head_pose_vector])))

    cos = math.cos(correction_angle * math.pi / 180)
    sin = math.sin(correction_angle * math.pi / 180)
    rotation_matrix = np.array([[cos, sin], [-sin, cos]])
    if rotation_direction > 0:
        rotation_matrix = np.array([[cos, -sin], [sin, cos]])

    return head_pose_vector.dot(rotation_matrix)


def get_head_data(frame_data, joint1, joint2, factor):
    """ Calculate the head keypoint based on a pair of face keypoints """
    unusable = False
    joint1_pos = dict_to_array(frame_data[joint1])[:2]
    joint2_pos = dict_to_array(frame_data[joint2])[:2]
    mid_shoulder = (dict_to_array(frame_data["leftShoulder"])[:2] + dict_to_array(frame_data["rightShoulder"])[:2]) / 2
    mid_face = (joint1_pos + joint2_pos) / 2
    shoulders_to_head = (mid_face - mid_shoulder)

    head_pose_up = perpendicular_normalized(joint1_pos, joint2_pos)
    head_pose_down = -head_pose_up
    # noinspection PyTypeChecker
    head_pose_vector = find_closest_angle_normalized(head_pose_up, head_pose_down, shoulders_to_head)

    if factor == ear_eye_factor:
        ear_to_eye = joint1_pos - joint2_pos
        head_pose_vector = correct_neck_angle(ear_to_eye, head_pose_vector)

    if head_pose_vector is None:
        mid_hip = (dict_to_array(frame_data["leftHip"])[:2] + dict_to_array(frame_data["rightHip"])[:2]) / 2
        head_pose_vector = perpendicular_normalized(mid_hip, mid_shoulder)
        unusable = True
    if head_pose_vector is None:
        head_pose_vector = np.array([0, -1])
    distance = np.linalg.norm(mid_shoulder - mid_face) * factor
    head_pose_vector *= distance
    head_position = mid_shoulder + head_pose_vector
    head_dict = array_to_dict(head_position)
    if unusable:
        head_dict["confidence"] = -10
    else:
        head_dict["confidence"] = (frame_data[joint1]["confidence"] + frame_data[joint2]["confidence"]) / 2
    return head_dict


def get_location_and_likelihood_data(frame_data, joint):
    """
    Obtains the the location and likelihood data for the given frame.
    frame_data is given in detectron format, joint_id is the noisefilter index.
    """

    # if joint == "bottomHead":
    #     return frame_data["neck"]
    #
    # if joint == "topHead":
    #     # "topHead" is calculated using facial keypoints
    #     both_ear_likelihood = sum_likelihoods(frame_data, "leftEar", "rightEar")
    #     both_eyes_likelihood = sum_likelihoods(frame_data, "leftEye", "rightEye")
    #     left_ear_eye_likelihood = sum_likelihoods(frame_data, "leftEye", "leftEar")
    #     right_ear_eye_likelihood = sum_likelihoods(frame_data, "rightEye", "rightEar")
    #     likelihood_arr = [both_ear_likelihood, both_eyes_likelihood,
    #                       left_ear_eye_likelihood, right_ear_eye_likelihood]
    #     max_sum_likelihood_index = likelihood_arr.index(max(likelihood_arr))
    #     if max_sum_likelihood_index == 0:
    #         # print("frame {}: both ears used".format(x + 1))
    #         return get_head_data(frame_data, "leftEar", "rightEar", ears_factor), "ears"
    #     elif max_sum_likelihood_index == 1:
    #         # print("frame {}: both eyes used".format(x + 1))
    #         return get_head_data(frame_data, "leftEye", "rightEye", eyes_factor), "eyes"
    #     elif max_sum_likelihood_index == 2:
    #         # print("frame {}: left eyes and ears used".format(x + 1))
    #         return get_head_data(frame_data, "leftEye", "leftEar", ear_eye_factor), "left-ear-eye"
    #     else:
    #         # print("frame {}: right eyes and ears used".format(x + 1))
    #         return get_head_data(frame_data, "rightEye", "rightEar", ear_eye_factor), "right-ear-eye"

    return frame_data[joint]


def get_modified_skeleton(input_data):
    """ Convert the uniqye (COCO-style + h36m keypoints) to 27 custom keypoints leaving out nose, left/right foot, left/right toe and adding bottomHead and topHead  """
    output_data = []
    for frame_idx in range(len(input_data)):
        frame_data = input_data[frame_idx]["points"]
        output_data.append({"points": {},
                            "filename": input_data[frame_idx]["filename"],
                            "index": input_data[frame_idx]["index"]})
        for part in vicon_gt_matrix:
            # if part == "topHead":
            #     # top head segment returns a pair from this function and we want to ignore the second member
            #     joint_data, _ = get_location_and_likelihood_data(frame_data, part)
            # else:
            joint_data = get_location_and_likelihood_data(frame_data, part)
            output_data[frame_idx]["points"][part] = joint_data
    return output_data




def interpret_confidence(confidence):
    # translating by 1 keeps lower bound at 0; scaling by 6 empirically makes about 50% of keypoints in range 0-1
    return np.log(confidence * 6 + 1)


def load_keypoint_data_2d():
    """ Load 2D keypoints from custom file format """
    frames = []
    with open(configuration["keypoint2D"]) as keypoint_file:
        frame_index = 0
        while True:
            # pop last char to remove newline
            image_path = keypoint_file.readline()[:-1]
            if image_path == "":
                break
            frame = {"points": {}, "filename": image_path, "index": frame_index}
            frames.append(frame)

            # read lines in order of points in face_matrix and then body_matrix
            for part in vicon_gt_matrix:
                x, y, logit = keypoint_file.readline().split()
                # likelihood = interpret_confidence(float(logit))
                likelihood = float(logit)
                frames[frame_index]["points"][part] = \
                    {"x": int(float(x.strip())), "y": int(float(y.strip())), "confidence": likelihood}

            # for part in h36m_matrix:
            #     x, y, logit = keypoint_file.readline().split()
            #     likelihood = interpret_confidence(float(logit))
            #     if part in frames[frame_index]["points"]:




            frame_index += 1

    return frames


def get_confidence(input_data):
    """ Generates confidence values from 2D keypoint data to be copied to 3D data for any shared keypoint names """
    confidence = []
    for frame_idx in range(len(input_data)):
        frame = input_data[frame_idx]["points"]
        confidence.append({})
        for part in vicon_gt_matrix:
            if part in frame:
                confidence[frame_idx][part] = frame[part]["confidence"]
            else:
                confidence[frame_idx][part] = joint_metadata[part]["tolerance"]
    return confidence


def load_keypoint_data_3d(confidence,config_key):
    """ Load 3D keypoints from custom file format """
    frames = []
    with open(configuration[config_key]) as keypoint_file:
        frame_index = 0
        while True:
            image_path = keypoint_file.readline()[:-1]
            if image_path == "":
                break
            frame = {"points": {}, "filename": image_path, "index": frame_index}
            frames.append(frame)

            for part in vicon_gt_matrix:
                x, y, z, logit = keypoint_file.readline().split()
                likelihood = float(logit)
                frames[frame_index]["points"][part] = \
                    {"x": float(x.strip()), "y": float(y.strip()), "z": float(z.strip())}
                frames[frame_index]["points"][part]["confidence"] = likelihood
                # frames[frame_index]["points"][part]["confidence"] = confidence[frame_index][part]

            frame_index += 1

    return frames


def load_faces_from_file():
    with open(configuration["faceFile"]) as face_file:
        face_data = json.load(face_file)
    return face_data


def plot_histograms(output_data, dataset):
    save_dir = os.path.join('Histograms',dataset)
    os.makedirs(save_dir, exist_ok=True)
    joint_list = ['leftElbow','rightElbow']
    for joint in joint_list:
        if not 'Elbow' in joint:
            data = output_data['angles'][joint].flexion * 180 / np.pi
        else:
            elbow_angle = []
            dim = []
            shoulder_angle = []
            for frame in output_data['frames']:
                if 'left' in joint:
                    side = 'left'
                else:
                    side = 'right'
                shoulder_angle.append(frame['points'][side+'Shoulder']['flexion'])
                elbow_angle.append(frame['points'][joint]['flexion'])
                dim.append(frame['points'][joint]['dim'])
            data = np.array([elbow_angle[i] for i in range(len(elbow_angle)) if (dim[i] == '3d' and shoulder_angle[i] >= 60)])
        plt.figure(figsize=[10,8])
        # fig, ax = plt.subplots()
        data = np.array([i for i in data if i !=-1])
        counts, bins, patches = plt.hist(data, bins=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100], alpha=0.5, color='b',ec='black')
        plt.title('Frequency histogram for '+joint+' for '+dataset+' job')
        plt.xlabel('Angles')
        plt.ylabel('Frequency')
        # ax.set_xticks(bins)
        plt.xticks(bins)
        plt.savefig(os.path.join(save_dir,joint+'.jpg'))
        plt.close()

def rotate_vectors(vectors, axis, theta):
    """ Rotates a set of vectors about an axis in 3D space using Rogrigues' rotation formula """
    vectors = np.array(vectors)
    single = False
    # if 1D vector is passed instead of 2D matrix
    if len(vectors.shape) == 1:
        vectors = np.array([vectors])
        single = True

    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = axis
    # https://mathworld.wolfram.com/RodriguesRotationFormula.html
    r = np.array([
        [c + x ** 2 * (1 - c), x * y * (1 - c) - z * s, y * s + x * z * (1 - c)],
        [z * s + x * y * (1 - c), c + y ** 2 * (1 - c), -x * s + y * z * (1 - c)],
        [-y * s + x * z * (1 - c), x * s + y * z * (1 - c), c + z ** 2 * (1 - c)]
    ])

    # rotate all vectors about axis
    out = np.matmul(vectors, r.transpose())
    if single:
        return out[0]
    return out


def get_hand_angles(frame, side):
    """ Calculates the ulnar/radial deviation, supination/pronation, and flexion/extension of a single hand """
    assert side == "left" or side == "right"

    shoulder_to_elbow = (dict_to_array(frame["points"][side + "Elbow"]) -
                         dict_to_array(frame["points"][side + "Shoulder"]))[:3]
    elbow_to_wrist = (dict_to_array(frame["points"][side + "Wrist"]) -
                      dict_to_array(frame["points"][side + "Elbow"]))[:3]
    wrist_to_palm = (dict_to_array(frame["points"][side + "Palm"]) -
                     dict_to_array(frame["points"][side + "Wrist"]))[:3]
    wrist_to_thumb = (dict_to_array(frame["points"][side + "Thumb"]) -
                      dict_to_array(frame["points"][side + "Wrist"]))[:3]

    # Ulnar / Radial calculation
    # vector representing the plane of the hand
    hand_plane = np.cross(wrist_to_palm, wrist_to_thumb)
    # vector representing the line of intersection between the hand plane and the plane perpendicular to the forearm
    plane_intersection_vector = np.cross(hand_plane, elbow_to_wrist)
    # if hand has full flexion/extension (gimbal lock)
    if np.linalg.norm(plane_intersection_vector) == 0:
        ur_angle = 0
    else:
        ur_angle = get_angle(wrist_to_thumb, plane_intersection_vector)
    # positive means flexion, negative means extension
    if np.dot(wrist_to_thumb, elbow_to_wrist) < 0:
        ur_angle = -ur_angle
    # rotate hand vectors within the plane of the hand to negate ulnar/radial deviation
    wrist_to_palm, wrist_to_thumb = rotate_vectors((wrist_to_palm, wrist_to_thumb), hand_plane, np.radians(ur_angle))

    # Supination / Pronation calculation
    # part of upper arm parallel to forearm
    part_parallel = np.dot(shoulder_to_elbow, elbow_to_wrist) / np.linalg.norm(elbow_to_wrist) ** 2 * elbow_to_wrist
    # vector perpendicular to forearm indicating direction of neutral position
    neutral_dir = shoulder_to_elbow - part_parallel
    # if arm is completely straight
    if np.linalg.norm(neutral_dir) == 0:
        sp_angle = 0
    else:
        sp_angle = 180 - get_angle(wrist_to_thumb, neutral_dir)
    # adjust for left hand
    if side == "left":
        sp_angle = -sp_angle
    # positive means supination, negative means pronation
    if np.dot(hand_plane, neutral_dir) < 0:
        sp_angle = -sp_angle
    # rotate hand vectors around forearm axis to negate supination/pronation
    wrist_to_palm, wrist_to_thumb = rotate_vectors((wrist_to_palm, wrist_to_thumb), elbow_to_wrist,
                                                   -np.radians(sp_angle))

    # Flexion / Extension calculation
    fe_angle = get_angle(wrist_to_palm, elbow_to_wrist)
    # adjust for left hand
    if side == "left":
        fe_angle = -fe_angle
    # positive means flexion, negative means extension
    if np.dot(hand_plane, elbow_to_wrist) < 0:
        fe_angle = -fe_angle

    return {"ulnar": ur_angle, "supination": sp_angle, "flexion": fe_angle}


def main(args):
    # Load in the text data containing file names and coord pairs
    configure(args)
    gen_keypoints_list = args.gen_kps_list
    img_shape = args.img_shape


    init_files()
    ingest_config_file()

    # load 2D and 3D keypoints
    input_data = load_keypoint_data_2d()
    # augment to include spinal keypoints
    input_data = get_modified_skeleton(input_data)
    if configuration["keypoint3D"] != "" and configuration["keypoint3D_MB"] != "" and configuration["keypoint3D_MB_untilted"] != "":
        # copy confidence values from 2D keypoints. Should be removed if 3D confidence can be obtained
        confidence = get_confidence(input_data)
        # input_data_3d = load_keypoint_data_3d(confidence,"keypoint3D")
        input_data_3d_MB = load_keypoint_data_3d(confidence,"keypoint3D_MB")
        input_data_3d_MB_untilted = load_keypoint_data_3d(confidence,"keypoint3D_MB_untilted")
        output_data = get_output_data(input_data,input_data_3d_MB=input_data_3d_MB,data_3d_MB=configuration['data3D_MB'],input_data_3d_MB_untilted=input_data_3d_MB_untilted,data_3d_MB_untilted=configuration['data3D_MB_untilted'],img_shape=img_shape)
    else:
        output_data = get_output_data(input_data)

    # plot_histograms(output_data,args.dataset)
    # plot_wrist_angles(output_data['angles'],args.dataset)
    segment_mapper = draw_skeleton_mp(output_data)
    write_json(output_data)
    return segment_mapper


def call(args):
    segment_mapper = main(parse_args(args))
    return segment_mapper


if __name__ == "__main__":
    main(parse_args())
