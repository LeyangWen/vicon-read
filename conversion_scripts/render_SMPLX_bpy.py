import pickle
import bpy
import argparse
import numpy as np
import time

# In blender script
# Step 1: Load SMPLX model using the addon
# Step 2: Set view to -z (looking down the z-axis)
# Step 3: Run the python original code in your normal venv, it will save a pkl file
# Step 4: Set in_blender to True, Run the commented out script in blender script


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default=r"W:/VEHS/VEHS-7M/Mesh/S01/Activity00_stageii.pkl")
    parser.add_argument('--camera_xcp', type=str, default=r"W:\VEHS\VEHS data collection round 3\processed\S01\FullCollection\Activity00.xcp")
    parser.add_argument('--output_file', type=str, default=r"C:\Users\wenleyan1\Downloads\SMPL_bpy.pkl")
    parser.add_argument('--file_type', type=str, default='mosh_pkl')
    parser.add_argument('--expression', type=str, default='pleasant')
    parser.add_argument('--frame', type=int, default=15913)
    parser.add_argument('--camera_id', type=int, default=3)
    parser.add_argument('--in_blender', type=bool, default=False)
    return parser.parse_args()


def get_expression(args):
    presets = {
        "pleasant": [0, .3, 0, -.892, 0, 0, 0, 0, -1.188, 0, .741, -2.83, 0, -1.48, 0, 0, 0, 0, 0, -.89, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .89, 0, 0, 2.67],
        "happy": [0.9, 0, .741, -2, .27, -.593, -.29, 0, .333, 0, 1.037, -1, 0, .7, .296, 0, 0, -1.037, 0, 0, 0, 1.037, 0, 3],
        "excited": [-.593, .593, .7, -1.55, -.32, -1.186, -.43, -.14, -.26, -.88, 1, -.74, 1, -.593, 0, 0, 0, 0, 0, 0, -.593],
        "sad": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 2, 2, -2, 1, 1.6, 2, 1.6],
        "frustrated": [0, 0, -1.33, 1.63, 0, -1.185, 2.519, 0, 0, -.593, -.444],
        "angry": [0, 0, -2.074, 1.185, 1.63, -1.78, 1.63, .444, .89, .74, -4, 1.63, -1.93, -2.37, -4],
    }
    return presets[args.expression]

def blender_axis_angle(axis_angle):
    return np.array([axis_angle[0], axis_angle[2], -axis_angle[1]])
    # return axis_angle

if __name__ == '__main__':
    args = parse_args()

    if args.in_blender:
        bpy.ops.object.smplx_load_pose(filepath=args.output_file)

        # need to run twice
        screenshot_file = r"C:\Users\wenleyan1\Downloads\smplx_screen"
        screenshot_file = f"{screenshot_file}\\frame_{str(args.frame)}_cam_{str(args.camera_id)}.png"
        bpy.ops.screen.screenshot(filepath=screenshot_file,
                                  hide_props_region=True)
    else:
        import ergo3d as eg
        import xml.etree.ElementTree as ET

        # Step 1: read camera orientation & translation
        vicon_camera_pos_map = {'51470934': "0: +x, +y, +z, back angle view",
                                '66920731': "1: -x, 0y, +++z, right side view",
                                '66920734': "2: 0x, -y, 0z, frontal view",
                                '66920758': "3: -x, -y, ++z, front angle view"}
        cameras = eg.batch_load_from_xcp(args.camera_xcp)

        camera = cameras[args.camera_id]
        print(f"Camera id: {camera.DEVICEID}, Camera position: {vicon_camera_pos_map[camera.DEVICEID]}")
        camera_orientation_quaternion = camera.ORIENTATION  # quaternion
        camera_orientation_axis_angle = eg.Camera.axis_angle_from_quaternion(camera_orientation_quaternion)
        print(f"Camera orientation axis: {camera_orientation_axis_angle}, angle: {np.linalg.norm(camera_orientation_axis_angle)*180/np.pi}")

        # Step 2: get SMPLX pose
        with open(args.input_file, "rb") as f:
            data = pickle.load(f)

        pose = data['fullpose'][args.frame]
        trans_mosh = data['trans'][args.frame]
        trans = np.array([0, 0, 0])  # x --> x in blender, y --> z in blender, z --> -y in blender
        global_orient = pose[:3]
        body_pose = pose[3:66]
        jaw_pose = pose[66:69]
        leye_pose = pose[69:72]
        reye_pose = pose[72:75]
        left_hand_pose = pose[75:120]
        right_hand_pose = pose[120:]
        global_orient = eg.Camera.rotate_axis_angle_by_axis_angle(blender_axis_angle(np.array([-np.pi / 2, 0, 0])), blender_axis_angle(global_orient))
        global_orient = eg.Camera.rotate_axis_angle_by_axis_angle(global_orient, blender_axis_angle(camera_orientation_axis_angle))
        # axis_angle: Rotations given as a vector in axis angle form, as a tensor of shape (..., 3), where the magnitude is the angle turned anticlockwise in radians around the vector's direction.

        format_data = {"global_orient": global_orient.reshape(1, -1),
                       "body_pose": body_pose.reshape(1, -1),
                       "jaw_pose": jaw_pose.reshape(1, -1),
                       "leye_pose": leye_pose.reshape(1, -1),
                       "reye_pose": reye_pose.reshape(1, -1),
                       "left_hand_pose": left_hand_pose.reshape(1, -1),
                       "right_hand_pose": right_hand_pose.reshape(1, -1),
                       "betas": data["betas"].reshape(1, -1),
                       "expression": get_expression(args),
                       "transl": trans.reshape(1, -1)
                       }

        output_file = r"C:\Users\wenleyan1\Downloads\SMPL_bpy.pkl"  # temp output file
        with open(output_file, "wb") as f:
            pickle.dump(format_data, f)
        print("Output file saved to", output_file)


        # for camera in cameras:
        #     print()
        #     print(f"Camera position: {vicon_camera_pos_map[camera.DEVICEID]}, Camera id: {camera.DEVICEID}")
        #     camera_orientation_quaternion = camera.ORIENTATION  # quaternion
        #     camera_orientation_axis_angle = eg.Camera.axis_angle_from_quaternion(camera_orientation_quaternion)
        #     print(f"Camera orientation axis: {camera_orientation_axis_angle}, angle: {np.linalg.norm(camera_orientation_axis_angle) * 180 / np.pi}")
        #     print("="*60)
        #
        # print(f"pose orientation axis: {pose[:3] }, angle: {np.linalg.norm(pose[:3] ) * 180 / np.pi}")



