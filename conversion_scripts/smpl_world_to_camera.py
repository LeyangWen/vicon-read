import numpy as np
import ergo3d as eg
from Skeleton import *
import pickle
import argparse
import os
import yaml
# import smplx

# from mosh SMPL to MotionBERT format in camera projection

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_SMPL_dir', type=str, default=r"W:\VEHS\VEHS-7M\Mesh\SMPL_pkl")
    parser.add_argument('--input_MB_file', type=str, default=r"W:\VEHS\VEHS data collection round 3\processed\VEHS_3D_downsample5_keep1_66.pkl")
    parser.add_argument('--camera_xcp_dir', type=str, default=r"W:\VEHS\VEHS data collection round 3\processed")  #S01\FullCollection\Activity00.xcp
    parser.add_argument('--split_config_file', type=str, default=r'config/experiment_config/VEHS-R3-622-MotionBert.yaml')
    parser.add_argument('--output_MB_file', type=str, default=None)
    args = parser.parse_args()
    with open(args.split_config_file, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
            base_folder = data['base_folder']
            args.val_keyword = data['val_keyword']
            args.test_keyword = data['test_keyword']
        except yaml.YAMLError as exc:
            print(args.split_config_file, exc)

    return args

def blender_axis_angle(axis_angle):
    return np.array([axis_angle[0], axis_angle[2], -axis_angle[1]])
    # return axis_angle

if __name__ == '__main__':
    args = parse_args()
    with open(args.input_MB_file, "rb") as f:
        MB_data = pickle.load(f)
    new_MB_data = {'train': {'smpl_pose': np.empty((0, 72)), 'smpl_shape': np.empty((0, 10)), 'camera_check': np.empty((0,))},
                   'validate': {'smpl_pose': np.empty((0, 72)), 'smpl_shape': np.empty((0, 10)), 'camera_check': np.empty((0,))},
                   'test': {'smpl_pose': np.empty((0, 72)), 'smpl_shape': np.empty((0, 10)), 'camera_check': np.empty((0,))},
                   }

    for root, dirs, files in os.walk(args.input_SMPL_dir):
        dirs.sort()  # Sort directories in-place
        print(root)
        files.sort(key=str.lower)  # Sort files in-place
        for file in files:
            if not file.endswith('.pkl'):
                continue
            activity = file.split('.')[0]  # Activity00 or activity00
            subject = root.split('\\')[-1]  # S01
            if any(keyword in root for keyword in args.val_keyword):
                train_val_test = 'validate'
            elif any(keyword in root for keyword in args.test_keyword):
                train_val_test = 'test'
            else:
                train_val_test = 'train'

            print(f"{subject} in {train_val_test} set, activity: {activity}, ")

            # Step 1: read SMPL pkl inputs
            with open(os.path.join(root, file), "rb") as f:
                smpl_data = pickle.load(f)
            smpl_pose = smpl_data['full_pose']
            frame_no = smpl_pose.shape[0]
            smpl_shape_one = smpl_data['betas']
            smpl_shape = np.tile(smpl_shape_one, (frame_no, 1))

            # Step 2: read camera orientation and rotate pose
            camera_xcp_file = os.path.join(args.camera_xcp_dir, subject, 'FullCollection', f"{activity}.xcp")
            cameras = eg.batch_load_from_xcp(camera_xcp_file)
            for camera_id, camera, in enumerate(cameras):
                print(f"Camera id: {camera.DEVICEID}, Camera position: {camera.POSITION}")
                camera_orientation_quaternion = camera.ORIENTATION  # quaternion
                camera_orientation_axis_angle = eg.Camera.axis_angle_from_quaternion(camera_orientation_quaternion)
                this_smpl_pose = smpl_pose.copy()
                for i in range(frame_no):  # todo: parallelize this in ergo3d.Camera.rotate_axis_angle_by_axis_angle
                    global_orient = smpl_pose[i, :3]
                    global_orient = eg.Camera.rotate_axis_angle_by_axis_angle(blender_axis_angle(np.array([-np.pi / 2, 0, 0])), blender_axis_angle(global_orient))
                    global_orient = eg.Camera.rotate_axis_angle_by_axis_angle(global_orient, blender_axis_angle(camera_orientation_axis_angle))
                    this_smpl_pose[i, :3] = global_orient
                this_camera_id_string_np = np.array([camera.DEVICEID] * frame_no)

                # Step 3: save to MotionBERT format
                new_MB_data[train_val_test]['smpl_pose'] = np.append(new_MB_data[train_val_test]['smpl_pose'], this_smpl_pose, axis=0)
                new_MB_data[train_val_test]['smpl_shape'] = np.append(new_MB_data[train_val_test]['smpl_shape'], smpl_shape, axis=0)
                new_MB_data[train_val_test]['camera_check'] = np.append(new_MB_data[train_val_test]['camera_check'], this_camera_id_string_np, axis=0)


    # Step 4: merge dictionary and save to pkl file
    for key in MB_data.keys():
        # check camera
        assert np.all(MB_data[key]['camera_name'] == new_MB_data[key]['camera_check']), f"Camera name mismatch in {key} set"
        print(f"Camera name check passed in {key} set")
        MB_data[key]['smpl_pose'] = new_MB_data[key]['smpl_pose']
        MB_data[key]['smpl_shape'] = new_MB_data[key]['smpl_shape']

    if args.output_MB_file is None:
        args.output_MB_file = args.input_MB_file.replace('.pkl', '_SMPL.pkl')
    with open(args.output_MB_file, "wb") as f:
        pickle.dump(MB_data, f)
    print(f"Output file saved to {args.output_MB_file}")


# file = r"C:\Users\Public\Documents\Vicon\vicon_coding_projects\MotionBERT\data\mesh\mesh_det_h36m.pkl"
# with open(file, "rb") as f:
#     MB_data = pickle.load(f)
# MB_data['test'].keys()  # dict_keys(['joint_2d', 'confidence', 'joint_cam', 'smpl_pose', 'smpl_shape', 'camera_name', 'action', 'source'])
# MB_data['test']['joint_2d'].shape  # (102280, 17, 2)
# MB_data['test']['confidence'].shape  # (102280, 17, 1)
# MB_data['test']['joint_cam'].shape  # (102280, 17, 3)
# MB_data['test']['smpl_pose'].shape  # (102280, 72)
# MB_data['test']['smpl_shape'].shape  # (102280, 10)
# MB_data['test']['camera_name']  # [102280]
# MB_data['test']['action']  # [102280]
# MB_data['test']['source']  # [102280]
