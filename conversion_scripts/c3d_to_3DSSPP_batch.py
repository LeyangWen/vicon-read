# c3d_to_MB.py

import c3d
from utility import *

from Skeleton import *
import pickle
import yaml
import argparse



if __name__ == '__main__':
    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_config_file', type=str, default=r'config/experiment_config/VEHS-R3-721-MotionBert.yaml')
    parser.add_argument('--skeleton_file', type=str, default=r'config\VEHS_ErgoSkeleton_info\Ergo-Skeleton-66.yaml')
    parser.add_argument('--downsample', type=int, default=5)
    parser.add_argument('--downsample_keep', type=int, default=1)
    parser.add_argument('--split_output', action='store_true')  # not implemented yet
    parser.add_argument('--output_type', type=list, default=[False, True, False, False], help='3D, 6D, SMPL, 3DSSPP')
    parser.add_argument('--output_file_name_end', type=str, default='_config6_tilt_corrected')
    parser.add_argument('--distort', action='store_false', help='consider camera distortion in the output 2D pose')
    parser.add_argument('--zero_camera_pitch', type=bool, default=True)
    args = parser.parse_args()


    # base_folder = r'C:\Users\Public\Documents\Vicon\data\Vicon_F\Round3\LeyangWen'
    split_config_file = args.split_config_file
    with open(split_config_file, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
            base_folder = data['base_folder']
            # base_folder = os.path.join(base_folder, 'LeyangWen')  # for testing
            val_keyword = data['val_keyword']
            test_keyword = data['test_keyword']
        except yaml.YAMLError as exc:
            print(split_config_file, exc)


    skeleton_file = args.skeleton_file
    downsample = args.downsample
    downsample_keep = args.downsample_keep

    output_smpl_dataset = {}
    count = 0
    dataset_statistics = {}
    total_frame_number = 0
    for root, dirs, files in os.walk(base_folder):
        dirs.sort()  # Sort directories in-place --> important, will change walk sequence
        files.sort(key=str.lower)  # Sort files in-place
        for file in files:
            if file.endswith('.c3d') and root[-6:] != 'backup' and (not file.startswith('ROM')) and (not file.endswith('_bad.c3d')):
                # val_keyword is a list of string, if any of them is in the root, then it is val set
                if any(keyword in root for keyword in val_keyword):
                    train_val_test = 'validate'
                elif any(keyword in root for keyword in test_keyword):
                    train_val_test = 'test'
                else:
                    train_val_test = 'train'
                    continue
                c3d_file = os.path.join(root, file)

                print(f"file: {file}, root: {root}, train_val_test: {train_val_test}")
                print(f'{count}: Starting on {c3d_file} as {train_val_test} set')
                this_skeleton = VEHSErgoSkeleton(skeleton_file)
                this_skeleton.load_c3d(c3d_file, analog_read=False)

                this_skeleton.calculate_joint_center()
                camera_xcp_file = c3d_file.replace('.c3d', '.xcp')

                # S01-A01 camera pitch
                # correcting pitch for 51470934, pitch: 9.1 degrees
                # correcting pitch for 66920731, pitch: 29.2 degrees
                # correcting pitch for 66920734, pitch: 6.8 degrees
                # correcting pitch for 66920758, pitch: 16.7 degrees

                batch_3DSSPP_batch_filename = c3d_file.replace('.c3d', '-3DSSPP.txt')
                this_skeleton.output_3DSSPP_loc(frame_range=[0, 3000, 10], loc_file=batch_3DSSPP_batch_filename)
                # self = this_skeleton
                break
                del this_skeleton

    if False:
        import subprocess
        import shutil
        ########################### Step 2: Run 3DSSPP ###########################
        # Get the initial modification time of the output file
        SSPP_CLI_folder = 'H:\\3DSSPP_all\Compiled\\3DSSPP 7.1.2 CLI'
        loc_file = batch_3DSSPP_batch_filename
        # export_file = os.path.join(SSPP_CLI_folder, 'export', 'batchinput_export.txt')  # constant if using wrapper
        # initial_mtime = os.stat(export_file).st_mtime
        loc_file = loc_file.replace('\\', '/')
        # copy the loc file to the 3DSSPP folder
        shutil.copy(loc_file, SSPP_CLI_folder)
        print(f"\n{'@' * 30} Subprocess start {'@' * 30}")
        # loc_file ="example_input_batch.txt"
        subprocess.call(['bash', '3DSSPP-script.sh', loc_file.split('/')[-1], '--avi', '4'], shell=True, cwd=SSPP_CLI_folder)  # '--autoclose'
        # careful to look for errors messages in terminal for the subprocess, will not stop code
        # wait_for_file_update(export_file, initial_mtime)  # Wait for the output file to be updated
        print(f"\n{'@' * 30} Subprocess end {'@' * 30}\n")
