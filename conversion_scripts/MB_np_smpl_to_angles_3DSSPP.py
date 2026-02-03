import argparse
import os.path
import pickle

import matplotlib.pyplot as plt
from scipy.stats import f_oneway

from Skeleton import *
from MB_np_to_visual import MB_input_pose_file_loader
import matplotlib
matplotlib.use('Qt5Agg')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=r'config/experiment_config/meshCompare/mesh-compare-v2.yaml')
    parser.add_argument('--skeleton_file', type=str, default=r'config/VEHS_ErgoSkeleton_info/Ergo-Skeleton-37.yaml')
    parser.add_argument("--input_type", choices=["mesh_17", "mesh_66", "mesh_SMPLEST", "3D", "6D", "pose_37"], default="mesh_17")
    # parser.add_argument("--output_type", choices=["22angles", "3DSSPP"], default=["22angles"], nargs="+")

    parser.add_argument('--angle_mode', type=str, default='paper')
    parser.add_argument('--clip_fill', type=bool, default=True)

    parser.add_argument('--output_frame_folder', type=str, default=None)
    parser.add_argument('--output_GT_frame_folder', type=str, default=None)
    parser.add_argument('--plot_mode', choices=['normal_view', 'camera_view', 'camera_side_view', '0_135_view'], default='normal_view')
    parser.add_argument('--MB_data_stride', type=int, default=243)  # may overwrite
    parser.add_argument('--debug_mode', default=False, type=bool)
    parser.add_argument('--merge_lr', default=True)

    # parser.add_argument('--name_list', type=list, default=[])
    args = parser.parse_args()
    with open(args.config_file, 'r') as stream:
        data = yaml.safe_load(stream)
        args.eval_key = data['eval_key']
        args.root_dir = data['root_dir']

        args.GT_6D_name_list = data['GT_6D_name_list']
        args.GT_file = os.path.join(args.root_dir,data['GT_file_pitch_correct'])

        args.estimate_3D_name_list = data['estimate_3D_name_list']
        args.estimate_3D_file = os.path.join(args.root_dir, data['estimate_3D_file'])
        if 'mesh_17' in args.input_type:
            args.estimate_mesh_file = os.path.join(args.root_dir, data['estimate_mesh_17_file'])
            args.MB_data_stride = 16
        elif 'mesh_66' in args.input_type:
            args.estimate_mesh_file = os.path.join(args.root_dir, data['estimate_mesh_66_file'])
            args.MB_data_stride = 16
        elif 'mesh_SMPLEST' in args.input_type:
            args.estimate_mesh_file = os.path.join(args.root_dir, data['estimate_mesh_SMPLEST_file'])
            args.MB_data_stride = 2
        if 'mesh' in args.input_type:
            args.GT_file = os.path.join(args.root_dir,data['GT_file'])

    if '3D' == args.input_type:
        args.output_dir = os.path.join(os.path.dirname(args.estimate_3D_file), 'results')
    elif '6D' == args.input_type:
        args.output_dir = os.path.join(os.path.dirname(args.estimate_6D_file), 'results')
    elif 'mesh' in args.input_type:
        args.output_dir = os.path.join(os.path.dirname(args.estimate_mesh_file), 'results')
    else:
        raise NotImplementedError("args.output_dir not set")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.file_start = None
    return args


def MB_output_pose_file_loader(args):
    if args.estimate_3D_file=='None':
        return None
    with open(args.estimate_3D_file, "rb") as f:
        output_np_pose = np.load(f)
    np_pose_shape = output_np_pose.shape
    output_np_pose = output_np_pose.reshape(-1, np_pose_shape[-2], np_pose_shape[-1])
    return output_np_pose


def MB_output_mesh_file_loader(args, npy_mode=True):
    file = args.estimate_mesh_file
    if file == 'None':
        return None
    if npy_mode:
        # For OOM --> First preprocess using "conversion_scripts/MB_SMPL_file_compress.py"
        # file = file.replace('.pkl', '_verts.npy')  # for all verts
        file = file.replace('.pkl', '_markers.npy')
        verts = np.load(file, mmap_mode='r')
        print("Loaded mesh output from:", file)
        return verts
    else:
        with open(file, "rb") as f:
            output_mesh_pose = pickle.load(f)
        print("Loaded mesh output from:", file)
        verts = output_mesh_pose.pop("verts").reshape(-1, 6890, 3)
        print("Reshaped verts to:", verts.shape)
        # verts_gt = output_mesh_pose['verts_gt'].reshape(-1, 6890, 3)
        # kp_3d = output_mesh_pose['kp_3d'].reshape(-1, 17, 3)
        del output_mesh_pose
        return verts

def SMPLest_output_mesh_dir_loader(args):
    if dir == 'None':
        return None
    verts = np.array([])
    file_start = []
    frame_id = 0
    file_id = 0
    for root, dirs, files in os.walk(args.estimate_mesh_file):
        dirs.sort()  # Sort directories in-place --> important, will change walk sequence
        files.sort(key=str.lower)  # Sort files in-place
        for file in files:
            if not file.endswith('.npy'):
                continue
            # print(root, dirs, file)
            file_path = os.path.join(root, file)
            cur_verts = np.load(file_path)
            frame_id_start = frame_id
            frame_id += cur_verts.shape[0]
            file_start.append([file_id, frame_id_start, frame_id, cur_verts.shape[0], file_path])
            file_id += 1
            # append verts
            verts = cur_verts if verts.size == 0 else np.concatenate((verts, cur_verts), axis=0)

    return verts, file_start

if __name__ == '__main__':
    # read arguments
    args = parse_args()
    # raise NotImplementedError


    print(f"Loading estimated poses from {args.input_type}...")
    if '3D' == args.input_type:
        pass
        estimate_3D_pose = MB_output_pose_file_loader(args)
        estimate_3D_pose = estimate_3D_pose[:1200] if args.debug_mode else estimate_3D_pose
        # Step 2: calculate MB-6D angles
        estimate_3D_skeleton = H36MSkeleton_angles("config/VEHS_ErgoSkeleton_info/H36M-17.yaml")
        print("Loading to skeleton...")
        estimate_3D_skeleton.load_name_list_and_np_points(args.estimate_3D_name_list, estimate_3D_pose)
        print("Calculating estimated angles...")
        estimate_3D_ergo_angles = {}
        for angle_name in estimate_3D_skeleton.angle_names:  # calling the angle calculation methods in skeleton class
            class_method_name = f'{angle_name}_angles'
            estimate_3D_ergo_angles[angle_name] = getattr(estimate_3D_skeleton, class_method_name)()
        estimate_3D_ergo_angles['back'] = estimate_3D_skeleton.back_angles(kpt_source = 'VEHS37kpts')
        estimate_ergo_angles = estimate_3D_ergo_angles

    elif '6D' == args.input_type:
        # estimate_6D_pose = MB_output_pose_file_loader(args)
        # # Step 2: calculate MB-6D angles
        # estimate_6D_skeleton = VEHSErgoSkeleton_angles(args.skeleton_file, mode=args.angle_mode, try_wrist=False)
        # estimate_6D_skeleton.load_name_list_and_np_points(args.name_list, estimate_6D_pose)
        # estimate_6D_ergo_angles = {}
        # for angle_name in estimate_6D_skeleton.angle_names:  # calling the angle calculation methods in skeleton class
        #     class_method_name = f'{angle_name}_angles'
        #     estimate_6D_ergo_angles[angle_name] = getattr(estimate_6D_skeleton, class_method_name)()
        pass
    elif 'mesh' in args.input_type:
        if 'SMPLEST' in args.input_type:
            estimate_mesh_vert, file_start = SMPLest_output_mesh_dir_loader(args)  #277912
            args.file_start = file_start
        else:
            estimate_mesh_vert = MB_output_mesh_file_loader(args)
        estimate_mesh_vert = estimate_mesh_vert[:1200] if args.debug_mode else estimate_mesh_vert

        up_right_frame = 0
        HDTP = estimate_mesh_vert[up_right_frame, 0]
        RHEEL = estimate_mesh_vert[up_right_frame, -2]
        LHEEL = estimate_mesh_vert[up_right_frame, -1]
        subject_height= 1800 # mm
        pose_height = np.linalg.norm(HDTP - (RHEEL + LHEEL) / 2) * 2
        scale = subject_height / pose_height
        print(f"Scaling mesh by {scale:.3f} to match subject height of {subject_height} mm")
        estimate_mesh_vert = estimate_mesh_vert * scale # needed for joint calculation


        # calculate MB-mesh angles
        print("Loading to skeleton...")
        estimate_mesh_skeleton = VEHSErgoSkeleton_angles(args.skeleton_file, mode=args.angle_mode, try_wrist=False)
        # estimate_mesh_skeleton.load_mesh(estimate_mesh_vert)  # for full verts inputs, very slow
        estimate_mesh_skeleton.load_mesh(estimate_mesh_vert, pre_saved=True)  # for full verts inputs, very slow
        estimate_mesh_skeleton.calculate_joint_center()
        print("Calculating estimated angles...")
        estimate_mesh_ergo_angles = {}
        for angle_name in estimate_mesh_skeleton.angle_names:  # calling the angle calculation methods in skeleton class
            class_method_name = f'{angle_name}_angles'
            estimate_mesh_ergo_angles[angle_name] = getattr(estimate_mesh_skeleton, class_method_name)()
        if 'SMPLEST' not in args.input_type: # z up for MB-mesh
            estimate_mesh_ergo_angles['back'] = estimate_mesh_skeleton.back_angles(up_axis=[0, 0, -1])

        estimate_ergo_angles = estimate_mesh_ergo_angles

    # # Step 2: Get GT angles to compare
    print("Loading GT poses...")
    GT_pose, factor_25d, clip_id = MB_input_pose_file_loader(args, get_clip_id=True, file_start = args.file_start)
    GT_pose = GT_pose[:1200] if args.debug_mode else GT_pose
    print("Calculating GT angles...")
    # calculate GT angles
    GT_skeleton = VEHSErgoSkeleton_angles(args.skeleton_file, mode=args.angle_mode, try_wrist=False)
    GT_skeleton.load_name_list_and_np_points(args.GT_6D_name_list, GT_pose)
    GT_ergo_angles = {}
    for angle_name in GT_skeleton.angle_names:  # calling the angle calculation methods in skeleton class
            class_method_name = f'{angle_name}_angles'
            GT_ergo_angles[angle_name] = getattr(GT_skeleton, class_method_name)()


    print("Calculating angle errors...")

    # copeid from np_to_angles

    # Step 3: visualize

    # # Hi Veeru, I used this to visualize the 3D pose frame by frame
    # frame = 10210
    # frame = 3000
    # for frame in [0, 1000, 3000, 9000, 12000]:
    #     GT_skeleton.plot_3d_pose_frame(frame=frame, coord_system="camera-px")
    #     estimate_mesh_skeleton.plot_3d_pose_frame(frame=frame, coord_system="camera-px")

    frame_range = [0, 1200]
    log = []
    anova_results = []
    average_error = {}
    target_angles = GT_skeleton.angle_names
    all_ja1 = None
    all_ja2 = None
    # target_angles = ['right_shoulder']
    for angle_index, this_angle_name in enumerate(target_angles):
        try:
            estimate_ergo_angles[this_angle_name]
        except:
            print(f"Angle {this_angle_name} not found in {args.input_type} estimate")
            continue
        # plot angles
        GT_fig, GT_ax = GT_ergo_angles[this_angle_name].plot_angles_old_style(joint_name=f"GT-{this_angle_name}", frame_range=frame_range, alpha=0.75, colors=['g', 'g', 'g'])
        estimate_fig, _ = estimate_ergo_angles[this_angle_name].plot_angles_old_style(joint_name=f"Est-{this_angle_name}", frame_range=frame_range, alpha=0.75, colors=['r', 'r', 'r'], overlay=[GT_fig, GT_ax])
        # plt.show()
        # GT_fig.savefig(f'frames/MB_angles/GT-{this_angle_name}.png')
        if not os.path.exists(os.path.join(args.output_dir, 'angle_plots', args.eval_key)):
            os.makedirs(os.path.join(args.output_dir, 'angle_plots', args.eval_key))
        if estimate_fig:
            estimate_fig.savefig(os.path.join(args.output_dir, 'angle_plots', args.eval_key, f'Est-{this_angle_name}.png'))

        ergo_angle_name = ['flexion', 'abduction', 'rotation']
        print_ergo_names = getattr(GT_ergo_angles[this_angle_name], 'ergo_name')
        print_angle_name = this_angle_name.replace('_', '').replace('right', 'R-').replace('left', 'L-').capitalize()
        flexion_errors = np.array([])
        abduction_errors = np.array([])
        rotation_errors = np.array([])
        for this_ergo_angle in ergo_angle_name:
            ja1 = getattr(estimate_ergo_angles[this_angle_name], this_ergo_angle)
            ja2 = getattr(GT_ergo_angles[this_angle_name], this_ergo_angle)
            if args.merge_lr:
                if 'right' in this_angle_name:
                    ja1_l = getattr(estimate_ergo_angles[this_angle_name.replace('right', 'left')], this_ergo_angle)
                    ja2_l = getattr(GT_ergo_angles[this_angle_name.replace('right', 'left')], this_ergo_angle)
                    try:
                        ja1 = np.concatenate([ja1, ja1_l])
                        ja2 = np.concatenate([ja2, ja2_l])
                    except(ValueError):
                        continue
                    print_angle_name = print_angle_name.replace('R-', '').capitalize()
                elif 'left' in this_angle_name:
                    continue
            print_ergo_name = print_ergo_names[this_ergo_angle].capitalize()
            if ja1 is not None:
                print("=====================================")
                print(f'{this_angle_name} - {this_ergo_angle}')
                all_ja1 = ja1 if all_ja1 is None else np.concatenate([all_ja1, ja1])
                all_ja2 = ja2 if all_ja2 is None else np.concatenate([all_ja2, ja2])
                # raise NotImplementedError
                # bland-Altman plot
                if not os.path.exists(os.path.join(args.output_dir, 'BA_plots', args.eval_key)):
                    os.makedirs(os.path.join(args.output_dir, 'BA_plots', args.eval_key))
                md, sd = bland_altman_plot(ja1, ja2, title=f'{print_angle_name}: {print_ergo_name}',
                                           save_path=os.path.join(args.output_dir, f'BA_plots/{args.eval_key}/{angle_index}-{this_angle_name}-{this_ergo_angle}.png'))
                # save_path=f'frames/MB_angles/BA_plots/{angle_index}-{this_angle_name}-{this_ergo_angle}.png')
                # print(f'Bland Altman: md: {md:.2f}, sd: {sd:.2f}')

                RMSE = root_mean_squared_error(ja1, ja2)
                MAE = mean_absolute_error(ja1, ja2)
                median_AE = median_absolute_error(ja1, ja2)
                merge_name = f"{print_angle_name}-{print_ergo_name}"
                average_error[merge_name] = angle_diff(ja1, ja2, input_rad=True, output_rad=False)

                angle_compare = AngleCompare(ja1, ja2)
                # 1,435,236 frames in test set, use Rice rule --> approx 225 bins
                # 	•	For n = 267{,}300: 128 bins
                # 	•	For n = 534{,}600: 162 bins --> 150
                bin_no = 150
                # make folder if not exits
                if not os.path.exists(os.path.join(args.output_dir, 'histograms', args.eval_key)):
                    os.makedirs(os.path.join(args.output_dir, 'histograms', args.eval_key))
                plot_error_histogram(angle_compare.diff_deg, bins=bin_no, title=f'{print_angle_name}: {print_ergo_name}',
                                     save_path=os.path.join(args.output_dir, f'histograms/{args.eval_key}/{angle_index}-{this_angle_name}-{this_ergo_angle}_hist.png'),
                                     plot_normal_curve=angle_compare.plot_normal_curve)
                # save_path=f'frames/MB_angles/histograms/{angle_index}-{this_angle_name}-{this_ergo_angle}_hist.png',
                error_dist = analyze_error_distribution(angle_compare.diff_deg)
                print(f'Error distribution: {error_dist}')

                # error_dist = analyze_error_distribution(angle_compare.diff_inliers_deg)
                # print(f'Error distribution: {error_dist}')

                # Calculate errors
                errors = ja1 - ja2
                if this_ergo_angle == 'flexion':
                    flexion_errors = errors.copy()
                elif this_ergo_angle == 'abduction':
                    abduction_errors = errors.copy()
                elif this_ergo_angle == 'rotation':
                    rotation_errors = errors.copy()

                print(f'MAE: {MAE:.2f}, RMSE: {RMSE:.2f}, median: {angle_compare.median_deg:.2f}')
                # this_log = [this_angle_name, this_ergo_angle, md, sd, MAE, RMSE]
                this_log = [print_angle_name, print_ergo_name, MAE, median_AE, RMSE, md, sd, error_dist['skewness'], error_dist['kurtosis'], angle_compare.median_deg, error_dist['IQR']]
                log.append(this_log)

        # # Perform ANOVA for each joint
        if flexion_errors.size > 0 and abduction_errors.size > 0 and rotation_errors.size > 0:
            cut = 1500
            f_stat, p_value = f_oneway(flexion_errors[:cut], abduction_errors[:cut] , rotation_errors[:cut])
            anova_results.append((this_angle_name, f_stat, p_value))
        else:
            f_stat, p_value = np.nan, np.nan

    # for all angles
    print("=====================================")
    print(f'All Angles')
    md, sd = bland_altman_plot(all_ja1, all_ja2, title=f'All Angles',
                               save_path=os.path.join(args.output_dir, f'BA_plots/{args.eval_key}/All.png'))
    RMSE = root_mean_squared_error(all_ja1, all_ja2)
    MAE = mean_absolute_error(all_ja1, all_ja2)
    median_AE = median_absolute_error(all_ja1, all_ja2)
    merge_name = f"{print_angle_name}-{print_ergo_name}"
    average_error[merge_name] = angle_diff(all_ja1, all_ja2, input_rad=True, output_rad=False)

    angle_compare = AngleCompare(all_ja1, all_ja2)
    plot_error_histogram(angle_compare.diff_deg, bins=bin_no, title=f'All Angles',
                         save_path=os.path.join(args.output_dir, f'histograms/{args.eval_key}/All_hist.png'),
                         plot_normal_curve=angle_compare.plot_normal_curve)
    save_path=f'frames/MB_angles/histograms/{angle_index}-{this_angle_name}-{this_ergo_angle}_hist.png',
    error_dist = analyze_error_distribution(angle_compare.diff_deg)
    print(f'Error distribution: {error_dist}')
    print(f'MAE: {MAE:.2f}, RMSE: {RMSE:.2f}, median: {angle_compare.median_deg:.2f}')
    # this_log = [this_angle_name, this_ergo_angle, md, sd, MAE, RMSE]
    this_log = ["All", "-", MAE, median_AE, RMSE, md, sd, error_dist['skewness'], error_dist['kurtosis'], angle_compare.median_deg, error_dist['IQR']]
    log.append(this_log)

    # print log as csv in console
    header = ["angle_name", "ergo_angle", "MAE", "median_AE", "RMSE", "diff_md", "dif_sd", "skewness", "kurtosis", "median", "IQR"]
    for i in log:
        for j in i:
            print(j, end=",")
        print()
    # also save log as csv
    with open(os.path.join(args.output_dir, f'{args.eval_key}_{args.angle_mode}_log.csv'), 'w') as f:
        f.write(f"{header}\n")
        for i in log:
            for j in i:
                f.write(str(j) + ",")
            f.write("\n")

    print("ANOVA Results:")
    print("Angle Name, F-Statistic, P-Value")
    for result in anova_results:
        print(f"{result[0]}, {result[1]:}, {result[2]}")

    # store Absolute Error for each angle in dict
    with open(os.path.join(args.output_dir, f'{args.eval_key}_{args.angle_mode}_AE.pkl'), 'wb') as f:
        pickle.dump(average_error, f)

    print(f"Store location: {args.output_dir}")





