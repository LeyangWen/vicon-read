import argparse
import os.path
import pickle
from Skeleton import *
from scipy.stats import f_oneway
from MB_np_to_visual import MB_input_pose_file_loader

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config_file', type=str, default=r'config/experiment_config/VEHS-6D-MB.yaml')
    # parser.add_argument('--skeleton_file', type=str, default=r'config/VEHS_ErgoSkeleton_info/Ergo-Skeleton-66.yaml')
    # parser.add_argument('--angle_mode', type=str, default='VEHS')

    # parser.add_argument('--config_file', type=str, default=r'config/experiment_config/H36M17kpts/VEHS-3D-MB.yaml')
    # parser.add_argument('--skeleton_file', type=str, default=r'config/VEHS_ErgoSkeleton_info/H36M-17.yaml')
    # parser.add_argument('--angle_mode', type=str, default='VEHS')

    parser.add_argument('--config_file', type=str, default=r'config/experiment_config/37kpts/Inference-RTMPose-MB-20fps-VEHS7M.yaml')  # config/experiment_config/VEHS-6D-MB.yaml') #
    parser.add_argument('--skeleton_file', type=str, default=r'config/VEHS_ErgoSkeleton_info/Ergo-Skeleton-37.yaml')
    parser.add_argument('--angle_mode', type=str, default='VEHS')
    parser.add_argument('--clip_fill', type=bool, default=False)



    parser.add_argument('--MB_data_stride', type=int, default=243)
    parser.add_argument('--debug_mode', default=True)
    parser.add_argument('--merge_lr', default=True)

    # parser.add_argument('--name_list', type=list, default=[])
    args = parser.parse_args()
    with open(args.config_file, 'r') as stream:
        data = yaml.safe_load(stream)
        args.name_list = data['name_list']
        args.GT_name_list = data['GT_name_list'] if 'GT_name_list' in data else data['name_list']
        args.GT_file = data['GT_file']
        args.eval_key = data['eval_key']
        args.estimate_file = data['estimate_file']
        if type(args.estimate_file) == str:
            args.output_dir = os.path.join(os.path.dirname(args.estimate_file), 'results')
        else:
            args.output_dir = os.path.join('/Users/leyangwen/Downloads/temp', 'results')
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    return args


def MB_output_pose_file_loader(args):
    with open(args.estimate_file, "rb") as f:
        output_np_pose = np.load(f)
    np_pose_shape = output_np_pose.shape
    output_np_pose = output_np_pose.reshape(-1, np_pose_shape[-2], np_pose_shape[-1])
    return output_np_pose


if __name__ == '__main__':
    # read arguments
    args = parse_args()

    if args.eval_key == 'dict':
        estimate_pose = []
        GT_pose = []
        for key, item in args.estimate_file.items():
            args.eval_key = key
            args.estimate_file = item
            this_estimate_pose = MB_output_pose_file_loader(args)
            estimate_pose.append(this_estimate_pose)

            this_gt_pose, _ = MB_input_pose_file_loader(args)
            GT_pose.append(this_gt_pose)
        estimate_pose = np.concatenate(estimate_pose, axis=0)
        GT_pose = np.concatenate(GT_pose, axis=0)
    else:
        GT_pose, _ = MB_input_pose_file_loader(args)
        estimate_pose = MB_output_pose_file_loader(args)
    # assert GT_pose.shape == estimate_pose.shape, f"GT_pose.shape: {GT_pose.shape}, estimate_pose.shape: {estimate_pose.shape}, they should be the same"
    assert GT_pose.shape[0] == estimate_pose.shape[0], f"GT_pose.shape: {GT_pose.shape}, estimate_pose.shape: {estimate_pose.shape}, frame no should be the same"

    if args.debug_mode:
        small_sample = 1200
        GT_pose = GT_pose[:small_sample]
        estimate_pose = estimate_pose[:small_sample]

    # Step 1: calculate GT angles
    GT_skeleton = VEHSErgoSkeleton_angles(args.skeleton_file, mode=args.angle_mode)
    # GT_skeleton = H36MSkeleton_angles(args.skeleton_file, mode=args.angle_mode)  # todo: remember to change this back to VEHSErgoSkeleton_angles
    GT_skeleton.load_name_list_and_np_points(args.GT_name_list, GT_pose)
    GT_ergo_angles = {}
    for angle_name in GT_skeleton.angle_names:  # calling the angle calculation methods in skeleton class
        class_method_name = f'{angle_name}_angles'
        GT_ergo_angles[angle_name] = getattr(GT_skeleton, class_method_name)()


    # Step 2: calculate MB angles
    estimate_skeleton = VEHSErgoSkeleton_angles(args.skeleton_file, mode=args.angle_mode)
    # estimate_skeleton = H36MSkeleton_angles(args.skeleton_file, mode=args.angle_mode)  # todo: remember to change this back to VEHSErgoSkeleton_angles
    estimate_skeleton.load_name_list_and_np_points(args.name_list, estimate_pose)
    estimate_ergo_angles = {}
    for angle_name in GT_skeleton.angle_names:  # calling the angle calculation methods in skeleton class
        class_method_name = f'{angle_name}_angles'
        try:
            estimate_ergo_angles[angle_name] = getattr(estimate_skeleton, class_method_name)()
        except:
            estimate_ergo_angles[angle_name] = estimate_skeleton.empty_angles()

    # Step 3: visualize
    
    # # Hi Veeru, I used this to visualize the 3D pose frame by frame
    # frame = 10210
    # frame = 10180
    # GT_skeleton.plot_3d_pose_frame(frame)
    # estimate_skeleton.plot_3d_pose_frame(frame)

    frame_range = [0, 1200]
    log = []
    anova_results = []
    average_error = {}
    target_angles = GT_skeleton.angle_names
    # target_angles = ['right_shoulder']
    for angle_index, this_angle_name in enumerate(target_angles):
        # plot angles
        GT_fig, GT_ax = GT_ergo_angles[this_angle_name].plot_angles(joint_name=f"GT-{this_angle_name}", frame_range=frame_range, alpha=0.75, colors=['g', 'g', 'g'])
        estimate_fig, _ = estimate_ergo_angles[this_angle_name].plot_angles(joint_name=f"Est-{this_angle_name}", frame_range=frame_range, alpha=0.75, colors=['r', 'r', 'r'], overlay=[GT_fig, GT_ax])
        # plt.show()
        # GT_fig.savefig(f'frames/MB_angles/GT-{this_angle_name}.png')
        if estimate_fig:
            estimate_fig.savefig(f'frames/MB_angles/Est-{this_angle_name}.png')


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
                # bland-Altman plot
                if not os.path.exists(os.path.join(args.output_dir, 'BA_plots', args.eval_key)):
                    os.makedirs(os.path.join(args.output_dir, 'BA_plots', args.eval_key))
                md, sd = bland_altman_plot(ja1, ja2, title=f'{print_angle_name}: {print_ergo_name}',
                                           save_path=os.path.join(args.output_dir,f'BA_plots/{args.eval_key}/{angle_index}-{this_angle_name}-{this_ergo_angle}.png'))
                                           #save_path=f'frames/MB_angles/BA_plots/{angle_index}-{this_angle_name}-{this_ergo_angle}.png')
                print(f'Bland Altman: md: {md:.2f}, sd: {sd:.2f}')

                RMSE = root_mean_squared_error(ja1, ja2)
                MAE = mean_absolute_error(ja1, ja2)
                median_AE = median_absolute_error(ja1, ja2)
                merge_name = f"{print_angle_name}-{print_ergo_name}"
                average_error[merge_name] = angle_diff(ja1, ja2, input_rad=True, output_rad=False)

                angle_compare = AngleCompare(ja1, ja2)
                # 1,435,236 frames in test set, use Rice rule --> approx 225 bins
                # make folder if not exits
                if not os.path.exists(os.path.join(args.output_dir, 'histograms', args.eval_key)):
                    os.makedirs(os.path.join(args.output_dir, 'histograms', args.eval_key))
                plot_error_histogram(angle_compare.diff_deg, bins=225, title=f'{print_angle_name}: {print_ergo_name}',
                                     save_path=os.path.join(args.output_dir, f'histograms/{args.eval_key}/{angle_index}-{this_angle_name}-{this_ergo_angle}_hist.png'),
                                     plot_normal_curve=angle_compare.plot_normal_curve)
                                     # save_path=f'frames/MB_angles/histograms/{angle_index}-{this_angle_name}-{this_ergo_angle}_hist.png',
                error_dist = analyze_error_distribution(angle_compare.diff_deg)
                print(f'Error distribution: {error_dist}')

                # error_dist = analyze_error_distribution(angle_compare.diff_inliers_deg)
                # print(f'Error distribution: {error_dist}')

                np.nanstd(ja1, axis=0)
                np.nanstd(ja1)

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
        # if flexion_errors.size > 0 and abduction_errors.size > 0 and rotation_errors.size > 0:
        #     cut = 1500
        #     f_stat, p_value = f_oneway(flexion_errors[:cut], abduction_errors[:cut] , rotation_errors[:cut])
        #     anova_results.append((this_angle_name, f_stat, p_value))
        # else:
        #     f_stat, p_value = np.nan, np.nan



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
                f.write(str(j)+",")
            f.write("\n")

    print("ANOVA Results:")
    print("Angle Name, F-Statistic, P-Value")
    for result in anova_results:
        print(f"{result[0]}, {result[1]:}, {result[2]}")

    # store Absolute Error for each angle in dict
    with open(os.path.join(args.output_dir, f'{args.eval_key}_{args.angle_mode}_AE.pkl'), 'wb') as f:
        pickle.dump(average_error, f)

    print(f"Store location: {args.output_dir}")
# generate merged bland-altman plot for left and right


""" powershell batch crop png files
cd frames/MB_angles/BA_plots
Remove-Item .\cropped\* -Recurse
$files = Get-ChildItem
foreach ($file in $files) {
    ffmpeg -i $file -vf "crop=515:450:74:22" "cropped/$file"
}
"""
""" in mac
#!/bin/bash

# Create the cropped directory if it doesn't exist
mkdir -p cropped

# Remove existing files in the cropped directory
rm -rf cropped/*

# Loop through all PNG files in the current directory
for file in *.png; do
    # Crop using FFmpeg
    ffmpeg -i "$file" -vf "crop=515:450:74:22" "cropped/$file"
done

"""

""" ffmpeg crop command
ffmpeg -i back-abduction.png -vf "crop=515:430:74:22" output.jpg
"""



