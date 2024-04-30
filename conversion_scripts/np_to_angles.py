import argparse
import pickle
from Skeleton import *



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=r'config\experiment_config\VEHS-6D-MB.yaml')
    parser.add_argument('--skeleton_file', type=str, default=r'config\VEHS_ErgoSkeleton_info\Ergo-Skeleton-66.yaml')
    parser.add_argument('--MB_data_stride', type=int, default=243)
    parser.add_argument('--debug_mode', action='store_false')

    # parser.add_argument('--name_list', type=list, default=[])
    args = parser.parse_args()
    with open(args.config_file, 'r') as stream:
        data = yaml.safe_load(stream)
        args.name_list = data['name_list']
        args.GT_file = data['GT_file']
        args.eval_key = data['eval_key']
        args.estimate_file = data['estimate_file']
    return args

def MB_input_pose_file_loader(args):
    with open(args.GT_file, "rb") as f:
        data = pickle.load(f)
    source = data[args.eval_key]['source']
    MB_clip_id = []
    k = 0
    for i in range(len(source)):  # MB clips each data into 243 frame segments, the last segment (<243) is discarded
        k += 1
        if k == args.MB_data_stride:
            k = 0
            good_id = list(range(i-args.MB_data_stride+1, i+1))
            MB_clip_id.extend(good_id)
        if i == len(source)-1:
            break
        if source[i] != source[i+1]:
            k = 0
    np_pose = data[args.eval_key]['joint3d_image'][MB_clip_id]
    return np_pose


def MB_output_pose_file_loader(args):
    with open(args.estimate_file, "rb") as f:
        output_np_pose = np.load(f)
    np_pose_shape = output_np_pose.shape
    output_np_pose = output_np_pose.reshape(-1, np_pose_shape[-2], np_pose_shape[-1])
    return output_np_pose


if __name__ == '__main__':
    # read arguments
    args = parse_args()

    GT_pose = MB_input_pose_file_loader(args)
    estimate_pose = MB_output_pose_file_loader(args)
    assert GT_pose.shape == estimate_pose.shape, f"GT_pose.shape: {GT_pose.shape}, estimate_pose.shape: {estimate_pose.shape}, they should be the same"


    if args.debug_mode:
        small_sample = 7500
        GT_pose = GT_pose[:small_sample]
        estimate_pose = estimate_pose[:small_sample]

    # Step 1: calculate GT angles
    GT_skeleton = VEHSErgoSkeleton_angles(args.skeleton_file)
    GT_skeleton.load_name_list_and_np_points(args.name_list, GT_pose)
    GT_ergo_angles = {}
    for angle_name in GT_skeleton.angle_names:  # calling the angle calculation methods in skeleton class
        class_method_name = f'{angle_name}_angles'
        GT_ergo_angles[angle_name] = getattr(GT_skeleton, class_method_name)()

    # Step 2: calculate MB angles
    estimate_skeleton = VEHSErgoSkeleton_angles(args.skeleton_file)
    estimate_skeleton.load_name_list_and_np_points(args.name_list, estimate_pose)
    estimate_ergo_angles = {}
    for angle_name in estimate_skeleton.angle_names:  # calling the angle calculation methods in skeleton class
        class_method_name = f'{angle_name}_angles'
        estimate_ergo_angles[angle_name] = getattr(estimate_skeleton, class_method_name)()

    # Step 3: visualize
    
    # Hi Veeru, I used this to visualize the 3D pose frame by frame
    frame = 10210
    frame = 10180
    GT_skeleton.plot_3d_pose_frame(frame)
    estimate_skeleton.plot_3d_pose_frame(frame)

    frame_range = [101800, 102200]
    log = []
    target_angles = GT_skeleton.angle_names
    # target_angles = ['right_shoulder']
    for angle_index, this_angle_name in enumerate(target_angles):
        pass
        # plot angles
        # GT_fig, GT_ax = GT_ergo_angles[this_angle_name].plot_angles(joint_name=f"GT-{this_angle_name}", frame_range=frame_range, alpha=0.75, colors=['g', 'g', 'g'])
        # estimate_fig, _ = estimate_ergo_angles[this_angle_name].plot_angles(joint_name=f"Est-{this_angle_name}", frame_range=frame_range, alpha=0.75, colors=['r', 'r', 'r'], overlay=[GT_fig, GT_ax])
        # plt.show()
        # GT_fig.savefig(f'frames/MB_angles/GT-{this_angle_name}.png')
        # estimate_fig.savefig(f'frames/MB_angles/Est-{this_angle_name}.png')
    #
    #     ergo_angle_name = ['flexion', 'abduction', 'rotation']
    #     print_ergo_names = getattr(estimate_ergo_angles[this_angle_name], 'ergo_name')
    #     print_angle_name = this_angle_name.replace('_', '').replace('right', 'R-').replace('left', 'L-').capitalize()
    #     for this_ergo_angle in ergo_angle_name:
    #         ja1 = getattr(estimate_ergo_angles[this_angle_name], this_ergo_angle)
    #         ja2 = getattr(GT_ergo_angles[this_angle_name], this_ergo_angle)
    #         print_ergo_name = print_ergo_names[this_ergo_angle].capitalize()
    #         if ja1 is not None:
    #             print("=====================================")
    #             print(f'{this_angle_name} - {this_ergo_angle}')
    #             # bland-Altman plot
    #             md, sd = bland_altman_plot(ja1, ja2, title=f'{print_angle_name}: {print_ergo_name}', save_path=f'frames/MB_angles/BA_plots/{angle_index}-{this_angle_name}-{this_ergo_angle}.png')
    #             print(f'Bland Altman: md: {md:.2f}, sd: {sd:.2f}')
    #
    #             RMSE = root_mean_squared_error(ja1, ja2)
    #             MAE = mean_absolute_error(ja1, ja2)
    #             print(f'MAE: {MAE:.2f}, RMSE: {RMSE:.2f}')
    #             this_log = [this_angle_name, this_ergo_angle, md, sd, MAE, RMSE]
    #             log.append(this_log)
    # print(f"Store location: {'frames/MB_angles/BA_plots/'}")
    # # print log as csv in console
    # print("angle_name,ergo_angle,diff_md,dif_sd,MAE,RMSE")
    # for i in log:
    #     for j in i:
    #         print(j, end=",")
    #     print()


    # generate merged bland-altman plot for left and right


""" powershell batch crop png files
cd frames/MB_angles/BA_plots
Remove-Item .\cropped\* -Recurse
$files = Get-ChildItem
foreach ($file in $files) {
    ffmpeg -i $file -vf "crop=515:450:74:22" "cropped/$file"
}
"""

""" ffmpeg crop command
ffmpeg -i back-abduction.png -vf "crop=515:430:74:22" output.jpg
"""



