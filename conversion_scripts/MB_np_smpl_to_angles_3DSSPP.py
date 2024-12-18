import argparse
import os.path
import pickle
from Skeleton import *
import matplotlib
matplotlib.use('Qt5Agg')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=r'config/experiment_config/mesh-compare.yaml')
    parser.add_argument('--skeleton_file', type=str, default=r'config/VEHS_ErgoSkeleton_info/Ergo-Skeleton-66.yaml')
    parser.add_argument("--input_type", choices=["mesh_17", "mesh_66", "3D", "6D"], default="3D")
    # parser.add_argument("--output_type", choices=["22angles", "3DSSPP"], default=["22angles"], nargs="+")

    parser.add_argument('--output_frame_folder', type=str, default=None)
    parser.add_argument('--output_GT_frame_folder', type=str, default=None)
    parser.add_argument('--plot_mode', choices=['normal_view', 'camera_view', 'camera_side_view', '0_135_view'], default='normal_view')
    parser.add_argument('--MB_data_stride', type=int, default=243)
    parser.add_argument('--debug_mode', default=True, type=bool)


    # parser.add_argument('--name_list', type=list, default=[])
    args = parser.parse_args()
    with open(args.config_file, 'r') as stream:
        data = yaml.safe_load(stream)
        args.eval_key = data['eval_key']
        args.root_dir = data['root_dir']

        args.GT_6D_name_list = data['GT_6D_name_list']
        args.GT_6D_file = os.path.join(args.root_dir,data['GT_6D_file'])

        args.estimate_3D_name_list = data['estimate_3D_name_list']
        args.estimate_3D_file = os.path.join(args.root_dir, data['estimate_3D_file'])
        if 'mesh_17' in args.input_type:
            args.estimate_mesh_file = os.path.join(args.root_dir, data['estimate_mesh_17_file'])
        elif 'mesh_66' in args.input_type:
            args.estimate_mesh_file = os.path.join(args.root_dir, data['estimate_mesh_66_file'])
    print(args.plot_mode)

    # args.output_frame_folder = os.path.dirname(args.estimate_file) if args.output_frame_folder is None else args.output_frame_folder
    # args.output_GT_frame_folder = os.path.dirname(args.GT_file) if args.output_GT_frame_folder is None else args.output_GT
    # args.output_frame_folder = os.path.join(args.output_frame_folder, args.plot_mode)
    # GT_base_folder = args.output_GT_frame_folder
    # args.output_GT_frame_folder = os.path.join(GT_base_folder, args.plot_mode)
    # args.output_2D_frame_folder = os.path.join(GT_base_folder, '2D')
    return args


def MB_output_pose_file_loader(args):
    if args.estimate_3D_file=='None':
        return None
    with open(args.estimate_3D_file, "rb") as f:
        output_np_pose = np.load(f)
    np_pose_shape = output_np_pose.shape
    output_np_pose = output_np_pose.reshape(-1, np_pose_shape[-2], np_pose_shape[-1])
    # todo: temp fix, need to run eval again with another dataset
    ## approx convert 50 fps to 20 fps
    # 1. double the frame rate
    output_np_pose = np.repeat(output_np_pose, 2, axis=0)
    # 2. take every 5th frame
    output_np_pose = output_np_pose[::5]
    return output_np_pose


def MB_input_pose_file_loader(args, file):
    if file=='None':
        return None
    # file = os.path.join(args.root_dir, args.GT_6D_file)
    with open(file, "rb") as f:
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

    # print(f'2.5d_factor: {data[args.eval_key]["2.5d_factor"]}')
    return np_pose


def MB_output_mesh_file_loader(args, file):
    if file=='None':
        return None
    with open(file, "rb") as f:
        output_mesh_pose = pickle.load(f)
    verts = output_mesh_pose['verts'].reshape(-1, 6890, 3)
    # verts_gt = output_mesh_pose['verts_gt'].reshape(-1, 6890, 3)
    # kp_3d = output_mesh_pose['kp_3d'].reshape(-1, 17, 3)
    return verts


def check_GT_file(args):
    with open(args.GT_file, "rb") as f:
        data = pickle.load(f)
    eval_keys = ['train', 'test', 'validate']
    args.eval_key = eval_keys[1]
    GT_pose = data[args.eval_key]['joint3d_image']

    # check for hands
    frame = 0
    print(f"Dist to middle base {np.linalg.norm(GT_pose[frame,5] - GT_pose[frame,0])} px")
    print(f"{(np.linalg.norm(GT_pose[frame, 5] - GT_pose[frame, 0])) * data[args.eval_key]['2.5d_factor'][frame]} mm")

    i = 0
    j = 0
    for frame in range(1, len(GT_pose)-1):
        frame_source = data[args.eval_key]['source'][frame]
        next_source = data[args.eval_key]['source'][frame+1]

        ## check 2: travel between frames
        travel = np.linalg.norm(GT_pose[frame,5] - GT_pose[frame + 1,5])
        if travel > 11.18*1000/20 and frame_source == next_source:  # 11.18 m/s boxer punch speed, 100 fps
            j += 1
            print(f"j-{j} - ave travel between frame {frame} &+1: {travel}")

        ## check 1: middle finger dist
        px_dist = np.linalg.norm(GT_pose[frame, 5] - GT_pose[frame, 0])
        mm_dist = px_dist * data[args.eval_key]['2.5d_factor'][frame]
        next_px_dist = np.linalg.norm(GT_pose[frame+1, 5] - GT_pose[frame+1, 0])
        next_mm_dist = next_px_dist * data[args.eval_key]['2.5d_factor'][frame+1]
        if abs(mm_dist-next_mm_dist)  > next_mm_dist*0.2 and frame_source == next_source:
            # print(f"Dist to middle base in frame {frame} &+1: {px_dist} - {next_px_dist} = {px_dist-next_px_dist} px")
            print(f"Dist to middle base in frame {frame} &+1: {mm_dist} - {next_mm_dist} = {mm_dist-next_mm_dist} mm")

        ## check 2.5
        if data[args.eval_key]['2.5d_factor'][frame]-data[args.eval_key]['2.5d_factor'][frame+1] > 3 and frame_source == next_source:
            i+=1
            print(f"i-{i} - 2.5d factor in frame {frame} &+1: {data[args.eval_key]['2.5d_factor'][frame]} - {data[args.eval_key]['2.5d_factor'][frame+1]} = {data[args.eval_key]['2.5d_factor'][frame]-data[args.eval_key]['2.5d_factor'][frame+1]}")




    frame = 120
    # frame = 39747
    # frame = 0
    GT_skeleton.plot_3d_pose_frame(frame=frame, coord_system="camera-px", plot_range=2000, mode='normal_view', center_key='Wrist')
    GT_skeleton.plot_3d_pose_frame(frame=frame+1, coord_system="camera-px", plot_range=2000, mode='normal_view', center_key='Wrist')
    print(data[args.eval_key]['source'][frame])
    print(data[args.eval_key]['source'][frame+1])



if __name__ == '__main__':
    # read arguments
    args = parse_args()

    if '3D' == args.input_type:
        pass
        estimate_3D_pose = MB_output_pose_file_loader(args)
        estimate_3D_pose = estimate_3D_pose[:1000] if args.debug_mode else estimate_3D_pose
        # Step 2: calculate MB-6D angles
        estimate_3D_skeleton = H36MSkeleton_angles(args.skeleton_file)
        estimate_3D_skeleton.load_name_list_and_np_points(args.estimate_3D_name_list, estimate_3D_pose)
        estimate_3D_ergo_angles = {}
        for angle_name in estimate_3D_skeleton.angle_names:  # calling the angle calculation methods in skeleton class
            class_method_name = f'{angle_name}_angles'
            estimate_3D_ergo_angles[angle_name] = getattr(estimate_3D_skeleton, class_method_name)()
        estimate_ergo_angles = estimate_3D_ergo_angles
    elif '6D' == args.input_type:
        # estimate_6D_pose = MB_output_pose_file_loader(args)
        # # Step 2: calculate MB-6D angles
        # estimate_6D_skeleton = VEHSErgoSkeleton_angles(args.skeleton_file)
        # estimate_6D_skeleton.load_name_list_and_np_points(args.name_list, estimate_6D_pose)
        # estimate_6D_ergo_angles = {}
        # for angle_name in estimate_6D_skeleton.angle_names:  # calling the angle calculation methods in skeleton class
        #     class_method_name = f'{angle_name}_angles'
        #     estimate_6D_ergo_angles[angle_name] = getattr(estimate_6D_skeleton, class_method_name)()
        pass
    elif 'mesh' in args.input_type:
        estimate_mesh_vert = MB_output_mesh_file_loader(args, args.estimate_mesh_file)
        estimate_mesh_vert = estimate_mesh_vert[:1000] if args.debug_mode else estimate_mesh_vert
        # calculate MB-mesh angles
        estimate_mesh_skeleton = VEHSErgoSkeleton_angles(args.skeleton_file)
        estimate_mesh_skeleton.load_mesh(estimate_mesh_vert)
        estimate_mesh_skeleton.calculate_joint_center()
        estimate_mesh_ergo_angles = {}
        for angle_name in estimate_mesh_skeleton.angle_names:  # calling the angle calculation methods in skeleton class
            class_method_name = f'{angle_name}_angles'
            estimate_mesh_ergo_angles[angle_name] = getattr(estimate_mesh_skeleton, class_method_name)()
        estimate_mesh_ergo_angles['back'] = estimate_mesh_skeleton.back_angles(up_axis=[0, 0, -1])

        estimate_ergo_angles = estimate_mesh_ergo_angles
    # Step 2: Get GT angles to compare
    GT_pose = MB_input_pose_file_loader(args, args.GT_6D_file)
    GT_pose = GT_pose[:1000] if args.debug_mode else GT_pose
    # calculate GT angles
    GT_skeleton = VEHSErgoSkeleton_angles(args.skeleton_file)
    GT_skeleton.load_name_list_and_np_points(args.GT_6D_name_list, GT_pose)
    GT_ergo_angles = {}
    for angle_name in GT_skeleton.angle_names:  # calling the angle calculation methods in skeleton class
            class_method_name = f'{angle_name}_angles'
            GT_ergo_angles[angle_name] = getattr(GT_skeleton, class_method_name)()



    # Step 3: visualize
    frame = 400
    # GT_skeleton.plot_3d_pose_frame(frame, coord_system="camera-px", plot_range=1000)
    # estimate_mesh_skeleton.plot_3d_pose_frame(frame, coord_system="world")

    frame_range = [0, 1000]
    log = []
    target_angles = GT_skeleton.angle_names
    # target_angles = ['right_shoulder']
    for angle_index, this_angle_name in enumerate(target_angles):
        # plot angles
        try:
            estimate_ergo_angles[this_angle_name]
        except:
            print(f"Angle {this_angle_name} not found in {args.input_type} estimate")
            continue
        GT_fig, GT_ax = GT_ergo_angles[this_angle_name].plot_angles(joint_name=f"GT-{this_angle_name}", frame_range=frame_range, alpha=0.75, colors=['g', 'g', 'g'])
        estimate_fig, estimate_ax = estimate_ergo_angles[this_angle_name].plot_angles(joint_name=f"Est-{this_angle_name}", frame_range=frame_range, alpha=0.75, colors=['r', 'r', 'r'], overlay=[GT_fig, GT_ax])
        plt.show()
        merge_angle_dir = os.path.join(args.root_dir, f'frames/MB_angles/Est-{this_angle_name}.png')
        if not os.path.exists(os.path.dirname(merge_angle_dir)):
            os.makedirs(os.path.dirname(merge_angle_dir))
        estimate_fig.savefig(merge_angle_dir)

        ergo_angle_name = ['flexion', 'abduction', 'rotation']
        print_ergo_names = getattr(estimate_ergo_angles[this_angle_name], 'ergo_name')
        print_angle_name = this_angle_name.replace('_', '').replace('right', 'R-').replace('left', 'L-').capitalize()
        for this_ergo_angle in ergo_angle_name:
            ja1 = getattr(estimate_ergo_angles[this_angle_name], this_ergo_angle)
            ja2 = getattr(GT_ergo_angles[this_angle_name], this_ergo_angle)
            print_ergo_name = print_ergo_names[this_ergo_angle].capitalize()
            if ja1 is not None:
                print("=====================================")
                print(f'{this_angle_name} - {this_ergo_angle}')
                # bland-Altman plot
                # md, sd = bland_altman_plot(ja1, ja2, title=f'{print_angle_name}: {print_ergo_name}',
                #                            save_path=f'frames/MB_angles/BA_plots/{angle_index}-{this_angle_name}-{this_ergo_angle}.png')
                # print(f'Bland Altman: md: {md:.2f}, sd: {sd:.2f}')
                md = 0
                sd = 0

                RMSE = root_mean_squared_error(ja1, ja2)
                MAE = mean_absolute_error(ja1, ja2)
                print(f'MAE: {MAE:.2f}, RMSE: {RMSE:.2f}')
                this_log = [this_angle_name, this_ergo_angle, md, sd, MAE, RMSE]
                log.append(this_log)
    print(f"Store location: {'frames/MB_angles/BA_plots/'}")
    # print log as csv in console
    print("angle_name,ergo_angle,diff_md,dif_sd,MAE,RMSE")
    for i in log:
        for j in i:
            print(j, end=",")
        print()

    # generate merged bland-altman plot for left and right







