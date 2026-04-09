import argparse
import os.path
import pickle

import numpy as np

from Skeleton import *
import matplotlib
from ergo3d import Point
matplotlib.use('Qt5Agg')
import csv
from SSPPOutput import SSPPV7Output

####################################################
# From 3DSSPP export output txt file
# Read and plot the results

####################################################

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--print_all', default=True, help='Print all frames without cutting segments')

    # parser.add_argument('--export_file', type=str, default=r'/Users/leyangwen/Library/CloudStorage/OneDrive-Umich/isaac_3dsspp/intervention_eval_data/Generated_flat_results-3DSSPP_export.txt')
    # parser.add_argument('--export_file', type=str, default=r'/Users/leyangwen/Library/CloudStorage/OneDrive-Umich/isaac_3dsspp/intervention_eval_data/Generated_step_results-3DSSPP_export.txt')
    # parser.add_argument('--export_file', type=str, default=r'/Users/leyangwen/Library/CloudStorage/OneDrive-Umich/isaac_3dsspp/intervention_eval_data/Mocap_results-3DSSPP_export.txt')

    parser.add_argument('--export_file', type=str, default=r'/Users/leyangwen/Library/CloudStorage/OneDrive-Umich/isaac_3dsspp/intervention_eval_data/export_results/flat_export.txt')
    # parser.add_argument('--export_file', type=str, default=r'/Users/leyangwen/Library/CloudStorage/OneDrive-Umich/isaac_3dsspp/intervention_eval_data/export_results/mocap_export.txt')
    # parser.add_argument('--export_file', type=str, default=r'/Users/leyangwen/Library/CloudStorage/OneDrive-Umich/isaac_3dsspp/intervention_eval_data/export_results/step_export.txt')


    # parser.add_argument('--print_all', default=False, help='Print all frames without cutting segments')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # read arguments
    args = parse_args()
    # read est file csv as a numpy array

    result = SSPPV7Output()
    result.load_file(args.export_file)

    eval_keys = result.show_category(subcategory='Summary')[:-3]
    hand_force_keys = [result.show_category(subcategory='Hand Forces')[0], result.show_category(subcategory='Hand Forces')[3]]
    # eval_keys = result.show_category(subcategory='Summary')[:2]
    # result.visualize_segment(result.all_segments, segment_eval_keys=eval_keys, verbose=True)

    eval_keys_SCP = result.show_category(subcategory='Summary')[2:-3]
    eval_keys_Compression = result.show_category(subcategory='Summary')[:2]
    csv_file = args.export_file.replace('.txt', '.csv')

    if args.print_all:
        # Print all frames without cutting segments
        task_name_key = 'Info - Task Name'
        frame_key = 'Info - Frame'
        output_keys = eval_keys_Compression + eval_keys_SCP + hand_force_keys
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["filename", csv_file])
            writer.writerow([task_name_key, frame_key] + output_keys)
            for i in range(len(result.df)):
                row = result.df.iloc[i]
                task_name = row[task_name_key] if task_name_key in result.df.columns else ''
                frame_num = row[frame_key] if frame_key in result.df.columns else i
                values = [row[k] for k in output_keys]
                writer.writerow([task_name, frame_num] + values)
        print(f"All frames written to {csv_file}")
    else:
        _, unique_segment_len = result.cut_segment()
        # write to csv
        # n = len(result.segments)//2
        # lift_lower = ['Lift', 'Lower']
        # start_heights = [0.2, 0.4, 0.6, 0.8]
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["filename", csv_file])
            writer.writerow(['Segment Index',] + eval_keys_Compression + eval_keys_SCP + hand_force_keys)
            for segment in result.segments:
                # segment = 4
                print()
                print("##################################################")
                print(f"index: {segment}")
                # eval_keys = result.show_category(subcategory='Strength Capability Percentile')
                output_frame = 0
                _, min_score, scores, _ = result.eval_segment(result.segments[segment], eval_keys_SCP, verbose=True, criteria=output_frame)
                _, _, compression_forces, _ = result.eval_segment(result.segments[segment], eval_keys_Compression, verbose=True, criteria=output_frame)
                _, _, hand_forces, _ = result.eval_segment(result.segments[segment], hand_force_keys, verbose=True, criteria=output_frame)
                writer.writerow([segment,] + list(compression_forces) + list(scores) + list(hand_forces))


