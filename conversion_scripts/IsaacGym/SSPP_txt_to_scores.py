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
    parser.add_argument('--config_file', type=str, default=r'config/experiment_config/IssacGym/test.yaml')
    parser.add_argument('--output_folder', type=str, default=None)


    # parser.add_argument('--name_list', type=list, default=[])
    args = parser.parse_args()
    with open(args.config_file, 'r') as stream:
        data = yaml.safe_load(stream)
        args.name_list = data['name_list']
        args.estimate_file = data['estimate_file']
        args.fps = data['fps']
        args.dim = data['dim']
        args.density = data['density']
    args.export_file = args.estimate_file.replace('.csv', '-3DSSPP_export.txt')
    print(args.estimate_file)
    args.mass = args.density * args.dim[0] * args.dim[1] * args.dim[2]  # kg
    args.output_folder = os.path.dirname(args.estimate_file) if args.output_folder is None else args.output_folder
    return args


if __name__ == '__main__':
    # read arguments
    args = parse_args()
    # read est file csv as a numpy array

    result = SSPPV7Output()
    result.load_file(args.export_file)
    _, unique_segment_len = result.cut_segment()
    eval_keys = result.show_category(subcategory='Summary')[:-3]
    # eval_keys = result.show_category(subcategory='Summary')[:2]
    # result.visualize_segment(result.all_segments, segment_eval_keys=eval_keys, verbose=True)

    eval_keys_SCP = result.show_category(subcategory='Summary')[2:-3]
    eval_keys_Compression = result.show_category(subcategory='Summary')[:2]
    # write to csv
    csv_file = args.export_file.replace('.txt', '.csv')
    n = len(result.segments)//2
    lift_lower = ['Lift', 'Lower']
    start_heights = [0.2, 0.4, 0.6, 0.8]
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", csv_file])
        writer.writerow(['Segment Index', 'Lift/lower', 'Lift/lower Height (m)'] + eval_keys_Compression + eval_keys_SCP)
        for segment in result.segments:
            # segment = 4
            print()
            print("##################################################")
            print(f"index: {segment}")
            # eval_keys = result.show_category(subcategory='Strength Capability Percentile')

            _, min_score, scores, _ = result.eval_segment(result.segments[segment], eval_keys_SCP, verbose=True, criteria='min_min')
            _, _, compression_forces, _ = result.eval_segment(result.segments[segment], eval_keys_Compression, verbose=True, criteria='min_max')
            writer.writerow([segment, lift_lower[segment//n], "start_heights[segment%n]"] + list(compression_forces) + list(scores))


