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
    result.visualize_segment(result.all_segments, segment_eval_keys=eval_keys, verbose=True)

    eval_keys = result.show_category(subcategory='Strength Capability Percentile')
    for segment in result.segments:
        # segment = 4
        print()
        print("##################################################")
        print(f"index: {segment}")
        _, min_score, scores, _ = result.eval_segment(result.segments[segment], eval_keys, verbose=True, criteria='min_min')
        print(f"min_score: {min_score}")