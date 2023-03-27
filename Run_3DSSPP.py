import Skeleton
from viconnexusapi import ViconNexus
import os
import yaml
import argparse


if __name__ == '__main__':
    # # manually specify the config file
    # config_file = r'F:\wen_storage\test\VEHS_ske\Test\Gunwoo\Test1\Gunwoo movement 02.yaml'
    # with open(config_file, 'r') as stream:
    #     try:
    #         data = yaml.safe_load(stream)
    #         c3d_file = data['c3d_file']
    #     except yaml.YAMLError as exc:
    #         print(config_file, exc)

    # use argparse to specify start, end, and step
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--start', type=int, default=0)
    parser.add_argument('-e', '--end', type=int, default=1000)
    parser.add_argument('-st', '--step', type=int, default=5)
    # optional config files
    parser.add_argument('--skeleton_file', type=str, default=r'C:\Users\Public\Documents\Vicon\vicon_coding_projects\vicon-read\config/Plugingait_info/plugingait_VEHS.yaml')
    parser.add_argument('--acronym_file', type=str, default=r'C:\Users\Public\Documents\Vicon\vicon_coding_projects\vicon-read\config/Plugingait_info/acronym.yaml')

    args = parser.parse_args()
    start = args.start
    end = args.end
    step = args.step
    skeleton_file = args.skeleton_file
    acronym_file = args.acronym_file

    # get the c3d file from Vicon Nexus
    vicon = ViconNexus.ViconNexus()
    trial_name = vicon.GetTrialName()
    subject_names = vicon.GetSubjectNames()
    weight = vicon.GetSubjectParam(subject_names[0], 'Bodymass')[0]
    height = vicon.GetSubjectParam(subject_names[0], 'Height')[0]

    c3d_file = os.path.join(trial_name[0],trial_name[1]+ '.c3d')
    skeleton = Skeleton.PulginGaitSkeleton(c3d_file, skeleton_file=skeleton_file, acronym_file=acronym_file)
    skeleton.set_weight_height(weight=weight, height=height/1000)
    # skeleton.set_weight_height(80, 1.844)
    skeleton.output_3DSSPP_loc(frame_range=[start,end,step])