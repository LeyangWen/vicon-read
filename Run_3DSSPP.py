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

    # -s 340 -e 3300 -s 10
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--start', type=int, default=100)
    parser.add_argument('-e', '--end', type=int, default=1700)
    parser.add_argument('--step', type=int, default=5)
    # optional config files
    parser.add_argument('--skeleton_file', type=str, default=r'C:\Users\Public\Documents\Vicon\vicon_coding_projects\vicon-read\config\VEHS_ErgoSkeleton_info\Ergo-Skeleton-66.yaml')

    args = parser.parse_args()
    start = args.start
    end = args.end
    step = args.step
    skeleton_file = args.skeleton_file

    # get the c3d file from Vicon Nexus
    vicon = ViconNexus.ViconNexus()
    trial_name = vicon.GetTrialName()
    subject_names = vicon.GetSubjectNames()
    weight = vicon.GetSubjectParam(subject_names[0], 'Bodymass')[0]
    height = vicon.GetSubjectParam(subject_names[0], 'Height')[0]
    gender = vicon.GetSubjectParam(subject_names[0], 'Gender_1M_2F_3O')[0]
    gender = "female" if gender==2 else "male"

    print(subject_names,weight, height, gender, trial_name)
    c3d_file = os.path.join(trial_name[0], trial_name[1] + '.c3d')
    # skeleton = Skeleton.PulginGaitSkeleton(c3d_file, skeleton_file=skeleton_file, acronym_file=acronym_file)
    skeleton = Skeleton.VEHSErgoSkeleton(skeleton_file)
    skeleton.load_c3d(c3d_file, analog_read=False, verbose=False)
    skeleton.calculate_joint_center()
    skeleton.set_weight_height(weight=weight, height=height/10)
    skeleton.set_gender(gender)
    # skeleton.set_weight_height(80, 1.844)
    skeleton.output_3DSSPP_loc(frame_range=[start,end,step])