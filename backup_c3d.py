import os
import datetime
from viconnexusapi import ViconNexus
import argparse

# read arguments
parser = argparse.ArgumentParser()
parser.add_argument('--note', type=str, default=None, help='Status of the backup, e.g. filtered, raw, corrected, gap filled etc.')


vicon = ViconNexus.ViconNexus()
trial_name = vicon.GetTrialName()

# create a new backup folder if it does not exist
backup_folder = os.path.join(trial_name[0], 'backup')
if not os.path.exists(backup_folder):
    os.mkdir(backup_folder)

# csv_file = os.path.join(trial_name[0],trial_name[1]+'.filtered_already.csv')
# check if the csv file exists
c3d_note = parser.parse_args().note

file_name = os.path.join(trial_name[0],trial_name[1]+'.c3d')
time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
backup_file_name = os.path.join(backup_folder, trial_name[1] +f'_{time_stamp}_{c3d_note}.c3d')
os.system(f'copy "{file_name}" "{backup_file_name}"')
print(f'Backup file created: {backup_file_name}')