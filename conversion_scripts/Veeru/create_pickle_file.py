# File from Veeru to overwrite Vicon-MB pkl files with the RTMPose 2D pose and confidence score
# Step 1: c3d_to_MB.py to generate Vicon GT pkl file
# Step 2: veeru_format_snippet.py to delete two activities
# Step 3: create_pickle_file.py to overwrite with RTMPose 2D pose and confidence score

import pickle
import numpy as np
import copy

motionbert_pkl_file = r'W:\VEHS\VEHS data collection round 3\processed\VEHS_6D_downsample5_keep1_config6_tilt_corrected_modified.pkl'
own_pkl_file = r'W:\VEHS\Training_Folder\Training_Folder\rtmpose_vicon_dataset_j24_f20_s1_with_conf_score.pkl'
new_pkl_file = motionbert_pkl_file.replace('.pkl', '_RTM2D.pkl')

def read_pkl(data_url):
    file = open(data_url,'rb')
    content = pickle.load(file)
    file.close()
    return content


motionbert_data_dict = read_pkl(motionbert_pkl_file)
own_data_dict = read_pkl(own_pkl_file)

## Test to see if the data sequence is the same (subject, camera, activity)
key_map = {'train': 'train', 'val': 'validate', 'test': 'test'}

# subject_map = {'3-1': 'LeyangWen', '3-2': 'YuxuanZhu', '3-3': 'WenZhou', '3-4': 'JohnSohn', '3-5': 'DanLi',
#                   '3-6': 'XinyanWang', '3-7': 'SeunguLee', '3-8': 'FrancisBeak', '3-9': 'RuiNie', '3-10': 'ZoliaPeng'}
subject_map = {'3-1': 'S01', '3-2': 'S02', '3-3': 'S03', '3-4': 'S04', '3-5': 'S05',
               '3-6': 'S06', '3-7': 'S07', '3-8': 'S08', '3-9': 'S09', '3-10': 'S10'}

for key in own_data_dict.keys():
    print(f"===================={key}====================")
    sources = motionbert_data_dict[key_map[key]]['source']
    cameras = motionbert_data_dict[key_map[key]]['camera_name']
    store_camera = ''
    i_last = 0
    i_intervals = []
    assert_bundle_list = []
    for i in range(len(cameras)):
        if cameras[i] != store_camera:
            store_camera = cameras[i]
            subject_name = sources[i].split('\\')[-3]
            action_name = sources[i].split('\\')[-1].split('.')[0].lower()
            camera_id = cameras[i]
            assert_bundle_list.append([subject_name, camera_id, action_name])
            i_intervals.append(i - i_last)
            i_last = i
    i_intervals.append(i - i_last+1)
    assert_frame_number = i_intervals[1:]

    length = len(own_data_dict[key]['name'])
    for i in range(length):
        correct_frame_number = own_data_dict[key]['gt2d'][i].shape[0]
        correct_subject_id = own_data_dict[key]['subject'][i]
        correct_subject_name = subject_map[correct_subject_id]
        correct_camera_id = own_data_dict[key]['camera'][i]['id']
        correct_action_name = own_data_dict[key]['action'][i].lower()
        correct_bundle = [correct_subject_name, correct_camera_id, correct_action_name]
        assert_bundle = assert_bundle_list[i]

        print(f"RTM: {correct_bundle} {correct_frame_number} <==> Vicon: {assert_bundle} {assert_frame_number[i]}")
        assert correct_bundle == assert_bundle, f"RTM: {correct_bundle} != Vicon: {assert_bundle}"
        assert correct_frame_number == assert_frame_number[i], f"RTM: {correct_frame_number} != Vicon: {assert_frame_number[i]} for {correct_bundle}"

## test end

new_motionbert_dict = copy.deepcopy(motionbert_data_dict)

train_gt_2d = np.concatenate(own_data_dict['train']['gt2d'])
train_confidence = np.concatenate(own_data_dict['train']['confidence'])

new_motionbert_dict['train']['joint_2d'] = train_gt_2d
new_motionbert_dict['train']['confidence'] = train_confidence
# new_motionbert_dict['train']['joint3d_image'][:,:,:2] = train_gt_2d

val_gt_2d = np.concatenate(own_data_dict['val']['gt2d'])
val_confidence = np.concatenate(own_data_dict['val']['confidence'])

new_motionbert_dict['validate']['joint_2d'] = val_gt_2d
new_motionbert_dict['validate']['confidence'] = val_confidence
# new_motionbert_dict['validate']['joint3d_image'][:,:,:2] = val_gt_2d


del new_motionbert_dict['test']

pickle.dump(new_motionbert_dict, open(new_pkl_file, "wb"))

test_dict = read_pkl(new_pkl_file)

print("Saved to:")
print(new_pkl_file)




