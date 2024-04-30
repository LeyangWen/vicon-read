# remove activity 4 for subject 2 and activity 1 for subject 3, so that it matches Veeru's RTMPose dataset format
import pickle
import os
import numpy as np

pkl_file = r"W:\VEHS\VEHS data collection round 3\processed\VEHS_6D_downsample5_keep1_config6.pkl"
with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

for key in data.keys():
    id_list = []
    print(f"{key} - {data[key].keys()}")
    #
    source = data[key]['source']
    for id, src in enumerate(source):
        keyword1 = 'S02\\FullCollection\\Activity04.c3d'
        keyword2 = 'S03\\FullCollection\\Activity01.c3d'
        flag = False
        for keyword in [keyword1, keyword2]:
            if keyword in src:
                flag = True
        if not flag:
            id_list.append(id)
    print(f"{key}: {len(id_list)} out of {len(source)}")
    # keep ids in data
    if len(id_list) != len(source):
        data[key]['source'] = [source[i] for i in id_list]
        data[key]['joint_2d'] = np.array([data[key]['joint_2d'][i] for i in id_list])
        data[key]['confidence'] = np.array([data[key]['confidence'][i] for i in id_list])
        data[key]['joint3d_image'] = np.array([data[key]['joint3d_image'][i] for i in id_list])
        data[key]['camera_name'] = np.array([data[key]['camera_name'][i] for i in id_list])
        data[key]['c3d_frame'] = [data[key]['c3d_frame'][i] for i in id_list]
        try:
            data[key]['action'] = np.array([data[key]['action'][i] for i in id_list])
            data[key]['joints_2.5d_image'] = np.array([data[key]['joints_2.5d_image'][i] for i in id_list])
            data[key]['2.5d_factor'] = np.array([data[key]['2.5d_factor'][i] for i in id_list])
        except:
            pass


# save pkl file
save_pkl_file = pkl_file.replace('.pkl', '_modified.pkl')
with open(save_pkl_file, 'wb') as f:
    pickle.dump(data, f)