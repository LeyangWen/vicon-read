import c3d
# from spacepy import pycdf
import os
import matplotlib.pyplot as plt
import numpy as np
import datetime
from mpl_toolkits.mplot3d import Axes3D
import json
from tqdm import tqdm

def plot_joint_axis(joint_axis_pts,label=None):
    # example
    # idx = [75,79] # head
    # idx = [75,79] # head
    # joint_axis_pts = points[idx[0]:idx[1]]
    # plot_joint_axis(joint_axis_pts,label = reader.point_labels[idx[0]])
    if label:
        print(label)
    # sgmentNameO segment Origin
    # segmentNameA Anterior axis
    # segmentNameP Proximal axis
    # segmentNameL Lateral axis
    axis_name = ['Anterior axis',
                'Lateral axis',
                'Proximal axis']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(joint_axis_pts[:, 0], joint_axis_pts[:, 1], joint_axis_pts[:, 2], c='r', marker='o')
    ax.set_xlabel('X (mm))')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    # connect points to joint_axis_pts[0]
    for i in range(1,4):
        ax.plot([joint_axis_pts[0, 0], joint_axis_pts[i, 0]],
                [joint_axis_pts[0, 1], joint_axis_pts[i, 1]],
                [joint_axis_pts[0, 2], joint_axis_pts[i, 2]])
        #print the length of each line
        #show axis name on the plot
        ax.text(joint_axis_pts[i, 0], joint_axis_pts[i, 1], joint_axis_pts[i, 2], axis_name[i - 1])
        print('length of {}: {}'.format(axis_name[i - 1], np.linalg.norm(joint_axis_pts[0] - joint_axis_pts[i])))
    ax.text(joint_axis_pts[0, 0], joint_axis_pts[0, 1], joint_axis_pts[0, 2], label)
    plt.show()


def create_dir(directory, is_base_dir=True):
    if is_base_dir:
        if not os.path.exists(directory):
            os.makedirs(directory)
    else:
        base_dir = os.path.dirname(directory)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)


def BMI_classUS(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Healthy weight'
    elif bmi < 30:
        return 'Overweight'
    elif bmi == 0:
        return 'Not available'
    else:
        return 'Obesity'


def BMI_caculate(weight, height):
    try:
        bmi = weight / (height / 100) ** 2
    except ZeroDivisionError:
        bmi = 0
    return bmi


def dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


def dist_array(p1s, p2s):
    return np.sqrt((p1s[:, 0] - p2s[:, 0]) ** 2 + (p1s[:, 1] - p2s[:, 1]) ** 2 + (p1s[:, 2] - p2s[:, 2]) ** 2)


def store_cdf(file_name, data, date='', kp_names='', subjectID='', TaskID='', CamID = '', jointName='', bbox=np.array([])):
    create_dir(os.path.dirname(file_name))
    if os.path.exists(file_name):
        os.remove(file_name)
    cdf = pycdf.CDF(file_name, '')
    cdf['Pose'] = data
    cdf.attrs['SubjectID'] = subjectID
    cdf.attrs['TaskID'] = TaskID
    cdf.attrs['CamID'] = CamID
    cdf.attrs['UpdateDate'] = datetime.datetime.now()
    cdf.attrs['CaptureDate'] = os.path.basename(date)
    cdf.attrs['KeypointNames'] = kp_names
    cdf['bbox'] = bbox
    cdf.close()

def empty_MotionBert_dataset_dict(joint_number, version="normal"):
    '''
    usage example:
    h36m_joint_names = ['PELVIS', 'RHIP', 'RKNEE', 'RANKLE', 'LHIP', 'LKNEE', 'LANKLE', 'T8', 'THORAX', 'C7', 'HEAD', 'LSHOULDER', 'LELBOW', 'LWRIST', 'RSHOULDER', 'RELBOW', 'RWRIST']
    output_3D_dataset = empty_MotionBert_dataset_dict(len(h36m_joint_names))  # 17
    custom_6D_joint_names = ['RPSIS', 'RASIS', 'LPSIS', 'LASIS', 'C7_d', 'SS', 'T8', 'XP', 'C7', 'HDTP', 'REAR', 'LEAR', 'RAP', 'RAP_f', 'RLE', 'RAP_b', 'RME', 'LAP', 'LAP_f', 'LLE', 'LAP_b', 'LME', 'LUS', 'LRS', 'RUS', 'RRS', 'RMCP5', 'RMCP2', 'LMCP5', 'LMCP2', 'LGT', 'LMFC', 'LLFC', 'RGT', 'RMFC', 'RLFC', 'RMM', 'RLM', 'LMM', 'LLM', 'LMTP1', 'LMTP5', 'LHEEL', 'RMTP1', 'RMTP5', 'RHEEL', 'HEAD', 'RSHOULDER', 'LSHOULDER', 'C7_m', 'THORAX', 'LELBOW', 'RELBOW', 'RWRIST', 'LWRIST', 'RHAND', 'LHAND', 'PELVIS', 'RHIP', 'RKNEE', 'RANKLE', 'RFOOT', 'LHIP', 'LKNEE', 'LANKLE', 'LFOOT']
    output_6D_dataset = empty_MotionBert_dataset_dict(len(custom_6D_joint_names))  # 66
    '''
    if version == "normal":  # V1 normal MB training
        return {
            'train': {
                'joint_2d': np.empty((0, joint_number, 2)),
                'confidence': np.empty((0, joint_number, 1)),
                'joint3d_image': np.empty((0, joint_number, 3)),
                'camera_name': np.empty((0,)),
                'source': [],
                'c3d_frame': []
            },
            'validate': {
                'joint_2d': np.empty((0, joint_number, 2)),
                'confidence': np.empty((0, joint_number, 1)),
                'joint3d_image': np.empty((0, joint_number, 3)),
                'joints_2.5d_image': np.empty((0, joint_number, 3)),
                '2.5d_factor': np.empty((0,)),
                'camera_name': np.empty((0,)),
                'action': [],
                'source': [],
                'c3d_frame': []
            },
            'test': {
                'joint_2d': np.empty((0, joint_number, 2)),
                'confidence': np.empty((0, joint_number, 1)),
                'joint3d_image': np.empty((0, joint_number, 3)),
                'joints_2.5d_image': np.empty((0, joint_number, 3)),
                '2.5d_factor': np.empty((0,)),
                'camera_name': np.empty((0,)),
                'action': [],
                'source': [],
                'c3d_frame': []
            }
        }
    elif version == "diversity_metric":  # V2 for diversity metric calculation only
        return {
            'train': {
                'joints_2.5d_image': np.empty((0, joint_number, 3)),
                'source': [],
            },
            'validate': {
                'joints_2.5d_image': np.empty((0, joint_number, 3)),
                'source': [],
            },
            'test': {
                'joints_2.5d_image': np.empty((0, joint_number, 3)),
                'source': [],
            }
        }

def empty_COCO_dataset_dict(joint_number):
    """
    {"info" : info, "images" : [image], "annotations" : [annotation], "licenses" : [license] }

    info{
    "year" : int, "version" : str, "description" : str, "contributor" : str, "url" : str, "date_created" : datetime,
    }

    image{
    "id" : int, "width" : int, "height" : int, "file_name" : str, "license" : int, "flickr_url" : str, "coco_url" : str, "date_captured" : datetime,
    }

    license{
    "id" : int, "name" : str, "url" : str,
    }

    annotation[0]{
    "keypoints" : [x1,y1,v1,...], "num_keypoints" : int, "[cloned]" : ...,
    }

    categories[{
    "keypoints" : [str], "skeleton" : [edge], "[cloned]" : ...,
    }]

    "[cloned]": denotes fields copied from object detection annotations defined above.
    """
    output = {}
    for this_train_val_test in ['train', 'validate', 'test']:
        empty_dict = {}
        empty_dict['info'] = {
            "year": 2023,
            "version": "1.0",
            "description": "VEHS-7M-5fps-Ergo37kpts",
            "contributor": "Leyang Wen",
            "url": ""}
        empty_dict['licenses'] = [
            {"url": "", "id": 99, "name": "VEHS-internal-use-only"}
        ]
        empty_dict['images'] = []
        empty_dict['annotations'] = []
        empty_dict['categories'] = [{
            "supercategory": "person",
            "id": 1,
            "name": "person",
            "keypoints": ['PELVIS', 'RWRIST', 'LWRIST', 'RHIP', 'LHIP',
                          'RKNEE', 'LKNEE', 'RANKLE', 'LANKLE', 'RFOOT',
                          'LFOOT', 'RHAND', 'LHAND', 'RELBOW', 'LELBOW',
                          'RSHOULDER', 'LSHOULDER', 'HEAD', 'THORAX', 'HDTP',
                          'REAR', 'LEAR', 'C7', 'C7_d', 'SS',
                          'RAP_b', 'RAP_f', 'LAP_b', 'LAP_f', 'RLE',
                          'RME', 'LLE', 'LME', 'RMCP2', 'RMCP5',
                          'LMCP2', 'LMCP5'],
            "skeleton": [
                        [1, 4],   # PELVIS to RHIP
                        [1, 5],   # PELVIS to LHIP
                        [4, 6],   # RHIP to RKNEE
                        [5, 7],   # LHIP to LKNEE
                        [6, 8],   # RKNEE to RANKLE
                        [7, 9],   # LKNEE to LANKLE
                        [8, 10],  # RANKLE to RFOOT
                        [9, 11],  # LANKLE to LFOOT
                        [2, 13],  # RWRIST to RELBOW
                        [3, 15],  # LWRIST to LELBOW
                        [13, 17], # RELBOW to RSHOULDER
                        [15, 18], # LELBOW to LSHOULDER
                        [12, 13], # RHAND to RELBOW
                        [14, 15], # LHAND to LELBOW
                        [17, 18], # RSHOULDER to LSHOULDER
                        [18, 19], # LSHOULDER to THORAX
                        [19, 20], # THORAX to HDTP
                        [20, 21], # HDTP to REAR
                        [20, 22], # HDTP to LEAR
                        [23, 24], # C7 to C7_d
                        [25, 26], # SS to RAP_b
                        [27, 28], # LAP_b to LAP_f
                        [29, 30], # RLE to RME
                        [31, 32], # LLE to LME
                        [33, 34], # RMCP2 to RMCP5
                        [35, 36]  # LMCP2 to LMCP5
                    ]
                }]
        output[this_train_val_test] = empty_dict
    return output



def append_output_xD_dataset(output_xD_dataset, this_train_val_test, append_outputxD_dict):
    for key in output_xD_dataset[this_train_val_test].keys():
        # print(key)
        if key == 'source' or key == 'c3d_frame':
            output_xD_dataset[this_train_val_test][key] = output_xD_dataset[this_train_val_test][key] + append_outputxD_dict[key]
        else:
            output_xD_dataset[this_train_val_test][key] = np.append(output_xD_dataset[this_train_val_test][key], append_outputxD_dict[key], axis=0)
    return output_xD_dataset


def append_COCO_xD_dataset(output_xD_dataset, this_train_val_test, append_outputxD_dict):
    for key in append_outputxD_dict.keys():  # images, annotations
        output_xD_dataset[this_train_val_test][key] = output_xD_dataset[this_train_val_test][key] + append_outputxD_dict[key]
    return output_xD_dataset


def save_COCO_json(json_data, name):
    for train_val_test in ['train', 'validate', 'test']:
        # np to list
        this_json_data = json_data[train_val_test]
        for frame in tqdm(range(len(this_json_data['annotations'])), desc="Processing Frames"):
            for key2 in this_json_data['annotations'][frame].keys():
                if type(this_json_data['annotations'][frame][key2]) == np.ndarray:
                    this_json_data['annotations'][frame][key2] = this_json_data['annotations'][frame][key2].tolist()
            # this_json_data['annotations'][frame]['keypoints'] = this_json_data['annotations'][frame].pop('keypoint')
        json_filename = name.replace('.pkl', f'_{train_val_test}.json')
        with open(f'{json_filename}', 'w') as f:
            json.dump(json_data[train_val_test], f)
        print(f"Saved {json_filename}")


def save_COCO_json_increment(json_data_ori, name):
    json_data = json_data_ori.copy()
    for train_val_test in ['train', 'validate', 'test']:
        # Prepare the output file path
        json_filename = name.replace('.json', f'_{train_val_test}.json')
        with open(json_filename, 'w') as f:
            this_json_data = json_data[train_val_test]

            # Write the initial metadata fields except annotations
            f.write('{"info": ')
            json.dump(this_json_data['info'], f)

            f.write(', "licenses": ')
            json.dump(this_json_data['licenses'], f)

            f.write(', "images": ')
            json.dump(this_json_data['images'], f)

            f.write(', "categories": ')
            json.dump(this_json_data['categories'], f)

            # Start writing annotations field, and write each annotation separately
            f.write(', "annotations":[')

            for frame in tqdm(range(len(this_json_data['annotations'])), desc=f"Serializing {train_val_test} data to JSON"):
                annotation = this_json_data['annotations'][frame]

                # Convert specific keys to a list if needed
                if 'joint_2d' in annotation:
                    joint_2d_np = annotation['joint_2d'].astype(int)
                    annotation['joint_2d'] = joint_2d_np.tolist()

                # Ensure bbox values are floats with two decimal places using NumPy
                if 'bbox' in annotation:
                    bbox_np = np.round(annotation['bbox'], 2)
                    annotation['bbox'] = bbox_np.tolist()

                # Serialize the current annotation
                json.dump(annotation, f)

                # Add a comma if this is not the last annotation
                if frame != len(this_json_data['annotations']) - 1:
                    f.write(',')

            # Close the annotations array and the JSON object
            f.write(']}')

        print(f"Saved {json_filename}")


def bbox_tlbr2tlwh(bbox):
    """
    Convert bounding box format from top-left-bottom-right to top-left-width-height
    """
    tl, br = bbox
    x1, y1 = tl
    x2, y2 = br
    return [x1, y1, x2 - x1, y2 - y1]


def non_zero_mean(arr):
    # Filter out the zero elements and calculate the mean of the non-zero elements
    return np.mean(arr[arr != 0])

def non_zero_median(arr):
    # Filter out the zero elements and calculate the mean of the non-zero elements
    return np.median(arr[arr != 0])


