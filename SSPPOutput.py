import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


class SSPPOutput:
    """
    This class is used to process 3DSSPP output files
    Functions include: read, visualize, evaluate

    Usage example:
    result = SSPPOutput() / SSPPV6Output() / SSPPV7Output() / SSPPV7WrapperOutput()
    result.load_file('3DSSPP_output.txt')  # or .exp file for v6
    result.show_category()
    result.show_category(subcategory='Strength Capability Percentile')
    result.header
    result.df['Info - Task Name'][200]
    result.cut_segment()
    result.eval_segment(result.segments, eval_keys)
    """
    def __init__(self):
        self.header = []
        self.header_category = {}
        self.df = None
        self.all_segments = {}
        self.segments = {}
        self.baseline_segments = {}
        self.unique_task_len = 0

    def set_header(self, header):
        # modify header to make them unique
        header_category = 'Info'
        header_cat_dict = {header_category: []}
        unique_header = []
        for index, header_item in enumerate(header):
            if header_item == 0.0:
                pass
            elif header_item.endswith('Start'):
                header_category = header_item.replace(' Start', '')
                header_cat_dict[header_category] = []
            else:
                header_item = header_category + ' - ' + header_item
                header_cat_dict[header_category].append(header_item)
            unique_header.append(header_item)

        self.header = unique_header
        self.header_category = header_cat_dict

    def load_file(self, file):
        """
        load .txt file as csv text and read into dict with subheaders
        for 3DSSPP v7.0
        header is included in .txt file
        """
        df = pd.read_csv(file, header=None, skiprows=lambda x: x % 2 == 0)  # skip every 1 line, first line is header
        header = list(pd.read_csv(file, header=None, nrows=1).iloc[0, :].values)  # read first row
        self.set_header(header=header)
        df.columns = self.header
        self.df = df

    def get_category(self, category='Info'):
        return self.df[self.header_category[category]]

    def show_category(self, subcategory=False):
        if subcategory:
            return self.header_category[subcategory]
        else:
            return list(self.header_category.keys())

    def cut_segment(self, baseline_key='Baseline'):
        """
        Input might contain multiple tasks, cut segments df by task name
        """
        segments = {}
        baseline_segments = {}
        # info_name = result.df.keys().item()
        task_names = self.df['Info - Task Name']
        _, start_ids = np.unique(task_names, return_index=True)
        start_ids = np.sort(start_ids)
        unique_task_names = task_names[start_ids]
        unique_task_len = len(unique_task_names)
        for _, unique_task_name in enumerate(unique_task_names):
            frame_count = len(task_names[task_names == unique_task_name])
            start_frame = start_ids[_]
            end_frame = start_frame + frame_count
            if baseline_key in str(unique_task_name):
                baseline_segments[unique_task_name] = self.df.iloc[start_frame:end_frame, :]
            else:
                segments[unique_task_name] = self.df.iloc[start_frame:end_frame, :]
            # print(segments[unique_task_name]['Info - Task Name'])
        self.segments = segments
        self.baseline_segments = baseline_segments
        self.all_segments = {**baseline_segments, **segments}
        self.unique_task_len = unique_task_len
        return unique_task_names, unique_task_len

    def eval_segment(self, segments, segment_eval_keys, verbose=False, criteria='min_mean'):
        """
        Perform evaluation on segments, such as calculating the mean, min, and max.

        Parameters:
        segments (dict): A dictionary of segments. For example, self.segments or self.baseline_segments, you can also pass in self.segments['key'] for one segment.
        segment_eval_keys (str or list): A key for evaluation. For example, 'Summary - Minimum Shoulder Percentile'.
                                         It can also be a list of such keys.
        verbose (bool, optional): If True, print detailed information during evaluation. Defaults to False.

        Returns:
        tuple: A tuple containing the id, score percentage, and task name.
        """
        segment_eval_keys = segment_eval_keys if isinstance(segment_eval_keys, list) else [segment_eval_keys]
        segments = segments if isinstance(segments, dict) else {"place_holder": segments}
        min_score_per_task = []
        min_scores = []
        for _key, _value in segments.items():
            segment_min_list = []
            segment_mean_list = []
            segment_max_list = []
            for eval_key in segment_eval_keys:
                frame_min = _value[eval_key].min()  # min value of each evaluation key
                frame_mean = _value[eval_key].mean()  # mean value of each evaluation key over the session
                frame_max = _value[eval_key].max()
                segment_min_list.append(frame_min)
                segment_mean_list.append(frame_mean)
                segment_max_list.append(frame_max)
                if verbose:
                    print(f"{_key}-{eval_key}: {frame_min}")
            if criteria == 'min_mean':
                # select criteria: segment_mean_min
                min_score_per_task.append(np.min(segment_mean_list))
                min_scores.append(segment_mean_list)
            elif criteria == 'min_min':
                # select criteria: segment_min_min
                min_score_per_task.append(np.min(segment_min_list))
                min_scores.append(segment_min_list)
            elif criteria == 'mean_min':
                # select criteria: segment_mean_max
                min_score_per_task.append(np.mean(segment_min_list))
                min_scores.append(segment_mean_list)
            elif criteria == 'min_max':
                # select criteria: segment_min_max
                min_score_per_task.append(np.min(segment_max_list))
                min_scores.append(segment_max_list)
        task_id_w_max_score = np.argmax(min_score_per_task)
        task_w_max_score = list(segments.keys())[task_id_w_max_score]
        best_segment = {task_w_max_score: segments[task_w_max_score]}
        return task_id_w_max_score, min_score_per_task[task_id_w_max_score], min_scores[task_id_w_max_score], task_w_max_score

    def visualize_segment(self, segments, segment_eval_keys, verbose=False):
        """
        Visualize segment results, panda index is x-axis

        Parameters:
        segments (dict): A dictionary of segments. In this version, you should pass in self.all_segments for good visuals. todo: allow for part of segments
        segment_eval_keys (str or list): A key for evaluation. For example, 'Summary - Minimum Shoulder Percentile'.
                                         It can also be a list of such keys.
        verbose (bool, optional): If True, print detailed information during evaluation. Defaults to False.

        Returns:
        tuple:
        """
        segment_eval_keys = segment_eval_keys if isinstance(segment_eval_keys, list) else [segment_eval_keys]
        segments = segments if isinstance(segments, dict) else {"place_holder": segments}
        figure_num = len(segment_eval_keys)
        fig, axs = plt.subplots(figure_num, 1, figsize=(10, figure_num*10))
        plt.subplots_adjust(hspace=0.75)
        total_frame_count = sum([len(_value) for _key, _value in segments.items()])  # Count the total frame count
        print(f"Total frame count: {total_frame_count}")
        plt.xlabel("Frames", fontsize=14)  # Set x-axis title
        # plot each segment in sequence, with a gray vertical line in between
        for eval_key_idx, eval_key in enumerate(segment_eval_keys):
            axs[eval_key_idx].set_title(eval_key)  # subheading
            axs[eval_key_idx].set_xlim(0, total_frame_count-1)  # Set x-axis limit for each subplot
            if "Percentile" in eval_key:
                axs[eval_key_idx].set_ylim(0, 110)
            for _key, _value in segments.items():
                axs[eval_key_idx].plot(_value[eval_key])
                # gray out the segment in between
                axs[eval_key_idx].axvspan(_value.index[0]-1, _value.index[0], facecolor='gray', alpha=0.8)
        if verbose:
            plt.show()

    def dump(self, segments, segment_eval_keys, file_name="output.csv"):
        """

        Parameters:
        segments (dict): A dictionary of segments. In this version, you should pass in self.all_segments for good visuals. todo: allow for part of segments
        segment_eval_keys (str or list): A key for evaluation. For example, 'Summary - Minimum Shoulder Percentile'.
                                         It can also be a list of such keys.
        verbose (bool, optional): If True, print detailed information during evaluation. Defaults to False.

        Returns:
        tuple:
        """
        segment_eval_keys = segment_eval_keys if isinstance(segment_eval_keys, list) else [segment_eval_keys]
        segments = segments if isinstance(segments, dict) else {"place_holder": segments}

        output = pd.DataFrame()
        # merge into one dataframe
        section = 0
        for _key, _value in segments.items():
            for eval_key_idx, eval_key in enumerate(segment_eval_keys):

                output = pd.concat([output, _value[eval_key]], axis=1)
            if file_name.endswith('.csv'):
                file_name = file_name.replace('.csv', f'_{section}.csv')
                # output panda with header and index
                output.to_csv(file_name, header=True)
            section += 1



class SSPPV6Output(SSPPOutput):
    def __init__(self):
        super().__init__()
        self.default_header = ['Analyst', 'Company Name', 'Units', 'Task Name', 'Gender', 'Height', 'Weight', 'Summary Start', 'L5/S1 Compression', 'L4/L5 Compression', 'Minimum Wrist Percentile',
                               'Minimum Elbow Percentile', 'Minimum Shoulder Percentile', 'Minimum Torso Percentile', 'Minimum Hip Percentile', 'Minimum Knee Percentile',
                               'Minimum Ankle Percentile', 'Coefficient of Friction', 'Left Load Fraction', 'Balance Status', 'Strength Capability Start', 'Right Wrist Flexion',
                               'Right Wrist Deviation', 'Right Forearm Rotation', 'Right Elbow Flexion', 'Right Humeral Rotation', 'Right Shoulder Rotation', 'Right Shoulder Abduction',
                               'Right Hip Flexion', 'Right Knee Flexion', 'Right Ankle Flexion', 'Left Wrist Flexion', 'Left Wrist Deviation', 'Left Forearm Rotation', 'Left Elbow Flexion',
                               'Left Humeral Rotation', 'Left Shoulder Rotation', 'Left Shoulder Abduction', 'Left Hip Flexion', 'Left Knee Flexion', 'Left Ankle Flexion', 'Torso Flexion',
                               'Torso Lateral Bending', 'Torso Rotation', 'Low Back Start', 'L5/S1 Compression', 'L5/S1 Compression SD', 'Sagittal Shear', 'Frontal Shear',
                               'Total L4/L5 Compression', 'Anterior L4/L5 Shear', 'Lateral L4/L5 Shear', 'Right Erector Force Magnitude', 'Right Erector Shear', 'Right Erector Force X',
                               'Right Erector Force Y', 'Right Erector Force Z', 'Right Rectus Force Magnitude', 'Right Rectus Shear', 'Right Rectus Force X', 'Right Rectus Force Y',
                               'Right Rectus Force Z', 'Right Internal Force Magnitude', 'Right Internal Shear', 'Right Internal Force X', 'Right Internal Force Y', 'Right Internal Force Z',
                               'Right External Force Magnitude', 'Right External Shear', 'Right External Force X', 'Right External Force Y', 'Right External Force Z',
                               'Right Latissimus Force Magnitude', 'Right Latissimus Shear', 'Right Latissimus Force X', 'Right Latissimus Force Y', 'Right Latissimus Force Z',
                               'Left Erector Force Magnitude', 'Left Erector Shear', 'Left Erector Force X', 'Left Erector Force Y', 'Left Erector Force Z', 'Left Rectus Force Magnitude',
                               'Left Rectus Shear', 'Left Rectus Force X', 'Left Rectus Force Y', 'Left Rectus Force Z', 'Left Internal Force Magnitude', 'Left Internal Shear',
                               'Left Internal Force X', 'Left Internal Force Y', 'Left Internal Force Z', 'Left External Force Magnitude', 'Left External Shear', 'Left External Force X',
                               'Left External Force Y', 'Left External Force Z', 'Left Latissimus Force Magnitude', 'Left Latissimus Shear', 'Left Latissimus Force X', 'Left Latissimus Force Y',
                               'Left Latissimus Force Z', 'Fatigue Start', 'Right Wrist Flexion Fifth', 'Right Wrist Flexion Fiftieth', 'Right Wrist Flexion Nintieth',
                               'Right Wrist Deviation Fifth', 'Right Wrist Deviation Fiftieth', 'Right Wrist Deviation Nintieth', 'Right Forearm Rotation Fifth', 'Right Forearm Rotation Fiftieth',
                               'Right Forearm Rotation Nintieth', 'Right Elbow Flexion Fifth', 'Right Elbow Flexion Fiftieth', 'Right Elbow Flexion Nintieth', 'Right Humeral Rotation Fifth',
                               'Right Humeral Rotation Fiftieth', 'Right Humeral Rotation Nintieth', 'Right Shoulder Rotation Fifth', 'Right Shoulder Rotation Fiftieth',
                               'Right Shoulder Rotation Nintieth', 'Right Shoulder Abduction Fifth', 'Right Shoulder Abduction Fiftieth', 'Right Shoulder Abduction Nintieth',
                               'Right Hip Flexion Fifth', 'Right Hip Flexion Fiftieth', 'Right Hip Flexion Nintieth', 'Right Knee Flexion Fifth', 'Right Knee Flexion Fiftieth',
                               'Right Knee Flexion Nintieth', 'Right Ankle Flexion Fifth', 'Right Ankle Flexion Fiftieth', 'Right Ankle Flexion Nintieth', 'Left Wrist Flexion Fifth',
                               'Left Wrist Flexion Fiftieth', 'Left Wrist Flexion Nintieth', 'Left Wrist Deviation Fifth', 'Left Wrist Deviation Fiftieth', 'Left Wrist Deviation Nintieth',
                               'Left Forearm Rotation Fifth', 'Left Forearm Rotation Fiftieth', 'Left Forearm Rotation Nintieth', 'Left Elbow Flexion Fifth', 'Left Elbow Flexion Fiftieth',
                               'Left Elbow Flexion Nintieth', 'Left Humeral Rotation Fifth', 'Left Humeral Rotation Fiftieth', 'Left Humeral Rotation Nintieth', 'Left Shoulder Rotation Fifth',
                               'Left Shoulder Rotation Fiftieth', 'Left Shoulder Rotation Nintieth', 'Left Shoulder Abduction Fifth', 'Left Shoulder Abduction Fiftieth',
                               'Left Shoulder Abduction Nintieth', 'Left Hip Flexion Fifth', 'Left Hip Flexion Fiftieth', 'Left Hip Flexion Nintieth', 'Left Knee Flexion Fifth',
                               'Left Knee Flexion Fiftieth', 'Left Knee Flexion Nintieth', 'Left Ankle Flexion Fifth', 'Left Ankle Flexion Fiftieth', 'Left Ankle Flexion Nintieth',
                               'Torso Flexion Fifth', 'Torso Flexion Fiftieth', 'Torso Flexion Nintieth', 'Torso Lateral Bending Fifth', 'Torso Lateral Bending Fiftieth',
                               'Torso Lateral Bending Nintieth', 'Torso Rotation Fifth', 'Torso Rotation Fiftieth', 'Torso Rotation Nintieth', 'Balance Start', 'COG X', 'COG Y', 'COP X', 'COP Y',
                               'Stability', 'Left load', 'Right load', 'Hand Forces Start', 'Right Force Magnitude', 'Right Vertical Angle', 'Right Horizontal Angle', 'Left Force Magnitude',
                               'Left Vertical Angle', 'Left Horizontal Angle', 'Segment Angles Start', 'Right Vertical Hand Angle', 'Right Horizontal Hand Angle', 'Right Hand Rotation Angle',
                               'Right Vertical Forearm Angle', 'Right Horizontal Forearm Angle', 'Right Vertical Upper Arm Angle', 'Right Horizontal Upper Arm Angle',
                               'Right Vertical Clavicle Angle', 'Right Horizontal Clavicle Angle', 'Right Vertical Upper Leg Angle', 'Right Horizontal Upper Leg Angle',
                               'Right Vertical Lower Leg Angle', 'Right Horizontal Lower Leg Angle', 'Right Vertical Foot Angle', 'Right Horizontal Foot Angle', 'Left Vertical Hand Angle',
                               'Left Horizontal Hand Angle', 'Left Hand Rotation Angle', 'Left Vertical Forearm Angle', 'Left Horizontal Forearm Angle', 'Left Vertical Upper Arm Angle',
                               'Left Horizontal Upper Arm Angle', 'Left Vertical Clavicle Angle', 'Left Horizontal Clavicle Angle', 'Left Vertical Upper Leg Angle',
                               'Left Horizontal Upper Leg Angle', 'Left Vertical Lower Leg Angle', 'Left Horizontal Lower Leg Angle', 'Left Vertical Foot Angle', 'Left Horizontal Foot Angle',
                               'Head Lateral Bending Angle', 'Head Flexion Angle', 'Head Axial Rotation Angle', 'Trunk Lateral Bending Angle', 'Trunk Flexion Angle', 'Trunk Axial Rotation Angle',
                               'Pelvis Lateral Bending Angle', 'Pelvis Flexion Angle', 'Pelvis Axial Rotation Angle', 'Posture Angles Start', 'Right Hand Flexion', 'Right Hand Deviation',
                               'Right Forearm Rotation', 'Right Elbow Included', 'Right Shoulder Vertical', 'Right Shoulder Horizontal', 'Right Humeral Rotation', 'Right Hip Included',
                               'Right Hip Vertical', 'Right Hip Horizontal', 'Right Femoral Rotation', 'Right Lower Leg Rotation', 'Right Knee Included', 'Right Ankle Included',
                               'Left Hand Flexion', 'Left Hand Deviation', 'Left Forearm Rotation', 'Left Elbow Included', 'Left Shoulder Vertical', 'Left Shoulder Horizontal',
                               'Left Humeral Rotation', 'Left Hip Included', 'Left Hip Vertical', 'Left Hip Horizontal', 'Left Femoral Rotation', 'Left Lower Leg Rotation', 'Left Knee Included',
                               'Left Ankle Included', 'Head Flexion Angle', 'Head Axial Rotation Angle', 'Head Lateral Bending Angle', 'Trunk Flexion From L5/S1', 'Adjusted Trunk Axial Rotation',
                               'Adjusted Trunk Lateral Bending', 'Pelvis Flexion', 'Pelvis Axial Rotation Angle', 'Pelvis Lateral Bending Angle', 'L5S1 Tilt Angle', 'Joint Locations Start',
                               'Right Hand X', 'Right Hand Y', 'Right Hand Z', 'Right Wrist X', 'Right Wrist Y', 'Right Wrist Z', 'Right Elbow X', 'Right Elbow Y', 'Right Elbow Z',
                               'Right Shoulder X', 'Right Shoulder Y', 'Right Shoulder Z', 'Right Hip X', 'Right Hip Y', 'Right Hip Z', 'Right IT X', 'Right IT Y', 'Right IT Z', 'Right Knee X',
                               'Right Knee Y', 'Right Knee Z', 'Right Ankle X', 'Right Ankle Y', 'Right Ankle Z', 'Right Heel X', 'Right Heel Y', 'Right Heel Z', 'Right Foot Center X',
                               'Right Foot Center Y', 'Right Foot Center Z', 'Right Ball of Foot X', 'Right Ball of Foot Y', 'Right Ball of Foot Z', 'Left Hand X', 'Left Hand Y', 'Left Hand Z',
                               'Left Wrist X', 'Left Wrist Y', 'Left Wrist Z', 'Left Elbow X', 'Left Elbow Y', 'Left Elbow Z', 'Left Shoulder X', 'Left Shoulder Y', 'Left Shoulder Z', 'Left Hip X',
                               'Left Hip Y', 'Left Hip Z', 'Left IT X', 'Left IT Y', 'Left IT Z', 'Left Knee X', 'Left Knee Y', 'Left Knee Z', 'Left Ankle X', 'Left Ankle Y', 'Left Ankle Z',
                               'Left Heel X', 'Left Heel Y', 'Left Heel Z', 'Left Foot Center X', 'Left Foot Center Y', 'Left Foot Center Z', 'Left Ball of Foot X', 'Left Ball of Foot Y',
                               'Left Ball of Foot Z', 'Tragion X', 'Tragion Y', 'Tragion Z', 'Nasion X', 'Nasion Y', 'Nasion Z', 'Top of Neck X', 'Top of Neck Y', 'Top of Neck Z', 'C7T1 X',
                               'C7T1 Y', 'C7T1 Z', 'SCJ X', 'SCJ Y', 'SCJ Z', 'L5S1 X', 'L5S1 Y', 'L5S1 Z', 'Center of Hips X', 'Center of Hips Y', 'Center of Hips Z', 'Center of ITs X',
                               'Center of ITs Y', 'Center of ITs Z', 'Center of Ankles X', 'Center of Ankles Y', 'Center of Ankles Z', 'Center of Balls of Feet X', 'Center of Balls of Feet Y',
                               'Center of Balls of Feet Z', 'Joint Forces Start', 'Right Hand X', 'Right Hand Y', 'Right Hand Z', 'Right Wrist X', 'Right Wrist Y', 'Right Wrist Z', 'Right Elbow X',
                               'Right Elbow Y', 'Right Elbow Z', 'Right Shoulder X', 'Right Shoulder Y', 'Right Shoulder Z', 'Right Hip X', 'Right Hip Y', 'Right Hip Z', 'Right IT X', 'Right IT Y',
                               'Right IT Z', 'Right Knee X', 'Right Knee Y', 'Right Knee Z', 'Right Ankle X', 'Right Ankle Y', 'Right Ankle Z', 'Right Heel X', 'Right Heel Y', 'Right Heel Z',
                               'Right Foot Center X', 'Right Foot Center Y', 'Right Foot Center Z', 'Right Ball of Foot X', 'Right Ball of Foot Y', 'Right Ball of Foot Z', 'Left Hand X',
                               'Left Hand Y', 'Left Hand Z', 'Left Wrist X', 'Left Wrist Y', 'Left Wrist Z', 'Left Elbow X', 'Left Elbow Y', 'Left Elbow Z', 'Left Shoulder X', 'Left Shoulder Y',
                               'Left Shoulder Z', 'Left Hip X', 'Left Hip Y', 'Left Hip Z', 'Left IT X', 'Left IT Y', 'Left IT Z', 'Left Knee X', 'Left Knee Y', 'Left Knee Z', 'Left Ankle X',
                               'Left Ankle Y', 'Left Ankle Z', 'Left Heel X', 'Left Heel Y', 'Left Heel Z', 'Left Foot Center X', 'Left Foot Center Y', 'Left Foot Center Z', 'Left Ball of Foot X',
                               'Left Ball of Foot Y', 'Left Ball of Foot Z', 'Tragion X', 'Tragion Y', 'Tragion Z', 'Nasion X', 'Nasion Y', 'Nasion Z', 'Top of Neck X', 'Top of Neck Y',
                               'Top of Neck Z', 'C7T1 X', 'C7T1 Y', 'C7T1 Z', 'SCJ X', 'SCJ Y', 'SCJ Z', 'L5S1 X', 'L5S1 Y', 'L5S1 Z', 'Center of Hips X', 'Center of Hips Y', 'Center of Hips Z',
                               'Center of ITs X', 'Center of ITs Y', 'Center of ITs Z', 'Center of Ankles X', 'Center of Ankles Y', 'Center of Ankles Z', 'Center of Balls of Feet X',
                               'Center of Balls of Feet Y', 'Center of Balls of Feet Z', 'Right Forward Seat X', 'Right Forward Seat Y', 'Right Forward Seat Z', 'Left Forward Seat X',
                               'Left Forward Seat Y', 'Left Forward Seat Z', 'Seat Back X', 'Seat Back Y', 'Seat Back Z', 'Joint Moments Start', 'Right Hand X', 'Right Hand Y', 'Right Hand Z',
                               'Right Wrist X', 'Right Wrist Y', 'Right Wrist Z', 'Right Elbow X', 'Right Elbow Y', 'Right Elbow Z', 'Right Shoulder X', 'Right Shoulder Y', 'Right Shoulder Z',
                               'Right Hip X', 'Right Hip Y', 'Right Hip Z', 'Right IT X', 'Right IT Y', 'Right IT Z', 'Right Knee X', 'Right Knee Y', 'Right Knee Z', 'Right Ankle X',
                               'Right Ankle Y', 'Right Ankle Z', 'Right Heel X', 'Right Heel Y', 'Right Heel Z', 'Right Foot Center X', 'Right Foot Center Y', 'Right Foot Center Z',
                               'Right Ball of Foot X', 'Right Ball of Foot Y', 'Right Ball of Foot Z', 'Left Hand X', 'Left Hand Y', 'Left Hand Z', 'Left Wrist X', 'Left Wrist Y', 'Left Wrist Z',
                               'Left Elbow X', 'Left Elbow Y', 'Left Elbow Z', 'Left Shoulder X', 'Left Shoulder Y', 'Left Shoulder Z', 'Left Hip X', 'Left Hip Y', 'Left Hip Z', 'Left IT X',
                               'Left IT Y', 'Left IT Z', 'Left Knee X', 'Left Knee Y', 'Left Knee Z', 'Left Ankle X', 'Left Ankle Y', 'Left Ankle Z', 'Left Heel X', 'Left Heel Y', 'Left Heel Z',
                               'Left Foot Center X', 'Left Foot Center Y', 'Left Foot Center Z', 'Left Ball of Foot X', 'Left Ball of Foot Y', 'Left Ball of Foot Z', 'Tragion X', 'Tragion Y',
                               'Tragion Z', 'Nasion X', 'Nasion Y', 'Nasion Z', 'Top of Neck X', 'Top of Neck Y', 'Top of Neck Z', 'C7T1 X', 'C7T1 Y', 'C7T1 Z', 'SCJ X', 'SCJ Y', 'SCJ Z', 'L5S1 X',
                               'L5S1 Y', 'L5S1 Z', 'Center of Hips X', 'Center of Hips Y', 'Center of Hips Z', 'Center of ITs X', 'Center of ITs Y', 'Center of ITs Z', 'Center of Ankles X',
                               'Center of Ankles Y', 'Center of Ankles Z', 'Center of Balls of Feet X', 'Center of Balls of Feet Y', 'Center of Balls of Feet Z']

    def load_file(self, file):
        """
        3DSSPP v6 generates a .exp file with no header, use set_header to set the default header
        load .exp as csv text and read into dict with subheaders
        """
        df = pd.read_csv(file, header=None)  # load csv file
        self.set_header(header=self.default_header)  # set header to self.header
        df.columns = self.header
        self.df = df


class SSPPV7Output(SSPPOutput):
    def __init__(self):
        super().__init__()

    def load_file(self, file):
        """
        3DSSPP v7 generates a .txt file with header every other line
        load .txt file as csv text and read into dict with subheaders
        """
        df = pd.read_csv(file, header=None, skiprows=lambda x: x % 2 == 0)  # skip every 1 line, first line is header
        header = list(pd.read_csv(file, header=None, nrows=1).iloc[0, :].values)  # read first row
        self.set_header(header=header)
        df.columns = self.header
        self.df = df


if __name__ == "__main__":
    case = 0

    if case == 0:  # visualization example
        # load file
        input_3DSSPP_folder = r'W:\3DSSPP_all\Compiled\3DSSPP 7.1.2 CLI\export_20240224'
        input_3DSSPP_file = r"batchinput_export.txt"

        result = SSPPV7Output()
        result.load_file(os.path.join(input_3DSSPP_folder, input_3DSSPP_file))
        _, unique_segment_len = result.cut_segment()

        print(_)

        for subcategory in ["Joint Forces", "Joint Moments", "Low Back"]:
            eval_keys = result.show_category(subcategory=subcategory)
            result.dump(result.all_segments, segment_eval_keys=eval_keys, file_name=f"output/{subcategory}.csv")


        # eval_keys = result.show_category(subcategory='Summary')[:6]
        # result.visualize_segment(result.all_segments, segment_eval_keys=eval_keys, verbose=True)


        # eval_keys = result.show_category(subcategory='Summary')[:-3]
        # result.visualize_segment(result.all_segments, segment_eval_keys=eval_keys, verbose=True)

    elif case == 1:  # mass evaluation example
        motion_smpl_folder_base = r'experiment\text2pose-20231113T194712Z-001\text2pose'
        text_prompts = ['A_person_half_kneel_with_one_leg_to_work_near_the_floor',
                        'A_person_half_squat_to_work_near_the_floor',
                        'A_person_move_a_box_from_left_to_right',
                        'A_person_raise_both_hands_above_his_head_and_keep_them_there',
                        'A_person_squat_to_carry_up_something']
        text_prompt = text_prompts[2]
        motion_smpl_folder = f'{motion_smpl_folder_base}\\{text_prompt}'

        search_string = '_export.txt'
        txt_files = [f for f in os.listdir(motion_smpl_folder) if search_string in f]

        txt_file = txt_files[0]
        print(f"loading {txt_file} ...")
        result = SSPPV7Output()
        result.load_file(os.path.join(motion_smpl_folder, txt_file))
        a = result.get_category('Info')
        # result.show_category()
        # result.show_category(subcategory='Strength Capability Percentile')
        # result.header
        # result.df['Info - Task Name'][200]
        result.cut_segment()
        # eval_keys = [
        #              'Summary - Minimum Wrist Percentile',
        #              'Summary - Minimum Elbow Percentile',
        #              'Summary - Minimum Shoulder Percentile',
        #              'Summary - Minimum Torso Percentile',
        #              'Summary - Minimum Neck Percentile',
        #              'Summary - Minimum Hip Percentile',
        #              'Summary - Minimum Knee Percentile',
        #              'Summary - Minimum Ankle Percentile'
        # ]
        eval_keys = result.show_category(subcategory='Strength Capability Percentile')

        ours = result.eval_segment(result.segments, eval_keys)
        result.eval_segment(result.segments[ours[-1]], eval_keys, verbose=True)
        baseline = result.eval_segment(result.baseline_segments, eval_keys, verbose=True)
        print("##################################################")
        print(f"text prompt: {text_prompt}")
        print(f"ours: {ours}")
        print(f"baseline: {baseline}")


