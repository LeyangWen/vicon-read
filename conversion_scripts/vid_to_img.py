import argparse
import ffmpeg
import os
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Convert video files to JPG images with downsampling")
    parser.add_argument("--input_extension", default=".avi", help="Video file extension")
    parser.add_argument('--split_config_file', type=str, default=r'config/experiment_config/VEHS-R3-721-MotionBert.yaml')
    parser.add_argument("--output_folder", default=r"W:\VEHS\VEHS-7M\images\5fps", help="Output folder for images")
    parser.add_argument("--output_extension", default=".jpg")
    # parser.add_argument("--verbose", default=False)
    parser.add_argument("--fps", type=int, default=5)

    args = parser.parse_args()

    split_config_file = args.split_config_file
    with open(split_config_file, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
            args.base_folder = data['base_folder']
            args.val_keyword = data['val_keyword']
            args.test_keyword = data['test_keyword']
        except yaml.YAMLError as exc:
            print(split_config_file, exc)
    return args


def convert_video_to_images(input_file, output_file_base, args):
    (
        ffmpeg
        .input(input_file)
        .filter('fps', fps=args.fps)
        .output(f'{output_file_base}-%06d{args.output_extension}')
        .overwrite_output()
        .run(quiet=False)
    )


if __name__ == '__main__':
    # avi to COCO format jpg
    args = parse_args()
    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)
    convert_count = 1
    for root, dirs, files in os.walk(args.base_folder):
        dirs.sort()  # Sort directories in-place
        files.sort(key=str.lower)  # Sort files in-place
        for file in files:
            if not file.endswith(args.input_extension):
                continue
            input_file = os.path.join(root, file)
            # print(f"Processing {input_file}")
            file_elements = input_file.split('\\')
            ### subject to change based on your filename structure
            subject = file_elements[-3]
            activity = file_elements[-1].split('.')[0].lower()
            camera_id = file_elements[-1].split('.')[1]
            if 'activity' not in activity or 'bad' in activity or 'S' not in subject:
                continue
            if any(keyword in root for keyword in args.val_keyword):
                train_val_test = 'validate'
            elif any(keyword in root for keyword in args.test_keyword):
                train_val_test = 'test'
            else:
                train_val_test = 'train'

            print(f"Extracted elements: {train_val_test} - sub: {subject}, act: {activity}, cam: {camera_id}")
            output_file_base = os.path.join(args.output_folder, train_val_test, f"{subject}-{activity}-{camera_id}")
            print(f"Output file base: {output_file_base}")
            # make dir if not exist
            os.makedirs(os.path.dirname(output_file_base), exist_ok=True)
            # get files number inside
            frames_before = len([name for name in os.listdir(os.path.dirname(output_file_base))])
            # raise NotImplementedError('1')
            try:
                convert_video_to_images(input_file, output_file_base, args)
                frames_after = len([name for name in os.listdir(os.path.dirname(output_file_base))])
                print(f"Successfully converted #{convert_count} {input_file}")
                print(f"{frames_after-frames_before} frames saved to {output_file_base}")
            except Exception as e:
                print(f"Error converting #{convert_count} {input_file}: {str(e)}")
            convert_count += 1
            print("#" * 50)

