import os
import argparse
import cv2

"""
python img_to_video.py --imgs_dir output/imgs/example_path --delete_imgs
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--imgs_dir", type=str,
                        default="/Volumes/Z/RTMPose/37kpts_rtmw_v4/Ricks_Videos_RTMW_freeze_backbone_neck_merged_epoch_best/"
                                "Rick_12")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--delete_imgs", action="store_true", default=True)
    parser.add_argument("--video_name", type=str, default=None)
    args = parser.parse_args()

    imgs_dir = args.imgs_dir
    fps = args.fps
    delete_imgs = args.delete_imgs
    args.video_name = os.path.basename(imgs_dir) if args.video_name is None else args.video_name

    fnames = os.listdir(imgs_dir)
    fnames = sorted(fnames)
    # remove all "." files
    fnames = [fn for fn in fnames if not fn.startswith(".")]

    print(f"{os.path.join(imgs_dir, fnames[0])} -> {os.path.join(imgs_dir, fnames[-1])}")

    imgs = [cv2.imread(os.path.join(imgs_dir, fn), cv2.IMREAD_COLOR) for fn in fnames]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    size = imgs[0].shape[:2]
    videoWriter = cv2.VideoWriter(os.path.join(imgs_dir, f"{args.video_name}.mp4"), fourcc, fps, (size[1], size[0]))

    for i in range(len(imgs)):
        # cv2.imshow("test", imgs[i])
        # cv2.waitKey(0)
        videoWriter.write(imgs[i])

    videoWriter.release()
    # get parent directory name
    print("video saved at {}".format(os.path.join(imgs_dir, f"{args.video_name}.mp4")))

    if delete_imgs:
        for i in range(len(imgs)):
            os.remove(os.path.join(imgs_dir, fnames[i]))