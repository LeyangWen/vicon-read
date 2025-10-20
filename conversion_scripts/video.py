import os
import argparse
import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--imgs_dir", type=str, required=True)
    parser.add_argument("--fps", type=int, default=20)  # 4 fps can not render
    parser.add_argument("--delete_imgs", action="store_true", default=False)

    args = parser.parse_args()

    imgs_dir = args.imgs_dir
    fps = args.fps
    delete_imgs = args.delete_imgs

    fnames = os.listdir(imgs_dir)
    fnames = sorted(fnames)
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
    imgs = []
    target_size = None
    for i, fn in enumerate(fnames):
        path = os.path.join(imgs_dir, fn)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"\n⚠️ Skipping {fn} (could not read)")
            continue
        if target_size is None:
            # (h, w)
            target_size = img.shape[:2]
        if img.shape[:2] != target_size:
            print(f"\n⚠️ Resizing {fn} from {img.shape[:2]} to {target_size}")
            img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
        imgs.append(img)

        # progress print every 100 images
        minutes = (i + 1) / fps / 60
        if (i + 1) % 100 == 0 or (i + 1) == len(fnames):
            print(f"\rRead {i + 1}/{len(fnames)} images (~{minutes:.2f} min of video)", end="", flush=True)
            # break

    if not imgs:
        raise RuntimeError(f"No valid images found in {imgs_dir}")

    # imgs = [cv2.imread(os.path.join(imgs_dir, fn), cv2.IMREAD_COLOR) for fn in fnames]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    size = imgs[0].shape[:2]
    out_dir = os.path.dirname(imgs_dir)
    out_name = os.path.basename(imgs_dir)+".mp4"
    videoWriter = cv2.VideoWriter(os.path.join(out_dir, out_name), fourcc, fps, (size[1], size[0]))

    for i in range(len(imgs)):
        # cv2.imshow("test", imgs[i])
        # cv2.waitKey(0)
        videoWriter.write(imgs[i])

    videoWriter.release()

    print("video saved at {}".format(os.path.join(out_dir, out_name)))

    if delete_imgs:
        for i in range(len(imgs)):
            os.remove(os.path.join(imgs_dir, fnames[i]))
