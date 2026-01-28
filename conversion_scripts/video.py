import os
import argparse
import cv2

VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

def is_image_file(name: str) -> bool:
    return name.lower().endswith(VALID_EXTS)

def make_writer(out_path: str, fps: int, width: int, height: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {out_path}")
    return writer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs_dir", type=str, required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--delete_imgs", action="store_true", default=False)
    parser.add_argument("--max_frames_per_video", type=int, default=54000)

    args = parser.parse_args()

    imgs_dir = args.imgs_dir
    fps = args.fps
    delete_imgs = args.delete_imgs
    max_frames_per_video = args.max_frames_per_video

    if not os.path.isdir(imgs_dir):
        raise FileNotFoundError(f"Not a directory: {imgs_dir}")

    fnames = sorted([f for f in os.listdir(imgs_dir) if is_image_file(f)])
    if not fnames:
        raise RuntimeError(f"No image files found in {imgs_dir}")

    out_dir = os.path.dirname(imgs_dir.rstrip("/"))
    base = os.path.basename(imgs_dir.rstrip("/"))

    # Find first readable frame to lock target size
    target_hw = None  # (h, w)
    first_idx = None
    for j, fn in enumerate(fnames):
        img = cv2.imread(os.path.join(imgs_dir, fn), cv2.IMREAD_COLOR)
        if img is not None:
            target_hw = img.shape[:2]
            first_idx = j
            break
    if target_hw is None:
        raise RuntimeError(f"All images unreadable in {imgs_dir}")

    target_h, target_w = target_hw

    # Writer state for splitting
    part = 1
    frames_in_part = 0
    total_written = 0
    written_files = []  # only those actually written (for safe deletion)
    writer = None

    def open_new_part(part_id: int):
        out_name = f"{base}_{part_id}.mp4" if max_frames_per_video else f"{base}.mp4"
        out_path = os.path.join(out_dir, out_name)
        return make_writer(out_path, fps, target_w, target_h), out_path

    # If splitting disabled (max_frames_per_video <= 0), write single file named base.mp4
    if max_frames_per_video <= 0:
        out_path = os.path.join(out_dir, f"{base}.mp4")
        writer = make_writer(out_path, fps, target_w, target_h)
        current_out_path = out_path
    else:
        writer, current_out_path = open_new_part(part)

    # Process sequentially (streaming)
    for i, fn in enumerate(fnames):
        path = os.path.join(imgs_dir, fn)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"\n⚠️ Skipping {fn} (could not read)")
            continue

        if img.shape[:2] != (target_h, target_w):
            print(f"\n⚠️ Resizing {fn} from {img.shape[:2]} to {(target_h, target_w)}")
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # split boundary: start a new part BEFORE writing this frame
        if max_frames_per_video > 0 and frames_in_part >= max_frames_per_video:
            writer.release()
            part += 1
            frames_in_part = 0
            writer, current_out_path = open_new_part(part)

        writer.write(img)
        written_files.append(path)
        frames_in_part += 1
        total_written += 1

        minutes = total_written / fps / 60
        if (total_written % 100 == 0) or (i + 1 == len(fnames)):
            print(f"\rWrote {total_written} frames (~{minutes:.2f} min)", end="", flush=True)

    writer.release()

    # Report outputs
    if max_frames_per_video <= 0:
        print(f"\nVideo saved at {current_out_path}")
    else:
        # You may have multiple outputs; show the directory + naming rule.
        print(f"\nVideos saved in {out_dir} as {base}_1.mp4, {base}_2.mp4, ...")

    if delete_imgs:
        # Delete only files that were actually written (and readable)
        for path in written_files:
            try:
                os.remove(path)
            except OSError as e:
                print(f"\n⚠️ Failed to delete {path}: {e}")
        print(f"\nDeleted {len(written_files)} images.")