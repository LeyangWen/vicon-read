# First use adobe Premiere to add marker pairs
# Then export markers as CSV/TSV (tab-separated) from source (no need for sequence)
# Put in csv and clip and video folders
# Run to separate clips


#!/usr/bin/env python3
import argparse, sys, re, subprocess, shutil, glob, math
from pathlib import Path
from typing import List, Tuple
import pandas as pd

# ---------- helpers ----------

def which_or_die(cmd: str):
    if shutil.which(cmd) is None:
        sys.exit(f"Error: '{cmd}' not found in PATH. On mac: brew install {cmd}")

def parse_markins(csv_path: Path) -> List[str]:
    """
    Return a list of 'In' timecodes from an Adobe marker CSV/TSV.
    - Adobe exports are usually TAB-separated, with UTF-8 BOM or UTF-16.
    - Only uses 'In' column (case-insensitive).
    """
    df = None
    for enc in ("utf-8-sig", "utf-16"):
        try:
            df = pd.read_csv(csv_path, sep="\t", encoding=enc, engine="python")
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        sys.stderr.write(f"ERROR: Could not read {csv_path.name} with utf-8-sig or utf-16\n")
        return []
    cols_lc = [str(c).strip().lower() for c in df.columns]
    if "in" not in cols_lc:
        sys.stderr.write(f"ERROR: No 'In' column in {csv_path.name}\n")
        return []
    key = df.columns[cols_lc.index("in")]
    # Keep non-empty
    return [str(v).strip() for v in df[key].dropna() if str(v).strip()]

def find_matching_videos(video_root: Path, prefix: str, timestamp: str) -> List[Path]:
    """Find all videos under video_root matching '<prefix>.*.<timestamp>.mp4'."""
    pattern = f"**/{prefix}.*.{timestamp}.mp4"
    return [Path(p) for p in glob.glob(str(video_root / pattern), recursive=True)]

def derive_prefix_and_timestamp(csv_filename_stem: str) -> Tuple[str, str]:
    """
    From '<prefix>.<camera>.<timestamp>' return (prefix, timestamp).
    Example: 'bag01.51470934.20250919201429' -> ('bag01', '20250919201429')
    """
    parts = csv_filename_stem.split(".")
    if len(parts) < 3:
        return parts[0], parts[-1] if parts else ("", "")
    return parts[0], parts[-1]

def ffprobe_fps(video_path: Path) -> float:
    """
    Return the video FPS as a float using ffprobe r_frame_rate (e.g., 30000/1001).
    """
    try:
        # Query r_frame_rate only, plain output
        out = subprocess.check_output(
            ["ffprobe", "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-of", "default=nk=1:nw=1",
             str(video_path)],
            stderr=subprocess.STDOUT
        ).decode("utf-8", errors="replace").strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed on {video_path}: {e.output.decode('utf-8', 'replace')}")
    # r_frame_rate can be like "30000/1001" or "30/1" or "25"
    if "/" in out:
        num, den = out.split("/", 1)
        try:
            fps = float(num) / float(den)
        except Exception:
            fps = 30.0
    else:
        try:
            fps = float(out)
        except Exception:
            fps = 30.0
    # Sanity clamp
    if not (1.0 <= fps <= 240.0):
        fps = 30.0
    return fps

def looks_like_frames_timecode(s: str) -> bool:
    """
    Heuristic: 'HH:MM:SS:FF' or 'H:MM:SS:FF' where last field is frames.
    """
    parts = s.split(":")
    return len(parts) == 4 and all(p.isdigit() for p in parts)

def timecode_to_seconds(tc: str, fps: float) -> float:
    """
    Convert 'HH:MM:SS:FF' to seconds using fps.
    If tc is already seconds or 'HH:MM:SS(.mmm)', return seconds as float.
    """
    tc = tc.replace(";", ":").strip()
    parts = tc.split(":")
    if looks_like_frames_timecode(tc):
        hh, mm, ss, ff = (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
        return hh*3600 + mm*60 + ss + (ff / fps)
    # If HH:MM:SS(.mmm)
    if len(parts) >= 3 and parts[0].isdigit():
        # Allow HH:MM:SS.mmm
        try:
            hh = int(parts[0]); mm = int(parts[1])
            ss = float(parts[2])
            return hh*3600 + mm*60 + ss
        except Exception:
            pass
    # Fallback: plain seconds string
    try:
        return float(tc)
    except Exception:
        raise ValueError(f"Unrecognized time format: {tc}")

def run_ffmpeg_clip(src: Path, out_path: Path, start_s: float, end_s: float, reencode: bool, crf: int, preset: str):
    if end_s <= start_s:
        raise ValueError(f"end <= start for {src.name}: {start_s} vs {end_s}")
    dur = max(0.001, end_s - start_s)
    # Use -ss (input seeking) + -t duration for robust cutting
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start_s:.3f}", "-i", str(src), "-t", f"{dur:.3f}"
    ]
    if reencode:
        cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                "-c:a", "copy", "-movflags", "+faststart"]
    else:
        cmd += ["-c", "copy"]
    cmd.append(str(out_path))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        # Re-run with full stderr to help debug
        print(f"\nffmpeg failed for {src.name} → {out_path.name}", file=sys.stderr)
        try:
            subprocess.run(cmd[:-1] + [str(out_path)], check=True)
        except Exception:
            pass
        raise

def ensure_outdir(root: Path, video_path: Path) -> Path:
    outdir = root / video_path.stem
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

# ---------- main processing ----------

def process_csv(csv_path: Path, video_root: Path, out_root: Path, reencode: bool, crf: int, preset: str):
    if csv_path.name.startswith("._") or csv_path.name.startswith("_"):
        print(f"Skipping CSV: {csv_path}")
        return

    # Derive pattern from CSV name: <prefix>.<camera>.<timestamp>.mp4.csv
    stem_with_mp4 = csv_path.name[:-4]  # strip .csv
    stem = stem_with_mp4[:-4] if stem_with_mp4.endswith(".mp4") else stem_with_mp4
    prefix, timestamp = derive_prefix_and_timestamp(stem)

    marks = parse_markins(csv_path)
    if len(marks) < 2:
        print(f"WARNING: No (or only one) In in {csv_path.name}; skipping.")
        return
    if len(marks) % 2 == 1:
        print(f"NOTE: Odd number of 'In' markers ({len(marks)}) in {csv_path.name}; last one will be ignored.")
        marks = marks[:-1]

    videos = find_matching_videos(video_root, prefix, timestamp)
    if not videos:
        print(f"WARNING: No videos found for CSV {csv_path.name} (pattern {prefix}.*.{timestamp}.mp4)")
        return

    print(f"CSV: {csv_path.name}  → {len(videos)} video(s)  [prefix='{prefix}', ts='{timestamp}']")

    # Build raw pairs (strings)
    pairs = [(marks[i], marks[i+1]) for i in range(0, len(marks), 2)]

    for src in videos:
        try:
            fps = ffprobe_fps(src)
        except Exception as e:
            print(f"WARNING: Could not determine FPS for {src.name}: {e}. Assuming 30.0.", file=sys.stderr)
            fps = 30.0

        # Convert timecodes to seconds using that file's FPS
        pairs_s: List[Tuple[float, float]] = []
        for a, b in pairs:
            start_s = timecode_to_seconds(a, fps)
            end_s = timecode_to_seconds(b, fps)
            if end_s > start_s:
                pairs_s.append((start_s, end_s))
            else:
                print(f"  Skip invalid pair: {a} -> {b}")

        if not pairs_s:
            print(f"  No valid time ranges for {src.name}")
            continue

        outdir = ensure_outdir(out_root, src)
        print(f"  Source: {src}")
        print(f"  FPS: {fps:.3f}")
        print(f"  Clips → {outdir}")

        for idx, (start_s, end_s) in enumerate(pairs_s, start=1):
            out_name = f"clip_{idx:02d}.mp4"
            out_path = outdir / out_name
            print(f"    → {out_name}  [start={start_s:.3f}s, end={end_s:.3f}s]")
            run_ffmpeg_clip(src, out_path, start_s, end_s, reencode=reencode, crf=crf, preset=preset)

        made = len(list(outdir.glob("clip_*.mp4")))
        print(f"  Done: {src.name} → {made} clip(s)\n")

def main():
    parser = argparse.ArgumentParser(description="Split videos into clips using CSV marker files.")
    parser.add_argument("--csv-dir", default=   '//Users/leyangwen/Documents/Isaac/MMH/imitation_motions/Terrain_lift/markers')
    parser.add_argument("--video-root", default='/Users/leyangwen/Documents/Isaac/MMH/imitation_motions/Terrain_lift/good/raw')
    parser.add_argument("--out-root", default=  "/Users/leyangwen/Documents/Isaac/MMH/imitation_motions/Terrain_lift/good/clips")
    parser.add_argument("--crf", type=int, default=18, help="x264 CRF quality (default: 18). Lower = better.")
    parser.add_argument("--preset", default="veryfast", help="x264 preset (default: veryfast).")
    parser.add_argument("--copy-video", action="store_true",
                        help="Use stream copy (-c copy) instead of re-encode (faster but keyframe-aligned cuts).")
    args = parser.parse_args()

    which_or_die("ffmpeg")
    which_or_die("ffprobe")

    csv_dir = Path(args.csv_dir).expanduser().resolve()
    video_root = Path(args.video_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    if not csv_dir.is_dir():
        sys.exit(f"--csv-dir not found: {csv_dir}")
    if not video_root.is_dir():
        sys.exit(f"--video-root not found: {video_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    csvs = sorted(csv_dir.glob("*.mp4.csv"))
    if not csvs:
        print(f"No CSVs found in {csv_dir} matching *.mp4.csv")
        return

    reencode = not args.copy_video
    for csv_path in csvs:
        process_csv(csv_path, video_root, out_root, reencode=reencode, crf=args.crf, preset=args.preset)

    print("All CSVs processed.")


if __name__ == "__main__":
    main()