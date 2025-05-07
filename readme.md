# VICON Reader

This repo contains tools for reading and processing Vicon data.
1. Scripts to use with an open Vicon Nexus session to read and save data in real-time
2. Scripts to process the saved c3d data
3. Supporting classes and functions

Here is a [YouTube playlist](https://youtube.com/playlist?list=PLjMAlxkYpRr0PwPyE3-LDrwiz8xIsNuma&si=HpLm-B3SFCHOHMK4) generated using this repo.
* YouTube video 3D pose preview:  
  ![Alt Text](/figures/Youtube_pose_preview.png)
* YouTube video ergonomic angle preview:  
  ![Alt Text](/figures/Youtube_angle_preview.png)

## 1. Scripts to use with an open Vicon Nexus session to read and save data in real-time
### 1.1. Tools for cleaning up Vicon data
* [detect_swap.py](nexus_tools/detect_swap.py): Detects if the markers are swapped and mark as Vicon events, also automatically swaps the markers in simpler cases
* [swap_by_frame.py](tools/swap_by_frame.py): Batch swap markers by frame number, normally between left and right
* [export_files.py](tools/export_files.py): Export files from Vicon Nexus session
* [backup_c3d.py](tools/backup_c3d.py): Backup c3d files in the backup folder before irreversible operations like smoothing. 

### 1.2. Tools for processing Vicon data
* [caculateVEHSskeleton_ergo.py](caculateVEHSskeleton_ergo.py): Calculate the 2D and 3D VEHS skeleton joint centers and angles using the VEHSR3 dataset skeleton. Also output cdf in H36M format
* [analog_compare.py](nexus_tools/analog_compare.py): Compare analog data from two analog forceplates/transducers, used for rough calibration of force and torque data
* [branch-drop_accel.acceltest.py](acceltest.py): contains code to validate Vicon acceleration calculations by dropping markers
* [branch-pendulum_accel.acceltest.py](acceltest.py): contains code to validate Vicon acceleration calculations by pendulum motion

## 2. Scripts to process the saved c3d data
* [c3d_to_angles.py](/conversion_script/c3d_to_angles.py): Reads the c3d files, calculates the 2D and 3D skeleton joint centers (66) and outputs in a MotionBert format. Also output the ergonomic joint angles
* [np_to_angles.py](/conversion_script/np_to_angles.py): Reads the c3d files, calculates the 2D and 3D skeleton joint centers as well as the ergonomic joint angles


## 3. Supporting classes and functions
* [ForceTorqueTransducer.py](tools/ForceTorqueTransducer.py): Class for calculating the force and torque from the force plate and a custom-wired analog AMTI force transducer
* [Camera.py](Camera.py): Class for projection, distortion correction, and other camera-related calculations
* [Skeleton.py](Skeleton.py): Class for defining pose skeleton structure, outputting 3DSSPP batch file, and visualization

## Figures
* 26 joint format visualization
![Alt Text](/figures/26joints.png)
* Ergonomic angle coordinate system definitions
![Alt Text](/figures/angle_coord.png)

### Useful snippets
- Powershell code to convert all avi files in a folder to mp4, keep original files
```powershell
cd 'W:\VEHS\VEHS data collection round 3\processed\DanLi\FullCollection'
# skip if mp4 already exists
Get-ChildItem -Filter *ctivity06.66920734.20230825192231.avi | ForEach-Object { If (!(Test-Path "$($_.DirectoryName)/$($_.BaseName).mp4")) {ffmpeg -i $_.FullName -c:v copy -c:a copy -y "$($_.DirectoryName)/$($_.BaseName).mp4"}}

```

```bash
# Base path to your processed data
BASE="/media/leyang/My Book/VEHS/VEHS data collection round 3/processed"

# Loop sessions S02 through S10
for session in S0{1..9} S10; do
  DIR="$BASE/$session/FullCollection"
  if [ ! -d "$DIR" ]; then
    echo "⚠️  Skipping missing directory: $DIR"
    continue
  fi

  echo "🔄 Processing session: $session"

  # look for *tivity04*.avi in that folder
  for avi in "$DIR"/*tivity03.51470934*.avi; do
    # if no match, the glob stays literal and avi won't exist
    if [ ! -e "$avi" ]; then
      echo "   • No *tivity03.51470934*.avi files in $DIR"
      break
    fi

    base="${avi%.*}"
    if [ ! -f "${base}.mp4" ]; then
      echo "   • Converting $(basename "$avi") → $(basename "${base}.mp4")"
      ffmpeg -i "$avi" -c:v copy -c:a copy -y "${base}.mp4"
    else
      echo "   • Skipping $(basename "$avi") (mp4 exists)"
    fi
  done

  echo
done

echo "✅ All sessions done."

```
### Copy over to Z drive SSD.

```bash
#!/usr/bin/env bash
# set -euo pipefail

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BASE_SRC="/media/leyang/My Book/VEHS/VEHS data collection round 3/processed"
BASE_DST="/media/leyang/Z/VEHS/VEHS7M/vid_mp4"
# ────────────────────────────────────────────────────────────────────────────────

shopt -s nullglob

for session in S0{1..9} S10; do
  SRC_DIR="$BASE_SRC/$session/FullCollection"
  DST_DIR="$BASE_DST/$session/FullCollection"

  if [[ ! -d "$SRC_DIR" ]]; then
    echo "⚠️  Skipping missing directory: $SRC_DIR"
    continue
  fi

  mkdir -p "$DST_DIR"
  echo "🔄 Processing session: $session"

  mp4_files=( "$SRC_DIR"/*.51470934*.mp4 )
  if (( ${#mp4_files[@]} == 0 )); then
    echo "   • No .mp4 files in $SRC_DIR"
  else
    for src in "${mp4_files[@]}"; do
      fname="$(basename "$src")"
      dst="$DST_DIR/$fname"

      if [[ ! -e "$dst" ]]; then
        echo "   • Copying $fname → $DST_DIR/"
        cp -- "$src" "$dst"
      else
        echo "   • Skipping $fname (already exists)"
      fi
    done
  fi

  echo
done

echo "✅ All sessions done."


```


- Batch rename of all files in one folder from "activity{x}" to "activity{y)"
```powershell
cd 'C:\Users\Public\Documents\Vicon\data\Vicon_F\Round3\XinyanWang\FullCollection'
$old_activity_name = 'activity09.'
$new_activity_name = 'activity00.'
Get-ChildItem -Filter $old_activity_name* | ForEach-Object {Rename-Item $_ -NewName ($_.Name -replace $old_activity_name, $new_activity_name)}

```


