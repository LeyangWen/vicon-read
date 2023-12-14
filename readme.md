# VICON Reader

This repo contains tools for reading and processing Vicon data.
1. Scripts to use with an open Vicon Nexus session to read and save data in real-time
2. Scripts to process the saved c3d data
3. Supporting classes and functions

Here is a [YouTube playlist](https://youtube.com/playlist?list=PLjMAlxkYpRr0PwPyE3-LDrwiz8xIsNuma&si=HpLm-B3SFCHOHMK4) generated using this repo.

## 1. Scripts to use with an open Vicon Nexus session to read and save data in real-time
### 1.1. Tools for cleaning up Vicon data
* [detect_swap.py](detect_swap.py): Detects if the markers are swapped and mark as Vicon events, also automatically swaps the markers in simpler cases
* [swap_by_frame.py](swap_by_frame.py): Batch swap markers by frame number, normally between left and right
* [export_files.py](export_files.py): Export files from Vicon Nexus session
* [backup_c3d.py](backup_c3d.py): Backup c3d files in the backup folder before irreversible operations like smoothing. 

### 1.2. Tools for processing Vicon data
* [caculateVEHSskeleton_ergo.py](caculateVEHSskeleton_ergo.py): Calculate the 2D and 3D VEHS skeleton joint centers and angles using the VEHSR3 dataset skeleton. Also output cdf in H36M format
* [analog_compare.py](analog_compare.py): Compare analog data from two analog forceplates/transducers, used for rough calibration of force and torque data
* [branch-drop_accel.acceltest.py](acceltest.py): contains code to validate Vicon acceleration calculations by dropping markers
* [branch-pendulum_accel.acceltest.py](acceltest.py): contains code to validate Vicon acceleration calculations by pendulum motion

## 2. Scripts to process the saved c3d data
* [caculateSkeleton.py](caculateSkeleton.py): Reads the c3d files, calculates the 2D and 3D skeleton joint centers (66) and outputs in a MotionBert format. 


## 3. Supporting classes and functions
* [ForceTorqueTransducer.py](ForceTorqueTransducer.py): Class for calculating the force and torque from the force plate and a custom-wired analog AMTI force transducer
* [Camera.py](Camera.py): Class for projection, distortion correction, and other camera-related calculations
* [Point.py](Point.py): Classes for 3D point, plane, coordinate system, and joint angles calculations.
* [Skeleton.py](Skeleton.py): Class for defining pose skeleton structure, outputting 3DSSPP batch file, and visualization

### Useful snippets
- Powershell code to convert all avi files in a folder to mp4, keep original files
```powershell
cd 'C:\Users\Public\Documents\Vicon\data\Vicon_F\Round3\LeyangWen\FullCollection'
# skip if mp4 already exists
Get-ChildItem -Filter *.avi | ForEach-Object { If (!(Test-Path "$($_.DirectoryName)/$($_.BaseName).mp4")) {ffmpeg -i $_.FullName -c:v copy -c:a copy -y "$($_.DirectoryName)/$($_.BaseName).mp4"}}

```
- Batch rename of all files in one folder from "activity{x}" to "activity{y)"
```powershell
cd 'C:\Users\Public\Documents\Vicon\data\Vicon_F\Round3\XinyanWang\FullCollection'
$old_activity_name = 'activity09.'
$new_activity_name = 'activity00.'
Get-ChildItem -Filter $old_activity_name* | ForEach-Object {Rename-Item $_ -NewName ($_.Name -replace $old_activity_name, $new_activity_name)}

```


