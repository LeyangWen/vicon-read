# VICON Reader

#### Snipets
- Powershell code to convert all avi files in a folder to mp4, keep original files
```powershell
cd 'C:\Users\Public\Documents\Vicon\data\Vicon_F\Round3\LeyangWen\FullCollection'
# skip if mp4 already exists
Get-ChildItem -Filter *.avi | ForEach-Object { If (!(Test-Path "$($_.DirectoryName)/$($_.BaseName).mp4")) {ffmpeg -i $_.FullName -c:v copy -c:a copy -y "$($_.DirectoryName)/$($_.BaseName).mp4"}}

```
- Batch rename of all files in one folder from "activity{x}" to "activit{y)"
```powershell
cd 'C:\Users\Public\Documents\Vicon\data\Vicon_F\Round3\DanLi\FullCollection'
$old_activity_name = 'activity001.'
$new_activity_name = 'activity01.'
Get-ChildItem -Filter $old_activity_name* | ForEach-Object {Rename-Item $_ -NewName ($_.Name -replace $old_activity_name, $new_activity_name)}

```
```

#### Visualizations
https://www.youtube.com/watch?v=BCuvBHVFi4Q&ab_channel=leyangwen