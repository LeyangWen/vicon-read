# VICON Reader

#### Snipets
- Powershell code to convert all avi files in a folder to mp4, keep original files
```powershell
cd 'C:\Users\Public\Documents\Vicon\data\Vicon_F\Round3\LeyangWen\FullCollection'
# skip if mp4 already exists
Get-ChildItem -Filter *.avi | ForEach-Object { If (!(Test-Path "$($_.DirectoryName)/$($_.BaseName).mp4")) {ffmpeg -i $_.FullName -c:v copy -c:a copy -y "$($_.DirectoryName)/$($_.BaseName).mp4"}}

```