# VICON Reader

#### Snippets
- Powershell code to convert all avi files in a folder to mp4, keep original files
```powershell
cd 'C:\Users\Public\Documents\Vicon\data\Vicon_F\Round3\LeyangWen\FullCollection'
# skip if mp4 already exists
Get-ChildItem -Filter *.avi | ForEach-Object { If (!(Test-Path "$($_.DirectoryName)/$($_.BaseName).mp4")) {ffmpeg -i $_.FullName -c:v copy -c:a copy -y "$($_.DirectoryName)/$($_.BaseName).mp4"}}

```
- Batch rename of all files in one folder from "activity{x}" to "activit{y)"
```powershell
cd 'C:\Users\Public\Documents\Vicon\data\Vicon_F\Round3\XinyanWang\FullCollection'
$old_activity_name = 'activity09.'
$new_activity_name = 'activity00.'
Get-ChildItem -Filter $old_activity_name* | ForEach-Object {Rename-Item $_ -NewName ($_.Name -replace $old_activity_name, $new_activity_name)}

```
```

#### Visualizations
https://www.youtube.com/watch?v=BCuvBHVFi4Q&ab_channel=leyangwen

```

### 3DSSPP Modifications
* **SSPPAPP/hom_doc.h**: 
```diff
-line 592: protected:

+line 592: public:
```
* **SSPPAPP/Hom_doc.cpp**:
```diff
-line 3389: if( lDiag.DoModal() == IDOK )
-line 3390: {
-line 3391:   mBatchFilePathName = lDiag.GetPathName();
-line 3392:   mBatchFileRootName = lDiag.GetFileTitle();
-line 3401:   CString lMsg("Begin processing batch file '");
-line 3402:   lMsg += mBatchFilePathName;
-line 3403:   lMsg += "'?";

-line 3405:   if(AfxMessageBox(lMsg, MB_OKCANCEL) == IDOK)
-line 3406:   {
-line 3407:     CString RootPath = mBatchFilePathName;
-line 3408:     RootPath.Delete(RootPath.GetLength() - 4,4);
-line 3409:     BatchFile   lBatchFile
-line 3410:     (
-line 3411:       LPCTSTR( mBatchFilePathName ),
-line 3412:       LPCTSTR( RootPath ) ,
-line 3413:       this
-line 3414:     );

-line 3416:     lBatchFile.Process();
-line 3417:  }
-line 3418:  else
-line 3419:  {
-line 3420:    AfxMessageBox( "Canceling batch processing." );
-line 3421:  }
-line 3422: }

+line 3424: string batchfilename = "./batchinput.txt";
+line 3425: string batchfileroot = "../export/batchinput";
+line 3426: bf(batchfilename, batchfileroot, this);
+line 3427: bf.process();

+line 1026: this->OnTaskinputBatch();
+line 1027: this->DoClose()
```

* **SSPPAPP/hom.cpp**:
```diff
-line 230: if (1)

+line 230: if (0)
```

* **SSPPAPP/Main_frm.cpp**:
```diff
-line 600: int lBoxAnswer = AfxMessageBox("Save chanegs to current task?", MB_YESNOCANCEL);

+line 600: int lBoxAnswer = 2(IDCANCEL);
+line 601: // AfxMessageBox("Save changes to current task?", MB_YESNOCANCEL);
```

* **SSPPAPP/BatchFile.cpp**:
```diff
-line 1695: AfxMessageBox(lMsg.str());

+line 1695: // AfxMessageBox(lMsg.str());
```

* **SSPPAPP/Logon.cpp**:

```diff
-line 30: if (Diag.DoModal() == IDOK)
{
                               //--- Intro 2
   // Diag.Intro_Message = this->Intro_Text_2();
   // if (Diag.DoModal() == IDOK)
   // {
                               //--- Intro 3
      // Diag.Intro_Message = this->Intro_Text_3();
      // if( Diag.DoModal() == IDOK )
      // {
            Permission_Granted = TRUE;
      // }         
      // else  AfxMessageBox( "Exiting program at user's request." ); //Intro 3 
   // }      
   // else  AfxMessageBox( "Exiting program at user's request." ); //Intro 2
}
-line 46: else  AfxMessageBox( "Exiting program at user's request." ); //Intro 1

+line 30: // if (Diag.DoModal() == IDOK)
+line 31: // {
                               //--- Intro 2
   // Diag.Intro_Message = this->Intro_Text_2();
   // if (Diag.DoModal() == IDOK)
   // {
                               //--- Intro 3
      // Diag.Intro_Message = this->Intro_Text_3();
      // if( Diag.DoModal() == IDOK )
      // {
            Permission_Granted = TRUE;
      // }         
      // else  AfxMessageBox( "Exiting program at user's request." ); //Intro 3 
   // }      
   // else  AfxMessageBox( "Exiting program at user's request." ); //Intro 2
+line 45: // }
+line 46: // else  AfxMessageBox( "Exiting program at user's request." ); //Intro 1 
```

