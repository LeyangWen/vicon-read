Script Name: replaceMissingMarker

Language: Matlab, Python (2.7)

Summary:

This script will replace a marker that is missing on a 4 marker rigid body cluster during a dynamic trial, but is present in a static trial. This code cannot be run in Nexus 2 as it requires the user to open trials manually in Nexus 2 and this cannot be done through a pipeline. It must be launched from either Matlab or Python. When executing the code from Matlab or Python, you will be prompted open the trial containing all 4 markers on the rigid body segment in Nexus 2 and press OK in either the Matlab or Python prompt. You will then be prompted to enter the names of the 4 markers on the rigid body segment and to select the subject name from a dropdown menu (Python) or type the name in (Matlab). After pressing "Submit" (Python) or "OK" (Matlab), it will then ask you if you want to load a DYNAMIC trial? If Yes is selected, you will need to load the first dynamic trial. The Matlab or Python prompt will ask you to save the trial within Nexus 2. It will then repeat the process. Press yes, load as many dynamic trials as you would like to run the operation on, and save them in Nexus 2. When all trials have been processed, select No, or Cancel. If a dynamic trial has the missing marker present in even a single frame, the code will not work. Instead, please use the script "replace4Macro.m" or "replace4Macro.py" (available to download via the Vicon website, see models and scripts)

Dependencies: 

Matlab: None (originally created in Matlab 2018a)
Python modules: numpy, easygui, tkinter

Run in Vicon Nexus: No. Must be run within Matlab or Python with Nexus running alongside.

Example data provided: Yes, (Processed in Nexus 2.11)

To run this code on the example data provided, please run the script from Python or Matlab.

The subjects name is 'Katlin', and the missing markers are the 'C7', 'T10', 'CLAV', 'STRN'.
The calibration trial that contains all 4 markers on the rigid body segment is called 'Cal'.
There are 4 trials provided where one marker (C7, T10, CLAV, STRN) is missing for the duration of
that trial.
Once the trial has been loaded, the appropriate marker will be recreated and you will be prompted to save the trial within Nexus and to select another trial with a marker on the rigid body segment missing for the entire trial if desired. Once all trials have been processed, press "No" when asked if you want a dynamic trial to be loaded.

Author: Nev Pires - Vicon Motion Systems, Inc.
