This script is designed to replicate the Replace4Macro that is used in
BodyBuilder. The original code can be found from our website
www.vicon.com/downloads under the heading "Models and Scripts", in the
download called "BodyBuilder Example Models".

Written by Nev Pires, 12/2/2020

This code will work if one marker on a rigid body segment is missing for
a few frames in the trial. If two or more markers on the rigid body
segment are missing in the same frame/s, they will not be replaced. If a
marker is missing for the entire trial, please use the companion code
"replaceMissingMarker.m" (for Matlab) or "replaceMissingMarker.py" for Python

The Python script will require you to have the following modules: 

- numpy
- tkinter
- viconnexusapi
- viconnexusutils

The inputs are the Subject Name: as a string which is the same as the VSK
name;
p1, p2, p3 and p4 are the markers on the rigid body segment.
These should be entered as strings, and they need to match the marker names in the VSK.
If one out of the four markers are missing (but is present in at least one frame), 
this code will replace the missing marker. In Python, the subject name will need to be 
chosen from a dropdown menu. 

To run the code on the example data provided, please load the trial "Torso Markers Missing" 
in Nexus. Load either "Run Matlab Operation" or "Run Python Operation" into the current pipeline
and choose replace4Macro.m" (for Matlab) or "replace4Macro.py" for Python, and run the pipeline. 

You will be prompted in Matlab to first enter the subjects name, in this example 'Katlin', and 
subsequently the names of the four markers ('C7', 'T10', 'CLAV', 'STRN') that make up the 
rigid body segments. In Python, you will need to enter the names of the four markers 
('C7', 'T10', 'CLAV', 'STRN') on the rigid body segment, and to choose the subject name ('Katlin')
from the dropdown menu. 

In the example trial provided, the C7, T10, CLAV and STRN markers are missing at different times
in the trial T10 and STRN are both missing at the same time for part of the trial. During 
the frames where only one of the markers is missing, those markers will be recovered. However, 
for the frames where two markers on the segment are missing, neither marker will be replaced. 
