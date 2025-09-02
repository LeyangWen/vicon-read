# This script will replace a marker that is missing on a 4
# marker rigid body cluster during a dynamic trial, but is present in a
# static trial.
#
# If the marker is present for at least 1 frame in the trial, this code will not work.
# Please use the companion code "replace4Macro.py" instead.
#
# This code must be run in Python and cannot be run in Nexus 2 as it requires the user to open trials
# manually in Nexus 2 and this cannot be done through a pipeline.
#
# When executing the code, you will be prompted to open the trial in Nexus
# 2 that contains all 4 markers on the rigid body segment, and to then press
# OK in the Python prompt.
# You will then be prompted to type the names of the four markers
# on the rigid body cluster, and to choose the subject's name via a dropdown menu in the Python prompt.
#
# After pressing "Submit", it will then ask you if you want to load a DYNAMIC
# trial? If Yes is selected, you will be prompted to load a dynamic trial where a marker on the
# rigid body segment is missing for the duration of the trial. Please load the trial in Nexus and press 'OK' in the
# Python prompt. After the marker has been replaced, you will be asked to save the trial in Nexus 2. Please save the
# trial in Nexus 2 and press 'OK' in the Python prompt. You will be asked if you would like to repeat the process for
# other trials. When you don't want to process other trials, please select 'No'.
#
# If a dynamic trial has the missing marker present in even a single frame, the code will
# not work. Instead, please use the script "replace4Macro.py"
#
# This script requires the following modules:
#
# - numpy
# - easygui
# - tkinter
#
# Written by Nev Pires, 1/11/2020
#

import math
import numpy as np
from easygui import *
import sys
sys.path.append('C:/Program Files (x86)/Vicon/Nexus2.11/SDK/Win32')
sys.path.append('C:/Program Files (x86)/Vicon/Nexus2.11/SDK/Python')

import ViconNexus
from NexusTrajectory import NexusTrajectory  # Import the Nexus Trajectory Module
from NexusSegment import NexusSegment  # Import the Nexus Segment Module
from tkinter import *
import tkinter as tk

# Prompt for the user to load the static trial within Nexus

loadStatic = msgbox('Please load the static trial, where all 4 markers on the rigid body segment are present in Nexus')

# Connects to Nexus and creates the Nexus object for the static trial

vicon = ViconNexus.ViconNexus()

# Gets the list of subject names

subjectList = vicon.GetSubjectNames()

class Input(tk.Frame):

    # Creates the input box where the 4 markers on the rigid body segment are to be entered and the subject name is to be selected from the dropdown menu

    def __init__(self, parent):

        tk.Frame.__init__(self, parent)
        self.parent = parent

        # Creates the dropdown menu with the subject names listed

        self.subject_selection = tk.StringVar()
        self.subject_selection.set(subjectList[0])

        # Creates 4 input boxes that correspond with the 4 markers that are on the rigid body segment

        self.marker_1_label = tk.Label(root, text="Marker 1: ")
        self.marker_1_entry = tk.Entry(root)

        self.marker_2_label = tk.Label(root, text="Marker 2: ")
        self.marker_2_entry = tk.Entry(root)

        self.marker_3_label = tk.Label(root, text="Marker 3: ")
        self.marker_3_entry = tk.Entry(root)

        self.marker_4_label = tk.Label(root, text="Marker 4: ")
        self.marker_4_entry = tk.Entry(root)

        # Creates the dropdown menu with the subject names listed. A default subject is chosen if multiple subjects are in the trial

        self.ideal_label = tk.Label(root, text="Subject Name: ")
        self.ideal_entry = tk.OptionMenu(root, self.subject_selection, *subjectList)

        # Creates a button to close the window and continue

        self.submit_button = tk.Button(text="Submit", command=self.close_window)

        # Creates the location of the text and input boxes for the dialog box

        self.marker_1_label.grid(row=0, column=0)
        self.marker_1_entry.grid(row=0, column=1)

        self.marker_2_label.grid(row=1, column=0)
        self.marker_2_entry.grid(row=1, column=1)

        self.marker_3_label.grid(row=2, column=0)
        self.marker_3_entry.grid(row=2, column=1)

        self.marker_4_label.grid(row=3, column=0)
        self.marker_4_entry.grid(row=3, column=1)

        self.ideal_label.grid(row=4, column=0)
        self.submit_button.grid(columnspan=2, row=5, column=0)

        self.ideal_entry.grid(row=4, column=1)

    # function to close the window

    def close_window(self):
        #global name
        #global ideal_type
        self.marker1 = self.marker_1_entry.get()
        self.marker2 = self.marker_2_entry.get()
        self.marker3 = self.marker_3_entry.get()
        self.marker4 = self.marker_4_entry.get()
        self.subject = self.subject_selection.get()
        #self.destroy()
        self.quit()

# Executes the above function and class

if __name__ == '__main__':
    root = tk.Tk()
    app = Input(root)
    root.mainloop()
    # Note the returned variables here
    # They must be assigned to external variables
    # for continued use
    marker1 = app.marker1
    marker2 = app.marker2
    marker3 = app.marker3
    marker4 = app.marker4
    subject = app.subject
    # Should only need root.destroy() to close down tkinter
    # But need to handle user cancelling the form instead
    try:
        root.destroy()
    except:
        sys.exit(1)

# Check to see if markers exist in the subject list

markerList = sorted([marker1, marker2, marker3, marker4])
fullMarkerList = vicon.GetMarkerNames(subject)

found = [marker in fullMarkerList for marker in markerList]
trajExists = [[] for item in range(len(found))]

for marker in range(len(markerList)):
    if found[marker] == True:
        trajExists[marker] = vicon.HasTrajectory(subject, markerList[marker])
        if trajExists[marker] == False:
            msgbox('Error: Marker' + ' ' + markerList[marker] + ' ' + 'has not been labeled in trial')
            sys.exit('Error: Marker' + ' ' + markerList[marker] + ' ' + 'has not been labeled in trial')
    else:
        msgbox('Error: Marker ' + ' ' + markerList[marker] + ' ' + ' does not exist in VSK')
        sys.exit('Error: Marker ' + ' ' + markerList[marker] + ' ' + ' does not exist in VSK')

markers = {names: NexusTrajectory(subject) for names in markerList}
markerRead = {names: markers[names].Read(names, vicon) for names in markerList}

# Define Segment if marker 1 (0 in Python) is missing in subsequent trials

s234 = NexusSegment(subject)
s234.Populate(markers[markerList[2]], markers[markerList[1]] - markers[markerList[2]], markers[markerList[2]] - markers[markerList[3]], 'xyz')

p1V1 = NexusTrajectory(subject)
p1V1 = s234.LocalisePoint(markers[markerList[0]])

# Define Segment if marker 2 (1 in Python) is missing in subsequent trials

s134 = NexusSegment(subject)
s134.Populate(markers[markerList[2]], markers[markerList[0]] - markers[markerList[2]], markers[markerList[2]] - markers[markerList[3]], 'xyz')

p2V1 = NexusTrajectory(subject)
p2V1 = s134.LocalisePoint(markers[markerList[1]])

# Define Segment if marker 3 (2 in Python) is missing in subsequent trials

s124 = NexusSegment(subject)
s124.Populate(markers[markerList[1]], markers[markerList[0]] - markers[markerList[1]], markers[markerList[1]] - markers[markerList[3]], 'xyz')

p3V1 = NexusTrajectory(subject)
p3V1 = s124.LocalisePoint(markers[markerList[2]])

# Define Segment if marker 4 (3 in Python) is missing in subsequent trials

s123 = NexusSegment(subject)
s123.Populate(markers[markerList[1]], markers[markerList[0]] - markers[markerList[1]], markers[markerList[1]] - markers[markerList[2]], 'xyz')

p4V1 = NexusTrajectory(subject)
p4V1 = s123.LocalisePoint(markers[markerList[3]])

# Load Dynamic Trial

yes = True

while yes == True:

    loadDynamicOption = ynbox("Load Dynamic Trial?", "Load Dynamic?", ["Yes", "No"])

    if loadDynamicOption == True:
        yes = True
    elif loadDynamicOption == False:
        yes = False
        break

    loadDynamic = msgbox("Please the load dynamic trial, where one marker on the rigid body segment is missing for the entirety of the trial in Nexus")

    vicon = ViconNexus.ViconNexus()  # Connect to Nexus for Dynamic Trial

    # Check to see which markers exist

    markerExist = {marker: vicon.HasTrajectory(subject, marker) for marker in markerList}

    # Check to see if the markers have any data for at least a frame. If the marker has data in a single frame, an error will be generated

    if all(markerExist.values()) == True:
        msgbox('Error: All markers have at least one frame of data in the trial. Please use the code "replace4Macro.py" code instead')
        sys.exit('Error: All markers have at least one frame of data in the trial. Please use the code "replace4Macro.py" code instead')

    # Determine which Local Coordinate System to use

    if markerExist[markerList[0]] == False:
        local = p1V1
    elif markerExist[markerList[1]] == False:
        local = p2V1
    elif markerExist[markerList[2]] == False:
        local = p3V1
    elif markerExist[markerList[3]] == False:
        local = p4V1

    local = [frames for frames in local if math.isnan(frames[0]) == False]

    # Extract the 3 remaining marker names, eliminating the missing marker

    dynamicMarkerList = sorted([markerName for markerName, exists in markerExist.items() if exists == True])

    # Get Dynamic Marker Data

    dynamicMarkers = {names: NexusTrajectory(subject) for names in dynamicMarkerList}
    dynamicMarkerRead = {names: dynamicMarkers[names].Read(names, vicon) for names in dynamicMarkerList}

    # Define Missing Segment

    dynamicSegment = NexusSegment(subject)
    dynamicSegment.Populate(dynamicMarkers[dynamicMarkerList[1]],
                                dynamicMarkers[dynamicMarkerList[0]] - dynamicMarkers[dynamicMarkerList[1]],
                                dynamicMarkers[dynamicMarkerList[1]] - dynamicMarkers[dynamicMarkerList[2]], 'xyz')

    # Perform Local to Global transformation recreating the missing marker

    globalPoint = NexusTrajectory(subject)
    globalPoint.SetPosition(dynamicSegment.GlobalisePoint(local[0]))

    # Find the missing marker name

    outputMarkerName = [marker for marker in markerList if marker not in dynamicMarkerList][0]

    # Extract Position Data to new array

    outputDataX = globalPoint._position[:,0].tolist()
    outputDataY = globalPoint._position[:,1].tolist()
    outputDataZ = globalPoint._position[:,2].tolist()

    outputDataExists = [False if np.isnan(frame) == True else True for frame in outputDataX]

    # Write data to the file

    setMissingMarker = vicon.SetTrajectory(subject, outputMarkerName, outputDataX, outputDataY, outputDataZ, outputDataExists)

    # Prompt to save data in Nexus

    saveDataPrompt = msgbox('Please Save the trial in Nexus')