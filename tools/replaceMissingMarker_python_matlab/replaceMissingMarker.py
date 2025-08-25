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
# - viconnexusapi
# - viconnexusutils
# - math
#
# Written by Nev Pires, 1/11/2020
#

from viconnexusapi import ViconNexus
import numpy as np
from viconnexusutils import NexusSegment
from viconnexusutils import NexusTrajectory
import tkinter as tk
from tkinter import messagebox
from tkinter.messagebox import *
import sys
from easygui import *
import math


def quit():
    root.destroy()


root = tk.Tk()
tk.Label(root,
         text="Please load the trial where all 4 markers are on the rigid body segment in Nexus").grid(column=0, row=0)
tk.Button(root, text="OK", command=quit).grid(column=0, row=1)
root.mainloop()

# Connects to Nexus
vicon = ViconNexus.ViconNexus()

# Gets the list of subject names
subject_list = vicon.GetSubjectNames()

# Get Start and End frames for the entire trial
start_frame, end_frame = vicon.GetTrialRegionOfInterest()


class Input(tk.Frame):

    # Creates the input box where the 4 markers on the rigid body segment are to be
    # entered and the subject name is to be selected from the dropdown menu

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent

        # Creates the dropdown menu with the subject names listed

        self.subject_selection = tk.StringVar()
        self.subject_selection.set(subject_list[0])

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
        self.ideal_entry = tk.OptionMenu(root, self.subject_selection, *subject_list)

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
        # global name
        # global ideal_type
        self.marker1 = self.marker_1_entry.get()
        self.marker2 = self.marker_2_entry.get()
        self.marker3 = self.marker_3_entry.get()
        self.marker4 = self.marker_4_entry.get()
        self.subject = self.subject_selection.get()
        # self.destroy()
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
marker_list = sorted([marker1, marker2, marker3, marker4])
full_marker_list = vicon.GetMarkerNames(subject)

found = [marker in full_marker_list for marker in marker_list]
trajExists = [[] for item in range(len(found))]

for marker in range(len(marker_list)):
    if found[marker]:
        trajExists[marker] = vicon.HasTrajectory(subject, marker_list[marker])
    else:
        messagebox.showinfo('ERROR', 'Error: Marker ' + ' ' + marker_list[marker] + ' ' + ' does not exist in VSK')
        sys.exit('Error: Marker ' + ' ' + marker_list[marker] + ' ' + ' does not exist in VSK')

# Check to see that the markers are properly labeled
for marker in marker_list:
    if vicon.HasTrajectory(subject, marker) == False:
        messagebox.showinfo('ERROR', 'Error: Marker ' + ' ' + marker + ' ' + ' is not labeled in the trial')
        sys.exit('Error: Marker ' + ' ' + marker + ' ' + ' is not labeled in the trial')

markers = {names: NexusTrajectory.NexusTrajectory(subject) for names in marker_list}
_ = {names: markers[names].Read(names, vicon) for names in marker_list}
markers = {names: np.array(markers[names].Position()) for names in marker_list}


def CreateSegment(subject, origin, p1, p2, p3):
    """Creates and populates the Nexus Segment. The segment has a rotation of xyz"""
    segment = NexusSegment.NexusSegment(subject)
    segment.Populate(origin, p1 - p2, p2 - p3, "xyz")

    return segment


# Create a dictionary for the permutations of the missing markers
segments = {"s123": {"segment": CreateSegment(subject, markers[marker_list[2]],
                                              markers[marker_list[1]], markers[marker_list[2]],
                                              markers[marker_list[3]]), marker_list[0]: markers[marker_list[0]]},
            "s023": {"segment": CreateSegment(subject, markers[marker_list[2]],
                                              markers[marker_list[0]], markers[marker_list[2]],
                                              markers[marker_list[3]]), marker_list[1]: markers[marker_list[1]]},
            "s013": {"segment": CreateSegment(subject, markers[marker_list[1]],
                                              markers[marker_list[0]], markers[marker_list[1]],
                                              markers[marker_list[3]]), marker_list[2]: markers[marker_list[2]]},
            "s012": {"segment": CreateSegment(subject, markers[marker_list[1]],
                                              markers[marker_list[0]], markers[marker_list[1]],
                                              markers[marker_list[2]]), marker_list[3]: markers[marker_list[3]]}}

local_point = {name: np.array(segments[segment_name]["segment"].LocalisePoint(segments[segment_name][name]))
               for segment_name in segments.keys() for name in segments[segment_name].keys() if name != "segment"}

# Load Dynamic Trial

yes = True

while yes:
    load_dynamic_option = ynbox("Load Dynamic Trial?", "Load Dynamic?", ["Yes", "No"])

    if load_dynamic_option:
        yes = True
    elif not load_dynamic_option:
        yes = False
        break

    load_dynamic = msgbox(
        "Please the load dynamic trial, where one marker on the rigid body segment is missing for the entirety of the trial in Nexus")

    vicon = ViconNexus.ViconNexus()  # Connect to Nexus for Dynamic Trial

    # Check to see which markers exist
    marker_exist = {marker: vicon.HasTrajectory(subject, marker) for marker in marker_list}

    # Check to see if the markers have any data for at least a frame.
    # If the marker has data in a single frame, an error will be generated
    if all(marker_exist.values()):
        msgbox(
            'Error: All markers have at least one frame of data in the trial. Please use the code "replace4Macro.py" code instead')
        sys.exit(
            'Error: All markers have at least one frame of data in the trial. Please use the code "replace4Macro.py" code instead')

    # Determine which Local Coordinate System to use
    if not marker_exist[marker_list[0]]:
        local = local_point[marker_list[0]]
    elif not marker_exist[marker_list[1]]:
        local = local_point[marker_list[1]]
    elif not marker_exist[marker_list[2]]:
        local = local_point[marker_list[2]]
    elif not marker_exist[marker_list[3]]:
        local = local_point[marker_list[3]]

    local = [frames for frames in local if math.isnan(frames[0]) == False]

    # Extract the 3 remaining marker names, eliminating the missing marker
    dynamic_marker_list = sorted([marker for marker, exists in marker_exist.items() if exists == True])

    # Get Dynamic Marker Data
    dynamic_markers = {name: NexusTrajectory.NexusTrajectory(subject) for name in dynamic_marker_list}
    _ = {name: dynamic_markers[name].Read(name, vicon) for name in dynamic_marker_list}
    dynamic_markers = {name: np.array(dynamic_markers[name].Position()) for name in dynamic_marker_list}

    # Define Missing Segment
    dynamic_segment = NexusSegment.NexusSegment(subject)
    dynamic_segment.Populate(dynamic_markers[dynamic_marker_list[1]],
                             dynamic_markers[dynamic_marker_list[0]] - dynamic_markers[dynamic_marker_list[1]],
                             dynamic_markers[dynamic_marker_list[1]] - dynamic_markers[dynamic_marker_list[2]],
                             'xyz')

    # Perform Local to Global transformation recreating the missing marker
    global_point = NexusTrajectory.NexusTrajectory(subject)
    global_point.SetPosition(dynamic_segment.GlobalisePoint(local[0]))

    # Find the missing marker name
    output_marker_name = [marker for marker in marker_list if marker not in dynamic_marker_list][0]

    output_data_exists = [False if np.isnan(frame) == True else
                          True for frame in np.array(global_point.Position())[:, 0]]

    # Write data to the file

    vicon.SetTrajectory(subject, output_marker_name, np.array(global_point.Position())[:, 0],
                        np.array(global_point.Position())[:, 1],
                        np.array(global_point.Position())[:, 2],
                        output_data_exists)

    # Prompt to save data in Nexus
    _ = msgbox('Please Save the trial in Nexus')
