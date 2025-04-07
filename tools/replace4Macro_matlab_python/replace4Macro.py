# This script is designed to replicate the Replace4Macro that is used in
# BodyBuilder. The original code can be found from our website
# www.vicon.com/downloads under the heading "Models and Scripts", in the
# download called "BodyBuilder Example Models".
#
# Written by Nev Pires, 12/2/2020
#
# This code will work if one marker on a rigid body segment is missing for
# a few frames in the trial. If two or more markers on the rigid body
# segment are missing in the same frame/s, they will not be replaced. If a
# marker is missing for the entire trial, please use the companion code
# "replaceMissingMarker.py"
#
# The inputs are the p1, p2, p3 and p4 are the markers on the rigid body segment.
# These should be entered as strings.
# The subject name must be selected from the drop down menu.
# If one out of the four markers are missing (but is present in at least one frame),
# this code will replace the missing marker.
#
# This script requires the following modules:
#
# - numpy
# - tkinter
# - viconnexusapi
# - viconnexusutils

from viconnexusapi import ViconNexus
import numpy as np
from viconnexusutils import NexusSegment
from viconnexusutils import NexusTrajectory
import tkinter as tk
from tkinter import messagebox
import sys

# Connects to Nexus
vicon = ViconNexus.ViconNexus()

# Gets the list of subject names
subject_list = vicon.GetSubjectNames()

# Get Start and End frames for the entire trial
start_frame, end_frame = vicon.GetTrialRegionOfInterest()


class Input(tk.Frame):

    # Creates the input box where the 4 markers on the rigid body segment are to be entered and the subject name is to be selected from the dropdown menu

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

# Create a dictionary for the segments based on the remaining markers
local_point = {name: np.array(segments[segment_name]["segment"].LocalisePoint(segments[segment_name][name]))
               for segment_name in segments.keys() for name in segments[segment_name].keys() if name != "segment"}

# Calculate the local point average
local_point_ave = {name: np.nanmean(local_point[name], axis=0) for name in local_point.keys()}

# Create Nexus Trajectory objects for each marker
p_markers = {name: NexusTrajectory.NexusTrajectory(subject) for name in local_point}

# Globalize each point
_ = {name: p_markers[name].SetPosition(segments[segment_name]["segment"].GlobalisePoint(local_point_ave[name]))
     for segment_name in segments.keys() for name in segments[segment_name].keys() if name != "segment"}

# Extract the position for each marker from the Nexus object
p_positions = {name: np.array(p_markers[name].Position()) for name in p_markers}

# Replace missing frames (NaN's) in original data with the calculated data
marker_data = {name: np.array([p_positions[name][frame] if all(np.isnan(markers[name][frame]))
               else markers[name][frame] for frame in range(start_frame-1, end_frame)]) for name in p_positions.keys()}

# Determine if marker exists or not for a specific frame
exists_new = {names: [False if all(np.isnan(frames)) else True for frames in marker_data[names]]
              for names in marker_list}

# setPosition = {names: markers[names].SetPosition(markerData[marker_list[names]]) for names in marker_list}
_ = {name: vicon.SetTrajectory(subject, name,
                               list(marker_data[name][:, 0]),
                               list(marker_data[name][:, 1]),
                               list(marker_data[name][:, 2]),
                               exists_new[name]) for name in marker_list}
