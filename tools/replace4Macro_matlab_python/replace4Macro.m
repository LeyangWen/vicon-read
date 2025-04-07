% This script is designed to replicate the Replace4Macro that is used in
% BodyBuilder. The original code can be found from our website
% www.vicon.com/downloads under the heading "Models and Scripts", in the
% download called "BodyBuilder Example Models".
%
% Written by Nev Pires, 12/2/2020
%
% This code will work if one marker on a rigid body segment is missing for
% a few frames in the trial. If two or more markers on the rigid body
% segment are missing in the same frame/s, they will not be replaced. If a
% marker is missing for the entire trial, please use the companion code
% "replaceMissingMarker.m"
%
% The inputs are the Subject Name: as a string which is the same as the VSK
% name;
% p1, p2, p3 and p4 are the markers on the rigid body segment.
% These should be entered as strings, and they need to match the marker names in the VSK.
% If one out of the four markers are missing (but is present in at least one frame), 
% this code will replace the
% missing marker.

%% Clear the command window
clear; clc;

%% Connect to Nexus
vicon = ViconNexus();

%% Create the dailog box for the subject name, and the four markers on the rigid body cluster
inputs = inputdlg({'Subject Name', 'Marker1', 'Marker2', 'Marker3', 'Marker4'}, 'Rigid Body Marker Clusters');

%% Parse the inputs into their respective variables 
subject = inputs{1};

marker_names = sort(inputs(2:end));

%% See if the subject name is in the list of subjects from Nexus

subject_list = vicon.GetSubjectNames();

if ~any(strcmp(subject, subject_list))
    questdlg('The Subject Name does not match the VSK. Please enter the correct Subject Name', ...
        '','Ok','Ok');
end

%% Get the Marker Information

full_marker_list = vicon.GetMarkerNames(subject);

found = logical(true(1,length(marker_names)));
for marker = 1:length(marker_names)
    found(marker) = false;
    for marker_list_name = 1:length(full_marker_list)
        if strcmp(full_marker_list(marker_list_name), marker_names{marker}) == 1
            found(marker) = true;
        end
    end
end

%% Check if Trial has all four markers present

trajExists = logical(true(1,length(marker_names)));

for marker = 1:length(marker_names)
    if found(marker) == true
        trajExists(marker) = vicon.HasTrajectory(subject, marker_names{marker});
    else
        error('A marker does not exist in the VSK')
    end
end

%% Get Marker Data

for marker = 1:length(marker_names)

    [x.(marker_names{marker}), y.(marker_names{marker}), z.(marker_names{marker}), e.(marker_names{marker})] = vicon.GetTrajectory(subject, marker_names{marker});
     
    for frame = 1:length(x.(marker_names{marker}))
        if e.(marker_names{marker})(frame) == false
            x.(marker_names{marker})(frame) = NaN;
            y.(marker_names{marker})(frame) = NaN;
            z.(marker_names{marker})(frame) = NaN;
        end
    end

    marker_data.(marker_names{marker}) = [x.(marker_names{marker}); y.(marker_names{marker}); z.(marker_names{marker})];

end

%% Create Nexus trajectory objects for each marker and populate them with the marker data that has already been imported

marker1 = NexusTrajectory(subject);
marker1.SetPosition(marker_data.(marker_names{1}));

marker2 = NexusTrajectory(subject);
marker2.SetPosition(marker_data.(marker_names{2}));

marker3 = NexusTrajectory(subject);
marker3.SetPosition(marker_data.(marker_names{3}));

marker4 = NexusTrajectory(subject);
marker4.SetPosition(marker_data.(marker_names{4}));

%% Define Segments if a single marker on the 4 marker cluster is missing

s234 = NexusSegment(subject); % Creates Nexus segment object in Matlab
s234.Populate(marker3, marker2 - marker3, marker3 - marker4, 'xyz'); % Creates a segment with markers 2, 3, 4 as marker 1 is missing

s134 = NexusSegment(subject); % Creates Nexus segment object in Matlab
s134.Populate(marker3, marker1 - marker3, marker3 - marker4, 'xyz'); % Creates a segment with markers 1, 3, 4 as marker 2 is missing

s124 = NexusSegment(subject); % Creates Nexus segment object in Matlab
s124.Populate(marker2, marker1 - marker2, marker2 - marker4, 'xyz'); % Creates a segment with markers 1, 2, 4 as marker 3 is missing

s123 = NexusSegment(subject); % Creates Nexus segment object in Matlab
s123.Populate(marker2, marker1 - marker2, marker2 - marker3, 'xyz'); % Creates a segment with markers 1, 2, 3 as marker 4 is missing

%% Localise the missing marker in the corresponding coordinate system (created from the 3 remaining markers)

p1_local = s234.LocalisePoint(marker1); % Localise p1 in segment s234
p2_local = s134.LocalisePoint(marker2); % Localise p2 in segment s134
p3_local = s124.LocalisePoint(marker3); % Localise p3 in segment s124
p4_local = s123.LocalisePoint(marker4); % Localise p4 in segment s123

%% Calculate the mean of the localised markers

p1mean = mean(p1_local','omitnan')'; % Calculates the average of p1 local point
p2mean = mean(p2_local','omitnan')'; % Calculates the average of p2 local point
p3mean = mean(p3_local','omitnan')'; % Calculates the average of p3 local point
p4mean = mean(p4_local','omitnan')'; % Calculates the average of p4 local point

%% Create Nexus Trajectory objects for each marker

p1 = NexusTrajectory(subject);
p2 = NexusTrajectory(subject);
p3 = NexusTrajectory(subject);
p4 = NexusTrajectory(subject);

%% Globalize each point

p1.SetPosition(s234.GlobalisePoint(p1mean));
p2.SetPosition(s134.GlobalisePoint(p2mean));
p3.SetPosition(s124.GlobalisePoint(p3mean));
p4.SetPosition(s123.GlobalisePoint(p4mean));

%% Extract the position for each marker from the Nexus object

position_outputs.(marker_names{1}) = p1.Position;
position_outputs.(marker_names{2}) = p2.Position;
position_outputs.(marker_names{3}) = p3.Position;
position_outputs.(marker_names{4}) = p4.Position;

%% Replace missing frames (NaN's) in original data with the calculated data

for marker = 1:length(marker_names)
    for frame = 1:length(marker_data.(marker_names{marker}))
        if all(isnan(marker_data.(marker_names{marker})(:,frame)))
            marker_data.(marker_names{marker})(:,frame) = position_outputs.(marker_names{marker})(:,frame);
        else
            marker_data.(marker_names{marker})(:,frame) = marker_data.(marker_names{marker})(:,frame);
        end
    end
end

%% Determine if marker exists for the frame

for marker = 1:length(marker_names)
    for frame = 1:length(marker_data.(marker_names{marker}))
        if all(isnan(marker_data.(marker_names{marker})(:,frame)))
            exists_new.(marker_names{marker})(frame) = false;
        else
            exists_new.(marker_names{marker})(frame) = true;
        end
    end
end

%% Set Trajectories

for marker = 1:length(marker_names)

    vicon.SetTrajectory(subject, marker_names{marker},...
        marker_data.(marker_names{marker})(1,:),...
        marker_data.(marker_names{marker})(2,:),...
        marker_data.(marker_names{marker})(3,:),...
        exists_new.(marker_names{marker}))

end
