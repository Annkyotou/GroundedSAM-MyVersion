
clear
close all



I1 = imread("2ndLab-1_semantic_pano.jpg");
figure;imshow(I1,[]);set(gca,"XDir",'reverse');

I2 = imread("2ndLab-1_pano_mask.tif");
figure;imshow(I2,[]);colormap("parula");set(gca,"XDir",'reverse');


%% File path
filename = 'Scan02_labeled_combined.ply';

%% Open and parse header
fid = fopen(filename, 'r');
if fid == -1
    error('Failed to open file.');
end

% Read header lines
line = fgetl(fid);
vertexCount = 0;
while ischar(line)
    if startsWith(line, 'element vertex')
        vertexCount = sscanf(line, 'element vertex %d');
    elseif strcmp(line, 'end_header')
        break;
    end
    line = fgetl(fid);
end

%% Read data
% Format: x y z r g b classification
formatSpec = '%f %f %f %u8 %u8 %u8 %u8';
data = textscan(fid, formatSpec, vertexCount);

fclose(fid);

%% Extract
X = data{1};
Y = data{2};
Z = data{3};
R = data{4};
G = data{5};
B = data{6};
Labels = data{7};

%% Build MATLAB pointCloud
ptCloud = pointCloud([X Y Z], 'Color', [R G B]);

%% Visualize points colored by RGB
figure;
pcshow(ptCloud);
title('Original RGB');

%% Visualize points colored by classification label
% Create a color map for labels
unique_labels = unique(Labels);
nLabels = numel(unique_labels);

label_colors = uint8(255 * rand(nLabels, 3));
label_map = containers.Map(unique_labels, 1:nLabels);

label_rgb = zeros(numel(Labels), 3, 'uint8');
for i = 1:numel(Labels)
    idx = label_map(Labels(i));
    label_rgb(i, :) = label_colors(idx, :);
end

ptCloud_labels = pointCloud([X Y Z], 'Color', label_rgb);

figure;
pcshow(ptCloud_labels);
title('Segment Labels (by Classification)');
