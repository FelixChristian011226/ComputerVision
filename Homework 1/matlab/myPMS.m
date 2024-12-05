function [N, rho] = L2_PMS_with_shadows(data, m, discard_percentage)
% L2_PMS_with_shadows - Photometric Stereo using L2 norm with shadow and highlight removal.
%   [N, rho] = L2_PMS_with_shadows(data, m, discard_percentage) estimates the surface normal 
%   and albedo for each pixel in the given data structure, and discards shadow and highlight 
%   observations by removing the darkest and brightest p% of pixel intensities.

% Inputs:
%  - data: A struct containing photometric stereo data.
%      data.s: The light source directions (nimages x 3).
%      data.L: The light source intensities (nimages x 3).
%      data.imgs: Cell array containing the images (nimages x 1).
%  - m: A vector of pixel indices that are valid (not masked).
%  - discard_percentage: The percentage of darkest and brightest pixels to discard.

% Outputs:
%  - N: The estimated surface normals (height x width x 3).
%  - rho: The estimated albedo (height x width).

% Get the number of images and image dimensions
num_images = size(data.s, 1);  % number of images
[height, width, ~] = size(data.imgs{1});  % image size

% Initialize the surface normal matrix and albedo matrix
N = zeros(height, width, 3);
rho = zeros(height, width);

% Stack all pixel values from all images (for valid pixels)
I = zeros(num_images, length(m));  % Image intensities for valid pixels
for i = 1:num_images
    img = double(data.imgs{i});  % Load the image as double
    img = img(m);  % Only take valid pixels
    I(i, :) = img;  % Store pixel values
end

% Process each pixel individually
for i = 1:length(m)
    % Get pixel intensities for this valid pixel across all images
    I_col = I(:, i);
    
    % Sort the intensities and discard the darkest and brightest p% values
    I_sorted = sort(I_col);
    num_discard = round(discard_percentage * num_images / 100);
    I_filtered = I_sorted(num_discard+1:end-num_discard);  % Filtered intensities
    
    % Recalculate the light matrix (using only the filtered intensities)
    L_filtered = data.s(1:length(I_filtered), :);  % Corresponding light directions
    
    % Solve for the surface normal using the least squares method
    A = L_filtered \ I_filtered;  % Solve L * N = I_filtered for N
    norm_A = A / norm(A);  % Normalize the surface normal
    
    % Assign the normalized normal and albedo (rho)
    [row, col] = ind2sub([height, width], m(i));  % Convert index to row, col
    N(row, col, :) = norm_A';  % Assign the normalized normal
    
    % Estimate the albedo (rho) as the ratio of the intensity and the dot product
    rho(row, col) = mean(I_filtered) / dot(norm_A, data.s(1, :)');
end

end
