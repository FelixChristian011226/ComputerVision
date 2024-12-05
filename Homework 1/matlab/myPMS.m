function N = L2_PMS(data, m, shadow_removal_percentage)
% L2_PMS - Photometric Stereo using L2 norm (least squares) with shadow and highlight removal.
%   N = L2_PMS(data, m, shadow_removal_percentage) estimates the surface normal for each pixel
%   in the given data structure. The m argument specifies the pixel indices that are valid (i.e., not masked).
%   shadow_removal_percentage is the percentage of darkest and brightest pixels to discard for shadow removal.

% Inputs:
%  - data: A struct containing photometric stereo data.
%      data.s: The light source directions (K x 3).
%      data.L: The light source intensities (optional, not used here).
%      data.imgs: Cell array containing the images (K x 1).
%  - m: A vector of pixel indices that are valid (not masked).
%  - shadow_removal_percentage: Percentage of darkest and brightest pixels to discard (e.g., 20%).

% Outputs:
%  - N: The estimated surface normals (height x width x 3).

% Get the number of images and image dimensions
num_images = size(data.s, 1);  % number of images
[height, width, ~] = size(data.imgs{1});  % image size

% Initialize the surface normal matrix
N = zeros(height, width, 3);

% Stack all pixel values from all images (for valid pixels)
I = zeros(num_images, length(m));  % Image intensities for valid pixels
for i = 1:num_images
    img = double(data.imgs{i});  % Load the image as double
    img = img(m);  % Only take valid pixels
    I(i, :) = img;  % Store pixel values
end

% Remove shadows and highlights by sorting pixel intensities at each valid pixel
for i = 1:length(m)
    I_col = I(:, i);  % Pixel intensities for this valid pixel across all images
    
    % Sort the intensities
    [sorted_I, idx] = sort(I_col);
    
    % Discard the darkest and brightest percentages
    num_to_remove = round(length(sorted_I) * shadow_removal_percentage / 100);
    valid_idx = idx(num_to_remove+1:end-num_to_remove);  % Indices of remaining intensities
    
    % Filter the corresponding light source directions
    s_filtered = data.s(valid_idx, :);  % Corresponding rows of light source directions
    I_col_filtered = sorted_I(num_to_remove+1:end-num_to_remove);  % Remaining intensities
    
    % Solve for the surface normal for this pixel using the least squares formula
    A = s_filtered \ I_col_filtered;  % Solve L * N = I_col_filtered for N
    norm_A = A / norm(A);  % Normalize the surface normal
    [row, col] = ind2sub([height, width], m(i));  % Convert index to row, col
    N(row, col, :) = norm_A';  % Assign the normalized normal
end

end
