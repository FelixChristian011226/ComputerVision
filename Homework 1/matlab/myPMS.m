function N = L2_PMS(data, m)
% L2_PMS - Photometric Stereo using L2 norm (least squares).
%   N = L2_PMS(data, m) estimates the surface normal for each pixel
%   in the given data structure. The m argument specifies the pixel
%   indices that are valid (i.e., not masked).

% Inputs:
%  - data: A struct containing photometric stereo data.
%      data.s: The light source directions (nimages x 3).
%      data.L: The light source intensities (nimages x 3).
%      data.imgs: Cell array containing the images (nimages x 1).
%  - m: A vector of pixel indices that are valid (not masked).

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

% Solve for surface normals using least squares
% We solve the equation I = L * N for N (the normal), where L is the light matrix.
L = data.s;  % Light source directions (nimages x 3)
% N = (L' * L)^(-1) * L' * I
% Note: We solve for N for each valid pixel.
for i = 1:length(m)
    % Solve for the surface normal for this pixel using the least squares formula
    I_col = I(:, i);  % Pixel intensities for this valid pixel across all images
    A = L \ I_col;  % Solve L * N = I_col for N
    norm_A = A / norm(A);  % Normalize the surface normal
    % Assign the normalized normal to the corresponding pixel in the output N
    [row, col] = ind2sub([height, width], m(i));  % Convert index to row, col
    N(row, col, :) = norm_A';  % Assign the normalized normal
end

end
