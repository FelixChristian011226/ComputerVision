function N = L2_PMS(data, m, shadow_removal_percentage)

num_images = size(data.s, 1);
[height, width, ~] = size(data.imgs{1});

N = zeros(height, width, 3);    % Surface normals
I = zeros(num_images, length(m));  % Image intensities for valid pixels

% Extract pixel intensities for each image
for i = 1:num_images
    img = double(data.imgs{i});
    img = img(m);
    I(i, :) = img;
end

% Remove shadows and highlights by sorting pixel intensities at each valid pixel
for i = 1:length(m)
    I_col = I(:, i);
    
    [sorted_I, idx] = sort(I_col);
    
    num_to_remove = round(length(sorted_I) * shadow_removal_percentage / 100);
    valid_idx = idx(num_to_remove+1:end-num_to_remove);
    
    % Filter the corresponding light source directions
    s_filtered = data.s(valid_idx, :);
    I_col_filtered = sorted_I(num_to_remove+1:end-num_to_remove); 
    
    % Solve for the surface normal for this pixel using the least squares formula
    A = s_filtered \ I_col_filtered;
    norm_A = A / norm(A);
    [row, col] = ind2sub([height, width], m(i));
    N(row, col, :) = norm_A';
end

end
