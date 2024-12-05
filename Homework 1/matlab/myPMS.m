function [N, albedo, re_rendered_img] = L2_PMS(data, m, shadow_removal_percentage)

    num_images = size(data.s, 1);
    [height, width, ~] = size(data.imgs{1});
    
    N = zeros(height, width, 3);      % Surface normals
    albedo = zeros(height, width);   % Albedo map
    re_rendered_img = zeros(height, width); % Re-rendered image
    I = zeros(num_images, length(m)); % Image intensities for valid pixels
    
    % Extract pixel intensities for each image
    for i = 1:num_images
        img = double(data.imgs{i});
        img = img(m);
        I(i, :) = img;
    end
    
    % Remove shadows and highlights, compute normals and albedo
    for i = 1:length(m)
        I_col = I(:, i);
        
        % Sort intensities and remove shadows/highlights
        [sorted_I, idx] = sort(I_col);
        num_to_remove = round(length(sorted_I) * shadow_removal_percentage / 100);
        valid_idx = idx(num_to_remove+1:end-num_to_remove);
        
        % Filter light source directions and intensities
        s_filtered = data.s(valid_idx, :);
        I_col_filtered = sorted_I(num_to_remove+1:end-num_to_remove);
        
        % Solve for normal and albedo
        A = s_filtered \ I_col_filtered;
        albedo_val = norm(A);

        norm_A = A / albedo_val;
        [row, col] = ind2sub([height, width], m(i));
        N(row, col, :) = norm_A';
        albedo(row, col) = albedo_val;
    end
    
    % Re-render the image using recovered normals and albedo
    viewing_direction = [0, 0, 1];
    for i = 1:height
        for j = 1:width
            normal = squeeze(N(i, j, :));
            if norm(normal) > 0
                re_rendered_img(i, j) = max(0, dot(normal, viewing_direction)) * albedo(i, j);
            end
        end
    end
    
    end
    