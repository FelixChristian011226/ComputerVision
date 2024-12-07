function [N, albedo, re_rendered_img] = L2_PMS(data, m, shadow_removal_percentage)

    num_images = size(data.s, 1);
    [height, width, ~] = size(data.imgs{1});
    
    N = zeros(height, width, 3);
    albedo = zeros(height, width, 3);
    re_rendered_img = zeros(height, width, 3);
    I = zeros(num_images, length(m), 3);
    
    % Extract pixel intensities for each image
    for c = 1:3
        for i = 1:num_images
            img = double(data.imgs{i});
            img = img(m);
            img = img / data.L(i, c);
            I(i, :, c) = img;
        end
    end
    
    for c = 1:3
    % Remove shadows and highlights, compute normals and albedo
        for i = 1:length(m)
            I_col = I(:, i, c);
            
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
            albedo(row, col, c) = albedo_val;
        end
    end
    % Re-render the image using recovered normals and albedo
    viewing_direction = [0, 0, 1];
    for c = 1:3
        for i = 1:height
            for j = 1:width
                normal = squeeze(N(i, j, :));
                if norm(normal) > 0
                    re_rendered_img(i, j, c) = max(0, dot(normal, viewing_direction)) * albedo(i, j, c);
                end
            end
        end
    end
    end
    