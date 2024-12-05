clc;
close all;
clear all;

dataFormat = 'PNG'; 

%==========01=========%
dataNameStack{1} = 'bear';
%==========02=========%
dataNameStack{2} = 'cat';
%==========03=========%
dataNameStack{3} = 'pot';
%==========04=========%
dataNameStack{4} = 'buddha';

for testId = 1 : 4
    dataName = [dataNameStack{testId}, dataFormat];
    datadir = ['..\pmsData\', dataName];
    bitdepth = 16;
    gamma = 1;
    resize = 1;  
    data = load_datadir_re(datadir, bitdepth, resize, gamma); 

    L = data.s;
    f = size(L, 1);
    [height, width, color] = size(data.mask);
    if color == 1
        mask1 = double(data.mask./255);
    else
        mask1 = double(rgb2gray(data.mask)./255);
    end
    mask3 = repmat(mask1, [1, 1, 3]);
    m = find(mask1 == 1);
    p = length(m);

    % %% Standard photometric stereo
    % Normal = myPMS(data, m);
    % 
    % %% Save results "png"
    % imwrite(uint8((Normal+1)*128).*uint8(mask3), strcat(dataName, '_Normal.png'));
    % 
    % %% Save results "mat"
    % save(strcat(dataName, '_Normal.mat'), 'Normal');
    % discard_percentage 是丢弃最暗和最亮的 10% 强度值
    discard_percentage = 10;  % 丢弃最暗和最亮的 10%

    % 调用 L2_PMS_with_shadows 函数来计算法向量和反射率
    [N, rho] = myPMS(data, m, discard_percentage);

    % 保存法向量和反射率结果
    imwrite(uint8((N + 1) * 128), strcat(dataName, '_Normal.png'));  % 保存法向量为图像
    save(strcat(dataName, '_Normal.mat'), 'N');  % 保存法向量为 .mat 文件
    imwrite(uint8(rho * 255), strcat(dataName, '_Albedo.png'));  % 保存反射率为图像
    save(strcat(dataName, '_Albedo.mat'), 'rho');  % 保存反射率为 .mat 文件
end
