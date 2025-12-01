clear; clc;
% ============== 用户参数设置 ==============
irPath = 'images/IR/a.tif';     % 红外图像路径
visPath = 'images/VIS/b.tif';   % 可见光图像路径
outputPath = 'results/fused_result.tif'; % 融合结果保存路径
% ========================================
    
image1 = imread(irPath);
image2 = imread(visPath); 

im1 = double(image1);
im2 = double(image2);

mean_image1 = mean(im1(:));
mean_image2 = mean(im2(:));

% Calculation of weights
alpha = 0.5;
beta = 0.5;  

% Perform exponential fusion
fused_image = (alpha * im1 ./ (im1 + im2) + beta * im2 ./ (im1 + im2)) .* (im1 + im2);

% 将融合后的图像归一化到 [0, 255] 范围
fused_image = mat2gray(fused_image);

% 显示融合前后的图像
figure;
subplot(1, 3, 1);
imshow(image1, []);
title('图像1');

subplot(1, 3, 2);
imshow(image2, []);
title('图像2');

subplot(1, 3, 3);
imshow(fused_image, []);
title('融合后的图像');