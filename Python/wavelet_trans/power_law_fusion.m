irPath = 'images/IR/a.tif';
visPath = 'images/VIS/b.tif';
outputPath = 'results/fused_result.tif';


F1 = imread(visPath); 
F2 = imread(irPath); 

F1 = im2double(F1);
F2 = im2double(F2);


n = 8;

Z = F1 .^ (1 - F2 / 2^n);

Z_normalized = mat2gray(Z);

% 显示融合后的图像
figure;
imshow(Z_normalized, []);
title('Fused Image using Power Law Transformation');