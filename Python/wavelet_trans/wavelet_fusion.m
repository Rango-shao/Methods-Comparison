% 图像融合主函数
function main_image_fusion()
    % ============== 用户参数设置 ==============
    irPath = 'images/IR/IR18.jpg';     % 红外图像路径
    visPath = 'images/VIS/VIS18.jpg';   % 可见光图像路径
    outputPath = 'results/fused_result-189*8/.jpg'; % 融合结果保存路径
    % ========================================
    
    % 读取图像
    irImg = imread(irPath);
    visImg = imread(visPath);
    
    % 统一图像尺寸
    if ~isequal(size(irImg), size(visImg))
        visImg = imresize(visImg, size(irImg(:,:,1)));
        fprintf('图像尺寸已统一\n');
    end
    
    % 转换为灰度图并归一化
    if ndims(irImg) == 3
        irImg = rgb2gray(irImg);
    end
    if ndims(visImg) == 3
        visImg = rgb2gray(visImg);
    end
    irImg = im2double(irImg);
    visImg = im2double(visImg);
    
    % 小波参数设置
    zt = 2;          % 分解层数
    wtype = 'haar';  % 小波基类型
    
    % 小波分解与融合
    [c0, s0] = Wave_Decompose(irImg, zt, wtype);
    [c1, s1] = Wave_Decompose(visImg, zt, wtype);
    Coef_Fusion = Fuse_Process(c0, c1, s0, s1);
    fusedImg = Wave_Reconstruct(Coef_Fusion, s0, wtype);
    
    % 保存结果
    imwrite(fusedImg, outputPath);
    fprintf('融合结果已保存至: %s\n', outputPath);
end