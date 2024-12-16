function image_processing_platform()
    % 创建主界面
    fig = uifigure('Name', '图像处理平台', 'Position', [100, 100, 1200, 800]);

    % 状态指示灯（左上角）
    statusLamp = uilamp(fig, 'Position', [10, 750, 20, 20], 'Color', 'red');
    entropyLabel = uilabel(fig, 'Text', '熵值: --', 'Position', [400, 740, 200, 30]);

    % 创建菜单栏
    menu = uimenu(fig, 'Text', '文件');
    uimenu(menu, 'Text', '打开', 'MenuSelectedFcn', @(src, event) openImage());
    uimenu(menu, 'Text', '保存', 'MenuSelectedFcn', @(src, event) saveImage());

    % 创建选项菜单
    optionsMenu = uimenu(fig, 'Text', '选项');
    uimenu(optionsMenu, 'Text', '重做', 'MenuSelectedFcn', @(src, event) resetImage());

    % 图像显示区（原图像、处理后的图像、RGB折线图并排）
    ax1 = axes(fig, 'Position', [0.05, 0.55, 0.25, 0.35]); % 原图像
    title(ax1, '原图像');

    ax2 = axes(fig, 'Position', [0.4, 0.55, 0.25, 0.35]); % 处理后的图像
    title(ax2, '处理后的图像');

    ax3 = axes(fig, 'Position', [0.75, 0.55, 0.25, 0.35]); % RGB折线图
    title(ax3, 'RGB频度');

  % 图像操作按钮区
btnPanel = uipanel(fig, 'Title', '图像处理操作', 'Position', [0.05, 0.05, 0.9, 0.4]);

% RGB调整控制面板
uilabel(fig, 'Text', '红色通道', 'Position', [880, 350, 80, 25]);
redSlider = uislider(fig, 'Position', [970, 360, 200, 3], 'Limits', [0, 2], 'Value', 1);
redSlider.ValueChangedFcn = @(src, event) adjustRGB();

uilabel(fig, 'Text', '绿色通道', 'Position', [880, 300, 80, 25]);
greenSlider = uislider(fig, 'Position', [970, 310, 200, 3], 'Limits', [0, 2], 'Value', 1);
greenSlider.ValueChangedFcn = @(src, event) adjustRGB();

uilabel(fig, 'Text', '蓝色通道', 'Position', [880, 250, 80, 25]);
blueSlider = uislider(fig, 'Position', [970, 260, 200, 3], 'Limits', [0, 2], 'Value', 1);
blueSlider.ValueChangedFcn = @(src, event) adjustRGB();

uilabel(fig, 'Text', '旋转角度', 'Position', [880, 200, 80, 25]);
rotateAngleEdit = uieditfield(fig, 'numeric', 'Position', [970, 200, 80, 25], 'Value', 0);
rotateAngleEdit.ValueChangedFcn = @(src, event) rotateImage();

 % 噪声添加控制区
    uilabel(fig, 'Text', '均值', 'Position', [180, 255, 80, 25]);%880, 60, 80, 25
    noiseMeanEdit = uieditfield(fig, 'numeric', 'Position', [210, 255, 30, 25], 'Value', 0);


 % 直方图按钮
    histogramButton = uibutton(fig, 'push', 'Text', '直方图', 'Position', [50, 350, 120, 40], ...
                               'ButtonPushedFcn', @(btn, event) showHistogram());
      % 增加灰度化和对比度增强按钮
    contrastEnhanceButton = uibutton(fig, 'push', 'Text', '灰度图', 'Position', [200, 350, 120, 40], ...
                                     'ButtonPushedFcn', @(btn, event) enhanceContrast());
      % 缩放变换按钮
    zoomButton = uibutton(fig, 'push', 'Text', '缩放变换', 'Position', [50, 300, 120, 40], ...
                          'ButtonPushedFcn', @(btn, event) zoomImage());
        % 噪声按钮
    gaussianNoiseButton = uibutton(fig, 'push', 'Text', '高斯噪声', 'Position', [50, 250, 120, 40], ...
                                   'ButtonPushedFcn', @(btn, event) addGaussianNoise());
    saltAndPepperButton = uibutton(fig, 'push', 'Text', '椒盐噪声', 'Position', [250, 250, 120, 40], ...
                                   'ButtonPushedFcn', @(btn, event) addSaltAndPepperNoise());
    % 目标提取按钮
extractButton = uibutton(fig, 'push', 'Text', '提取目标', 'Position', [50, 200, 120, 40], ...
                        'ButtonPushedFcn', @(btn, event) extractTarget());
  % LBP特征提取按钮
    lbpButton = uibutton(fig, 'push', 'Text', '提取原图LBP特征', 'Position', [50, 150, 120, 40], ...
                         'ButtonPushedFcn', @(btn, event) extractLBP('original'));

    % 提取目标LBP特征按钮
    lbpTargetButton = uibutton(fig, 'push', 'Text', '提取目标LBP特征', 'Position', [200, 150, 120, 40], ...
                               'ButtonPushedFcn', @(btn, event) extractLBP('target'));

    % HOG特征提取按钮
    hogButton = uibutton(fig, 'push', 'Text', '提取原图HOG特征', 'Position', [50, 100, 120, 40], ...
                         'ButtonPushedFcn', @(btn, event) extractHOG('original'));

    % 提取目标HOG特征按钮
    hogTargetButton = uibutton(fig, 'push', 'Text', '提取目标HOG特征', 'Position', [200, 100, 120, 40], ...
                               'ButtonPushedFcn', @(btn, event) extractHOG('target'));

      % 添加缩放滑块
    zoomSlider = uislider(fig, 'Position', [970, 160, 200, 3], 'Limits', [0.1, 5], 'Value', 1);
    zoomSliderLabel = uilabel(fig, 'Text', '缩放比例: 1.0', 'Position', [880, 150, 200, 30]);

    % 当滑块的值变化时更新显示的缩放比例
    zoomSlider.ValueChangedFcn = @(src, event) updateZoomValue(src, zoomSliderLabel);


% 创建下拉框按钮
 uilabel(fig, 'Text', '频域操作', 'Position', [500, 350, 80, 25]);
 dropdown = uidropdown(fig, ......
    'Position', [560, 350, 100, 30], ...
    'Items', {'无', '傅里叶变换', '离散余弦变换'}, ...
    'ValueChangedFcn', @(src, event) applyFrequencyDomainOperation(src.Value));

 % 平滑操作按钮
  uilabel(fig, 'Text', '平滑操作', 'Position', [500, 300, 80, 25]);
    dropdown = uidropdown(fig, 'Position', [560, 300, 100, 30], ...
        'Items', {'无', '均值平滑'}, ...
        'ValueChangedFcn', @(src, event) applySmoothingOperation(src.Value));

    % 锐化操作选择下拉框
    uilabel(fig, 'Text', '锐化操作', 'Position', [500, 250, 80, 25]);
    sharpenDropdown = uidropdown(fig, 'Position', [560, 250, 100, 30], ...
        'Items', {'无', 'Robert', 'Prewitt', 'Sobel', '拉普拉斯'}, ...
        'ValueChangedFcn', @(src, event) applySharpeningOperation(src.Value));

     uilabel(fig, 'Text', '滤波操作', 'Position', [500, 200, 80, 25]);
    dropdown = uidropdown(fig, 'Position', [560, 200, 100, 30], ...
        'Items', {'无', '低通滤波', '高通滤波', '带通滤波', '带阻滤波'}, ...
        'ValueChangedFcn', @(src, event) applyFilteringDomainOperation(src.Value));



% 调用 drawnow 强制更新图形
drawnow;

% 调试输出
disp(fig.Children);  % 查看窗口的所有控件
disp(btnPanel.Children);  % 查看面板内的控件
disp(btnPanel.Position);
disp(redSlider.Position);
disp(greenSlider.Position);
disp(blueSlider.Position);
disp(rotateAngleEdit.Position);

    % 初始化图像变量
    img = [];
    processedImg = [];

    % 打开图像函数
    function openImage()
        [file, path] = uigetfile({'*.jpg;*.png;*.bmp', '所有图像文件'}, '选择图像');
        if file
            img = imread(fullfile(path, file));
            % 显示原图像
            imshow(img, 'Parent', ax1);
            
            % 手动计算图像的熵值
            grayImg = rgb2grayCustom(img); % 转换为灰度图像
            entropyValue = calculateEntropy(grayImg); % 计算熵值
            entropyLabel.Text = ['熵值: ', num2str(entropyValue)];
            
            % 计算并显示RGB通道的灰度频度图
            plotRGBHistogram(img);
            
            % 更新状态指示灯为绿色表示成功加载
            statusLamp.Color = 'green';
        end
    end

    % 手动实现RGB到灰度图像的转换
    function grayImg = rgb2grayCustom(img)
        % 获取图像的尺寸
        [height, width, ~] = size(img);
        
        % 初始化灰度图像
        grayImg = zeros(height, width);
        
        % 遍历每个像素进行加权平均
        for i = 1:height
            for j = 1:width
                R = img(i, j, 1);
                G = img(i, j, 2);
                B = img(i, j, 3);
                % 使用加权公式进行转换
                grayImg(i, j) = 0.2989 * R + 0.5870 * G + 0.1140 * B;
            end
        end
        
        % 转换为uint8格式
        grayImg = uint8(grayImg);
    end

    % 手动计算图像的熵值
    function entropyValue = calculateEntropy(grayImg)
        % 计算灰度图像的直方图
        [counts, ~] = imhist(grayImg);
        
        % 归一化直方图，得到每个灰度级的概率
        totalPixels = numel(grayImg);
        probabilities = counts / totalPixels;
        
        % 移除概率为零的项
        probabilities = probabilities(probabilities > 0);
        
        % 使用香农熵公式计算熵值
        entropyValue = -sum(probabilities .* log2(probabilities));
    end

    % 绘制RGB通道的灰度频度图
    function plotRGBHistogram(image)
        % 分离RGB通道
        redChannel = image(:,:,1);
        greenChannel = image(:,:,2);
        blueChannel = image(:,:,3);
        
        % 计算每个通道的直方图
        [countsR, binsR] = imhist(redChannel);
        [countsG, binsG] = imhist(greenChannel);
        [countsB, binsB] = imhist(blueChannel);
        
        % 绘制折线图
        hold(ax3, 'off');
        plot(ax3, binsR, countsR, 'r', 'LineWidth', 2); hold(ax3, 'on');
        plot(ax3, binsG, countsG, 'g', 'LineWidth', 2);
        plot(ax3, binsB, countsB, 'b', 'LineWidth', 2);
        xlabel(ax3, '灰度级');
        ylabel(ax3, '像素数');
        legend(ax3, '红色通道', '绿色通道', '蓝色通道');
        hold(ax3, 'off');
    end

    % 保存图像函数
    function saveImage()
        if isempty(processedImg)
            msgbox('请先处理图像', '错误', 'error');
            return;
        end
        [file, path] = uiputfile({'*.jpg;*.png;*.bmp', '图像文件'}, '保存图像');
        if file
            imwrite(processedImg, fullfile(path, file));
            msgbox('图像已保存', '提示', 'help');
        end
    end

    % 重做函数
    function resetImage()
        if isempty(img)
            msgbox('没有加载图像', '错误', 'error');
            return;
        end
        % 恢复到原始图像
        processedImg = img;
        imshow(img, 'Parent', ax2);
        % 更新RGB三通道的灰度频度图
        plotRGBHistogram(img);
        statusLamp.Color = 'yellow'; % 状态指示灯设置为黄色，表示可以操作
    end

    % 调整RGB通道函数
    function adjustRGB()
        if isempty(img)
            return;
        end
        % 获取当前RGB通道的调整比例
        redFactor = redSlider.Value;
        greenFactor = greenSlider.Value;
        blueFactor = blueSlider.Value;
        
        % 调整RGB通道
        adjustedImg = img;
        adjustedImg(:,:,1) = uint8(double(img(:,:,1)) * redFactor);
        adjustedImg(:,:,2) = uint8(double(img(:,:,2)) * greenFactor);
        adjustedImg(:,:,3) = uint8(double(img(:,:,3)) * blueFactor);
        
        % 显示调整后的图像
        processedImg = adjustedImg;
        imshow(adjustedImg, 'Parent', ax2);
        plotRGBHistogram(adjustedImg);  % 更新RGB频度图
    end
 % 旋转图像函数
    function rotateImage()
        if isempty(img)
            return;
        end
        % 获取旋转角度
        angle = rotateAngleEdit.Value;
        
        % 使用imrotate函数旋转图像
        rotatedImg = imrotate(img, angle);
        
        % 显示旋转后的图像
        processedImg = rotatedImg;
        imshow(rotatedImg, 'Parent', ax2);
        plotRGBHistogram(rotatedImg);  % 更新RGB频度图
    end

    function showHistogram()
    % 弹出新窗口显示灰度直方图、均衡化后图像、匹配后图像
    histFig = uifigure('Name', '直方图处理', 'Position', [200, 200, 1000, 600]);

    % 创建三个子图区域，确保它们有足够的空间
    ax1 = axes(histFig, 'Position', [0.05, 0.3, 0.3, 0.6]); % 灰度直方图
    title(ax1, '灰度直方图', 'FontSize', 12);
    
    ax2 = axes(histFig, 'Position', [0.4, 0.3, 0.3, 0.6]); % 直方图均衡化
    title(ax2, '直方图均衡化', 'FontSize', 12);
    
    ax3 = axes(histFig, 'Position', [0.75, 0.3, 0.3, 0.6]); % 直方图匹配（规定化）
    title(ax3, '直方图匹配', 'FontSize', 12);

    % 处理图像
    if isempty(img)
        return;
    end
    
    % 转换为灰度图像
    grayImg = rgb2gray(img);
    
    % 绘制灰度直方图
    histogramData = computeHistogram(grayImg);
    bar(ax1, histogramData, 'BarWidth', 1);
    
    % 直方图均衡化
    eqImg = histogramEqualization(grayImg);
    imshow(eqImg, 'Parent', ax2);
    
    % 直方图匹配（规定化）
    %targetImg = zeros(size(grayImg), 'like', grayImg); % 可以设置目标图像
    targetImg = imread('C:\Users\10835\Desktop\beijing.jpg'); 
    if size(targetImg, 3) == 3
    targetImg = rgb2gray(targetImg);
    end
    matchedImg = histogramMatching(grayImg, targetImg);
    imshow(matchedImg, 'Parent', ax3);
end

    % 计算灰度直方图
    function histData = computeHistogram(grayImage)
        histData = zeros(1, 256);
        for i = 1:numel(grayImage)
            histData(grayImage(i) + 1) = histData(grayImage(i) + 1) + 1;
        end
    end
    
    % 直方图均衡化
    function eqImage = histogramEqualization(grayImage)
        % 计算累积分布函数 (CDF)
        histData = computeHistogram(grayImage);
        cdf = cumsum(histData) / numel(grayImage);
        % 归一化映射
        eqImage = uint8(255 * cdf(double(grayImage) + 1));
    end
    
    % 直方图匹配
    function matchedImage = histogramMatching(sourceImg, targetImg)
    % 计算源图像和目标图像的灰度直方图
    sourceHist = computeHistogram(sourceImg);
    targetHist = computeHistogram(targetImg);
    
    % 计算源图像和目标图像的累积分布函数 (CDF)
    sourceCDF = cumsum(sourceHist) / numel(sourceImg);
    targetCDF = cumsum(targetHist) / numel(targetImg);
    
    % 创建一个空的匹配图像
    matchedImage = zeros(size(sourceImg), 'like', sourceImg);
    
    % 直方图匹配过程
    for i = 1:numel(sourceImg)
        % 找到最接近的目标CDF值，并映射到目标像素值
        [~, idx] = min(abs(sourceCDF(sourceImg(i) + 1) - targetCDF));
        
        % 确保映射后的像素值位于有效范围内
        matchedImage(i) = idx - 1;  % -1 是因为索引从0开始
    end
    
    % 确保匹配后的图像值在[0, 255]范围内
    matchedImage = uint8(matchedImage);
    
    % 显示一些调试信息
    fprintf('源图像灰度最小值: %d, 最大值: %d\n', min(sourceImg(:)), max(sourceImg(:)));
    fprintf('目标图像灰度最小值: %d, 最大值: %d\n', min(targetImg(:)), max(targetImg(:)));
    fprintf('匹配图像灰度最小值: %d, 最大值: %d\n', min(matchedImage(:)), max(matchedImage(:)));
    end

% 对比度增强
function enhanceContrast()
    if isempty(img)
        return;
    end
    
    % 灰度化图像
    grayImg = rgb2grayCustom(img);

    % 创建一个弹出窗口，显示四张图像（2x2布局）
    figure('Name', '图像处理结果', 'Position', [200, 200, 1200, 800]);

    % 第一个子图：灰度图
    subplot(2, 2, 1);
    imshow(grayImg);
    title('灰度图');

    % 第二个子图：线性对比度增强
    linearEnhanced = imadjust(grayImg, [0.2, 0.8], [0, 1]);
    subplot(2, 2, 2);
    imshow(linearEnhanced);
    title('线性对比度增强');

    % 第三个子图：对数变换增强
    grayImgDouble = double(grayImg);  % 将灰度图像转换为 double 类型
    c = 255 / log(1 + max(grayImgDouble(:))); % 缩放常数
    logEnhanced = c * log(1 + grayImgDouble); % 对数变换
    subplot(2, 2, 3);
    imshow(uint8(logEnhanced));  % 显示图像时转换回 uint8 类型
    title('对数变换增强');

    % 第四个子图：指数变换增强
    gamma = 1.5;  % 可以调整gamma值来控制指数变换的强度
    expEnhanced = 255 * (grayImgDouble / 255) .^ gamma;  % 归一化到[0, 1]范围后进行指数变换
    subplot(2, 2, 4);
    imshow(uint8(expEnhanced));
    title('指数变换增强');
end

 % 更新显示的缩放比例
    function updateZoomValue(slider, label)
        zoomFactor = slider.Value;
        label.Text = ['缩放比例: ', num2str(zoomFactor, '%.1f')];
    end

    % 缩放变换函数
    function zoomImage()
    if isempty(img)
        return;
    end

    % 获取缩放比例
    scale = zoomSlider.Value;

    % 计算新的图像大小
    [rows, cols, channels] = size(img);
    newRows = round(rows * scale);
    newCols = round(cols * scale);

    % 输出新的图像尺寸，用于调试
    disp(['原始尺寸：', num2str(rows), 'x', num2str(cols)]);
    disp(['缩放后的尺寸：', num2str(newRows), 'x', num2str(newCols)]);

    % 创建一个新的空图像，确保数据类型与原图一致
    zoomedImg = zeros(newRows, newCols, channels, 'like', img); 

    % 使用最近邻插值进行缩放
    for r = 1:newRows
        for c = 1:newCols
            oldR = round(r / scale);
            oldC = round(c / scale);
            oldR = min(max(oldR, 1), rows);  % 防止超出范围
            oldC = min(max(oldC, 1), cols);  % 防止超出范围
            zoomedImg(r, c, :) = img(oldR, oldC, :);
        end
    end

    % 显示缩放后的图像
    processedImg = zoomedImg;
    imshow(processedImg, 'Parent', ax2);
end

% 高斯噪声添加函数
function addGaussianNoise()
    if ~isempty(img)
        meanNoise = noiseMeanEdit.Value;  % 获取用户输入的噪声均值
        sigma = 0.1;  % 设置标准差
        noisyImg = double(img) + meanNoise + sigma * randn(size(img));  % 转换为double进行加法操作
        noisyImg = uint8(noisyImg);  % 强制转换为uint8，确保图像像素为有效值范围
        
        % 防止像素值超出0-255范围
        noisyImg(noisyImg > 255) = 255;  
        noisyImg(noisyImg < 0) = 0;
        
        % 显示噪声图像
        imshow(noisyImg, 'Parent', ax2);
        processedImg = noisyImg;
        disp('添加高斯噪声');
    end
end

% 椒盐噪声添加函数
function addSaltAndPepperNoise()
    if ~isempty(img)
        meanNoise = noiseMeanEdit.Value;  % 获取用户输入的噪声均值
        noisyImg = img;
        p = 0.01;  % 控制椒盐噪声的概率（噪声强度），0.01 代表1%的像素
        numSalt = round(p * numel(img));  % 计算盐噪声的数量
        numPepper = numSalt;  % 胡椒噪声数量与盐噪声相同

        % 随机生成盐噪声（白色）
        saltIndices = randperm(numel(img), numSalt);
        noisyImg(saltIndices) = 255;

        % 随机生成胡椒噪声（黑色）
        pepperIndices = randperm(numel(img), numPepper);
        noisyImg(pepperIndices) = 0;

        % 显示带噪声图像
        imshow(noisyImg, 'Parent', ax2);
        processedImg = noisyImg;
        disp('添加椒盐噪声');
    end
end

% 平滑操作
function applySmoothingOperation(operation)
    if strcmp(operation, '均值平滑') && ~isempty(img)
        % 步骤 1: 直接应用均值平滑
        processedImg = mean_smooth_custom(img, 3);  % 使用3x3均值滤波器

        % 显示处理后的图像
        imshow(processedImg, 'Parent', ax2);

        % 计算并显示熵值
        entropyValue = entropy(rgb2gray(processedImg));  % 使用灰度图计算熵值
        entropyLabel.Text = ['熵值: ', num2str(entropyValue, '%.2f')];
    end
end

% 自定义均值平滑函数
function smoothedImg = mean_smooth_custom(inputImg, kernelSize)
    [rows, cols, channels] = size(inputImg);  % 获取图像大小
    padSize = floor(kernelSize / 2);
    paddedImg = padarray(inputImg, [padSize, padSize], 'replicate', 'both');  % 填充边缘
    smoothedImg = zeros(rows, cols, channels, 'uint8');  % 创建输出图像

    for c = 1:channels  % 对每个颜色通道进行处理
        for r = 1:rows
            for col = 1:cols
                % 取邻域像素
                neighborhood = paddedImg(r:r+kernelSize-1, col:col+kernelSize-1, c);
                smoothedImg(r, col, c) = mean(neighborhood(:));  % 计算均值
            end
        end
    end
end

 % 处理频域变换函数
    function applyFrequencyDomainOperation(option)
        if isempty(img)
            return;
        end
        switch option
            case '傅里叶变换'
                processedImg = applyFourierTransform(img);
            case '离散余弦变换'
                processedImg = applyDCT(img);
            otherwise
                processedImg = img; % 无操作
        end
        % 显示处理后的图像
        imshow(processedImg, 'Parent', ax2);
    end

    % 自定义傅里叶变换
    function transformedImg = applyFourierTransform(inputImg)
        % 将图像转换为灰度图像
        grayImg = rgb2grayCustom(inputImg);

        % 计算二维傅里叶变换
        F = fftshift(fft2(double(grayImg)));

        % 计算频谱幅值
        magnitude = log(1 + abs(F));

        % 显示频谱（反向变换得到的图像）
        transformedImg = uint8(abs(ifft2(ifftshift(F))));
    end

    % 自定义离散余弦变换
    function transformedImg = applyDCT(inputImg)
        % 将图像转换为灰度图像
        grayImg = rgb2grayCustom(inputImg);

        % 计算二维DCT
        transformedImg = dct2(grayImg);

        % 通过逆DCT恢复图像
        transformedImg = uint8(idct2(transformedImg));
    end

  % 锐化操作选择回调函数
    function applySharpeningOperation(operation)
        if isempty(img)
            return; % 如果图像为空，直接返回
        end

        switch operation
            case 'Robert'
                processedImg = robertSharpening(img);
            case 'Prewitt'
                processedImg = prewittSharpening(img);
            case 'Sobel'
                processedImg = sobelSharpening(img);
            case '拉普拉斯'
                processedImg = laplacianSharpening(img);
            otherwise
                processedImg = img; % 不做任何处理
        end

        % 显示处理后的图像
        imshow(processedImg, 'Parent', ax2);
    end

    % Robert算子锐化
    function out = robertSharpening(inputImg)
        % 将图像转换为灰度图像
        grayImg = rgb2gray(inputImg);
        % 计算Robert算子
        robertX = [1 0; 0 -1];
        robertY = [0 1; -1 0];
        gradX = imfilter(double(grayImg), robertX);
        gradY = imfilter(double(grayImg), robertY);
        out = uint8(sqrt(gradX.^2 + gradY.^2));
    end

    % Prewitt算子锐化
    function out = prewittSharpening(inputImg)
        grayImg = rgb2gray(inputImg);
        prewittX = [-1 0 1; -1 0 1; -1 0 1];
        prewittY = [-1 -1 -1; 0 0 0; 1 1 1];
        gradX = imfilter(double(grayImg), prewittX);
        gradY = imfilter(double(grayImg), prewittY);
        out = uint8(sqrt(gradX.^2 + gradY.^2));
    end

    % Sobel算子锐化
    function out = sobelSharpening(inputImg)
        grayImg = rgb2gray(inputImg);
        sobelX = [-1 0 1; -2 0 2; -1 0 1];
        sobelY = [-1 -2 -1; 0 0 0; 1 2 1];
        gradX = imfilter(double(grayImg), sobelX);
        gradY = imfilter(double(grayImg), sobelY);
        out = uint8(sqrt(gradX.^2 + gradY.^2));
    end

    % 拉普拉斯算子锐化
    function out = laplacianSharpening(inputImg)
        grayImg = rgb2gray(inputImg);
        laplacian = [0 1 0; 1 -4 1; 0 1 0];
        out = imfilter(double(grayImg), laplacian);
        out = uint8(out);
    end
   % 频域操作的回调函数
    function applyFilteringDomainOperation(opType)
        if isempty(img)
            return;
        end

        % 转换为灰度图像
        grayImg = rgb2grayCustom(img);

        % 将图像转换到频域
        f = fft2(double(grayImg));
        fshift = fftshift(f); % 频谱移至中心

        % 选择滤波器
        switch opType
            case '低通滤波'
                processedImg = lowPassFilter(fshift);
            case '高通滤波'
                processedImg = highPassFilter(fshift);
            case '带通滤波'
                processedImg = bandPassFilter(fshift);
            case '带阻滤波'
                processedImg = bandStopFilter(fshift);
            otherwise
                processedImg = grayImg;
        end

        % 将频域处理后的图像转换回空间域
        processedImg = real(ifft2(ifftshift(processedImg)));
        
        % 显示处理后的图像
        imshow(uint8(processedImg), 'Parent', ax2);
    end

    % 低通滤波器（简单的阈值）
    function result = lowPassFilter(fshift)
        [M, N] = size(fshift);
        % 创建低通滤波器的掩码
        cutoff = 30; % 截止频率
        [U, V] = meshgrid(1:N, 1:M);
        D = sqrt((U - N / 2).^2 + (V - M / 2).^2);
        H = double(D <= cutoff); % 低通滤波器
        result = fshift .* H;
    end

    % 高通滤波器
    function result = highPassFilter(fshift)
        [M, N] = size(fshift);
        % 创建高通滤波器的掩码
        cutoff = 30; % 截止频率
        [U, V] = meshgrid(1:N, 1:M);
        D = sqrt((U - N / 2).^2 + (V - M / 2).^2);
        H = double(D > cutoff); % 高通滤波器
        result = fshift .* H;
    end

    % 带通滤波器
    function result = bandPassFilter(fshift)
        [M, N] = size(fshift);
        % 创建带通滤波器的掩码
        low_cutoff = 10;
        high_cutoff = 50;
        [U, V] = meshgrid(1:N, 1:M);
        D = sqrt((U - N / 2).^2 + (V - M / 2).^2);
        H = double((D > low_cutoff) & (D < high_cutoff)); % 带通滤波器
        result = fshift .* H;
    end

    % 带阻滤波器
    function result = bandStopFilter(fshift)
        [M, N] = size(fshift);
        % 创建带阻滤波器的掩码
        low_cutoff = 10;
        high_cutoff = 50;
        [U, V] = meshgrid(1:N, 1:M);
        D = sqrt((U - N / 2).^2 + (V - M / 2).^2);
        H = double(~((D > low_cutoff) & (D < high_cutoff))); % 带阻滤波器
        result = fshift .* H;
    end

%提取目标
    function extractTarget()
    if isempty(img)
        uialert(fig, '请先打开一张图像！', '错误');
        return;
    end
    
    % 转换为灰度图像
    grayImg = rgb2grayCustom(img); 
    
    % 去噪声：应用高斯滤波（手动实现卷积）
    grayImg = gaussianFilter(grayImg, 2);  % 使用标准差为2的高斯滤波
    
    % 阈值化处理：Otsu方法手动计算阈值
    threshold = otsuThreshold(grayImg);    % 使用Otsu方法计算阈值
    binaryImg = grayImg > threshold;      % 使用计算出的阈值进行二值化
    
    % 边缘检测：手动实现Sobel算子
    edges = sobelEdgeDetection(binaryImg);   % 使用Sobel算子进行边缘检测
    
    % 连通组件分析：手动实现连通区域分析
    labeledImg = connectedComponents(edges); % 实现连通区域标记
    
    % 给不同区域上色
    rgbLabeledImg = label2rgb(labeledImg, 'hsv', 'k', 'shuffle'); % 给不同区域上色

       % 保存处理后的目标图像（用于后续LBP提取）
    processedImg = rgbLabeledImg;
    
    % 显示处理结果
    imshow(rgbLabeledImg, 'Parent', ax2); % 显示提取后的目标图像
    
    % 更新状态指示灯为绿色
    statusLamp.Color = 'green';


end


% 辅助函数：高斯滤波（卷积实现）
function filteredImg = gaussianFilter(img, sigma)
    % 创建高斯核
    size = 2 * ceil(3 * sigma) + 1;  % 高斯核的大小
    [X, Y] = meshgrid(-floor(size/2):floor(size/2), -floor(size/2):floor(size/2));
    gaussianKernel = exp(-(X.^2 + Y.^2) / (2 * sigma^2));
    gaussianKernel = gaussianKernel / sum(gaussianKernel(:));  % 归一化
    
    % 图像卷积
    filteredImg = conv2(img, gaussianKernel, 'same');
end

% 辅助函数：Otsu方法计算阈值
function threshold = otsuThreshold(img)
    % 计算图像的直方图
    histVals = histcounts(img(:), 0:256);
    prob = histVals / sum(histVals);  % 归一化为概率分布
    
    % 计算类间方差
    total = sum(prob);
    sumB = 0;
    weightB = 0;
    maximum = 0;
    sum1 = sum((0:255) .* prob);
    
    for t = 1:255
        weightB = weightB + prob(t);
        if weightB == 0
            continue;
        end
        
        weightF = total - weightB;
        if weightF == 0
            break;
        end
        
        sumB = sumB + (t - 1) * prob(t);
        meanB = sumB / weightB;
        meanF = (sum1 - sumB) / weightF;
        
        betweenClassVar = weightB * weightF * (meanB - meanF)^2;
        if betweenClassVar > maximum
            maximum = betweenClassVar;
            threshold = t - 1;
        end
    end
end

% 辅助函数：Sobel边缘检测
function edges = sobelEdgeDetection(img)
    % Sobel算子的水平和垂直滤波器
    sobelX = [-1, 0, 1; -2, 0, 2; -1, 0, 1];
    sobelY = [-1, -2, -1; 0, 0, 0; 1, 2, 1];
    
    % 图像的水平和垂直梯度
    gradX = conv2(img, sobelX, 'same');
    gradY = conv2(img, sobelY, 'same');
    
    % 计算梯度的幅值
    edges = sqrt(gradX.^2 + gradY.^2);
    
    % 二值化边缘图像
    edges = edges > 0.2 * max(edges(:));  % 设置一个阈值来确定边缘
end

% 辅助函数：连通区域分析（基于深度优先搜索DFS）
function labeledImg = connectedComponents(binaryImg)
    [rows, cols] = size(binaryImg);
    labeledImg = zeros(rows, cols);  % 初始化标签矩阵
    currentLabel = 1;  % 当前标签，从1开始
    
    % 创建一个访问标记矩阵
    visited = false(rows, cols);
    
    % 深度优先搜索（DFS）函数
    function dfs(r, c, label)
        stack = [r, c];  % 栈用于模拟递归
        while ~isempty(stack)
            [x, y] = deal(stack(end, 1), stack(end, 2));
            stack(end, :) = [];  % 弹出当前点
            
            if visited(x, y)
                continue;
            end
            
            visited(x, y) = true;
            labeledImg(x, y) = label;
            
            % 四个邻域
            for dx = -1:1
                for dy = -1:1
                    if abs(dx) + abs(dy) == 1  % 只考虑四个方向
                        nx = x + dx;
                        ny = y + dy;
                        if nx > 0 && nx <= rows && ny > 0 && ny <= cols && binaryImg(nx, ny) == 1 && ~visited(nx, ny)
                            stack = [stack; nx, ny];  % 入栈
                        end
                    end
                end
            end
        end
    end
    
    % 遍历图像的每一个像素
    for r = 1:rows
        for c = 1:cols
            if binaryImg(r, c) == 1 && ~visited(r, c)
                dfs(r, c, currentLabel);  % 从未访问的目标像素开始DFS
                currentLabel = currentLabel + 1;
            end
        end
    end
end

% 手动实现LBP特征提取
    function lbpFeatures = extractLBP(mode)
    % 检查是否打开了图像
    if isempty(img)
        uialert(fig, '请先打开一张图像！', '错误');
        return;
    end

    % 根据模式选择灰度图像
    if strcmp(mode, 'original')
        grayImg = rgb2grayCustom(img);  % 对原始图像进行灰度转换
    elseif strcmp(mode, 'target')
        if isempty(processedImg)
            uialert(fig, '请先提取目标！', '错误');
            return;
        end
        grayImg = rgb2grayCustom(processedImg); % 对提取的目标图像进行灰度转换
    else
        uialert(fig, '无效的模式！', '错误');
        return;
    end

    % 使用自定义的LBP算法，得到LBP特征图像
    lbpFeatures = customLBP(grayImg);

    % 确保lbpFeatures是二维图像数据
    if size(lbpFeatures, 3) > 1
        % 如果lbpFeatures是RGB图像或多通道图像，转换成灰度图
        lbpFeatures = rgb2gray(lbpFeatures);
    end

    % 在ax2上显示LBP特征图像
    axes(ax2);  % 激活ax2作为当前坐标轴
    imshow(lbpFeatures, [], 'Parent', ax2); % 显示图像到ax2
    title(ax2, 'LBP 提取');
    %colormap gray;  % 使用灰度色图
    %colorbar;       % 显示颜色条

    % 如果你只想查看前几个特征
    disp(lbpFeatures(1:min(5,end), 1:min(5,end)));  % 输出前5个元素
end

  % 手动实现HOG特征提取
    function hogFeatures = extractHOG(mode)
        if isempty(img)
            uialert(fig, '请先打开一张图像！', '错误');
            return;
        end

        if strcmp(mode, 'original')
            grayImg = rgb2grayCustom(img);  % 对原始图像进行灰度转换
        elseif strcmp(mode, 'target')
            if isempty(processedImg)
                uialert(fig, '请先提取目标！', '错误');
                return;
            end
            grayImg = rgb2grayCustom(processedImg); % 对提取的目标图像进行灰度转换
        end

        % 使用自定义的HOG算法
        hogFeatures = customHOG(grayImg, ax2);  % 传入ax2作为显示HOG图像的目标
        disp('HOG特征：');
        disp(hogFeatures);
    end


% 自定义LBP算法（计算LBP特征）
function lbp = customLBP(I)
    [rows, cols] = size(I);
    lbp = zeros(rows-2, cols-2);  % 初始化LBP特征矩阵

    for i = 2:rows-1
        for j = 2:cols-1
            % 获取3x3邻域
            neighborhood = I(i-1:i+1, j-1:j+1);  
            center = neighborhood(2, 2);  % 中心像素值

            % 比较邻域像素与中心像素，生成二进制模式
            binaryPattern = neighborhood > center;  % 大于中心的为1，否则为0

            % 将二进制模式转化为十进制数
            lbpValue = 0;
            % 遍历8个邻域像素并计算二进制模式的十进制值
            for k = 1:8
                lbpValue = lbpValue + binaryPattern(k) * 2^(k-1);
            end

            % 存储LBP值
            lbp(i-1, j-1) = lbpValue;  % 存储LBP值
        end
    end
end

 % 自定义HOG算法（计算HOG特征）
    function hog = customHOG(I, ax2)
        % 转换为灰度图（如果不是灰度图）
        if size(I, 3) == 3
            I = rgb2gray(I);
        end

        % 计算图像梯度
        [Gx, Gy] = gradient(double(I));  % 计算X和Y方向的梯度
        magnitude = sqrt(Gx.^2 + Gy.^2);  % 梯度幅值
        angle = atan2d(Gy, Gx);  % 梯度方向（角度）

        % 将梯度方向量化为9个方向（0-180度分成9个区间）
        binEdges = -90:20:90;
        [counts, ~] = histcounts(angle(:), binEdges);

        % 归一化梯度幅值
        hog = counts / sum(counts);  % 归一化HOG特征

        % 可视化梯度幅值图像，在ax2区域显示
        imshow(magnitude, [], 'Parent', ax2);  % 显示梯度幅值图像
        title(ax2, 'HOG Magnitude');
    end



end


