function [annotatedImg, detectedBoxes, detectedScores, detectedLabels] = detectLesionsYOLO(I, onnxPath, confThresh, iouThresh)
% Run YOLOv8 ONNX inference on fundus image and overlay detected lesions

if nargin < 2 || isempty(onnxPath)
    onnxPath = fullfile('models', 'detection', 'best.onnx');
end
if nargin < 3 || isempty(confThresh)
    confThresh = 0.25;
end
if nargin < 4 || isempty(iouThresh)
    iouThresh = 0.45;
end

classNames  = ["Microaneurysm", "Hemorrhage", "Hard Exudate", "Soft Exudate"];
classColors = {'yellow', 'red', 'cyan', 'magenta'};

I_display = imresize(I, [512 512]);

detectedBoxes  = zeros(0, 4);
detectedScores = zeros(0, 1);
detectedLabels = strings(0, 1);
annotatedImg   = I_display;

if ~exist(onnxPath, 'file')
    annotatedImg = insertText(I_display, [20, 20], ...
        'YOLOv8 model not found', ...
        'FontSize', 14, 'BoxColor', 'black', 'TextColor', 'yellow');
    return;
end

try
    persistent cachedYoloNet cachedOnnxPath;
    if isempty(cachedYoloNet) || ~strcmp(cachedOnnxPath, onnxPath)
        if exist('importNetworkFromONNX', 'file')
            cachedYoloNet = importNetworkFromONNX(onnxPath);
        elseif exist('importONNXNetwork', 'file')
            cachedYoloNet = importONNXNetwork(onnxPath);
        else
            error('Deep Learning Toolbox ONNX import function not found.');
        end
        cachedOnnxPath = onnxPath;
    end
    yoloNet = cachedYoloNet;
    
    I_resized = im2single(imresize(I, [512 512]));
    inputData = permute(I_resized, [3 2 1]);
    inputData = reshape(inputData, [3 512 512 1]);
    inputData = permute(inputData, [2 1 3 4]);
    
    dlInput = dlarray(inputData, 'SSCB');
    rawOut  = predict(yoloNet, dlInput);
    outData = extractdata(rawOut);
    
    outData = squeeze(outData);
    if size(outData, 1) > size(outData, 2)
        outData = outData';
    end
    
    boxes_norm = outData(1:4, :);
    scores_all = outData(5:end, :);
    
    [maxScores, classIndices] = max(scores_all, [], 1);
    validIdx = find(maxScores >= confThresh);
    
    if ~isempty(validIdx)
        cx = boxes_norm(1, validIdx);
        cy = boxes_norm(2, validIdx);
        bw = boxes_norm(3, validIdx);
        bh = boxes_norm(4, validIdx);
        
        xmin = cx - bw / 2;
        ymin = cy - bh / 2;
        
        % Scale to 512x512 if normalized
        if max(cx) <= 1.05
            xmin = xmin * 512;
            ymin = ymin * 512;
            bw   = bw   * 512;
            bh   = bh   * 512;
        end
        
        candidateBoxes  = [xmin', ymin', bw', bh'];
        candidateScores = maxScores(validIdx)';
        candidateLabels = classNames(classIndices(validIdx))';
        
        if exist('bboxnms', 'file')
            [selectedIdx, ~] = bboxnms(candidateBoxes, candidateScores, iouThresh);
        else
            [~, sortOrder] = sort(candidateScores, 'descend');
            selectedIdx = sortOrder(1:min(15, numel(sortOrder)));
        end
        
        detectedBoxes  = candidateBoxes(selectedIdx, :);
        detectedScores = candidateScores(selectedIdx);
        detectedLabels = candidateLabels(selectedIdx);
        
        annotText = cell(size(detectedBoxes, 1), 1);
        boxColors = cell(size(detectedBoxes, 1), 1);
        for k = 1:size(detectedBoxes, 1)
            annotText{k} = sprintf('%s %.2f', detectedLabels(k), detectedScores(k));
            cIdx = find(classNames == detectedLabels(k), 1);
            if isempty(cIdx), cIdx = 1; end
            boxColors{k} = classColors{cIdx};
        end
        
        if ~isempty(detectedBoxes)
            annotatedImg = insertObjectAnnotation(I_display, 'rectangle', ...
                detectedBoxes, annotText, 'Color', boxColors, 'LineWidth', 2, 'FontSize', 11);
        end
    end
    
catch ME
    warning('YOLO inference failed: %s', ME.message);
end

end
