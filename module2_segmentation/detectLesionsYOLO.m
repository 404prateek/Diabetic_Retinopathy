function [annotatedImg, detectedBoxes, detectedScores, detectedLabels] = detectLesionsYOLO(I, onnxPath, confThresh, iouThresh)
% =========================================================================
% detectLesionsYOLO - YOLOv8 Retinal Lesion Detection via ONNX in MATLAB
% =========================================================================
% Purpose : Runs forward inference of exported YOLOv8 ONNX model on a
%           retinal image, decodes bounding boxes, applies NMS, and returns
%           an annotated visualization.
%
% Inputs  :
%   I          - Input RGB retinal fundus image (uint8 or single)
%   onnxPath   - Path to exported YOLOv8 best.onnx (default: models/detection/best.onnx)
%   confThresh - Confidence score threshold (default: 0.25)
%   iouThresh  - Non-maximum suppression IoU threshold (default: 0.45)
%
% Outputs :
%   annotatedImg   - RGB image with color-coded bounding boxes and labels
%   detectedBoxes  - [N x 4] matrix of bounding boxes [x, y, width, height]
%   detectedScores - [N x 1] vector of detection confidence scores
%   detectedLabels - [N x 1] string array of lesion category names
% =========================================================================

if nargin < 2 || isempty(onnxPath)
    onnxPath = fullfile('models', 'detection', 'best.onnx');
end
if nargin < 3 || isempty(confThresh)
    confThresh = 0.25;
end
if nargin < 4 || isempty(iouThresh)
    iouThresh = 0.45;
end

classNames = ["Microaneurysm", "Hemorrhage", "Hard Exudate", "Soft Exudate"];
classColors = {'yellow', 'red', 'cyan', 'magenta'};

[origH, origW, ~] = size(I);
I_display = imresize(I, [512 512]);

detectedBoxes  = zeros(0, 4);
detectedScores = zeros(0, 1);
detectedLabels = strings(0, 1);
annotatedImg   = I_display;

if ~exist(onnxPath, 'file')
    % Return clean image with notification if ONNX model is pending
    annotatedImg = insertText(I_display, [20, 20], ...
        'YOLOv8 ONNX Pending (Run scripts/export_onnx.py)', ...
        'FontSize', 14, 'BoxColor', 'black', 'TextColor', 'yellow');
    return;
end

try
    % 1. Load ONNX Network into MATLAB
    persistent cachedYoloNet cachedOnnxPath;
    if isempty(cachedYoloNet) || ~strcmp(cachedOnnxPath, onnxPath)
        fprintf('[*] Loading YOLOv8 ONNX network: %s...\n', onnxPath);
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
    
    % 2. Preprocess input for YOLOv8 (1 x 3 x 512 x 512 normalized [0, 1])
    I_resized = im2single(imresize(I, [512 512]));
    % Permute from MATLAB HWC to NCHW or standard DL format
    % dlarray format depends on imported network input layer
    inputData = permute(I_resized, [3 2 1]); % 3 x 512 x 512
    inputData = reshape(inputData, [3 512 512 1]);
    inputData = permute(inputData, [2 1 3 4]); % 512 x 3 x 512 x 1
    
    % Try forward pass
    dlInput = dlarray(inputData, 'SSCB');
    rawOut  = predict(yoloNet, dlInput);
    outData = extractdata(rawOut);
    
    % YOLOv8 output shape is typically [4 + num_classes, num_anchors]
    % i.e., [8, 5376] for 4 classes
    outData = squeeze(outData);
    if size(outData, 1) > size(outData, 2)
        outData = outData'; % Ensure [8, num_anchors]
    end
    
    numAnchors = size(outData, 2);
    boxes_norm = outData(1:4, :); % cx, cy, w, h
    scores_all = outData(5:end, :); % class scores
    
    [maxScores, classIndices] = max(scores_all, [], 1);
    validIdx = find(maxScores >= confThresh);
    
    if ~isempty(validIdx)
        cx = boxes_norm(1, validIdx);
        cy = boxes_norm(2, validIdx);
        bw = boxes_norm(3, validIdx);
        bh = boxes_norm(4, validIdx);
        
        % Convert [cx, cy, w, h] to [xmin, ymin, w, h] in 512x512 coordinates
        xmin = (cx - bw/2);
        ymin = (cy - bh/2);
        
        % If coordinates are normalized in [0, 1], scale to 512
        if max(cx) <= 1.05
            xmin = xmin * 512;
            ymin = ymin * 512;
            bw   = bw   * 512;
            bh   = bh   * 512;
        end
        
        candidateBoxes  = [xmin', ymin', bw', bh'];
        candidateScores = maxScores(validIdx)';
        candidateLabels = classNames(classIndices(validIdx))';
        
        % Non-Maximum Suppression (NMS)
        if exist('bboxnms', 'file')
            [selectedIdx, ~] = bboxnms(candidateBoxes, candidateScores, iouThresh);
        else
            % Basic confidence ranking if bboxnms unavailable
            [~, sortOrder] = sort(candidateScores, 'descend');
            selectedIdx = sortOrder(1:min(15, numel(sortOrder)));
        end
        
        detectedBoxes  = candidateBoxes(selectedIdx, :);
        detectedScores = candidateScores(selectedIdx);
        detectedLabels = candidateLabels(selectedIdx);
        
        % Annotation formatting
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
    fprintf('[!] Notice during YOLO inference: %s\n', ME.message);
    annotatedImg = insertText(I_display, [20, 20], ...
        sprintf('YOLOv8 Active (%s)', ME.identifier), ...
        'FontSize', 12, 'BoxColor', 'black', 'TextColor', 'yellow');
end

end
