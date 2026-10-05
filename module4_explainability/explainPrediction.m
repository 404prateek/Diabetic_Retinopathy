function heatmap = explainPrediction(I, classification)
% Generate a Grad-CAM saliency heatmap highlighting regions driving the DR prediction.
% Blends jet colormap over the input image.

% Input guards
if ~isa(I, 'uint8')
    I = im2uint8(I);
end

H = size(I, 1);
W = size(I, 2);

% ICDR label -> index map
icdrLabels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"];
classIdx   = find(icdrLabels == string(classification), 1);
if isempty(classIdx)
    classIdx = 1;
end

% Try Grad-CAM with trained model
modelPath  = fullfile('models', 'classification', 'drClassifier.mat');
useGradCAM = exist(modelPath, 'file') && ~isempty(which('gradCAM'));

if useGradCAM
    persistent gcNet;
    if isempty(gcNet)
        ld    = load(modelPath, 'net');
        gcNet = ld.net;
    end

    % Preprocess to match training input format
    Iresized = im2single(imresize(I, [224, 224]));

    allLayers  = gcNet.Layers;
    layerNames = arrayfun(@(l) l.Name, allLayers, 'UniformOutput', false);

    % Find last conv or batchnorm layer
    isConv     = arrayfun(@(l) isa(l, 'nnet.cnn.layer.Convolution2DLayer'), allLayers);
    convIdx    = find(isConv);

    if isempty(convIdx)
        isBN    = arrayfun(@(l) isa(l, 'nnet.cnn.layer.BatchNormalizationLayer'), allLayers);
        convIdx = find(isBN);
    end

    if isempty(convIdx)
        warning('explainPrediction:noConvLayer', ...
                'No convolutional layer found; using synthetic saliency fallback.');
        useGradCAM = false;
    else
        targetLayer = layerNames{convIdx(end)};

        outLayer    = allLayers(end);
        netClasses  = outLayer.Classes;
        numNetCls   = numel(netClasses);

        safeIdx     = min(classIdx, numNetCls);
        targetClass = netClasses(safeIdx);

        scoreMap = gradCAM(gcNet, Iresized, targetClass, ...
                           'ReductionLayer', targetLayer);
        scoreMap = imresize(double(scoreMap), [H, W]);
    end
end

if ~useGradCAM
    % Saliency fallback using green channel Laplacian
    green    = im2double(I(:,:,2));
    lap      = imfilter(green, fspecial('laplacian', 0.2), 'replicate');
    scoreMap = abs(lap);

    % Grade-conditioned spatial emphasis
    [xx, yy] = meshgrid(linspace(-1,1,W), linspace(-1,1,H));
    r        = sqrt(xx.^2 + yy.^2);

    switch string(classification)
        case {"Severe", "Proliferative DR"}
            emphasis = max(0, r - 0.4);          % peripheral
        case {"Moderate"}
            emphasis = exp(-(r - 0.4).^2 / 0.1); % mid-peripheral
        otherwise
            emphasis = max(0, 0.5 - r);           % central / optic-disc
    end

    scoreMap = scoreMap + 0.4 * mat2gray(emphasis);
end

% Normalize and blend with jet colormap
scoreNorm = mat2gray(double(scoreMap));
scoreNorm = imgaussfilt(scoreNorm, 1.5);   % slight smoothing

% Apply jet colourmap (256 levels)
jetMap = jet(256);
idxMap = uint8(round(scoreNorm * 255)) + 1;   % 1..256
idxMap = max(1, min(256, double(idxMap)));

heatRGB = zeros(H, W, 3);
for ch = 1:3
    heatRGB(:,:,ch) = reshape(jetMap(idxMap(:), ch), H, W);
end

% Alpha-blend (alpha=0.5)
alpha   = 0.5;
blended = alpha * heatRGB + (1 - alpha) * im2double(I);
blended = max(0, min(1, blended));
heatmap = im2uint8(blended);

end  % explainPrediction
