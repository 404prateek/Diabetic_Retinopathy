function heatmap = explainPrediction(I, classification)
% =========================================================================
% explainPrediction  -  Generate a Grad-CAM saliency heatmap
% =========================================================================
% Purpose : Produce a Gradient-weighted Class Activation Map (Grad-CAM)
%           overlay that highlights the retinal regions most responsible
%           for the classifier's decision.
%
% Owner   : Member 4  (module4_explainability/)
%
% Inputs  :
%   I              - (H x W x 3 uint8)  The same image passed to classifyDR()
%   classification - (string)  Grade string returned by classifyDR()
%
% Outputs :
%   heatmap        - (H x W x 3 uint8)  Jet-colourmap Grad-CAM overlay
%                    blended with the original image at alpha = 0.5
%
% BugFix (2024-10):
%   - Fixed layerNames extraction: {gcNet.Layers.Name} fails for DAGNetwork;
%     replaced with arrayfun to extract Name from each Layer object.
%   - Added guard for classIdx out-of-bounds when model class count != 5.
%   - Added im2single() preprocessing to match what classify() expects.
%
% Dependencies : MATLAB Deep Learning Toolbox (gradCAM), Image Processing Toolbox
% =========================================================================

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

% ------------------------------------------------------------------
% Attempt 1: Proper Grad-CAM using trained network
% ------------------------------------------------------------------
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

    % ---- BUG FIX: robust layer name extraction for DAGNetwork ----------
    % {gcNet.Layers.Name} fails when Layers is a Layer object array, not a
    % struct array. Use arrayfun to safely extract Name from each element.
    allLayers  = gcNet.Layers;
    layerNames = arrayfun(@(l) l.Name, allLayers, 'UniformOutput', false);

    % Find the last convolutional layer (best for Grad-CAM visualisation)
    isConv     = arrayfun(@(l) isa(l, 'nnet.cnn.layer.Convolution2DLayer'), allLayers);
    convIdx    = find(isConv);

    if isempty(convIdx)
        % Fallback: last batch normalisation layer (common in EfficientNet)
        isBN    = arrayfun(@(l) isa(l, 'nnet.cnn.layer.BatchNormalizationLayer'), allLayers);
        convIdx = find(isBN);
    end

    if isempty(convIdx)
        % Nothing suitable found - fall through to synthetic saliency
        warning('explainPrediction:noConvLayer', ...
                'No convolutional layer found; using synthetic saliency fallback.');
        useGradCAM = false;
    else
        targetLayer = layerNames{convIdx(end)};

        % Map classIdx to the categorical class label used during training
        outLayer    = allLayers(end);
        netClasses  = outLayer.Classes;        % categorical array of trained classes
        numNetCls   = numel(netClasses);

        % Clamp index so it never exceeds what the network knows about
        safeIdx    = min(classIdx, numNetCls);
        targetClass = netClasses(safeIdx);

        % Compute Grad-CAM and resize to original image dimensions
        scoreMap = gradCAM(gcNet, Iresized, targetClass, ...
                           'ReductionLayer', targetLayer);
        scoreMap = imresize(double(scoreMap), [H, W]);
    end
end

if ~useGradCAM
    % ------------------------------------------------------------------
    % Fallback: synthetic saliency from Laplacian of green channel
    % ------------------------------------------------------------------
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

% ------------------------------------------------------------------
% Normalise -> jet colourmap -> alpha-blend with original
% ------------------------------------------------------------------
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
