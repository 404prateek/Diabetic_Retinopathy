function [classification, confidence] = classifyDR(I)
% Classify diabetic retinopathy grade in a fundus image using EfficientNet-B0.
% Returns ICDR grade string and softmax confidence.

% 1. Load the trained network (cached across calls via persistent)
persistent net;
if isempty(net)
    modelPath = fullfile('models', 'classification', 'drClassifier.mat');
    if ~exist(modelPath, 'file')
        error('classifyDR:ModelNotFound', ...
              ['Trained model not found at: %s\n' ...
               'Run trainClassifier(imdsTrain, imdsVal) first.'], modelPath);
    end
    loadedData = load(modelPath, 'net');
    net        = loadedData.net;
    fprintf('[classifyDR] Loaded classifier from %s\n', modelPath);
end

% 2. Preprocess: resize to 224x224, convert to single [0,1]
%    (EfficientNet-B0 expects single-precision input normalised to [0,1])
imgResized = imresize(I, [224, 224]);
imgSingle  = im2single(imgResized);   % uint8 -> single, divides by 255

% 3. Run inference
[predLabel, scores] = classify(net, imgSingle);

% scores is (1 x numClasses) row vector of softmax probabilities
confidence = double(max(scores(:)));

% 4. Map network output label to ICDR string
%    Training data sub-folders are named '0'..'4' (APTOS convention)
icdrLabels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"];
gradeNum   = str2double(string(predLabel));

if ~isnan(gradeNum) && gradeNum >= 0 && gradeNum <= 4
    % Label came from folder-name numeric convention (0..4)
    classification = icdrLabels(gradeNum + 1);
elseif ismember(string(predLabel), icdrLabels)
    % Label already is an ICDR text string
    classification = string(predLabel);
else
    % Unknown label - pass through as-is
    classification = string(predLabel);
    warning('classifyDR:UnknownLabel', ...
            'Predicted label "%s" not in standard ICDR set.', predLabel);
end

end  % classifyDR
