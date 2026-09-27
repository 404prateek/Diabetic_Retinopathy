function [classification, confidence] = classifyDR(I)
% =========================================================================
% classifyDR  -  Classify diabetic retinopathy grade in a fundus image
% =========================================================================

    % 1. Load the trained network from models/classification/
    persistent net;
    if isempty(net)
        modelPath = fullfile('models', 'classification', 'drClassifier.mat');
        if ~exist(modelPath, 'file')
            error('Trained model file not found at %s. Ensure trainClassifier() has run.', modelPath);
        end
        loadedData = load(modelPath, 'net');
        net = loadedData.net;
    end

    % 2. Preprocess and resize input image to 224x224 (EfficientNet-B0 requirement)
    imgResized = imresize(I, [224, 224]);

    % 3. Run inference to get class prediction and softmax probabilities
    [predLabel, scores] = classify(net, imgResized);

    % 4. Extract highest softmax probability confidence score
    confidence = double(max(scores));

    % 5. Map predicted output to required ICDR string description
    icdrLabels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"];
    
    % Convert prediction label (e.g. '0', '1', '2', '3', '4') to numerical index
    gradeNum = str2double(string(predLabel));

    if isnan(gradeNum)
        % If folder names were already named as text descriptions
        classification = string(predLabel);
    else
        % Map grade index 0..4 to array positions 1..5
        idx = gradeNum + 1;
        if idx >= 1 && idx <= 5
            classification = icdrLabels(idx);
        else
            classification = string(predLabel);
        end
    end

end