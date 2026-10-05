% DR Screening Pipeline Driver Script

clc; clear; close all;

DATA_PATH = 'data/APTOS';

% Load dataset
fprintf('[1/3] Loading dataset from: %s\n', DATA_PATH);
imds = loadDataset(DATA_PATH);

if isempty(imds.Files)
    error('No images found in "%s". Add dataset files first (see README).', DATA_PATH);
end

% Pick test image
testImagePath = imds.Files{1};
fprintf('[2/3] Test image: %s\n', testImagePath);
I = imread(testImagePath);

% Run pipeline
fprintf('[3/3] Running mainPipeline...\n');
result = mainPipeline(I);

% Display results
fprintf('\n--- Pipeline Result ---\n');
fprintf('Status         : %s\n', result.status);
fprintf('Quality Status : %s\n', result.quality.status);

if result.status == "COMPLETE"
    fprintf('DR Grade       : %s (confidence: %.1f%%)\n', ...
            result.classification, result.confidence * 100);

    figure('Name', 'DR Screening Result', 'NumberTitle', 'off');

    subplot(1, 3, 1);
    imshow(result.image);
    title('Enhanced Fundus Image');

    subplot(1, 3, 2);
    imshow(result.segmentation.lesionMask);
    title(sprintf('Lesion Mask  (%d regions)', result.segmentation.lesionCount));

    subplot(1, 3, 3);
    imshow(result.heatmap);
    title(sprintf('Grad-CAM: %s', result.classification));

    sgtitle('Explainable DR Screening  |  PS ID 26038');
else
    fprintf('Image REJECTED by quality check. No further processing.\n');
    figure;
    imshow(I);
    title('REJECTED Image');
end
