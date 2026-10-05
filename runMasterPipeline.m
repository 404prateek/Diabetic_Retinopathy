% =========================================================================
% runMasterPipeline.m - End-to-End Diabetic Retinopathy Dashboard
% =========================================================================
% Purpose : Standalone quick-run script that directly invokes the U-Net
%           segmentation (via predict) and EfficientNet-B0 classification
%           (via classify) and displays a two-panel dashboard.
%
% Usage   : Run from repo root >> runMasterPipeline
%
% Note    : This script requires trained models:
%             models/segmentation/vessel_unet.mat   (trainVessels_Corrected.m)
%             models/classification/drClassifier.mat (runTraining.m)
%
% Dependencies : MATLAB Deep Learning Toolbox, Image Processing Toolbox
% =========================================================================
clear; clc;

% 1. Auto-Detect an Image to Test
testFolder = fullfile('data', 'APTOS', 'test_images');
imageFiles = dir(fullfile(testFolder, '*.png'));

if isempty(imageFiles)
    % Fallback: try train_images
    imageFiles = dir(fullfile('data', 'APTOS', 'train_images', '*.png'));
    if isempty(imageFiles)
        error('No PNG images found. Ensure data/APTOS/test_images/ or train_images/ contains images.');
    end
    testFolder = fullfile('data', 'APTOS', 'train_images');
end

testImagePath = fullfile(testFolder, imageFiles(1).name);
fprintf('Testing on image: %s\n', imageFiles(1).name);
I = imread(testImagePath);

% 2. Load U-Net (Segmentation - Module 2 | dlnetwork via trainnet)
fprintf('Loading U-Net Segmentation Model...\n');
unetPath   = fullfile('models', 'segmentation', 'vessel_unet.mat');
if ~exist(unetPath, 'file')
    error('U-Net model not found at %s. Run trainVessels_Corrected.m first.', unetPath);
end
loadedUnet = load(unetPath, 'net');
unetModel  = loadedUnet.net;

% 3. Load EfficientNet-B0 (Classification - Module 3)
fprintf('Loading EfficientNet-B0 Classification Model...\n');
classPath   = fullfile('models', 'classification', 'drClassifier.mat');
if ~exist(classPath, 'file')
    error('Classifier not found at %s. Run runTraining.m first.', classPath);
end
loadedClass = load(classPath, 'net');
classModel  = loadedClass.net;

% 4. Run U-Net Segmentation via predict() (dlnetwork API)
fprintf('Running Vessel Segmentation...\n');
I_unet    = im2single(imresize(I, [512 512]));           % dlnetwork expects single [0,1]
probMap   = predict(unetModel, I_unet);                  % returns H x W x numClasses x 1
vesselProb      = probMap(:,:,2,1);                      % channel 2 = vessel probability
vesselOnlyMask  = vesselProb > 0.5;                      % binary logical mask

% 5. Run EfficientNet Classification via classify()
fprintf('Running DR Classification...\n');
I_class              = im2single(imresize(I, [224 224])); % single [0,1] for EfficientNet
[predLabel, scores]  = classify(classModel, I_class);

% 6. Map Output to Clinical Labels
icdrLabels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"];
gradeNum   = str2double(string(predLabel));
if isnan(gradeNum) && ismember(string(predLabel), icdrLabels)
    finalDiagnosis = string(predLabel);
    gradeNum       = find(icdrLabels == string(predLabel), 1) - 1;
elseif ~isnan(gradeNum) && gradeNum >= 0 && gradeNum <= 4
    finalDiagnosis = icdrLabels(gradeNum + 1);
else
    finalDiagnosis = string(predLabel);
    gradeNum       = -1;
end
confidence = double(max(scores(:))) * 100;

% 7. Display Final Clinical Dashboard
figure('Name', 'Automated DR Diagnostic Dashboard', 'Position', [100, 100, 1000, 450]);

subplot(1, 2, 1);
% Overlay vessel pixels in bright red using labeloverlay
B = labeloverlay(imresize(I, [512 512]), vesselOnlyMask, 'Colormap', [1 0 0], 'Transparency', 0.4);
imshow(B);
title('Module 2: U-Net Vessel Masking');

subplot(1, 2, 2);
imshow(imresize(I, [224 224]));
if gradeNum >= 0
    title(sprintf('Module 3: Classification\nDiagnosis: %s (Grade %d)\nConfidence: %.1f%%', ...
        finalDiagnosis, gradeNum, confidence));
else
    title(sprintf('Module 3: Classification\nDiagnosis: %s\nConfidence: %.1f%%', ...
        finalDiagnosis, confidence));
end