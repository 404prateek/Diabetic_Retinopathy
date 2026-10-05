% runMasterPipeline.m
% End-to-end dashboard: vessel segmentation + lesion detection + DR classification

clear; clc;

% Find test image
testFolder = fullfile('data', 'APTOS', 'test_images');
imageFiles = dir(fullfile(testFolder, '*.png'));

if isempty(imageFiles)
    testFolder = fullfile('data', 'APTOS', 'train_images');
    imageFiles = dir(fullfile(testFolder, '*.png'));
    if isempty(imageFiles)
        error('No PNG images found in data/APTOS/test_images or train_images.');
    end
end

testImagePath = fullfile(testFolder, imageFiles(1).name);
fprintf('Testing on image: %s\n', imageFiles(1).name);
I = imread(testImagePath);

% Load vessel U-Net
fprintf('Loading U-Net Segmentation Model...\n');
unetPath = fullfile('models', 'segmentation', 'vessel_unet.mat');
if ~exist(unetPath, 'file')
    error('U-Net model not found at %s. Run trainVessels_Corrected.m first.', unetPath);
end
loadedUnet = load(unetPath, 'net');
unetModel  = loadedUnet.net;

% Load DR classifier
fprintf('Loading EfficientNet-B0 Classification Model...\n');
classPath = fullfile('models', 'classification', 'drClassifier.mat');
if ~exist(classPath, 'file')
    error('Classifier not found at %s. Run runTraining.m first.', classPath);
end
loadedClass = load(classPath, 'net');
classModel  = loadedClass.net;

% Vessel segmentation
fprintf('Running Vessel Segmentation...\n');
I_unet         = im2single(imresize(I, [512 512]));
probMap        = predict(unetModel, I_unet);
vesselProb     = probMap(:,:,2,1);
vesselOnlyMask = vesselProb > 0.5;

% YOLO lesion detection
fprintf('Running YOLOv8 Lesion Detection...\n');
addpath('module2_segmentation');
yoloOnnxPath = fullfile('models', 'detection', 'best.onnx');

I_vesselFree = imresize(I, [512 512]);
I_vesselFree(repmat(vesselOnlyMask, [1 1 3])) = 0;

[annotatedLesions, bboxes, bScores, bLabels] = detectLesionsYOLO(I_vesselFree, yoloOnnxPath);

% EfficientNet grading
fprintf('Running DR Classification...\n');
I_class             = im2single(imresize(I, [224 224]));
[predLabel, scores] = classify(classModel, I_class);

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

% Dashboard display
figure('Name', 'Automated DR Diagnostic & Lesion Detection Dashboard', 'Position', [60, 100, 1350, 450]);

subplot(1, 3, 1);
B = labeloverlay(imresize(I, [512 512]), vesselOnlyMask, 'Colormap', [1 0 0], 'Transparency', 0.4);
imshow(B);
title('Module 2: U-Net Vessel Masking', 'FontWeight', 'bold');

subplot(1, 3, 2);
imshow(annotatedLesions);
if isempty(bboxes)
    title(sprintf('YOLOv8: Lesion Detection\n(MA, HE, EX, SE Bounding Boxes)'), 'FontWeight', 'bold');
else
    title(sprintf('YOLOv8: %d Lesions Detected\n(MA: %d, HE: %d, EX: %d, SE: %d)', ...
        size(bboxes, 1), sum(bLabels=="Microaneurysm"), sum(bLabels=="Hemorrhage"), ...
        sum(bLabels=="Hard Exudate"), sum(bLabels=="Soft Exudate")), 'FontWeight', 'bold');
end

subplot(1, 3, 3);
imshow(imresize(I, [224 224]));
if gradeNum >= 0
    title(sprintf('Module 3: DR Classification\nDiagnosis: %s (Grade %d)\nConfidence: %.1f%%', ...
        finalDiagnosis, gradeNum, confidence), 'FontWeight', 'bold');
else
    title(sprintf('Module 3: DR Classification\nDiagnosis: %s\nConfidence: %.1f%%', ...
        finalDiagnosis, confidence), 'FontWeight', 'bold');
end