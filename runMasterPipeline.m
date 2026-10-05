% =========================================================================
% runMasterPipeline.m - End-to-End Diabetic Retinopathy Dashboard
% =========================================================================
clear; clc;

% 1. Auto-Detect an Image to Test
testFolder = fullfile('data', 'APTOS', 'test_images');
imageFiles = dir(fullfile(testFolder, '*.png')); 

if isempty(imageFiles)
    error('No PNG images found in %s. Please ensure your images are there.', testFolder);
end

testImagePath = fullfile(testFolder, imageFiles(1).name);
fprintf('Testing on image: %s\n', imageFiles(1).name);
I = imread(testImagePath);

% 2. Load U-Net (Segmentation - Module 2)
fprintf('Loading U-Net Segmentation Model...\n');
loadedUnet = load(fullfile('models', 'segmentation', 'vessel_unet.mat')); 
unetModel = loadedUnet.net; 

% 3. Load EfficientNet-B0 (Classification - Module 3)
fprintf('Loading EfficientNet-B0 Classification Model...\n');
loadedClass = load(fullfile('models', 'classification', 'drClassifier.mat'));
classModel = loadedClass.net;

% 4. Run U-Net Segmentation (Execution forced to CPU)
fprintf('Running Vessel Segmentation...\n');
I_unet = imresize(I, [512 512]); 
vesselMask = semanticseg(I_unet, unetModel, 'ExecutionEnvironment', 'cpu');

% 5. Run EfficientNet Classification (Execution forced to CPU)
fprintf('Running DR Classification...\n');
I_class = imresize(I, [224 224]); 
[predLabel, scores] = classify(classModel, I_class, 'ExecutionEnvironment', 'cpu');

% 6. Map Output to Clinical Labels
icdrLabels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"];
gradeNum = str2double(string(predLabel));
finalDiagnosis = icdrLabels(gradeNum + 1);
confidence = max(scores) * 100;

% 7. Display Final Clinical Dashboard
figure('Name', 'Automated DR Diagnostic Dashboard', 'Position', [100, 100, 1000, 450]);

% Extract binary mask for vessel class (ignoring background)
cats = categories(vesselMask);
vesselOnlyMask = (vesselMask == cats{end}); % Selects the vessel class

subplot(1, 2, 1);
% Overlay ONLY the vessel pixels in bright red
B = labeloverlay(I_unet, vesselOnlyMask, 'Colormap', [1 0 0], 'Transparency', 0.4);
imshow(B);
title('Module 2: U-Net Vessel Masking');

subplot(1, 2, 2);
imshow(I_class);
title(sprintf('Module 3: Classification\nDiagnosis: %s (Grade %d)\nConfidence: %.1f%%', ...
    finalDiagnosis, gradeNum, confidence));