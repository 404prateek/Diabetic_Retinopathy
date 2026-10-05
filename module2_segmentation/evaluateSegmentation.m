% =========================================================================
% evaluateSegmentation.m - Robust Vessel Segmentation Evaluation
% =========================================================================
clear; clc;

% 1. Dynamically locate project paths
scriptFolder = fileparts(mfilename('fullpath'));
projectRoot  = fileparts(scriptFolder);

% 2. Automatically locate Image Directory
candidateImageDirs = { ...
    fullfile(projectRoot, 'data', 'DRIVE', 'test', 'images'), ...
    fullfile(projectRoot, 'data', 'DRIVE', 'training', 'images'), ...
    fullfile(projectRoot, 'data', 'DRIVE', 'images') ...
};

testImageDir = '';
for k = 1:numel(candidateImageDirs)
    if exist(candidateImageDirs{k}, 'dir')
        testImageDir = candidateImageDirs{k};
        break;
    end
end

if isempty(testImageDir)
    error('Could not locate DRIVE images folder under data/DRIVE/');
end

% 3. Automatically locate Mask Directory (1st_manual contains vessel GT)
candidateMaskDirs = { ...
    fullfile(projectRoot, 'data', 'DRIVE', 'test', '1st_manual'), ...
    fullfile(projectRoot, 'data', 'DRIVE', 'training', '1st_manual'), ...
    fullfile(projectRoot, 'data', 'DRIVE', '1st_manual'), ...
    fullfile(projectRoot, 'data', 'DRIVE', 'masks') ...
};

testMaskDir = '';
for k = 1:numel(candidateMaskDirs)
    if exist(candidateMaskDirs{k}, 'dir')
        testMaskDir = candidateMaskDirs{k};
        break;
    end
end

if isempty(testMaskDir)
    error('Could not locate DRIVE ground-truth mask folder under data/DRIVE/');
end

% 4. Set up Datastores
imdsTest = imageDatastore(testImageDir);
pxdsTest = imageDatastore(testMaskDir, 'FileExtensions', {'.tif', '.gif', '.png', '.jpg'}); 

numImages = numel(imdsTest.Files);
totalDice = 0;
totalIoU = 0;

fprintf('Evaluating %d images for Vessel Segmentation...\n', numImages);
fprintf('Image Path: %s\n', testImageDir);
fprintf('Mask Path:  %s\n\n', testMaskDir);

% 5. Loop through test set, predict, and calculate metrics
for i = 1:numImages
    % Read image and ground-truth mask
    img = readimage(imdsTest, i);
    groundTruth = readimage(pxdsTest, i);
    
    % Ensure ground truth is binary logical (vessels = true/1)
    if size(groundTruth, 3) > 1
        groundTruth = rgb2gray(groundTruth);
    end
    gtLogical = groundTruth > 0; 
    
    % Run model inference
    segResult = runSegmentation(img);
    
    % Extract mask from prediction struct
    if isfield(segResult, 'vesselMask')
        rawMask = segResult.vesselMask;
    elseif isfield(segResult, 'binaryMask')
        rawMask = segResult.binaryMask;
    elseif isfield(segResult, 'mask')
        rawMask = segResult.mask;
    else
        fn = fieldnames(segResult);
        rawMask = segResult.(fn{1});
    end
    
    % Convert prediction to strict logical boolean array
    if iscategorical(rawMask)
        predLogical = (rawMask == "vessel") | (rawMask == "1");
    elseif islogical(rawMask)
        predLogical = rawMask;
    else
        predLogical = rawMask > 0;
    end
    
    % Resize ground truth to match prediction dimensions (512x512)
    gtResized = imresize(gtLogical, size(predLogical), 'nearest');
    
    % Handle inverted mask logic if background was assigned as 1
    if sum(predLogical(:)) > (numel(predLogical) * 0.7)
        predLogical = ~predLogical;
    end
    
    % Print diagnostic details on Image 1
    if i == 1
        fprintf('--- DIAGNOSTIC AUDIT (Image 1) ---\n');
        fprintf('Image 1 File: %s\n', imdsTest.Files{1});
        fprintf('Mask 1 File:  %s\n', pxdsTest.Files{1});
        fprintf('Predicted Vessel Pixels: %d\n', sum(predLogical(:)));
        fprintf('Ground Truth Vessel Pixels: %d\n', sum(gtResized(:)));
        fprintf('----------------------------------\n\n');
    end
    
    % Calculate Dice and IoU
    imgDice = dice(predLogical, gtResized);
    imgIoU  = jaccard(predLogical, gtResized);
    
    totalDice = totalDice + imgDice;
    totalIoU  = totalIoU + imgIoU;
end

% 6. Print Final Results
avgDice = totalDice / numImages;
avgIoU = totalIoU / numImages;

fprintf('=== Module 2 Vessel Evaluation Metrics ===\n');
fprintf('Average Dice Coefficient: %.4f (Target: > 0.82)\n', avgDice);
fprintf('Average IoU (Jaccard):    %.4f\n', avgIoU);
fprintf('==========================================\n');