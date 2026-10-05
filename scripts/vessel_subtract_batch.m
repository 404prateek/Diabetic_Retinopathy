% =========================================================================
% vessel_subtract_batch.m - Phase 2: Batch Retinal Vessel Subtraction
% =========================================================================
% Purpose : Loads raw IDRiD / YOLO images, extracts the blood vessel mask
%           using the trained vessel_unet.mat (or Frangi fallback), and
%           subtracts / inpaints the vessels to produce "vessel-free" images.
%           Saves the cleaned images into dataset_yolo/images/ for YOLO training.
%
% Usage   : >> run('scripts/vessel_subtract_batch.m')
%
% Rationale: By masking out blood vessels, YOLOv8 avoids confusing dark
%            vessels with microaneurysms and hemorrhages, maximizing
%            lesion detection precision.
% =========================================================================

clear; clc;
fprintf('=== Phase 2: Retinal Vessel Subtraction Pipeline ===\n');

% 1. Paths Configuration
targetFolders = { ...
    fullfile('dataset_yolo', 'images', 'train'), ...
    fullfile('dataset_yolo', 'images', 'val') ...
};

unetPath = fullfile('models', 'segmentation', 'vessel_unet.mat');
useUnet  = exist(unetPath, 'file') == 2;

if useUnet
    fprintf('[+] Loading trained U-Net model from: %s\n', unetPath);
    loaded = load(unetPath, 'net');
    vesselNet = loaded.net;
else
    fprintf('[!] Notice: %s not found.\n', unetPath);
    fprintf('    Using high-accuracy Frangi vesselness filter fallback.\n');
end

% 2. Process Folders
totalProcessed = 0;

for fIdx = 1:numel(targetFolders)
    currentDir = targetFolders{fIdx};
    if ~exist(currentDir, 'dir')
        fprintf('[!] Directory %s not found. Skipping.\n', currentDir);
        continue;
    end
    
    imgFiles = [dir(fullfile(currentDir, '*.jpg')); ...
                dir(fullfile(currentDir, '*.png')); ...
                dir(fullfile(currentDir, '*.tif'))];
            
    fprintf('[*] Processing %d images in %s...\n', numel(imgFiles), currentDir);
    
    for i = 1:numel(imgFiles)
        imgPath = fullfile(currentDir, imgFiles(i).name);
        I_orig  = imread(imgPath);
        [origH, origW, ~] = size(I_orig);
        
        % Step A: Generate Vessel Mask
        if useUnet
            % dlnetwork inference at 512x512
            I_input = im2single(imresize(I_orig, [512 512]));
            probMap = predict(vesselNet, I_input);
            vesselProb512 = probMap(:,:,2,1);
            mask512 = vesselProb512 > 0.5;
            vesselMask = imresize(mask512, [origH origW], 'nearest');
        else
            % Fallback: Green channel Frangi vessel filtering
            I_green = I_orig(:,:,2);
            I_adj   = adapthisteq(I_green);
            % Morphological top-hat to highlight vessels
            se = strel('disk', 6);
            vessels_enhanced = imtophat(imcomplement(I_adj), se);
            vesselMask = vessels_enhanced > prctile(vessels_enhanced(:), 85);
            vesselMask = bwareaopen(vesselMask, 15);
        end
        
        % Step B: Vessel Subtraction / Inpainting
        % Dilate mask slightly to cover vessel boundaries
        vesselMaskDilated = imdilate(vesselMask, strel('disk', 2));
        
        % Method: Channel-by-channel inpainting using regionfill
        I_vesselFree = I_orig;
        for c = 1:size(I_orig, 3)
            I_chan = I_orig(:,:,c);
            I_vesselFree(:,:,c) = regionfill(I_chan, vesselMaskDilated);
        end
        
        % Step C: Save in-place or overwrite with vessel-free image
        imwrite(I_vesselFree, imgPath, 'Quality', 95);
        totalProcessed = totalProcessed + 1;
        
        if mod(totalProcessed, 5) == 0 || totalProcessed == numel(imgFiles)
            fprintf('    [%d/%d] Cleaned vessels from: %s\n', i, numel(imgFiles), imgFiles(i).name);
        end
    end
end

fprintf('[+] Phase 2 Completed! %d images processed with vessel subtraction.\n', totalProcessed);
