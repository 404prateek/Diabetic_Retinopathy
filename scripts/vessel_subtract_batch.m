% Batch vessel subtraction script for dataset_yolo images
clear; clc;

targetFolders = { ...
    fullfile('dataset_yolo', 'images', 'train'), ...
    fullfile('dataset_yolo', 'images', 'val') ...
};

unetPath = fullfile('models', 'segmentation', 'vessel_unet.mat');
useUnet  = exist(unetPath, 'file') == 2;

if useUnet
    loaded = load(unetPath, 'net');
    vesselNet = loaded.net;
else
    fprintf('vessel_unet.mat not found, using Frangi fallback\n');
end

totalProcessed = 0;

for fIdx = 1:numel(targetFolders)
    currentDir = targetFolders{fIdx};
    if ~exist(currentDir, 'dir')
        continue;
    end
    
    imgFiles = [dir(fullfile(currentDir, '*.jpg')); ...
                dir(fullfile(currentDir, '*.png')); ...
                dir(fullfile(currentDir, '*.tif'))];
            
    for i = 1:numel(imgFiles)
        imgPath = fullfile(currentDir, imgFiles(i).name);
        I_orig  = imread(imgPath);
        [origH, origW, ~] = size(I_orig);
        
        if useUnet
            I_input = im2single(imresize(I_orig, [512 512]));
            probMap = predict(vesselNet, I_input);
            vesselProb512 = probMap(:,:,2,1);
            mask512 = vesselProb512 > 0.5;
            vesselMask = imresize(mask512, [origH origW], 'nearest');
        else
            I_green = I_orig(:,:,2);
            I_adj   = adapthisteq(I_green);
            se = strel('disk', 6);
            vessels_enhanced = imtophat(imcomplement(I_adj), se);
            vesselMask = vessels_enhanced > prctile(vessels_enhanced(:), 85);
            vesselMask = bwareaopen(vesselMask, 15);
        end
        
        % Inpaint vessels using regionfill
        vesselMaskDilated = imdilate(vesselMask, strel('disk', 2));
        I_vesselFree = I_orig;
        for c = 1:size(I_orig, 3)
            I_chan = I_orig(:,:,c);
            I_vesselFree(:,:,c) = regionfill(I_chan, vesselMaskDilated);
        end
        
        imwrite(I_vesselFree, imgPath, 'Quality', 95);
        totalProcessed = totalProcessed + 1;
    end
end

fprintf('Done. Processed %d images.\n', totalProcessed);
