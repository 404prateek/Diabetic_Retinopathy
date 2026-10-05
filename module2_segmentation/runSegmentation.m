function segmentation = runSegmentation(I)
% =========================================================================
% runSegmentation  -  Segment retinal vessels and DR lesions in a fundus image
% =========================================================================
% Purpose : Apply a trained deep segmentation model to detect retinal vessels.
% Owner   : Member 2  (module2_segmentation/)
% =========================================================================
persistent net
if isempty(net)
    modelPath = fullfile("models", "segmentation", "vessel_unet.mat");
    if ~exist(modelPath, 'file')
        error('Model file not found at %s. Ensure vessel U-Net training has run.', modelPath);
    end
    loaded = load(modelPath, 'net');
    net = loaded.net;
end

% Capture original dimensions to scale masks back properly
[H, W, ~] = size(I);

% 1. Resize I to network input size and normalize to [0, 1]
Iresized = im2single(imresize(I, [512 512]));

% 2. Run modern dlnetwork prediction (compatible with trainnet)
% predict returns probabilities for [background, vessel] across spatial grid
probMap = predict(net, Iresized);

% Extract the vessel probability channel (index 2)
vesselProb = probMap(:,:,2,1);

% 3. Apply threshold (0.5) to generate binary mask
vesselMaskResized = vesselProb > 0.5;

% 4. Scale back to original image dimensions
vesselMask = imresize(vesselMaskResized, [H, W], 'nearest');

% Lesion model is pending IDRiD integration; initialize as empty logical mask
lesionMask = false(H, W);
cc = bwconncomp(lesionMask);
lesionCount = cc.NumObjects;

% 5. Populate struct contract
segmentation.vesselMask = vesselMask;
segmentation.lesionMask = lesionMask;
segmentation.lesionCount = lesionCount;
segmentation.dice = NaN;
segmentation.iou = NaN;
end