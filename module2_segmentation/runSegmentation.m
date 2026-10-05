function segmentation = runSegmentation(I)
% Segment retinal blood vessels and lesions.
% Uses trained U-Net if available in models/segmentation/vessel_unet.mat,
% with a classical Frangi filter fallback.

if ~isa(I, 'uint8')
    I = im2uint8(I);
end

[H, W, ~] = size(I);

% Check for trained U-Net model
vesselModelPath = fullfile('models', 'segmentation', 'vessel_unet.mat');
useDeepModel    = exist(vesselModelPath, 'file');

if useDeepModel
    % --- Deep Learning path (dlnetwork / predict API) -----------------
    persistent net;

    if isempty(net)
        loaded = load(vesselModelPath, 'net');
        net    = loaded.net;
        fprintf('[runSegmentation] Loaded vessel U-Net from %s\n', vesselModelPath);
    end

    % Resize to 512x512 and normalize to single [0,1] for dlnetwork
    Iresized = im2single(imresize(I, [512 512]));

    % predict() returns (H x W x numClasses x N) probability map
    probMap    = predict(net, Iresized);

    % Channel 2 = vessel probability
    vesselProb        = probMap(:,:,2,1);
    vesselMaskResized = vesselProb > 0.5;

    % Scale mask back to original image dimensions
    vesselMask = imresize(vesselMaskResized, [H, W], 'nearest');

    % Lesion segmentation: pending IDRiD integration (placeholder)
    lesionMask = false(H, W);

else
    % Classical morphological fallback
    green      = im2double(I(:,:,2));
    greenClahe = adapthisteq(green, 'ClipLimit', 0.01, 'NumTiles', [8 8]);

    % Vessel segmentation via Frangi vesselness
    vesselMask = frangiVesselness(greenClahe);

    % Lesion segmentation
    se_large = strel('disk', 10);

    % Hard exudates: bright structures (top-hat)
    topHat       = imtophat(green, se_large);
    brightLesion = safeBinaryThreshold(topHat, 0.6);

    % Microaneurysms / haemorrhages: dark structures (bottom-hat)
    bottomHat  = imbothat(green, se_large);
    darkLesion = safeBinaryThreshold(bottomHat, 0.5);

    % Suppress optic disc (large bright circular region) from lesion mask
    discMask     = imdilate(imopen(green > 0.85, strel('disk', 5)), strel('disk', 20));
    brightLesion = brightLesion & ~discMask;

    % Combine + morphological clean-up (remove tiny specks < 5 px)
    lesionRaw  = brightLesion | darkLesion;
    lesionMask = bwareaopen(lesionRaw, 5);
end

% ------------------------------------------------------------------
% Count distinct lesion connected components
% ------------------------------------------------------------------
cc = bwconncomp(lesionMask, 8);

% Populate output struct
segmentation.vesselMask  = vesselMask;
segmentation.lesionMask  = logical(lesionMask);
segmentation.lesionCount = cc.NumObjects;
segmentation.dice        = NaN;
segmentation.iou         = NaN;

end % runSegmentation


function mask = safeBinaryThreshold(img, factor)
% Safe binary thresholding handling zero signal
if max(img(:)) < eps
    mask = false(size(img));
    return;
end
thresh = graythresh(img) * factor;
mask   = img > thresh;
end


function vesselMask = frangiVesselness(img)
% Multi-scale Frangi vesselness filter

scales = [1, 2, 3];
beta   = 0.5;
imgMax = max(img(:));

% Guard against zero-image
c = max(0.5 * imgMax, 1e-6);

[H, W]     = size(img);
vesselness = zeros(H, W);

for sigma = scales
    [Hxx, Hxy, Hyy] = hessian2D(img, sigma);

    tmp     = sqrt(max((Hxx - Hyy).^2 + 4 * Hxy.^2, 0));
    lambda1 = 0.5 * ((Hxx + Hyy) - tmp);
    lambda2 = 0.5 * ((Hxx + Hyy) + tmp);

    Rb = lambda1 ./ (lambda2 + eps);
    S  = sqrt(lambda1.^2 + lambda2.^2);

    v              = exp(-Rb.^2 / (2 * beta^2)) .* (1 - exp(-S.^2 / (2 * c^2)));
    v(lambda2 >= 0) = 0;

    vesselness = max(vesselness, v);
end

if max(vesselness(:)) < eps
    vesselMask = false(H, W);
    return;
end

thresh     = graythresh(vesselness) * 0.5;
vesselMask = logical(vesselness > thresh);

vesselMask = bwareaopen(vesselMask, 20);
vesselMask = imfill(vesselMask, 'holes');

end


function [Dxx, Dxy, Dyy] = hessian2D(img, sigma)
% 2-D scale-normalised Hessian via Gaussian derivative filters
sz  = ceil(3 * sigma) * 2 + 1;
[x, y] = meshgrid(-(sz-1)/2 : (sz-1)/2, -(sz-1)/2 : (sz-1)/2);

rSq = x.^2 + y.^2;
G   = exp(-rSq / (2 * sigma^2));

Gxx = sigma^2 * (x.^2 / sigma^4 - 1/sigma^2) .* G;
Gyy = sigma^2 * (y.^2 / sigma^4 - 1/sigma^2) .* G;
Gxy = sigma^2 * (x .* y / sigma^4)            .* G;

Dxx = imfilter(img, Gxx, 'replicate');
Dyy = imfilter(img, Gyy, 'replicate');
Dxy = imfilter(img, Gxy, 'replicate');
end
