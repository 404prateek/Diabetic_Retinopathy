function segmentation = runSegmentation(I)
% =========================================================================
% runSegmentation  -  Segment retinal vessels and DR lesions in a fundus image
% =========================================================================
% Purpose : Apply a trained deep segmentation model (e.g. U-Net) to detect
%           retinal blood vessels and diabetic retinopathy lesions such as
%           microaneurysms, haemorrhages, and hard exudates.
%
% Owner   : Member 2  (module2_segmentation/)
%
% Inputs  :
%   I             - (H x W x 3 uint8)  Enhanced RGB fundus image.
%
% Outputs :
%   segmentation  - struct with fields:
%     .vesselMask   (H x W logical)  Binary mask of detected blood vessels
%     .lesionMask   (H x W logical)  Binary mask of all detected lesions
%     .lesionCount  (int)            Total number of distinct lesion regions
%     .dice         (double)         Dice vs ground-truth (NaN if unavailable)
%     .iou          (double)         IoU vs ground-truth  (NaN if unavailable)
%
% BugFix (2024-10):
%   - Fixed divide-by-zero in Frangi when c=0 (all-black image)
%   - Fixed graythresh on all-zero top-hat/bottom-hat → all pixels passing
%   - Both fixes are guarded with max(..., eps) and early-exit on zero maps
%
% Dependencies : MATLAB Image Processing Toolbox
%               (MATLAB Deep Learning Toolbox optional – for U-Net inference)
% =========================================================================

% Input guard
if ~isa(I, 'uint8')
    I = im2uint8(I);
end

H = size(I, 1);
W = size(I, 2);

% ------------------------------------------------------------------
% Attempt 1: Load pre-trained U-Net models from models/segmentation/
% ------------------------------------------------------------------
vesselModelPath = fullfile('models', 'segmentation', 'unet_vessel.mat');
lesionModelPath = fullfile('models', 'segmentation', 'unet_lesion.mat');

useDeepModel = exist(vesselModelPath, 'file') && exist(lesionModelPath, 'file');

if useDeepModel
    % --- Deep Learning path -------------------------------------------
    persistent netVessel netLesion;

    if isempty(netVessel)
        v         = load(vesselModelPath, 'net');
        netVessel = v.net;
    end
    if isempty(netLesion)
        l         = load(lesionModelPath, 'net');
        netLesion = l.net;
    end

    % Determine network input size (assume square)
    inSz = netVessel.Layers(1).InputSize(1:2);

    % Preprocess: resize + normalise to single [0,1]
    Iresized = im2single(imresize(I, inSz));

    % Run semantic segmentation
    vesselSeg = semanticseg(Iresized, netVessel);
    lesionSeg = semanticseg(Iresized, netLesion);

    % Convert categorical output to logical masks at original resolution
    vesselMaskSmall = vesselSeg == categorical({'vessel'});
    lesionMaskSmall = lesionSeg == categorical({'lesion'});

    segmentation.vesselMask = imresize(uint8(vesselMaskSmall), [H, W], 'nearest') > 0;
    segmentation.lesionMask = imresize(uint8(lesionMaskSmall), [H, W], 'nearest') > 0;

else
    % --- Classical morphological fallback --------------------------------
    green = im2double(I(:,:,2));

    % CLAHE on green channel for contrast enhancement
    greenClahe = adapthisteq(green, 'ClipLimit', 0.01, 'NumTiles', [8 8]);

    % Vessel segmentation via Frangi vesselness (Hessian ridge detector)
    vesselMask = frangiVesselness(greenClahe);  % helper at bottom of file

    % ---- Lesion segmentation ----
    se_large = strel('disk', 10);

    % Hard exudates: bright structures (top-hat)
    topHat     = imtophat(green, se_large);
    brightLesion = safeBinaryThreshold(topHat, 0.6);

    % Microaneurysms / haemorrhages: dark structures (bottom-hat)
    bottomHat  = imbothat(green, se_large);
    darkLesion = safeBinaryThreshold(bottomHat, 0.5);

    % Suppress optic disc (large bright circular region) from lesion mask
    discMask     = imdilate(imopen(green > 0.85, strel('disk', 5)), strel('disk', 20));
    brightLesion = brightLesion & ~discMask;

    % Combine + morphological clean-up (remove tiny specks < 5 px)
    lesionRaw  = brightLesion | darkLesion;
    lesionClean = bwareaopen(lesionRaw, 5);

    segmentation.vesselMask = vesselMask;
    segmentation.lesionMask = logical(lesionClean);
end

% ------------------------------------------------------------------
% Count distinct lesion connected components
% ------------------------------------------------------------------
cc = bwconncomp(segmentation.lesionMask, 8);
segmentation.lesionCount = cc.NumObjects;

% Dice and IoU - NaN at inference time (no ground-truth)
segmentation.dice = NaN;
segmentation.iou  = NaN;

end % runSegmentation


% =========================================================================
% Helper: safe binary thresholding that handles all-zero maps
% =========================================================================
function mask = safeBinaryThreshold(img, factor)
% Returns false(size) if image has no signal, otherwise Otsu threshold * factor.
if max(img(:)) < eps
    mask = false(size(img));
    return;
end
thresh = graythresh(img) * factor;
mask   = img > thresh;
end


% =========================================================================
% Helper: Frangi 2-D vesselness filter (Hessian-based ridge detector)
% =========================================================================
function vesselMask = frangiVesselness(img)
% Multi-scale Frangi vesselness – handles all-black images safely.

scales    = [1, 2, 3];   % Gaussian sigma values
beta      = 0.5;          % blob-shape sensitivity
imgMax    = max(img(:));

% BUG FIX: guard c against zero-image (prevents div-by-zero in exp term)
c = max(0.5 * imgMax, 1e-6);

[H, W]     = size(img);
vesselness = zeros(H, W);

for sigma = scales
    [Hxx, Hxy, Hyy] = hessian2D(img, sigma);

    % Eigenvalues of the 2×2 Hessian at every pixel
    tmp     = sqrt(max((Hxx - Hyy).^2 + 4 * Hxy.^2, 0));
    lambda1 = 0.5 * ((Hxx + Hyy) - tmp);   % smaller eigenvalue
    lambda2 = 0.5 * ((Hxx + Hyy) + tmp);   % larger  eigenvalue

    % Frangi vesselness (1998) - only for dark ridges (lambda2 < 0)
    Rb = lambda1 ./ (lambda2 + eps);
    S  = sqrt(lambda1.^2 + lambda2.^2);

    v             = exp(-Rb.^2 / (2 * beta^2)) .* (1 - exp(-S.^2 / (2 * c^2)));
    v(lambda2 >= 0) = 0;    % suppress blob-like / flat regions

    vesselness = max(vesselness, v);
end

% BUG FIX: handle all-zero vesselness (blank image)
if max(vesselness(:)) < eps
    vesselMask = false(H, W);
    return;
end

thresh     = graythresh(vesselness) * 0.5;
vesselMask = logical(vesselness > thresh);

% Morphological clean-up
vesselMask = bwareaopen(vesselMask, 20);
vesselMask = imfill(vesselMask, 'holes');

end


% =========================================================================
% Helper: 2-D scale-normalised Hessian via Gaussian derivative filters
% =========================================================================
function [Dxx, Dxy, Dyy] = hessian2D(img, sigma)
sz  = ceil(3 * sigma) * 2 + 1;
[x, y] = meshgrid(-(sz-1)/2 : (sz-1)/2, -(sz-1)/2 : (sz-1)/2);

rSq = x.^2 + y.^2;
G   = exp(-rSq / (2 * sigma^2));

% Scale-normalised 2nd-order Gaussian derivatives
Gxx = sigma^2 * (x.^2 / sigma^4 - 1/sigma^2) .* G;
Gyy = sigma^2 * (y.^2 / sigma^4 - 1/sigma^2) .* G;
Gxy = sigma^2 * (x .* y / sigma^4)            .* G;

Dxx = imfilter(img, Gxx, 'replicate');
Dyy = imfilter(img, Gyy, 'replicate');
Dxy = imfilter(img, Gxy, 'replicate');
end
