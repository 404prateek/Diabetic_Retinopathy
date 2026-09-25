function enhanced = enhanceFundus(I)
% =========================================================================
% enhanceFundus  -  Pre-process a borderline-quality fundus image
% =========================================================================
% Purpose : Apply enhancement steps (CLAHE, green-channel normalisation,
%           background subtraction) to images flagged "ENHANCE" by
%           assessQuality(). Output should pass assessQuality() as "PASS".
%
% Owner   : Member 1  (data/, preprocessing/, module1_quality/)
%
% Inputs  :
%   I        - (H x W x 3 uint8)  RGB fundus image (raw / degraded)
%
% Outputs :
%   enhanced - (H x W x 3 uint8)  Enhanced RGB image, same size as I
%
% Usage example:
%   q = assessQuality(I);
%   if q.status == "ENHANCE"
%       I = enhanceFundus(I);
%   end
%
% Planned steps (owner to implement):
%   1. Subtract background illumination via morphological opening.
%   2. Apply CLAHE (adapthisteq) on L* channel of L*a*b* space.
%   3. Normalise green channel to zero mean / unit std (Ben Graham method).
%   4. Resize to standard canvas (e.g. 512x512) with circular mask padding.
%
% Dependencies : MATLAB Image Processing Toolbox
% =========================================================================

% Convert input to double for processing
I = im2double(I);

% ---------------------------------------------------------
% 1. Background illumination subtraction
% ---------------------------------------------------------
% Use morphological opening to estimate the smooth background.
grayImage = rgb2gray(I);

se = strel('disk', 15);
background = imopen(grayImage, se);

% Subtract background illumination
corrected = grayImage - background;

% Normalize corrected image to [0, 1]
corrected = mat2gray(corrected);

% ---------------------------------------------------------
% 2. CLAHE on L* channel of Lab color space
% ---------------------------------------------------------
labImage = rgb2lab(I);

L = labImage(:,:,1) / 100;

% Apply adaptive histogram equalization
L_enhanced = adapthisteq(L, 'ClipLimit', 0.01);

labImage(:,:,1) = L_enhanced * 100;

enhancedLab = lab2rgb(labImage);

% ---------------------------------------------------------
% 3. Green-channel normalization
% ---------------------------------------------------------
green = enhancedLab(:,:,2);

greenMean = mean(green(:));
greenStd = std(green(:));

if greenStd > 0
    greenNormalized = (green - greenMean) / greenStd;
else
    greenNormalized = green;
end

% Convert normalized green channel back to [0,1]
greenNormalized = mat2gray(greenNormalized);

enhancedLab(:,:,2) = greenNormalized;

% ---------------------------------------------------------
% 4. Combine enhancement with background correction
% ---------------------------------------------------------
% Use the enhanced RGB image as the main result.
enhanced = enhancedLab;

% Add a small amount of illumination correction.
enhanced = enhanced .* (0.8 + 0.2 * corrected);

% Ensure valid image range
enhanced = min(max(enhanced, 0), 1);

% Convert back to uint8
enhanced = im2uint8(enhanced);

% Preserve the original spatial dimensions
if ~isequal(size(enhanced), size(I))
    enhanced = imresize(enhanced, [size(I,1), size(I,2)]);
end

end
