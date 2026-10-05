function enhanced = enhanceFundus(I)
% Pre-process borderline fundus images using illumination correction and CLAHE.

% Convert input to double for processing
I = im2double(I);

% Background illumination subtraction
grayImage = rgb2gray(I);
se = strel('disk', 15);
background = imopen(grayImage, se);
corrected = grayImage - background;
corrected = mat2gray(corrected);

% CLAHE on L* channel in Lab color space
labImage = rgb2lab(I);
L = labImage(:,:,1) / 100;
L_enhanced = adapthisteq(L, 'ClipLimit', 0.01);
labImage(:,:,1) = L_enhanced * 100;
enhancedLab = lab2rgb(labImage);

% Green-channel normalization
green = enhancedLab(:,:,2);
greenMean = mean(green(:));
greenStd = std(green(:));

if greenStd > 0
    greenNormalized = (green - greenMean) / greenStd;
else
    greenNormalized = green;
end
greenNormalized = mat2gray(greenNormalized);
enhancedLab(:,:,2) = greenNormalized;

% Combine enhancement with illumination correction
enhanced = enhancedLab;
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
