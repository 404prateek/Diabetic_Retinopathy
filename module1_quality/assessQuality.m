function result = assessQuality(I)
% Assess image quality of a fundus photograph (sharpness and brightness)
% to determine if it passes the quality gate for screening.

% Guard: handle both uint8 and double inputs
if ~isa(I, 'uint8')
    I = im2uint8(I);
end

% Guard: must be an RGB image
if size(I, 3) ~= 3
    error('assessQuality:BadInput', 'Input image must be H x W x 3 RGB uint8.');
end

% Convert RGB image to grayscale
grayImage = rgb2gray(I);

% Convert to double for numerical calculations
grayImage = double(grayImage);

% Compute Laplacian response
laplacianKernel = [0 1 0; 1 -4 1; 0 1 0];
laplacianImage = imfilter(grayImage, laplacianKernel, 'replicate');

% Variance of Laplacian = sharpness/focus score
result.sharpness = var(laplacianImage(:));

% Extract green channel
greenChannel = double(I(:,:,2));

% Calculate brightness statistics
result.brightnessMean = mean(greenChannel(:));
result.brightnessStd  = std(greenChannel(:));

% Quality thresholds
% Initial calibration based on real IDRiD fundus-image testing.
% These values are tunable and are not clinically validated.

if result.sharpness < 5
    result.status = "REJECT";

elseif result.sharpness < 30
    result.status = "ENHANCE";

else
    result.status = "PASS";
end

end  % assessQuality
