function result = assessQuality(I)
% =========================================================================
% assessQuality  -  Assess the quality of a fundus image
% =========================================================================
% Purpose : Analyse a raw fundus photograph and decide whether it is usable
%           for automated DR screening.
%
% Owner   : Member 1  (data/, preprocessing/, module1_quality/)
%
% Inputs  :
%   I       - (H x W x 3 uint8)  RGB fundus image
%
% Outputs :
%   result  - struct with fields:
%     .status         (string)  "PASS" | "ENHANCE" | "REJECT"
%     .sharpness      (double)  Laplacian-variance focus score
%     .brightnessMean (double)  Mean pixel intensity, green channel [0-255]
%     .brightnessStd  (double)  Std  pixel intensity, green channel
%
% Usage example:
%   q = assessQuality(imread('data/APTOS/sample.png'));
%   if q.status == "REJECT", return; end
%
% Thresholds (to be tuned by owner):
%   sharpness < 50         -> REJECT
%   50 <= sharpness < 100  -> ENHANCE
%   sharpness >= 100       -> PASS  (subject to brightness bounds)
%
% Dependencies : MATLAB Image Processing Toolbox
% =========================================================================

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
result.brightnessStd = std(greenChannel(:));

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

