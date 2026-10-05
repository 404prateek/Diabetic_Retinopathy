% Unit tests for Module 4: Explainability

fprintf('Running Module 4 Tests: Explainability...\n');

%% Create a dummy 256x256 synthetic fundus image and fake pipeline outputs
[xx, yy] = meshgrid(1:256, 1:256);
mask      = sqrt((xx-128).^2 + (yy-128).^2) < 100;
dummyImg  = uint8(zeros(256,256,3));
dummyImg(:,:,2) = uint8(mask * 120);

% Dummy structs mimicking module outputs
dummyQuality.status         = "PASS";
dummyQuality.sharpness      = 120.5;
dummyQuality.brightnessMean = 85.0;
dummyQuality.brightnessStd  = 12.3;

dummySeg.vesselMask  = false(256,256);
dummySeg.lesionMask  = false(256,256);
dummySeg.lesionCount = 0;
dummySeg.dice        = NaN;
dummySeg.iou         = NaN;

dummyClass = "Moderate";
dummyConf  = 0.82;

% Test 1: explainPrediction returns uint8 RGB of correct size
fprintf('Test 1: explainPrediction() output...\n');
heatmap = explainPrediction(dummyImg, dummyClass);

assert(isa(heatmap, 'uint8'),                    'FAIL: heatmap must be uint8');
assert(ndims(heatmap) == 3,                      'FAIL: heatmap must be 3-channel (H x W x 3)');
assert(isequal(size(heatmap,1), size(dummyImg,1)) && ...
       isequal(size(heatmap,2), size(dummyImg,2)), ...
       'FAIL: heatmap spatial size must match input image');

fprintf('  PASS  (size=%dx%dx%d)\n\n', size(heatmap,1), size(heatmap,2), size(heatmap,3));

% Test 2: generateReport runs without error (smoke test)
fprintf('Test 2: generateReport() smoke test...\n');
try
    generateReport(dummyImg, dummyQuality, dummySeg, dummyClass, dummyConf, heatmap);
    fprintf('  PASS  (generateReport completed without error)\n\n');
catch ME
    fprintf('  FAIL: generateReport threw: %s\n\n', ME.message);
    rethrow(ME);
end

fprintf('Module 4: All tests passed.\n');
