%% test_module1.m  -  Unit tests for Module 1: Quality Assessment
% =========================================================================
% Project : Explainable AI for DR Screening (PS ID 26038)
% Owner   : Member 1  (module1_quality/)
% Purpose : Minimal smoke-tests for assessQuality() and enhanceFundus().
%           Run from repo root:  >> run('tests/test_module1.m')
% =========================================================================

fprintf('=== Module 1 Tests: Quality Assessment ===\n\n');

%% Create a dummy 256x256 synthetic fundus image (grey disk on black)
[xx, yy] = meshgrid(1:256, 1:256);
mask      = sqrt((xx-128).^2 + (yy-128).^2) < 100;
dummyImg  = uint8(zeros(256, 256, 3));
dummyImg(:,:,1) = uint8(mask * 80);
dummyImg(:,:,2) = uint8(mask * 120);   % Green channel brightest (retina-like)
dummyImg(:,:,3) = uint8(mask * 60);

%% --- Test 1: assessQuality returns required fields -----------------------
fprintf('Test 1: assessQuality() output struct fields...\n');
result = assessQuality(dummyImg);

assert(isfield(result, 'status'),         'FAIL: result missing field "status"');
assert(isfield(result, 'sharpness'),      'FAIL: result missing field "sharpness"');
assert(isfield(result, 'brightnessMean'), 'FAIL: result missing field "brightnessMean"');
assert(isfield(result, 'brightnessStd'),  'FAIL: result missing field "brightnessStd"');

validStatuses = ["PASS", "ENHANCE", "REJECT"];
assert(ismember(result.status, validStatuses), ...
       'FAIL: result.status must be "PASS", "ENHANCE", or "REJECT"');
assert(isnumeric(result.sharpness) && isscalar(result.sharpness), ...
       'FAIL: result.sharpness must be a numeric scalar');
assert(isnumeric(result.brightnessMean) && isscalar(result.brightnessMean), ...
       'FAIL: result.brightnessMean must be a numeric scalar');
assert(isnumeric(result.brightnessStd)  && isscalar(result.brightnessStd), ...
       'FAIL: result.brightnessStd must be a numeric scalar');

fprintf('  PASS  (status="%s", sharpness=%.2f)\n\n', result.status, result.sharpness);

%% --- Test 2: enhanceFundus returns same spatial size ---------------------
fprintf('Test 2: enhanceFundus() output size...\n');
enhanced = enhanceFundus(dummyImg);

assert(isa(enhanced, 'uint8'),            'FAIL: enhanced must be uint8');
assert(isequal(size(enhanced), size(dummyImg)), ...
       'FAIL: enhanced size must match input size');

fprintf('  PASS  (size=%dx%dx%d)\n\n', size(enhanced,1), size(enhanced,2), size(enhanced,3));
%% --- Test 3: Quality gate rejects severe blur ----------------------------

severelyBlurred = imgaussfilt(dummyImg, 15);
blurResult = assessQuality(severelyBlurred);

assert(blurResult.sharpness < 5, ...
    'FAIL: severe blur should have sharpness < 5');

assert(blurResult.status == "REJECT", ...
    'FAIL: severe blur should be REJECT');

fprintf('Test 3: Severe blur correctly rejected... PASS\n\n');


%% --- Test 4: Enhancement function produces valid output -----------------

enhancedTest = enhanceFundus(dummyImg);

assert(isa(enhancedTest, 'uint8'), ...
    'FAIL: enhanced image must be uint8');

assert(isequal(size(enhancedTest), size(dummyImg)), ...
    'FAIL: enhanced image size must match input');

fprintf('Test 4: Enhancement output validation... PASS\n\n');
fprintf('=== Module 1: ALL TESTS PASSED ===\n');
