% Unit tests for Module 2: Segmentation

fprintf('Running Module 2 Tests: Segmentation...\n');

%% Create a dummy 256x256 synthetic fundus image
[xx, yy] = meshgrid(1:256, 1:256);
mask      = sqrt((xx-128).^2 + (yy-128).^2) < 100;
dummyImg  = uint8(zeros(256,256,3));
dummyImg(:,:,2) = uint8(mask * 120);

% Test 1: runSegmentation returns required fields
fprintf('Test 1: runSegmentation() output struct fields...\n');
seg = runSegmentation(dummyImg);

assert(isfield(seg, 'vesselMask'),   'FAIL: seg missing field "vesselMask"');
assert(isfield(seg, 'lesionMask'),   'FAIL: seg missing field "lesionMask"');
assert(isfield(seg, 'lesionCount'),  'FAIL: seg missing field "lesionCount"');
assert(isfield(seg, 'dice'),         'FAIL: seg missing field "dice"');
assert(isfield(seg, 'iou'),          'FAIL: seg missing field "iou"');

assert(islogical(seg.vesselMask) && isequal(size(seg.vesselMask), [256 256]), ...
       'FAIL: vesselMask must be 256x256 logical');
assert(islogical(seg.lesionMask) && isequal(size(seg.lesionMask), [256 256]), ...
       'FAIL: lesionMask must be 256x256 logical');
assert(isnumeric(seg.lesionCount) && isscalar(seg.lesionCount), ...
       'FAIL: lesionCount must be a numeric scalar');

fprintf('  PASS  (lesionCount=%d)\n\n', seg.lesionCount);

fprintf('Module 2: All tests passed.\n');
