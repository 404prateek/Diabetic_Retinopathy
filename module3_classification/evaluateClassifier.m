% Calculate Sensitivity, Specificity, and Quadratic Weighted Kappa (QWK) on test set
clear; clc;

% 1. Define Paths
projectRoot = pwd;
aptosDir    = fullfile(projectRoot, 'data', 'APTOS');
testImgDir  = fullfile(aptosDir, 'test_images'); 

% 2. Auto-Detect Test CSV
candidateTestCSVs = {fullfile(aptosDir, 'test.csv'), fullfile(aptosDir, 'test-1.csv')};
testCsvPath = '';
for k = 1:numel(candidateTestCSVs)
    if isfile(candidateTestCSVs{k})
        testCsvPath = candidateTestCSVs{k}; break;
    end
end

if isempty(testCsvPath)
    error('Could not locate test CSV in %s.', aptosDir);
end

% 3. Map Test Data Safely (Case-Insensitive)
fprintf('Mapping test image files on disk...\n');
testTbl = readtable(testCsvPath);

if ismember('diagnosis', testTbl.Properties.VariableNames)
    labelsRaw = testTbl.diagnosis;
else
    labelsRaw = testTbl.dr_level;
end

dirFiles = dir(testImgDir);
dirFiles = dirFiles(~[dirFiles.isdir]);
fileMap = containers.Map('KeyType', 'char', 'ValueType', 'char');
for k = 1:numel(dirFiles)
    [~, fnameClean, fext] = fileparts(dirFiles(k).name);
    fileMap(lower(char(fnameClean))) = fullfile(testImgDir, [fnameClean, fext]);
end

idCodes = string(testTbl.id_code);
validPaths = {};
validLabels = [];

for i = 1:height(testTbl)
    [~, cleanID, ~] = fileparts(char(idCodes(i)));
    cleanKey = lower(cleanID);
    if isKey(fileMap, cleanKey)
        validPaths{end+1, 1} = fileMap(cleanKey); %#ok<AGROW>
        validLabels(end+1, 1) = labelsRaw(i); %#ok<AGROW>
    end
end

if isempty(validPaths)
    error('Zero test images matched. Please check your test folder and CSV.');
end

imdsTest = imageDatastore(validPaths, 'Labels', categorical(validLabels));
fprintf('Successfully matched %d test images.\n\n', numel(validPaths));

% 4. Load Trained Model
modelPath = fullfile(projectRoot, 'models', 'classification', 'drClassifier.mat');
fprintf('Loading trained model from %s...\n', modelPath);
loaded = load(modelPath, 'net');
net = loaded.net;

% 5. Run Inference
fprintf('Running predictions on test set (this may take a minute)...\n');
imdsTest.ReadSize = 1;
augimdsTest = augmentedImageDatastore([224 224 3], imdsTest);

predLabels = classify(net, augimdsTest);
trueLabels = imdsTest.Labels;

% 6. Calculate Confusion Matrix & Metrics
confMat = confusionmat(trueLabels, predLabels);
numClasses = 5;
totalSamples = sum(confMat, 'all');

fprintf('\n======================================================\n');
fprintf('                CLINICAL METRICS (TEST SET)             \n');
fprintf('======================================================\n');

for i = 1:numClasses
    TP = confMat(i, i);
    FN = sum(confMat(i, :)) - TP;
    FP = sum(confMat(:, i)) - TP;
    TN = totalSamples - (TP + FN + FP);
    
    sensitivity = TP / (TP + FN);
    specificity = TN / (TN + FP);
    
    if isnan(sensitivity), sensitivity = 0; end
    if isnan(specificity), specificity = 0; end
    
    fprintf('Class %d (Grade %d): Sensitivity = %5.2f%% | Specificity = %5.2f%%\n', ...
        i-1, i-1, sensitivity*100, specificity*100);
end

% 7. Compute Quadratic Weighted Kappa (QWK)
W = zeros(numClasses, numClasses);
for i = 1:numClasses
    for j = 1:numClasses
        W(i,j) = ((i - j)^2) / ((numClasses - 1)^2);
    end
end

rowSums = sum(confMat, 2);
colSums = sum(confMat, 1);
E = (rowSums * colSums) / totalSamples;

observedLoss = sum(W .* confMat, 'all');
expectedLoss = sum(W .* E, 'all');
qwk = 1 - (observedLoss / expectedLoss);

fprintf('------------------------------------------------------\n');
fprintf('Quadratic Weighted Kappa (QWK): %.4f\n', qwk);
fprintf('======================================================\n');

% 8. Plot Confusion Matrix
figure('Name', 'Confusion Matrix');
chart = confusionchart(trueLabels, predLabels);
chart.Title = sprintf('Diabetic Retinopathy Classification (QWK: %.4f)', qwk);
chart.RowSummary = 'row-normalized';
chart.ColumnSummary = 'column-normalized';