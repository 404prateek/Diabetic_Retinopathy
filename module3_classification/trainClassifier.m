function net = trainClassifier(imdsTrain, imdsValidation)
% =========================================================================
% trainClassifier  -  Fine-tune EfficientNet-B0 for DR severity classification
% =========================================================================
% Inputs  : imdsTrain      - imageDatastore of training images with Labels
%           imdsValidation - imageDatastore of validation images with Labels
% Outputs : net            - Trained network saved to models/classification/drClassifier.mat
%
% BugFix (2024-10): Fixed layer discovery from broken isprop('LearnableParameters')
%   to isa() type-checks.  Added explicit class names to classificationLayer.
%   Note: trainNetwork is the correct API for lgraph/DAGNetwork (R2023b).
%         Use trainnet() only when migrating to dlnetwork.
% =========================================================================

% 1. Load pre-trained EfficientNet-B0 base architecture
fprintf('[trainClassifier] Loading EfficientNet-B0...\n');
baseNet = efficientnetb0;
lgraph  = layerGraph(baseNet);
layers  = lgraph.Layers;

% -----------------------------------------------------------------------
% 2. Locate layers to replace using isa() type checks (robust to all nets)
% -----------------------------------------------------------------------
% Find the last fully-connected layer (the classification head)
isFCLayer    = arrayfun(@(l) isa(l, 'nnet.cnn.layer.FullyConnectedLayer'), layers);
fcLayerNames = {layers(isFCLayer).Name};
lastFCName   = fcLayerNames{end};

% Find the classification output layer
isOutLayer   = arrayfun(@(l) isa(l, 'nnet.cnn.layer.ClassificationOutputLayer'), layers);
outLayerNames = {layers(isOutLayer).Name};
lastOutName  = outLayerNames{end};

fprintf('[trainClassifier] Replacing "%s" and "%s"...\n', lastFCName, lastOutName);

% -----------------------------------------------------------------------
% 3. Build replacement layers for 5-class ICDR grading
% -----------------------------------------------------------------------
numClasses  = 5;   % Grades 0..4
classNames  = {'0','1','2','3','4'};   % Must match sub-folder names in data/

newFc = fullyConnectedLayer(numClasses, ...
    'Name',                  'dr_fc', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor',   10);

% Explicit class names avoid the deprecated auto-inference path in R2023b
newClassLayer = classificationLayer( ...
    'Name',    'dr_classoutput', ...
    'Classes', classNames);

lgraph = replaceLayer(lgraph, lastFCName,  newFc);
lgraph = replaceLayer(lgraph, lastOutName, newClassLayer);

% -----------------------------------------------------------------------
% 4. Data augmentation pipeline
% -----------------------------------------------------------------------
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandRotation',    [-20 20], ...
    'RandXScale',      [0.9 1.1], ...
    'RandYScale',      [0.9 1.1]);

% EfficientNet-B0 expects 224x224 RGB
augimdsTrain      = augmentedImageDatastore([224 224 3], imdsTrain,      'DataAugmentation', imageAugmenter);
augimdsValidation = augmentedImageDatastore([224 224 3], imdsValidation);

% -----------------------------------------------------------------------
% 5. Training options
% -----------------------------------------------------------------------
options = trainingOptions('adam', ...
    'MiniBatchSize',       16, ...
    'MaxEpochs',           10, ...
    'InitialLearnRate',    1e-4, ...
    'LearnRateSchedule',   'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 5, ...
    'L2Regularization',    1e-4, ...
    'ValidationData',      augimdsValidation, ...
    'ValidationFrequency', 30, ...
    'Shuffle',             'every-epoch', ...
    'Verbose',             true, ...
    'Plots',               'training-progress', ...
    'OutputNetwork',       'best-validation-loss');   % Save best checkpoint

% -----------------------------------------------------------------------
% 6. Train
% -----------------------------------------------------------------------
fprintf('[trainClassifier] Starting training...\n');
net = trainNetwork(augimdsTrain, lgraph, options);  %#ok<TRAIN>
% Note: trainNetwork is the stable API for DAGNetwork in R2023b.
% Use trainnet() with dlnetwork only if you migrate to the new DL API.

% -----------------------------------------------------------------------
% 7. Save to models/classification/drClassifier.mat
% -----------------------------------------------------------------------
if ~exist(fullfile('models', 'classification'), 'dir')
    mkdir(fullfile('models', 'classification'));
end

savePath = fullfile('models', 'classification', 'drClassifier.mat');
save(savePath, 'net');
fprintf('[trainClassifier] Saved trained classifier to: %s\n', savePath);

end  % trainClassifier
