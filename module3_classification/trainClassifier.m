function net = trainClassifier(imdsTrain, imdsValidation)
% 1. Load pre-trained EfficientNet-b0 base architecture
baseNet = efficientnetb0;
lgraph = layerGraph(baseNet);

% 2. Dynamically locate and replace the final classification layers for 5 classes
layers = lgraph.Layers;
isLearnable = arrayfun(@(l) isprop(l, 'Weights') || isprop(l, 'LearnableParameters'), layers);
learnableLayers = layers(isLearnable);
lastLearnableLayer = learnableLayers(end);
classLayer = layers(end);

numClasses = 5; % Grades 0 to 4 (ICDR scale)
newFc = fullyConnectedLayer(numClasses, 'Name', 'new_fc', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);
newClassLayer = classificationLayer('Name', 'new_classoutput');

lgraph = replaceLayer(lgraph, lastLearnableLayer.Name, newFc);
lgraph = replaceLayer(lgraph, classLayer.Name, newClassLayer);

% 3. Configure Data Augmentation (Flips and Rotations)
imageAugmenter = imageDataAugmenter(...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandRotation', [-20 20]);

% Resize images to 224x224 (EfficientNet-B0 input size requirement)
augimdsTrain = augmentedImageDatastore([224 224], imdsTrain, ...
    'DataAugmentation', imageAugmenter);
augimdsValidation = augmentedImageDatastore([224 224], imdsValidation);

% 4. Set Training Options
options = trainingOptions('adam', ...
    'MiniBatchSize', 16, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-4, ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% 5. Train Network
net = trainNetwork(augimdsTrain, lgraph, options);

% 6. Save Model to models/classification/drClassifier.mat as per repo spec
if ~exist('models/classification', 'dir')
    mkdir('models/classification');
end

savePath = fullfile('models', 'classification', 'drClassifier.mat');
save(savePath, 'net');
fprintf('Successfully saved trained classifier to %s\n', savePath);
end