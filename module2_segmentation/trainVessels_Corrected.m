% =========================================================================
% trainVessels_Corrected.m - Modern U-Net with Custom Dice Loss
% =========================================================================
clear; clc;

% 1. Lock in the EXACT paths verified by your evaluation script
scriptFolder = fileparts(mfilename('fullpath'));
projectRoot  = fileparts(scriptFolder);
imageDir = fullfile(projectRoot, 'data', 'DRIVE', 'images');
maskDir  = fullfile(projectRoot, 'data', 'DRIVE', 'masks');

% 2. Setup Datastores (0 = background, 255 = vessel)
classNames = ["background", "vessel"];
pixelValues = [0 255];

imds = imageDatastore(imageDir);
pxds = pixelLabelDatastore(maskDir, classNames, pixelValues);

% 3. Combine and Add Augmentation
dsTrain = combine(imds, pxds);
dsTrain = transform(dsTrain, @resizeAndAugmentData);

% 4. Create Modern U-Net
imageSize = [512 512 3];
numClasses = 2;
net = unet(imageSize, numClasses);

% 5. GPU Options (20 Epochs)
options = trainingOptions("adam", ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 8, ...
    'ExecutionEnvironment', 'gpu', ...
    'Plots', "training-progress");

% 6. Train the Network using the Custom DICE Loss Function
fprintf('Starting U-Net training with CUSTOM DICE loss on GPU...\n');
trainedNet = trainnet(dsTrain, net, @diceLossFunction, options);

% 7. Save Model directly to the team's target folder
saveDir = fullfile(projectRoot, 'models', 'segmentation');
if ~exist(saveDir, 'dir'); mkdir(saveDir); end
savePath = fullfile(saveDir, 'vessel_unet.mat');

% Rename to 'net' to match the team contract
net = trainedNet; 
save(savePath, "net");
fprintf('Training complete! Model successfully saved to %s\n', savePath);

% =========================================================================
% Helper Function 1: Resizing and Data Augmentation
% =========================================================================
function data = resizeAndAugmentData(data)
img = imresize(data{1}, [512 512]);
mask = imresize(data{2}, [512 512], 'nearest');

% 50% chance to flip horizontally/vertically
if rand > 0.5; img = fliplr(img); mask = fliplr(mask); end
if rand > 0.5; img = flipud(img); mask = flipud(mask); end

data{1} = img;
data{2} = mask;
end

% =========================================================================
% Helper Function 2: Custom Dice Loss Math for trainnet
% =========================================================================
function loss = diceLossFunction(Y, T)
% Y: Network Predictions, T: Ground Truth Targets
% Calculate intersection and union over spatial and batch dimensions (1, 2, 4)
intersection = sum(Y .* T, [1 2 4]);
denominator = sum(Y, [1 2 4]) + sum(T, [1 2 4]);

% Add a tiny smoothing factor (1e-5) to prevent division by zero
diceScores = (2 * intersection + 1e-5) ./ (denominator + 1e-5);

% Loss is 1 minus the average Dice score across classes
loss = mean(1 - diceScores, "all");
end