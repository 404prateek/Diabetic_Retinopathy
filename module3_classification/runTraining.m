% Master script to map APTOS dataset and fine-tune EfficientNet-B0
clear; clc;
% 1. Define paths
projectRoot = pwd;
aptosDir    = fullfile(projectRoot, 'data', 'APTOS');
trainImgDir = fullfile(aptosDir, 'train_images'); % Updated to match your folder
valImgDir   = fullfile(aptosDir, 'val_images');   % Already matches your folder
if ~exist(valImgDir, 'dir')
    valImgDir = fullfile(aptosDir, 'val'); % Fallback folder name
end

% 2. Auto-Detect Training CSV Name
candidateTrainCSVs = { ...
    fullfile(aptosDir, 'train-1.csv'), ...
    fullfile(aptosDir, 'train.csv'), ...
    fullfile(aptosDir, 'train_1.csv') ...
};

trainCsvPath = '';
for k = 1:numel(candidateTrainCSVs)
    if isfile(candidateTrainCSVs{k})
        trainCsvPath = candidateTrainCSVs{k};
        break;
    end
end

if isempty(trainCsvPath)
    error('Could not locate training CSV in %s.', aptosDir);
end

% 3. Auto-Detect Validation CSV Name
candidateValCSVs = { ...
    fullfile(aptosDir, 'valid.csv'), ...
    fullfile(aptosDir, 'val.csv'), ...
    fullfile(aptosDir, 'validation.csv') ...
};

valCsvPath = '';
for k = 1:numel(candidateValCSVs)
    if isfile(candidateValCSVs{k})
        valCsvPath = candidateValCSVs{k};
        break;
    end
end

if isempty(valCsvPath)
    error('Could not locate validation CSV in %s.', aptosDir);
end

fprintf('Using Training CSV:   %s\n', trainCsvPath);
fprintf('Using Validation CSV: %s\n\n', valCsvPath);

% 4. Dynamically build Training Datastore
fprintf('Mapping training image files on disk...\n');
trainTbl = readtable(trainCsvPath);
[imdsTrain, numTrain] = buildDatastoreFromTable(trainImgDir, trainTbl);
fprintf('Successfully matched %d training images.\n\n', numTrain);

% 5. Dynamically build Validation Datastore
fprintf('Mapping validation image files on disk...\n');
valTbl = readtable(valCsvPath);
[imdsValidation, numVal] = buildDatastoreFromTable(valImgDir, valTbl);
fprintf('Successfully matched %d validation images.\n\n', numVal);

% 6. Call trainClassifier function
fprintf('Initiating EfficientNet-B0 training via trainClassifier.m...\n');
net = trainClassifier(imdsTrain, imdsValidation);

fprintf('Training complete and model successfully saved!\n');


% File and label datastore resolver
function [imds, matchedCount] = buildDatastoreFromTable(imgFolder, tbl)
    % 1. Extract diagnosis or dr_level column
    if ismember('diagnosis', tbl.Properties.VariableNames)
        labelsRaw = tbl.diagnosis;
    elseif ismember('dr_level', tbl.Properties.VariableNames)
        labelsRaw = tbl.dr_level;
    else
        error('CSV table must contain a "diagnosis" or "dr_level" column.');
    end
    
    % 2. Catalog all actual image files on disk
    dirFiles = dir(imgFolder);
    dirFiles = dirFiles(~[dirFiles.isdir]);
    
    if isempty(dirFiles)
        error('Directory "%s" is empty or contains no files.', imgFolder);
    end
    
    fileMap = containers.Map('KeyType', 'char', 'ValueType', 'char');
    for k = 1:numel(dirFiles)
        [~, fnameClean, fext] = fileparts(dirFiles(k).name);
        % Store lowercase key for case-insensitive matching
        fileMap(lower(char(fnameClean))) = fullfile(imgFolder, [fnameClean, fext]);
    end
    
    % 3. Convert CSV id_code column to string array properly
    idCodes = string(tbl.id_code);
    idCodes = strtrim(idCodes);
    
    validPaths = {};
    validLabels = [];
    
    for i = 1:height(tbl)
        % Strip extension if CSV ID contains one (e.g., '10_left.png')
        [~, cleanID, ~] = fileparts(char(idCodes(i)));
        cleanKey = lower(cleanID);
        
        if isKey(fileMap, cleanKey)
            validPaths{end+1, 1} = fileMap(cleanKey); %#ok<AGROW>
            validLabels(end+1, 1) = labelsRaw(i); %#ok<AGROW>
        end
    end
    
    % 4. Print diagnostic details if zero matches found
    if isempty(validPaths)
        sampleCSV = idCodes(1:min(3, height(tbl)));
        sampleDisk = keys(fileMap);
        sampleDisk = sampleDisk(1:min(3, numel(sampleDisk)));
        
        fprintf('\n--- MAPPING DIAGNOSTIC ERROR ---\n');
        fprintf('Target Image Folder:  %s\n', imgFolder);
        fprintf('Sample CSV id_codes:  %s\n', strjoin(sampleCSV, ', '));
        fprintf('Sample Disk filenames: %s\n', strjoin(string(sampleDisk), ', '));
        fprintf('--------------------------------\n');
        error('Zero image files in %s matched the IDs in the CSV table.', imgFolder);
    end
    
    % Convert labels to categorical
    catLabels = categorical(validLabels);
    
    % Create imageDatastore
    imds = imageDatastore(validPaths, 'Labels', catLabels);
    matchedCount = numel(validPaths);
end