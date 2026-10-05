function imds = loadDataset(dataPath)
% Load a fundus image dataset from disk into an imageDatastore.

% Validate that the dataset path exists
if ~isfolder(dataPath)
    error('loadDataset:InvalidPath', ...
        'Dataset folder does not exist: %s', dataPath);
end

% Create image datastore
imds = imageDatastore(dataPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'FileExtensions', {'.png', '.jpg', '.jpeg', '.tif'});

end
