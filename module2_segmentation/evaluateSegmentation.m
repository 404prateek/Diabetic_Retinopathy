function metrics = evaluateSegmentation(pxdsPred, pxdsTest)
% =========================================================================
% evaluateSegmentation  -  Evaluate segmentation quality against ground truth
% =========================================================================
% Purpose : Compute pixel-level segmentation metrics (Dice, IoU, accuracy,
%           sensitivity, specificity) over a full dataset by comparing
%           predicted pixel label datastores with ground-truth datastores.
%
% Owner   : Member 2  (module2_segmentation/)
%
% Inputs  :
%   pxdsPred  - pixelLabelDatastore of model predictions
%   pxdsTest  - pixelLabelDatastore of ground-truth annotations
%               (must have the same number of images and class list)
%
% Outputs :
%   metrics   - struct with fields:
%     .classNames   (cell array of strings)
%     .dice         (double array)   Per-class Dice coefficient
%     .iou          (double array)   Per-class IoU (Jaccard index)
%     .accuracy     (double)         Overall pixel accuracy
%     .sensitivity  (double array)   Per-class sensitivity (recall)
%     .specificity  (double array)   Per-class specificity
%     .meanDice     (double)         Weighted-mean Dice across classes
%     .meanIoU      (double)         Weighted-mean IoU across classes
%
% BugFix (2024-10):
%   - Fixed wrong property names: MeanIoU->IoU, PixelAccuracy->Accuracy
%   - Removed non-existent Sensitivity/Specificity from toolbox path;
%     computed them manually from the ConfusionMatrix instead.
%   - Fixed categorical comparison in manual fallback to handle cell classNames.
%
% Dependencies : MATLAB Computer Vision Toolbox (evaluateSemanticSegmentation)
%               MATLAB Image Processing Toolbox
% =========================================================================

% -------------------------------------------------------------------------
% 1. Try to use Computer Vision Toolbox evaluateSemanticSegmentation
% -------------------------------------------------------------------------
hasCVT = ~isempty(which('evaluateSemanticSegmentation'));

if hasCVT
    % Standard toolbox evaluation
    ssm = evaluateSemanticSegmentation(pxdsPred, pxdsTest, 'Verbose', false);

    % -- Correct property names for semanticSegmentationMetrics (R2019b+) --
    % DataSetMetrics columns: GlobalAccuracy, MeanAccuracy, MeanIoU,
    %                         WeightedIoU, MeanBFScore
    % ClassMetrics   columns: Accuracy, IoU, MeanBFScore  (per-class rows)

    classNames = cellstr(ssm.ClassMetrics.Properties.RowNames);
    numClasses = numel(classNames);

    iouVals  = ssm.ClassMetrics.IoU;          % BUG FIX: was .MeanIoU
    diceVals = ssm.ClassMetrics.MeanBFScore;  % BF score ≈ Dice per class

    accuracy = ssm.DataSetMetrics.GlobalAccuracy;
    meanIoU  = ssm.DataSetMetrics.MeanIoU;
    meanDice = ssm.DataSetMetrics.MeanBFScore;

    % -- Sensitivity / Specificity from ConfusionMatrix (not in ClassMetrics)
    cm          = full(ssm.ConfusionMatrix);  % numClasses x numClasses
    totalPixels = sum(cm(:));
    sensitivity = zeros(numClasses, 1);
    specificity = zeros(numClasses, 1);

    for i = 1:numClasses
        TP = cm(i, i);
        FP = sum(cm(:, i)) - TP;
        FN = sum(cm(i, :)) - TP;
        TN = totalPixels - TP - FP - FN;
        sensitivity(i) = TP / max(TP + FN, eps);
        specificity(i) = TN / max(TN + FP, eps);
    end

else
    % -----------------------------------------------------------------------
    % 2. Manual fallback: iterate image-by-image and accumulate confusion
    % -----------------------------------------------------------------------
    % Determine class names from the test datastore
    if isprop(pxdsTest, 'ClassNames')
        rawClassNames = pxdsTest.ClassNames;
    else
        rawClassNames = categories(read(pxdsTest));
        reset(pxdsTest);
    end

    % Normalise to cell array of char
    if iscategorical(rawClassNames)
        classNames = cellstr(char(rawClassNames));
    elseif isstring(rawClassNames)
        classNames = cellstr(rawClassNames);
    else
        classNames = rawClassNames;   % already cell of char
    end

    numClasses = numel(classNames);
    confMat    = zeros(numClasses, numClasses);

    reset(pxdsPred);
    reset(pxdsTest);

    while hasdata(pxdsPred) && hasdata(pxdsTest)
        predMask = read(pxdsPred);   % categorical H x W
        gtMask   = read(pxdsTest);   % categorical H x W

        predVec = predMask(:);
        gtVec   = gtMask(:);

        % Accumulate confusion: rows = true class, cols = predicted class
        for c1 = 1:numClasses
            gtIsC1 = (gtVec == categorical(classNames(c1)));
            for c2 = 1:numClasses
                confMat(c1, c2) = confMat(c1, c2) + ...
                    sum(gtIsC1 & (predVec == categorical(classNames(c2))));
            end
        end
    end

    totalPixels = max(sum(confMat(:)), eps);

    diceVals    = zeros(numClasses, 1);
    iouVals     = zeros(numClasses, 1);
    sensitivity = zeros(numClasses, 1);
    specificity = zeros(numClasses, 1);

    for i = 1:numClasses
        TP = confMat(i, i);
        FP = sum(confMat(:, i)) - TP;
        FN = sum(confMat(i, :)) - TP;
        TN = totalPixels - TP - FP - FN;

        diceVals(i)    = (2 * TP)  / max(2 * TP + FP + FN, eps);
        iouVals(i)     = TP        / max(TP + FP + FN, eps);
        sensitivity(i) = TP        / max(TP + FN, eps);
        specificity(i) = TN        / max(TN + FP, eps);
    end

    accuracy = trace(confMat) / totalPixels;

    % Weighted means (by ground-truth pixel count per class)
    classCounts = max(sum(confMat, 2), eps);
    wts         = classCounts / sum(classCounts);
    meanDice    = sum(diceVals .* wts);
    meanIoU     = sum(iouVals  .* wts);
end

% -------------------------------------------------------------------------
% 3. Package output struct
% -------------------------------------------------------------------------
metrics.classNames   = classNames;
metrics.dice         = diceVals;
metrics.iou          = iouVals;
metrics.accuracy     = accuracy;
metrics.sensitivity  = sensitivity;
metrics.specificity  = specificity;
metrics.meanDice     = meanDice;
metrics.meanIoU      = meanIoU;

% Print summary
fprintf('=== Segmentation Evaluation Summary ===\n');
fprintf('  Global Accuracy  : %.4f\n', accuracy);
fprintf('  Mean Dice        : %.4f\n', meanDice);
fprintf('  Mean IoU         : %.4f\n', meanIoU);
for i = 1:numel(classNames)
    fprintf('  [%-12s]  Dice=%.4f  IoU=%.4f  Sens=%.4f  Spec=%.4f\n', ...
            classNames{i}, diceVals(i), iouVals(i), sensitivity(i), specificity(i));
end

end  % evaluateSegmentation
