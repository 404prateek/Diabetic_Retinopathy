function metrics = evaluateClassifier(net, imdsTest)
% =========================================================================
% evaluateClassifier  -  Evaluate DR classification performance on a test set
% =========================================================================

    % 1. Resize test images for network input (224x224 for EfficientNet-B0)
    augimdsTest = augmentedImageDatastore([224 224], imdsTest);

    % 2. Predict probability scores and obtain class predictions
    scores = predict(net, augimdsTest);
    [~, maxIdx] = max(scores, [], 2);

    % Extract actual ground-truth categorical labels
    actualLabels = imdsTest.Labels;
    classes = categories(actualLabels);
    numClasses = numel(classes);

    % Create predicted categorical labels matching ground-truth categories
    predCategorical = categorical(classes(maxIdx), classes);

    % 3. Confusion Matrix (rows = true, cols = pred)
    cm = confusionmat(actualLabels, predCategorical);
    totalSamples = sum(cm(:));

    % 4. Top-1 Overall Accuracy
    accuracy = sum(diag(cm)) / totalSamples;

    % 5. Compute Per-Class Precision, Recall (Sensitivity), and F1 Score
    precision = zeros(1, numClasses);
    recall    = zeros(1, numClasses);
    f1Score   = zeros(1, numClasses);

    for i = 1:numClasses
        tp = cm(i, i);
        fp = sum(cm(:, i)) - tp;
        fn = sum(cm(i, :)) - tp;

        if (tp + fp) > 0
            precision(i) = tp / (tp + fp);
        end
        if (tp + fn) > 0
            recall(i) = tp / (tp + fn);
        end
        if (precision(i) + recall(i)) > 0
            f1Score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
        end
    end

    % 6. Compute Quadratic Weighted Kappa (QWK)
    actualNum = double(actualLabels);
    predNum   = double(predCategorical);

    % Weight matrix calculation
    w = zeros(numClasses, numClasses);
    for i = 1:numClasses
        for j = 1:numClasses
            w(i,j) = ((i - j)^2) / ((numClasses - 1)^2);
        end
    end

    % Expected matrix calculation
    histActual  = sum(cm, 2);
    histPred    = sum(cm, 1);
    expectedMat = (histActual * histPred) / totalSamples;

    obsErr = sum(sum(w .* cm)) / totalSamples;
    expErr = sum(sum(w .* expectedMat)) / totalSamples;

    if expErr == 0
        kappaScore = 1.0;
    else
        kappaScore = 1.0 - (obsErr / expErr);
    end

    % 7. Compute One-vs-Rest ROC AUC per class
    rocAUC = zeros(1, numClasses);
    for i = 1:numClasses
        binaryTrue = (actualNum == i);
        classScores = scores(:, i);

        try
            [~, ~, ~, auc] = perfcurve(binaryTrue, classScores, true);
            rocAUC(i) = auc;
        catch
            rocAUC(i) = NaN;
        end
    end

    % 8. ICDR Class Names Mapping
    icdrMap = containers.Map({'0','1','2','3','4'}, ...
        {'No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'});

    classNames = cell(1, numClasses);
    for i = 1:numClasses
        cName = char(classes(i));
        if isKey(icdrMap, cName)
            classNames{i} = icdrMap(cName);
        else
            classNames{i} = cName;
        end
    end

    % 9. Build and Return Output Struct
    metrics = struct();
    metrics.accuracy        = accuracy;
    metrics.confusionMatrix = cm;
    metrics.classNames      = classNames;
    metrics.precision       = precision;
    metrics.recall          = recall;
    metrics.f1Score        = f1Score;
    metrics.kappaScore     = kappaScore;
    metrics.rocAUC         = rocAUC;

end