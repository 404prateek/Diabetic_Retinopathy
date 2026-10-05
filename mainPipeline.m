function result = mainPipeline(I)
% End-to-end DR screening pipeline: Quality -> Segmentation -> Classification -> Grad-CAM.
% Short-circuits to REJECT if image quality fails initial gate.

try
    % Quality Assessment
    fprintf('[M1] Assessing image quality...\n');
    quality = assessQuality(I);
    result.quality = quality;
    fprintf('     Status: %s  (sharpness=%.2f, brightness=%.1f)\n', ...
            quality.status, quality.sharpness, quality.brightnessMean);

    % Short-circuit on REJECT
    if quality.status == "REJECT"
        fprintf('[M1] Image REJECTED - skipping further processing.\n');
        result.image          = I;
        result.segmentation   = [];
        result.classification = "";
        result.confidence     = NaN;
        result.heatmap        = [];
        result.status         = "REJECT";
        return;
    end

    % Enhancement if borderline
    if quality.status == "ENHANCE"
        fprintf('[M1] Borderline quality - applying CLAHE enhancement...\n');
        I = enhanceFundus(I);
        postEnhQuality = assessQuality(I);
        fprintf('     Post-enhancement sharpness=%.2f (status=%s)\n', ...
                postEnhQuality.sharpness, postEnhQuality.status);
    end
    result.image = I;

    % Segmentation
    fprintf('[M2] Running vessel & lesion segmentation...\n');
    result.segmentation = runSegmentation(I);
    fprintf('     Lesion count: %d\n', result.segmentation.lesionCount);

    % Classification
    fprintf('[M3] Classifying DR severity...\n');
    [result.classification, result.confidence] = classifyDR(I);
    fprintf('     Grade: %s  (confidence=%.1f%%)\n', ...
            result.classification, result.confidence * 100);

    % Explainability & Report
    fprintf('[M4] Generating Grad-CAM heatmap...\n');
    result.heatmap = explainPrediction(I, result.classification);

    fprintf('[M4] Writing PDF report...\n');
    generateReport(I, quality, result.segmentation, ...
                   result.classification, result.confidence, result.heatmap);

    result.status = "COMPLETE";
    fprintf('[OK] Pipeline complete.\n');

catch ME
    warning('mainPipeline:error', 'Pipeline failed: %s', ME.message);
    % Fill in safe defaults so caller can always inspect result struct
    if ~isfield(result, 'image'),          result.image          = I;    end
    if ~isfield(result, 'quality'),        result.quality        = struct('status','ERROR','sharpness',NaN,'brightnessMean',NaN,'brightnessStd',NaN); end
    if ~isfield(result, 'segmentation'),   result.segmentation   = [];   end
    if ~isfield(result, 'classification'), result.classification = "";   end
    if ~isfield(result, 'confidence'),     result.confidence     = NaN;  end
    if ~isfield(result, 'heatmap'),        result.heatmap        = [];   end
    result.status = "ERROR";
    rethrow(ME);
end

end
