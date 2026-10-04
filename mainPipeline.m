function result = mainPipeline(I)
% =========================================================================
% mainPipeline  -  End-to-end DR screening pipeline for a single image
% =========================================================================
% Purpose : Orchestrate all four modules (Quality -> Segmentation ->
%           Classification -> Explainability) for a single fundus image.
%           Short-circuits to REJECT after Module 1 if quality.status is
%           "REJECT", skipping expensive inference steps.
%           Returns a single struct with all intermediate and final results.
%
% Owner   : Member 4  (integration, mainPipeline.m)
%
% Inputs  :
%   I       - (H x W x 3 uint8)  Raw RGB fundus image
%
% Outputs :
%   result  - struct with fields:
%     .image          (H x W x 3 uint8)  The (possibly enhanced) image used
%     .quality        struct              Output of assessQuality()
%     .segmentation   struct | []        Output of runSegmentation()
%                                        ([] if quality.status == "REJECT")
%     .classification (string) | ""      Output of classifyDR()
%                                        ("" if quality.status == "REJECT")
%     .confidence     (double) | NaN     Softmax confidence
%                                        (NaN if quality.status == "REJECT")
%     .heatmap        (H x W x 3) | []  Grad-CAM overlay
%                                        ([] if quality.status == "REJECT")
%     .status         (string)           "COMPLETE" | "REJECT" | "ERROR"
%
% Usage example:
%   I      = imread('data/APTOS/sample.png');
%   result = mainPipeline(I);
%   if result.status == "COMPLETE"
%       imshow(result.heatmap);
%   end
%
% Pipeline flow:
%   [M1] assessQuality  -> REJECT? -> short-circuit, return result
%                       -> ENHANCE? -> enhanceFundus -> continue
%   [M2] runSegmentation
%   [M3] classifyDR
%   [M4] explainPrediction
%        generateReport
% =========================================================================

try
    % ------------------------------------------------------------------
    % Step 1 – Quality Assessment (Module 1)
    % ------------------------------------------------------------------
    fprintf('[M1] Assessing image quality...\n');
    quality = assessQuality(I);
    result.quality = quality;
    fprintf('     Status: %s  (sharpness=%.2f, brightness=%.1f)\n', ...
            quality.status, quality.sharpness, quality.brightnessMean);

    % ------------------------------------------------------------------
    % Step 2 – Short-circuit on REJECT
    % ------------------------------------------------------------------
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

    % ------------------------------------------------------------------
    % Step 3 – Enhancement if needed (Module 1)
    % ------------------------------------------------------------------
    if quality.status == "ENHANCE"
        fprintf('[M1] Borderline quality - applying CLAHE enhancement...\n');
        I = enhanceFundus(I);
        % Re-assess to confirm enhancement helped
        postEnhQuality = assessQuality(I);
        fprintf('     Post-enhancement sharpness=%.2f (status=%s)\n', ...
                postEnhQuality.sharpness, postEnhQuality.status);
    end
    result.image = I;

    % ------------------------------------------------------------------
    % Step 4 – Segmentation (Module 2)
    % ------------------------------------------------------------------
    fprintf('[M2] Running vessel & lesion segmentation...\n');
    result.segmentation = runSegmentation(I);
    fprintf('     Lesion count: %d\n', result.segmentation.lesionCount);

    % ------------------------------------------------------------------
    % Step 5 – Classification (Module 3)
    % ------------------------------------------------------------------
    fprintf('[M3] Classifying DR severity...\n');
    [result.classification, result.confidence] = classifyDR(I);
    fprintf('     Grade: %s  (confidence=%.1f%%)\n', ...
            result.classification, result.confidence * 100);

    % ------------------------------------------------------------------
    % Step 6 – Explainability + Report (Module 4)
    % ------------------------------------------------------------------
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
