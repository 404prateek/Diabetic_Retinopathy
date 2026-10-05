function generateReport(I, quality, segmentation, classification, confidence, heatmap)
% Export a structured PDF screening report compiling pipeline outputs.

% Ensure results directory exists
if ~exist('results', 'dir')
    mkdir('results');
end

% Timestamp
ts       = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
baseName = fullfile('results', [ts '_DRReport']);

% Build report figure
fig = figure('Name',            'DR Screening Report', ...
             'NumberTitle',     'off', ...
             'Visible',         'off', ...
             'PaperUnits',      'inches', ...
             'PaperSize',       [8.5, 11], ...
             'PaperPosition',   [0.25, 0.25, 8, 10.5], ...
             'PaperOrientation','portrait', ...
             'Color',           'w');

% -- Row 1: Original | Grad-CAM --
subplot(3, 2, 1);
imshow(I);
title('Original Fundus Image', 'FontWeight', 'bold');

subplot(3, 2, 2);
if ~isempty(heatmap)
    imshow(heatmap);
    title(sprintf('Grad-CAM Saliency (%s)', classification), 'FontWeight', 'bold');
else
    axis off;
    text(0.5, 0.5, 'Heatmap unavailable', 'HorizontalAlignment', 'center', 'Units', 'normalized');
    title('Grad-CAM', 'FontWeight', 'bold');
end

% -- Row 2: Vessel mask | Lesion mask --
if ~isempty(segmentation)
    subplot(3, 2, 3);
    imshow(segmentation.vesselMask);
    title('Vessel Mask', 'FontWeight', 'bold');

    subplot(3, 2, 4);
    imshow(segmentation.lesionMask);
    title(sprintf('Lesion Mask (%d regions)', segmentation.lesionCount), 'FontWeight', 'bold');
else
    subplot(3, 2, 3); axis off;
    text(0.5, 0.5, 'Segmentation skipped', 'HorizontalAlignment', 'center', 'Units', 'normalized');
    subplot(3, 2, 4); axis off;
    text(0.5, 0.5, '(rejected at quality gate)', 'HorizontalAlignment', 'center', 'Units', 'normalized');
end

% -- Row 3: Metrics text block --
ax = subplot(3, 2, [5 6]);
axis(ax, 'off');

% Referral recommendation map
referralMap = containers.Map( ...
    {'No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'}, ...
    {'Routine – rescreen in 2 years', ...
     'Lifestyle counselling – rescreen in 1 year', ...
     'Refer within 4 weeks', ...
     'Refer within 1 week (URGENT)', ...
     'SAME-DAY URGENT REFERRAL'});

classStr = char(classification);
if isKey(referralMap, classStr)
    referralText = referralMap(classStr);
else
    referralText = 'Consult ophthalmologist';
end

% Build metric lines
qLine1 = sprintf('Quality Status  : %s', quality.status);
qLine2 = sprintf('Sharpness       : %.2f', quality.sharpness);
qLine3 = sprintf('Brightness Mean : %.1f  /  Std : %.1f', ...
                  quality.brightnessMean, quality.brightnessStd);

if isnan(confidence)
    cLine1 = sprintf('DR Grade        : N/A (image rejected)');
    cLine2 = 'Confidence      : N/A';
    cLine3 = 'Recommendation  : N/A';
else
    cLine1 = sprintf('DR Grade        : %s', classStr);
    cLine2 = sprintf('Confidence      : %.1f%%', confidence * 100);
    cLine3 = sprintf('Recommendation  : %s', referralText);
end

if ~isempty(segmentation)
    sLine1 = sprintf('Lesion Regions  : %d', segmentation.lesionCount);
    if ~isnan(segmentation.dice)
        sLine2 = sprintf('Dice / IoU      : %.4f / %.4f', ...
                          segmentation.dice, segmentation.iou);
    else
        sLine2 = 'Dice / IoU      : N/A (no ground-truth)';
    end
else
    sLine1 = 'Segmentation    : skipped (quality REJECT)';
    sLine2 = '';
end

disclaimer = ['DISCLAIMER: Research prototype only. ' ...
              'Not a certified medical device. ' ...
              'All outputs must be reviewed by a qualified ophthalmologist.'];

reportText = sprintf( ...
    ['DR-Screen  |  Report  |  %s\n' ...
     '____________________________________________\n\n' ...
     'QUALITY ASSESSMENT\n  %s\n  %s\n  %s\n\n' ...
     'CLASSIFICATION\n  %s\n  %s\n  %s\n\n' ...
     'SEGMENTATION\n  %s\n  %s\n\n' ...
     '____________________________________________\n' ...
     '%s'], ...
    char(datetime('now', 'Format', 'dd-MMM-yyyy HH:mm')), ...
    qLine1, qLine2, qLine3, ...
    cLine1, cLine2, cLine3, ...
    sLine1, sLine2, ...
    disclaimer);

text(ax, 0.02, 0.98, reportText, ...
    'Units',           'normalized', ...
    'VerticalAlignment','top', ...
    'FontName',        'Courier', ...
    'FontSize',        7.5, ...
    'Interpreter',     'none');

sgtitle('DR-Screen  |  Explainable AI Screening Report  |  PS ID 26038', ...
        'FontWeight', 'bold', 'FontSize', 11);

% Export to PDF (PNG fallback)
pdfFile = [baseName '.pdf'];
pngFile = [baseName '.png'];

try
    print(fig, pdfFile, '-dpdf', '-fillpage');
    fprintf('[M4] Report saved: %s\n', pdfFile);
catch ME
    warning('generateReport:pdfFailed', ...
            'PDF export failed (%s). Saving PNG instead.', ME.message);
    saveas(fig, pngFile);
    fprintf('[M4] Report saved (PNG fallback): %s\n', pngFile);
end

close(fig);

end  % generateReport
