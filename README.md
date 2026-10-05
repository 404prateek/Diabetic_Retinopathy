<div align="center">

# DR-Screen
### Explainable AI for Diabetic Retinopathy Screening in Rural India

[![SIH 2024](https://img.shields.io/badge/Smart%20India%20Hackathon-2024-orange?style=for-the-badge)](https://www.sih.gov.in/)
[![PS ID](https://img.shields.io/badge/PS%20ID-26038-blue?style=for-the-badge)](https://www.sih.gov.in/)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2023b%2B-red?style=for-the-badge&logo=mathworks)](https://www.mathworks.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Pipeline-Fully%20Integrated-brightgreen?style=for-the-badge)]()

*Automated fundus screening - Grad-CAM explainability - Offline-capable - Rural clinic ready*

</div>

---

## The Problem

India has **over 77 million diabetic patients** - the world's second-largest diabetic population. Diabetic Retinopathy (DR) is the leading cause of preventable blindness among working-age adults. Yet:

- Most rural primary health centres have **no ophthalmologist on site**
- Patients travel 50-200 km to reach specialist care, often **too late**
- Even when a fundus camera is available, **no local expertise exists** to interpret the image
- Existing AI tools are black boxes - clinicians cannot trust or explain the decision

**DR-Screen** puts a five-grade AI screener directly in the hands of an ASHA worker or nurse. It tells them *what* it found and *where* on the retina it found it - in a printable report they can hand to the patient for referral.

---

## System Architecture

```
+------------------------------------------------------------------+
|                        DR-Screen Pipeline                        |
+------------------------------------------------------------------+

  Fundus Camera Input
         |
         v
  +------------------+   sharpness/brightness
  |  Module 1        |   thresholds fail?
  |  Quality         +--------- REJECT -------->  Skip + Log
  |  Assessment      |
  +--------+---------+
           | borderline -> CLAHE enhance -> re-assess
           | PASS
           v
  +------------------+
  |  Module 2        |   Frangi vesselness (classical) or U-Net
  |  Vessel &        |   . retinal vessel mask
  |  Lesion Seg      |   . lesion mask (MAs, HEs, EXs)
  +--------+---------+   . lesion count
           |
           v
  +------------------+
  |  Module 3        |   Fine-tuned EfficientNet-B0
  |  DR Grading      |   5-class ICDR severity
  |  Classifier      |   + softmax confidence score
  +--------+---------+
           |
           v
  +------------------+   Grad-CAM on last conv layer
  |  Module 4        |   jet heatmap blended on original
  |  Grad-CAM &      +-----------------------------+
  |  PDF Report      |                             |
  +--------+---------+                             v
           |                         +---------------------+
           v                         |  SimEvents Model    |
  results/<id>_DRReport.pdf          |  Clinic queue sim   |
    . original + heatmap             |  (throughput study) |
    . vessel / lesion masks          +---------------------+
    . grade, confidence, referral
```

---

## DR Grading Scale (ICDR)

| Grade | Label | Referral Action |
|-------|-------|----------------|
| 0 | **No DR** | Routine - rescreen in 2 years |
| 1 | **Mild NPDR** | Lifestyle counselling - rescreen in 1 year |
| 2 | **Moderate NPDR** | Refer to ophthalmologist within **4 weeks** |
| 3 | **Severe NPDR** | Refer within **1 week** |
| 4 | **Proliferative DR** | **URGENT** - same-day referral |

---

## Repository Structure

```
Diabetic_Retinopathy/
|
+-- preprocessing/
|   +-- loadDataset.m           # imageDatastore builder for all supported datasets
|
+-- module1_quality/            # Image quality gate
|   +-- assessQuality.m         # Laplacian sharpness + green-channel brightness
|   +-- enhanceFundus.m         # CLAHE + Ben Graham normalisation
|
+-- module2_segmentation/       # Vessel & lesion segmentation
|   +-- runSegmentation.m       # Frangi vesselness OR U-Net inference
|   +-- evaluateSegmentation.m  # Dice / IoU over full test datastore
|
+-- module3_classification/     # DR severity classification
|   +-- classifyDR.m            # EfficientNet-B0 inference -> grade + confidence
|   +-- trainClassifier.m       # Transfer learning fine-tune + save
|   +-- evaluateClassifier.m    # Confusion matrix, QWK kappa, AUC
|
+-- module4_explainability/     # Clinical explainability
|   +-- explainPrediction.m     # Grad-CAM heatmap (+ synthetic fallback)
|   +-- generateReport.m        # PDF report (image + heatmap + metrics)
|
+-- simulink/                   # SimEvents clinic throughput model
+-- app/                        # MATLAB App Designer screening UI
+-- data/                       # Dataset drop-points (contents gitignored)
|   +-- APTOS/  IDRiD/  DRIVE/  STARE/  Messidor/
+-- models/                     # Trained .mat weights (gitignored)
+-- tests/                      # Per-module smoke tests (no dataset required)
+-- results/                    # Generated reports (gitignored)
+-- main.m                      # Quick-start driver script
+-- mainPipeline.m              # Full end-to-end pipeline function
```

---

## What Was Built

This section documents all contributions merged into the main branch.

### Module 1 - Quality Assessment (`module1_quality/`)

**Owner: Member 1**

| File | Description |
|------|-------------|
| `assessQuality.m` | Computes Laplacian-variance sharpness + green-channel brightness statistics. Returns PASS / ENHANCE / REJECT status. Input guards added for uint8 enforcement and RGB dimension check. |
| `enhanceFundus.m` | Background subtraction via morphological opening, CLAHE on L* channel of Lab space, Ben Graham green-channel normalisation, illumination correction blend. |

### Module 2 - Vessel & Lesion Segmentation (`module2_segmentation/`)

**Owner: Member 2**

| File | Description |
|------|-------------|
| `runSegmentation.m` | Dual-path implementation: (1) U-Net semantic segmentation when `models/segmentation/unet_vessel.mat` and `unet_lesion.mat` exist; (2) Classical fallback using multi-scale Frangi vesselness filter (scale-normalised 2D Hessian eigenvalue decomposition) for vessels + morphological top-hat/bottom-hat lesion detection. Safe thresholding guards added for all-zero images. |
| `evaluateSegmentation.m` | Dual-path evaluation: uses `evaluateSemanticSegmentation` from CV Toolbox when licensed; falls back to manual confusion-matrix accumulation computing Dice, IoU, sensitivity, specificity per class. Fixed API property names (`IoU` not `MeanIoU`; sensitivity/specificity computed from `ConfusionMatrix`). |

### Module 3 - DR Classification (`module3_classification/`)

**Owner: Member 3**

| File | Description |
|------|-------------|
| `trainClassifier.m` | Fine-tunes EfficientNet-B0 for 5-class ICDR grading. Fixed layer discovery using `isa()` type-checks (replaces broken `isprop('LearnableParameters')`). Explicit class names on `classificationLayer`. L2 regularisation, piecewise LR schedule, `OutputNetwork='best-validation-loss'`. |
| `classifyDR.m` | Loads trained network (persistent cache), resizes to 224x224, converts to `single [0,1]` via `im2single()` (critical fix - uint8 gave wrong results), runs `classify()`, maps output to ICDR string. |
| `evaluateClassifier.m` | Confusion matrix, per-class precision/recall/F1, quadratic weighted kappa (QWK), one-vs-rest ROC AUC. |

### Module 2.5 - Hybrid YOLOv8 Lesion Detection (`scripts/`, `models/detection/`)

**Four-Phase Architecture: MATLAB Preprocessing -> Python YOLOv8 Training -> MATLAB Dashboard**

| Phase | Script / Component | Description |
|-------|--------------------|-------------|
| **Phase 1: Dataset Acquisition & Formatting** | [`scripts/download_idrid.py`](file:///scripts/download_idrid.py)<br>[`scripts/convert_idrid_to_yolo.py`](file:///scripts/convert_idrid_to_yolo.py) | Downloads or synthesizes IDRiD lesion masks (MA, HE, EX, SE), extracts connected component bounding boxes, normalizes coordinates to YOLOv8 format (`<class> <xc> <yc> <w> <h>`), splits into `train`/`val`, and generates `dataset_yolo/data.yaml`. |
| **Phase 2: Vessel Subtraction Preprocessing** | [`scripts/vessel_subtract_batch.m`](file:///scripts/vessel_subtract_batch.m)<br>[`scripts/vessel_subtract_batch.py`](file:///scripts/vessel_subtract_batch.py) | Segments blood vessels via `vessel_unet.mat` (or Frangi filter fallback) and applies inpainting (`regionfill` / Telea) to produce vessel-free retinal images, preventing confusion between dark vessels and microaneurysms/hemorrhages. |
| **Phase 3: YOLOv8s Training (Python)** | [`scripts/train_yolov8.py`](file:///scripts/train_yolov8.py)<br>`dataset_yolo/data.yaml` | Trains YOLOv8s (Small) configured with `batch=8` and `imgsz=512` to maximize lesion feature extraction while strictly honoring the 4 GB VRAM ceiling of the NVIDIA RTX 3050. |
| **Phase 4: ONNX Export & Dashboard Integration** | [`scripts/export_onnx.py`](file:///scripts/export_onnx.py)<br>[`module2_segmentation/detectLesionsYOLO.m`](file:///module2_segmentation/detectLesionsYOLO.m)<br>[`runMasterPipeline.m`](file:///runMasterPipeline.m) | Exports trained `.pt` weights to static opset-12 ONNX (`best.onnx`) with single output tensor `[1, 8, 5376]`, synchronizes with `C:/SIH_Work/Diabetic_Retinopathy`, and updates `runMasterPipeline.m` to display a 3-panel clinical dashboard (Left: U-Net vessels, Center: YOLOv8 lesion bounding boxes, Right: EfficientNet classification). |

### Module 4 - Explainability & Report (`module4_explainability/`)

**Owner: Member 4**

| File | Description |
|------|-------------|
| `explainPrediction.m` | Grad-CAM using trained EfficientNet-B0 when available. Fixed `{gcNet.Layers.Name}` crash for DAGNetwork - replaced with `arrayfun(@(l) l.Name, ...)`. Added `safeIdx` clamp for class index bounds. Synthetic Laplacian + grade-guided spatial emphasis fallback when no model present. |
| `generateReport.m` | 6-panel figure: Original, Grad-CAM, vessel mask, lesion mask, metrics text block. Includes quality stats, ICDR grade, confidence, referral recommendation, lesion count, disclaimer. PDF export via `print()`; PNG fallback on systems without PDF driver. Replaced deprecated `datestr(now)` with `datetime()`. |

### Integration (`mainPipeline.m`)

**Owner: Member 4**

Full M1 -> M2 -> M3 -> M4 orchestration. REJECT short-circuits after quality gate. ENHANCE branch enhances then re-assesses. Wrapped in `try/catch` with safe struct defaults on ERROR. Progress logged to console at each stage.

---

## Quick Start

**Clone and set up:**
```bash
git clone https://github.com/404prateek/Diabetic_Retinopathy.git
cd Diabetic_Retinopathy
```

**In MATLAB (R2023b or later):**
```matlab
% 1. Add everything to path
addpath(genpath(pwd));

% 2. Smoke-test all modules (no dataset needed - uses synthetic images)
run('tests/test_module1.m')   % PASS (no model needed)
run('tests/test_module2.m')   % PASS (classical fallback runs)
run('tests/test_module3.m')   % SKIP if no trained model
run('tests/test_module4.m')   % PASS (synthetic Grad-CAM fallback)

% 3. Train the classifier (requires APTOS dataset in data/APTOS/)
imds = loadDataset('data/APTOS');
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');
trainClassifier(imdsTrain, imdsVal);   % saves models/classification/drClassifier.mat

% 4. Run full pipeline on one image
main
```

---

## Supported Datasets

| Dataset | Images | Task | Where to get it |
|---------|--------|------|----------------|
| **APTOS 2019** | 3,662 | 5-class DR grading | [Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection) |
| **IDRiD** | 516 | Indian fundus + pixel lesion masks | [IEEE DataPort](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) |
| **DRIVE** | 40 | Vessel segmentation ground truth | [Grand Challenge](https://drive.grand-challenge.org/) |
| **STARE** | 397 | Vessel + pathology | [Clemson](http://cecas.clemson.edu/~ahoover/stare/) |
| **Messidor** | 1,200 | DR grading (French hospitals) | [ADCIS](https://www.adcis.net/en/third-party/messidor/) |

> Download files, place under `data/<DatasetName>/` - they are gitignored automatically.

---

## Required MATLAB Toolboxes

| Toolbox | Used in |
|---------|---------|
| Image Processing Toolbox | All modules |
| Deep Learning Toolbox | Classification, Grad-CAM |
| Computer Vision Toolbox | Segmentation evaluation (optional - fallback available) |
| Statistics & Machine Learning Toolbox | Classifier metrics (kappa, AUC) |
| SimEvents | Clinic queue simulation |

Minimum MATLAB version: **R2023b**

---

## Target Metrics

| Module | Metric | Target |
|--------|--------|--------|
| Quality gate | Sensitivity for unusable images | > 90% |
| Vessel segmentation | Dice coefficient (DRIVE) | > 0.82 |
| Lesion segmentation | Dice coefficient (IDRiD) | > 0.70 |
| DR classifier | Quadratic-weighted kappa (APTOS) | > 0.85 |
| DR classifier | Sensitivity for Severe + PDR | > 0.95 |
| End-to-end | Inference time per image (CPU) | < 10 s |

---

## Team Meraki - Ownership Map

| Member | Files Owned | Deliverable |
|--------|-------------|-------------|
| Member 1 | `data/`, `preprocessing/`, `module1_quality/` | Quality gate + enhancement |
| Member 2 | `module2_segmentation/` | Vessel & lesion segmentation |
| Member 3 | `module3_classification/` | Trained classifier + evaluation |
| Member 4 | `module4_explainability/`, `app/`, `simulink/`, `mainPipeline.m` | Grad-CAM, PDF report, App UI, integration |

**Branching convention:** `feature/<name>-<module>` - PR requires review from Member 4.
**Rule:** Never change a function signature - other modules depend on the contract.

---

## Disclaimer

> This system is a **decision-support research prototype**, not a certified medical device.
> It must not be used as a standalone diagnostic tool.
> All outputs must be reviewed by a qualified ophthalmologist before clinical action is taken.

---

<div align="center">
<i>Built for Smart India Hackathon 2024 - PS ID 26038 - Team Meraki</i>
</div>
