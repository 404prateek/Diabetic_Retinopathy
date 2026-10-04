# App Designer - DR Screening UI

## Purpose
A MATLAB **App Designer** (.mlapp) application providing a clean, point-of-care
interface for a rural screening operator (nurse/ASHA worker) to upload a fundus
photograph, run the full AI pipeline, and review the results - all without
opening the MATLAB command window.

---

## Planned App File
`app/DRScreeningApp.mlapp`  (to be created by Member 4 in App Designer)

---

## UI Layout

```
+----------------------------------------------------------+
|  DR Screening Tool  |  PS ID 26038                       |
+---------------------------+------------------------------+
|                           |                              |
|   [ Upload Image ]        |   Original Image             |
|                           |                              |
|   [ Run Screening ]       |   Grad-CAM Heatmap           |
|                           |                              |
+---------------------------+------------------------------+
|  Quality                  |  Classification              |
|  Status:  PASS / ENHANCE  |  Grade:      Moderate        |
|  Sharpness:  142.3        |  Confidence: 82.4 %          |
|  Brightness: 88.1 +/- 14.5|                              |
+---------------------------+------------------------------+
|  Recommendation:                                         |
|  "Refer to ophthalmologist within 4 weeks (Moderate DR)" |
|                                 [ Generate PDF Report ]  |
+----------------------------------------------------------+
```

---

## UI Components (App Designer Widgets)

| Component | Widget Type | ID |
|-----------|------------|-----|
| Upload Image button | `uibutton` | `btnUpload` |
| Run Screening button | `uibutton` | `btnRun` |
| Original image axes | `uiaxes` | `axOriginal` |
| Grad-CAM axes | `uiaxes` | `axHeatmap` |
| Quality status label | `uilabel` | `lblQualityStatus` |
| Sharpness value | `uilabel` | `lblSharpness` |
| Brightness mean +/- std | `uilabel` | `lblBrightness` |
| DR grade label | `uilabel` | `lblGrade` |
| Confidence label | `uilabel` | `lblConfidence` |
| Recommendation text | `uitextarea` | `txtRecommendation` |
| Generate Report button | `uibutton` | `btnReport` |

---

## Recommendation Logic (to implement in App)

| DR Grade | Recommendation |
|----------|---------------|
| No DR | "No DR detected. Repeat screening in 2 years." |
| Mild | "Mild NPDR. Lifestyle counselling. Repeat in 1 year." |
| Moderate | "Moderate NPDR. Refer to ophthalmologist within 4 weeks." |
| Severe | "Severe NPDR. Urgent referral within 1 week." |
| Proliferative DR | "Proliferative DR. URGENT referral - same day if possible." |

---

## TODO (Owner: Member 4)

1. Open MATLAB -> **App Designer** -> New App (blank).
2. Design layout matching the wireframe above.
3. `btnUpload` callback: `uigetfile`, `imread`, store in `app.CurrentImage`.
4. `btnRun` callback: call `mainPipeline(app.CurrentImage)`, populate all labels
   and display images on `axOriginal` / `axHeatmap`.
5. `btnReport` callback: call `generateReport(...)`.
6. Save as `app/DRScreeningApp.mlapp`.
7. Test on a sample image from `data/APTOS/` before team demo.

---

> **Disclaimer**: This app is a decision-support prototype, not a diagnostic device.
> All AI outputs must be reviewed by a qualified clinician before clinical action.
