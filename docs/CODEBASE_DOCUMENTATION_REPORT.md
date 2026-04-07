# Boea hygrometrica Drought Tolerance Pipeline — Codebase Documentation Report

**Generated:** 2026-04-06  
**Repository root:** `d:/Projects/TARP`  

This report is derived **only** from files and artifacts present in this workspace, including:
- Code: `app.py`, `src/*.py`
- Datasets checked into `data/`
- Model artifacts checked into `models/` and `runs/`
- Existing documentation in `README.md` and `docs/`

---

## 1. Project Overview

### What this project does
This repository implements a multi-stage pipeline that:
1. (Optionally) screens a leaf image for plant disease using a CNN.
2. Detects and crops the leaf region using a YOLOv8 object detector.
3. Extracts morphological features from the cropped leaf using OpenCV.
4. Predicts leaf water potential (Ψ_leaf) via a linear regression model and survival probability via logistic regression.
5. Presents results in an interactive Streamlit dashboard.

**Primary UI:** `app.py` (Streamlit app).  
**Primary pipeline modules:** `src/morphology.py`, `src/water_potential.py`, `src/yolo_detect.py`, `src/cnn_train.py`.  

### Purpose, goals, and intended use cases (as evidenced in repo docs)
- `README.md` describes the project as an “advanced models and analytic pipelines … to predict and monitor the physiological desiccation responses” of *Boea hygrometrica*.
- `docs/PIPELINE_METHODOLOGY.md` specifies the pipeline goal: “automate the physiological monitoring of desiccation tolerance” by mapping visual proxies (folding angle θ and width ratio w) to Ψ_leaf and survival probability.

### Intended users (evidenced)
The repository is structured for:
- Running a demonstration/analysis dashboard (`streamlit run app.py`).
- Training or re-training models via standalone modules (`python src/*.py`).

---

## 2. Dataset

This project uses multiple datasets, all present in `data/`.

### 2.1 Kaggle leaf detection dataset (object detection training)
**Where it is in the repo:**
- Images: `data/leaf_detection_data/train/` (1132 image files present)
- Annotations: `data/leaf_detection_data/train.csv` (5346 rows)

**What the annotation file contains (evidence):**
- Columns: `image_id,width,height,bbox` (see first line(s) of `data/leaf_detection_data/train.csv`)
- `bbox` is stored as a Python-list-like string, e.g. `"[473, 273, 289, 335]"`.

**Size and splits actually produced in this repo (measured from current workspace):**
- Unique images referenced in CSV: **1130**
- YOLO split output (created under `data/yolo_dataset/`):
  - Train: **904** images + **904** label files
  - Val: **226** images + **226** label files
- Total bounding boxes (YOLO label lines):
  - Train bboxes: **4282**
  - Val bboxes: **1064**
  - Total: **5346** (matches CSV row count)

**Features and target variable:**
- Inputs: RGB images (JPEG/PNG).
- Supervision: bounding boxes per image.
- Target: a **single detection class** called `Leaf` (see `data/yolo_dataset/data.yaml`: `nc: 1`, `names: ['Leaf']`).

**Preprocessing and conversion steps (implemented in code):**
Conversion is implemented in `src/convert_kaggle_to_yolo.py`.

Key steps (directly from code):
```python
# 80/20 split (reproducible)
random.seed(42)
random.shuffle(unique_images)
split_idx = int(0.8 * len(unique_images))
train_imgs = set(unique_images[:split_idx])

# COCO-like bbox string -> YOLO normalized coordinates
bbox = ast.literal_eval(row['bbox'])
xmin, ymin, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

x_center = (xmin + w / 2.0) / img_width
y_center = (ymin + h / 2.0) / img_height
norm_w = w / img_width
norm_h = h / img_height

yolo_line = f"0 {x_center} {y_center} {norm_w} {norm_h}\n"
```

**Why it was chosen / suitability (evidence):**
- `README.md` states the project “now uses a real Kaggle dataset (1,130 images)” and uses it specifically to train the “YOLO Leaf-Cropper” for precise cropping prior to morphology.

**Alternative datasets and reasons for exclusion (evidence):**
- The codebase itself does not enumerate alternative leaf-detection datasets.
- The existing narrative doc `docs/PROJECT_REPORT.md` mentions alternatives for disease classification (not leaf detection). No alternative leaf-detection dataset choice rationale is implemented in code.

### 2.2 PlantVillage dataset (disease classification)
**Where it is in the repo:**
- `data/plant_village/` contains **39** class directories with **55,448** images total.
- The directory list includes a `Background_without_leaves/` class folder, which is why the on-disk class folder count is 39.

**How it is loaded (code):**
- `src/cnn_train.py` loads from the **local** `data/plant_village` folder using `tf.keras.utils.image_dataset_from_directory(..., validation_split=0.2, seed=123, ...)`.

Key preprocessing (directly from `src/cnn_train.py`):
```python
normalization_layer = tf.keras.layers.Rescaling(1./255)
ds_train = ds_train.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
```

**Target variable:**
- Multi-class disease label inferred from the folder name (directory-per-class).

**Why it was chosen / suitability (evidence):**
- `docs/PIPELINE_METHODOLOGY.md` states the intent: disease screening is used to avoid confounding drought analysis with biotic stress.

**Alternative datasets and reasons for exclusion (evidence):**
- `docs/PROJECT_REPORT.md` contains an “Alternatives not used” table for disease datasets (e.g., iNaturalist), but this is narrative documentation rather than code-enforced logic.

### 2.3 Synthetic physiological calibration data (regression training)
**Where it is in the repo:**
- Generated in code: `src/water_potential.py` function `generate_synthetic_calibration_data()`.
- Plot output: `data/simulated_outputs/water_potential_model.png`.
- Serialized models: `models/regression/psi_model.pkl` and `models/regression/surv_model.pkl`.

**Features and targets (evidence in code):**
- Inputs: `theta` (folding angle in degrees), `w` (width ratio).
- Targets:
  - `psi_leaf_true` (continuous, MPa)
  - `survival_data` (binary 0/1)

Key generation logic:
```python
np.random.seed(42)
theta_data = np.random.uniform(20, 90, n_samples)

w_data = 0.3 + (theta_data / 90.0) * 0.7 + np.random.normal(0, 0.05, n_samples)
w_data = np.clip(w_data, 0.2, 1.0)

psi_leaf_true = 0.04 * theta_data + 1.5 * w_data - 5.5 + np.random.normal(0, 0.2, n_samples)

z = 2.0 + 1.2 * psi_leaf_true
survival_prob = 1 / (1 + np.exp(-z))
survival_data = np.random.binomial(1, survival_prob)
```

**Important limitation (evidence):**
`src/water_potential.py` explicitly states: “since we don't have physical pressure chamber data”, calibration data is synthetic.

---

## 3. Models Used

This section lists **all models/algorithms implemented** in code and/or present as artifacts.

### 3.1 YOLOv8n leaf detector
**Implementation:** `src/yolo_detect.py`  
**Weights present:** `runs/detect/models/yolo/train/weights/best.pt` (6,200,618 bytes)  
**Training outputs present:** `runs/detect/models/yolo/train/*` (plots, CSV metrics).

**What it predicts (evidence):**
- A single class `Leaf` (`data/yolo_dataset/data.yaml`).

**Why chosen (evidence in docs):**
- `docs/PIPELINE_METHODOLOGY.md` states YOLOv8n (Nano) is used for automated leaf localization as an ROI cropper.

**Alternatives not implemented:**
- No other detector architecture is implemented in code.

### 3.2 Simple CNN disease classifier (Keras Sequential)
**Implementation:** `src/cnn_train.py`  
**Weights present:** `models/cnn/simple_cnn_disease_classifier.h5` (78,359,072 bytes).

**Architecture (evidence in code):**
3 Conv2D + MaxPool blocks → Flatten → Dense(256) → Dense(num_classes).

**Why chosen (evidence):**
- `src/cnn_train.py` states it matches the “original 3-layer Proof-of-Concept from Untitled6.ipynb”.

**Alternatives implemented separately:**
- ResNet50 transfer learning exists in `src/resnet_train.py`.

### 3.3 ResNet50 transfer-learning disease classifier
**Implementation:** `src/resnet_train.py`  
**Artifacts present:** The script writes `training_curves.png`, `confusion_matrix.png`, and `resnet_boea_disease_classifier.h5`, but these files are **not present** in the current workspace root.

**Important note (evidence):**
- `out.txt` contains a TensorFlow Datasets (`tfds`) error trace referencing PlantVillage dataset preparation. This indicates that at least one run attempted TFDS-based PlantVillage loading and failed in the environment at that time.

### 3.4 Morphological phenotyping (OpenCV, deterministic)
**Implementation:** `src/morphology.py` (`class BoeaMorphologyAnalyzer`)  
This is not a trained model; it is a deterministic feature extractor.

Features computed (evidence in code):
- θ folding angle: ellipse fit orientation (`cv2.fitEllipse`)
- w width ratio: bounding rectangle width divided by baseline width (`cv2.boundingRect`)
- κ curvature proxy: `1 - solidity` from convex hull area ratio
- σ symmetry deviation: left vs right mask area difference

### 3.5 Physiological regression models (Scikit-learn)
**Implementation:** `src/water_potential.py`  
**Artifacts present:**
- `models/regression/psi_model.pkl` (LinearRegression)
- `models/regression/surv_model.pkl` (LogisticRegression)

**Why chosen (evidence in code/docs):**
- The code implements the mathematical form described in `docs/PIPELINE_METHODOLOGY.md` and references “Section 6.5 of the Report” in comments.

---

## 4. Workflow

### 4.1 Online (interactive) workflow — Streamlit dashboard
The end-to-end pipeline is orchestrated in `app.py`:

1. **Load models** (cached):
   - Loads regression pickles; falls back to synthetic training if missing.
   - Instantiates `BoeaMorphologyAnalyzer(baseline_width=300)`.
   - Attempts to load YOLO weights from `runs/detect/models/yolo/train/weights/best.pt`.
   - Attempts to load CNN weights from `models/cnn/simple_cnn_disease_classifier.h5`.

2. **Disease check** (optional):
   - If CNN is available and “mock DL” is unchecked, runs CNN inference on a 128×128 resized image.

3. **Leaf detection + crop** (optional):
   - If YOLO is available and “mock DL” is unchecked, runs YOLO and crops the first detected box.

4. **Morphological analysis**:
   - Saves a temporary image (`temp_leaf.jpg`) and calls `analyzer.analyze_image(temp_path)`.

5. **Physiology prediction**:
   - Predict Ψ_leaf via `psi_model.predict([[theta, w]])`.
   - Predict survival probability via `surv_model.predict_proba([[pred_psi]])`.

6. **Display results**:
   - Shows θ, w, Ψ_leaf, and survival probability.
   - Assigns a status using thresholds (in code):
     - Ψ > −1.0 → “Well-Watered”
     - −2.5 < Ψ ≤ −1.0 → “Moderate Stress”
     - Ψ ≤ −2.5 → “Severe Desiccation”

### 4.2 Offline workflows (training + batch profiling)
- **Dataset conversion:** `python src/convert_kaggle_to_yolo.py` creates `data/yolo_dataset/`.
- **YOLO training:** `python src/yolo_detect.py` trains YOLOv8 and generates run artifacts.
- **Regression training:** `python src/water_potential.py` generates synthetic calibration data, trains regression models, writes PKLs and plot.
- **Integration test:** `python src/test_pipeline.py` runs YOLO→Morph→Regression on one validation image and saves `debug_crop.jpg`.
- **Batch profiling:** `python src/generate_real_results.py` runs inference on all validation images and writes:
  - `data/real_physiological_results.csv`
  - `data/real_desiccation_distribution.png`

---

## 5. Implementation Stages

The repository includes an existing staged narrative in `docs/PROJECT_REPORT.md`. Below is a code-grounded version of those stages.

### Stage A — Notebook prototype
**Evidence:** `notebooks/Untitled6.ipynb` and `docs/summary.txt`.
- Implements PlantVillage loading via TFDS, a simple CNN, and early OpenCV masking/edge experiments.

### Stage B — Modularization into `src/`
**Evidence:** `src/cnn_train.py`, `src/morphology.py`, `src/water_potential.py`, `src/yolo_detect.py`.
- Each major stage is a standalone Python module.

### Stage C — Real dataset conversion to YOLO format
**Evidence:** `src/convert_kaggle_to_yolo.py`, `data/yolo_dataset/`.
- Converts `train.csv` to YOLO label files and creates an 80/20 split.

### Stage D — Training outputs committed for YOLO
**Evidence:** `runs/detect/models/yolo/train/results.csv` + associated plots.
- YOLO training artifacts are present and include final metrics.

### Stage E — End-user dashboard integration
**Evidence:** `app.py`.
- Adds caching, temporary file handling, and status labels.

### Problems/bugs encountered (evidence)
- `out.txt` contains a TFDS PlantVillage error trace (dataset_info missing / TFDS builder preparation issues). The current `src/cnn_train.py` avoids TFDS by loading from `data/plant_village`.

---

## 6. Project File & Folder Structure

This section documents the repository structure. For directories that contain very large numbers of files (datasets), contents are summarized with counts.

### Root
| Path | Type | Purpose / connections |
|---|---|---|
| `app.py` | file | Streamlit UI and pipeline orchestration; imports `src/morphology.py` and `src/water_potential.py`; loads YOLO/CNN artifacts if present. |
| `README.md` | file | High-level execution instructions and pipeline steps. |
| `requirements.txt` | file | Python dependency list (TensorFlow, Ultralytics, Streamlit, OpenCV, scikit-learn, etc.). |
| `yolov8n.pt` | file | Ultralytics YOLOv8 nano base weights (used to initialize training). |
| `out.txt` | file | Captured TFDS/TensorFlow stderr + a TFDS PlantVillage failure trace (historical run artifact). |
| `debug_crop.jpg` | file | Output crop created by `src/test_pipeline.py`. |
| `src/` | dir | All Python pipeline modules. |
| `data/` | dir | All datasets and analysis outputs. |
| `models/` | dir | Saved model artifacts and (duplicate) YOLO run folder. |
| `runs/` | dir | Ultralytics YOLO run folder with plots, CSVs, weights. |
| `docs/` | dir | Academic-style documentation and methodology. |
| `notebooks/` | dir | Prototype notebook(s). |

### `src/` (code modules)
| Path | Purpose |
|---|---|
| `src/cnn_train.py` | Trains and saves the simple CNN disease classifier to `models/cnn/simple_cnn_disease_classifier.h5`. |
| `src/resnet_train.py` | Trains a ResNet50 transfer learning classifier; expects TFDS PlantVillage; writes images and model to root when run. |
| `src/yolo_detect.py` | Generates `data/yolo_dataset/data.yaml` and trains YOLOv8n; produces run artifacts. |
| `src/convert_kaggle_to_yolo.py` | Converts Kaggle CSV bboxes into YOLO label files and splits train/val. |
| `src/morphology.py` | OpenCV feature extraction: θ, w, κ, σ. |
| `src/water_potential.py` | Synthetic calibration + LinearRegression (Ψ) + LogisticRegression (survival); writes PKLs and plot. |
| `src/test_pipeline.py` | End-to-end CLI test: YOLO crop → morphology → regression; saves `debug_crop.jpg`. |
| `src/generate_real_results.py` | Batch inference over val images; writes `data/real_physiological_results.csv` and `data/real_desiccation_distribution.png`. |
| `src/generate_dummy_yolo.py` | Generates a small synthetic YOLO dataset (note: uses 3 classes, which does not match current `nc: 1` config in `data.yaml`). |
| `src/__init__.py` | Empty package marker. |

### `data/` (datasets + outputs)
| Path | Purpose |
|---|---|
| `data/leaf_detection_data/train.csv` | Kaggle annotation CSV (5346 bbox rows). |
| `data/leaf_detection_data/train/` | Leaf detection images (1132 files present). |
| `data/leaf_detection_data/test/leaf/` | Small test leaf images (7 files). |
| `data/yolo_dataset/` | YOLO-ready dataset produced by conversion script: images + labels for train/val; plus `data.yaml`. |
| `data/plant_village/` | Disease dataset folder: 39 class directories, 55,448 images. |
| `data/simulated_outputs/water_potential_model.png` | Regression diagnostics plot created by `src/water_potential.py`. |
| `data/real_physiological_results.csv` | Batch inference outputs (224 rows). |
| `data/real_desiccation_distribution.png` | Scatter + histogram plot from batch inference. |

### `models/`
| Path | Purpose |
|---|---|
| `models/cnn/simple_cnn_disease_classifier.h5` | Saved Keras CNN weights. |
| `models/regression/psi_model.pkl` | Saved scikit-learn LinearRegression model. |
| `models/regression/surv_model.pkl` | Saved scikit-learn LogisticRegression model. |
| `models/yolo/runs/detect/train/` | Duplicate copy of YOLO run artifacts (mirrors `runs/detect/models/yolo/train`). |

### `runs/`
| Path | Purpose |
|---|---|
| `runs/detect/models/yolo/train/` | YOLO training outputs: plots, `results.csv`, and `weights/`. |

---

## 7. The `runs/` Folder — All Graphs and Images

This section enumerates **every image file** present under `runs/detect/models/yolo/train/` and interprets it based on the metrics CSV and typical Ultralytics output semantics.

### Training summary metrics (for reference)
From `runs/detect/models/yolo/train/results.csv` (epoch 15, last row):
- Precision(B): **0.64688**
- Recall(B): **0.63534**
- mAP50(B): **0.66563**
- mAP50-95(B): **0.43179**

### 7.1 `runs/detect/models/yolo/train/results.png`
- **What it is:** Multi-panel plot of training/validation losses and detection metrics over epochs.
- **Why this plot type:** Ultralytics standard training dashboard; shows convergence + metric trajectory in one view.
- **Interpretation (evidence):** Metrics panels rise over epochs, culminating near mAP50≈0.665 and mAP50-95≈0.432, matching `results.csv`.

### 7.2 `runs/detect/models/yolo/train/results.csv` (not an image, but the source of plotted metrics)
- Contains per-epoch numeric values that back all metric plots.

### 7.3 `runs/detect/models/yolo/train/confusion_matrix.png`
- **What it is:** Confusion matrix for `Leaf` vs `background`.
- **Why this plot type:** Shows error modes (misses vs false detections) for the detector.
- **Interpretation:** Consistent with recall ~0.635; indicates a non-trivial miss rate.

### 7.4 `runs/detect/models/yolo/train/confusion_matrix_normalized.png`
- **What it is:** Normalized confusion matrix.
- **Why this plot type:** Normalization makes proportions comparable.
- **Interpretation (visible values):** The `Leaf`→`Leaf` cell is ~0.71 and `Leaf`→`background` ~0.29 in the normalized matrix image, indicating a substantial fraction of leaf instances are missed in this evaluation view.

### 7.5 `runs/detect/models/yolo/train/BoxPR_curve.png`
- **What it is:** Precision–Recall curve.
- **Why this plot type:** Standard for object detection quality; PR is informative under class imbalance.
- **Interpretation (legend):** “Leaf 0.665” indicates AP@0.5 for the `Leaf` class ≈ 0.665.

### 7.6 `runs/detect/models/yolo/train/BoxF1_curve.png`
- **What it is:** F1 score vs confidence threshold.
- **Why this plot type:** Helps choose an operating confidence threshold.
- **Interpretation (legend):** Peak F1 ≈ **0.64** at confidence ≈ **0.334**.

### 7.7 `runs/detect/models/yolo/train/BoxP_curve.png`
- **What it is:** Precision vs confidence threshold.
- **Why this plot type:** Shows how tightening threshold improves precision.
- **Interpretation:** Precision increases as confidence increases (typical behavior).

### 7.8 `runs/detect/models/yolo/train/BoxR_curve.png`
- **What it is:** Recall vs confidence threshold.
- **Why this plot type:** Shows how tightening threshold reduces recall.
- **Interpretation:** Recall decreases as confidence increases (typical tradeoff).

### 7.9 `runs/detect/models/yolo/train/labels.jpg`
- **What it is:** Ultralytics label diagnostics: instances-per-class and bbox center/size distributions.
- **Why this plot type:** Reveals dataset characteristics (bbox densities, common sizes/positions).
- **Interpretation (visible number):** Shows **4282** instances for the `Leaf` class (this matches the computed YOLO train label-line total 4282).

### 7.10 Training batch mosaics
These are training-time visualization mosaics used for debugging/inspection.

- `runs/detect/models/yolo/train/train_batch0.jpg`
- `runs/detect/models/yolo/train/train_batch1.jpg`
- `runs/detect/models/yolo/train/train_batch2.jpg`
- `runs/detect/models/yolo/train/train_batch285.jpg`
- `runs/detect/models/yolo/train/train_batch286.jpg`
- `runs/detect/models/yolo/train/train_batch287.jpg`

**What they represent:** Mosaic/augmented training batches with labels drawn.  
**Why this plot type:** Quickly verifies that labels align with objects after augmentation.  
**Interpretation:** Visual confirmation that multiple leaf instances can appear per image, consistent with multiple bboxes per `image_id` in `data/leaf_detection_data/train.csv`.

### 7.11 Validation batch label/prediction mosaics
- `runs/detect/models/yolo/train/val_batch0_labels.jpg`
- `runs/detect/models/yolo/train/val_batch0_pred.jpg`
- `runs/detect/models/yolo/train/val_batch1_labels.jpg`
- `runs/detect/models/yolo/train/val_batch1_pred.jpg`
- `runs/detect/models/yolo/train/val_batch2_labels.jpg`
- `runs/detect/models/yolo/train/val_batch2_pred.jpg`

**What they represent:** Side-by-side style mosaics of ground-truth labels vs model predictions.  
**Why this plot type:** Qualitative validation of box placement and coverage.  
**Interpretation (based on visible examples):** The model predicts many leaf boxes across images, including small leaves; this aligns with the dataset having many small bboxes (see `labels.jpg` width/height distribution).

---

## 8. Results & Evaluation

This section reports metrics that are **actually present** as artifacts in the repository.

### 8.1 YOLOv8 leaf detector (object detection)
**Source of truth:** `runs/detect/models/yolo/train/results.csv`.

**Final epoch metrics (epoch 15):**
| Metric | Value |
|---|---:|
| Precision(B) | 0.64688 |
| Recall(B) | 0.63534 |
| mAP50(B) | 0.66563 |
| mAP50-95(B) | 0.43179 |

**Interpretation in plain language:**
- mAP50≈0.666 indicates the detector localizes leaves reasonably well at IoU=0.5.
- mAP50-95≈0.432 is lower, indicating bounding boxes are less consistently “tight” under stricter IoU thresholds.
- Precision and recall in the mid-0.6s indicate both false positives and false negatives occur; this matters because `app.py` crops only the first detected box.

### 8.2 Water potential regression (LinearRegression) and survival probability (LogisticRegression)
**Artifacts present:**
- `models/regression/psi_model.pkl`
- `models/regression/surv_model.pkl`
- `data/simulated_outputs/water_potential_model.png`

**Evaluation metric present in code:**
- `src/water_potential.py` computes RMSE on the same synthetic dataset used for fitting:
```python
rmse = np.sqrt(mean_squared_error(y_psi, preds))
```

**Interpretation:**
- The regression plot (`water_potential_model.png`) shows predicted vs true Ψ_leaf close to the identity line, indicating low in-sample error on the synthetic calibration data.

**Important caveat (evidence):**
- The calibration data is synthetic (“we don't have physical pressure chamber data”), so this RMSE is not a real-world physiological validation metric.

### 8.3 Batch inference outputs (not evaluation, but measured outputs)
**Source:** `data/real_physiological_results.csv` (224 rows).

**Status counts:**
- Healthy: 95
- Stressed: 60
- Severe: 69

**Ranges:**
- θ: 0.02 → 90.0
- Predicted Ψ_leaf: −5.0707 → 0.3169
- Survival probability: 0.01436 → 0.93077

These values are predictions produced by the pipeline, not accuracy metrics against ground truth.

### 8.4 CNN/ResNet evaluation metrics
- The current workspace includes the CNN weights file, but does **not** include persisted evaluation outputs (accuracy curves, confusion matrices) for the CNN.
- The ResNet training script would generate `training_curves.png` and `confusion_matrix.png`, but these artifacts are not present in the workspace.

---

## 9. Challenges & Limitations

These points are grounded in code paths and artifact contents.

1. **Synthetic physiology calibration (major):**
   - `src/water_potential.py` trains regression models on synthetic data due to missing pressure-chamber ground truth.
   - As a result, regression “performance” is in-sample on synthetic data and may not generalize to real Boea measurements.

2. **YOLO class mismatch across scripts:**
   - `data/yolo_dataset/data.yaml` defines `nc: 1` with class `Leaf`.
   - `src/generate_dummy_yolo.py` generates 3 classes (`Healthy`, `Mild`, `Severe`), which does not match `nc: 1`.

3. **Cropping logic uses only the first detection:**
   - `app.py` and `src/test_pipeline.py` crop only the first YOLO box; images frequently contain multiple leaves (evidenced by multiple bboxes per image in `train.csv`).

4. **Morphology depends on fixed HSV thresholds:**
   - `src/morphology.py` uses a fixed green HSV range `[25,40,40]` to `[90,255,255]`. Non-green leaves, unusual lighting, or background vegetation can cause failures or noisy contours.

5. **Baseline width is hard-coded in the UI path:**
   - `app.py` instantiates `BoeaMorphologyAnalyzer(baseline_width=300)`. Different input resolutions can produce width ratios > 1 (observed in `data/real_physiological_results.csv`).

6. **Historical TFDS issue captured in `out.txt`:**
   - Indicates TFDS-based PlantVillage loading previously failed; current `cnn_train.py` mitigates by loading from local directory instead.

---

## Appendix: Key artifact inventory (measured)

### Model/weight files
| Path | Size (bytes) |
|---|---:|
| `models/cnn/simple_cnn_disease_classifier.h5` | 78,359,072 |
| `models/regression/psi_model.pkl` | 601 |
| `models/regression/surv_model.pkl` | 831 |
| `yolov8n.pt` | 6,549,796 |
| `runs/detect/models/yolo/train/weights/best.pt` | 6,200,618 |
| `runs/detect/models/yolo/train/weights/last.pt` | 6,200,618 |

### Dataset counts
| Dataset | Evidence | Count |
|---|---|---:|
| Kaggle leaf detection images | `data/leaf_detection_data/train/` | 1132 files present |
| Unique images referenced by CSV | `data/leaf_detection_data/train.csv` | 1130 |
| Bbox rows (total) | `data/leaf_detection_data/train.csv` | 5346 |
| YOLO train images | `data/yolo_dataset/images/train/` | 904 |
| YOLO val images | `data/yolo_dataset/images/val/` | 226 |
| YOLO train bbox instances | label lines | 4282 |
| YOLO val bbox instances | label lines | 1064 |
| PlantVillage class folders | `data/plant_village/` | 39 |
| PlantVillage images | recursive count | 55,448 |
| Batch inference rows | `data/real_physiological_results.csv` | 224 |
