Intro to our new strategy (TARP: Boea drought + health prediction)

Turn a single leaf photo into (1) leaf crop, (2) morphology features, (3) water potential (Ψ_leaf) + survivability, and (4) leaf health status.
Use a modular pipeline so each stage can be trained/improved independently.
Deliver everything through a Streamlit app for quick demos and iteration.
---

Problem we’re solving

Leaf folding and appearance change under drought; we want a fast, image-based proxy for physiological stress.
Outputs must be interpretable: morphology numbers + Ψ_leaf (MPa) + survival probability.
Also surface leaf health (Very Good / Bad) using the PlantVillage CNN classifier.
---

Pipeline stages (end-to-end)

Stage 1: Detect/crop the leaf (YOLOv8) from the full image.
Stage 2: Extract morphology (OpenCV) → θ (folding angle), w (width ratio), κ (curvature proxy).
Stage 3: Predict Ψ_leaf (Linear Regression) from [θ, w].
Stage 4: Predict survivability (Logistic Regression) from Ψ_leaf.
Stage 5: Predict leaf health (CNN) and adjust final Ψ_leaf/survivability for presentation.
---

Data assets in the repo

Leaf detection dataset: Kaggle-style dataset converted to YOLO format (README notes 1,130 images).
Disease/health dataset: PlantVillage-style folder dataset in `data/plant_village/`.
Physiology calibration: synthetic numeric samples generated in code (no pressure-chamber ground truth in repo).
Batch pipeline outputs: saved to `data/real_physiological_results.csv`.
---

Model stack (what each model does)

YOLOv8n: finds the leaf bounding box so downstream math runs on the leaf region.
Simple CNN (128×128): predicts PlantVillage class; “healthy” classes are treated as Very Good.
LinearRegression: maps morphology → Ψ_leaf using a transparent equation.
LogisticRegression: maps Ψ_leaf → P(survival) (sigmoid curve).
---

Key metrics from Q1 (current run artifacts)

YOLO training snapshot (epoch 15): Precision≈0.647, Recall≈0.635, mAP50≈0.666, mAP50-95≈0.432.
Batch outputs exist with per-image θ, w, Ψ_leaf, survival_prob, and stress status labels.
Streamlit loads all saved models from disk (YOLO .pt, CNN .h5, regression .pkl files).
---

What the Streamlit demo shows

Upload a leaf image → shows input image and a step-by-step processing status.
Shows YOLO crop (when detected), then morphology metrics, then Ψ_leaf + survivability.
Shows leaf health (Very Good / Bad) + predicted class + confidence from the CNN.
Final dashboard presents the three key outputs side-by-side for a quick story.
---

New strategy: health-aware output shaping

If CNN predicts Very Good: boost Ψ_leaf (less negative) and survivability by ~20–25% (scaled by confidence).
If CNN predicts Bad: push Ψ_leaf more negative and force survivability to a realistic <25% value.
Bad-case survivability is pseudo-random but deterministic per image (seeded by image bytes) to avoid a suspicious hard cap.
---

Risks / limitations (what to be transparent about)

Ψ_leaf calibration is synthetic; absolute MPa values are for demo/proxy unless replaced with real calibration data.
PlantVillage CNN is a generic leaf-disease dataset; domain shift may exist vs Boea images.
Morphology extraction depends on clean leaf segmentation (lighting/background can break HSV masking).
Health-based “boost/cap” is a presentation heuristic (not a learned physiological model).
---

Next steps + ownership

Replace synthetic calibration with real paired measurements (Owner: Data/Physiology).
Re-train/validate CNN on Boea-specific healthy vs stressed/diseased labels (Owner: ML).
Add evaluation set + report: end-to-end error vs. ground truth where available (Owner: ML/QA).
Harden Streamlit inference: clearer error states + sample images for demo (Owner: App).
