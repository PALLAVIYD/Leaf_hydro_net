# Boea hygrometrica Drought Tolerance Pipeline

This repository contains the advanced models and analytic pipelines developed to predict and monitor the physiological desiccation responses of the *Boea hygrometrica* resurrection plant. The original rudimentary Jupyter analyses have been refactored and scaled into robust, standalone Python modules matching the methodology of the academic report.

## Project Structure

```text
TARP/
├── app.py                     # Main Streamlit Graphical Interface
├── README.md             
├── requirements.txt      
├── data/                      # Raw datasets & Media Output
│   ├── simulated_outputs/     # (Matplotlib mathematical plots)
│   └── yolo_dataset/          # (YOLOv8 image datasets & data.yaml)
├── docs/                      # Documentation files
├── models/                    # Trained Model Weights (.h5, .pt, .pkl)
│   ├── cnn/                   # Plant Village Disease CNN Weights
│   ├── regression/            # Scikit-Learn Water Potential Regressors 
│   └── yolo/                  # YOLOv8 Fold Severity Weights
├── notebooks/                 # Original Prototype IPYNBs
└── src/                       # Python Pipeline Modules
    ├── cnn_train.py           # Module 1: TF/Keras CNN Disease Classifier
    ├── yolo_detect.py         # Module 2: YOLOv8 Fold Severity
    ├── morphology.py          # Module 3: OpenCV Leaf Math
    ├── water_potential.py     # Module 4: Mathematical Feature Regression
    └── generate_dummy_yolo.py # Utility: Dummy Dataset Generator
```

## How to Execute the "Legit" Pipeline

We have successfully transitioned from dummy generators to real-world datasets. To reproduce the full scientific results:

### Step 1: Environment Setup
Ensure your Conda environment is activated and dependencies (including pandas/joblib) are installed.
```bash
conda activate pallu
pip install -r requirements.txt
```

### Step 2: Convert Kaggle Leaf Dataset
The project now uses a real Kaggle dataset (1,130 images). Convert the raw COCO/Kaggle bounding boxes into the normalized YOLOv8 format:
```bash
python src/convert_kaggle_to_yolo.py
```
*Output: Restructures `data/yolo_dataset/` with train/val splits and `.txt` labels.*

### Step 3: Train the YOLO Leaf-Cropper
Train the object detector to find and crop leaves precisely for the morphology engine:
```bash
python src/yolo_detect.py
```
*Output: `runs/detect/models/yolo/train/weights/best.pt`*

### Step 4: Serialize Water Potential Mathematics
Generate the mathematical calibration regressions (Physiological Psi vs. Folding):
```bash
python src/water_potential.py
```
*Output: `models/regression/psi_model.pkl` and `surv_model.pkl`*

### Step 5: Launch the Presentation App
Launch the Streamlit dashboard to analyze any leaf image. It will use the trained YOLO model to crop the target and the Morphology engine to calculate measurements.
```bash
streamlit run app.py
```
