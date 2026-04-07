# Boea hygrometrica Drought Pipeline Methodology

This document outlines the scientific methodology, mathematical foundations, and performance results of the **Boea Drought Pipeline**. The pipeline is designed to automate the physiological monitoring of desiccation tolerance in resurrection plants.

---

## 1. Pipeline Overview

The pipeline operates in four distinct stages, transitioning from raw image classification to high-precision physiological regression:

1.  **Module 1: Biotic Screening (CNN)**: Filters out diseased specimens.
2.  **Module 2: Specimen Localization (YOLOv8)**: Crops exactly to the leaf region.
3.  **Module 3: Morphological Phenotyping (OpenCV)**: Extracts folding angles and width ratios.
4.  **Module 4: Physiological Prediction (ML Regression)**: Calculates Water Potential ($\Psi_{leaf}$) and Survival Probability.

---

## 2. Module 1: Pre-Screening (CNN)

To ensure the desiccation results are not confounded by biotic stress, we utilize a **Sequential Convolutional Neural Network (CNN)**.

*   **Architecture**: Keras Sequential model with 3 Convolutional/Pooling blocks.
*   **Dataset**: PlantVillage (38,000+ images across 38 classes).
*   **Purpose**: Validates the "Biological Baseline" of the plant before proceeding to drought analysis.

---

## 3. Module 2: YOLOv8 Leaf-Cropper

The **YOLOv8n (Nano)** architecture is utilized for automated leaf detection. Training was performed on a dataset of **1,130 real images** of plant leaves.

### Performance Results
The model achieved strong convergence with the following metrics:
- **mAP50**: 0.665
- **mAP50-95**: 0.432

![YOLO Training Results](../runs/detect/models/yolo/train/results.png)
*Figure 1: Training and Validation Metrics for the YOLOv8 Leaf-Cropper.*

### Detection Visualization
The model successfully isolates the leaf ROI (Region of Interest) from complex backgrounds, which is critical for accurate morphological math.

![YOLO Prediction](../runs/detect/models/yolo/train/val_batch0_pred.jpg)
*Figure 2: Sample validation batch showing high-confidence leaf bounding boxes.*

---

## 4. Module 3: Morphological Phenotyping

Once cropped, the leaf is analyzed mathematically using OpenCV. We extract two primary features established as proxies for desiccation in the literature:

### 4.1 Folding Angle ($\theta$)
Calculated by finding the primary contour and measuring the angle of the major axis relative to the baseline.
$$ \theta = \arccos\left(\frac{|\vec{v} \cdot \vec{u}|}{\|\vec{v}\| \|\vec{u}\|}\right) $$
*Where $\vec{v}$ is the dominant contour vector and $\vec{u}$ is the horizontal baseline.*

### 4.2 Width Reduction Ratio ($w$)
The ratio of the current bounding width to the baseline width of a fully hydrated leaf.
$$ w = \frac{W_{current}}{W_{baseline}} $$

---

## 5. Module 4: Physiological Regression

The final module translates the observed morphology into internal physiological states using **Scikit-Learn Multi-Output Regression**.

### 5.1 Water Potential Prediction ($\Psi_{leaf}$)
Operating on the biological principle that mechanical folding reflects turgor loss, we use a Linear Regression model:
$$ \Psi_{leaf} = \alpha\theta + \beta w + \gamma $$

### 5.2 Survival Probability
A Logistic Regression model predicts the absolute survival probability based on the predicted $\Psi_{leaf}$:
$$ P(\text{survival}) = \frac{1}{1 + e^{-(\delta_0 + \delta_1 \Psi_{leaf})}} $$

### Calibration Results
![Water Potential Curve](../data/simulated_outputs/water_potential_model.png)
*Figure 3: Predicted vs. True Water Potential and the Logistic Survival Curve.*

---

## 6. Conclusion
The **Boea Drought Pipeline** represents a robust bridge between computer vision and physiological botany. By utilizing real-world training data (mAP 0.66) and scientifically grounded regression models, it enables high-throughput, non-destructive monitoring of drought tolerance in desiccation experiments.
