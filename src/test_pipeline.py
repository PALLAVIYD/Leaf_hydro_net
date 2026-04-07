import sys
import os
import cv2
import numpy as np
import joblib
from PIL import Image
from ultralytics import YOLO

# Add src to path
sys.path.append('src')
from morphology import BoeaMorphologyAnalyzer
from water_potential import train_water_potential_model, train_survival_model

def test_full_pipeline(image_path):
    print(f"--- Testing Pipeline for: {image_path} ---")
    
    # 1. Load Models
    print("Loading models...")
    yolo_model = YOLO('runs/detect/models/yolo/train/weights/best.pt')
    psi_model = joblib.load('models/regression/psi_model.pkl')
    surv_model = joblib.load('models/regression/surv_model.pkl')
    analyzer = BoeaMorphologyAnalyzer(baseline_width=300)
    
    # 2. YOLO Detect & Crop
    print("Running YOLO detection...")
    img = Image.open(image_path).convert("RGB")
    results = yolo_model(img)[0]
    
    if len(results.boxes) > 0:
        box = results.boxes[0].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        print(f"Detected leaf at {box} with confidence {results.boxes[0].conf[0].item():.2f}")
        cropped_img = img.crop((x1, y1, x2, y2))
        
        # Save crop for inspection
        crop_path = "debug_crop.jpg"
        cropped_img.save(crop_path)
        print(f"Saved cropped leaf to {crop_path}")
    else:
        print("No leaf detected by YOLO! Using original image.")
        crop_path = image_path
        
    # 3. Morphology
    print("Running Morphological Analysis...")
    morph_results = analyzer.analyze_image(crop_path)
    
    if "Error" in morph_results:
        print(f"Morphology Error: {morph_results['Error']}")
        return
        
    theta = morph_results['Folding_Angle_Theta']
    w = morph_results['Width_Ratio_w']
    print(f"Measurements: Theta={theta} deg, Width Ratio={w}")
    
    # 4. Regression
    print("Running Predictive Regression...")
    X_input = np.array([[theta, w]])
    pred_psi = psi_model.predict(X_input)[0]
    pred_surv = surv_model.predict_proba([[pred_psi]])[0][1]
    
    print(f"\n--- FINAL PREDICTION ---")
    print(f"Predicted Leaf Water Potential: {pred_psi:.2f} MPa")
    print(f"Predicted Survival Probability: {pred_surv*100:.1f}%")

if __name__ == "__main__":
    # Pick the first image from validation set
    val_dir = 'data/yolo_dataset/images/val'
    if os.path.exists(val_dir):
        files = os.listdir(val_dir)
        if files:
            sample_img = os.path.join(val_dir, files[0])
            test_full_pipeline(sample_img)
        else:
            print("No images found in validation directory.")
    else:
        print("Validation directory missing.")
