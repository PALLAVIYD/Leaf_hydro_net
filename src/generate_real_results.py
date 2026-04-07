import os
import sys
import cv2
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append('src')
try:
    from morphology import BoeaMorphologyAnalyzer
except ImportError:
    # If already in workspace root
    from src.morphology import BoeaMorphologyAnalyzer

def generate_scientific_profile():
    print("🚀 Initializing Global Dataset Profiling...")
    
    # 1. Load Models & Tools
    try:
        yolo_model = YOLO('runs/detect/models/yolo/train/weights/best.pt')
        psi_model = joblib.load('models/regression/psi_model.pkl')
        surv_model = joblib.load('models/regression/surv_model.pkl')
        analyzer = BoeaMorphologyAnalyzer(baseline_width=300)
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return

    val_dir = 'data/yolo_dataset/images/val'
    if not os.path.exists(val_dir):
        print(f"❌ Validation directory not found: {val_dir}")
        return
        
    image_files = [f for f in os.listdir(val_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"📊 Found {len(image_files)} images in validation set.")
    
    results_list = []
    
    # 2. Process Images
    for i, img_name in enumerate(image_files):
        img_path = os.path.join(val_dir, img_name)
        if (i+1) % 20 == 0:
            print(f"   Processing image {i+1}/{len(image_files)}...")
            
        try:
            # Stage 1: YOLO Detection
            img = Image.open(img_path).convert("RGB")
            yolo_res = yolo_model(img, verbose=False)[0]
            
            if len(yolo_res.boxes) > 0:
                box = yolo_res.boxes[0].xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                cropped_img = img.crop((x1, y1, x2, y2))
                temp_crop_path = "temp_batch_crop.jpg"
                cropped_img.save(temp_crop_path)
            else:
                temp_crop_path = img_path # Fallback to full image
                
            # Stage 2: Morphological Extraction
            morph = analyzer.analyze_image(temp_crop_path)
            
            if "Error" not in morph:
                theta = morph['Folding_Angle_Theta']
                w = morph['Width_Ratio_w']
                
                # Stage 3: Physiological Regression
                X_fit = np.array([[theta, w]])
                pred_psi = psi_model.predict(X_fit)[0]
                pred_surv = surv_model.predict_proba([[pred_psi]])[0][1]
                
                results_list.append({
                    'image_id': img_name,
                    'theta_deg': theta,
                    'width_ratio': w,
                    'pred_psi_mpa': pred_psi,
                    'survival_prob': pred_surv,
                    'status': 'Healthy' if pred_psi > -1.5 else ('Stressed' if pred_psi > -3.0 else 'Severe')
                })
            
            if os.path.exists("temp_batch_crop.jpg"):
                os.remove("temp_batch_crop.jpg")
                
        except Exception as e:
            continue

    # 3. Save Results
    res_df = pd.DataFrame(results_list)
    output_csv = 'data/real_physiological_results.csv'
    res_df.to_csv(output_csv, index=False)
    print(f"✅ Scientific data saved to: {output_csv}")
    
    # 4. Generate Visualization
    plt.figure(figsize=(12, 6))
    
    # Regression Spread
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=res_df, x='theta_deg', y='pred_psi_mpa', hue='status', palette='viridis')
    plt.title('Empirical Folding vs. Water Potential')
    plt.xlabel('Folding Angle (θ) degrees')
    plt.ylabel('Predicted $\Psi_{leaf}$ (MPa)')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Survival Distribution
    plt.subplot(1, 2, 2)
    sns.histplot(data=res_df, x='survival_prob', bins=20, kde=True, color='green')
    plt.title('Dataset Survival Probability Distribution')
    plt.xlabel('Survival Probability')
    plt.ylabel('Count')
    
    plt.tight_layout()
    output_plot = 'data/real_desiccation_distribution.png'
    plt.savefig(output_plot)
    print(f"🎨 Generated statistical distribution plot: {output_plot}")

if __name__ == "__main__":
    generate_scientific_profile()
