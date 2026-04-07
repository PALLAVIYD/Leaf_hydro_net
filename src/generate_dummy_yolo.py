import cv2
import numpy as np
import os
import random

# Script to generate a synthetic YOLO dataset for Boea Folding Severity
# This allows YOLOv8 to train and output best.pt so the pipeline compiles!

IM_DIR = "data/yolo_dataset/images/train"
LB_DIR = "data/yolo_dataset/labels/train"
os.makedirs(IM_DIR, exist_ok=True)
os.makedirs(LB_DIR, exist_ok=True)

# Classes: 0: Healthy, 1: Mild Fold, 2: Severe Fold
CLASSES = ["Healthy", "Mild", "Severe"]

print("Generating synthetic YOLO dataset...")
for i in range(50):
    # Create blank white image (256x256)
    img = np.ones((256, 256, 3), dtype=np.uint8) * 255
    
    # Pick a random class
    cls_idx = random.randint(0, 2)
    
    # Draw a mock "leaf"
    # Healthy = wide green oval
    # Mild = narrower
    # Severe = very thin
    center = (128, 128)
    if cls_idx == 0:
        cv2.ellipse(img, center, (80, 40), 0, 0, 360, (0, 150, 0), -1) # W: 160
        bbox_w, bbox_h = 160, 80
    elif cls_idx == 1:
        cv2.ellipse(img, center, (40, 40), 0, 0, 360, (0, 120, 0), -1) # W: 80
        bbox_w, bbox_h = 80, 80
    else:
        cv2.ellipse(img, center, (15, 40), 0, 0, 360, (50, 100, 50), -1) # W: 30
        bbox_w, bbox_h = 30, 80
        
    # Standardized YOLO coordinate format: class x_center y_center width height (normalized)
    x_c = 128 / 256.0
    y_c = 128 / 256.0
    w_norm = bbox_w / 256.0
    h_norm = bbox_h / 256.0
    
    img_name = f"dummy_{i}.jpg"
    txt_name = f"dummy_{i}.txt"
    
    cv2.imwrite(os.path.join(IM_DIR, img_name), img)
    # create validation set dummy duplicates
    os.makedirs("data/yolo_dataset/images/val", exist_ok=True)
    os.makedirs("data/yolo_dataset/labels/val", exist_ok=True)
    cv2.imwrite(os.path.join("data/yolo_dataset/images/val", img_name), img)
    
    with open(os.path.join(LB_DIR, txt_name), "w") as f:
        f.write(f"{cls_idx} {x_c} {y_c} {w_norm} {h_norm}\n")
    with open(os.path.join("data/yolo_dataset/labels/val", txt_name), "w") as f:
        f.write(f"{cls_idx} {x_c} {y_c} {w_norm} {h_norm}\n")

print("Created 50 synthetic leaf images in data/yolo_dataset/")
