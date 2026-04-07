import os
import shutil
import pandas as pd
import ast
import random

def convert():
    print("Starting Kaggle to YOLO dataset conversion...")
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_img_dir = os.path.join(base_dir, 'data', 'leaf_detection_data', 'train')
    csv_path = os.path.join(base_dir, 'data', 'leaf_detection_data', 'train.csv')
    
    out_dir = os.path.join(base_dir, 'data', 'yolo_dataset')
    
    # Refresh Output Directories
    if os.path.exists(out_dir):
        print("Clearing old YOLO dataset directory...")
        shutil.rmtree(out_dir)
        
    for split in ['train', 'val']:
        os.makedirs(os.path.join(out_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'labels', split), exist_ok=True)
        
    # Read Kaggle annotations
    df = pd.read_csv(csv_path)
    
    # Split into train/val
    unique_images = df['image_id'].unique().tolist()
    random.seed(42)
    random.shuffle(unique_images)
    
    split_idx = int(0.8 * len(unique_images))
    train_imgs = set(unique_images[:split_idx])
    
    print(f"Total Unique Images: {len(unique_images)}")
    print(f"Train set: {len(train_imgs)}, Validation set: {len(unique_images) - len(train_imgs)}")
    
    # Write Labels and Copy Images
    missing_images = 0
    copied_images = set()

    for idx, row in df.iterrows():
        img_id = row['image_id']
        img_width = row['width']
        img_height = row['height']
        
        # ast.literal_eval parses the string "[x, y, w, h]" into a python list
        try:
            bbox = ast.literal_eval(row['bbox'])
            xmin, ymin, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        except Exception as e:
            continue
            
        # YOLO Normalization 
        x_center = (xmin + w / 2.0) / img_width
        y_center = (ymin + h / 2.0) / img_height
        norm_w = w / img_width
        norm_h = h / img_height
        
        yolo_line = f"0 {x_center} {y_center} {norm_w} {norm_h}\n"
        
        split_name = 'train' if img_id in train_imgs else 'val'
        
        label_file = os.path.join(out_dir, 'labels', split_name, img_id.replace('.jpg', '.txt').replace('.png', '.txt'))
        img_source_path = os.path.join(source_img_dir, img_id)
        img_dest_path = os.path.join(out_dir, 'images', split_name, img_id)
        
        # Append to txt file
        with open(label_file, 'a') as f:
            f.write(yolo_line)
            
        # Copy image if not already copied for this split
        if img_id not in copied_images:
            if os.path.exists(img_source_path):
                shutil.copy(img_source_path, img_dest_path)
                copied_images.add(img_id)
            else:
                missing_images += 1
                
    if missing_images > 0:
        print(f"Warning: {missing_images} images referenced in CSV were not found in source directory.")
    print("Conversion Complete!")

if __name__ == "__main__":
    convert()
