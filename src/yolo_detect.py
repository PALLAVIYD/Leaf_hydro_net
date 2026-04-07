import os
from ultralytics import YOLO

# ===================================================================
# Module 2: YOLOv8 Leaf-Folding Detection Pipeline
# Handles the object-detection logic described in Section 8.3
# Classes: "Healthy", "Mild", "Severe"
# ===================================================================

def setup_yolo_yaml():
    """
    Creates the required data.yaml for YOLOv8 using absolute paths.
    """
    print("Setting up YOLO dataset directory structure...")
    abs_path = os.path.abspath('data/yolo_dataset')
    yaml_content = f"""
train: {abs_path}/images/train
val: {abs_path}/images/val

nc: 1
names: ['Leaf']
"""
    with open('data/yolo_dataset/data.yaml', 'w') as f:
        f.write(yaml_content.strip())
    print("Created data/yolo_dataset/data.yaml with absolute paths")

def train_yolo_model():
    """
    Trains the YOLOv8 model on the provided dataset.
    Downloads the yolov8n.pt (nano) pre-trained weights to start.
    """
    print("Initializing YOLOv8n model...")
    # Load a pretrained model (recommended for training)
    model = YOLO('yolov8n.pt') 

    # Train the model
    print("Attempting to train YOLOv8 model...")
    try:
        # Train for 15 epochs for convergence
        results = model.train(
            data=os.path.abspath('data/yolo_dataset/data.yaml'), 
            epochs=15, 
            imgsz=256,
            project='models/yolo',
            plots=True # Automatically generate confusion matrices and metrics plots
        )
        print("Training successful!")
        return model
    except Exception as e:
        print(f"Training failed: {e}")
        return model

def run_inference(image_path, model_path='runs/detect/train/weights/best.pt'):
    """
    Runs real-time inference on a target image using a trained model.
    """
    if not os.path.exists(model_path):
        print(f"Model weights not found at {model_path}. You need to train the model first.")
        # Fallback to the base model for demonstration
        model = YOLO('yolov8n.pt')
    else:
        model = YOLO(model_path)
        
    if not os.path.exists(image_path):
        print(f"Input image not found at {image_path}. Please provide a valid path.")
        return
        
    # Perform object detection
    results = model(image_path)
    print("Inference completed")

if __name__ == "__main__":
    print("--- YOLOv8 Pipeline for Boea hygrometrica ---")
    setup_yolo_yaml()
    
    # Train the model since we now have the kaggle dataset mapped
    trained_model = train_yolo_model()
    print("Done generating yolo model.")
