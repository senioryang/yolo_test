from ultralytics import YOLO
import os
from pathlib import Path
import torch
import yaml

def update_yaml_path(yaml_path, dataset_root):
    # Update the 'path' in data.yaml to the absolute path on the current machine
    # This ensures it works on both Windows and Linux without manual editing
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    data['path'] = str(dataset_root.absolute())
    
    with open(yaml_path, 'w') as f:
        # Increase priority of these fields for readability
        yaml.dump(data, f, sort_keys=False)
    print(f"Updated data.yaml 'path' to: {data['path']}")

def train():
    # Base paths relative to script
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATASET_DIR = BASE_DIR / "dataset"
    DATA_YAML_PATH = DATASET_DIR / "data.yaml"
    PROJECT_DIR = BASE_DIR / "runs/detect"
    
    # 1. Update YAML with current absolute paths (Critical for cross-platform)
    if not DATA_YAML_PATH.exists():
        print(f"Error: {DATA_YAML_PATH} not found. Run generate_labels.py first.")
        return
    update_yaml_path(DATA_YAML_PATH, DATASET_DIR)

    # 2. Detect Environment (Windows CPU vs Linux GPU)
    # Using torch to check CUDA availability
    use_gpu = torch.cuda.is_available()
    
    if use_gpu:
        print("🚀 GPU Detected! Configuring for High-Performance Training (Linux/RTX 4090 mode)")
        device = 0 # Use first GPU
        batch_size = 32 # RTX 4090 has 24GB VRAM, can handle large batches
        epochs = 100
        workers = 8
        imgsz = 640
    else:
        print("🐌 CPU/Debug Mode Detected (Windows mode)")
        device = 'cpu'
        batch_size = 2 # Minimal batch for debugging
        # Increased for small datasets to allow learning (was 3)
        epochs = 50 
        workers = 0 # Multi-processing on Windows can be buggy
        # Increased image size to 640. 320 is too small for text/diagram features 
        imgsz = 640 
        print("  - Image size: 640")
        print("  - Batch size: 2")
        print("  - Running 50 epochs for small dataset training")

    # Initialize model
    model = YOLO("yolov8n.pt")

    # Train
    results = model.train(
        data=str(DATA_YAML_PATH),
        project=str(PROJECT_DIR),
        name="train_run",
        
        # Dynamic parameters
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        workers=workers, 
        
        # Additional useful settings
        exist_ok=True, # Overwrite existing project/name
        save=True 
    )

if __name__ == '__main__':
    train()
