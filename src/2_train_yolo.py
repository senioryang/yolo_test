from ultralytics import YOLO
import os
from pathlib import Path
import torch
import yaml
from dotenv import load_dotenv

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
    # Base paths logic
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(PROJECT_ROOT / ".env")
    
    solo_base = os.getenv("SOLO_BASE_PATH")
    if solo_base:
        if isinstance(solo_base, str):
            solo_base = Path(solo_base)
        print(f"Using configured SOLO_BASE_PATH: {solo_base}")
        BASE_DIR = solo_base
    else:
        BASE_DIR = PROJECT_ROOT
        
    DATASET_DIR = BASE_DIR / "dataset"
    # IMPORTANT: The data.yaml MUST be inside the dataset folder as per YOLO conventions
    # But generate_labels.py writes it there
    DATA_YAML_PATH = DATASET_DIR / "data.yaml"
    
    # Update YAML content (path field) to ensure absolute path correct
    if DATA_YAML_PATH.exists():
        with open(DATA_YAML_PATH, 'r') as f:
            y = yaml.safe_load(f)
        y['path'] = str(DATASET_DIR.absolute().as_posix())
        with open(DATA_YAML_PATH, 'w') as f:
            yaml.dump(y, f, sort_keys=False)
    
    PROJECT_DIR = BASE_DIR / "runs/detect"
    
    # 1. Update YAML with current absolute paths (Critical for cross-platform)
    if not DATA_YAML_PATH.exists():
        print(f"Error: {DATA_YAML_PATH} not found. Run generate_labels.py first.")
        return

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
