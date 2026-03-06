import cv2
import os
import glob
from pathlib import Path
import random
import shutil
from dotenv import load_dotenv

def verify_labels(image_dir, label_dir, output_dir, num_samples=5):
    """
    Verifies YOLO labels by drawing bounding boxes on images and saving them.
    
    Args:
        image_dir (str or Path): Directory containing images.
        label_dir (str or Path): Directory containing label txt files.
        output_dir (str or Path): Directory to save debug images.
        num_samples (int): Number of random samples to verify. If None, verify all.
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    
    # Create or clean output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all label files
    label_files = list(label_dir.glob("*.txt"))
    
    if not label_files:
        print(f"No label files found in {label_dir}")
        return

    if num_samples:
        label_files = random.sample(label_files, min(num_samples, len(label_files)))
    
    print(f"Verifying {len(label_files)} samples...")
    
    for label_path in label_files:
        # Construct corresponding image path
        # Assuming image has same stem but different extension (.jpg, .jpeg, .png)
        image_stem = label_path.stem
        image_found = False
        image_path = None
        
        for ext in ['.jpeg', '.jpg', '.png']:
            potential_path = image_dir / (image_stem + ext)
            if potential_path.exists():
                image_path = potential_path
                image_found = True
                break
        
        if not image_found:
            print(f"Image not found for label: {label_path.name}")
            continue
            
        print(f"Processing: {image_path.name}")
        
        # Read Image
        # cv2.imread usually doesn't handle unicode paths well on Windows
        # Use numpy fromfile workaround
        import numpy as np
        img_array = np.fromfile(str(image_path), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"Failed to load image: {image_path}")
            continue
            
        height, width = img.shape[:2]
        
        # Read Labels
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Draw Bounding Boxes
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5: continue
            
            cls = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            
            # Convert normalized YOLO coordinates to pixel coordinates
            x1 = int((x_center - w/2) * width)
            y1 = int((y_center - h/2) * height)
            x2 = int((x_center + w/2) * width)
            y2 = int((y_center + h/2) * height)

            
            # Crop image
            # Ensure coordinates are within bounds for cropping
            x1_c, y1_c = max(0, x1), max(0, y1)
            x2_c, y2_c = min(width, x2), min(height, y2)
            
            if x2_c > x1_c and y2_c > y1_c:
                crop = img[y1_c:y2_c, x1_c:x2_c]
                crop_dir = output_dir / "crops"
                crop_dir.mkdir(exist_ok=True)
                crop_filename = crop_dir / f"{image_stem}_cls{cls}_idx{i}.jpg"
                
                # Save crop
                is_success, buffer = cv2.imencode(".jpg", crop)
                if is_success:
                    with open(crop_filename, "wb") as f_crop:
                        f_crop.write(buffer)

            # Draw rectangle on main image
            # Color based on class (arbitrary)
            color = (0, 255, 0) # Green
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{cls}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save annotated image
        out_path = output_dir / ("vis_" + image_path.name)
        # cv2.imwrite also might fail with unicode on Windows, verify
        # Trying standard imwrite first. If issues, use imencode
        success = cv2.imwrite(str(out_path), img)
        if not success:
             # Retry with imencode
             is_success, buffer = cv2.imencode(".jpg", img)
             if is_success:
                 with open(out_path, "wb") as f:
                     f.write(buffer)
        
        print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    # Base paths
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
        
    IMAGE_DIR = BASE_DIR / "dataset" / "images" / "train"
    LABEL_DIR = BASE_DIR / "dataset" / "labels" / "train"
    OUTPUT_DIR = BASE_DIR / "runs" / "debug"
    
    verify_labels(IMAGE_DIR, LABEL_DIR, OUTPUT_DIR, num_samples=None)
