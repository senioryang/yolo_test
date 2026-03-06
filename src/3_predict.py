from ultralytics import YOLO
import cv2
import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

def predict(image_path, model_path):
    # Load the trained model
    model = YOLO(model_path)
    
    # Run inference
    # conf=0.7 is confidence threshold
    results = model(image_path, conf=0.7)
    
    # Process results
    for result in results:
        boxes = result.boxes
        if len(boxes) == 0:
            print("No question images detected.")
            continue
            
        print(f"Detected {len(boxes)} question images.")
        
        # Print detailed coordinate information
        for i, box in enumerate(boxes):
            # box.xyxy provides [x1, y1, x2, y2] coordinates
            coords = box.xyxy[0].tolist()
            
            # --- Save Crop Logic ---
            x1, y1, x2, y2 = map(int, coords)
            # Ensure coordinates fit image
            h_img, w_img = result.orig_img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            
            crop_img = result.orig_img[y1:y2, x1:x2]
            if crop_img.size > 0:
                crop_filename = f"crop_{i+1}_{Path(image_path).stem}.jpg"
                crop_path = Path(image_path).parent / crop_filename
                
                # Unicode-safe save using cv2.imencode + numpy.tofile
                success_enc, encoded_img = cv2.imencode(".jpg", crop_img)
                if success_enc:
                    encoded_img.tofile(str(crop_path))
                    print(f"    Saved crop to {crop_path.name}")
            # -----------------------

            # Also get normalized xywhn format for comparison with labels
            xywhn = box.xywhn[0].tolist()
            conf = box.conf[0].item()
            
            print(f"  Box {i+1}:")
            print(f"    Pixel Coords (xyxy): x1={coords[0]:.1f}, y1={coords[1]:.1f}, x2={coords[2]:.1f}, y2={coords[3]:.1f}")
            print(f"    YOLO Label (xywhn):  0 {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}")
            print(f"    Conf={conf:.4f}")

        # Visualize
        im_array = result.plot()  # plot a BGR numpy array of predictions
        im = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB) # RGB conversion if displaying with matplotlib
        
        # Save or display
        output_path = Path(image_path).parent / f"predicted_{Path(image_path).name}"
        cv2.imwrite(str(output_path), im_array)
        print(f"Saved prediction to {output_path}")

if __name__ == '__main__':
    # Determine base directory
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
    
    # Configure paths relative to project root
    # Note: Because the data folder structure changed, verify this path exists
    # Example: BASE_DIR / "data/source_pages/book_8785_page_0001.jpeg"
    TEST_IMAGE_DIR = BASE_DIR / "data/source_pages"
    
    # Find a test image dynamically
    if TEST_IMAGE_DIR.exists():
        potential_images = list(TEST_IMAGE_DIR.glob("*.jpeg")) + list(TEST_IMAGE_DIR.glob("*.jpg"))
        if potential_images:
            TEST_IMAGE = potential_images[0]
        else:
             TEST_IMAGE = BASE_DIR / "data/test_image.jpg" # Fallback
    else:
        TEST_IMAGE = BASE_DIR / "data/test_image.jpg"
        
    MODEL_PATH = BASE_DIR / "runs/detect/train_run/weights/best.pt"
    
    print(f"Using Test Image: {TEST_IMAGE}")
    print(f"Looking for model at: {MODEL_PATH}")
    
    if TEST_IMAGE.exists() and MODEL_PATH.exists():
        predict(str(TEST_IMAGE), str(MODEL_PATH))
    else:
        print("Please ensure test image and trained model exist.")
