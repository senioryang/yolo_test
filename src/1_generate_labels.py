import cv2
import os
import shutil
import re
import numpy as np
from pathlib import Path
import random
from collections import defaultdict

def cv2_imread_unicode(path):
    """
    Reads an image using cv2.imdecode and numpy.fromfile to handle unicode paths on Windows.
    """
    try:
        # np.fromfile reads binary data; cv2.imdecode decodes it
        img_array = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print(f"Error reading image {path}: {e}")
        return None

def convert_to_yolo_format(img_width, img_height, x, y, w, h):
    # YOLO format: class x_center y_center width height (normalized)
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def find_crop_location(page_img, crop_img):
    # Use template matching to find the crop in the page
    
    # Ensure images are grayscale for matching
    if len(page_img.shape) == 3:
        page_gray = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)
    else:
        page_gray = page_img
        
    if len(crop_img.shape) == 3:
        crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    else:
        crop_gray = crop_img
        
    # Check if crop is larger than page (extracted data error)
    if crop_gray.shape[0] > page_gray.shape[0] or crop_gray.shape[1] > page_gray.shape[1]:
        return None

    res = cv2.matchTemplate(page_gray, crop_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # Threshold for a "good" match. 
    # High threshold reduces false positives which is crucial when searching multiple pages
    threshold = 0.85 
    if max_val >= threshold:
        return max_loc # Top-left corner (x, y)
    return None

def process_dataset(source_pages_dir, source_crops_dir, output_dir, split_ratio=0.8, lookahead=2):
    """
    lookahead: Number of additional pages to search if not found on the expected page.
               If lookahead=2, caches Page N, N+1, N+2.
    """
    pages_dir = Path(source_pages_dir)
    crops_dir = Path(source_crops_dir)
    base_output = Path(output_dir)
    
    # Define paths
    train_img_dir = base_output / "images/train"
    val_img_dir = base_output / "images/val"
    train_lbl_dir = base_output / "labels/train"
    val_lbl_dir = base_output / "labels/val"
    
    # Get all page images sorted naturally (to ensure page_1, page_2 sequence is correct)
    # Recursively search for common image formats
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    page_files = []
    for ext in extensions:
        page_files.extend(list(pages_dir.glob(ext)))
    
    # Sort pages by name to establish the "sequence"
    page_files.sort(key=lambda x: x.name) 
    page_map = {p.stem: i for i, p in enumerate(page_files)} # Map filename_no_ext -> index
    
    print(f"Found {len(page_files)} source pages.")
    
    # Dictionary to store matches: page_filepath -> list of label strings
    page_labels = defaultdict(list)
    
    # Iterate through all crop folders
    # We assume the folder name matches one of the page names (the "expected" page)
    crop_dirs = [d for d in crops_dir.iterdir() if d.is_dir()]
    
    print(f"Processing {len(crop_dirs)} crop directories with Look-ahead={lookahead}...")

    total_crops_processed = 0
    total_crops_matched = 0

    for c_dir in crop_dirs:
        expected_page_name = c_dir.name
        
        # Find where this page starts in our sorted list
        # If the crop folder name doesn't match a page file exactly, we can't guess sequence.
        # You might need fuzzy matching here if names vary (e.g. "Page001" vs "page_1")
        if expected_page_name not in page_map:
            print(f"  [Skip] Crop folder '{expected_page_name}' does not match any source page filename.")
            continue
            
        start_idx = page_map[expected_page_name]
        
        # Get all crops in this folder
        current_crops = []
        for ext in extensions:
            current_crops.extend(list(c_dir.glob(ext)))
            
        if not current_crops:
            continue

        total_crops_processed += len(current_crops)
        
        # For each crop, search in [start_idx, start_idx + lookahead]
        for crop_path in current_crops:
            crop_img = cv2_imread_unicode(crop_path)
            if crop_img is None: continue
            
            h_crop, w_crop = crop_img.shape[:2]
            found_location = False

            # [New] Dynamic Page Range Logic
            # Check if crop filename contains range info, e.g., "q1_p3-5_img.jpg" or "crop_p12_a.jpg"
            # This is a placeholder for custom logic. 
            # If you have external metadata (JSON/CSV), you would load that at the start of the script.
            # Here we demonstrate determining range from filename or falling back to default lookahead.
            
            # Default search range: [start_index, start_index + lookahead]
            current_search_start = start_idx
            current_search_end = min(start_idx + lookahead + 1, len(page_files))
            
            # [New] Dynamic Page Range Logic
            # Check if crop filename contains range info: q{id}_p{start}-p{end}_{idx}.{ext}
            match = re.search(r'p(\d+)-p(\d+)_', crop_path.name)
            if match:
                try:
                    p_start_meta = int(match.group(1))
                    p_end_meta = int(match.group(2))
                    
                    # Calculate logical page span
                    # If start p=5, end p=7. Span is 2 extra pages.
                    page_span = p_end_meta - p_start_meta
                    if page_span < 0: page_span = 0 # Safety check
                    
                    # Update search end. 
                    # We start at start_idx (which corresponds to p_start_meta).
                    # We want to cover up to p_end_meta.
                    # Plus lookahead to handle minor misalignments.
                    current_search_end = min(start_idx + page_span + 1 + lookahead, len(page_files))
                except ValueError:
                    pass

            
            # SEARCH LOOP
            for target_idx in range(current_search_start, current_search_end):
                target_page_path = page_files[target_idx]
                
                # Load page image (In a real pro script, we would cache these to avoid re-reading)
                page_img = cv2_imread_unicode(target_page_path)
                if page_img is None: continue
                
                loc = find_crop_location(page_img, crop_img)
                
                if loc:
                    # Found it!
                    x, y = loc
                    h_page, w_page = page_img.shape[:2]
                    
                    label = convert_to_yolo_format(w_page, h_page, x, y, w_crop, h_crop)
                    
                    # Store result against the ACTUAL matched page, not the expected one
                    page_labels[target_page_path].append(label)
                    
                    # Calculate offset for logging
                    found_offset = target_idx - start_idx
                    if found_offset != 0:
                        print(f"    [Correction] Crop '{crop_path.name}' expected on '{expected_page_name}' but found on '{target_page_path.stem}'")
                    
                    found_location = True
                    total_crops_matched += 1
                    break # Stop looking for this crop
            
            if not found_location:
                 # Optional: print(f"    [Fail] Could not find '{crop_path.name}' in {expected_page_name} or next {lookahead} pages.")
                 pass

    print(f"Matching complete. Found {total_crops_matched}/{total_crops_processed} crops.")
    print(f"Writing dataset...")

    # Write Process: Iterate through ALL source pages to include standard samples AND negative samples
    # Negative samples (images with no objects) are important to reduce False Positives.
    for page_path in page_files:
        labels = page_labels.get(page_path, [])
        
        # Determine split (randomly assign this PAGE to train or val)
        is_train = random.random() < split_ratio
        target_img_dir = train_img_dir if is_train else val_img_dir
        target_lbl_dir = train_lbl_dir if is_train else val_lbl_dir
        
        # Copy Image
        shutil.copy(str(page_path), str(target_img_dir / page_path.name))
        
        # Write Labels (if empty list, it creates an empty file - which is correct for negative samples)
        label_file = target_lbl_dir / f"{page_path.stem}.txt"
        with open(label_file, "w") as f:
            if labels:
                f.write("\n".join(labels))
            else:
                pass # Create empty file

    print("Dataset generation finished.")

if __name__ == "__main__":
    # Configure paths relative to this script
    BASE_DIR = Path(__file__).resolve().parent.parent
    SOURCE_PAGES = BASE_DIR / "data/source_pages"
    SOURCE_CROPS = BASE_DIR / "data/source_crops"
    OUTPUT_DIR = BASE_DIR / "dataset"
    
    # Create directories if they don't exist
    for p in [SOURCE_PAGES, SOURCE_CROPS, OUTPUT_DIR]:
        p.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing data in: {BASE_DIR}")
    # Increased lookahead to 2 (searches Current, Next, Next+1)
    process_dataset(SOURCE_PAGES, SOURCE_CROPS, OUTPUT_DIR, lookahead=2)
