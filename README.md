# YOLO Question Image Detector

Project for training a YOLO model to detect question-related images in textbook pages.

## Process Overview

Since the original data lacks bounding box coordinates (only having the cropped images), this project uses a two-step process:

1.  **Automatic Label Generation**: Uses Template Matching (OpenCV) to find the location of known cropped images within the full page images. This generates the necessary YOLO training labels.
2.  **Model Training**: Trains a YOLOv8 model using the generated dataset.

## Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation

You must organize your data into the following structure before running the scripts:

1.  **Page Images**: Place all full PDF page images in `data/source_pages/`.
    *   Example: `data/source_pages/page_001.jpg`

2.  **Cropped Images**: Place the cropped question images in `data/source_crops/`.
    *   **Crucial**: Create a subfolder for each page with the exact same name as the page image (minus extension).
    *   Example: If you have `page_001.jpg`, create folder `data/source_crops/page_001/` and put all question images for that page inside it (e.g., `image_1.jpg`, `fig_a.png`).

## Usage

### 1. Generate Labels
Run the generation script to create the dataset:
```bash
python src/1_generate_labels.py
```
This will populate `dataset/images` and `dataset/labels`. Check the console output to see how many crops were successfully matched.

### 2. Train Model
Run the training script:
```bash
python src/2_train_yolo.py
```
This will start training a YOLOv8 Nano model. Results will be saved to `runs/detect/train_run`.

### 3. Inference / Prediction
To test the model on a new page:
```bash
python src/3_predict.py
```
(Edit the script to point to your specific test image).
