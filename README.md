# Project 4 — Image & Text Recognition (Basic)

DecodeLabs AI Industrial Training Kit (Batch 2026) — "Building the Machine's Optic Nerve"

This project implements **both** recognition paths described in the brief, so you can submit whichever (or both) your milestone requires:

- **Path 1 — OCR** (`ocr_recognition.py`): reads text out of an image using `pytesseract`.
- **Path 2 — Object Detection** (`object_detection.py`): detects and localizes real-world objects using a pre-trained **MobileNet-SSD** deep learning model via OpenCV's DNN module.

Both paths follow the exact pipeline and pass/fail rules from the brief:

## The 4-step Milestone Validation (both scripts satisfy all 4)

| # | Requirement | How it's satisfied |
|---|---|---|
| 1 | **Library Integration** | `pytesseract` (OCR engine wrapper) / `cv2.dnn` + MobileNet-SSD |
| 2 | **Pre-Processing Integrity** | Grayscale → Gaussian Blur → Adaptive (Otsu) Thresholding for OCR; 4D blob construction (mean subtraction + scaling) for detection |
| 3 | **Accuracy Benchmarking** | Hard-coded **80% minimum confidence gate** — anything below is dropped |
| 4 | **Visual Confirmation** | Annotated output image with bounding boxes + labels + confidence %, saved to disk |

## Files

| File | Purpose |
|---|---|
| `ocr_recognition.py` | Path 1: full OCR pipeline with pre-processing + 80% confidence gate |
| `object_detection.py` | Path 2: full object-detection pipeline with 80% confidence gate |
| `make_sample_image.py` | Generates a sample text image if you don't have your own |
| `models/` | Folder for the two MobileNet-SSD model files (see setup below) |

## Setup

### 1. Install Tesseract OCR engine (system-level, required for Path 1)
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows: https://github.com/UB-Mannheim/tesseract/wiki
```

### 2. Install Python packages
```bash
pip install pytesseract opencv-python-headless pillow numpy
```

### 3. (Path 2 only) Download the pre-trained MobileNet-SSD model files
```bash
mkdir -p models
curl -L -o models/MobileNetSSD_deploy.prototxt \
  https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt
curl -L -o models/MobileNetSSD_deploy.caffemodel \
  https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/mobilenet_iter_73000.caffemodel
```

## Usage

**Path 1 — OCR:**
```bash
python make_sample_image.py        # optional, if you have no test image
python ocr_recognition.py sample_input.png
```
Outputs: recognized text + confidence in the console, plus `ocr_output.png` (annotated) and `ocr_preprocessed.png` (grayscale/blur/threshold result).

**Path 2 — Object Detection:**
```bash
python object_detection.py your_photo.jpg
```
Outputs: detected object list + confidence in the console, plus `detection_output.png` (annotated with bounding boxes).

## Notes on the 80% Confidence Gate

Both scripts implement the exact logic from the brief:
```python
if confidence >= 0.80:
    draw_box_and_label()
else:
    drop_detection()
```
This favors precision over recall — it minimizes false positives at the cost of occasionally missing a low-confidence true detection, which is the standard tradeoff the brief calls for.
