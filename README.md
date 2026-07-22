# Image and Text Recognition (Basic)
**DecodeLabs AI Industrial Training Kit (Batch 2026): "Building the Machine's Optic Nerve"**

This project implements **both** recognition paths described in the brief, so you can submit whichever (or both) your milestone requires:
- **Path 1: OCR** (`ocr-recognition.py`)  reads text out of an image using `pytesseract`
- **Path 2: Object Detection** (`object-detection.py`)  detects and localizes real-world objects using a pre-trained **MobileNet-SSD** deep learning model via OpenCV's DNN module

Both paths follow the exact pipeline and pass/fail rules from the brief.

## The 4-Step Milestone Validation
Both scripts satisfy all four requirements.

| # | Requirement | How It's Satisfied |
|---|---|---|
| 1 | **Library Integration** | `pytesseract` (OCR engine wrapper) / `cv2.dnn` with MobileNet-SSD |
| 2 | **Pre-Processing Integrity** | Grayscale, Gaussian Blur, and Adaptive (Otsu) Thresholding for OCR; 4D blob construction (mean subtraction and scaling) for detection |
| 3 | **Accuracy Benchmarking** | Hard-coded 80% minimum confidence gate — anything below is dropped |
| 4 | **Visual Confirmation** | Annotated output image with bounding boxes, labels, and confidence percentage, saved to disk |

## Files
| File | Purpose |
|---|---|
| `ocr-recognition.py` | Path 1: full OCR pipeline with pre-processing and 80% confidence gate |
| `object-detection.py` | Path 2: full object-detection pipeline with 80% confidence gate |
| `make-sample-image.py` | Generates a sample text image if you don't have your own |
| `models/` | Folder for the two MobileNet-SSD model files (see setup below) |

## Setup
### 1. Install Tesseract OCR Engine (system-level, required for Path 1)

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows: https://github.com/UB-Mannheim/tesseract/wiki
```

### 2. Install Python Packages
```bash
pip install pytesseract opencv-python-headless pillow numpy
```

### 3. Download the Pre-Trained MobileNet-SSD Model Files (Path 2 only)

```bash
mkdir -p models
curl -L -o models/mobilenet-ssd-deploy.prototxt \
  https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt
curl -L -o models/mobilenet-ssd-deploy.caffemodel \
  https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/mobilenet_iter_73000.caffemodel
```

## Usage
**Path 1: OCR**

```bash
python make-sample-image.py        # optional, if you have no test image
python ocr-recognition.py sample-input.png
```

Outputs recognized text and confidence to the console, plus `ocr-output.png` (annotated) and `ocr-preprocessed.png` (grayscale, blur, and threshold result).

**Path 2: Object Detection**
```bash
python object-detection.py your-photo.jpg
```

Outputs the detected object list and confidence to the console, plus `detection-output.png` (annotated with bounding boxes).

## Notes on the 80% Confidence Gate
Both scripts implement the exact logic from the brief:

```python
if confidence >= 0.80:
    draw_box_and_label()
else:
    drop_detection()
```

This favors precision over recall. It minimizes false positives at the cost of occasionally missing a low-confidence true detection, which is the standard tradeoff the brief calls for.

---

## Author

**Sumaira Safeer**
Computer Engineer

LinkedIn: [linkedin.com/in/sumaira-safeer-948804418](https://www.linkedin.com/in/sumaira-safeer-948804418/)
GitHub: [github.com/SumairaSafeer](https://github.com/SumairaSafeer)
