# Project — Image & Text Recognition (Basic)

**Status: ✅ Complete — Path 1 (OCR) implemented, tested, and passing all milestone checks.**

---

## Overview
This project implements text recognition (OCR) using `pytesseract` on top of an
OpenCV pre-processing pipeline. The script reads an input image, cleans it up,
runs OCR, filters results by confidence, and produces an annotated output image.

---

## Files

| File | Name | Description |
|---|---|---|
| `ocr_recognition.py` | OCR Recognition Script | Path 1 full pipeline:   grayscale → Gaussian blur → Otsu thresholding → Tesseract OCR → 80% confidence gate → annotated output. |
| `object_detection.py` | Object Detection Script | Path 2 (optional/not used in this submission) MobileNet-SSD object detection with the same 80% confidence gate. |
| `make_sample_image.py` | Sample Image Generator | Generates a plain test image with text, for use when no other input image is available. |
| `sample_input.png` | Sample Input Image | Auto-generated test image used for the first test run. |
| `ocr_preprocessed.png` | Pre-processing Output | Shows the grayscale → blur → Otsu threshold result before OCR is applied. |
| `ocr_output.png` | Annotated OCR Output | Final image with recognized text boxed, labeled, and confidence percentages shown. |
| `MobileNetSSD_deploy.prototxt` | Model Architecture File | Network architecture definition for Path 2 (weights downloaded separately, not included). |
| `README.md` | This File | Setup instructions, usage, and results summary. |

---

## Milestone Requirements — How Each Was Satisfied

| # | Requirement | Result |
|---|---|---|
| 1 | **Library Integration** | Used `pytesseract` as the OCR engine wrapper around the Tesseract binary. |
| 2 | **Pre-Processing Integrity** | Applied grayscale conversion → Gaussian blur → Otsu adaptive thresholding before OCR.|
| 3 | **Accuracy Benchmarking** | Hard-coded 80% minimum confidence gate — any word below 80% is dropped and not annotated. |
| 4 | **Visual Confirmation** | Annotated output image saved to disk with bounding boxes, text, and confidence % |

---

## Setup

### 1. Install the Tesseract OCR engine (system-level)
```bash
# Windows: installer from https://github.com/UB-Mannheim/tesseract/wiki
# macOS
brew install tesseract
# Ubuntu/Debian
sudo apt-get install tesseract-ocr
```

### 2. Install Python packages
```bash
pip install pytesseract opencv-python-headless pillow numpy
```

### 3. Verify installation
```bash
tesseract --version
```

---

## Usage
```bash
python ocr_recognition.py sample_input.png
```

Outputs:
- Console: recognized text + average confidence + word-by-word breakdown.
- `ocr_preprocessed.png`  grayscale/blur/threshold result.
- `ocr_output.png`  final annotated image.

---

## Test Results
### Test Run 1 — Sample generated image (`sample_input.png`)
```
===== OCR RECOGNITION RESULTS =====
Recognized text : DecodeLabs Project 4
Average confidence: 94.0%
Word-by-word:
  - 'DecodeLabs'  ->  90% confidence
  - 'Project'     ->  96% confidence
  - '4'           ->  96% confidence
====================================
```
All words passed the 80% gate and were annotated in `ocr_output.png`.

### Test Run 2 — Real-world image (banner/graphic with mixed text and design elements)
The script correctly detected and annotated multiple text regions with confidence
scores (e.g. "SAFEER" 81%, "Engineer" and "Embedded" ~94%). One region ("SUMAIRA")
was partially misread as "UMAIRA" because the leading "S" overlapped a graphic
element in the image, which affected character segmentation. This is expected
OCR behavior rather than a script defect, it demonstrates the confidence gate
correctly filtering results based on how cleanly text can be segmented from
its surroundings, and highlights that OCR accuracy is sensitive to image
clarity, contrast, and overlapping visual elements.

---

## Notes on the 80% Confidence Gate
```python
if confidence >= 0.80:
    draw_box_and_label()
else:
    drop_detection()
```
This favors precision over recall — it minimizes false positives at the cost of
occasionally missing a low-confidence true detection, which is the standard
tradeoff specified in the project brief.

---

## Conclusion
Path 1 (OCR) was implemented and tested successfully on two different images,
a clean generated sample and a complex real-world graphic. All four milestone
requirements (library integration, pre-processing, confidence gate, visual
confirmation) were met and demonstrated with console output and annotated
image files.

## Author
**Sumaira Safeer**
**Computer Engineer**
