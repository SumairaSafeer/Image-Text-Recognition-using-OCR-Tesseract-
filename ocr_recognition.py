import sys
import cv2
import pytesseract

CONFIDENCE_THRESHOLD = 80  # The 80% Gate mandated by Project 4


# ---------------------------------------------------------------
# STEP 1: THE LOGIC SKELETON - SYSTEMATIC IMAGE PRE-PROCESSING
# ---------------------------------------------------------------
def preprocess_image(image):
    """
    Cleans raw visual data before recognition:
      1. Grayscale Conversion -> collapses the 3D RGB matrix into a
         1D intensity matrix, removing distracting color data.
      2. Gaussian Blur -> smooths micro-imperfections and artifact
         noise.
      3. Adaptive Thresholding (Otsu's Method) -> forces every pixel
         to commit to pure black or white for maximum contrast.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu's method automatically calculates the optimal cutoff
    # intensity instead of a fixed guess.
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresh


# ---------------------------------------------------------------
# STEP 2: RECOGNITION - TRANSFER LEARNING VIA PRE-TRAINED TESSERACT
# ---------------------------------------------------------------
def run_ocr(processed_image, psm: int = 6):
    """
    Runs Tesseract's OCR engine (a pre-trained CNN + Bi-LSTM pipeline)
    on the cleaned image and returns word-level text + confidence data.

    PSM (Page Segmentation Mode) guide from the PDF:
      --psm 3  : Fully automatic (default, mixed layouts)
      --psm 6  : Single uniform block of text (documents/labels)
      --psm 7  : Single text line (headers, plates)
      --psm 11 : Sparse, scattered text (invoices/receipts)
    """
    config = f"--psm {psm}"
    data = pytesseract.image_to_data(
        processed_image, config=config, output_type=pytesseract.Output.DICT
    )
    return data


# ---------------------------------------------------------------
# STEP 3: THE 80% CONFIDENCE GATE + VISUAL CONFIRMATION
# ---------------------------------------------------------------
def apply_confidence_gate(original_image, ocr_data, threshold: int = CONFIDENCE_THRESHOLD):
    """
    Implements the exact gatekeeper logic from the PDF:

        if confidence >= 0.80:
            draw_box_and_label()
        else:
            drop_detection()

    Only words that clear the 80% threshold are drawn and kept in the
    final recognized-text output, minimizing false positives.
    """
    accepted_words = []
    n_boxes = len(ocr_data["text"])

    for i in range(n_boxes):
        word = ocr_data["text"][i].strip()
        conf = int(float(ocr_data["conf"][i])) if ocr_data["conf"][i] != "-1" else -1

        if not word or conf == -1:
            continue

        if conf >= threshold:
            x, y, w, h = (
                ocr_data["left"][i],
                ocr_data["top"][i],
                ocr_data["width"][i],
                ocr_data["height"][i],
            )
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(
                original_image, f"{word} ({conf}%)", (x, max(y - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 1, cv2.LINE_AA
            )
            accepted_words.append((word, conf))
        # else: drop_detection() -> silently discarded, exactly as spec'd

    return original_image, accepted_words


# ---------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------
def main(image_path: str):
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    processed = preprocess_image(original)
    ocr_data = run_ocr(processed, psm=6)
    annotated, accepted_words = apply_confidence_gate(original.copy(), ocr_data)

    print("\n===== OCR RECOGNITION RESULTS =====")
    if not accepted_words:
        print(f"No text cleared the {CONFIDENCE_THRESHOLD}% confidence gate.")
    else:
        full_text = " ".join(w for w, _ in accepted_words)
        avg_conf = sum(c for _, c in accepted_words) / len(accepted_words)
        print(f"Recognized text : {full_text}")
        print(f"Average confidence: {avg_conf:.1f}%")
        print("\nWord-by-word:")
        for word, conf in accepted_words:
            print(f"  - '{word}'  ->  {conf}% confidence")
    print("====================================\n")

    cv2.imwrite("ocr_output.png", annotated)
    cv2.imwrite("ocr_preprocessed.png", processed)
    print("Saved annotated result to ocr_output.png")
    print("Saved pre-processed (grayscale+blur+threshold) image to ocr_preprocessed.png")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_input.png"
    main(path)
