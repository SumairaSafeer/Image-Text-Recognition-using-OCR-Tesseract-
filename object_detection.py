import sys
import cv2
import numpy as np

CONFIDENCE_THRESHOLD = 0.80  # The 80% Gate mandated by Project 4
PROTOTXT = "models/MobileNetSSD_deploy.prototxt"
MODEL = "models/MobileNetSSD_deploy.caffemodel"

# The 20 object classes MobileNet-SSD was pre-trained on (PASCAL VOC)
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
]


# ---------------------------------------------------------------
# STEP 1: TRANSFER LEARNING - LOAD THE PRE-TRAINED NETWORK
# ---------------------------------------------------------------
def load_model(prototxt: str = PROTOTXT, model: str = MODEL):
    """
    Loads MobileNet-SSD: a network that has already learned universal
    visual concepts (edges, shapes, gradients) from millions of images.
    We are not training from scratch - we're inheriting its knowledge.
    """
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    return net


# ---------------------------------------------------------------
# STEP 2: PRE-PROCESSING - 4D BLOB CONSTRUCTION
# ---------------------------------------------------------------
def build_blob(image):
 
    blob = cv2.dnn.blobFromImage(
        image, scalefactor=0.007843, size=(300, 300),
        mean=127.5, swapRB=False, crop=False,
    )
    return blob


# ---------------------------------------------------------------
# STEP 3: DECODING THE MATRIX - BOUNDING BOXES + THE 80% GATE
# ---------------------------------------------------------------
def detect_objects(net, image, threshold: float = CONFIDENCE_THRESHOLD):
    """
    Runs a single forward pass (Single Shot Detector) and decodes the
    network's normalized coordinate output into real pixel bounding
    boxes, applying the exact gatekeeper rule from the PDF:

        if confidence >= 0.80:
            draw_box_and_label()
        else:
            drop_detection()
    """
    (h, w) = image.shape[:2]
    blob = build_blob(image)
    net.setInput(blob)
    detections = net.forward()  # shape: [1, 1, N, 7]

    accepted = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])

        if confidence >= threshold:
            class_id = int(detections[0, 0, i, 1])
            label = CLASSES[class_id] if class_id < len(CLASSES) else "unknown"

            # Translation: normalized coords (0-1) -> real pixel coords
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 200, 0), 2)
            text = f"{label}: {confidence * 100:.1f}%"
            y = startY - 10 if startY - 10 > 10 else startY + 15
            cv2.putText(
                image, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2
            )
            accepted.append((label, confidence))
        # else: drop_detection() -> silently discarded, exactly as spec'd

    return image, accepted


# ---------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------
def main(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    net = load_model()
    annotated, accepted = detect_objects(net, image.copy())

    print("\n===== OBJECT DETECTION RESULTS =====")
    if not accepted:
        print(f"No objects cleared the {int(CONFIDENCE_THRESHOLD * 100)}% confidence gate.")
    else:
        for label, conf in accepted:
            print(f"  - {label:<12} -> {conf * 100:.1f}% confidence")
    print("=====================================\n")

    cv2.imwrite("detection_output.png", annotated)
    print("Saved annotated result to detection_output.png")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "test_photo.png"
    main(path)
