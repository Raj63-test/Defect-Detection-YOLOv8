"""
detector.py
-----------
Loads the trained YOLOv8 model (best.pt) and runs inference on images.
This file is imported by app.py — it never trains, only predicts.

Usage (from app.py):
    from detector import detect_defects
    result_image, detections = detect_defects(pil_image)
"""

from ultralytics import YOLO
from PIL import Image
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
#
# YOLO("best.pt") loads once when the module is imported.
# This means the model is loaded into memory ONCE when app.py starts,
# not on every single request — which is important for performance.
#
# If best.pt is in the same folder as detector.py, just "best.pt" works.
# If it's somewhere else, use the full path: YOLO("/path/to/best.pt")
# ─────────────────────────────────────────────────────────────────────────────
print("Loading YOLOv8 model...")
model = YOLO("best.pt")
print(f"Model loaded. Classes: {list(model.names.values())}")


# ─────────────────────────────────────────────────────────────────────────────
# CLASS LABELS AND COLOURS
#
# These map each class_id to a human-readable label and a display colour.
# Colours are used by results.plot() to draw boxes — one colour per class
# makes it easy to tell defect types apart visually.
# ─────────────────────────────────────────────────────────────────────────────
CLASS_INFO = {
    # Bottle defects
    0: {"label": "broken_large",  "color": (220, 50,  50)},   # red
    1: {"label": "broken_small",  "color": (255, 140,  0)},   # orange
    2: {"label": "contamination", "color": (255, 215,  0)},   # yellow

    # Metal nut defects
    3: {"label": "bent",          "color": (50,  205, 50)},   # green
    4: {"label": "color",         "color": (0,   191, 255)},  # sky blue
    5: {"label": "flip",          "color": (138,  43, 226)},  # purple
    6: {"label": "scratch",       "color": (255,  20, 147)},  # pink
    7: {"label": "thread",        "color": (0,   255, 127)},  # spring green
}


def detect_defects(
    image: Image.Image,
    conf_threshold: float = 0.25,
    iou_threshold:  float = 0.45,
) -> tuple[Image.Image, list[dict], str]:
    """
    Run YOLOv8 inference on a PIL image.

    Args:
        image          : PIL Image (RGB) — the product photo to inspect
        conf_threshold : minimum confidence to show a detection (0.0–1.0)
                         0.25 = show detections the model is 25%+ confident about
                         increase to 0.5 if you're getting too many false positives
        iou_threshold  : overlap threshold for NMS (Non-Maximum Suppression)
                         0.45 is the YOLOv8 default — usually fine to leave as is

    Returns:
        annotated_image : PIL Image with bounding boxes drawn on it
        detections      : list of dicts, one per detected defect
        summary_text    : plain text summary for display in the Gradio UI
    """

    # ── Convert PIL → numpy array ─────────────────────────────────────────────
    # YOLOv8 accepts numpy arrays (H, W, 3) in RGB format.
    # PIL images are already RGB so no channel conversion needed.
    img_array = np.array(image)

    # ── Run inference ─────────────────────────────────────────────────────────
    # model() returns a list of Results objects — one per image.
    # Since we pass a single image, we take index [0].
    results = model(
        img_array,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,          # suppress per-inference console output
    )[0]

    # ── Draw bounding boxes ───────────────────────────────────────────────────
    # results.plot() returns a numpy array with boxes, labels, and
    # confidence scores drawn directly onto the image.
    # We convert it back to PIL for Gradio to display.
    annotated_array = results.plot(
        line_width=2,           # box border thickness in pixels
        font_size=12,           # label font size
    )
    annotated_image = Image.fromarray(annotated_array)

    # ── Parse detections into structured dicts ────────────────────────────────
    # results.boxes contains all detected bounding boxes.
    # Each box has: cls (class id), conf (confidence), xyxy (pixel coords)
    detections = []

    for box in results.boxes:
        class_id   = int(box.cls.item())
        confidence = round(float(box.conf.item()), 3)

        # Pixel coordinates of the bounding box
        x1, y1, x2, y2 = [round(v) for v in box.xyxy[0].tolist()]

        # Box dimensions in pixels
        box_w = x2 - x1
        box_h = y2 - y1

        # Get label from our CLASS_INFO dict
        # Fall back to model.names if class_id not in CLASS_INFO (safety net)
        label = CLASS_INFO.get(class_id, {}).get("label") or model.names.get(class_id, f"class_{class_id}")

        detections.append({
            "class_id"  : class_id,
            "label"     : label,
            "confidence": confidence,
            "bbox_pixels": [x1, y1, x2, y2],   # top-left, bottom-right
            "bbox_size" : f"{box_w}×{box_h}px",
        })

    # Sort by confidence descending — highest confidence defect first
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    # ── Build summary text ────────────────────────────────────────────────────
    # This string is shown in the Gradio textbox next to the annotated image.
    if not detections:
        summary_text = "✅ No defects detected — product appears normal."
    else:
        lines = [f"⚠️  {len(detections)} defect(s) detected:\n"]
        for i, d in enumerate(detections, 1):
            lines.append(
                f"  {i}. {d['label']}"
                f"  —  confidence: {d['confidence']:.1%}"
                f"  —  size: {d['bbox_size']}"
            )
        summary_text = "\n".join(lines)

    return annotated_image, detections, summary_text


def get_model_info() -> dict:
    """
    Return basic info about the loaded model.
    Used by app.py to display model details in the Gradio UI.
    """
    return {
        "model_file"  : "best.pt",
        "num_classes" : len(model.names),
        "class_names" : list(model.names.values()),
        "task"        : model.task,
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# Run this file directly to verify the model loads and inference works:
#   python detector.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    print("\n── Model info ──────────────────────────────")
    info = get_model_info()
    for k, v in info.items():
        print(f"  {k}: {v}")

    # Try inference on a test image if one exists
    test_dir = "test_images"
    if os.path.exists(test_dir):
        test_files = [f for f in os.listdir(test_dir) if f.endswith((".png", ".jpg"))]
        if test_files:
            test_path = os.path.join(test_dir, test_files[0])
            print(f"\n── Running test inference on: {test_path}")
            img = Image.open(test_path).convert("RGB")
            annotated, detections, summary = detect_defects(img)
            print(summary)
            print(f"\nAnnotated image size: {annotated.size}")
            print("detector.py is working correctly ✓")
        else:
            print("\nNo test images found in test_images/ — skipping inference test.")
            print("Add a .png or .jpg file to test_images/ and run again.")
    else:
        print("\ntest_images/ folder not found — skipping inference test.")
        print("Model loaded successfully ✓")