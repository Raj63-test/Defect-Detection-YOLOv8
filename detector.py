"""
detector.py
-----------
Loads the trained YOLOv8 model and runs inference/tracking on images and videos.
Designed for both Streamlit and standalone usage.
"""

from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import time
import subprocess

# Cache for loaded models to avoid reloading weights repeatedly
_loaded_models = {}

def load_model(weights_path: str) -> YOLO:
    """Load a YOLOv8 model from the given path (uses caching)."""
    # Resolve relative paths relative to this script directory if not found in CWD
    if not os.path.isabs(weights_path) and not os.path.exists(weights_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        resolved_path = os.path.join(script_dir, weights_path)
        if os.path.exists(resolved_path):
            weights_path = resolved_path

    if weights_path not in _loaded_models:
        print(f"Loading YOLO model from {weights_path}...")
        # Check if file exists; if not, fall back to downloading standard model if it's a model name
        if not os.path.exists(weights_path) and not weights_path.endswith('.pt'):
            # E.g. yolov8n.pt or yolov8n-seg.pt
            _loaded_models[weights_path] = YOLO(weights_path)
        else:
            _loaded_models[weights_path] = YOLO(weights_path)
    return _loaded_models[weights_path]



def get_class_info(model: YOLO) -> dict:
    """Generate display labels and colors dynamically based on the model classes."""
    import random
    presets = {
        "broken_large": (220, 50, 50),
        "broken_small": (255, 140, 0),
        "contamination": (255, 215, 0),
        "bent": (50, 205, 50),
        "color": (0, 191, 255),
        "flip": (138, 43, 226),
        "scratch": (255, 20, 147),
        "thread": (0, 255, 127),
        "fire": (255, 69, 0),
        "smoke": (128, 128, 128)
    }
    class_info = {}
    for cid, name in model.names.items():
        color = presets.get(name)
        if not color:
            random.seed(cid)
            color = (random.randint(50, 220), random.randint(50, 220), random.randint(50, 220))
        class_info[cid] = {"label": name, "color": color}
    return class_info


def detect_defects(
    image: Image.Image,
    conf_threshold: float = 0.25,
    iou_threshold:  float = 0.45,
    model = None,
    show_boxes: bool = True
) -> tuple[Image.Image, list[dict], str]:
    """
    Run YOLOv8 inference on a PIL image.
    Supports both detection and segmentation models.
    """
    # Fallback to local default model if not provided
    if model is None:
        model = load_model("best.pt")

    img_array = np.array(image)

    # Run inference
    results = model(
        img_array,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )[0]

    # Draw annotations (boxes/masks)
    annotated_array = results.plot(
        line_width=2,
        font_size=12,
        boxes=show_boxes
    )
    annotated_image = Image.fromarray(annotated_array)

    # Parse detections into structured dictionary
    detections = []
    class_info = get_class_info(model)

    if results.boxes is not None:
        for box in results.boxes:
            class_id = int(box.cls.item())
            confidence = round(float(box.conf.item()), 3)

            # Bounding box coordinates
            x1, y1, x2, y2 = [round(v) for v in box.xyxy[0].tolist()]
            box_w = x2 - x1
            box_h = y2 - y1

            label = class_info.get(class_id, {}).get("label") or model.names.get(class_id, f"class_{class_id}")

            detection_entry = {
                "class_id"  : class_id,
                "label"     : label,
                "confidence": confidence,
                "bbox_pixels": [x1, y1, x2, y2],
                "bbox_size" : f"{box_w}×{box_h}px",
            }

            # If segmentation masks are available, extract mask area or statistics if needed
            if hasattr(results, 'masks') and results.masks is not None:
                detection_entry["has_mask"] = True

            detections.append(detection_entry)

    # Sort by confidence descending
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    # Build summary text
    if not detections:
        summary_text = "✅ No anomalies/defects detected."
    else:
        lines = [f"⚠️  {len(detections)} detection(s) found:\n"]
        for i, d in enumerate(detections, 1):
            mask_tag = " [with Mask]" if d.get("has_mask") else ""
            lines.append(
                f"  {i}. {d['label']}{mask_tag}"
                f"  —  confidence: {d['confidence']:.1%}"
                f"  —  size: {d['bbox_size']}"
            )
        summary_text = "\n".join(lines)

    return annotated_image, detections, summary_text


def process_video(
    model: YOLO,
    input_path: str,
    output_path: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    progress_callback = None,
    show_boxes: bool = True
) -> dict:
    """
    Read input video, track defects frame-by-frame, save annotated output video,
    and convert to web-playable H.264 format using ffmpeg.
    """
    cap = cv2_cap = cv2 = None
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV (cv2) is not installed in the current environment.")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open input video file: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    temp_output_path = output_path + ".temp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

    frame_idx = 0
    start_time = time.time()
    total_inference_time = 0.0
    total_objects_tracked = 0
    unique_ids = set()

    # Track frame times to compute exact FPS metrics
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            t_start = time.time()
            # Perform ByteTrack tracking
            results = model.track(
                frame,
                persist=True,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )[0]
            t_end = time.time()
            total_inference_time += (t_end - t_start)

            # Draw tracking bounding boxes/masks
            annotated_frame = results.plot(boxes=show_boxes)
            out.write(annotated_frame)

            # Track unique IDs
            if results.boxes is not None and results.boxes.id is not None:
                ids = results.boxes.id.int().tolist()
                unique_ids.update(ids)
                total_objects_tracked = max(total_objects_tracked, len(unique_ids))

            if progress_callback:
                progress_callback(frame_idx, total_frames)
    finally:
        cap.release()
        out.release()

    total_time = time.time() - start_time
    avg_fps = frame_idx / total_time if total_time > 0 else 0.0
    avg_inference_latency = (total_inference_time / frame_idx * 1000) if frame_idx > 0 else 0.0

    # Convert the temp output video to web-playable H.264 using FFmpeg
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except OSError:
            pass

    try:
        cmd = [
            'ffmpeg', '-y', '-i', temp_output_path,
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',
            output_path
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
    except Exception as e:
        print(f"FFmpeg H.264 re-encoding failed: {e}")
        # Fallback to the temp output file as the final output
        if os.path.exists(temp_output_path):
            os.rename(temp_output_path, output_path)

    return {
        "total_frames": frame_idx,
        "avg_fps": round(avg_fps, 2),
        "avg_inference_latency_ms": round(avg_inference_latency, 2),
        "total_time_sec": round(total_time, 2),
        "unique_objects_tracked": total_objects_tracked,
        "task": model.task
    }


def get_model_info(model = None) -> dict:
    """Return basic info about the loaded model."""
    if model is None:
        model = load_model("best.pt")
    return {
        "num_classes" : len(model.names),
        "class_names" : list(model.names.values()),
        "task"        : model.task,
    }


if __name__ == "__main__":
    print("\n── Model info ──────────────────────────────")
    model = load_model("best.pt")
    info = get_model_info(model)
    for k, v in info.items():
        print(f"  {k}: {v}")

    test_dir = "test_images"
    if os.path.exists(test_dir):
        test_files = [f for f in os.listdir(test_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
        if test_files:
            test_path = os.path.join(test_dir, test_files[0])
            print(f"\n── Running test inference on: {test_path}")
            img = Image.open(test_path).convert("RGB")
            annotated, detections, summary = detect_defects(img, model=model)
            print(summary)
            print(f"\nAnnotated image size: {annotated.size}")
            print("detector.py is working correctly ✓")
        else:
            print("\nNo test images found in test_images/ — skipping inference test.")
    else:
        print("Model loaded successfully ✓")