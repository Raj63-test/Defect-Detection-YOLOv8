"""
MVTec AD Bottle Dataset → YOLOv8-Seg Conversion Script
=====================================================
Converts MVTec AD "bottle" category (pixel masks) into
YOLOv8-Seg polygon segmentation format and generates data.yaml.

WHAT THIS SCRIPT PRODUCES:
---------------------------
yolo_bottle_seg_dataset/
├── images/
│   ├── train/              ← images for training
│   └── val/                ← images for validation
├── labels/
│   ├── train/              ← .txt files with polygon segment contours
│   └── val/
└── data.yaml               ← config file YOLOv8-Seg reads

WHAT A YOLO SEG LABEL FILE LOOKS LIKE (e.g. 000.txt):
---------------------------------------------------
0 0.512000 0.423000 0.520000 0.430000 0.530000 0.440000 ...
Each line = one detected object in polygon format:
  [class_id] [x1] [y1] [x2] [y2] ... [xn] [yn]
  All values are normalised 0.0–1.0 relative to image size.

USAGE — paste this into Google Colab:
--------------------------------------
# Step 1: upload this file to Colab
from google.colab import files
files.upload()   # upload convert_mvtec_bottle_to_yolo_seg.py

# Step 2: run it
!python convert_mvtec_bottle_to_yolo_seg.py \
    --mvtec_path  /content/bottle \
    --output_path /content/yolo_bottle_seg_dataset

# Step 3: train YOLOv8 Segmentation
from ultralytics import YOLO
model = YOLO("yolov8n-seg.pt")
model.train(data="/content/yolo_bottle_seg_dataset/data.yaml",
            epochs=50, imgsz=640, batch=16)
"""

import os
import shutil
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml

# BOTTLE DEFECT CLASSES
DEFECT_CLASSES = [
    "broken_large",    # class_id = 0
    "broken_small",    # class_id = 1
    "contamination",   # class_id = 2
]

def mask_to_polygons(mask_path: Path, min_pixel_area: int = 25) -> list[list[float]]:
    """
    Read a binary mask PNG and return YOLOv8-Seg polygon contours.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"    [WARN] Cannot read mask: {mask_path.name} — skipping")
        return []

    img_h, img_w = mask.shape

    # Binarize mask
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_TC89_KCOS  # Gives high-quality polygons with fewer vertices
    )

    polygons = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_pixel_area:
            continue

        # Convert contour coordinates to normalized 0-1 values
        poly = []
        for point in contour:
            px, py = point[0]
            # Normalise
            nx = px / img_w
            ny = py / img_h
            poly.extend([nx, ny])
        
        # YOLOv8-seg needs at least 3 points (6 coords) to form a polygon
        if len(poly) >= 6:
            polygons.append(poly)

    return polygons


def write_label_file(label_path: Path, class_id: int, polygons: list[list[float]]):
    """Write YOLO segmentation annotation file. One polygon per line."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for poly in polygons:
            poly_str = " ".join([f"{coord:.6f}" for coord in poly])
            f.write(f"{class_id} {poly_str}\n")


def write_empty_label(label_path: Path):
    """Write empty label file (negative example)."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("")


def copy_image(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def convert(mvtec_path: str, output_path: str, val_split: float = 0.2, seed: int = 42):
    mvtec = Path(mvtec_path)
    out_dir = Path(output_path)

    # Create output folders
    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    stats = {cls: {"found": 0, "skipped": 0} for cls in DEFECT_CLASSES}
    count = {"train": 0, "val": 0}

    print("\n[1/3] Converting defective images into segmentation polygons...")
    
    all_defect_samples = []

    for class_id, defect_class in enumerate(DEFECT_CLASSES):
        img_dir = mvtec / "test" / defect_class
        mask_dir = mvtec / "ground_truth" / defect_class

        if not img_dir.exists():
            print(f"  [SKIP] No folder found for '{defect_class}' in test/")
            continue

        image_files = sorted(img_dir.glob("*.png"))
        print(f"  {defect_class} (class_id={class_id}): {len(image_files)} images")

        for img_path in image_files:
            mask_name = img_path.stem + "_mask.png"
            mask_path = mask_dir / mask_name

            if not mask_path.exists():
                print(f"    [WARN] Missing mask: {mask_name}")
                stats[defect_class]["skipped"] += 1
                continue

            polygons = mask_to_polygons(mask_path)

            if not polygons:
                print(f"    [WARN] No valid polygon from {mask_name}")
                stats[defect_class]["skipped"] += 1
                continue

            all_defect_samples.append((img_path, class_id, defect_class, polygons))
            stats[defect_class]["found"] += 1

    # Shuffle and split into train/val
    random.shuffle(all_defect_samples)
    split_idx = int(len(all_defect_samples) * (1 - val_split))
    train_defects = all_defect_samples[:split_idx]
    val_defects = all_defect_samples[split_idx:]

    # Write training labels and images
    for img_path, class_id, defect_class, polygons in train_defects:
        stem = f"{defect_class}_{img_path.stem}"
        dst_img = out_dir / "images" / "train" / (stem + ".png")
        dst_label = out_dir / "labels" / "train" / (stem + ".txt")
        copy_image(img_path, dst_img)
        write_label_file(dst_label, class_id, polygons)
        count["train"] += 1

    # Write validation labels and images
    for img_path, class_id, defect_class, polygons in val_defects:
        stem = f"{defect_class}_{img_path.stem}"
        dst_img = out_dir / "images" / "val" / (stem + ".png")
        dst_label = out_dir / "labels" / "val" / (stem + ".txt")
        copy_image(img_path, dst_img)
        write_label_file(dst_label, class_id, polygons)
        count["val"] += 1

    # Add negative (good) images
    print("\n[2/3] Adding defect-free images as negative examples...")
    good_train_imgs = sorted((mvtec / "train" / "good").glob("*.png"))
    good_test_imgs = sorted((mvtec / "test"  / "good").glob("*.png"))
    all_good = good_train_imgs + good_test_imgs

    print(f"  Found {len(all_good)} defect-free images.")
    random.shuffle(all_good)
    good_split = int(len(all_good) * (1 - val_split))
    good_train = all_good[:good_split]
    good_val = all_good[good_split:]

    good_count = {"train": 0, "val": 0}

    for img_path in good_train:
        dst_img = out_dir / "images" / "train" / f"good_{img_path.name}"
        dst_label = out_dir / "labels" / "train" / f"good_{img_path.stem}.txt"
        copy_image(img_path, dst_img)
        write_empty_label(dst_label)
        count["train"] += 1
        good_count["train"] += 1

    for img_path in good_val:
        dst_img = out_dir / "images" / "val" / f"good_{img_path.name}"
        dst_label = out_dir / "labels" / "val" / f"good_{img_path.stem}.txt"
        copy_image(img_path, dst_img)
        write_empty_label(dst_label)
        count["val"] += 1
        good_count["val"] += 1

    # Write data.yaml
    print("\n[3/3] Writing data.yaml...")
    yaml_data = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(DEFECT_CLASSES),
        "names": DEFECT_CLASSES,
    }

    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    total = count["train"] + count["val"]
    print("\n" + "=" * 55)
    print("  CONVERSION TO SEGMENTATION FORMAT COMPLETE")
    print("=" * 55)
    print(f"  Output path : {out_dir.resolve()}")
    print(f"  Total images: {total}")
    print(f"    Train      : {count['train']}  ({count['train']/total*100:.0f}%)")
    print(f"    Val        : {count['val']}   ({count['val']/total*100:.0f}%)")
    print()
    print("  Defect class breakdown:")
    for cls in DEFECT_CLASSES:
        s = stats[cls]
        print(f"    [{DEFECT_CLASSES.index(cls)}] {cls:<16}: {s['found']} found, {s['skipped']} skipped")
    print(f"    [neg] good images    : {good_count['train']} train / {good_count['val']} val")
    print()
    print("  Ready to train. Paste in Colab:")
    print(f"  from ultralytics import YOLO")
    print(f"  model = YOLO('yolov8n-seg.pt')")
    print(f"  model.train(data='{yaml_path}', epochs=50, imgsz=640, batch=16)")
    print("=" * 55)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MVTec AD bottle dataset to YOLOv8-Seg format")
    parser.add_argument("--mvtec_path", default="/content/bottle", help="Path to extracted bottle/ folder")
    parser.add_argument("--output_path", default="/content/yolo_bottle_seg_dataset", help="Where to write YOLOv8 dataset")
    parser.add_argument("--val_split", default=0.2, type=float, help="Validation fraction (default 0.2)")
    parser.add_argument("--seed", default=42, type=int, help="Random seed (default 42)")
    args = parser.parse_args()

    convert(args.mvtec_path, args.output_path, args.val_split, args.seed)
