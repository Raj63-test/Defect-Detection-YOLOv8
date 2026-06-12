"""
MVTec AD Bottle Dataset → YOLOv8 Conversion Script
=====================================================
Converts MVTec AD "bottle" category (pixel masks) into
YOLOv8 bounding box format and generates data.yaml.

HOW MVTec BOTTLE IS STRUCTURED (what you download):
----------------------------------------------------
bottle/
├── train/
│   └── good/               ← 209 defect-free training images (normal bottles)
├── test/
│   ├── good/               ← 20  defect-free test images
│   ├── broken_large/       ← 24  images of bottles with large breaks
│   ├── broken_small/       ← 24  images of bottles with small breaks
│   └── contamination/      ← 24  images of bottles with contamination inside
└── ground_truth/
    ├── broken_large/       ← binary PNG masks: white pixels = defect area
    ├── broken_small/
    └── contamination/

WHAT THIS SCRIPT PRODUCES:
---------------------------
yolo_bottle_dataset/
├── images/
│   ├── train/              ← images for training
│   └── val/                ← images for validation
├── labels/
│   ├── train/              ← .txt files with bounding boxes
│   └── val/
└── data.yaml               ← config file YOLOv8 reads

WHAT A YOLO LABEL FILE LOOKS LIKE (e.g. 000.txt):
---------------------------------------------------
0 0.512300 0.423100 0.180400 0.095200
1 0.312000 0.601000 0.092000 0.043000

Each line = one detected object:
  [class_id]  [x_center]  [y_center]  [width]  [height]
  All values are normalised 0.0–1.0 relative to image size.
  class_id 0 = broken_large
  class_id 1 = broken_small
  class_id 2 = contamination

USAGE — paste this into Google Colab:
--------------------------------------
# Step 1: upload this file to Colab
from google.colab import files
files.upload()   # upload convert_mvtec_bottle_to_yolo.py

# Step 2: run it
!python convert_mvtec_bottle_to_yolo.py \
    --mvtec_path  /content/bottle \
    --output_path /content/yolo_bottle_dataset

# Step 3: train
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="/content/yolo_bottle_dataset/data.yaml",
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


# ─────────────────────────────────────────────────────────────────────────────
# BOTTLE DEFECT CLASSES
# Index position = class_id used in YOLO label files
# ─────────────────────────────────────────────────────────────────────────────
DEFECT_CLASSES = [
    "broken_large",    # class_id = 0  (big structural breaks/chips)
    "broken_small",    # class_id = 1  (small chips, hairline cracks)
    "contamination",   # class_id = 2  (dirt/liquid contamination inside)
]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: MASK → BOUNDING BOXES
#
# WHY WE NEED THIS:
# MVTec stores defect locations as PNG images where:
#   - Black pixel (0)   = normal area
#   - White pixel (255) = defect area
#
# YOLOv8 needs rectangles, not pixel masks.
# This function finds the white regions and draws tight rectangles around them.
#
# WHY WE USE CONTOURS (not just min/max pixel coords):
# A bottle can have TWO separate chips — one at the top, one at the bottom.
# If we just take min/max coords, we get one giant box covering the whole bottle.
# Contours find each SEPARATE white region individually → one box per defect.
# ─────────────────────────────────────────────────────────────────────────────
def mask_to_bboxes(mask_path: Path, min_pixel_area: int = 25) -> list[list[float]]:
    """
    Read a binary mask PNG and return YOLO-format bounding boxes.

    Args:
        mask_path      : path to the _mask.png file
        min_pixel_area : ignore white regions smaller than this (noise filter)
                         25 pixels = 5x5 square minimum

    Returns:
        List of [x_center, y_center, width, height] — all in 0.0–1.0 range
    """

    # Read as grayscale (we only care about black vs white, not colour)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        # File exists but OpenCV couldn't read it (corrupted or wrong format)
        print(f"    [WARN] Cannot read mask: {mask_path.name} — skipping")
        return []

    # Image dimensions — needed to normalise coordinates to 0–1
    img_h, img_w = mask.shape

    # ── Threshold ────────────────────────────────────────────────────────────
    # Convert to pure black/white in case of any grey anti-aliasing pixels
    # Pixels > 127 → 255 (white/defect), pixels ≤ 127 → 0 (black/normal)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # ── Find contours ─────────────────────────────────────────────────────────
    # RETR_EXTERNAL  = only outer boundaries (no nested contours)
    # CHAIN_APPROX_SIMPLE = compress horizontal/vertical lines (saves memory)
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    bboxes = []

    for contour in contours:

        # ── Filter tiny noise ────────────────────────────────────────────────
        # Masks sometimes have stray white pixels from JPEG compression artifacts.
        # We ignore any region smaller than min_pixel_area pixels.
        area = cv2.contourArea(contour)
        if area < min_pixel_area:
            continue

        # ── Get bounding rectangle ───────────────────────────────────────────
        # cv2.boundingRect returns pixel coordinates: (x, y, width, height)
        # where (x, y) is the TOP-LEFT corner
        x, y, w, h = cv2.boundingRect(contour)

        # ── Add padding ──────────────────────────────────────────────────────
        # Tight bounding boxes that clip the defect edge hurt recall.
        # We add 2% of the box dimensions as padding on each side.
        pad_x = max(2, int(w * 0.02))   # at least 2px, or 2% of width
        pad_y = max(2, int(h * 0.02))

        # Clamp to image boundaries so we don't go negative or exceed image size
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        w = min(img_w - x, w + 2 * pad_x)
        h = min(img_h - y, h + 2 * pad_y)

        # ── Convert to YOLO format ───────────────────────────────────────────
        # YOLO wants: [cx, cy, w, h] all normalised to image size
        # cx/cy = center of the box (not top-left corner)
        cx = (x + w / 2) / img_w   # 0.0 = far left,  1.0 = far right
        cy = (y + h / 2) / img_h   # 0.0 = very top,  1.0 = very bottom
        nw = w / img_w              # box width  as fraction of image width
        nh = h / img_h              # box height as fraction of image height

        bboxes.append([cx, cy, nw, nh])

    return bboxes


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: WRITE YOLO LABEL FILE
#
# One .txt file per image. Each line describes one object:
#   class_id  cx  cy  w  h
#
# IMPORTANT: If the image has NO defects (good/ images), we write an EMPTY
# .txt file — not a missing file. Missing = ignored by YOLO. Empty = "no objects."
# This distinction matters for training quality.
# ─────────────────────────────────────────────────────────────────────────────
def write_label_file(label_path: Path, class_id: int, bboxes: list[list[float]]):
    """Write YOLO annotation file. One bbox per line."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for cx, cy, w, h in bboxes:
            # 6 decimal places is standard — more is unnecessary
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def write_empty_label(label_path: Path):
    """Write an empty label file = image with no objects (negative example)."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("")


def copy_image(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONVERSION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def convert(mvtec_path: str, output_path: str, val_split: float = 0.2, seed: int = 42):

    mvtec   = Path(mvtec_path)
    out_dir = Path(output_path)

    # Create output folder structure
    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    random.seed(seed)   # fix seed → same split every time you run

    stats = {cls: {"found": 0, "skipped": 0} for cls in DEFECT_CLASSES}
    count = {"train": 0, "val": 0}

    # ─────────────────────────────────────────────────────────────────────────
    # PART A: Defective images
    # For each defect class: read image + read mask → extract bboxes → save
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[1/3] Converting defective images...")
    print(f"      Classes: {DEFECT_CLASSES}\n")

    all_defect_samples = []   # collect all then shuffle before splitting

    for class_id, defect_class in enumerate(DEFECT_CLASSES):
        img_dir  = mvtec / "test" / defect_class
        mask_dir = mvtec / "ground_truth" / defect_class

        if not img_dir.exists():
            print(f"  [SKIP] No folder found for '{defect_class}' in test/")
            continue

        image_files = sorted(img_dir.glob("*.png"))
        print(f"  {defect_class} (class_id={class_id}): {len(image_files)} images")

        for img_path in image_files:
            # MVTec mask filename = original filename + "_mask.png"
            # e.g. 000.png → 000_mask.png
            mask_name = img_path.stem + "_mask.png"
            mask_path = mask_dir / mask_name

            if not mask_path.exists():
                print(f"    [WARN] Missing mask: {mask_name}")
                stats[defect_class]["skipped"] += 1
                continue

            bboxes = mask_to_bboxes(mask_path)

            if not bboxes:
                print(f"    [WARN] No valid bbox from {mask_name} (all regions too small)")
                stats[defect_class]["skipped"] += 1
                continue

            all_defect_samples.append((img_path, class_id, defect_class, bboxes))
            stats[defect_class]["found"] += 1

    # Shuffle then split into train/val
    random.shuffle(all_defect_samples)
    split_idx     = int(len(all_defect_samples) * (1 - val_split))
    train_defects = all_defect_samples[:split_idx]
    val_defects   = all_defect_samples[split_idx:]

    for img_path, class_id, defect_class, bboxes in train_defects:
        # Use class name as prefix to avoid filename collisions between classes
        stem      = f"{defect_class}_{img_path.stem}"
        dst_img   = out_dir / "images" / "train" / (stem + ".png")
        dst_label = out_dir / "labels" / "train" / (stem + ".txt")
        copy_image(img_path, dst_img)
        write_label_file(dst_label, class_id, bboxes)
        count["train"] += 1

    for img_path, class_id, defect_class, bboxes in val_defects:
        stem      = f"{defect_class}_{img_path.stem}"
        dst_img   = out_dir / "images" / "val" / (stem + ".png")
        dst_label = out_dir / "labels" / "val" / (stem + ".txt")
        copy_image(img_path, dst_img)
        write_label_file(dst_label, class_id, bboxes)
        count["val"] += 1

    # ─────────────────────────────────────────────────────────────────────────
    # PART B: Defect-free (good) images as negative examples
    #
    # WHY THIS IS ESSENTIAL:
    # Without negative examples, YOLOv8 has only ever seen defective bottles
    # during training. It will draw boxes on EVERYTHING — including perfectly
    # normal bottles — because it has never learned what "normal" looks like.
    #
    # We add all good/ training images with EMPTY label files.
    # Empty label file = "this image contains zero objects."
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2/3] Adding defect-free images as negative examples...")

    good_train_imgs = sorted((mvtec / "train" / "good").glob("*.png"))
    good_test_imgs  = sorted((mvtec / "test"  / "good").glob("*.png"))
    all_good        = good_train_imgs + good_test_imgs

    print(f"  Found {len(good_train_imgs)} good train + {len(good_test_imgs)} good test = {len(all_good)} total")

    random.shuffle(all_good)
    good_split    = int(len(all_good) * (1 - val_split))
    good_train    = all_good[:good_split]
    good_val      = all_good[good_split:]

    good_count = {"train": 0, "val": 0}

    for img_path in good_train:
        dst_img   = out_dir / "images" / "train" / f"good_{img_path.name}"
        dst_label = out_dir / "labels" / "train" / f"good_{img_path.stem}.txt"
        copy_image(img_path, dst_img)
        write_empty_label(dst_label)
        count["train"]       += 1
        good_count["train"]  += 1

    for img_path in good_val:
        dst_img   = out_dir / "images" / "val" / f"good_{img_path.name}"
        dst_label = out_dir / "labels" / "val" / f"good_{img_path.stem}.txt"
        copy_image(img_path, dst_img)
        write_empty_label(dst_label)
        count["val"]       += 1
        good_count["val"]  += 1

    # ─────────────────────────────────────────────────────────────────────────
    # PART C: Write data.yaml
    #
    # This is the config file YOLOv8 reads before training.
    # It tells YOLOv8:
    #   - where to find images
    #   - how many classes there are
    #   - what each class is called
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[3/3] Writing data.yaml...")

    yaml_data = {
        "path"  : str(out_dir.resolve()),   # absolute path — no ambiguity
        "train" : "images/train",
        "val"   : "images/val",
        "nc"    : len(DEFECT_CLASSES),      # number of classes = 3
        "names" : DEFECT_CLASSES,           # class names list
    }

    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    # Print the yaml content so you can verify it visually in Colab output
    print(f"\n  data.yaml contents:")
    print(f"  {'─'*35}")
    for key, val in yaml_data.items():
        print(f"  {key}: {val}")
    print(f"  {'─'*35}")

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    total = count["train"] + count["val"]
    print("\n" + "=" * 55)
    print("  CONVERSION COMPLETE")
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
    print(f"  model = YOLO('yolov8n.pt')")
    print(f"  model.train(data='{yaml_path}', epochs=50, imgsz=640, batch=16)")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert MVTec AD bottle dataset to YOLOv8 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default Colab paths:
  python convert_mvtec_bottle_to_yolo.py

  # Custom paths:
  python convert_mvtec_bottle_to_yolo.py \\
      --mvtec_path  /content/bottle \\
      --output_path /content/yolo_bottle_dataset \\
      --val_split   0.2
        """
    )
    parser.add_argument("--mvtec_path",  default="/content/bottle",             help="Path to extracted bottle/ folder")
    parser.add_argument("--output_path", default="/content/yolo_bottle_dataset", help="Where to write YOLOv8 dataset")
    parser.add_argument("--val_split",   default=0.2, type=float,               help="Validation fraction (default 0.2)")
    parser.add_argument("--seed",        default=42,  type=int,                 help="Random seed (default 42)")
    args = parser.parse_args()

    convert(args.mvtec_path, args.output_path, args.val_split, args.seed)