---
title: Defect Detection YOLOv8
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
---

# 🔍 Real-Time Surface Defect Detection

> A computer vision system that detects surface defects on industrial products in real time using a custom-trained YOLOv8 model. Trained on the **MVTec AD** benchmark dataset across **2 product categories** and **8 defect classes**.

---

Defect Detection Demo

---

## 📋 Table of Contents

- [Overview](#Overview)
- [Model Performance](#Model-Performance)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [How It Works](#how-it-works)
- [Results](#results)
- [What I'd Improve Next](#what-id-improve-next)
- [Tech Stack](#tech-stack)

---

## Overview

Surface defect detection is a critical step in industrial quality control. Manual inspection is slow, inconsistent, and expensive. This project trains a YOLOv8 object detection model to automatically identify defects on **bottles** and **metal nuts** — detecting issues like cracks, chips, contamination, and surface scratches.

The model outputs bounding boxes with class labels and confidence scores on each image, making it directly usable in a real-time inspection pipeline.

**Key highlights:**

- Trained from scratch on MVTec AD — the standard industrial anomaly detection benchmark
- Handles 8 distinct defect types across 2 product categories
- Interactive Gradio demo for live image upload and detection
- Includes negative examples (defect-free images) to reduce false positives
- Per-class evaluation with precision, recall, and mAP metrics

## Model Performance

Evaluated on the held-out validation set (20% of total data, never seen during training).

### Overall metrics


| Metric        | Score |
| ------------- | ----- |
| **mAP50**     | 0.782 |
| **mAP50-95**  | 0.521 |
| **Precision** | 0.814 |
| **Recall**    | 0.763 |


### Training curves

Training results
Confusion Matrix

---

## Dataset

**MVTec AD** — the standard benchmark for industrial anomaly detection, published at CVPR 2019 by MVTec Software GmbH.


| Property             | Value                                                                                           |
| -------------------- | ----------------------------------------------------------------------------------------------- |
| Source               | [mvtec.com/company/research/datasets](https://www.mvtec.com/company/research/datasets/mvtec-ad) |
| Categories used      | Bottle, Metal nut                                                                               |
| Total defect classes | 8                                                                                               |
| Train images         | ~360                                                                                            |
| Val images           | ~72                                                                                             |
| Image resolution     | 900×900 px                                                                                      |
| License              | Free for research use                                                                           |


### Classes

```
Bottle:     0-broken_large  1-broken_small  2-contamination
Metal nut:  3-bent          4-color         5-flip   6-scratch   7-thread
```

### Preprocessing

- Pixel masks converted to YOLO bounding box format using contour detection
- 80/20 train/val split with fixed random seed (42) for reproducibility
- Defect-free images included as negative examples to reduce false positives
- Augmentation: horizontal flip, HSV shift, rotation (±10°), brightness contrast

> The full dataset is not included in this repo (4.7 GB). Download from the MVTec website and run the conversion script — see [Quickstart](#quickstart).

---

## Project Structure

```
defect-detection-yolov8/
│
├── app.py                          # Gradio demo interface
├── detector.py                     # YOLOv8 inference logic
├── requirements.txt                # Python dependencies
├── best.pt                         # Trained model weights (6 MB)
│
├── convert_mvtec_bottle_to_yolo.py     # Bottle dataset conversion script
├── convert_mvtec_metalnut_to_yolo.py   # Metal nut dataset conversion script
│
├── test_images/                    # Sample images for demo
│   ├── sample1.png
│   └── sample2.png
│
└── results/                        # Evaluation outputs
    ├── results.png                 # Training metric curves
    ├── PR_curve.png                # Precision-recall curve
    └── confusion_matrix_normalized.png
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/defect-detection-yolov8.git
cd defect-detection-yolov8
pip install -r requirements.txt
```

### 2. Run the demo (weights included)

```bash
python app.py
# Opens at http://localhost:7860
# Upload any bottle or metal nut image to see detections
```

## How It Works

```
Input image
    ↓
YOLOv8n backbone (pretrained on COCO, fine-tuned on MVTec)
    ↓
Detection head → bounding boxes + class labels + confidence scores
    ↓
results.plot() → annotated output image
    ↓
Gradio UI displays result
```

***Why YOLOv8?***
YOLOv8 is a single-stage detector — it predicts bounding boxes and class probabilities in one forward pass, making it fast enough for real-time inspection. Compared to two-stage detectors like Faster R-CNN, it's simpler to train and deploy while achieving competitive mAP on small datasets.

***Why MVTec AD?***
MVTec AD is the standard benchmark for industrial defect detection, used in hundreds of research papers. Using it makes results directly comparable to published work and demonstrates awareness of real-world industrial datasets.

---

## Results

### Inference examples

Inference result 1

*Bounding boxes drawn with class label and confidence score. Confidence threshold: 0.25.*

---

## What I'd Improve Next

- **Deployment** — containerise with Docker, deploy inference service on Azure AKS with horizontal pod autoscaling and GPU node pools
- **MLOps pipeline** — add MLflow experiment tracking, CI/CD via Azure DevOps with mAP quality gate before any model promotion
- **More categories** — extend to all 15 MVTec AD categories for a fully general industrial inspection system
- **ONNX export** — export to ONNX with INT8 quantisation for 3–4× inference speedup without accuracy loss
- **Data drift monitoring** — add Evidently AI to detect when production image distribution shifts and trigger automatic retraining

---

## Tech Stack


| Tool                                                                 | Purpose                    |
| -------------------------------------------------------------------- | -------------------------- |
| [YOLOv8](https://github.com/ultralytics/ultralytics)                 | Object detection model     |
| [PyTorch](https://pytorch.org)                                       | Deep learning framework    |
| [Gradio](https://gradio.app)                                         | Interactive demo interface |
| [OpenCV](https://opencv.org)                                         | Mask-to-bbox conversion    |
| [Google Colab](https://colab.research.google.com)                    | Free GPU training (T4)     |
| [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) | Industrial defect dataset  |


---

## License

MIT License — free to use for personal and commercial projects. See [LICENSE](LICENSE) for details.

---

## Author

**Rajan Niranjan** — Machine Learning/Computer Vision  Engineer

[GitHub](https://github.com/Raj63-test)
[LinkedIn](https://www.linkedin.com/in/rajan-niranjan-193568179/)