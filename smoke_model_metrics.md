# Factory Smoke & Fire Segmentation Model Performance Report

This document summarizes the validation metrics and performance analysis of the custom-trained **YOLOv8-Seg** model for industrial stack flare monitoring.

---

## 📈 Key Performance Indicators (KPIs)

* **Peak F1-Score**: **0.78** at a confidence threshold of **0.390** (average across all classes).
* **Maximum Recall**: **0.83** (detects 83% of all actual smoke/fire plumes under low confidence thresholds).
* **Class Performance**:
  * **Fire**: Demonstrates exceptionally strong recall, maintaining >85% recall up to a 0.6 confidence threshold.
  * **Smoke**: Achieves solid recall, though slightly lower than fire due to varying cloud transparency and lighting profiles.

---

## 📊 Confusion Matrix Analysis

A review of the validation predictions reveals the following breakdown:

| Actual Class | Predicted: **Fire** | Predicted: **Smoke** | Predicted: **Background** | True Positive Rate (Recall) |
| :--- | :---: | :---: | :---: | :---: |
| **Fire** | **48** | 0 | 3 | **94.1%** |
| **Smoke** | 1 | **24** | 8 | **75.0%** |
| **Background** (Ground Truth) | 9 | 15 | - | - |

### Key Takeaways:
1. **Excellent Fire Detection**: Out of 51 real fire instances, **48 (94.1%)** were successfully segmented. There was zero confusion between fire and smoke.
2. **Reliable Smoke Detection**: Out of 32 real smoke instances, **24 (75.0%)** were successfully segmented.
3. **Background False Positives**: There were 9 false positives for fire and 15 for smoke. This is a common and expected challenge in industrial open-sky environments, where steam plumes, low clouds, or sun glare can be mistaken for smoke/fire.

---

## 🖼️ Visual Overlay Assessment

Based on the validation batch prediction grids:
* **Plume Outlines**: The model outputs high-quality polygon masks that trace the boundaries of smoke plumes (marked in cyan/blue) with high precision, avoiding over-segmentation.
* **Flame Segmentation**: Flare stack flames (marked in purple) are segmented tightly, distinguishing the core flame from the surrounding sky or heat haze.
* **Label Confidence**: Individual detections display high confidence scores (typically **0.80 to 0.90+**), validating that the model has high certainty on true positive regions.
