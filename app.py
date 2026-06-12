import gradio as gr 
from ultralytics import YOLO
from PIL import Image
import numpy as np

model = YOLO("best.pt")

def detect_defects(image):
    results = model(np.array(image))[0]
    annotated = Image.fromarray(results.plot())


    detections=[]
    for box in results.boxes:
        label = model.names[int(box.cls)]
        conf = round(float(box.conf),2)
        detections.append(f"{label}:{conf}")
    

    summary = "\n".join(detections) if detections else "NO defects Detected"

    return annotated, summary


demo = gr.Interface(
    fn= detect_defects,
    inputs = gr.Image(type="pil", label="Upload Product Image"),
    outputs = [
        gr.Image(label="Detection result"),
        gr.Textbox(label = "Detections")
    ],
    title= "Real-Time Defect Detection",
    description="Upload an image to detect surface defects using YOLOv8",
    examples=["test_images/sample1.png", "test_images/sample2.png", "test_images/sample3.png","test_images/sample4.png","test_images/sample5.png"]
)

# launch the gradio app.
demo.launch()