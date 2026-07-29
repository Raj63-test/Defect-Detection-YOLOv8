import streamlit as st
from PIL import Image
import os
from detector import detect_defects

# Page Configuration
st.set_page_config(
    page_title="Real-Time Surface Defect Detection",
    page_icon="🔍",
    layout="wide",
)

# Custom Sleek Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #f3f4f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2563eb;
        margin-bottom: 1.5rem;
    }
    .result-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: monospace;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title & Description
st.markdown('<div class="main-title">🔍 Real-Time Surface Defect Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">An industrial quality control system powered by custom-trained YOLOv8. Upload a product photo to automatically locate and classify defects.</div>',
    unsafe_allow_html=True,
)

# Sidebar Configuration
st.sidebar.image("test_images/sample1.png", caption="Defect Detection YOLOv8", use_column_width=True)
st.sidebar.title("Configuration")

# Threshold Sliders
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Minimum confidence score required to display a detection box.",
)

iou_threshold = st.sidebar.slider(
    "IoU Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.45,
    step=0.05,
    help="Overlap threshold for Non-Maximum Suppression (NMS) to filter double detections.",
)

# Main UI Structure: Left (Inputs) | Right (Outputs)
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("1. Input Image")
    
    # Example Selector
    examples_dir = "test_images"
    example_images = []
    if os.path.exists(examples_dir):
        example_images = sorted([
            f for f in os.listdir(examples_dir) 
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
    
    selected_example = None
    if example_images:
        example_choice = st.selectbox(
            "Or select an example image:",
            options=["-- Upload your own --"] + example_images
        )
        if example_choice != "-- Upload your own --":
            selected_example = os.path.join(examples_dir, example_choice)

    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload Product Image (Bottle or Metal Nut)",
        type=["png", "jpg", "jpeg"],
        help="Upload an image to inspect it for defects."
    )

    # Load Image
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
    elif selected_example is not None:
        image = Image.open(selected_example).convert("RGB")
        st.info(f"Loaded example: {os.path.basename(selected_example)}")
    
    if image is not None:
        st.image(image, caption="Original Image", use_column_width=True)
    else:
        st.info("Please upload an image or select a sample image from the list above.")

with col2:
    st.subheader("2. Inspection Results")
    
    if image is not None:
        # Run detection
        with st.spinner("Inspecting product with YOLOv8..."):
            annotated_image, detections, summary_text = detect_defects(
                image=image,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
        
        # Display annotated image
        st.image(annotated_image, caption="Annotated Detections", use_column_width=True)
        
        # Display Summary
        st.subheader("Defect Summary")
        if not detections:
            st.success(summary_text)
        else:
            st.warning(summary_text)
            
            # Detailed breakdown
            with st.expander("Show detailed detection data"):
                st.write(detections)
    else:
        st.write("Results will appear here after you load an input image.")

# Footer
st.markdown("---")
st.markdown(
    "Trained on **MVTec AD** dataset (Bottle and Metal Nut categories) | Powered by Ultralytics YOLOv8 & Streamlit"
)
