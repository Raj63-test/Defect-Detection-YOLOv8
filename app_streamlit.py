import streamlit as st
from PIL import Image
import os
import glob
import time
from detector import load_model, detect_defects, process_video, get_model_info

# Page Configuration
st.set_page_config(
    page_title="Industrial Defect Detection & Tracking Pipeline",
    page_icon="🔍",
    layout="wide",
)

# Custom Premium Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-title {
        font-size: 1.2rem;
        color: #4b5563;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        padding: 1.2rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
        border-top: 4px solid #3b82f6;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3a8a;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 5px solid #2563eb;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown('<div class="main-title">🔍 Industrial Defect Analysis & Tracking Pipeline</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">A production-ready computer vision pipeline supporting real-time detection, segmentation, and ByteTrack object tracking on images and videos.</div>',
    unsafe_allow_html=True,
)

# Dynamic Model Discovery
weights_options = {}

# Priority defaults
if os.path.exists("best.pt"):
    weights_options["[Detect] MVTec Defect Detection (best.pt)"] = "best.pt"
if os.path.exists("smoke_best.pt"):
    weights_options["[Segment] Smoke Segmentation (smoke_best.pt)"] = "smoke_best.pt"
if os.path.exists("../defect_segmentation_yolov8/best.pt"):
    weights_options["[Segment] Fire & Smoke Segmentation (defect_segmentation_yolov8/best.pt)"] = "../defect_segmentation_yolov8/best.pt"

# Scan workspace for additional model files
for p in glob.glob("*.pt") + glob.glob("../*/*.pt") + glob.glob("../Smart-Monitoring/**/*.pt", recursive=True):
    abs_path = os.path.abspath(p)
    base_name = os.path.basename(p)
    # Label properly
    task = "detect"
    if "seg" in base_name.lower() or "segment" in abs_path.lower() or "semantic" in abs_path.lower():
        task = "segment"
    elif "classify" in abs_path.lower():
        task = "classify"
        
    label = f"[{task.upper()}] {base_name} ({os.path.dirname(p)})"
    if label not in weights_options and p not in weights_options.values():
        weights_options[label] = p

# Standard model options to always allow quick testing
weights_options["[DETECT] YOLOv8n Pretrained (COCO)"] = "yolov8n.pt"
weights_options["[SEGMENT] YOLOv8n-Seg Pretrained (COCO)"] = "yolov8n-seg.pt"

# Sidebar Configuration
st.sidebar.title("🛠️ Configuration")

selected_model_label = st.sidebar.selectbox(
    "Select Model Weights",
    options=list(weights_options.keys())
)
weights_path = weights_options[selected_model_label]

# Load selected model
with st.sidebar.spinner("Loading YOLO model..."):
    try:
        model = load_model(weights_path)
        model_info = get_model_info(model)
        st.sidebar.success(f"Loaded: `{model.task.upper()}` Model")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        st.stop()

# Display Model Info in Sidebar
with st.sidebar.expander("📊 Model Details", expanded=True):
    st.markdown(f"**Task:** `{model_info['task'].upper()}`")
    st.markdown(f"**Classes ({model_info['num_classes']}):**")
    st.code(", ".join(model_info['class_names']))

# Sidebar settings
st.sidebar.markdown("---")
st.sidebar.subheader("Hyperparameters")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Minimum confidence score required to draw a box/mask.",
)

iou_threshold = st.sidebar.slider(
    "IoU Threshold (NMS)",
    min_value=0.1,
    max_value=1.0,
    value=0.45,
    step=0.05,
    help="Overlap threshold for Non-Maximum Suppression.",
)

# Input Mode Selection
input_mode = st.radio(
    "Select Input Media Type",
    options=["🖼️ Image Inspection", "🎥 Video Tracking (ByteTrack)"],
    horizontal=True
)

st.markdown("---")

if input_mode == "🖼️ Image Inspection":
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.subheader("1. Input Image Selection")
        
        # Example selector
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
                
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["png", "jpg", "jpeg"],
            help="Upload an image to inspect for anomalies."
        )
        
        # Load the selected image
        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
        elif selected_example is not None:
            image = Image.open(selected_example).convert("RGB")
            st.info(f"Loaded sample image: {os.path.basename(selected_example)}")
            
        if image is not None:
            st.image(image, caption="Original Input Image", use_container_width=True)
        else:
            st.info("Please upload an image or choose a sample from the menu above.")
            
    with col2:
        st.subheader("2. Inspection & Annotation Results")
        
        if image is not None:
            # Inference execution
            with st.spinner("Processing image through vision pipeline..."):
                t0 = time.time()
                annotated_image, detections, summary_text = detect_defects(
                    image=image,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    model=model
                )
                latency = (time.time() - t0) * 1000
                
            # Metric visualization cards
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Latency (Inference)</div>
                        <div class="metric-value">{latency:.1f} ms</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with m_col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Detections Count</div>
                        <div class="metric-value">{len(detections)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            # Display annotated image
            st.image(annotated_image, caption=f"Annotated Output ({model.task.upper()})", use_container_width=True)
            
            # Display summary
            st.subheader("Inspection Summary")
            if not detections:
                st.success(summary_text)
            else:
                st.warning(summary_text)
                
            with st.expander("🔍 Show detailed detection data"):
                st.write(detections)
        else:
            st.write("Awaiting image input...")

else:
    # Video Tracking Mode
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.subheader("1. Video Source Selection")
        
        # Look for existing test videos
        examples_video_dir = "test_videos"
        example_videos = []
        if os.path.exists(examples_video_dir):
            example_videos = sorted([
                f for f in os.listdir(examples_video_dir)
                if f.lower().endswith((".mp4", ".avi", ".mov"))
            ])
            
        selected_video_example = None
        if example_videos:
            video_choice = st.selectbox(
                "Select a sample video:",
                options=["-- Upload your own --"] + example_videos
            )
            if video_choice != "-- Upload your own --":
                selected_video_example = os.path.join(examples_video_dir, video_choice)
                
        # File uploader
        uploaded_video = st.file_uploader(
            "Upload Video (MP4, AVI, MOV)",
            type=["mp4", "avi", "mov"],
            help="Upload a video to run tracking inference."
        )
        
        video_path = None
        if uploaded_video is not None:
            # Save uploaded video to a temporary local file
            temp_in_path = "temp_uploaded_video.mp4"
            with open(temp_in_path, "wb") as f:
                f.write(uploaded_video.read())
            video_path = temp_in_path
            st.video(video_path)
        elif selected_video_example is not None:
            video_path = selected_video_example
            st.video(video_path)
            st.info(f"Loaded sample video: {os.path.basename(video_path)}")
        else:
            st.info("Please upload a video or select a sample to begin.")
            
    with col2:
        st.subheader("2. Tracking & Pipeline Analysis")
        
        if video_path is not None:
            # Output path configuration
            output_video_path = "processed_tracking_output.mp4"
            
            # Action button
            if st.button("🚀 Start Video Tracking & Pipeline Analysis", use_container_width=True):
                # Progress and placeholders
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                
                def update_progress(current, total):
                    pct = float(current) / float(total) if total > 0 else 0.0
                    progress_bar.progress(pct)
                    status_text.text(f"Analyzing and tracking: Frame {current} of {total}...")
                
                with st.spinner("Processing video frame-by-frame..."):
                    metrics = process_video(
                        model=model,
                        input_path=video_path,
                        output_path=output_video_path,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                        progress_callback=update_progress
                    )
                
                status_text.success("Processing & H.264 video compression complete!")
                progress_bar.progress(1.0)
                
                # Show pipeline metrics
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Pipeline FPS</div>
                            <div class="metric-value">{metrics['avg_fps']} FPS</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with m_col2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Inference Latency</div>
                            <div class="metric-value">{metrics['avg_inference_latency_ms']} ms</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with m_col3:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Unique Objects Tracked</div>
                            <div class="metric-value">{metrics['unique_objects_tracked']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Render the final tracking output video
                st.video(output_video_path)
                
                # Download Button
                with open(output_video_path, "rb") as file:
                    st.download_button(
                        label="📥 Download Processed Tracking Video",
                        data=file,
                        file_name="processed_defect_tracking.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
                
                # Clean up uploaded temp file if used
                if video_path == "temp_uploaded_video.mp4" and os.path.exists("temp_uploaded_video.mp4"):
                    try:
                        os.remove("temp_uploaded_video.mp4")
                    except OSError:
                        pass
        else:
            st.write("Awaiting video source selection...")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #9ca3af; font-size: 0.85rem;'>"
    "Production-Grade Defect Analysis System | Powered by Ultralytics YOLOv8, ByteTrack, OpenCV, & Streamlit"
    "</div>",
    unsafe_allow_html=True
)
