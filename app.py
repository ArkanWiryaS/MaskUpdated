import streamlit as st
import cv2
import numpy as np
import torch
import os
import tempfile
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import matplotlib.pyplot as plt
from ultralytics import YOLO

st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@200..800&display=swap');
    
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Manrope', sans-serif;
    }
    /* Apply Manrope to all basic elements */
    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif !important;
    }
    /* Apply to all text elements */
    p, div, h1, h2, h3, h4, h5, h6, label, span {
        font-family: 'Manrope', sans-serif !important;
    }
    /* Apply to streamlit elements */
    .stTextInput, .stSelectbox, .stMultiselect {
        font-family: 'Manrope', sans-serif !important;
    }
    /* Apply to dataframes */
    .dataframe {
        font-family: 'Manrope', sans-serif !important;
    }
    .main-header {
        font-family: 'Manrope', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-family: 'Manrope', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: white;
        margin-bottom: 1rem;
    }
    .info-box { 
        font-family: 'Manrope', sans-serif;
        background-color: #EFF6FF;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #3B82F6;
    }
    .card {
        font-family: 'Manrope', sans-serif;
        border-radius: 10px;
        padding: 20px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .stButton>button {
        font-family: 'Manrope', sans-serif;
        background-color: #3B82F6;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1E3A8A;
    }
    /* New Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1E3A8A;
        color: white;
        font-family: 'Manrope', sans-serif !important;
        border-radius : 0px 38px 38px 0px;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: white !important;
        font-family: 'Manrope', sans-serif !important;
    }
    section[data-testid="stSidebar"] .stRadio div:nth-child(2) {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 5px;
        padding: 10px;
        font-family: 'Manrope', sans-serif !important;
    }
    .sidebar-content {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        font-family: 'Manrope', sans-serif !important;
    }
    .sidebar-title {
        font-size: 1.8rem; 
        font-weight: 700;
        color: white;
        margin-bottom: 1.5rem;
        padding-bottom: 0.7rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        font-family: 'Manrope', sans-serif !important;
    }
    .sidebar-header {
        color: #60A5FA;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        font-family: 'Manrope', sans-serif !important;
    }
    .sidebar-subheader {
        color: white;
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        font-family: 'Manrope', sans-serif !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the YOLOv8 model
@st.cache_resource
def load_model():
    # Fix for PyTorch 2.6+ loading issues by explicitly setting weights_only=False
    from ultralytics.nn.tasks import torch_safe_load
    
    # Save original function for later restoration
    original_torch_load = torch.load
    
    # Override torch.load temporarily to use weights_only=False
    torch.load = lambda *args, **kwargs: original_torch_load(*args, **kwargs, weights_only=False)
    
    try:
        model = YOLO("best.pt")
        return model
    finally:
        # Restore original torch.load function
        torch.load = original_torch_load

# Function to process image
def process_image(image, model):
    # Convert image to RGB if it has an alpha channel (RGBA)
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    # Make sure image is RGB (3 channels)
    elif len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    results = model(image)
    result_img = results[0].plot()
    return result_img, results[0]

# Function to display detection results
def display_results(results):
    if len(results.boxes) > 0:
        st.markdown("### Detection Results")
        
        # Extract class names and confidence scores
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        class_names = [results.names[class_id] for class_id in class_ids]
        confidence_scores = results.boxes.conf.cpu().numpy()
        
        # Create a simple results table
        result_data = {"Class": class_names, "Confidence": confidence_scores}
        st.dataframe(result_data, use_container_width=True)
        
        # Count detections by class
        class_counts = {}
        for cls in class_names:
            if cls in class_counts:
                class_counts[cls] += 1
            else:
                class_counts[cls] = 1
        
        # Display counts
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Total detections: {len(class_names)}")
        with col2:
            for cls, count in class_counts.items():
                if "mask" in cls.lower():
                    st.success(f"{cls}: {count}")
                else:
                    st.error(f"{cls}: {count}")

# Webcam processing class
class MaskDetector(VideoProcessorBase):
    def __init__(self, model):
        self.model = model
        self.error_count = 0
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Always get BGR format
        
        try:
            # Process the frame
            processed_img, _ = process_image(img, self.model)
            self.error_count = 0  # Reset error count on success
            return frame.from_ndarray(processed_img)
        except Exception as e:
            self.error_count += 1
            # If we've had too many errors, just pass through the original frame
            if self.error_count > 5:
                # Draw error text on the image
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, "Detection Error", (50, 50), font, 1, (0, 0, 255), 2)
            return frame.from_ndarray(img)

# Main function
def main():
    apply_custom_css()
    
    # Title and introduction
    st.markdown('<h1 class="main-header">Face Mask Detection</h1>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        <p style="font-family: 'Manrope', sans-serif; font-weight: 500;">
        This application uses a trained YOLOv8 model to detect whether people are wearing face masks correctly.
        You can either upload an image or use your webcam for real-time detection.
        </p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Load the model
    with st.spinner("Loading model..."):
        model = load_model()
    
    # Sidebar
    with st.sidebar:
        st.markdown('<p class="sidebar-title">Face Mask Detection</p>', unsafe_allow_html=True)
        
        # Detection Mode
        st.markdown('<p class="sidebar-header">Detection Mode</p>', unsafe_allow_html=True)
        detection_mode = st.radio(
            "Choose your detection method:",
            ["Upload Image", "Use Webcam"]
        )
    
    if detection_mode == "Upload Image":
        st.markdown('<p class="sub-header">Upload Image</p>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                # Convert the uploaded file to an image
                image = Image.open(uploaded_file).convert('RGB')  # Convert to RGB
                img_array = np.array(image)
                
                # Display the uploaded image
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Image")
                    st.image(img_array, channels="RGB", use_column_width=True)
                
                # Process the image
                with st.spinner("Processing image..."):
                    try:
                        result_image, results = process_image(img_array, model)
                        
                        with col2:
                            st.subheader("Processed Image")
                            st.image(result_image, channels="RGB", use_column_width=True)
                        
                        # Display detection results
                        display_results(results)
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
                        st.info("Try uploading a different image or check if the image format is supported.")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
    else:  # Webcam mode
        st.markdown('<p class="sub-header">Webcam Detection</p>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            st.write("Click 'Start' to begin real-time face mask detection using your webcam.")
            
            webrtc_ctx = webrtc_streamer(
                key="mask-detection",
                video_processor_factory=lambda: MaskDetector(model),
                rtc_configuration=RTCConfiguration(
                    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                ),
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            
            if webrtc_ctx.state.playing:
                st.write("Webcam is streaming. The model is detecting face masks in real-time.")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: gray;'>© 2023 Face Mask Detection App</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()