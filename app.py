import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io
import base64
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="🔍 CounterFIt - Currency Analysis",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .filter-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #e0e0e0;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .filter-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border-color: #667eea;
    }
    
    .upload-area {
        border: 3px dashed #667eea;
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        background: linear-gradient(135deg, #f8f9ff 0%, #e8ecff 100%);
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, #f0f4ff 0%, #e0e8ff 100%);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9ff 0%, #e8ecff 100%);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🔍 CounterFIt</h1>
    <h3>Advanced Currency Note Analysis & Counterfeit Detection</h3>
    <p>Powered by Computer Vision & Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎛️ Analysis Controls")
    
    # Filter selection
    st.markdown("### 📊 Select Analysis Method")
    filter_options = [
        "Original", "Grayscale", "Canny Edge Detection", "Sobel Filter",
        "HSV Colorspace", "Gaussian Blur", "Laplacian Filter",
        "Thresholding (Binary Image)", "Texture Analysis",
        "Segmentation (K-means)", "PCA (Dimensionality Reduction)",
        "Corner Detection (Harris)", "Feature Detection (SIFT)"
    ]
    
    selected_filter = st.selectbox(
        "Choose your analysis method:",
        filter_options,
        index=0
    )
    
    st.markdown("---")
    
    # Analysis parameters
    st.markdown("### ⚙️ Parameters")
    
    if "Canny Edge Detection" in selected_filter:
        canny_low = st.slider("Canny Low Threshold", 50, 200, 100)
        canny_high = st.slider("Canny High Threshold", 150, 300, 200)
    
    if "Gaussian Blur" in selected_filter:
        blur_kernel = st.slider("Blur Kernel Size", 3, 15, 5, step=2)
    
    if "Thresholding" in selected_filter:
        threshold_value = st.slider("Threshold Value", 0, 255, 127)
    
    if "Segmentation" in selected_filter:
        k_clusters = st.slider("Number of Clusters", 2, 8, 4)
    
    st.markdown("---")
    
    # Information panel
    st.markdown("### ℹ️ About")
    st.info("""
    **CounterFIt** uses advanced computer vision techniques to analyze currency notes and detect potential counterfeits.
    
    Each filter reveals different aspects of the note that can help identify authenticity.
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 📸 Upload Currency Images")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose one or more currency images...",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload high-quality images for best analysis results"
    )

with col2:
    st.markdown("## 📊 Quick Stats")
    
    if uploaded_files:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(uploaded_files)}</h3>
            <p>Images Uploaded</p>
        </div>
        """, unsafe_allow_html=True)
        
        # File info
        for i, file in enumerate(uploaded_files):
            st.info(f"**{file.name}** ({file.size // 1024} KB)")
    else:
        st.markdown("""
        <div class="metric-card">
            <h3>0</h3>
            <p>Images Uploaded</p>
        </div>
        """, unsafe_allow_html=True)

# Analysis functions
def apply_filter(img, filter_name, **params):
    """Apply selected filter to image with parameters"""
    if filter_name == "Original":
        return img, "Check overall look of the note: portrait, borders, hologram."
    
    # Convert to grayscale for processing
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    if filter_name == "Grayscale":
        return gray, "Grayscale highlights contrasts. Fake notes may have dull/uneven printing."
    
    elif filter_name == "Canny Edge Detection":
        low = params.get('canny_low', 100)
        high = params.get('canny_high', 200)
        edges = cv2.Canny(gray, low, high)
        return edges, f"Edges reveal micro-text, borders. Fakes often have broken/blurred edges. (Low: {low}, High: {high})"
    
    elif filter_name == "Sobel Filter":
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        mag = cv2.magnitude(sobelx, sobely)
        return mag, "Sobel highlights directional edges. Compare fine patterns and portrait clarity."
    
    elif filter_name == "HSV Colorspace":
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        return hsv, "HSV shows ink variations. Fakes may miss fluorescent tones."
    
    elif filter_name == "Gaussian Blur":
        kernel_size = params.get('blur_kernel', 5)
        blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        return blur, f"Blur can expose inconsistencies. Genuine notes keep structure even under blur. (Kernel: {kernel_size}x{kernel_size})"
    
    elif filter_name == "Laplacian Filter":
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return lap, "Laplacian shows sharp intensity changes. Fake notes may lack fine depth."
    
    elif filter_name == "Thresholding (Binary Image)":
        thresh_val = params.get('threshold_value', 127)
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        return binary, f"Binary form highlights watermark/logo areas. Genuine notes show hidden symbols. (Threshold: {thresh_val})"
    
    elif filter_name == "Texture Analysis":
        g_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        g_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        texture = cv2.magnitude(g_x, g_y)
        return texture, "Texture patterns (like paper fibers, security threads) differ in fakes."
    
    elif filter_name == "Segmentation (K-means)":
        k = params.get('k_clusters', 4)
        Z = img.reshape((-1,3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        segmented = center[label.flatten()].reshape(img.shape)
        return segmented, f"Segmentation separates portrait, serial, watermark. Compare structure. (Clusters: {k})"
    
    elif filter_name == "PCA (Dimensionality Reduction)":
        flat = np.float32(gray.reshape(-1,1))
        mean, eigenvectors = cv2.PCACompute(flat, mean=None, maxComponents=1)
        reduced = np.dot(flat-mean, eigenvectors.T).reshape(gray.shape)
        return reduced, "PCA reduces redundancy. Genuine notes retain main patterns; fakes distort."
    
    elif filter_name == "Corner Detection (Harris)":
        dst = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        img_corners = img.copy()
        img_corners[dst>0.01*dst.max()] = [255,0,0]
        return img_corners, "Corners highlight alignment marks, serial edges. Fakes misalign often."
    
    elif filter_name == "Feature Detection (SIFT)":
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        img_sift = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return img_sift, f"SIFT shows keypoints. Genuine holograms/portraits have dense keypoints. (Features: {len(kp)})"
    
    return img, "No filter applied."

# Display results
if uploaded_files:
    st.markdown("## 🔍 Analysis Results")
    
    # Prepare parameters
    params = {}
    if "Canny Edge Detection" in selected_filter:
        params['canny_low'] = canny_low
        params['canny_high'] = canny_high
    if "Gaussian Blur" in selected_filter:
        params['blur_kernel'] = blur_kernel
    if "Thresholding" in selected_filter:
        params['threshold_value'] = threshold_value
    if "Segmentation" in selected_filter:
        params['k_clusters'] = k_clusters
    
    # Process and display images
    cols = st.columns(min(3, len(uploaded_files)))
    
    for i, uploaded_file in enumerate(uploaded_files):
        col_idx = i % len(cols)
        
        with cols[col_idx]:
            # Load image
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            # Apply filter
            filtered_img, description = apply_filter(img_array, selected_filter, **params)
            
            # Display
            st.markdown(f"""
            <div class="filter-card">
                <h4>📄 {uploaded_file.name}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            if len(filtered_img.shape) == 2:
                st.image(filtered_img, caption=f"{selected_filter}", use_column_width=True)
            else:
                st.image(filtered_img, caption=f"{selected_filter}", use_column_width=True)
            
            st.info(description)
            
            # Download button
            if len(filtered_img.shape) == 2:
                # Convert grayscale to RGB for PIL
                pil_img = Image.fromarray(filtered_img.astype(np.uint8))
            else:
                pil_img = Image.fromarray(filtered_img.astype(np.uint8))
            
            buf = io.BytesIO()
            pil_img.save(buf, format='PNG')
            buf.seek(0)
            
            st.download_button(
                label=f"💾 Download {selected_filter}",
                data=buf.getvalue(),
                file_name=f"{uploaded_file.name.split('.')[0]}_{selected_filter.replace(' ', '_')}.png",
                mime="image/png"
            )

else:
    # Welcome message when no images uploaded
    st.markdown("""
    <div class="upload-area">
        <h2>🚀 Ready to Analyze Currency?</h2>
        <p>Upload one or more currency images above to begin your analysis!</p>
        <p>Our advanced computer vision algorithms will help you detect potential counterfeits.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature showcase
    st.markdown("## ✨ Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="filter-card">
            <h3>🔍 Edge Detection</h3>
            <p>Advanced algorithms to detect fine details and borders</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="filter-card">
            <h3>🎨 Color Analysis</h3>
            <p>HSV and texture analysis for ink consistency</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="filter-card">
            <h3>🧠 AI Segmentation</h3>
            <p>Machine learning-based image segmentation</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🔍 CounterFIt - Advanced Currency Analysis Platform</p>
    <p>Built with ❤️ using Streamlit, OpenCV, and Computer Vision</p>
</div>
""", unsafe_allow_html=True)
