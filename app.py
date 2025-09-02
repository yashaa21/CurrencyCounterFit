import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# ───────────────────────── Page configuration (must be first Streamlit call) ─────────────────────────
st.set_page_config(
    page_title="🔍 CounterFIt - Currency Analysis",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ───────────────────────── Helpers ─────────────────────────
def to_uint8(img):
    """Safely convert any numeric image to uint8 [0,255]."""
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr
    # normalize per-image
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx == mn:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr - mn) * (255.0 / (mx - mn))
    return np.clip(scaled, 0, 255).astype(np.uint8)

def ensure_rgb(pil_img):
    """Ensure 3-channel RGB numpy array (handle L/LA/RGBA)."""
    if pil_img.mode not in ("RGB", "L"):
        pil_img = pil_img.convert("RGBA")
        bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
        pil_img = Image.alpha_composite(bg, pil_img).convert("RGB")
    elif pil_img.mode == "L":
        pil_img = pil_img.convert("RGB")
    return np.array(pil_img)

def draw_harris_on_image(base_rgb, gray):
    """Draw Harris corners (red) safely on a color image."""
    gray32 = np.float32(gray)
    dst = cv2.cornerHarris(gray32, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    overlay = base_rgb.copy()
    mask = dst > 0.01 * dst.max()
    overlay[mask] = [255, 0, 0]
    return overlay

def pca_first_component(gray):
    """Safe PCA 1-component visualization using OpenCV if available, else numpy."""
    flat = gray.reshape(-1, 1).astype(np.float32)
    try:
        mean, eig = cv2.PCACompute(flat, mean=None, maxComponents=1)
        proj = np.dot(flat - mean, eig.T).reshape(gray.shape)
    except Exception:
        # Numpy fallback
        x = flat - flat.mean(axis=0)
        U, S, Vt = np.linalg.svd(x, full_matrices=False)
        proj = (U[:, 0] * S[0]).reshape(gray.shape)
    return to_uint8(proj)

def kmeans_segmentation(rgb, k):
    Z = rgb.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    attempts = 10
    _, label, center = cv2.kmeans(Z, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    segmented = center[label.flatten()].reshape(rgb.shape)
    return segmented

# ───────────────────────── Custom CSS ─────────────────────────
st.markdown("""
<style>
    /* Global body text color for dark theme components not explicitly styled */
    body {
        color: white; /* Default text color for the overall app background */
    }

    /* Main header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white; /* Header text is white */
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }

    /* Metric card for quick stats */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white; /* Metric card text is white */
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    /* Filter card for key features and image details */
    .filter-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #e0e0e0;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        min-height: 150px; /* Ensures consistent height for feature boxes */
    }

    /* Specific text color for elements within filter card */
    .filter-card h3, .filter-card h4, .filter-card p {
        color: #333333 !important; /* Darker text for readability on white background */
    }

    /* Hover effect for filter card */
    .filter-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border-color: #667eea;
    }

    /* File upload area styling */
    .upload-area {
        border: 3px dashed #667eea;
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        background: linear-gradient(135deg, #f8f9ff 0%, #e8ecff 100%);
        margin: 2rem 0;
        transition: all 0.3s ease;
    }

    /* Specific text color for elements within upload area */
    .upload-area h2, .upload-area p {
        color: #333333 !important; /* Darker text for readability on light background */
    }

    /* Hover effect for upload area */
    .upload-area:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, #f0f4ff 0%, #e0e8ff 100%);
    }

    /* Streamlit button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    /* Hover effect for buttons */
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }

    /* Fix for Streamlit input widgets (text input, number input, selectbox) */
    .stTextInput > div > div > input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] span {
        color: #000000 !important;      /* Black text */
        background-color: #ffffff !important; /* White background */
        border-radius: 8px !important;
    }

    /* Fix for Streamlit text area */
    .stTextArea textarea {
        color: #000000 !important;
        background-color: #ffffff !important;
        border-radius: 8px !important;
    }

    /* Fix for Streamlit slider handle text */
    .stSlider > div > div > div {
        color: #000000 !important;
    }

    /* Footer styling */
    div[data-testid="stVerticalBlock"] > div > div > div > div:last-child p {
        color: #666 !important; /* Specific color for footer text */
    }
</style>
""", unsafe_allow_html=True)

# ───────────────────────── Header ─────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔍 CounterFIt</h1>
    <h3>Advanced Currency Note Analysis & Counterfeit Detection</h3>
    <p>Powered by Computer Vision & Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ───────────────────────── Sidebar ─────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Analysis Controls")
    st.markdown("### 📊 Select Analysis Method")
    filter_options = [
        "Original", "Grayscale", "Canny Edge Detection", "Sobel Filter",
        "HSV Colorspace", "Gaussian Blur", "Laplacian Filter",
        "Thresholding (Binary Image)", "Texture Analysis",
        "Segmentation (K-means)", "PCA (Dimensionality Reduction)",
        "Corner Detection (Harris)", "Feature Detection (SIFT/ORB)"
    ]
    selected_filter = st.selectbox("Choose your analysis method:", filter_options, index=0)

    st.markdown("---")
    st.markdown("### ⚙️ Parameters")

    if selected_filter == "Canny Edge Detection":
        canny_low = st.slider("Canny Low Threshold", 0, 255, 100)
        canny_high = st.slider("Canny High Threshold", 1, 255, 200)

    if selected_filter == "Gaussian Blur":
        blur_kernel = st.slider("Blur Kernel Size (odd)", 3, 31, 5, step=2)

    if selected_filter == "Thresholding (Binary Image)":
        threshold_value = st.slider("Threshold Value", 0, 255, 127)

    if selected_filter == "Segmentation (K-means)":
        k_clusters = st.slider("Number of Clusters", 2, 8, 4)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("""
    **CounterFIt** uses computer vision to analyze currency notes for possible counterfeits.
    Each method reveals different aspects to inspect.
    """)

# ───────────────────────── Main content ─────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 📸 Upload Currency Images")
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
        for f in uploaded_files:
            try:
                size_kb = (f.size or 0) // 1024
            except Exception:
                size_kb = 0
            st.info(f"**{f.name}** ({size_kb} KB)")
    else:
        st.markdown("""
        <div class="metric-card">
            <h3>0</h3>
            <p>Images Uploaded</p>
        </div>
        """, unsafe_allow_html=True)

# ───────────────────────── Analysis function ─────────────────────────
def apply_filter(img_rgb, filter_name, **params):
    """
    img_rgb: uint8 RGB image
    returns: (display_img_uint8, description)
    """
    img_rgb = to_uint8(img_rgb)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    if filter_name == "Original":
        return img_rgb, "Check overall look of the note: portrait, borders, hologram."

    if filter_name == "Grayscale":
        return gray, "Grayscale highlights contrasts. Fake notes may have dull/uneven printing."

    if filter_name == "Canny Edge Detection":
        low = int(params.get('canny_low', 100))
        high = int(params.get('canny_high', 200))
        # Ensure valid ordering
        low, high = min(low, high), max(low, high)
        edges = cv2.Canny(gray, low, high)
        return edges, f"Edges reveal micro-text, borders. Fakes often have broken/blurred edges. (Low: {low}, High: {high})"

    if filter_name == "Sobel Filter":
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = cv2.magnitude(sobelx, sobely)
        return to_uint8(mag), "Sobel highlights directional edges. Compare fine patterns and portrait clarity."

    if filter_name == "HSV Colorspace":
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        # Visualize as false-color by feeding HSV directly (distinct look), but ensure uint8
        return to_uint8(hsv), "HSV emphasizes hue/value variations that may differ in counterfeits."

    if filter_name == "Gaussian Blur":
        k = int(params.get('blur_kernel', 5))
        # kernel must be odd and >=1
        if k % 2 == 0:
            k += 1
        k = max(k, 1)
        blur = cv2.GaussianBlur(img_rgb, (k, k), 0)
        return blur, f"Blur can expose inconsistencies. Genuine notes keep structure even under blur. (Kernel: {k}×{k})"

    if filter_name == "Laplacian Filter":
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return to_uint8(lap), "Laplacian shows sharp intensity changes. Fake notes may lack fine depth."

    if filter_name == "Thresholding (Binary Image)":
        t = int(params.get('threshold_value', 127))
        _, binary = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
        return binary, f"Binary form highlights watermark/logo areas. Genuine notes show hidden symbols. (Threshold: {t})"

    if filter_name == "Texture Analysis":
        gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        texture = cv2.magnitude(gx, gy)
        return to_uint8(texture), "Texture patterns (paper fibers, security threads) often differ in fakes."

    if filter_name == "Segmentation (K-means)":
        k = int(params.get('k_clusters', 4))
        segmented = kmeans_segmentation(img_rgb, k)
        return segmented, f"Segmentation separates portrait, serial, watermark regions. (Clusters: {k})"

    if filter_name == "PCA (Dimensionality Reduction)":
        comp = pca_first_component(gray)
        return comp, "PCA emphasizes major variance patterns; genuine notes retain strong principal structures."

    if filter_name == "Corner Detection (Harris)":
        corners = draw_harris_on_image(img_rgb, gray)
        return corners, "Corners highlight alignment marks/serial edges. Fakes often show misalignment."

    if filter_name == "Feature Detection (SIFT/ORB)":
        # Prefer SIFT if available (opencv-contrib), else fallback to ORB
        img_disp = img_rgb.copy()
        try:
            sift = cv2.SIFT_create()
            kp, _ = sift.detectAndCompute(gray, None)
            img_disp = cv2.drawKeypoints(img_rgb, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            return img_disp, f"SIFT keypoints overlaid. Dense, stable features appear on genuine security elements. (Features: {len(kp)})"
        except Exception:
            orb = cv2.ORB_create(nfeatures=1000)
            kp, _ = orb.detectAndCompute(gray, None)
            img_disp = cv2.drawKeypoints(img_rgb, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            return img_disp, f"ORB keypoints (SIFT unavailable). Still useful for spotting consistent security features. (Features: {len(kp)})"

    return img_rgb, "No filter applied."

# ───────────────────────── Display results ─────────────────────────
if uploaded_files:
    st.markdown("## 🔍 Analysis Results")

    # collect params
    params = {}
    if selected_filter == "Canny Edge Detection":
        params['canny_low'] = canny_low
    if selected_filter == "Canny Edge Detection":
        params['canny_high'] = canny_high
    if selected_filter == "Gaussian Blur":
        params['blur_kernel'] = blur_kernel
    if selected_filter == "Thresholding (Binary Image)":
        params['threshold_value'] = threshold_value
    if selected_filter == "Segmentation (K-means)":
        params['k_clusters'] = k_clusters

    cols = st.columns(min(3, len(uploaded_files)))
    for i, uf in enumerate(uploaded_files):
        col = cols[i % len(cols)]
        with col:
            pil = Image.open(uf)
            img_rgb = ensure_rgb(pil)  # robust to L/LA/RGBA
            out_img, description = apply_filter(img_rgb, selected_filter, **params)

            st.markdown(f"""
            <div class="filter-card">
                <h4>📄 {uf.name}</h4>
            </div>
            """, unsafe_allow_html=True)

            # Streamlit will handle 2D (grayscale) and 3D images
            st.image(to_uint8(out_img), caption=f"{selected_filter}", use_column_width=True)

            st.info(description)

            # Download button
            out_uint8 = to_uint8(out_img)
            # If grayscale, save as 'L'; else 'RGB'
            if out_uint8.ndim == 2:
                pil_out = Image.fromarray(out_uint8, mode="L")
            else:
                pil_out = Image.fromarray(out_uint8, mode="RGB")
            buf = io.BytesIO()
            pil_out.save(buf, format='PNG')
            buf.seek(0)
            st.download_button(
                label=f"💾 Download {selected_filter}",
                data=buf.getvalue(),
                file_name=f"{uf.name.rsplit('.', 1)[0]}_{selected_filter.replace(' ', '_').replace('/', '-')}.png",
                mime="image/png"
            )
else:
    st.markdown("""
    <div class="upload-area">
        <h2>🚀 Ready to Analyze Currency?</h2>
        <p>Upload one or more currency images above to begin your analysis!</p>
        <p>Our advanced computer vision algorithms will help you detect potential counterfeits.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## ✨ Key Features")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="filter-card">
            <h3>🔍 Edge Detection</h3>
            <p>Advanced algorithms to detect fine details and borders</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="filter-card">
            <h3>🎨 Color Analysis</h3>
            <p>HSV and texture analysis for ink consistency</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="filter-card">
            <h3>🧠 AI Segmentation</h3>
            <p>Machine learning-based image segmentation</p>
        </div>
        """, unsafe_allow_html=True)

# ───────────────────────── Footer ─────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🔍 CounterFIt - Advanced Currency Analysis Platform</p>
    <p>Built with ❤️ using Streamlit & OpenCV</p>
</div>
""", unsafe_allow_html=True)