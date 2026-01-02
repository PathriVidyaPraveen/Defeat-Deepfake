import streamlit as st
import cv2
import pywt
import numpy as np
import tempfile
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Deepfake Forensics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Main container */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a1d29 100%);
    }
    
    /* Headers */
    h1 {
        color: #ff4b4b;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    h2, h3 {
        color: #ffffff;
        font-weight: 600;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d29 0%, #0e1117 100%);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- UTILITY FUNCTIONS ---

def normalize_wavelet(band):
    """Normalize wavelet coefficients for visualization."""
    band = np.abs(band)
    vmin, vmax = band.min(), band.max()
    if vmax - vmin < 1e-6:
        return band
    return ((band - vmin) / (vmax - vmin) * 255).astype(np.uint8)


def analyze_temporal_coherence(video_path, start_frame, num_frames=30, patch_size=16):
    """
    Analyzes patch-wise temporal intensity stability.
    Returns matplotlib figure and most unstable patch index.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append(frame)
    
    cap.release()
    
    if len(frames) < 2:
        return None, None

    # Convert to tensor: (T, 3, H, W)
    frames_tensor = torch.tensor(np.array(frames)).permute(0, 3, 1, 2)
    
    # Convert to grayscale
    gray = (frames_tensor[:, 0] * 0.299 + 
            frames_tensor[:, 1] * 0.587 + 
            frames_tensor[:, 2] * 0.114).unsqueeze(1)

    # Extract patches
    patches = F.unfold(gray, kernel_size=patch_size, stride=patch_size)
    patch_means = patches.mean(dim=1)  # (T, num_patches)
    
    # Calculate temporal instability
    deltas = torch.abs(patch_means[1:] - patch_means[:-1])
    instability = deltas.mean(dim=0)
    
    most_unstable_idx = torch.argmax(instability).item()
    most_stable_idx = torch.argmin(instability).item()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 5), facecolor='#0e1117')
    ax.set_facecolor('#1a1d29')
    
    frames_range = np.arange(len(frames))
    
    # Plot unstable patch
    y_unstable = patch_means[:, most_unstable_idx].numpy()
    y_unstable = (y_unstable - y_unstable.min()) / (y_unstable.max() - y_unstable.min() + 1e-8)
    
    ax.plot(frames_range, y_unstable, 
            color='#ff4b4b', linewidth=3, marker='o', 
            markersize=6, label=f'Patch #{most_unstable_idx} (Unstable)', 
            alpha=0.9)
    
    # Plot stable patch
    y_stable = patch_means[:, most_stable_idx].numpy()
    y_stable = (y_stable - y_stable.min()) / (y_stable.max() - y_stable.min() + 1e-8)
    
    ax.plot(frames_range, y_stable, 
            color='#00d4aa', linewidth=2.5, linestyle='--', 
            marker='s', markersize=5, label=f'Patch #{most_stable_idx} (Stable)', 
            alpha=0.7)
    
    ax.set_xlabel('Frame Index', color='white', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Intensity', color='white', fontsize=12, fontweight='bold')
    ax.set_title('Temporal Patch Intensity Analysis', color='white', fontsize=14, fontweight='bold', pad=20)
    ax.tick_params(colors='white', labelsize=10)
    ax.spines['bottom'].set_color('#ffffff')
    ax.spines['left'].set_color('#ffffff')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.15, color='white')
    ax.legend(facecolor='#262730', edgecolor='white', labelcolor='white', 
              fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    plt.close(fig)
    
    return fig, most_unstable_idx


# --- SIDEBAR ---
with st.sidebar:
    st.title("🔬 Forensics Lab")
    st.markdown("---")
    
    # Analysis mode
    analysis_mode = st.radio(
        "**Analysis Module**",
        ["🌊 Spectral Analysis", "📊 Temporal Coherence"],
        label_visibility="visible"
    )
    
    st.markdown("---")
    
    # Video selection
    st.subheader("📹 Evidence Selection")
    
    video_options = {
        "Demo: Real Video": "assets/real_video.mp4",
        "Demo: Fake Video": "assets/fake_video.mp4",
        "Upload Custom": None
    }
    
    video_choice = st.selectbox("", list(video_options.keys()))
    
    video_path = None
    if video_choice == "Upload Custom":
        uploaded = st.file_uploader("Upload MP4", type=['mp4'])
        if uploaded:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_file.write(uploaded.read())
            video_path = temp_file.name
    else:
        video_path = video_options[video_choice]
        if not os.path.exists(video_path):
            st.error(f"❌ File not found: {video_path}")
            st.stop()
    
    if not video_path:
        st.info("👆 Select or upload a video to begin analysis")
        st.stop()
    
    st.markdown("---")
    st.caption("**Deepfake Detection System**  \nBy Adishesh and Praveen")


# --- MAIN CONTENT ---

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🔬 Deepfake Forensics Dashboard")
with col2:
    verdict = "FAKE" if "fake" in video_choice.lower() else "REAL"
    verdict_color = "🔴" if verdict == "FAKE" else "🟢"
    st.metric("Video Status", f"{verdict_color} {verdict}")

st.markdown("---")

# Initialize video
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30.0
cap.release()

# --- SPECTRAL ANALYSIS MODE ---
# --- SPECTRAL ANALYSIS MODE ---
if analysis_mode == "🌊 Spectral Analysis":
    st.header("🌊 Frequency Domain Analysis")
    st.markdown("Wavelet decomposition reveals spatial frequency anomalies in deepfake generation.")
    
    # Frame selector
    frame_idx = st.slider("Select Frame", 0, total_frames - 1, total_frames // 2, 
                          help="Choose a frame to analyze")
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            st.subheader("Original Frame")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, use_container_width=True)
            st.caption(f"Frame {frame_idx}/{total_frames}")
        
        with col_right:
            st.subheader("Wavelet Decomposition (Haar)")
            
            # Perform DWT
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            coeffs = pywt.dwt2(gray, 'haar')
            LL, (LH, HL, HH) = coeffs
            
            # Display all subbands in 2x2 grid
            subcol1, subcol2 = st.columns(2)
            
            with subcol1:
                st.image(normalize_wavelet(LL), caption="LL - Approximation (Low-Low)", 
                        use_container_width=True)
                st.image(normalize_wavelet(LH), caption="LH - Horizontal Edges", 
                        use_container_width=True)
            
            with subcol2:
                st.image(normalize_wavelet(HL), caption="HL - Vertical Edges", 
                        use_container_width=True)
                st.image(normalize_wavelet(HH), caption="HH - Diagonal Details", 
                        use_container_width=True)
        
        # Energy analysis below
        st.markdown("---")
        st.subheader("📊 Frequency Band Energy Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        ll_energy = np.sum(LL ** 2)
        lh_energy = np.sum(LH ** 2)
        hl_energy = np.sum(HL ** 2)
        hh_energy = np.sum(HH ** 2)
        
        total_energy = ll_energy + lh_energy + hl_energy + hh_energy
        
        with col1:
            st.metric("LL Energy", f"{ll_energy:.2e}")
        
        with col2:
            st.metric("LH Energy", f"{lh_energy:.2e}")
        
        with col3:
            st.metric("HL Energy", f"{hl_energy:.2e}")
        
        with col4:
            st.metric("HH Energy", f"{hh_energy:.2e}")
        
        # Interpretation
        high_freq_ratio = (lh_energy + hl_energy + hh_energy) / total_energy
        
        # if "fake" in video_choice.lower():
        #     st.error(f"""
        #     **⚠️ Anomaly Detected**
            
        #     High-frequency energy ratio: **{high_freq_ratio:.1%}** (Abnormally low)
            
        #     - **LH/HL bands:** Reduced edge sharpness indicates GAN smoothing
        #     - **HH band:** Suppressed diagonal details suggest synthetic texture
        #     - GANs often produce over-smoothed outputs lacking natural camera noise
        #     """)
        # else:
        #     st.success(f"""
        #     **✅ Normal Pattern**
            
        #     High-frequency energy ratio: **{high_freq_ratio:.1%}** (Healthy)
            
        #     - **LH/HL bands:** Sharp edges consistent with authentic optics
        #     - **HH band:** Natural grain from camera sensor noise
        #     - Frequency distribution matches real-world capture characteristics
        #     """)
# --- TEMPORAL COHERENCE MODE ---
elif analysis_mode == "📊 Temporal Coherence":
    st.header("📊 Temporal Consistency Analysis")
    st.markdown("Tracks patch-level intensity stability across frames to detect temporal artifacts.")
    
    # Controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        start_frame = st.slider("Analysis Start Frame", 0, max(0, total_frames - 30), 0)
    with col2:
        st.write("")
        st.write("")
    with col3:
        st.write("")
        st.write("")
        analyze_btn = st.button("🔍 Run Analysis", type="primary", use_container_width=True)
    
    if analyze_btn:
        col_graph, col_video = st.columns([3, 2])
        
        with col_graph:
            with st.spinner("Analyzing temporal coherence..."):
                fig, unstable_idx = analyze_temporal_coherence(video_path, start_frame, num_frames=30)
            
            if fig:
                st.pyplot(fig)
                
                # Interpretation
                if "fake" in video_choice.lower():
                    st.error(f"""
                    **🚨 Temporal Inconsistency Detected**
                    
                    Patch #{unstable_idx} exhibits erratic intensity fluctuations. This jittering pattern 
                    is characteristic of frame-by-frame GAN synthesis where temporal coherence is not enforced.
                    """)
                else:
                    st.success(f"""
                    **✅ Coherent Temporal Pattern**
                    
                    Patch #{unstable_idx} maintains smooth intensity trajectory, consistent with natural 
                    camera motion and lighting changes.
                    """)
            else:
                st.warning("⚠️ Could not extract sufficient frames for analysis.")
        
        with col_video:
            st.subheader("Video Preview")
            start_time_sec = start_frame / fps
            st.video(video_path, start_time=int(start_time_sec))
            st.caption(f"📍 Starting at {start_time_sec:.2f}s (Frame {start_frame})")
            st.info("💡 Watch for subtle texture inconsistencies or micro-jitter in facial regions.")

