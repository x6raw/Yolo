import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import pandas as pd
import time

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="YOLO AI Vision",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #111827 40%,
        #1e293b 100%
    );
    color: white;
}

/* HERO */
.hero {
    padding: 40px;
    border-radius: 24px;
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 30px;
    text-align: center;
    box-shadow: 0px 10px 40px rgba(0,0,0,0.3);
}

.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    color: white;
    margin-bottom: 10px;
}

.hero-sub {
    font-size: 1.2rem;
    color: #cbd5e1;
}

/* CARDS */
.card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(16px);
    border-radius: 22px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0px 8px 30px rgba(0,0,0,0.25);
}

/* METRIC */
.metric-card {
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
}

.metric-number {
    font-size: 2.5rem;
    font-weight: bold;
    color: #38bdf8;
}

.metric-label {
    color: #cbd5e1;
    font-size: 1rem;
}

/* BUTTON */
.stButton > button {
    width: 100%;
    border-radius: 15px;
    height: 3rem;
    border: none;
    background: linear-gradient(90deg,#0ea5e9,#38bdf8);
    color: white;
    font-weight: bold;
    font-size: 16px;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0px 0px 20px rgba(56,189,248,0.5);
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04);
    padding: 20px;
    border-radius: 20px;
}

/* TABLE */
[data-testid="stDataFrame"] {
    border-radius: 20px;
    overflow: hidden;
}

.footer {
    text-align:center;
    color:#94a3b8;
    margin-top:40px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HERO SECTION
# ---------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-title">🤖 YOLO AI Vision</div>
    <div class="hero-sub">
        Moderne Object Detection mit YOLOv8 & Streamlit
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# MODEL LOAD
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:

    st.title("⚙️ Einstellungen")

    conf_threshold = st.slider(
        "Confidence Threshold",
        0.1,
        1.0,
        0.25,
        0.05
    )

    st.markdown("---")

    st.markdown("""
    ### 📌 Tipps
    
    - Nutze hochauflösende Bilder
    - Personen & Fahrzeuge funktionieren sehr gut
    - Niedriger Threshold = mehr Erkennungen
    """)

    st.markdown("---")

    st.success("✅ Modell geladen")

# ---------------------------------------------------
# UPLOAD
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Bild hochladen",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # Animation
    with st.spinner("🧠 KI analysiert das Bild..."):
        time.sleep(1)

        img_array = np.array(image)

        results = model.predict(
            source=img_array,
            conf=conf_threshold,
            save=False
        )

    result_img = results[0].plot()

    # ---------------------------------------------------
    # IMAGE DISPLAY
    # ---------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🖼️ Originalbild")
        st.image(image, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎯 KI-Erkennung")
        st.image(result_img, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------
    # STATS
    # ---------------------------------------------------
    boxes = results[0].boxes
    total_objects = len(boxes)

    avg_conf = 0
    if total_objects > 0:
        avg_conf = np.mean([float(box.conf[0]) for box in boxes])

    st.markdown("## 📊 Analyse")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{total_objects}</div>
            <div class="metric-label">Objekte erkannt</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{avg_conf:.2f}</div>
            <div class="metric-label">Ø Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        unique_classes = len(set([int(box.cls[0]) for box in boxes]))

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{unique_classes}</div>
            <div class="metric-label">Klassen erkannt</div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------------------------------
    # OBJECT TABLE
    # ---------------------------------------------------
    st.markdown("## 📋 Erkannte Objekte")

    data = []

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        data.append({
            "Objekt": model.names[cls_id],
            "Confidence": round(conf, 2)
        })

    if len(data) > 0:
        df = pd.DataFrame(data)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning("Keine Objekte erkannt.")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("""
<div class="footer">
    Entwickelt mit ❤️ | Streamlit + YOLOv8
</div>
""", unsafe_allow_html=True)

