import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import pandas as pd

# ---------------------------
# Seitenkonfiguration
# ---------------------------
st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------
# Custom CSS Design
# ---------------------------
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
        color: white;
    }

    .title {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #38bdf8;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        color: #cbd5e1;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }

    .stButton>button {
        background-color: #38bdf8;
        color: white;
        border-radius: 12px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #0ea5e9;
        transform: scale(1.03);
    }

    .box {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
    }

    .metric-card {
        background: #1e293b;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        color: white;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Titel
# ---------------------------
st.markdown('<div class="title">🤖 YOLO Object Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Lade ein Bild hoch und erkenne Objekte mit YOLOv8</div>',
    unsafe_allow_html=True
)

# ---------------------------
# Modell laden
# ---------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # eigenes Modell möglich

model = load_model()

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("⚙️ Einstellungen")

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.4,
        step=0.05
    )

    st.markdown("---")
    st.info("💡 Tipp: Nutze hochauflösende Bilder für bessere Ergebnisse.")

# ---------------------------
# File Upload
# ---------------------------
uploaded_file = st.file_uploader(
    "📤 Bild hochladen",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------
# Verarbeitung
# ---------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    # Layout mit zwei Spalten
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🖼️ Originalbild")
        st.image(image, use_column_width=True)

    # Bild in numpy konvertieren
    img_array = np.array(image)

    # Vorhersage
    results = model.predict(img_array, conf=conf_threshold)

    # Ergebnisbild
    result_img = results[0].plot()

    with col2:
        st.markdown("### 🎯 Erkanntes Bild")
        st.image(result_img, use_column_width=True)

    # ---------------------------
    # Statistiken
    # ---------------------------
    boxes = results[0].boxes

    st.markdown("## 📊 Analyse")

    total_objects = len(boxes)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{total_objects}</h2>
            <p>Erkannte Objekte</p>
        </div>
        """, unsafe_allow_html=True)

    avg_conf = 0
    if total_objects > 0:
        avg_conf = np.mean([float(box.conf[0]) for box in boxes])

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{avg_conf:.2f}</h2>
            <p>Durchschnittliche Confidence</p>
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------
    # Tabellenansicht
    # ---------------------------
    st.markdown("## 📋 Erkannte Objekte")

    data = []

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        data.append({
            "Objekt": model.names[cls_id],
            "Confidence": f"{conf:.2f}"
        })

    if len(data) > 0:
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("Keine Objekte erkannt.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown(
    "<center>Erstellt mit ❤️ und Streamlit + YOLOv8</center>",
    unsafe_allow_html=True
)
