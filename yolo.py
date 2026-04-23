import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Modell laden (du kannst hier dein eigenes YOLO26-Modell einsetzen)
model = YOLO("yolov8n.pt")  # ersetze durch dein Modell: "yolo26.pt"

st.title("YOLO Object Detection mit Streamlit")

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Originalbild", use_column_width=True)

    # Bild in numpy konvertieren
    img_array = np.array(image)

    # YOLO Vorhersage
    results = model(img_array)

    # Ergebnis rendern
    result_img = results[0].plot()

    st.image(result_img, caption="Erkanntes Bild", use_column_width=True)

    # Optional: Bounding Box Infos anzeigen
    st.subheader("Erkannte Objekte:")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        st.write(f"Klasse: {model.names[cls_id]}, Confidence: {conf:.2f}")
