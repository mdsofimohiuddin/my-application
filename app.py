import streamlit as st
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageClassification

st.title("AI Image / Video / Text Detector")

# Load model once
@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("umm-maybe/AI-image-detector")
    model = AutoModelForImageClassification.from_pretrained("umm-maybe/AI-image-detector")
    return processor, model

processor, model = load_model()

uploaded_file = st.file_uploader(
    "Upload Image, Text, or Video",
    type=["png", "jpg", "jpeg", "txt", "mp4", "mov"]
)

if uploaded_file:
    st.success("File uploaded!")
    st.write("File name:", uploaded_file.name)

    # ------------ TEXT DETECTION ------------
    if uploaded_file.type == "text/plain":
        content = uploaded_file.read().decode("utf-8")
        st.write(content)
        
        # simple text detection placeholder
        st.json({"label": "Real", "score": 0.90})

    # ------------ IMAGE DETECTION (REAL) ------------
    elif uploaded_file.type.startswith("image/"):
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        scores = torch.softmax(outputs.logits, dim=1)[0]
        real_score = float(scores[0])
        ai_score = float(scores[1])

        st.json({
            "label": "AI Generated" if ai_score > real_score else "Real",
            "ai_score": ai_score,
            "real_score": real_score
        })

    # ------------ VIDEO (PLACEHOLDER) ------------
    elif uploaded_file.type.startswith("video/"):
        st.video(uploaded_file)
        st.json({"label": "Real", "score": 0.6})
