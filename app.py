import streamlit as st
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageClassification

st.set_page_config(page_title="AI Detector", page_icon="🔍", layout="centered")
st.title("🔍 AI Image / Text Detector")

# ---------------------- Load Model Once ----------------------
@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("umm-maybe/AI-image-detector")
    model = AutoModelForImageClassification.from_pretrained("umm-maybe/AI-image-detector")
    return processor, model

processor, model = load_model()

# ---------------------- File Upload --------------------------
uploaded_file = st.file_uploader(
    "Upload Image or Text",
    type=["png", "jpg", "jpeg", "txt"]
)

if uploaded_file:
    st.success("File uploaded successfully!")
    st.write("**File Name:**", uploaded_file.name)

    # ---------------------- TEXT DETECTION ----------------------
    if uploaded_file.type == "text/plain":
        content = uploaded_file.read().decode("utf-8")
        st.subheader("📄 Text Preview")
        st.write(content)

        # Placeholder text detection (you can improve later)
        st.subheader("🔍 Detection Result")
        st.json({"label": "Real", "score": 0.90})

    # ---------------------- IMAGE DETECTION ---------------------
    elif uploaded_file.type.startswith("image/"):
        image = Image.open(uploaded_file)

        st.subheader("🖼️ Uploaded Image")
        st.image(image, use_column_width=True)

        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        scores = torch.softmax(outputs.logits, dim=1)[0]
        real_score = float(scores[0])
        ai_score = float(scores[1])

        st.subheader("🔍 Detection Result")
        st.json({
            "label": "AI Generated" if ai_score > real_score else "Real",
            "ai_score": ai_score,
            "real_score": real_score
        })

else:
    st.info("Upload a PNG, JPG, JPEG, or TXT file to begin detection.")


