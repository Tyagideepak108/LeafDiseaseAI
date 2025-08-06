# app.py (Updated)
import streamlit as st
from PIL import Image
import json
import tensorflow as tf
from predict import predict_disease

# --- Class Names (Alphabetical Order - Bahut Zaroori) ---
# Training ke time ImageDataGenerator folders ko alphabetically sort karta hai.
# Isliye yahan bhi order wahi hona chahiye.
sugarcane_class_names = sorted([
    "Banded Chlorosis", "Brown Spot", "BrownRust", "Dried Leaves", "Grassy shoot",
    "Healthy Leaves", "Pokkah Boeng", "Sett Rot", "smut", "Viral Disease", "Yellow Leaf"
])

other_class_names = sorted([
    "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
])

# --- Model Paths ---
models_config = {
    "Sugarcane": {
        "model_path": "models/sugarcane_disease_model_best.h5",
        "info_path": "sugercane_info.json",
        "class_names": sugarcane_class_names
    },
    "Other Crops": {
        "model_path": "models/other_crops_model_best.h5",
        "info_path": "other_diseases_info.json",
        "class_names": other_class_names
    }
}

st.set_page_config(page_title="🌱 Leaf Disease Detector", layout="wide")
st.title("🌿 Leaf Disease Detection App (Bilingual)")

# --- Model Loading (Efficient Way using Cache) ---
# Ye function model ko sirf ek baar load karega aur cache mein save kar lega.
@st.cache_resource
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# --- Sidebar ---
selected_crop = st.sidebar.selectbox("🌾 Select Crop Type", list(models_config.keys()))
config = models_config[selected_crop]
model = load_model(config["model_path"])
class_names = config["class_names"]

# Load disease info JSON
try:
    with open(config["info_path"], "r", encoding="utf-8") as f:
        disease_info = json.load(f)
except Exception as e:
    st.error(f"❌ Error loading disease info: {e}")
    st.stop()

# --- Image Input ---
upload_method = st.radio("📸 Choose Input Method", ["Upload Image", "Use Webcam"])
image_input = None
if upload_method == "Upload Image":
    uploaded = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image_input = Image.open(uploaded)
        st.image(image_input, caption="Uploaded Leaf Image", use_column_width=True)

elif upload_method == "Use webcam":
    cam_image = st.camera_input("Take a picture")
    if cam_image:
        image_input = Image.open(cam_image)
        st.image(image_input, caption="Captured Leaf Image", use_column_width=True)

# --- Prediction ---
if image_input and model:
    with st.spinner("🔍 Analyzing..."):
        predicted_class, confidence = predict_disease(image_input, model, class_names)
        st.success(f"✅ Prediction: **{predicted_class}** ({confidence:.2f}% confidence)")

        # Lookup JSON Info
        disease_key = predicted_class.replace(" ", "_").replace("___", "_").replace("__", "_")
        matched_info = disease_info.get(predicted_class) or disease_info.get(disease_key)

        if matched_info:
            st.subheader("📄 Disease Information")
            st.markdown(f"**🌱 Hindi Name:** {matched_info.get('hindi_name', '❌ Not available')}")
            st.markdown(f"**💠 English Cause:** {matched_info.get('cause', '❌ Not available')}")
            st.markdown(f"**🔍 Hindi Cause:** {matched_info.get('cause_hindi', matched_info.get('cause', '❌'))}")
            st.markdown(f"**💊 English Treatment:** {matched_info.get('treatment', '❌ Not available')}")
            st.markdown(f"**🌿 Hindi Treatment:** {matched_info.get('treatment_hindi', matched_info.get('treatment', '❌'))}")
        else:
            st.warning("ℹ️ No detailed info available for this prediction.")