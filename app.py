import streamlit as st
from PIL import Image
import json
import tensorflow as tf
from predict import predict_disease, preprocess_image
from gtts import gTTS
import io
import os
import pandas as pd
import datetime

# --- Class Names ---
sugarcane_class_names = sorted([
    "Banded Chlorosis", "Brown Spot", "BrownRust", "Dried Leaves", "Grassy shoot",
    "Healthy Leaves", "Pokkah Boeng", "Sett Rot", "smut", "Viral Disease", "Yellow Leaf" , "Red Stripe(viral Disease)"
])

other_class_names = sorted([
    "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
])

# --- Model and Info Paths ---
models_config = {
    "Sugarcane": {
        "model_path": "models/sugarcane_phase2_best.h5",
        "info_path": "sugercane_info.json",
        "class_names": sugarcane_class_names
    },
    "Other Crops": {
        "model_path": "models/other_crops_model_best.h5",
        "info_path": "other_diseases_info.json",
        "class_names": other_class_names
    }
}

# ====================================================
# --- Page Config + CSS + Language Toggle + Labels ---
# ====================================================
st.set_page_config(page_title="🌱 Leaf Disease Detector", layout="wide", page_icon="🌿")

def load_css():
    st.markdown(
        """
        <style>
        .main {background-color: #f8fff4;}
        .stButton button {
            border-radius: 12px;
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
        }
        .stButton button:hover {background-color: #45a049;}
        .prediction-card {
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
            margin-bottom: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
load_css()

# --- Language & Navigation ---
if "lang" not in st.session_state:
    st.session_state.lang = "en"

st.sidebar.header("⚙️ Settings")
lang_option = st.sidebar.radio("🗣️ Language", ["English 🇬🇧", "Hindi 🇮🇳"])
if lang_option == "Hindi 🇮🇳":
    st.session_state.lang = "hi"
else:
    st.session_state.lang = "en"

# --- Labels Dictionary ---
labels = {
    "en": {
        "title": "🌿 Leaf Disease Detection App",
        "upload_header": "Upload an Image",
        "input_method": "Choose Input Method",
        "upload_button": "Upload Image",
        "webcam_button": "Use Webcam",
        "prediction_result": "Prediction Result",
        "disease_info": "Disease Information",
        "treatment_preventive": "Treatment & Preventive Measures",
        "organic": "Organic",
        "chemical": "Chemical",
        "preventive": "Preventive",
        "feedback_question": "Is this prediction correct?",
        "yes_button": "Yes 👍",
        "no_button": "No 👎",
        "feedback_form_title": "Please tell us what was wrong:",
        "submit_button": "Submit",
        "thanks_success": "Thanks! Feedback saved.",
        "dashboard_title": "Feedback Dashboard",
        "raw_data": "Raw Data",
        "correct_incorrect": "Correct vs Incorrect",
        "top_predictions": "Top Predicted Diseases",
        "no_feedback": "No feedback available yet.",
        "analyzing": "🔍 Analyzing...",
        "not_confident": "⚠️ Model not fully confident, showing top 3 possible predictions:",
        "select_details": "👉 Select one to view details:",
        "no_details": "ℹ️ No details available for this disease.",
        "play_audio": "🔊 Play Audio",
        "home": "Home",
        "dashboard": "Dashboard",
        "crop_select": "🌾 Select Crop Type",
        "feedback_title": "📝 Feedback"
    },
    "hi": {
        "title": "🌿 पत्तों की बीमारी का पता लगाने वाला ऐप",
        "upload_header": "एक तस्वीर अपलोड करें",
        "input_method": "इनपुट विधि चुनें",
        "upload_button": "तस्वीर अपलोड करें",
        "webcam_button": "वेबकैम का उपयोग करें",
        "prediction_result": "पूर्वानुमान का परिणाम",
        "disease_info": "बीमारी की जानकारी",
        "treatment_preventive": "उपचार और निवारक उपाय",
        "organic": "जैविक",
        "chemical": "रासायनिक",
        "preventive": "निवारक",
        "feedback_question": "क्या यह पूर्वानुमान सही है?",
        "yes_button": "हाँ 👍",
        "no_button": "नहीं 👎",
        "feedback_form_title": "कृपया बताएं कि क्या गलत था:",
        "submit_button": "जमा करें",
        "thanks_success": "धन्यवाद! प्रतिक्रिया सहेज ली गई है।",
        "dashboard_title": "प्रतिक्रिया डैशबोर्ड",
        "raw_data": "कच्चा डेटा",
        "correct_incorrect": "सही बनाम गलत",
        "top_predictions": "शीर्ष अनुमानित बीमारियाँ",
        "no_feedback": "अभी कोई प्रतिक्रिया उपलब्ध नहीं है।",
        "analyzing": "🔍 विश्लेषण कर रहा है...",
        "not_confident": "⚠️ मॉडल पूरी तरह से आश्वस्त नहीं है, शीर्ष 3 संभावित पूर्वानुमान दिखा रहा है:",
        "select_details": "👉 विवरण देखने के लिए एक चुनें:",
        "no_details": "ℹ️ इस बीमारी के लिए कोई विवरण उपलब्ध नहीं है।",
        "play_audio": "🔊 ऑडियो चलाएं",
        "home": "होम",
        "dashboard": "डैशबोर्ड",
        "crop_select": "🌾 फसल का प्रकार चुनें",
        "feedback_title": "📝 प्रतिक्रिया"
    }
}
L = labels[st.session_state.lang]

page = st.sidebar.radio("📌 Go to", [L["home"], L["dashboard"]])

# --- Function: TTS ---
def text_to_speech(text, lang):
    if not text:
        return None
    try:
        tts = gTTS(text, lang=lang)
        audio_bytes_io = io.BytesIO()
        tts.write_to_fp(audio_bytes_io)
        return audio_bytes_io
    except Exception as e:
        st.error(f"❌ Error generating audio: {e}")
        return None

# --- HOME PAGE ---
if page == L["home"]:
    st.title(L["title"])

    @st.cache_resource
    def load_model(model_path):
        try:
            # Fix for DepthwiseConv2D compatibility
            return tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None

    # Sidebar crop selection
    st.sidebar.header(L["crop_select"])
    selected_crop = st.sidebar.selectbox(L["crop_select"], list(models_config.keys()))
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

    # Image Input
    st.header(L["upload_header"])
    upload_method = st.radio(L["input_method"], [L["upload_button"], L["webcam_button"]])
    image_input = None
    if upload_method == L["upload_button"]:
        uploaded = st.file_uploader(L["upload_button"], type=["jpg", "jpeg", "png"])
        if uploaded:
            image_input = Image.open(uploaded)
            st.image(image_input, caption=L["upload_button"], use_column_width=True)
    elif upload_method == L["webcam_button"]:
        cam_image = st.camera_input(L["webcam_button"])
        if cam_image:
            image_input = Image.open(cam_image)
            st.image(image_input, caption=L["webcam_button"], use_column_width=True)

    # ---- Prediction Logic ----
    final_pred = None
    final_conf = None

    if image_input and model:
        with st.spinner(L["analyzing"]):
            predicted_class, confidence, is_confident = predict_disease(image_input, model, class_names)

        if not is_confident:  # Show top-3 choices
            st.warning(L["not_confident"])
            processed_img = preprocess_image(image_input, model)
            predictions = model.predict(processed_img)[0]
            top_indices = predictions.argsort()[-3:][::-1]
            top_preds = [(class_names[i], predictions[i]) for i in top_indices]

            selected_disease = st.radio(
                L["select_details"],
                [f"{name} ({conf*100:.2f}%)" for name, conf in top_preds],
                key="top3_radio"
            )
            final_pred = selected_disease.split(" (")[0]
            final_conf = dict(top_preds)[final_pred] * 100
        else:
            final_pred, final_conf = predicted_class, confidence

        # --- Show Disease Info ---
        if final_pred:
            st.subheader(L["prediction_result"])
            st.markdown(
                f"<div style='border-radius:12px;padding:12px;margin-top:10px;"
                f"background-color:#1e1e2f;box-shadow:2px 2px 10px rgba(0,0,0,0.5);'>"
                f"<h4 style='color:#4CAF50;'>{final_pred}</h4>"
                f"<p style='color:#ccc;'>Confidence: {final_conf:.2f}%</p>"
                "</div>", unsafe_allow_html=True
            )

            matched_info = disease_info.get(final_pred)
            if matched_info:
                st.subheader(L["disease_info"])
                st.markdown(f"**Hindi Name:** {matched_info.get('hindi_name','N/A')}")

                eng_cause = matched_info.get("cause", "N/A")
                st.markdown(f"**Cause (English):** {eng_cause}")
                if st.button("🔊 Play Cause (English)", key="eng_audio"):
                    audio_io = text_to_speech(eng_cause, 'en')
                    if audio_io: st.audio(audio_io.getvalue(), format='audio/mp3')

                hin_cause = matched_info.get("cause_hindi", "N/A")
                st.markdown(f"**Cause (Hindi):** {hin_cause}")
                if st.button("🔊 Play Cause (Hindi)", key="hin_audio"):
                    audio_io = text_to_speech(hin_cause, 'hi')
                    if audio_io: st.audio(audio_io.getvalue(), format='audio/mp3')

                st.divider()
                st.subheader(L["treatment_preventive"])
                tab1, tab2, tab3 = st.tabs([f"🌿 {L['organic']}", f"🧪 {L['chemical']}", f"🛡️ {L['preventive']}"])
                with tab1:
                    for step in matched_info.get("organic_treatment", []):
                        st.markdown(f"- {step}")
                with tab2:
                    for step in matched_info.get("chemical_treatment", []):
                        st.markdown(f"- {step}")
                with tab3:
                    for step in matched_info.get("preventive_measures", []):
                        st.markdown(f"- {step}")
            else:
                st.warning(L["no_details"])

            # --- Feedback Section ---
            st.divider()
            st.header(L["feedback_title"])
            st.write(L["feedback_question"])

            # Save image for feedback
            if not os.path.exists("feedback_images"):
                os.makedirs("feedback_images")
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"feedback_images/{final_pred}_{timestamp_str}.png"
            image_input.save(image_filename)

            col_yes, col_no = st.columns(2)
            if col_yes.button(L["yes_button"]):
                feedback_data = {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "prediction": final_pred,
                    "confidence": f"{final_conf:.2f}%",
                    "is_correct": "Yes",
                    "comment": "N/A",
                    "image_path": image_filename
                }
                pd.DataFrame([feedback_data]).to_csv(
                    "feedback.csv", mode='a', header=not os.path.exists("feedback.csv"), index=False
                )
                st.success(L["thanks_success"])

            if col_no.button(L["no_button"]):
                st.session_state.show_comment_box = True

            if st.session_state.get("show_comment_box"):
                with st.form("feedback_form"):
                    user_comment = st.text_area(L["feedback_form_title"])
                    submit = st.form_submit_button(L["submit_button"])
                    if submit:
                        feedback_data = {
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "prediction": final_pred,
                            "confidence": f"{final_conf:.2f}%",
                            "is_correct": "No",
                            "comment": user_comment,
                            "image_path": image_filename
                        }
                        pd.DataFrame([feedback_data]).to_csv(
                            "feedback.csv", mode='a', header=not os.path.exists("feedback.csv"), index=False
                        )
                        st.success(L["thanks_success"])
                        st.session_state.show_comment_box = False
                        st.rerun()

# --- DASHBOARD PAGE ---
elif page == L["dashboard"]:
    st.title(L["dashboard_title"])

    if not os.path.exists("feedback.csv"):
        st.info(L["no_feedback"])
    else:
        df = pd.read_csv("feedback.csv")
        st.subheader(L["raw_data"])
        st.dataframe(df)

        st.subheader(L["correct_incorrect"])
        st.bar_chart(df["is_correct"].value_counts())

        st.subheader(L["top_predictions"])
        st.bar_chart(df["prediction"].value_counts().head(5))

        st.write(f"✅ Correct: {len(df[df['is_correct']=='Yes'])}")
        st.write(f"❌ Incorrect: {len(df[df['is_correct']=='No'])}")
