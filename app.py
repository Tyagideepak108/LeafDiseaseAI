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
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Custom DepthwiseConv2D to handle compatibility
class CompatibleDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        # Remove 'groups' parameter if present
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

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

# --- Page Config + CSS + Language Toggle + Labels ---

st.set_page_config(page_title="🌱 Leaf Disease Detector", layout="wide", page_icon="🌿")

# function css 
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Aur fir function ko call karo
load_css("style.css")

# --- Language & Navigation ---
if "lang" not in st.session_state:
    st.session_state.lang = "en"

# Initialize prediction history
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# Modern Sidebar Navigation

st.sidebar.header("⚙️ Settings")

# Language Toggle
st.sidebar.markdown("**🗣️ Language:**")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🇬🇧 EN", key="lang_en"):
        st.session_state.lang = "en"
with col2:
    if st.button("🇮🇳 HI", key="lang_hi"):
        st.session_state.lang = "hi"

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
        "not_confident": "⚠️ Model is not confident with this image. Please take a clearer image of the leaf for better results.",
        "select_details": "👉 Select one to view details:",
        "no_details": "ℹ️ No details available for this disease.",
        "play_audio": "🔊 Play Audio",
        "home": "Home",
        "dashboard": "Dashboard",
        "crop_select": "🌾 Select Crop Type",
        "feedback_title": "📝 Feedback",
        "history": "History",
        "prediction_history": "📜 Prediction History",
        "clear_history": "🗑️ Clear History",
        "no_history": "No predictions yet. Upload an image to start!",
        "batch_upload": "📎 Batch Upload",
        "upload_multiple": "Upload Multiple Images",
        "processing_images": "Processing images...",
        "batch_results": "Batch Processing Results",
        "analytics_dashboard": "📊 Analytics Dashboard",
        "total_predictions": "Total Predictions",
        "accuracy": "Accuracy",
        "avg_confidence": "Avg Confidence",
        "diseases_found": "Diseases Found",
        "trends": "📈 Trends",
        "crops": "🌾 Crops",
        "diseases": "🦠 Diseases",
        "details": "📊 Details",
        "prediction_trends": "📈 Prediction Trends",
        "confidence_distribution": "📊 Confidence Distribution",
        "weekly_pattern": "📅 Weekly Pattern",
        "crop_analysis": "🌾 Crop Analysis",
        "crop_accuracy": "🎯 Crop-wise Accuracy",
        "disease_analysis": "🦠 Disease Analysis",
        "disease_severity": "⚠️ Disease Severity (by frequency)",
        "detailed_analytics": "📊 Detailed Analytics",
        "filter_date": "Filter by date range",
        "filter_crop": "Filter by crop",
        "showing_records": "Showing",
        "records": "records",
        "export_csv": "📾 Export Data as CSV",
        "download_csv": "Download CSV",
        "successful_predictions": "Successful Predictions",
        "made_by": "Developed with ❤️ by Deepak Tyagi" # ADDED FOR FOOTER
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
        "not_confident": "⚠️ मॉडल इस तस्वीर से आश्वस्त नहीं है। बेहतर परिणाम के लिए कृपया पत्ते की स्पष्ट तस्वीर लें।",
        "select_details": "👉 विवरण देखने के लिए एक चुनें:",
        "no_details": "ℹ️ इस बीमारी के लिए कोई विवरण उपलब्ध नहीं है।",
        "play_audio": "🔊 ऑडियो चलाएं",
        "home": "होम",
        "dashboard": "डैशबोर्ड",
        "crop_select": "🌾 फसल का प्रकार चुनें",
        "feedback_title": "📝 प्रतिक्रिया",
        "history": "इतिहास",
        "prediction_history": "📜 पूर्वानुमान इतिहास",
        "clear_history": "🗑️ इतिहास साफ़ करें",
        "no_history": "अभी तक कोई पूर्वानुमान नहीं। शुरू करने के लिए एक तस्वीर अपलोड करें!",
        "batch_upload": "📎 बैच अपलोड",
        "upload_multiple": "कई तस्वीरें अपलोड करें",
        "processing_images": "तस्वीरें प्रोसेस कर रहे हैं...",
        "batch_results": "बैच प्रोसेसिंग के परिणाम",
        "analytics_dashboard": "📊 एनालिटिक्स डैशबोर्ड",
        "total_predictions": "कुल पूर्वानुमान",
        "accuracy": "सटीकता",
        "avg_confidence": "औसत विश्वसनीयता",
        "diseases_found": "रोग मिले",
        "trends": "📈 रुझान",
        "crops": "🌾 फसलें",
        "diseases": "🦠 रोग",
        "details": "📊 विवरण",
        "prediction_trends": "📈 पूर्वानुमान रुझान",
        "confidence_distribution": "📊 विश्वसनीयता वितरण",
        "weekly_pattern": "📅 साप्ताहिक पैटर्न",
        "crop_analysis": "🌾 फसल विश्लेषण",
        "crop_accuracy": "🎯 फसल-वार सटीकता",
        "disease_analysis": "🦠 रोग विश्लेषण",
        "disease_severity": "⚠️ रोग की गंभीरता (आवृत्ति के आधार पर)",
        "detailed_analytics": "📊 विस्तृत एनालिटिक्स",
        "filter_date": "दिनांक सीमा के अनुसार फ़िल्टर करें",
        "filter_crop": "फसल के अनुसार फ़िल्टर करें",
        "showing_records": "दिखा रहे हैं",
        "records": "रिकॉर्ड",
        "export_csv": "📾 डेटा को CSV के रूप में निर्यात करें",
        "download_csv": "CSV डाउनलोड करें",
        "successful_predictions": "सफल पूर्वानुमान",
        "made_by": "दीपक त्यागी द्वारा विकसित ❤️ के साथ " # ADDED FOR FOOTER
    }
}
L = labels[st.session_state.lang]

# Initialize current page with English labels to avoid reset
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"  # Use key instead of label

# Navigation Menu
st.sidebar.markdown("---")
st.sidebar.markdown("**📌 Navigation**")

# Navigation buttons with consistent keys
navigation_items = [
    ("🏠", "home", L["home"]),
    ("📊", "dashboard", L["dashboard"]),
    ("📜", "history", L["history"]),
    ("📋", "batch_upload", L["batch_upload"])
]

for icon, page_key, page_label in navigation_items:
    if st.sidebar.button(f"{icon} {page_label}", key=f"nav_{page_key}", use_container_width=True):
        st.session_state.current_page = page_key

# Map current page key to label for comparison
page_mapping = {
    "home": L["home"],
    "dashboard": L["dashboard"],
    "history": L["history"],
    "batch_upload": L["batch_upload"]
}

page = page_mapping[st.session_state.current_page]

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

# --- Function: Load Model ---
@st.cache_resource
def load_model(model_path, crop_name):
    try:
        # Custom objects for compatibility
        custom_objects = {
            'DepthwiseConv2D': CompatibleDepthwiseConv2D
        }
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None



# --- HOME PAGE ---
if page == L["home"]:
    st.title(L["title"])



    # Sidebar crop selection
    st.sidebar.header(L["crop_select"])
    selected_crop = st.sidebar.selectbox(L["crop_select"], list(models_config.keys()))
    
    # Clear cache if crop changed
    if "current_crop" not in st.session_state:
        st.session_state.current_crop = selected_crop
    elif st.session_state.current_crop != selected_crop:
        st.cache_resource.clear()
        st.session_state.current_crop = selected_crop
    
    config = models_config[selected_crop]
    model = load_model(config["model_path"], selected_crop)  # Pass crop name for unique caching
    class_names = config["class_names"]

    # Load disease info JSON
    try:
        with open(config["info_path"], "r", encoding="utf-8") as f:
            disease_info = json.load(f)
    except Exception as e:
        st.error(f"❌ Error loading disease info: {e}")
        st.stop()

    # Image Input - Preserve image on language change
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None
    if "webcam_image" not in st.session_state:
        st.session_state.webcam_image = None
        
    st.header(L["upload_header"])
    upload_method = st.radio(L["input_method"], [L["upload_button"], L["webcam_button"]])
    image_input = None
    
    if upload_method == L["upload_button"]:
        uploaded = st.file_uploader(L["upload_button"], type=["jpg", "jpeg", "png"])
        if uploaded:
            st.session_state.uploaded_image = Image.open(uploaded)
            st.session_state.webcam_image = None
        if st.session_state.uploaded_image:
            image_input = st.session_state.uploaded_image
            st.image(image_input, caption=L["upload_button"], use_container_width=True)
            
    elif upload_method == L["webcam_button"]:
        cam_image = st.camera_input(L["webcam_button"])
        if cam_image:
            st.session_state.webcam_image = Image.open(cam_image)
            st.session_state.uploaded_image = None
        if st.session_state.webcam_image:
            image_input = st.session_state.webcam_image
            st.image(image_input, caption=L["webcam_button"], use_container_width=True)

    # ---- Prediction Logic ----
    final_pred = None
    final_conf = None

    if image_input and model:
        with st.spinner(L["analyzing"]):
            # Debug info
            st.write(f"🔍 Using model for: {selected_crop}")
            # st.write(f"📊 Model expects {len(class_names)} classes")
            # st.write(f"📋 Model path: {config['model_path']}")
            
            # Verify model output matches class names
            if model is not None:
                try:
                    # Test prediction shape
                    test_shape = model.output_shape
                    expected_classes = test_shape[-1] if test_shape else "Unknown"
                    # st.write(f"🤖 Model output classes: {expected_classes}")
                    
                    if expected_classes != len(class_names):
                        st.error(f"⚠️ Model mismatch! Model has {expected_classes} outputs but {len(class_names)} class names provided.")
                        st.stop()
                except:
                    pass
            
            predicted_class, confidence, is_confident = predict_disease(image_input, model, class_names)

        if not is_confident:  # Show message to take better image
            st.warning(L["not_confident"])
            st.info("💡 Tips for better image: \n- Use good lighting \n- Focus on the leaf clearly \n- Avoid blurry images \n- Capture the diseased area properly")
            st.stop()  # Stop processing and don't show predictions
        else:
            final_pred, final_conf = predicted_class, confidence
            
            # Save to prediction history
            history_entry = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "crop": selected_crop,
                "disease": final_pred,
                "confidence": final_conf,
                "image": image_input.copy()  # Store a copy of the image
            }
            st.session_state.prediction_history.insert(0, history_entry)  # Add to beginning
            
            # Keep only last 10 predictions
            if len(st.session_state.prediction_history) > 10:
                st.session_state.prediction_history = st.session_state.prediction_history[:10]

        # --- Show Disease Info ---
        if final_pred:
            st.subheader(L["prediction_result"])
            
            # Display disease name based on language
            display_name = final_pred
            if st.session_state.lang == "hi":
                matched_info_temp = disease_info.get(final_pred)
                if matched_info_temp and matched_info_temp.get('hindi_name'):
                    display_name = matched_info_temp.get('hindi_name')
            
            confidence_text = "Confidence" if st.session_state.lang == "en" else "विश्वसनीयता"
            
            st.markdown(
                f"<div style='border-radius:12px;padding:12px;margin-top:10px;"
                f"background-color:#1e1e2f;box-shadow:2px 2px 10px rgba(0,0,0,0.5);'>"
                f"<h4 style='color:#4CAF50;'>{display_name}</h4>"
                f"<p style='color:#ccc;'>{confidence_text}: {final_conf:.2f}%</p>"
                "</div>", unsafe_allow_html=True
            )

            matched_info = disease_info.get(final_pred)
            if matched_info:
                st.subheader(L["disease_info"])
                
                # Language-specific display
                if st.session_state.lang == "hi":
                    # Hindi mode - show only Hindi
                    hindi_name = matched_info.get('hindi_name', final_pred)
                    st.markdown(f"**रोग का नाम:** {hindi_name}")
                    
                    hin_cause = matched_info.get("cause_hindi", "N/A")
                    st.markdown(f"**कारण:** {hin_cause}")
                    if st.button("🔊 ऑडियो सुनें", key="hin_audio"):
                        audio_io = text_to_speech(hin_cause, 'hi')
                        if audio_io: st.audio(audio_io.getvalue(), format='audio/mp3')
                else:
                    # English mode - show only English
                    st.markdown(f"**Disease Name:** {final_pred}")
                    
                    eng_cause = matched_info.get("cause", "N/A")
                    st.markdown(f"**Cause:** {eng_cause}")
                    if st.button("🔊 Play Audio", key="eng_audio"):
                        audio_io = text_to_speech(eng_cause, 'en')
                        if audio_io: st.audio(audio_io.getvalue(), format='audio/mp3')

                st.divider()
                st.subheader(L["treatment_preventive"])
                tab1, tab2, tab3 = st.tabs([f"🌿 {L['organic']}", f"🧪 {L['chemical']}", f"🛡️ {L['preventive']}"])
                
                # Get language-specific treatment data
                if st.session_state.lang == "hi":
                    organic_treatments = matched_info.get("organic_treatment", [])
                    chemical_treatments = matched_info.get("chemical_treatment", [])
                    preventive_measures = matched_info.get("preventive_measures", [])
                else:
                    organic_treatments = matched_info.get("organic_treatment_en", [])
                    chemical_treatments = matched_info.get("chemical_treatment_en", [])
                    preventive_measures = matched_info.get("preventive_measures_en", [])
                
                with tab1:
                    for step in organic_treatments:
                        st.markdown(f"- {step}")
                with tab2:
                    for step in chemical_treatments:
                        st.markdown(f"- {step}")
                with tab3:
                    for step in preventive_measures:
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

# --- ANALYTICS DASHBOARD PAGE ---
elif page == L["dashboard"]:
    st.title(L["analytics_dashboard"])
    
    # Combine feedback data and prediction history
    feedback_exists = os.path.exists("feedback.csv")
    history_exists = len(st.session_state.prediction_history) > 0
    
    if not feedback_exists and not history_exists:
        st.info(L["no_feedback"])
    else:
        # Load and prepare data
        all_data = []
        
        # Add feedback data
        if feedback_exists:
            df_feedback = pd.read_csv("feedback.csv")
            df_feedback['source'] = 'feedback'
            # Add crop column if missing (for old data)
            if 'crop' not in df_feedback.columns:
                df_feedback['crop'] = 'Unknown'
            all_data.append(df_feedback)
        
        # Add prediction history
        if history_exists:
            history_df = pd.DataFrame(st.session_state.prediction_history)
            history_df['is_correct'] = 'Unknown'
            history_df['comment'] = 'N/A'
            history_df['image_path'] = 'N/A'
            history_df['source'] = 'history'
            # Rename columns to match feedback data
            history_df = history_df.rename(columns={'disease': 'prediction'})
            all_data.append(history_df[['timestamp', 'prediction', 'confidence', 'crop', 'is_correct', 'comment', 'source']])
        
        # Combine all data
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
            combined_df['date'] = combined_df['timestamp'].dt.date
            
            # Fix confidence column - convert to numeric
            combined_df['confidence'] = pd.to_numeric(combined_df['confidence'].astype(str).str.replace('%', ''), errors='coerce')
            
            # --- KEY METRICS ---
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Total Predictions", len(combined_df))
            
            with col2:
                if feedback_exists:
                    accuracy = len(combined_df[combined_df['is_correct']=='Yes']) / len(combined_df[combined_df['is_correct'].isin(['Yes', 'No'])]) * 100
                    st.metric("🎯 Accuracy", f"{accuracy:.1f}%")
                else:
                    st.metric("🎯 Accuracy", "N/A")
            
            with col3:
                avg_confidence = combined_df['confidence'].mean()
                st.metric("📊 Avg Confidence", f"{avg_confidence:.1f}%")
            
            with col4:
                unique_diseases = combined_df['prediction'].nunique()
                st.metric("🦠 Diseases Found", unique_diseases)
            
            st.divider()
            
            # --- ANALYTICS TABS ---
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "🌾 Crops", "🦠 Diseases", "📊 Details"])
            
            with tab1:
                st.subheader("📈 Prediction Trends")
                
                # Daily predictions
                daily_counts = combined_df.groupby('date').size().reset_index(name='count')
                st.line_chart(daily_counts.set_index('date'))
                
                # Confidence distribution
                st.subheader("📊 Confidence Distribution")
                confidence_bins = pd.cut(combined_df['confidence'], bins=[0, 50, 70, 85, 100], labels=['Low (0-50%)', 'Medium (50-70%)', 'High (70-85%)', 'Very High (85-100%)'])
                st.bar_chart(confidence_bins.value_counts())
                
                # Weekly pattern
                combined_df['weekday'] = combined_df['timestamp'].dt.day_name()
                weekday_counts = combined_df['weekday'].value_counts()
                st.subheader("📅 Weekly Pattern")
                st.bar_chart(weekday_counts)
            
            with tab2:
                st.subheader("🌾 Crop Analysis")
                
                if 'crop' in combined_df.columns:
                    crop_counts = combined_df['crop'].value_counts()
                    st.bar_chart(crop_counts)
                    
                    # Crop-wise accuracy
                    if feedback_exists:
                        crop_accuracy = combined_df[combined_df['is_correct'].isin(['Yes', 'No'])].groupby('crop')['is_correct'].apply(lambda x: (x=='Yes').mean() * 100)
                        st.subheader("🎯 Crop-wise Accuracy")
                        st.bar_chart(crop_accuracy)
                else:
                    st.info("Crop data not available in older predictions")
            
            with tab3:
                st.subheader("🦠 Disease Analysis")
                
                # Top diseases
                disease_counts = combined_df['prediction'].value_counts().head(10)
                st.bar_chart(disease_counts)
                
                # Disease severity (based on frequency)
                st.subheader("⚠️ Disease Severity (by frequency)")
                severity_data = disease_counts.head(5)
                for disease, count in severity_data.items():
                    severity = "High" if count > severity_data.mean() else "Medium" if count > severity_data.mean()/2 else "Low"
                    color = "🔴" if severity == "High" else "🟡" if severity == "Medium" else "🟢"
                    st.write(f"{color} **{disease}**: {count} cases ({severity} risk)")
            
            with tab4:
                st.subheader("📊 Detailed Analytics")
                
                # Filter options
                col_filter1, col_filter2 = st.columns(2)
                
                with col_filter1:
                    date_filter = st.date_input("Filter by date range", value=[combined_df['date'].min(), combined_df['date'].max()])
                
                with col_filter2:
                    if 'crop' in combined_df.columns:
                        crop_filter = st.multiselect("Filter by crop", combined_df['crop'].unique(), default=combined_df['crop'].unique())
                    else:
                        crop_filter = []
                
                # Apply filters
                filtered_df = combined_df.copy()
                if len(date_filter) == 2:
                    filtered_df = filtered_df[(filtered_df['date'] >= date_filter[0]) & (filtered_df['date'] <= date_filter[1])]
                if crop_filter and 'crop' in combined_df.columns:
                    filtered_df = filtered_df[filtered_df['crop'].isin(crop_filter)]
                
                # Show filtered data
                st.write(f"**Showing {len(filtered_df)} records**")
                # Only show columns that exist
                display_cols = ['timestamp', 'prediction', 'confidence', 'is_correct']
                if 'crop' in filtered_df.columns:
                    display_cols.insert(3, 'crop')
                st.dataframe(filtered_df[display_cols].sort_values('timestamp', ascending=False))
                
                # Export button
                if st.button("💾 Export Data as CSV"):
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"leaf_disease_analytics_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

# --- HISTORY PAGE ---
elif page == L["history"]:
    st.title(L["prediction_history"])
    
    if not st.session_state.prediction_history:
        st.info(L["no_history"])
    else:
        # Clear history button
        if st.button(L["clear_history"]):
            st.session_state.prediction_history = []
            st.success("History cleared!" if st.session_state.lang == "en" else "इतिहास साफ़ कर दिया गया!")
            st.rerun()
        
        st.write(f"**Total Predictions:** {len(st.session_state.prediction_history)}")
        
        # Display history in cards
        for i, entry in enumerate(st.session_state.prediction_history):
            with st.container():
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(entry["image"], width=150)
                
                with col2:
                    st.markdown(f"**🕒 Time:** {entry['timestamp']}")
                    st.markdown(f"**🌾 Crop:** {entry['crop']}")
                    st.markdown(f"**🦠 Disease:** {entry['disease']}")
                    st.markdown(f"**📊 Confidence:** {entry['confidence']:.2f}%")
                
                st.divider()

# --- BATCH UPLOAD PAGE ---
elif page == L["batch_upload"]:
    st.title(L["batch_upload"])
    
    # Crop selection for batch processing
    st.sidebar.header(L["crop_select"])
    selected_crop = st.sidebar.selectbox(L["crop_select"], list(models_config.keys()), key="batch_crop")
    
    config = models_config[selected_crop]
    model = load_model(config["model_path"], selected_crop)
    class_names = config["class_names"]
    
    # Load disease info
    try:
        with open(config["info_path"], "r", encoding="utf-8") as f:
            disease_info = json.load(f)
    except Exception as e:
        st.error(f"❌ Error loading disease info: {e}")
        st.stop()
    
    # Multiple file upload
    uploaded_files = st.file_uploader(
        L["upload_multiple"], 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )
    
    if uploaded_files and model:
        st.write(f"**Selected {len(uploaded_files)} images for processing**")
        
        # Process button
        if st.button(f"🚀 Process {len(uploaded_files)} Images"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"{L['processing_images']} {i+1}/{len(uploaded_files)}")
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                try:
                    image = Image.open(uploaded_file)
                    predicted_class, confidence, is_confident = predict_disease(image, model, class_names)
                    
                    results.append({
                        "filename": uploaded_file.name,
                        "image": image,
                        "prediction": predicted_class if is_confident else "Low Confidence",
                        "confidence": confidence if is_confident else 0,
                        "is_confident": is_confident
                    })
                except Exception as e:
                    results.append({
                        "filename": uploaded_file.name,
                        "image": None,
                        "prediction": "Error",
                        "confidence": 0,
                        "is_confident": False,
                        "error": str(e)
                    })
            
            status_text.text("✅ Processing Complete!")
            progress_bar.empty()
            
            # Display results
            st.subheader(L["batch_results"])
            
            # Summary stats
            confident_predictions = sum(1 for r in results if r["is_confident"])
            st.metric("Successful Predictions", f"{confident_predictions}/{len(results)}")
            
            # Results in columns
            cols = st.columns(2)
            for i, result in enumerate(results):
                with cols[i % 2]:
                    if result["image"]:
                        st.image(result["image"], width=200)
                    
                    st.write(f"**File:** {result['filename']}")
                    
                    if result["is_confident"]:
                        # Get display name
                        display_name = result["prediction"]
                        if st.session_state.lang == "hi":
                            matched_info = disease_info.get(result["prediction"])
                            if matched_info and matched_info.get('hindi_name'):
                                display_name = matched_info.get('hindi_name')
                        
                        st.success(f"🌿 **{display_name}**")
                        st.write(f"Confidence: {result['confidence']:.2f}%")
                    else:
                        st.warning("⚠️ Low confidence - unclear image")
                    
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    
                    st.divider()

# --- FOOTER ---
# ==================================
footer_html = f"""
<div class="footer">
    <p>{L['made_by']} | <a href="https://github.com/Tyagideepak108" target="_blank">GitHub</a> | <a href="https://www.linkedin.com/in/tyagi-deepak/" target="_blank">LinkedIn</a></p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)

