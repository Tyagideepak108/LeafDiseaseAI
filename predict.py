# predict.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def preprocess_image(image, model):
    """Resize + preprocess image for the given model"""
    if image is None or model is None:
        return None  # Guard clause

    try:
        # Convert to RGB if image has alpha channel
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        input_shape = model.input_shape[1:3]
        image = image.resize(input_shape)
        image_array = img_to_array(image)
        image_array = tf.expand_dims(image_array, axis=0)
        processed_image = preprocess_input(image_array)
        return processed_image
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        return None

def predict_disease(image, model, class_names, threshold=60):
    """Return predicted class, confidence %, is_confident"""
    processed_image = preprocess_image(image, model)
    if processed_image is None:
        return None, 0.0, False  #  Safe return

    try:
        predictions = model.predict(processed_image)
        predicted_index = np.argmax(predictions)
        predicted_class = class_names[predicted_index]
        confidence = float(predictions[0][predicted_index]) * 100
        is_confident = confidence >= threshold
        
        # Debug info
        print(f"[DEBUG] Model output shape: {predictions.shape}")
        print(f"[DEBUG] Number of classes: {len(class_names)}")
        print(f"[DEBUG] Predicted index: {predicted_index}")
        print(f"[DEBUG] Predicted class: {predicted_class}")
        print(f"[DEBUG] Top 3 predictions: {np.argsort(predictions[0])[-3:][::-1]}")
        
        return predicted_class, confidence, is_confident
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return None, 0.0, False
