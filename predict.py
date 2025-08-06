# predict.py (Updated)
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def predict_disease(image, model, class_names):
    """
    Ye function ek pre-loaded model leta hai aur prediction karta hai.
    """
    # Model ke anusaar input shape match karein (224, 224)
    input_shape = model.input_shape[1:3]
    image = image.resize(input_shape)
    
    # Image ko array mein convert karein
    image_array = img_to_array(image)
    
    # Batch dimension add karein
    image_array = tf.expand_dims(image_array, axis=0)
    
    # MobileNetV2 ke liye sahi preprocessing istemaal karein
    processed_image = preprocess_input(image_array)

    # Prediction karein
    predictions = model.predict(processed_image)
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[0][predicted_index]) * 100

    return predicted_class, confidence