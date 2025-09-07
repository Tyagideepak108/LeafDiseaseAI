# other_crops_train.py (Improved Version)
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

# --- 1. Parameters ---
data_dir = 'PlantVillage' 

# MobileNetV2 224x224 images par train hua hai
img_height, img_width = 224, 224
batch_size = 32
epochs = 25

# --- 2. Data Augmentation & Loading ---
# Training data ke liye augmentation aur preprocessing
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation data ke liye sirf preprocessing (augmentation nahi)
validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_data = validation_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

# --- 3. Class Imbalance ko Handle Karna ---
print("Calculating class weights...")
class_labels = np.unique(train_data.classes)
class_weights_calculated = compute_class_weight(
    class_weight='balanced',
    classes=class_labels,
    y=train_data.classes
)
class_weights = dict(zip(class_labels, class_weights_calculated))
print("Calculated Class Weights:", class_weights)

# --- 4. Transfer Learning Model ---
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(img_height, img_width, 3)
)
base_model.trainable = False  # Base model ko freeze karein

# Nayi layers add karein
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# --- 5. Model Compile Karna ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- 6. Callbacks ---
os.makedirs('models', exist_ok=True)
checkpoint = ModelCheckpoint(
    'models/other_crops_model_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)
callbacks = [checkpoint, early_stop]

# --- 7. Model Train Karna ---
print("\nStarting model training...")
model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=callbacks,
    class_weight=class_weights  # Class weights ka istemal karein
)

# Final model ko save karna zaroori nahi, kyunki checkpoint best model save kar raha hai
# Lekin agar aap chahein to kar sakte hain
model.save('models/other_crops_model_final.h5')
print("\nTraining complete. Best model saved as 'other_crops_model_best.h5'")