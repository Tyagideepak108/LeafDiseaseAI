import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # Important
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

# --- 1. Parameters ---
# Apne dataset ka path yahan daalein
data_dir = 'sugercane'

# MobileNetV2 224x224 images par train hua hai, isliye hum yahi size istemaal karenge
img_height, img_width = 224, 224
batch_size = 32
epochs = 25

# --- 2. Data Loading & Augmentation ---
# ImageDataGenerator ko MobileNetV2 ke liye update kiya gaya hai
# Rescale ki jagah hum preprocess_input function ka istemal karenge
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, # MobileNetV2 ka pre-processing function
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

# --- 3. Class Imbalance ko Handle karna ---
# Class weights calculate karein taaki model kam images wali classes par zyada dhyaan de
class_labels = np.unique(train_data.classes)
class_weights_calculated = compute_class_weight(
    class_weight='balanced',
    classes=class_labels,
    y=train_data.classes
)
class_weights = dict(zip(class_labels, class_weights_calculated))
print("Calculated Class Weights:", class_weights)

# --- 4. Transfer Learning Model Banana ---
# MobileNetV2 base model load karein (bina upar ki classification layer ke)
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(img_height, img_width, 3)
)

# Base model ki layers ko freeze kar dein
base_model.trainable = False

# Nayi classification layers add karein
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# --- 5. Model Compile karna ---
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # Learning rate ko fine-tuning ke liye adjust kiya
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 6. Callbacks ---
# Sunischit karein ki 'models' directory maujood hai
os.makedirs('models', exist_ok=True)

checkpoint = ModelCheckpoint(
    'models/sugarcane_disease_model_best.h5',
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

# --- 7. Model Train karna ---
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=callbacks,
    class_weight=class_weights # Class imbalance ko handle karne ke liye
)

# Final model save karein
model.save('models/sugarcane_disease_model_final.h5')
print("\nTraining complete. Best model saved as 'sugarcane_disease_model_best.h5' and final model as 'sugarcane_disease_model_final.h5'")

