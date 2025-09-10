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
data_dir = 'sugercane'   
img_height, img_width = 224, 224
batch_size = 32
epochs_phase1 = 10   # Warmup epochs
epochs_phase2 = 15   # Fine-tuning epochs

# --- 2. Data Loading & Augmentation ---
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.3,
    shear_range=0.15,
    brightness_range=[0.7, 1.3],  # Real-life images ke liye brightness augment
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

# --- 3. Class Weights ---
class_labels = np.unique(train_data.classes)
class_weights_calculated = compute_class_weight(
    class_weight='balanced',
    classes=class_labels,
    y=train_data.classes
)
class_weights = dict(zip(class_labels, class_weights_calculated))
print("Calculated Class Weights:", class_weights)

# --- 4. Base Model ---
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(img_height, img_width, 3)
)

# --- 5. Build Model ---
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_data.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# --- 6. Phase 1: Warmup Training ---
base_model.trainable = False
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

os.makedirs('models', exist_ok=True)
checkpoint = ModelCheckpoint(
    'models/sugarcane_phase1_best.h5',
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

print("\n--- Phase 1: Training Classifier Only ---\n")
history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs_phase1,
    callbacks=[checkpoint, early_stop],
    class_weight=class_weights
)

# --- 7. Phase 2: Fine-Tuning ---
# Last 30 layers unfreeze karenge
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_ft = ModelCheckpoint(
    'models/sugarcane_phase2_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

print("\n--- Phase 2: Fine-Tuning Last Layers ---\n")
history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs_phase2,
    callbacks=[checkpoint_ft, early_stop],
    class_weight=class_weights
)

# --- 8. Save Final Model ---
model.save('models/sugarcane_disease_model_final.h5')
print("\n✅ Training complete. Best models saved in 'models/' folder")
