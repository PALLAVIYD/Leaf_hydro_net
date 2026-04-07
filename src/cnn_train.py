import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report

# ===================================================================
# Module 1 (Updated): Basic CNN Plant Disease Classifier
# Matches the original 3-layer Proof-of-Concept from Untitled6.ipynb
# ===================================================================

# Hyperparameters
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 5

def load_data():
    """Loads the PlantVillage dataset directly bypassing TFDS bug."""
    print("Loading PlantVillage dataset from data/plant_village...")
    
    ds_train = tf.keras.utils.image_dataset_from_directory(
        'data/plant_village',
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )
    
    ds_test = tf.keras.utils.image_dataset_from_directory(
        'data/plant_village',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )
    
    class_names = ds_train.class_names
    NUM_CLASSES = len(class_names)
    print(f"Loaded {NUM_CLASSES} classes.")

    normalization_layer = tf.keras.layers.Rescaling(1./255)
    ds_train = ds_train.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

    ds_test = ds_test.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds_train, ds_test, NUM_CLASSES, class_names

def build_simple_cnn(num_classes):
    """Builds the 3-Layer CNN mapped exactly from the Proof of Concept."""
    print("Building Simple CNN model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    # Ensure matplotlib doesn't hang in terminal
    import matplotlib
    matplotlib.use('Agg')

    ds_train, ds_test, num_classes, class_names = load_data()
    model = build_simple_cnn(num_classes)
    
    print("\n--- Starting Training ---")
    history = model.fit(ds_train, validation_data=ds_test, epochs=EPOCHS)
    
    # Save the simple model
    model.save('models/cnn/simple_cnn_disease_classifier.h5')
    print("Simple CNN Model saved to models/cnn/simple_cnn_disease_classifier.h5")
