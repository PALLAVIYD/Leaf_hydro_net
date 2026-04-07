import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report

# ===================================================================
# Module 1: ResNet50-based Plant Disease Classifier
# Represents the 38-class plant disease classifier mentioned in Sec 8.3
# ===================================================================

# Hyperparameters
IMG_SIZE = 224 # ResNet50 expected input size
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

def preprocess_image(image, label):
    """Resizes and normalizes the image for ResNet50."""
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    # ResNet50 expects inputs preprocessed via the built-in function
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, label

def load_data():
    """Loads the PlantVillage dataset from TFDS."""
    print("Loading PlantVillage dataset...")
    (ds_train, ds_val), ds_info = tfds.load(
        'plant_village',
        split=['train[:80%]', 'train[80%:]'],
        as_supervised=True,
        with_info=True
    )
    
    NUM_CLASSES = ds_info.features['label'].num_classes
    CLASS_NAMES = ds_info.features['label'].names
    print(f"Loaded {NUM_CLASSES} classes.")

    ds_train = ds_train.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    ds_val = ds_val.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val, NUM_CLASSES, CLASS_NAMES

def build_resnet_model(num_classes):
    """Builds the ResNet50 transfer learning model."""
    print("Building ResNet50 model...")
    # Load the base model with pre-trained ImageNet weights
    base_model = tf.keras.applications.ResNet50(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model
    base_model.trainable = False

    # Add custom classification head
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)  # Regularization
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, base_model

def plot_training_history(history):
    """Plots training and validation accuracy/loss."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('training_curves.png')
    print("Saved training_curves.png")

def evaluate_and_plot_confusion_matrix(model, ds_val, class_names):
    """Evaluates the model and plots the confusion matrix."""
    print("Evaluating model and generating confusion matrix...")
    y_true = []
    y_pred = []

    for images, labels in ds_val:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues", xticklabels=False, yticklabels=False)
    plt.title("ResNet50 Confusion Matrix (38 Classes)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig('confusion_matrix.png')
    print("Saved confusion_matrix.png")

if __name__ == "__main__":
    # Ensure matplotlib doesn't try to open windows
    import matplotlib
    matplotlib.use('Agg')
    
    ds_train, ds_val, num_classes, class_names = load_data()
    model, base_model = build_resnet_model(num_classes)
    
    print("\n--- Starting Initial Training (Frozen Head) ---")
    history = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS)
    
    print("\n--- Starting Fine-Tuning ---")
    # Unfreeze the top layers of the model
    base_model.trainable = True
    # Freeze all layers except the last 15
    for layer in base_model.layers[:-15]:
        layer.trainable = False
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower learning rate
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fine-tune for additional epochs
    fine_tune_epochs = 5
    total_epochs = EPOCHS + fine_tune_epochs
    history_fine = model.fit(ds_train, validation_data=ds_val, epochs=total_epochs, initial_epoch=history.epoch[-1])
    
    # Save visualizations
    plot_training_history(history_fine)
    evaluate_and_plot_confusion_matrix(model, ds_val, class_names)
    
    # Save the final model
    model.save('resnet_boea_disease_classifier.h5')
    print("Model saved to resnet_boea_disease_classifier.h5")
