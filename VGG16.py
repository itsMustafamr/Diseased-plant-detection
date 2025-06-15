 # -*- coding: utf-8 -*-
"""
Leaf Health Classification: Training and Testing

Created on: [Nov. 25, 2024]
@author: rshastri
"""

import os
# -*- coding: utf-8 -*-
"""
Leaf Health Classification: Training and Testing

Created on: [Nov. 25, 2024]
@author: rshastri
"""

import os
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_auc_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt

# Paths
train_data_dir = r"C:\Users\rshastri\Documents\Project\a\b\NPDD\Overall\Train"
test_data_dir = r"C:\Users\rshastri\Documents\Project\a - Copy\b\NPDD\Test"
model_path = 'leaf_health_model_overall.keras'

# Path to save confusion matrix plot
save_path = r"C:\Users\rshastri\Documents\Project\confusion_matrix_VGG_500_dpi.png"

# Image settings
img_size = (224, 224)  # VGG16 expects images of size 224x224
batch_size = 32

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    channel_shift_range=0.1
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Training data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

# Testing data
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False  # Do not shuffle to match predictions with true labels
)

# CONTROL: Choose to train or test
train_model = False  # Set to True for training, False for testing

if train_model:
    # Load the VGG16 model without the top classification layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers for binary classification
    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)  # Binary classification output

    # Define the model
    model = Model(inputs=base_model.input, outputs=outputs)

    # Compile the model with a reduced learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Reduced learning rate
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    print("Starting model training...")
    model.fit(
        train_generator,
        epochs=50  # Adjust epochs as needed
    )

    # Save the trained model
    model.save(model_path)
    print("Model trained and saved.")
else:
    # Load the pre-trained model
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please train the model first.")

    model = load_model(model_path)

    # Evaluate model on test data
    print("\nEvaluating model on test data...")
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Get predictions and true labels
    test_generator.reset()  # Reset the generator to start from the first batch
    predictions = (model.predict(test_generator) > 0.5).astype("int32")
    true_labels = test_generator.classes

    # Evaluate metrics
    f1 = f1_score(true_labels, predictions, average='binary', pos_label=0)  # Diseased = positive (pos_label=0)
    precision = precision_score(true_labels, predictions, pos_label=0)
    recall = recall_score(true_labels, predictions, pos_label=0)
    auc_pr = roc_auc_score(true_labels, model.predict(test_generator))  # AUC-PR uses probabilities

    cm = confusion_matrix(true_labels, predictions)
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp)
    g_mean = np.sqrt(recall * specificity)
    balanced_accuracy = (recall + specificity) / 2

    print("\nF1 Score: {:.4f}".format(f1))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("AUC-PR: {:.4f}".format(auc_pr))
    print("Specificity: {:.4f}".format(specificity))
    print("G-Mean: {:.4f}".format(g_mean))
    print("Balanced Accuracy: {:.4f}".format(balanced_accuracy))

    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Diseased", "Healthy"], yticklabels=["Diseased", "Healthy"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # Save confusion matrix plot
    plt.savefig(save_path, dpi=500)  # Save the figure at 500 DPI
    print(f"Confusion matrix saved to: {save_path}")

    # Show confusion matrix plot
    plt.show()
