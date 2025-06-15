import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
import os

# Define directories for training and testing data
train_dir = '../archive/train/'
test_dir = '../archive/test/'

# Image size and other parameters
img_height, img_width = 150, 150
batch_size = 32
epochs = 20
input_shape = (img_height, img_width, 3)

# Data augmentation and preprocessing for training
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2]
)

# Only rescale the test images
test_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',  # Binary classification
    shuffle=True
)

# Load and preprocess testing data
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)


# Build an improved CNN model
model = Sequential()

# Block 1
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())

# Block 2
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())

# Block 3
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())

# Block 4
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())

# Flatten and Fully Connected Layers
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])


# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=(train_generator.samples + batch_size - 1) // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=(test_generator.samples + batch_size - 1)// batch_size
)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples + batch_size - 1 // batch_size)
print(f'Test accuracy: {test_acc * 100:.2f}%')

preds = model.predict(test_generator, steps=test_generator.samples + batch_size - 1 // batch_size)

# Convert probabilities to binary predictions
preds_binary = (preds > 0.5).astype(int)

# Get the true labels
true_labels = test_generator.classes

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_labels[:len(preds_binary)], preds_binary)

# Calculate F1 score
f1 = f1_score(true_labels[:len(preds_binary)], preds_binary)

# Print the confusion matrix and F1 score
print("Confusion Matrix:")
print(conf_matrix)

print(f"F1 Score: {f1:.2f}")

import matplotlib.pyplot as plt
import seaborn as sns

# Plot confusion matrix
def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Assuming you have already calculated 'conf_matrix'
plot_confusion_matrix(conf_matrix)

from sklearn.metrics import recall_score, precision_score, roc_auc_score, balanced_accuracy_score

# Calculate Recall
recall = recall_score(true_labels[:len(preds_binary)], preds_binary)

# Calculate Precision
precision = precision_score(true_labels[:len(preds_binary)], preds_binary)

# Calculate AUC-PR
auc_pr = roc_auc_score(true_labels[:len(preds_binary)], preds.flatten())

# Calculate Specificity
# Specificity = TN / (TN + FP)
tn, fp, fn, tp = confusion_matrix(true_labels[:len(preds_binary)], preds_binary).ravel()
specificity = tn / (tn + fp)

# Calculate G-Mean
g_mean = (recall * specificity) ** 0.5

# Calculate Balanced Accuracy
balanced_acc = balanced_accuracy_score(true_labels[:len(preds_binary)], preds_binary)

# Print Metrics
print(f"Recall: {recall:.2f}")
print(f"Precision: {precision:.2f}")
print(f"AUC-PR: {auc_pr:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"G-Mean: {g_mean:.2f}")
print(f"Balanced Accuracy: {balanced_acc:.2f}")


# Save the model
model.save('plant_disease_cnn_model.h5')
