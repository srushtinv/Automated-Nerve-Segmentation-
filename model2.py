import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directories for the data
input_dir = 'data'
train_dir = os.path.join(input_dir, 'train')
val_dir = os.path.join(input_dir, 'val')

# Data augmentation for the training dataset
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize the image pixels
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation dataset (no augmentation, only normalization)
val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Flow data from the directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Resize images to 224x224 for compatibility
    batch_size=32,
    class_mode='binary'  # Binary classification (affected vs. not affected)
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Custom CNN Model Architecture
def create_model(input_shape=(224, 224, 3)):
    model = models.Sequential()

    # Block 1: Convolutional Layer + Max Pooling + Dropout
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 2: Convolutional Layer + Max Pooling + Dropout
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 3: Convolutional Layer + Max Pooling + Dropout
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Flatten the output to feed it into fully connected layers
    model.add(layers.Flatten())

    # Fully Connected Layer
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))

    # Output Layer (binary classification: affected vs not affected)
    model.add(layers.Dense(1, activation='sigmoid'))  # 1 node, sigmoid activation

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',  # Binary classification
                  metrics=['accuracy'])

    return model

# Create the model
model = create_model()

# Display the model summary
model.summary()

# Train the model using the train and validation data generators
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=1,  # You can adjust the number of epochs
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size
)

# Save the trained model
model.save('ultrasound_classifier.h5')

# Optionally: Plot training & validation accuracy/loss
import matplotlib.pyplot as plt

# Plot training & validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training & validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
