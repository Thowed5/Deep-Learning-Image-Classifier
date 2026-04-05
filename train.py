import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 5 # Reduced for faster execution in sandbox
NUM_CLASSES = 10 # Example: assuming 10 classes

# --- 1. Data Preprocessing ---
# For demonstration, we'll simulate data. In a real scenario, you'd load from disk.
# Let's create dummy data directories and files for demonstration purposes.
import os
import numpy as np

def create_dummy_data(base_dir, num_classes, samples_per_class, img_height, img_width):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    for i in range(num_classes):
        class_dir = os.path.join(base_dir, f'class_{i}')
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        for j in range(samples_per_class):
            dummy_image = np.random.randint(0, 256, (img_height, img_width, 3), dtype=np.uint8)
            # In a real scenario, you'd save actual image files (e.g., using PIL)
            # For this dummy, we just create a placeholder file
            with open(os.path.join(class_dir, f'image_{j}.png'), 'wb') as f:
                f.write(b'dummy_image_data')

# Create dummy training and validation data
train_dir = '/home/ubuntu/Deep-Learning-Image-Classifier/data/train'
val_dir = '/home/ubuntu/Deep-Learning-Image-Classifier/data/validation'
create_dummy_data(train_dir, NUM_CLASSES, 20, IMG_HEIGHT, IMG_WIDTH) # 20 samples per class
create_dummy_data(val_dir, NUM_CLASSES, 5, IMG_HEIGHT, IMG_WIDTH)   # 5 samples per class


train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.2) # This split is for flow_from_directory

# Load training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training')

# Load validation data
validation_generator = train_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation')

# --- 2. Model Definition ---
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# --- 3. Model Compilation ---
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 4. Model Training ---
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# --- 5. Save the Model ---
model.save('/home/ubuntu/Deep-Learning-Image-Classifier/models/image_classifier_model.h5')
print("Model training complete and saved to models/image_classifier_model.h5")

# Optional: Plot training history
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
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
plt.savefig('/home/ubuntu/Deep-Learning-Image-Classifier/training_history.png')
print("Training history plot saved to training_history.png")

# Create requirements.txt
with open('/home/ubuntu/Deep-Learning-Image-Classifier/requirements.txt', 'w') as f:
    f.write('tensorflow==2.10.0\n') # Specify a common version
    f.write('numpy\n')
    f.write('matplotlib\n')
print("requirements.txt created.")
