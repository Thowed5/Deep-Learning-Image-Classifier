import tensorflow as tf
from tensorflow.keras import layers, models

def build_simple_cnn(input_shape, num_classes):
    """
    Builds a simple Convolutional Neural Network (CNN) for image classification.

    Args:
        input_shape (tuple): The shape of the input images (height, width, channels).
        num_classes (int): The number of output classes for classification.

    Returns:
        tf.keras.Model: A compiled Keras model.
    """
    model = models.Sequential([
        # Convolutional Layer 1
        layers.Conv2D(32, (3, 3), activation=\'relu\', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Convolutional Layer 2
        layers.Conv2D(64, (3, 3), activation=\'relu\'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Convolutional Layer 3
        layers.Conv2D(128, (3, 3), activation=\'relu\'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Flatten the output for the dense layers
        layers.Flatten(),

        # Dense Layer 1
        layers.Dense(256, activation=\'relu\'),
        layers.Dropout(0.5),

        # Output Layer
        layers.Dense(num_classes, activation=\'softmax\')
    ])

    # Compile the model
    model.compile(
        optimizer=\'adam\',
        loss=\'categorical_crossentropy\',
        metrics=[\'accuracy\']
    )
    return model


def preprocess_image(image_path, target_size=(128, 128)):
    """
    Loads and preprocesses an image for model inference.

    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Desired size of the image (height, width).

    Returns:
        numpy.ndarray: Preprocessed image array, ready for model input.
    """
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch dimension
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array


if __name__ == "__main__":
    # Example usage:
    input_shape = (128, 128, 3)  # Example input shape for color images
    num_classes = 10  # Example: 10 different categories

    # Build the model
    model = build_simple_cnn(input_shape, num_classes)
    model.summary()

    print("\nModel architecture defined successfully.")
    print("You can now train this model with your dataset.")

    # Example of creating a dummy image file for testing preprocess_image
    dummy_image_path = "dummy_test_image.png"
    try:
        from PIL import Image
        import numpy as np
        dummy_img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
        dummy_img.save(dummy_image_path)
        print(f"Dummy image saved to {dummy_image_path}")

        # Preprocess the dummy image
        processed_img = preprocess_image(dummy_image_path)
        print(f"Processed image shape: {processed_img.shape}")
        os.remove(dummy_image_path)
        print(f"Dummy image {dummy_image_path} removed.")
    except ImportError:
        print("Pillow not installed. Cannot create dummy image for preprocessing test.")
        print("Please install Pillow: pip install Pillow")
    except Exception as e:
        print(f"An error occurred during dummy image processing: {e}")

