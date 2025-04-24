import tensorflow as tf
import numpy as np
import cv2
import os

# ✅ Load the trained model
model = tf.keras.models.load_model("C:/Users/Nehal/Desktop/Ancient_text/Image_Pipeline/sanskrit_tamil_classifier.h5")


# ✅ Function to preprocess an image
def preprocess_image(image_path):
    """
    Load and preprocess the input image for model prediction.
    - Converts image to grayscale
    - Resizes it to 224x224 (same as training data)
    - Normalizes pixel values to [0, 1]
    - Expands dimensions to match model input shape
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    img = cv2.resize(img, (224, 224))  # Resize to match model input size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=[0, -1])  # Expand dimensions to match (1, 224, 224, 1)
    return img

# ✅ Function to make predictions
def predict_language(image_path):
    """
    Predicts the language of a given manuscript image.
    - Outputs "Sanskrit" or "Tamil" with confidence score.
    """
    image = preprocess_image(image_path)
    prediction = model.predict(image)[0][0]  # Get prediction score
    print(prediction)
    # Convert to class label (Assuming 0 = Sanskrit, 1 = Tamil)
    predicted_class = "Tamil" if prediction > 0.001 else "Sanskrit"
    confidence = prediction if prediction > 0.001 else 1 - prediction

    print(f"Predicted Language: {predicted_class} (Confidence: {confidence:.2f})")
    return predicted_class

# ✅ Example: Predict on a test image
# image_path = "C:/Users/Nehal/Desktop/Ancient_text/tam2.jpg"  # Change this to your test image path
# predict_language(image_path)
