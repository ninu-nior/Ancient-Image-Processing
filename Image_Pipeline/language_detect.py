import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained model
char_model2 = load_model("C:/Users/Nehal/Desktop/Ancient_text/Image_Pipeline/best_model (2).keras")

def classify_manuscript(image_path):
    """
    Classifies words in the manuscript image as Sanskrit or Tamil, draws bounding boxes, 
    and saves the labeled output image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        str: The final classified language of the manuscript.
        dict: Breakdown of votes for each language.
        str: Path to the saved labeled image.
    """
    # Define classes
    classes = ["sanskrit", "tamil"]
    votes = {"sanskrit": 0, "tamil": 0}

    def preprocess_image(image, img_height=64, img_width=192):
        """Preprocess an image for model prediction."""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        image = cv2.resize(image, (img_width, img_height))  # Resize
        image = img_to_array(image) / 255.0  # Normalize
        image = np.expand_dims(image, axis=-1)  # Ensure shape (192, 64, 1)
        image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 192, 64, 1)
        return image

    # Load input image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    output_image = image.copy()  # Copy for visualization
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operations
    kernel = np.ones((5, 15), np.uint8)  # Adjust kernel size if needed
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process each detected word
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if w > 20 and h > 20:  # Ensure significant size
            word_image = image[y:y+h, x:x+w]  # Extract word ROI
            preprocessed_word = preprocess_image(word_image)  # Preprocess image

            # Predict using model
            prediction = char_model2.predict(preprocessed_word)
            predicted_class = np.argmax(prediction)  # 0 = Sanskrit, 1 = Tamil
            language = classes[predicted_class]
            votes[language] += 1

            # Draw bounding box and label
            color = (0, 255, 0) if language == "sanskrit" else (0, 0, 255)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(output_image, language, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Final classification using majority voting
    predicted_page_language = max(votes, key=votes.get)

    # Save the image with bounding boxes
    output_path = image_path.replace(".png", "_labeled.png")
    cv2.imwrite(output_path, output_image)

    return predicted_page_language, votes, output_path

# Example usage
#image_path = "C:/Users/Nehal/Desktop/Ancient_text/Image_Pipeline/final_output/output.png"
#predicted_language, votes, output_image_path = classify_manuscript(image_path)

#print(f"Final Manuscript Classification: {predicted_language}")
#print(f"Votes Breakdown: {votes}")
#print(f"Processed image saved at: {output_image_path}")
