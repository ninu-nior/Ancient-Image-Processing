import cv2
import pytesseract
from langdetect import detect, DetectorFactory

# Ensure consistent language detection results
DetectorFactory.seed = 0  

# Set the path to Tesseract-OCR if needed (for Windows users)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_image_language(image_path):
    # Load image
    image = cv2.imread(image_path)
    
    # Convert to grayscale for better OCR accuracy
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Extract text from image using Tesseract
    extracted_text = pytesseract.image_to_string(gray)
    
    if not extracted_text.strip():
        return "No text detected"
    
    # Detect language
    detected_lang = detect(extracted_text)
    
    return detected_lang, extracted_text

# Example usage
image_path = "C:/Users/Nehal/Desktop/Ancient_text/Image_Pipeline/final_output/output.png"
language, text = detect_image_language(image_path)
print(f"Detected Language: {language}")
print(f"Extracted Text: {text}")
