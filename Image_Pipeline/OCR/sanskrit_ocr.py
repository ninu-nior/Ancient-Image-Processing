# import pytesseract
# from PIL import Image
# import cv2
# import os

# # Set paths
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

# # Process Hindi text
# image_path_hindi = r"C:\Users\Nehal\Desktop\Ancient_text\Image_Pipeline/final_output/output.png"
# image_hindi = cv2.imread(image_path_hindi)
# text_hindi = pytesseract.image_to_string(image_hindi, lang="hin")
# print("Extracted Hindi Text:\n", text_hindi)

# # Process Sanskrit text
# image_path_sanskrit = r"C:\Users\Nehal\Desktop\Ancient_text\sanskrit_text_image.png"
# image_sanskrit = cv2.imread(image_path_sanskrit)
# text_sanskrit = pytesseract.image_to_string(image_sanskrit, lang="san")
# print("Extracted Sanskrit Text:\n", text_sanskrit)
import pytesseract
import cv2
# Set the Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Set the TESSDATA_PREFIX
import os
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"
image = cv2.imread("C:/Users/Nehal/Desktop/Ancient_text/Image_Pipeline/final_output/output.png")
# Now run OCR
text = pytesseract.image_to_string(image, lang="san")

print(text)