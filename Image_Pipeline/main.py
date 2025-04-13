from main_image_processing import main_func
from OCR.tamil_ocr import get_tamil
from OCR.sanskrit_ocr import get_sanskrit
from AI_Interpretation.sample import get_response
#from detect import predict_language,preprocess_image
from language_detect import classify_manuscript
import cv2
input_path="C:/Users/Nehal/Desktop/Ancient_text/tamil_3.png"
output_path=main_func(input_path)
print(output_path)
detect_lang=classify_manuscript(output_path)[0]
if detect_lang=="sanskrit":
    data=get_sanskrit(output_path)
if detect_lang=="tamil":
    data=get_tamil(output_path)
print("Language: ",detect_lang)

ai_response=get_response(data,detect_lang)