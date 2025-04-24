from main_image_processing import main_func
#from OCR.tamil_ocr import get_tamil
from OCR.sanskrit_ocr import get_sanskrit
from AI_Interpretation.sample import get_response
#from detect import predict_language,preprocess_image
import cv2
input_path="uploads/sanskrit.jpg"
#output_path=main_func(input_path)
#print(output_path)
#detect_lang=predict_language(output_path)
#if detect_lang=="Sanskrit":
data=get_sanskrit("C:/Users/adityOneDrive/Documents/Image_Pipeline/Image_Pipeline/final_output/output.png")
#if detect_lang=="Tamil":
 #   data=get_tamil(output_path)
#print("Language: ",detect_lang)
detect_lang="Sanskrit"
ai_response=get_response(data,detect_lang)