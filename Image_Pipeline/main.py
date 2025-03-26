from main_image_processing import main_func
from OCR.tamil_ocr import get_tamil
from AI_Interpretation.sample import get_response
output_path=main_func("C:/Users/Nehal/Desktop/Ancient_text/tam2.jpg")
print(output_path)
data=get_tamil(output_path)
ai_response=get_response(data)