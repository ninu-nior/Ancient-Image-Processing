import cv2
from image_processing import wavelet_denoising,apply_clahe
from load_esrgan import load_esrgan_model
from esrgan_1 import upscale_with_esrgan
from adaptive_background_subtraction import adaptive_background_subtraction
from savuola import sauvola_background_subtraction
from sharpening_and_morphing import unsharp_masking
from ocr_specific import ocr_specifics1
from esrgan2 import upscale_image

def main_func(input_path):
    input_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    print("converted to grayscale")
    cv2.imwrite('Step_outputs/1_grayscale.png', input_image)
    
    wavelet_denoised=wavelet_denoising(input_image)
    cv2.imwrite('Step_outputs/2_wavelet_denoised.png', wavelet_denoised)
    print("wavelet denoising done")
    
    clahe_enhanced=apply_clahe(wavelet_denoised)    
    cv2.imwrite('Step_outputs/3_clahe_enhanced.png', clahe_enhanced)
    print("CLAHE done")
    
    esrgan_enhanced_path=upscale_image('Step_outputs/3_clahe_enhanced.png','Step_outputs/4_esrgan_enhanced.png')
    # model=load_esrgan_model()    
    # esrgan_enhanced=upscale_with_esrgan(clahe_enhanced,model)
    esrgan_enhanced=cv2.imread(esrgan_enhanced_path)
    print("ESRGAN done")
    # esrgan_enhanced=clahe_enhanced
    
    background_seperated=adaptive_background_subtraction(esrgan_enhanced_path)
    cv2.imwrite('Step_outputs/5_background_seperated.png', background_seperated)
    print("background subtraction done")
    
    savuola_output=sauvola_background_subtraction(background_seperated)
    cv2.imwrite('Step_outputs/6_savuola_output.png', savuola_output)
    print("SAUVOLA done")
    
    unsharp_masking_output=unsharp_masking(savuola_output)
    cv2.imwrite('Step_outputs/7_unsharp_masking_output.png', unsharp_masking_output)
    print("unsharp masking done")
    
    # ocr_specifics= ocr_specifics1(unsharp_masking_output)
    # cv2.imwrite('Step_outputs/8_ocr_specifics.png', ocr_specifics)
    # print("ocr specifics done")
    
    final_op_path=upscale_image('Step_outputs/7_unsharp_masking_output.png',"final_output/output.png")
    
    print("final output saved")
    return final_op_path
    
# main()
main_func("C:/Users/Nehal/Desktop/Ancient_text/input_3.png")