import cv2
import numpy as np

# Read image
# img = cv2.imread("cleaned_text_sauvola.png")

def unsharp_masking(img):
# # Apply Unsharp Masking'
#     img = cv2.imread(img_path)

    gaussian = cv2.GaussianBlur(img, (0, 0), 3)
    sharpened = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

    # cv2.imwrite("final_sharpened.png", sharpened)
    kernel = np.ones((2,2), np.uint8)  
    morph = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel)
    return morph
# cv2.imwrite("final_morphed.png", morph)
