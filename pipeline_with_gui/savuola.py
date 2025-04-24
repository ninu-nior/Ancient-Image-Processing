import cv2
import numpy as np
#import matplotlib.pyplot as plt
from skimage.filters import threshold_sauvola

def sauvola_background_subtraction(img, window_size=35, k=0.10):
    # Load the image in grayscale
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Sauvola thresholding
    thresh_sauvola = threshold_sauvola(img, window_size=window_size, k=k)
    
    # Binarize image: Set pixels below the threshold to 0 (background) and others to 255 (text)
    binary_image = (img > thresh_sauvola).astype(np.uint8) * 255

    return binary_image

# # Load and process the upscaled image
# input_image_path = "cleaned_text_adaptive.png"  # Path to your upscaled image
# processed_image = sauvola_background_subtraction(input_image_path, window_size=35, k=0.15)  # Adjust parameters as needed

# Display the processed image
# plt.imshow(processed_image, cmap='gray')
# plt.axis('off')
# plt.show()

# # Save the result
# cv2.imwrite("cleaned_text_sauvola.png", processed_image)
