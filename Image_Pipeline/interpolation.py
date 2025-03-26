

import cv2
import numpy as np

# Load the image
img = cv2.imread("C:/Users/Nehal/Desktop/Ancient Text Processing/new_pipeline/cleaned_text_sauvola.png")

# Define the scaling factor (e.g., 2x)
scale_factor = 2

# Apply Bicubic Interpolation
bicubic_upscaled = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

# Apply Lanczos Interpolation
lanczos_upscaled = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LANCZOS4)

# Save the results
cv2.imwrite("bicubic_upscaled.png", bicubic_upscaled)
cv2.imwrite("lanczos_upscaled.png", lanczos_upscaled)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(bicubic_upscaled)
plt.title("Bicubic Upscaled")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(lanczos_upscaled)
plt.title("Lanczos Upscaled")
plt.axis("off")

plt.show()
