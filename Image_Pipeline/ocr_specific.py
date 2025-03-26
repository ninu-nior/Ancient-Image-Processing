import cv2
import numpy as np

# Load the image
img = cv2.imread("final_morphed.png", cv2.IMREAD_GRAYSCALE)
def ocr_specifics1(img):
# Apply a morphological closing operation to reconnect strokes
    kernel = np.ones((2, 1), np.uint8)  # Small kernel to avoid over-thickening
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
    smoothed = cv2.bilateralFilter(closed, d=9, sigmaColor=75, sigmaSpace=75)
    return smoothed

# Optional: Adaptive Threshold to enhance contrast
# enhanced = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                 cv2.THRESH_BINARY, 11, 2)

# Save the improved image
# cv2.imwrite("improved_output.png", smoothed)  # Change to `enhanced` if using thresholding
