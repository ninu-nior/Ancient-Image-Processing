import cv2
def adaptive_background_subtraction(image_path):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    
    background = cv2.GaussianBlur(img, (65, 65), 0)  

    text_foreground = cv2.divide(img, background,scale=255)  


    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 44)) 
    enhanced_img = clahe.apply(text_foreground)

    return enhanced_img

# # Load and process the upscaled image
# input_image_path = "upscaled_text2.png"  # Path to your upscaled image
# processed_image = adaptive_background_subtraction(input_image_path)

# # Display the processed image
# plt.imshow(processed_image, cmap='gray')
# plt.axis('off')
# plt.show()

# # Save the result
# cv2.imwrite("cleaned_text_adaptive.png", processed_image)
