import cv2
import numpy as np
#import torch
import RRDBNet_arch as arch
from skimage.restoration import denoise_wavelet

def wavelet_denoising(image):
    denoised = denoise_wavelet(image, method='BayesShrink', mode='soft', wavelet_levels=4,
                               wavelet='db1', rescale_sigma=True)
    denoised = (denoised * 255).astype(np.uint8)
    return denoised

def apply_clahe(image, clip_limit=1.0, tile_grid_size=(4, 4)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image)
    return enhanced

def enhance_text_image(image):
    denoised = wavelet_denoising(image)
    enhanced = apply_clahe(denoised)
    return enhanced

# # Load the grayscale image
# image = cv2.imread('C:/Users/Nehal/Desktop/Ancient Text Processing/sanskrit.png', cv2.IMREAD_GRAYSCALE)
# enhanced_image = enhance_text_image(image)

# # Save the contrast-enhanced image
# cv2.imwrite('enhanced_text.png', enhanced_image)

# ESRGAN Inference
def upscale_with_esrgan(image_path, model_path='C:/Users/Nehal/Desktop/Ancient Text Processing/ESRGAN/models/RRDB_ESRGAN_x4.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    torch.backends.cudnn.benchmark = True
    # Load the model
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    # Preprocess the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0  # Normalize to [0, 1]
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float().unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img).squeeze(0).cpu().numpy()
    
    # Postprocess the output
    output = np.transpose(output, (1, 2, 0))
    output = (output * 255).clip(0, 255).astype(np.uint8)
    

    # Save the upscaled image
    cv2.imwrite('upscaled_text2.png', output)

# Upscale the enhanced image
# upscale_with_esrgan('enhanced_text.png')
# cv2.imshow('Upscaled Image', cv2.imread('upscaled_text.png'))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# image = cv2.imread('upscaled_text.png')
# plt.imshow(image)
# plt.axis('off')  # Hide axes
# plt.show()