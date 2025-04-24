

import cv2
import torch
import numpy as np

# ESRGAN Inference
def upscale_with_esrgan(img, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ",device)
    # torch.backends.cudnn.benchmark = True
    # # Load the model
    # model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    # model.load_state_dict(torch.load(model_path), strict=True)
    # model.eval()
    # model = model.to(device)

    # Preprocess the image
    # img = cv2.imread(image_path, cv2.IMREAD_COLOR)
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

# image = cv2.imread('upscaled_text.png')
# plt.imshow(image)
# plt.axis('off')  # Hide axes
# plt.show()