import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
from skimage.restoration import denoise_wavelet
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_sauvola
# Load the model once globally
def load_esrgan_model(model_path='C:/Users/Nehal/Desktop/Ancient Text Processing/ESRGAN/models/RRDB_ESRGAN_x4.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    torch.backends.cudnn.benchmark = True
    # Load the model
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)
    return model

# Load the model once at the start



