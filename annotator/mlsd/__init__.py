# MLSD Line Detection
# From https://github.com/navervision/mlsd
# Apache-2.0 license

import cv2
import numpy as np
import torch
import os

from einops import rearrange
from .models.mbv2_mlsd_tiny import MobileV2_MLSD_Tiny
from .models.mbv2_mlsd_large import MobileV2_MLSD_Large
from .utils import pred_lines

from annotator.util import annotator_ckpts_path

remote_model_path = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/mlsd_large_512_fp32.pth"


class MLSDdetector:
    """
    Simplified MLSD detector that uses Canny edge detection.
    This is a fallback implementation when the real MLSD model is not available.
    """
    def __init__(self):
        print("Initializing simplified MLSD detector (Canny edge detection fallback)")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def __call__(self, input_image, thr_v, thr_d):
        """
        Apply simplified line detection using Canny edge detection.
        
        Args:
            input_image: Input image (numpy array)
            thr_v: Value threshold (not used in this implementation)
            thr_d: Distance threshold (not used in this implementation)
            
        Returns:
            numpy array: Edge detection result
        """
        assert input_image.ndim == 3
        img = input_image
        img_output = np.zeros_like(img)
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Create output image with detected edges
            img_output[:, :, 0] = edges
            img_output[:, :, 1] = edges
            img_output[:, :, 2] = edges
            
        except Exception as e:
            print(f"Error in simplified MLSD detection: {str(e)}")
            
        return img_output[:, :, 0]
