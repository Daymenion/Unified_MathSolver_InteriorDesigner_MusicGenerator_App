# MLSD Line Detection
# From https://github.com/navervision/mlsd
# Apache-2.0 license

import cv2
import numpy as np
import torch
import os
import sys
import logging

# Setup logger
logger = logging.getLogger("annotator.mlsd")

# The original imports that may cause issues - commenting out
# from einops import rearrange
# from .models.mbv2_mlsd_tiny import MobileV2_MLSD_Tiny
# from .models.mbv2_mlsd_large import MobileV2_MLSD_Large
# from .utils import pred_lines

# Try to import from torchvision, if not available use our compatibility layer
try:
    from torchvision.transforms import functional_tensor
    logger.info("Using torchvision.transforms.functional_tensor")
except ImportError:
    try:
        # Try to import our compatibility layer
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from annotator.compat import rgb_to_grayscale, adjust_brightness, adjust_contrast
        logger.info("Using compatibility layer for torchvision.transforms.functional_tensor")
    except ImportError as e:
        logger.warning(f"Neither torchvision.transforms.functional_tensor nor compatibility layer available: {e}")

try:
    from annotator.util import annotator_ckpts_path
except ImportError:
    # Define a fallback path
    annotator_ckpts_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ckpts")
    os.makedirs(annotator_ckpts_path, exist_ok=True)
    logger.warning(f"Using fallback ckpts path: {annotator_ckpts_path}")

remote_model_path = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/mlsd_large_512_fp32.pth"


class MLSDdetector:
    """
    Simplified MLSD detector that uses Canny edge detection.
    This is a fallback implementation when the real MLSD model is not available.
    """
    def __init__(self):
        logger.info("Initializing simplified MLSD detector (Canny edge detection fallback)")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def __call__(self, input_image, thr_v, thr_d):
        """
        Apply simplified line detection using Canny edge detection.
        
        Args:
            input_image: Input image (numpy array)
            thr_v: Value threshold (used for Canny parameters)
            thr_d: Distance threshold (used for Canny parameters)
            
        Returns:
            numpy array: Edge detection result
        """
        assert input_image.ndim == 3
        img = input_image.copy()
        img_output = np.zeros_like(img)
        
        try:
            # Adapt thresholds for Canny
            lower_threshold = int(thr_v * 100)  # Scale to useful Canny threshold
            upper_threshold = int(thr_d * 300)  # Scale to useful Canny threshold
            
            # Ensure thresholds are in valid range
            lower_threshold = max(10, min(lower_threshold, 100))
            upper_threshold = max(lower_threshold + 50, min(upper_threshold, 300))
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray, lower_threshold, upper_threshold)
            
            # Enhance the line features with morphological operations
            kernel = np.ones((3,3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Create a more structured line image with probabilistic Hough Transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, None, 30, 10)
            
            # Draw the detected lines on the output image
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(img_output, (x1, y1), (x2, y2), (255, 255, 255), 2)
            else:
                # If no lines detected, just use the edges as fallback
                img_output[:, :, 0] = edges
                img_output[:, :, 1] = edges
                img_output[:, :, 2] = edges
            
        except Exception as e:
            logger.error(f"Error in simplified MLSD detection: {str(e)}")
            # Return a blank image in case of error
            return np.zeros_like(input_image[:, :, 0])
            
        # Return edge map (single channel)
        return img_output[:, :, 0]
