"""
Compatibility module to provide fallbacks for missing torchvision functions.

This is mainly to provide fallbacks for torchvision.transforms.functional_tensor
which is needed by some of the annotator modules.
"""

import torch
import numpy as np
import logging

logger = logging.getLogger("annotator.compat")

def _is_tensor_a_torch_image(x):
    """
    Return whether the tensor is a torch image or not (PyTorch reimplementation).
    """
    return x.ndim >= 2

def _blend(img1, img2, ratio):
    """
    Blend between two images/batches of images (PyTorch reimplementation).
    """
    return img1.lerp(img2, ratio)

def rgb_to_grayscale(img, num_output_channels=1):
    """
    Convert RGB image to grayscale version (PyTorch reimplementation).
    
    Args:
        img (Tensor): RGB Image to be converted to grayscale.
        num_output_channels (int): number of channels of the output image. Value can be 1 or 3. Default, 1.
    Returns:
        Tensor: Grayscale version of the image.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError(f"tensor is not a torch image. Got {type(img)}")

    if img.shape[-3] != 3:
        raise TypeError(f"Input image tensor should have 3 channels, but got {img.shape[-3]} channels")

    # Apply the conversion formula for RGB to grayscale from ITU-R BT.601 standard:
    # Y = 0.299 R + 0.587 G + 0.114 B
    r, g, b = img.unbind(dim=-3)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gray = gray.unsqueeze(dim=-3)

    if num_output_channels == 1:
        return gray
    elif num_output_channels == 3:
        return gray.expand_as(img)
    else:
        raise ValueError(f"num_output_channels should be 1 or 3, got {num_output_channels}")

def adjust_brightness(img, brightness_factor):
    """
    Adjust brightness of an image (PyTorch reimplementation).
    
    Args:
        img (Tensor): Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.
    Returns:
        Tensor: Brightness adjusted image.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError(f"tensor is not a torch image. Got {type(img)}")
    
    if brightness_factor < 0:
        raise ValueError(f"brightness_factor ({brightness_factor}) is not non-negative.")

    if img.dtype == torch.uint8:
        # Convert to float to avoid clipping issues
        img = img.float()
        img = img.mul(brightness_factor).clamp(0, 255).byte()
    else:
        img = img.mul(brightness_factor).clamp(0, 1)
    
    return img

def adjust_contrast(img, contrast_factor):
    """
    Adjust contrast of an image (PyTorch reimplementation).
    
    Args:
        img (Tensor): Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non-negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    Returns:
        Tensor: Contrast adjusted image.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError(f"tensor is not a torch image. Got {type(img)}")
    
    if contrast_factor < 0:
        raise ValueError(f"contrast_factor ({contrast_factor}) is not non-negative.")

    dtype = img.dtype
    if dtype == torch.uint8:
        img = img.float()
    
    mean = torch.mean(rgb_to_grayscale(img).to(img.dtype))

    result = mean + (img - mean) * contrast_factor
    
    if dtype == torch.uint8:
        return result.clamp(0, 255).to(dtype)
    else:
        return result.clamp(0, 1).to(dtype)

# Export all functions to simulate functional_tensor module
__all__ = ['rgb_to_grayscale', 'adjust_brightness', 'adjust_contrast', '_blend']

# Log that we're using the compatibility module
logger.info("Using compatibility fallbacks for torchvision.transforms.functional_tensor") 