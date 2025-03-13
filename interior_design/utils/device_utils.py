"""
Utility functions for device management in ML models.
"""

import torch
import os
import numpy as np
from typing import Union, Tuple, Optional

def get_device() -> torch.device:
    """
    Gets the appropriate device for running ML models.
    
    Returns:
        torch.device: CUDA device if available, otherwise CPU
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_memory_info() -> dict:
    """
    Gets information about available memory on the current device.
    
    Returns:
        dict: Dictionary containing memory information
    """
    if torch.cuda.is_available():
        try:
            # Get CUDA memory usage
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            
            return {
                "device": "cuda",
                "total_memory_GB": t / (1024**3),
                "reserved_memory_GB": r / (1024**3),
                "allocated_memory_GB": a / (1024**3),
                "free_memory_GB": (t - r) / (1024**3)
            }
        except Exception:
            pass
    
    # Return CPU memory info as fallback
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "device": "cpu",
            "total_memory_GB": mem.total / (1024**3),
            "available_memory_GB": mem.available / (1024**3),
            "used_memory_GB": mem.used / (1024**3),
            "free_memory_GB": mem.free / (1024**3)
        }
    except ImportError:
        return {"device": "cpu", "memory_info": "unavailable"}

def setup_device_settings(use_half_precision: bool = True) -> Tuple[torch.device, torch.dtype]:
    """
    Sets up device settings for optimal performance.
    
    Args:
        use_half_precision: Whether to use half precision (float16) on CUDA devices
        
    Returns:
        Tuple[torch.device, torch.dtype]: Device and data type to use
    """
    device = get_device()
    
    # Use half precision only if CUDA is available and requested
    if device.type == "cuda" and use_half_precision:
        dtype = torch.float16
    else:
        dtype = torch.float32
        
    return device, dtype

def is_gpu_suitable_for_model(min_vram_gb: float = 4.0) -> bool:
    """
    Checks if the available GPU has sufficient VRAM for a model.
    
    Args:
        min_vram_gb: Minimum required VRAM in GB
        
    Returns:
        bool: Whether the GPU is suitable
    """
    if not torch.cuda.is_available():
        return False
    
    try:
        # Get total memory in bytes and convert to GB
        memory_info = get_memory_info()
        total_vram_gb = memory_info.get("total_memory_GB", 0)
        return total_vram_gb >= min_vram_gb
    except Exception:
        # In case of error, assume not suitable
        return False

def auto_select_device(model_complexity: str = "medium") -> torch.device:
    """
    Automatically selects the best device based on model complexity.
    
    Args:
        model_complexity: Complexity of the model ("low", "medium", "high")
        
    Returns:
        torch.device: Appropriate device
    """
    # Define minimum VRAM requirements based on complexity
    vram_requirements = {
        "low": 2.0,  # 2GB VRAM
        "medium": 4.0,  # 4GB VRAM
        "high": 8.0  # 8GB VRAM
    }
    
    # Get the requirement for the given complexity
    min_vram = vram_requirements.get(model_complexity, 4.0)
    
    # Check if CUDA is available and meets requirements
    if torch.cuda.is_available() and is_gpu_suitable_for_model(min_vram):
        return torch.device("cuda")
    else:
        return torch.device("cpu") 