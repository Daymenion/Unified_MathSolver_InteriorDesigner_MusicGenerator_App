import torch
import os

def get_device():
    """
    Get the appropriate device (CUDA or CPU) for running models.
    Returns torch.device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        print("CUDA is not available. Using CPU instead. This will be much slower.")
        return torch.device("cpu")

def ensure_directory_exists(dir_path):
    """
    Make sure a directory exists, create it if it doesn't.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")
