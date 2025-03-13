"""
Annotator package for image annotation (edge detection, MLSD, etc.)
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logger = logging.getLogger("annotator")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Define paths
annotator_path = Path(__file__).parent.absolute()
ckpts_path = annotator_path / "ckpts"

# Create ckpts directory if it doesn't exist
os.makedirs(ckpts_path, exist_ok=True)

# Add annotator dir to path to ensure imports work
if str(annotator_path.parent) not in sys.path:
    sys.path.insert(0, str(annotator_path.parent))

# Check for torchvision and add fallback import mechanism
try:
    import torchvision
    logger.info(f"Torchvision version: {torchvision.__version__}")
except ImportError:
    logger.warning("Torchvision not found. Using fallback mechanisms for annotators.")
    
# Create a helper function to safely import modules
def safe_import(module_path, fallback=None):
    try:
        if '.' in module_path:
            module_name, attr_name = module_path.rsplit('.', 1)
            module = __import__(module_name, fromlist=[attr_name])
            return getattr(module, attr_name)
        else:
            return __import__(module_path)
    except ImportError as e:
        logger.warning(f"Error importing {module_path}: {str(e)}")
        if fallback:
            return fallback
        return None

# Make the package importable 