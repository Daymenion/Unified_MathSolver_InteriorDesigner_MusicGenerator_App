"""
Interior Design AI module.

This module provides functionality for transforming interior design styles using AI.
"""

import os
import logging
from pathlib import Path

# Setup logging
logger = logging.getLogger("interior_design")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Define base path
BASE_PATH = Path(__file__).parent
ANNOTATOR_PATH = BASE_PATH / "annotator"

# Add initialize function
def initialize():
    """Initialize the interior design module."""
    # Create necessary directories
    os.makedirs(ANNOTATOR_PATH / "ckpts", exist_ok=True)
    os.makedirs(Path("data/outputs/interior_design"), exist_ok=True)
    
    # Log initialization
    logger.info("Interior Design module initialized")
    
    # Return True to indicate successful initialization
    return True

# Import main components for easy access
try:
    from .style_transformer import StyleTransformer
    __all__ = ['StyleTransformer', 'initialize']
    logger.info("Successfully imported StyleTransformer")
except ImportError as e:
    logger.warning(f"Error importing StyleTransformer: {str(e)}")
    __all__ = ['initialize'] 