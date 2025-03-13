"""
Test script for the style transformer functionality.

This script can be used to validate that the style transformer works correctly
with all the fallback mechanisms we've implemented.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import time
import numpy as np
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("style_transform_test")

# Ensure we can import from the parent directory
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import our style transformer
try:
    from interior_design.style_transformer import StyleTransformer
    logger.info("Successfully imported StyleTransformer")
except ImportError as e:
    logger.error(f"Error importing StyleTransformer: {str(e)}")
    sys.exit(1)

def test_style_transformer(image_path=None, style="Modern", room_type=None, output_dir="output", use_local_sd=True):
    """
    Test the StyleTransformer with a sample image.
    
    Args:
        image_path: Path to sample image (if None, will create a test image)
        style: Style to transform to
        room_type: Room type (if None, will be auto-detected)
        output_dir: Directory to save output images
        use_local_sd: Whether to use local Stable Diffusion or API
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the style transformer
    transformer = StyleTransformer()
    logger.info(f"Initialized StyleTransformer with device: {transformer.device}")
    
    # Create or load sample image
    if image_path and os.path.exists(image_path):
        try:
            image = Image.open(image_path)
            logger.info(f"Loaded test image from: {image_path}")
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            logger.info("Creating a synthetic test image instead")
            image = create_test_image()
    else:
        logger.info("Creating a synthetic test image")
        image = create_test_image()
    
    # Save original image
    original_path = os.path.join(output_dir, "original.png")
    image.save(original_path)
    logger.info(f"Saved original image to: {original_path}")
    
    # Test all components individually
    logger.info("Testing individual components...")
    
    # 1. Test HWC3 and resize functions
    try:
        # Initialize MLSD detector to access the utility functions
        transformer._init_mlsd_detector()
        
        # Test HWC3 function
        img_array = np.array(image)
        hwc3_result = transformer.HWC3(img_array)
        logger.info(f"HWC3 test passed: input shape {img_array.shape}, output shape {hwc3_result.shape}")
        
        # Test resize function
        resized = transformer.resize_image(hwc3_result, 512)
        logger.info(f"Resize test passed: output shape {resized.shape}")
        
        # Save resized image
        resized_path = os.path.join(output_dir, "resized.png")
        Image.fromarray(resized).save(resized_path)
        logger.info(f"Saved resized image to: {resized_path}")
    except Exception as e:
        logger.error(f"Error testing utility functions: {str(e)}")
    
    # 2. Test MLSD detector
    try:
        if transformer.mlsd_detector:
            img_array = np.array(image)
            detected_map = transformer.mlsd_detector(transformer.resize_image(transformer.HWC3(img_array), 512), 0.1, 0.1)
            detected_path = os.path.join(output_dir, "detected_lines.png")
            Image.fromarray(detected_map).save(detected_path)
            logger.info(f"MLSD detector test passed. Saved detected lines to: {detected_path}")
    except Exception as e:
        logger.error(f"Error testing MLSD detector: {str(e)}")
    
    # 3. Test the full transformation pipeline
    logger.info(f"Testing style transformation to {style} style using {'local SD' if use_local_sd else 'Hugging Face API'}...")
    try:
        start_time = time.time()
        result = transformer.transform_style(
            image, 
            style=style, 
            room_type=room_type, 
            use_local_sd=use_local_sd,
            strength=0.7,
            num_inference_steps=20
        )
        elapsed = time.time() - start_time
        
        if result.get("success"):
            output_path = os.path.join(output_dir, f"transformed_{style.lower()}.png")
            result["transformed_image"].save(output_path)
            logger.info(f"Style transformation successful! Saved to: {output_path}")
            logger.info(f"Transformation took {elapsed:.2f} seconds")
            logger.info(f"Method used: {result.get('method', 'unknown')}")
            logger.info(f"Room type: {result.get('room_type', 'unknown')}")
            
            # Save the prompt for reference
            prompt_path = os.path.join(output_dir, f"prompt_{style.lower()}.txt")
            with open(prompt_path, "w") as f:
                f.write(result.get("prompt", "No prompt available"))
            logger.info(f"Saved transformation prompt to: {prompt_path}")
        else:
            logger.error(f"Style transformation failed: {result.get('error', 'Unknown error')}")
    except Exception as e:
        logger.error(f"Error during style transformation: {str(e)}")
    
    logger.info("Test completed!")
    return output_dir

def create_test_image(size=(640, 480)):
    """Create a simple synthetic room image for testing."""
    # Create a blank canvas
    img = Image.new('RGB', size, color=(240, 240, 240))
    
    # Create simple room elements using PIL
    from PIL import ImageDraw
    
    draw = ImageDraw.Draw(img)
    
    # Draw floor
    draw.rectangle([(0, size[1]*2//3), (size[0], size[1])], fill=(180, 150, 100))
    
    # Draw back wall
    draw.rectangle([(0, 0), (size[0], size[1]*2//3)], fill=(220, 220, 220))
    
    # Draw a window
    window_width = size[0] // 4
    window_height = size[1] // 4
    window_left = (size[0] - window_width) // 2
    window_top = size[1] // 6
    draw.rectangle(
        [(window_left, window_top), (window_left + window_width, window_top + window_height)], 
        fill=(135, 206, 235)
    )
    
    # Draw window frame
    draw.rectangle(
        [(window_left, window_top), (window_left + window_width, window_top + window_height)], 
        outline=(100, 100, 100), width=5
    )
    
    # Draw a simple sofa
    sofa_width = size[0] // 2
    sofa_height = size[1] // 5
    sofa_left = (size[0] - sofa_width) // 2
    sofa_top = size[1]*2//3 - sofa_height
    draw.rectangle(
        [(sofa_left, sofa_top), (sofa_left + sofa_width, sofa_top + sofa_height)], 
        fill=(160, 82, 45)
    )
    
    # Draw a coffee table
    table_width = size[0] // 5
    table_height = size[1] // 15
    table_left = (size[0] - table_width) // 2
    table_top = sofa_top + sofa_height + size[1] // 15
    draw.rectangle(
        [(table_left, table_top), (table_left + table_width, table_top + table_height)], 
        fill=(139, 69, 19)
    )
    
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the StyleTransformer")
    parser.add_argument("--image", type=str, help="Path to sample image", default=None)
    parser.add_argument("--style", type=str, help="Style to transform to", default="Modern")
    parser.add_argument("--room", type=str, help="Room type", default=None)
    parser.add_argument("--output", type=str, help="Output directory", default="output")
    parser.add_argument("--api", action="store_true", help="Use Hugging Face API instead of local SD")
    
    args = parser.parse_args()
    
    test_style_transformer(
        image_path=args.image,
        style=args.style,
        room_type=args.room,
        output_dir=args.output,
        use_local_sd=not args.api
    ) 