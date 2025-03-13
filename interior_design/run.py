#!/usr/bin/env python
"""
Command-line interface for Interior Design style transformation.

This script provides a simple way to run the style transformation from the command line.
It uses all the fallback mechanisms we've implemented to ensure reliable operation.
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from PIL import Image
import glob
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("interior_design_cli")

# Ensure we can import from the parent directory
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    from interior_design import StyleTransformer, initialize
    logger.info("Successfully imported Interior Design modules")
    initialize()
except ImportError as e:
    logger.error(f"Error importing StyleTransformer: {str(e)}")
    try:
        # Try direct import as fallback
        from style_transformer import StyleTransformer
        logger.info("Successfully imported StyleTransformer using direct import")
    except ImportError as e2:
        logger.error(f"Failed to import StyleTransformer: {str(e2)}")
        sys.exit(1)

def get_supported_styles():
    """Get the list of supported styles."""
    try:
        # Create a temporary StyleTransformer to access the styles
        transformer = StyleTransformer()
        return transformer.styles
    except Exception as e:
        logger.warning(f"Error getting supported styles: {str(e)}")
        # Return a default list if we can't get the actual supported styles
        return [
            "Modern", "Contemporary", "Minimalist", "Light Luxury", "Industrial", 
            "Vintage", "Rustic", "Scandinavian", "Art Deco", "Gothic", "Japanese"
        ]

def get_supported_rooms():
    """Get the list of supported room types."""
    try:
        # Create a temporary StyleTransformer to access the room types
        transformer = StyleTransformer()
        return transformer.rooms
    except Exception as e:
        logger.warning(f"Error getting supported room types: {str(e)}")
        # Return a default list if we can't get the actual supported room types
        return [
            "Living Room", "Bedroom", "Bathroom", "Kitchen", "Dining Room", 
            "Office", "Study", "Patio", "Balcony"
        ]

def find_sample_images(sample_dir=None):
    """Find sample images in the sample directory."""
    if not sample_dir:
        # Try to find sample directory
        possible_paths = [
            Path(current_dir) / "samples",
            Path(current_dir) / "../samples",
            Path(current_dir) / "../data/samples",
            Path(current_dir) / "data/samples",
        ]
        
        for path in possible_paths:
            if path.exists() and path.is_dir():
                sample_dir = path
                break
    
    if not sample_dir or not os.path.exists(sample_dir):
        logger.warning("Sample directory not found")
        return []
    
    # Find all image files
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
        image_paths.extend(glob.glob(os.path.join(sample_dir, ext)))
    
    return image_paths

def transform_room(args):
    """Transform a room based on command-line arguments."""
    # Initialize the style transformer
    transformer = StyleTransformer()
    logger.info(f"Initialized StyleTransformer on device: {transformer.device}")
    
    # Get the input image
    if args.image and os.path.exists(args.image):
        image_path = args.image
    else:
        # If no image specified or image doesn't exist, find sample images
        sample_images = find_sample_images(args.sample_dir)
        if sample_images:
            image_path = random.choice(sample_images)
            logger.info(f"Using random sample image: {image_path}")
        else:
            logger.error("No input image specified and no sample images found")
            return
    
    try:
        # Load the image
        image = Image.open(image_path)
        logger.info(f"Loaded image: {image_path}")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save original image
        orig_path = os.path.join(args.output_dir, "original.png")
        image.save(orig_path)
        logger.info(f"Saved original image to: {orig_path}")
        
        # Time the transformation
        start_time = time.time()
        
        # Transform the style
        result = transformer.transform_style(
            image, 
            style=args.style,
            room_type=args.room,
            use_local_sd=not args.api,
            strength=args.strength,
            num_inference_steps=args.steps
        )
        
        elapsed = time.time() - start_time
        
        if result.get("success"):
            # Save the transformed image
            output_filename = f"transformed_{args.style.lower()}.png"
            if args.output:
                output_filename = args.output
            
            output_path = os.path.join(args.output_dir, output_filename)
            result["transformed_image"].save(output_path)
            
            logger.info(f"Style transformation successful! Saved to: {output_path}")
            logger.info(f"Transformation took {elapsed:.2f} seconds")
            logger.info(f"Method used: {result.get('method', 'unknown')}")
            
            # Save the prompt for reference
            prompt_path = os.path.join(args.output_dir, f"prompt_{args.style.lower()}.txt")
            with open(prompt_path, "w") as f:
                f.write(result.get("prompt", "No prompt available"))
            logger.info(f"Saved transformation prompt to: {prompt_path}")
            
            return output_path
        else:
            logger.error(f"Style transformation failed: {result.get('error', 'Unknown error')}")
    except Exception as e:
        logger.error(f"Error during transformation: {str(e)}")

def main():
    """Main entry point for the command-line interface."""
    # Get supported styles and rooms
    styles = get_supported_styles()
    rooms = get_supported_rooms()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Interior Design Style Transformer CLI")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--style", type=str, choices=styles, default="Modern", help="Style to transform to")
    parser.add_argument("--room", type=str, choices=rooms, help="Room type (if not provided, will be auto-detected)")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save output files")
    parser.add_argument("--output", type=str, help="Output filename (default: transformed_[style].png)")
    parser.add_argument("--api", action="store_true", help="Use Hugging Face API instead of local SD")
    parser.add_argument("--strength", type=float, default=0.7, help="Transformation strength (0.0 to 1.0)")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--sample-dir", type=str, help="Directory containing sample images")
    
    args = parser.parse_args()
    
    # Transform the room
    transform_room(args)

if __name__ == "__main__":
    main() 