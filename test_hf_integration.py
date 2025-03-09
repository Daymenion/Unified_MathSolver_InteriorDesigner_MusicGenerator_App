"""
Test script to verify Hugging Face API integration for image generation.

This script tests both text-to-image and image-to-image capabilities
using the Hugging Face Inference API.
"""

import os
import sys
import logging
from dotenv import load_dotenv
from PIL import Image
from pathlib import Path
from io import BytesIO
from datetime import datetime

# Ensure the project root is in the Python path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Try to import the huggingface_hub library
try:
    from huggingface_hub import InferenceClient
except ImportError:
    print("Error: huggingface_hub not installed. Please install it with 'pip install huggingface_hub'")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('hf_test')

# Load environment variables
load_dotenv()

# Test configuration
TEST_CONFIG = {
    "text_to_image": {
        "model": "stabilityai/stable-diffusion-xl-base-1.0",  # Default model
        "prompt": "A modern living room with minimalist furniture, large windows, and plants",
        "negative_prompt": "poor quality, blurry, distorted",
        "guidance_scale": 7.5,
        "num_inference_steps": 20,  # Lower for faster testing
        "width": 512,
        "height": 512
    },
    "image_to_image": {
        "model": "stabilityai/stable-diffusion-xl-refiner-1.0",  # Updated model for img2img
        "prompt": "Transform this room into a gothic style with dark colors, ornate furniture, dramatic lighting, stained glass windows, medieval inspired decor, high quality, professional interior design",
        "negative_prompt": "poor quality, blurry, distorted, unrealistic, bad architecture, extra walls, extra doors",
        "guidance_scale": 7.5,
        "num_inference_steps": 40,  # More steps for better quality
        "strength": 0.75  # Stronger transformation
    },
    "output_dir": "data/test_outputs"
}


def setup_test_environment():
    """
    Set up the test environment, creating directories if needed.
    """
    # Create output directory if it doesn't exist
    Path(TEST_CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
    
    # Check if the HF token is available
    hf_token = os.environ.get("HUGGING_FACE_API_TOKEN")
    if not hf_token:
        logger.error("HUGGING_FACE_API_TOKEN environment variable not set")
        print("Please set the HUGGING_FACE_API_TOKEN environment variable")
        return False
    
    # Verify that sample input image exists
    sample_image_path = "data/inputs/interior_design/living-room-victorian.png"
    if not os.path.exists(sample_image_path):
        logger.error(f"Sample input image not found at {sample_image_path}")
        print(f"Sample input image not found at {sample_image_path}")
        return False
    
    return True


def test_text_to_image(client):
    """
    Test text-to-image generation using Hugging Face Inference API.
    
    Args:
        client: Hugging Face InferenceClient
        
    Returns:
        bool: True if test passed, False otherwise
    """
    logger.info("Testing text-to-image generation...")
    
    try:
        # Set up the client with no caching
        client.headers["x-use-cache"] = "0"
        
        # Get config values
        config = TEST_CONFIG["text_to_image"]
        
        # Use text-to-image endpoint
        result = client.text_to_image(
            prompt=config["prompt"],
            model=config["model"],
            negative_prompt=config["negative_prompt"],
            guidance_scale=config["guidance_scale"],
            num_inference_steps=config["num_inference_steps"],
            width=config["width"],
            height=config["height"]
        )
        
        # Save the result
        output_path = os.path.join(TEST_CONFIG["output_dir"], "text_to_image_test.jpg")
        result.save(output_path)
        
        logger.info(f"Text-to-image test passed. Result saved to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Text-to-image test failed: {str(e)}")
        return False


def test_image_to_image(client):
    """
    Test image-to-image transformation using Hugging Face Inference API.
    
    Args:
        client: Hugging Face InferenceClient
        
    Returns:
        bool: True if test passed, False otherwise
    """
    logger.info("Testing image-to-image transformation...")
    
    try:
        # Set up the client with no caching
        client.headers["x-use-cache"] = "0"
        
        # Get config values
        config = TEST_CONFIG["image_to_image"]
        
        # Load the sample image
        sample_image_path = "data/inputs/interior_design/living-room-victorian.png"
        image = Image.open(sample_image_path)
        
        # Resize image if needed - SDXL Refiner works best with specific dimensions
        max_size = 1024
        aspect_ratio = image.width / image.height
        
        if aspect_ratio > 1:  # Landscape
            new_width = min(max_size, image.width)
            new_height = int(new_width / aspect_ratio)
        else:  # Portrait or square
            new_height = min(max_size, image.height)
            new_width = int(new_height * aspect_ratio)
        
        # Resize to appropriate dimensions
        if image.width > max_size or image.height > max_size:
            logger.info(f"Resizing image from {image.width}x{image.height} to {new_width}x{new_height}")
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Create byte stream for the image
        byte_stream = BytesIO()
        image.save(byte_stream, format='PNG', quality=95)
        byte_stream.seek(0)
        
        # Use image-to-image endpoint
        logger.info(f"Calling image-to-image API with model: {config['model']}")
        logger.info(f"Prompt: {config['prompt']}")
        
        result = client.image_to_image(
            model=config["model"],
            image=byte_stream,
            prompt=config["prompt"],
            negative_prompt=config["negative_prompt"],
            guidance_scale=config["guidance_scale"],
            num_inference_steps=config["num_inference_steps"],
            strength=config["strength"]
        )
        
        # Check if result is valid
        if result is None:
            logger.error("API returned None instead of an image")
            return False
        
        # Save the result
        output_dir = Path(TEST_CONFIG["output_dir"])
        output_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"image_to_image_test_{timestamp}.png"
        
        result.save(output_path)
        logger.info(f"Image-to-image transformation successful. Result saved to {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Image-to-image test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Main test function"""
    print("Starting Hugging Face API integration test...")
    
    # Set up test environment
    if not setup_test_environment():
        print("Failed to set up test environment. Exiting.")
        return
    
    # Initialize the Hugging Face client
    hf_token = os.environ.get("HUGGING_FACE_API_TOKEN")
    client = InferenceClient(token=hf_token)
    
    # Run tests
    text_to_image_result = test_text_to_image(client)
    image_to_image_result = test_image_to_image(client)
    
    # Print summary
    print("\n=== Test Results ===")
    print(f"Text-to-image: {'PASSED' if text_to_image_result else 'FAILED'}")
    print(f"Image-to-image: {'PASSED' if image_to_image_result else 'FAILED'}")
    
    if text_to_image_result and image_to_image_result:
        print("\nAll tests passed! Hugging Face API integration is working correctly.")
        print(f"Check the output images in {TEST_CONFIG['output_dir']}")
    else:
        print("\nSome tests failed. Please check the logs for details.")


if __name__ == "__main__":
    main() 