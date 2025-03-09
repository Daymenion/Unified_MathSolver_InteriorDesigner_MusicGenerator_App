"""
Style Transformer module for the Interior Design app.

This module contains the core functionality for transforming room styles.
"""

import os
import requests
import json
import base64
from PIL import Image
from io import BytesIO
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np

from common.utils import validate_image, preprocess_image, image_to_base64
from common.config import INTERIOR_DESIGN_SETTINGS, API_SETTINGS, HUGGINGFACE_SETTINGS
from common.ai_service import OpenAIService

# Import for Hugging Face Inference API
from huggingface_hub import InferenceClient


class StyleTransformer:
    """
    Style transformer for interior design images.
    
    This class handles the complete workflow for transforming room styles using AI.
    """
    
    def __init__(self):
        """Initialize the style transformer with required services."""
        self.ai_service = OpenAIService()
        self.settings = INTERIOR_DESIGN_SETTINGS
        self.styles = self.settings["supported_styles"]
        self.rooms = self.settings["supported_rooms"]
        self.generation_params = self.settings["image_generation"]
        
        # Initialize Hugging Face Inference client if API token is available
        self.hf_token = os.environ.get("HUGGING_FACE_API_TOKEN", None)
        self.use_hf_api = self.hf_token is not None
        if self.use_hf_api:
            self.hf_client = InferenceClient(token=self.hf_token)
            
        # Mapping of styles to SDXL Refiner model
        self.style_to_model = {
            "Modern": {
                "model": "stabilityai/stable-diffusion-xl-refiner-1.0",
            },
            "Soho": {
                "model": "stabilityai/stable-diffusion-xl-refiner-1.0",
            },
            "Gothic": {
                "model": "stabilityai/stable-diffusion-xl-refiner-1.0",
            }
        }
    
    def identify_room_type(self, image: Union[Image.Image, str]) -> str:
        """
        Identify the type of room in the image.
        
        Args:
            image: PIL Image object or path to image file
            
        Returns:
            str: Room type (e.g., 'Living Room', 'Kitchen')
        """
        prompt = (
            f"Identify the type of room shown in this image. "
            f"Choose one from this list: {', '.join(self.rooms)}.\n"
            f"Reply with ONLY the room type name, nothing else."
        )
        
        room_type = self.ai_service.analyze_image(image, prompt)
        return room_type.strip()
    
    def generate_transformation_prompt(self, room_type: str, style: str) -> str:
        """
        Generate a highly detailed prompt for style transformation using LLM.
        
        This method uses OpenAI to generate a detailed prompt optimized for image-to-image
        style transformation that maintains the composition while completely
        transforming all interior elements (furniture, walls, colors, etc).
        
        Args:
            room_type: Type of room (e.g., 'Living Room', 'Kitchen')
            style: Design style (e.g., 'Modern', 'Soho', 'Gothic')
            
        Returns:
            str: Complete transformation prompt optimized for image-to-image models
        """
        # First, prepare a detailed instruction for the LLM
        llm_instruction = f"""
        Create a highly detailed prompt for an AI image-to-image transformation model that will DRAMATICALLY transform a {room_type} 
        into {style} style interior design. The prompt should create a COMPLETE VISUAL TRANSFORMATION with:
        
        1. ONLY maintain the basic room layout and general positioning/perspective - EVERYTHING ELSE MUST BE COMPLETELY CHANGED.
        2. DRASTICALLY transform ALL interior elements to create a completely different {style} aesthetic:
           - Replace ALL furniture with entirely different style-appropriate pieces (different shapes, materials, colors)
           - Completely transform wall treatments, colors, and textures to {style} aesthetics
           - Replace flooring with entirely different materials and finishes typical of {style} design
           - Transform windows and window treatments to match {style} style completely
           - Replace ALL lighting fixtures with style-specific alternatives
           - Change ALL decorative elements, artwork, and accessories to match the style
        3. Create STRONG VISUAL CONTRAST between the original room and the transformed room
        4. Use a completely different color palette appropriate for {style} design
        5. Include specific, vivid details about materials, textures, colors, and unique features of {style} design
        6. Describe dramatic lighting changes that enhance the {style} atmosphere
        
        The prompt should be highly detailed and descriptive, around 200-300 words, specifically optimized for 
        stabilityai/stable-diffusion-xl-refiner-1.0 model doing image-to-image transformation.
        IMPORTANT: Format as a direct prompt only, creating the most DRAMATIC TRANSFORMATION possible while only maintaining the basic room layout.
        """
        
        # Generate prompt using LLM
        detailed_prompt = self.ai_service.generate_text(llm_instruction, temperature=0.7)
        
        # Add enhancement for image quality but emphasize dramatic transformation
        quality_enhancements = (
            "dramatic transformation, completely different style, high-quality detailed texture, "
            "professional interior photography, architectural visualization, ultra detailed, "
            "8k resolution, perfect lighting, photorealistic rendering"
        )
        
        # Add specific negative prompts to avoid quality issues but not restrict style changes
        negative_additions = "poor quality, blurry, distorted, bad resolution, text, watermark, signature"
        
        # Store the negative prompt for later use
        self.last_negative_prompt = negative_additions
        
        # Return the complete detailed prompt
        return f"{detailed_prompt} {quality_enhancements}"
    
    def transform_style(
        self, 
        image_file, 
        style: str,
        room_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete workflow to transform a room's style.
        
        Args:
            image_file: Uploaded image file object
            style: Design style to transform to
            room_type: Optional room type (if not provided, will be auto-detected)
            
        Returns:
            Dict[str, Any]: Transformation results
        """
        # Validate the image
        is_valid, error_message = validate_image(image_file)
        if not is_valid:
            return {"success": False, "error": error_message}
        
        try:
            # Store the original image path or name if available
            if hasattr(image_file, 'name'):
                self.last_image_path = image_file.name
            elif isinstance(image_file, str):
                self.last_image_path = image_file
            else:
                self.last_image_path = "room"
                
            # Preprocess the image
            processed_image = preprocess_image(image_file)
            
            # Identify room type if not provided
            if not room_type:
                room_type = self.identify_room_type(processed_image)
                if room_type not in self.rooms:
                    room_type = self.rooms[0]  # Default to first room type
            
            # Generate the transformation prompt using LLM
            prompt = self.generate_transformation_prompt(room_type, style)
            
            # Store the prompt for use in _transform_with_hf_api
            self.last_prompt = prompt
            
            # Save the prompt to file for review (even if API fails)
            self._save_prompt_to_file(room_type, style, prompt)
            
            # Use Hugging Face API for transformation
            if not self.use_hf_api:
                return {"success": False, "error": "Hugging Face API is not configured. Please set up API credentials."}
            
            try:
                transformed_image = self._transform_with_hf_api(processed_image, style, room_type)
                
                return {
                    "success": True,
                    "original_image": processed_image,
                    "transformed_image": transformed_image,
                    "room_type": room_type,
                    "style": style,
                    "prompt": prompt,
                    "method": "huggingface_api"
                }
            except Exception as e:
                # If transformation failed, return the error
                return {"success": False, "error": f"Error transforming image: {str(e)}"}
            
        except Exception as e:
            return {"success": False, "error": f"Error transforming image: {str(e)}"}
            
    def _save_prompt_to_file(self, room_type: str, style: str, prompt: str):
        """
        Save the generated prompt to a file for manual inspection.
        
        Args:
            room_type: Type of room
            style: Design style
            prompt: Generated transformation prompt
        """
        try:
            import os
            from datetime import datetime
            
            # Create output directory if it doesn't exist
            output_dir = "data/outputs/interior_design"
            os.makedirs(output_dir, exist_ok=True)
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.basename(self.last_image_path) if hasattr(self, 'last_image_path') else "room"
            filename = f"{base_name.split('.')[0]}_{style.lower()}_{timestamp}_prompt.txt"
            filepath = os.path.join(output_dir, filename)
            
            # Write the prompt to file
            with open(filepath, 'w') as f:
                f.write(f"Room Type: {room_type}\n")
                f.write(f"Style: {style}\n\n")
                f.write(f"Transformation Prompt:\n{prompt}")
                
            print(f"Saved transformation prompt to: {filepath}")
            
        except Exception as e:
            print(f"Error saving prompt to file: {str(e)}")
            # Non-critical error, don't raise
    
    def _transform_with_hf_api(self, image: Image.Image, style: str, room_type: str) -> Optional[Image.Image]:
        """
        Transform an image using Hugging Face's Inference API with SDXL Refiner.
        
        Args:
            image: Input image
            style: Design style
            room_type: Type of room
            
        Returns:
            Transformed image or None if transformation failed
        """
        if not self.use_hf_api:
            raise ValueError("Hugging Face API is not configured. Please set up API credentials.")
        
        try:
            # Get model info
            style_info = self.style_to_model.get(style, self.style_to_model["Modern"])
            model_id = style_info["model"]
            
            # Process the image - ensure correct size to avoid API issues
            # Resize if needed (SDXL works best with specific dimensions)
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
                image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Save image to a byte stream
            byte_stream = BytesIO()
            image.save(byte_stream, format='PNG', quality=95)
            byte_stream.seek(0)
            
            # Set up the client and disable caching to get fresh results
            self.hf_client.headers["x-use-cache"] = "0"
            
            # The negative prompt was stored when generating the transformation prompt
            negative_prompt = "poor quality, blurry, distorted, disfigured, deformed, bad architecture, text, watermark, signature"
            
            # Optimized parameters for SDXL Refiner with more dramatic transformation
            result_image = self.hf_client.image_to_image(
                model=model_id,
                image=byte_stream,
                prompt=self.last_prompt,
                negative_prompt=negative_prompt,
                guidance_scale=8.0,      # Higher guidance scale for better prompt adherence
                strength=0.95,           # Very high strength for dramatic transformation
                num_inference_steps=50   # More steps for higher quality
            )
            
            # If the call was successful but returned None, raise an error
            if result_image is None:
                raise ValueError("Hugging Face API returned None for transformed image")
                
            return result_image
            
        except Exception as e:
            # Log more details about the error for debugging
            import traceback
            print(f"Error in Hugging Face API transformation: {str(e)}")
            traceback.print_exc()
            raise  # Re-raise to handle it in the transform_style method 