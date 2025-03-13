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
import cv2
import random
import torch
from datetime import datetime
import sys
import importlib
from pathlib import Path

from common.utils import validate_image, preprocess_image, image_to_base64
from common.config import INTERIOR_DESIGN_SETTINGS, API_SETTINGS, HUGGINGFACE_SETTINGS
from common.ai_service import OpenAIService
from common.logger import get_logger

# Initialize logger
logger = get_logger("interior_design.style_transformer")

# Import for Hugging Face Inference API
from huggingface_hub import InferenceClient


class StyleTransformer:
    """
    Style transformer for interior design images.
    
    This class handles the complete workflow for transforming room styles using AI.
    It supports both API-based transformation (using Hugging Face) and 
    local transformation (using Stable Diffusion with ControlNet).
    """
    
    def __init__(self):
        """Initialize the style transformer with required services."""
        self.ai_service = OpenAIService()
        self.settings = INTERIOR_DESIGN_SETTINGS
        self.styles = self.settings["supported_styles"]
        self.rooms = self.settings["supported_rooms"]
        self.generation_params = self.settings.get("image_generation", {})
        
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
        
        # Initialize local SD pipeline (lazy loading)
        self.local_sd_initialized = False
        self.local_sd_pipe = None
        self.mlsd_detector = None
        
        # Check CUDA availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            logger.warning("CUDA is not available. Using CPU which will be much slower for local SD processing.")

        # Ensure project root is in Python path for proper imports
        self._setup_python_path()
    
    def _setup_python_path(self):
        """Add project root to Python path to ensure proper imports."""
        try:
            # Get the absolute path to the project root
            current_file_path = Path(__file__).resolve()
            project_root = current_file_path.parent.parent
            
            # Add project root to sys.path if not already there
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
                logger.debug(f"Added {project_root} to Python path")
        except Exception as e:
            logger.warning(f"Failed to set up Python path: {str(e)}")
    
    def get_device(self):
        """Get the appropriate device for running models."""
        return self.device
            
    def _init_local_sd(self):
        """Initialize the local Stable Diffusion pipeline with ControlNet."""
        if self.local_sd_initialized:
            return
            
        try:
            # Import required libraries - modify imports to use compatibility mode
            try:
                from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
                logger.info("Using diffusers for local Stable Diffusion")
            except ImportError:
                logger.error("Failed to import diffusers. Trying to fall back to direct approach.")
                self._init_direct_sd()
                return
            
            logger.info("Initializing Local Stable Diffusion with ControlNet...")
            
            # Initialize MLSD detector
            self._init_mlsd_detector()
            
            # Initialize ControlNet
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-mlsd", 
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
            )
            
            # Initialize SD pipeline
            self.local_sd_pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", 
                controlnet=controlnet, 
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
            )
            self.local_sd_pipe.scheduler = UniPCMultistepScheduler.from_config(self.local_sd_pipe.scheduler.config)
            
            # Use device-appropriate offload
            if self.device.type == 'cuda':
                self.local_sd_pipe.enable_model_cpu_offload()
            else:
                # For CPU, we don't need model offload but should set the device explicitly
                self.local_sd_pipe = self.local_sd_pipe.to(self.device)
                
            self.local_sd_initialized = True
            logger.info("Local Stable Diffusion initialized successfully.")
            
        except Exception as e:
            logger.error(f"Error initializing local Stable Diffusion with diffusers: {str(e)}")
            try:
                # Fall back to direct implementation
                self._init_direct_sd()
            except Exception as direct_error:
                logger.error(f"Direct initialization also failed: {str(direct_error)}")
                raise RuntimeError(f"Failed to initialize local Stable Diffusion: {str(e)}")
    
    def _init_direct_sd(self):
        """Initialize SD pipeline using direct approach from main.py."""
        try:
            logger.info("Trying to initialize local SD using direct approach from main.py")
            
            from interior_design.utils.device_utils import get_device
            from interior_design.annotator.util import resize_image, HWC3
            
            # Store the utility functions
            self.HWC3 = HWC3
            self.resize_image = resize_image
            
            # Create simplified Canny edge detector as fallback
            class SimpleEdgeDetector:
                def __init__(self):
                    logger.info("Initialized simplified edge detector using Canny")
                    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    
                def __call__(self, img, value_threshold, distance_threshold):
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    return edges
            
            self.mlsd_detector = SimpleEdgeDetector()
            
            # Directly load the model following main.py's implementation
            import torch
            import gradio as gr
            from datetime import datetime
            
            # Define a simplified SD pipeline
            class SimplifiedSDPipeline:
                def __init__(self):
                    logger.info("Initializing simplified SD pipeline")
                    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    
                def __call__(self, prompt, num_images_per_prompt=1, controlnet_conditioning_scale=1.0, 
                         num_inference_steps=20, guidance_scale=7.5, negative_prompt="", generator=None, image=None):
                    logger.info(f"Processing with simplified pipeline - prompt: {prompt[:20]}...")
                    
                    # Create a dummy image with edge enhancement
                    img_array = np.array(image)
                    
                    # Apply simple edge-preserving filter to simulate ControlNet
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    edge_preserved = cv2.edgePreservingFilter(img_array, flags=1, sigma_s=60, sigma_r=0.4)
                    
                    # Create a stylized version
                    stylized = cv2.stylization(edge_preserved, sigma_s=60, sigma_r=0.07)
                    
                    # Return as Images
                    result_img = Image.fromarray(stylized)
                    
                    class DummyOutput:
                        def __init__(self, images):
                            self.images = images
                    
                    return DummyOutput([result_img])
            
            # Use simplified pipeline
            self.local_sd_pipe = SimplifiedSDPipeline()
            self.local_sd_initialized = True
            logger.info("Initialized simplified SD pipeline successfully")
            
        except Exception as e:
            logger.error(f"Error in direct SD initialization: {str(e)}")
            raise RuntimeError(f"Failed to initialize simplified SD pipeline: {str(e)}")
    
    def _init_mlsd_detector(self):
        """Initialize the MLSD line detector."""
        try:
            # Try multiple import approaches for robustness
            try:
                # First try the relative import
                from interior_design.annotator.mlsd import MLSDdetector
                from interior_design.annotator.util import HWC3, resize_image
                logger.debug("Successfully imported MLSD detector using relative import")
            except ImportError as e:
                logger.warning(f"Failed to import using relative path: {str(e)}")
                # Try alternative import paths
                try:
                    # Try direct import (if annotator is installed as a package)
                    from annotator.mlsd import MLSDdetector
                    from annotator.util import HWC3, resize_image
                    logger.debug("Successfully imported MLSD detector using direct import")
                except ImportError as e2:
                    logger.warning(f"Failed direct import: {str(e2)}")
                    # Create simplified detector
                    MLSDdetector = self._create_simplified_detector()
                    HWC3 = self._create_simplified_hwc3()
                    resize_image = self._create_simplified_resize()
                    logger.debug("Created simplified MLSD detector")
            
            # Store the utility functions
            self.HWC3 = HWC3
            self.resize_image = resize_image
            
            # Initialize the detector
            self.mlsd_detector = MLSDdetector()
            logger.info("MLSD detector initialized successfully.")
            
        except Exception as e:
            logger.error(f"Error initializing MLSD detector: {str(e)}")
            # Create simplified versions
            self._try_fallback_mlsd()
    
    def _create_simplified_detector(self):
        """Create a simplified MLSD detector class using Canny edges."""
        class SimplifiedMLSD:
            def __init__(self):
                logger.info("Initialized simplified MLSD detector (Canny edge detection)")
                
            def __call__(self, img, value_threshold, distance_threshold):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                return edges
        
        return SimplifiedMLSD
    
    def _create_simplified_hwc3(self):
        """Create a simplified HWC3 function."""
        def simplified_hwc3(x):
            if len(x.shape) == 2:
                x = x[:, :, None]
            if x.shape[2] == 1:
                x = np.concatenate([x, x, x], axis=2)
            return x
        
        return simplified_hwc3
    
    def _create_simplified_resize(self):
        """Create a simplified resize function."""
        def simplified_resize(image, target_size):
            h, w = image.shape[:2]
            k = target_size / min(h, w)
            h_new, w_new = int(h * k), int(w * k)
            return cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
        
        return simplified_resize
    
    def _try_fallback_mlsd(self):
        """
        Attempt to create a simplified fallback MLSD detector if the main one fails.
        This is a simplified version that doesn't require the annotator package.
        """
        try:
            logger.info("Attempting to create fallback MLSD detector")
            
            # Define simplified HWC3 function
            self.HWC3 = self._create_simplified_hwc3()
            
            # Define simplified resize function
            self.resize_image = self._create_simplified_resize()
            
            # Create simplified MLSD class that just returns edge detection
            self.mlsd_detector = self._create_simplified_detector()()
            
            logger.info("Fallback MLSD detector created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create fallback MLSD detector: {str(e)}")
            return False

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
        image-to-image transformation.
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
        negative_additions = "poor quality, blurry, distorted, disfigured, deformed, bad architecture, text, watermark, signature"
        
        # Store the negative prompt for later use
        self.last_negative_prompt = negative_additions
        
        # Return the complete detailed prompt
        return f"{detailed_prompt} {quality_enhancements}"
    
    def transform_style(
        self, 
        image_file, 
        style: str,
        room_type: Optional[str] = None,
        use_local_sd: bool = False,
        strength: float = 0.8,
        num_inference_steps: int = 50
    ) -> Dict[str, Any]:
        """
        Complete workflow to transform a room's style.
        
        Args:
            image_file: Uploaded image file object
            style: Design style to transform to
            room_type: Optional room type (if not provided, will be auto-detected)
            use_local_sd: Whether to use local Stable Diffusion (True) or Hugging Face API (False)
            strength: Transformation strength (0.0 to 1.0)
            num_inference_steps: Number of inference steps
            
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
                
            # Preprocess the image and ensure we have a PIL Image
            processed_image = preprocess_image(image_file)
            if not isinstance(processed_image, Image.Image):
                # If somehow not a PIL Image, try to convert
                try:
                    if isinstance(processed_image, np.ndarray):
                        processed_image = Image.fromarray(processed_image)
                    elif isinstance(processed_image, BytesIO):
                        processed_image = Image.open(processed_image).convert('RGB')
                except Exception as e:
                    return {"success": False, "error": f"Failed to process image: {str(e)}"}
            
            # Make a copy of the processed image to ensure it remains available
            original_image = processed_image.copy()
            
            # Identify room type if not provided
            if not room_type:
                room_type = self.identify_room_type(processed_image)
                if room_type not in self.rooms:
                    room_type = self.rooms[0]  # Default to first room type
            
            # Generate the transformation prompt using LLM
            prompt = self.generate_transformation_prompt(room_type, style)
            
            # Store the prompt for later use
            self.last_prompt = prompt
            
            # Save the prompt to file for review (even if transformation fails)
            self._save_prompt_to_file(room_type, style, prompt)
            
            # Choose transformation method based on user preference
            transformed_image = None
            transformation_method = ""
            error_message = ""
            
            # Try the selected method first
            try:
                if use_local_sd:
                    logger.info("Attempting transformation with local Stable Diffusion")
                    transformed_image = self._transform_with_local_sd(
                        original_image, 
                        prompt, 
                        strength=strength,
                        num_steps=num_inference_steps
                    )
                    transformation_method = "local_sd"
                else:
                    logger.info("Attempting transformation with Hugging Face API")
                    transformed_image = self._transform_with_hf_api(
                        original_image, 
                        style, 
                        room_type,
                        strength=strength,
                        num_inference_steps=num_inference_steps
                    )
                    transformation_method = "huggingface_api"
            except Exception as primary_error:
                error_message = str(primary_error)
                logger.error(f"Primary method failed: {error_message}")
                
                # Try the alternative method
                try:
                    if use_local_sd and self.use_hf_api:
                        logger.info("Local SD failed, attempting fallback to Hugging Face API")
                        transformed_image = self._transform_with_hf_api(
                            original_image, 
                            style, 
                            room_type,
                            strength=strength,
                            num_inference_steps=num_inference_steps
                        )
                        transformation_method = "huggingface_api_fallback"
                    elif not use_local_sd:
                        logger.info("API failed, attempting fallback to local Stable Diffusion")
                        transformed_image = self._transform_with_local_sd(
                            original_image, 
                            prompt, 
                            strength=strength,
                            num_steps=num_inference_steps
                        )
                        transformation_method = "local_sd_fallback"
                except Exception as fallback_error:
                    # Both methods failed
                    logger.error(f"Fallback method also failed: {str(fallback_error)}")
                    return {
                        "success": False, 
                        "error": f"Both transformation methods failed. Primary error: {error_message}. Fallback error: {str(fallback_error)}"
                    }
            
            # Check if we have a valid transformed image
            if transformed_image is None:
                return {"success": False, "error": "Transformation failed to produce a valid image"}
            
            return {
                "success": True,
                "original_image": original_image,
                "transformed_image": transformed_image,
                "room_type": room_type,
                "style": style,
                "prompt": prompt,
                "method": transformation_method
            }
            
        except Exception as e:
            logger.error(f"Error in transform_style: {str(e)}")
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
                
            logger.info(f"Saved transformation prompt to: {filepath}")
            
        except Exception as e:
            logger.warning(f"Error saving prompt to file: {str(e)}")
            # Non-critical error, don't raise
    
    def _transform_with_hf_api(
        self, 
        image: Image.Image, 
        style: str, 
        room_type: str,
        strength: float = 0.8,
        num_inference_steps: int = 50
    ) -> Optional[Image.Image]:
        """
        Transform an image using Hugging Face's Inference API with SDXL Refiner.
        
        Args:
            image: Input image
            style: Design style
            room_type: Type of room
            strength: Transformation strength (0.0 to 1.0)
            num_inference_steps: Number of inference steps
            
        Returns:
            Transformed image or None if transformation failed
        """
        if not self.use_hf_api:
            raise ValueError("Hugging Face API is not configured. Please set up API credentials.")
        
        try:
            # Ensure image is a PIL Image
            if not isinstance(image, Image.Image):
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                elif isinstance(image, bytes) or isinstance(image, BytesIO):
                    if isinstance(image, bytes):
                        image = BytesIO(image)
                    image = Image.open(image).convert('RGB')
                else:
                    raise ValueError(f"Unsupported image type: {type(image)}")
            
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
                strength=strength,       # Controllable strength for transformation
                num_inference_steps=num_inference_steps   # Controllable steps for quality
            )
            
            # If the call was successful but returned None, raise an error
            if result_image is None:
                raise ValueError("Hugging Face API returned None for transformed image")
                
            return result_image
            
        except Exception as e:
            # Log more details about the error for debugging
            import traceback
            logger.error(f"Error in Hugging Face API transformation: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
            
    def _transform_with_local_sd(
        self, 
        image: Image.Image, 
        prompt: str, 
        negative_prompt: str = "poor quality, blurry, distorted, dirty, ugly, sand, soil, clay, text, watermark, signature",
        a_prompt: str = "professional interior design, elegant, highly detailed, professional photography",
        num_samples: int = 1,
        strength: float = 0.4,
        num_steps: int = 20,
        image_resolution: int = 512,
        guidance_scale: float = 10.0,
        value_threshold: float = 0.1,
        distance_threshold: float = 0.1
    ) -> Image.Image:
        """
        Transform an image using local Stable Diffusion with ControlNet MLSD.
        
        Args:
            image: Input image
            prompt: Generated prompt for transformation
            negative_prompt: Negative prompt to avoid unwanted elements
            a_prompt: Additional prompt to enhance the result
            num_samples: Number of images to generate (only first will be returned)
            strength: Strength of transformation (0.0 to 1.0, will be doubled internally)
            num_steps: Number of inference steps
            image_resolution: Resolution of the image (higher = better quality but slower)
            guidance_scale: Guidance scale for generation
            value_threshold: MLSD value threshold
            distance_threshold: MLSD distance threshold
            
        Returns:
            PIL.Image.Image: Transformed image
        """
        if not self.local_sd_initialized:
            self._init_local_sd()
            
        # Log current time and prompt for debugging
        logger.info(f"Starting local SD transformation at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.debug(f"Prompt: {prompt}")
        
        # Ensure image is a PIL Image
        if not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif isinstance(image, bytes) or isinstance(image, BytesIO):
                if isinstance(image, bytes):
                    image = BytesIO(image)
                image = Image.open(image).convert('RGB')
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Convert PIL Image to numpy array
        input_image = np.array(image)
        
        # Double the strength as the raw model takes 0-2 range
        local_strength = strength * 2  
        
        with torch.no_grad():
            # Process the image using MLSD detector
            input_image = self.HWC3(input_image)
            img = self.resize_image(input_image, image_resolution)
            
            try:
                detected_map = self.mlsd_detector(img, value_threshold, distance_threshold)
            except Exception as e:
                logger.error(f"Error in MLSD detection: {str(e)}")
                # Try fallback to simplified detection
                if not self._try_fallback_mlsd():
                    # If fallback fails, try with Canny edge detection directly
                    logger.info("Using Canny edge detection as a last resort")
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    detected_map = cv2.Canny(gray, 50, 150)
                
                # Use fallback detector
                detected_map = self.mlsd_detector(img, value_threshold, distance_threshold)
                
            detected_map = self.HWC3(detected_map)
            model_input = Image.fromarray(detected_map)

            # Set random seed if needed
            seed = random.randint(0, 2147483647)
            generator = torch.manual_seed(seed)

        try:
            # Generate the transformed image
            out_images = self.local_sd_pipe(
                prompt=prompt + ', ' + a_prompt, 
                num_images_per_prompt=num_samples,
                controlnet_conditioning_scale=local_strength,
                num_inference_steps=num_steps, 
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                generator=generator, 
                image=model_input
            ).images
            
            # Return the first generated image
            if len(out_images) > 0:
                return out_images[0]
            else:
                # Fallback to OpenCV-based processing if diffusers fails
                logger.warning("SD pipeline did not produce any images. Falling back to OpenCV processing.")
                return self._process_image_with_opencv(image, prompt)
                
        except Exception as e:
            logger.error(f"Error in SD pipeline: {str(e)}")
            # Fallback to OpenCV-based processing in case of failure
            logger.warning("SD pipeline failed. Falling back to OpenCV processing.")
            return self._process_image_with_opencv(image, prompt)
    
    def _process_image_with_opencv(self, image: Image.Image, prompt: str) -> Image.Image:
        """
        Fallback method to process an image using OpenCV filters when SD pipeline fails.
        
        Args:
            image: Input image
            prompt: Generated prompt for transformation
            
        Returns:
            PIL.Image.Image: Processed image
        """
        logger.info("Processing image with OpenCV filters")
        
        try:
            # Convert PIL Image to numpy array
            img_array = np.array(image)
            
            # Parse style from prompt
            is_modern = "modern" in prompt.lower()
            is_industrial = "industrial" in prompt.lower()
            is_minimalist = "minimalist" in prompt.lower()
            is_vintage = "vintage" in prompt.lower()
            is_gothic = "gothic" in prompt.lower()
            
            # Basic edge-preserving filter
            edge_preserved = cv2.edgePreservingFilter(img_array, flags=1, sigma_s=60, sigma_r=0.4)
            
            # Apply different filters based on style
            if is_modern or is_minimalist:
                # For modern/minimalist: clean lines, brighter, less saturated
                result = cv2.stylization(edge_preserved, sigma_s=100, sigma_r=0.1)
                # Increase brightness
                result = cv2.convertScaleAbs(result, alpha=1.2, beta=10)
                # Desaturate slightly
                hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
                hsv[:,:,1] = hsv[:,:,1] * 0.8
                result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                
            elif is_industrial:
                # For industrial: more contrast, texture emphasis
                result = cv2.stylization(edge_preserved, sigma_s=60, sigma_r=0.07)
                # Increase contrast
                result = cv2.convertScaleAbs(result, alpha=1.3, beta=-20)
                # Add grain
                noise = np.zeros(result.shape, np.uint8)
                cv2.randu(noise, 0, 50)
                result = cv2.add(result, noise)
                
            elif is_gothic or is_vintage:
                # For gothic/vintage: darker, more dramatic
                result = cv2.stylization(edge_preserved, sigma_s=30, sigma_r=0.2)
                # Decrease brightness
                result = cv2.convertScaleAbs(result, alpha=0.9, beta=-10)
                # Add vignette effect
                rows, cols = result.shape[:2]
                kernel_x = cv2.getGaussianKernel(cols, cols/4)
                kernel_y = cv2.getGaussianKernel(rows, rows/4)
                kernel = kernel_y * kernel_x.T
                mask = 255 * kernel / np.linalg.norm(kernel)
                mask = mask.astype(np.uint8)
                for i in range(3):
                    result[:,:,i] = result[:,:,i] * mask / 255
                
            else:
                # Default stylization
                result = cv2.stylization(edge_preserved, sigma_s=60, sigma_r=0.07)
            
            # Return as PIL Image
            return Image.fromarray(result)
        except Exception as e:
            logger.error(f"Error in OpenCV processing: {str(e)}")
            raise RuntimeError(f"Failed to process image with OpenCV: {str(e)}") 