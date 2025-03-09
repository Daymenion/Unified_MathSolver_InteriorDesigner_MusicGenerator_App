"""
AI Service module for integration with OpenAI API.

This module provides functions to interact with the OpenAI API for:
1. Text completion/generation
2. Image analysis (OCR)
3. Code generation and execution
"""

import os
import json
import base64
import time
import httpx
from typing import Dict, List, Union, Optional, Any
from PIL import Image
from io import BytesIO

from .config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS
from .utils import image_to_base64

# Import for Hugging Face Inference API
from huggingface_hub import InferenceClient

class OpenAIService:
    """Service class for OpenAI API integration."""
    
    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None, timeout: int = 60, max_retries: int = 3):
        """
        Initialize the OpenAI service.
        
        Args:
            api_key: OpenAI API key (defaults to environment variable)
            api_base: OpenAI API base URL (defaults to environment variable)
            timeout: Timeout for API calls in seconds
            max_retries: Maximum number of retries for failed API calls
        """
        self.api_key = api_key or OPENAI_API_KEY
        self.api_base = api_base or "https://api.openai.com/v1"
        
        # API configuration
        self.model = OPENAI_MODEL
        self.temperature = OPENAI_TEMPERATURE
        self.max_tokens = OPENAI_MAX_TOKENS
        self.max_retries = max_retries
        self.retry_delay = 2  # seconds
        
        try:
            # Create HTTP client without proxy settings
            self.http_client = httpx.Client(timeout=timeout)
        except Exception as e:
            print(f"Error initializing HTTP client: {str(e)}")
            self.http_client = None
        
        # Initialize Hugging Face Inference client if token is available
        self.hf_token = os.environ.get("HUGGING_FACE_API_TOKEN", "")
        self.use_hf_api = bool(self.hf_token) and os.environ.get("HUGGINGFACE_USE_API", "true").lower() in ["true", "1", "yes"]
        
        if self.use_hf_api:
            self.hf_client = InferenceClient(token=self.hf_token)
            # Disable caching to get fresh results
            self.hf_client.headers["x-use-cache"] = "0"
    
    def get_headers(self) -> Dict[str, str]:
        """Get headers for API calls."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def _handle_api_error(self, response: httpx.Response) -> Dict[str, Any]:
        """Handle API error responses and provide detailed errors."""
        error_detail = "Unknown error"
        try:
            error_data = response.json()
            if 'error' in error_data:
                error_detail = error_data['error'].get('message', str(error_data))
            else:
                error_detail = str(error_data)
        except:
            error_detail = response.text or f"HTTP {response.status_code}"
            
        return {"error": error_detail, "status_code": response.status_code}
    
    def generate_text(
        self, 
        prompt: str, 
        temperature: Optional[float] = None, 
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text using OpenAI API.
        
        Args:
            prompt: The text prompt to generate from
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            str: Generated text
        """
        if not self.http_client:
            raise ValueError("OpenAI client not properly initialized. Check API key.")
            
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens
        }
        
        # Retry logic for API calls
        retries = 0
        while retries <= self.max_retries:
            try:
                response = self.http_client.post(
                    url,
                    headers=self.get_headers(),
                    json=payload
                )
                
                if response.status_code == 200:
                    response_json = response.json()
                    return response_json['choices'][0]['message']['content']
                else:
                    error_info = self._handle_api_error(response)
                    
                    # Decide whether to retry based on status code
                    if response.status_code in [429, 500, 502, 503, 504]:
                        retries += 1
                        if retries <= self.max_retries:
                            sleep_time = self.retry_delay * (2 ** (retries - 1))  # Exponential backoff
                            print(f"Retrying in {sleep_time} seconds... (Attempt {retries}/{self.max_retries})")
                            time.sleep(sleep_time)
                            continue
                    
                    # If we reach here, we're out of retries or shouldn't retry
                    raise ValueError(f"OpenAI API error: {error_info['error']}")
            except httpx.RequestError as e:
                print(f"Network error when calling OpenAI API: {str(e)}")
                
                if retries < self.max_retries:
                    retries += 1
                    sleep_time = self.retry_delay * (2 ** (retries - 1))
                    print(f"Retrying in {sleep_time} seconds... (Attempt {retries}/{self.max_retries})")
                    time.sleep(sleep_time)
                else:
                    raise ValueError(f"Network error when calling OpenAI API: {str(e)}")
        
        # If we reach here, all retries failed
        raise ValueError("All retries failed when calling OpenAI API")
    
    def analyze_image(
        self, 
        image: Union[Image.Image, str], 
        prompt: str
    ) -> str:
        """
        Analyze an image using OpenAI's vision capabilities.
        
        Args:
            image: PIL Image object or path to image file
            prompt: Instructions for analyzing the image
            
        Returns:
            str: Analysis result
        """
        if not self.http_client:
            raise ValueError("OpenAI client not properly initialized. Check API key.")
            
        # Convert image to base64
        base64_image = image_to_base64(image)
        
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            "max_tokens": self.max_tokens
        }
        
        # Retry logic for API calls
        retries = 0
        while retries <= self.max_retries:
            try:
                response = self.http_client.post(
                    url,
                    headers=self.get_headers(),
                    json=payload
                )
                
                if response.status_code == 200:
                    response_json = response.json()
                    return response_json['choices'][0]['message']['content']
                else:
                    error_info = self._handle_api_error(response)
                    
                    # Decide whether to retry based on status code
                    if response.status_code in [429, 500, 502, 503, 504]:
                        retries += 1
                        if retries <= self.max_retries:
                            sleep_time = self.retry_delay * (2 ** (retries - 1))  # Exponential backoff
                            print(f"Retrying in {sleep_time} seconds... (Attempt {retries}/{self.max_retries})")
                            time.sleep(sleep_time)
                            continue
                    
                    # If we reach here, we're out of retries or shouldn't retry
                    raise ValueError(f"OpenAI API error: {error_info['error']}")
            except httpx.RequestError as e:
                print(f"Network error when calling OpenAI API: {str(e)}")
                
                if retries < self.max_retries:
                    retries += 1
                    sleep_time = self.retry_delay * (2 ** (retries - 1))
                    print(f"Retrying in {sleep_time} seconds... (Attempt {retries}/{self.max_retries})")
                    time.sleep(sleep_time)
                else:
                    raise ValueError(f"Network error when calling OpenAI API: {str(e)}")
        
        # If we reach here, all retries failed
        raise ValueError("All retries failed when calling OpenAI API")
    
    def generate_and_execute_code(
        self, 
        problem_description: str,
        problem_type: str = "math"
    ) -> Dict[str, str]:
        """
        Generate and execute code for solving a problem.
        
        Args:
            problem_description: Description of the problem to solve
            problem_type: Type of problem (e.g., 'math', 'statistics')
            
        Returns:
            Dict[str, str]: Results with code and solution
        """
        # Create a prompt for code generation
        if problem_type == "math":
            prompt = (
                f"Generate Python code to solve this math problem. "
                f"Use sympy, numpy, and other appropriate libraries. "
                f"Problem: {problem_description}\n\n"
                f"Write only the Python code without any explanation. "
                f"The code should print the step-by-step solution and the final answer."
            )
        else:
            prompt = (
                f"Generate Python code to solve this problem: {problem_description}\n\n"
                f"Write only the Python code without any explanation. "
                f"The code should print the step-by-step solution and the final answer."
            )
            
        # Generate the code
        code = self.generate_text(prompt)
        
        # Now generate explanation and solution
        explanation_prompt = (
            f"Here's a Python code to solve a {problem_type} problem:\n\n"
            f"```python\n{code}\n```\n\n"
            f"Execute this code and provide:\n"
            f"1. A step-by-step explanation of how the solution works\n"
            f"2. The final answer\n"
            f"Format the solution clearly with LaTeX for any mathematical notation."
        )
        
        explanation = self.generate_text(explanation_prompt)
        
        return {
            "code": code,
            "explanation": explanation,
            "problem_type": problem_type
        }

    def text_to_image(
        self,
        prompt: str,
        model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        negative_prompt: str = "poor quality, blurry, distorted"
    ) -> Optional[Image.Image]:
        """
        Generate an image from text using Hugging Face API.
        
        Args:
            prompt: Text prompt for image generation
            model: Model to use
            width: Width of the image
            height: Height of the image
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale for generation
            negative_prompt: Negative prompt to avoid unwanted elements
            
        Returns:
            PIL.Image.Image or None: Generated image
        """
        if not self.use_hf_api:
            print("Hugging Face API not configured")
            return None
        
        try:
            result = self.hf_client.text_to_image(
                prompt=prompt,
                model=model,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height
            )
            return result
        except Exception as e:
            print(f"Error in text-to-image generation: {str(e)}")
            return None

    def image_to_image(
        self,
        image: Union[str, Image.Image, BytesIO],
        prompt: str,
        model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        strength: float = 0.7,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        negative_prompt: str = "poor quality, blurry, distorted"
    ) -> Optional[Image.Image]:
        """
        Transform an image using Hugging Face API.
        
        Args:
            image: Input image (PIL Image, path, or BytesIO)
            prompt: Text prompt for transformation
            model: Model to use
            strength: Transformation strength (0.0 to 1.0)
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale for generation
            negative_prompt: Negative prompt to avoid unwanted elements
            
        Returns:
            PIL.Image.Image or None: Transformed image
        """
        if not self.use_hf_api:
            print("Hugging Face API not configured")
            return None
        
        try:
            # If image is a file path, open it
            if isinstance(image, str) and os.path.isfile(image):
                image = Image.open(image)
            
            result = self.hf_client.image_to_image(
                model=model,
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                strength=strength
            )
            return result
        except Exception as e:
            print(f"Error in image-to-image transformation: {str(e)}")
            return None 