"""
Music Generator module for creating lyrics and cover art.

This module contains the core functionality for generating personalized song lyrics
and cover art based on user preferences.
"""

import os
import json
import base64
from PIL import Image
from io import BytesIO
from typing import Dict, Any, List, Optional, Union, Tuple
import math
import io
import re
import random
import colorsys
import time
import numpy as np
from string import punctuation
from datetime import datetime
from pathlib import Path

from common.ai_service import OpenAIService
from common.utils import (
    image_to_base64, 
    format_timestamp, 
    save_output, 
    sanitize_filename, 
    create_directory,
    print_debug
)
from common.config import MUSIC_GENERATOR_SETTINGS

# Import for Hugging Face Inference API
from huggingface_hub import InferenceClient


class MusicGenerator:
    """
    Music generator for lyrics and cover art.
    
    This class handles the complete workflow for generating song lyrics and cover art
    based on user preferences.
    """
    
    def __init__(self):
        """Initialize the music generator with required services."""
        self.ai_service = OpenAIService()
        self.settings = MUSIC_GENERATOR_SETTINGS
        self.genres = self.settings["supported_genres"]
        self.moods = self.settings["supported_moods"]
        self.purposes = self.settings["supported_purposes"]
        
        # Create output directory if it doesn't exist
        self.output_dir = Path(self.settings["output_directory"])
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Hugging Face Inference client if API token is available
        self.hf_token = os.environ.get("HUGGING_FACE_API_TOKEN", None)
        self.use_hf_api = self.hf_token is not None
        if self.use_hf_api:
            self.hf_client = InferenceClient(token=self.hf_token)
            
        # Mapping of genres to text-to-image model for cover art generation
        self.genre_to_model = {
            "Pop": {
                "model": "stabilityai/stable-diffusion-xl-base-1.0"
            },
            "Rock": {
                "model": "stabilityai/stable-diffusion-xl-base-1.0"
            },
            "Jazz": {
                "model": "stabilityai/stable-diffusion-xl-base-1.0"
            },
            "Hip Hop": {
                "model": "stabilityai/stable-diffusion-xl-base-1.0"
            },
            "Classical": {
                "model": "stabilityai/stable-diffusion-xl-base-1.0"
            },
            "Electronic": {
                "model": "stabilityai/stable-diffusion-xl-base-1.0"
            },
            "Country": {
                "model": "stabilityai/stable-diffusion-xl-base-1.0"
            }
        }
    
    def generate_lyrics(
        self, 
        genre: str, 
        mood: str, 
        purpose: str, 
        custom_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate song lyrics based on user preferences.
        
        Args:
            genre: Music genre (e.g., 'Pop', 'Rock')
            mood: Emotional mood (e.g., 'Happy', 'Sad')
            purpose: Song purpose (e.g., 'For my love')
            custom_description: Optional custom description
            
        Returns:
            Dict[str, Any]: Generated lyrics data with structured content
        """
        if not genre or not isinstance(genre, str):
            return {"success": False, "error": "Genre is required and must be a string"}
            
        if not mood or not isinstance(mood, str):
            return {"success": False, "error": "Mood is required and must be a string"}
            
        if not purpose or not isinstance(purpose, str):
            return {"success": False, "error": "Purpose is required and must be a string"}
            
        # Build the prompt
        prompt = (
            f"Write original song lyrics for a {genre} song with a {mood} mood. "
            f"The song is {purpose}."
        )
        
        if custom_description:
            prompt += f"\nAdditional context: {custom_description}"
        
        prompt += """
The song should have:
- A clear verse-chorus structure
- At least 2 verses and a chorus
- Optional bridge
- Consistent rhyme scheme
- Thematic coherence around the purpose
- Emotional resonance matching the mood
- A catchy title that reflects the song's theme

FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:
TITLE: [Song Title]

VERSE 1:
[Verse 1 lyrics]

CHORUS:
[Chorus lyrics]

VERSE 2:
[Verse 2 lyrics]

[Include BRIDGE if appropriate]

[End with CHORUS]
"""
        
        try:
            # Generate the lyrics
            lyrics_raw = self.ai_service.generate_text(prompt, temperature=0.7, max_tokens=1500)
            
            if not lyrics_raw or not isinstance(lyrics_raw, str):
                return {"success": False, "error": "Failed to generate lyrics: No valid text returned from AI service"}
            
            # Parse the lyrics to extract title and content
            title = "Untitled"
            content = lyrics_raw
            
            # Extract title if present
            if "TITLE:" in lyrics_raw:
                title_parts = lyrics_raw.split("TITLE:", 1)[1].split("\n", 1)
                if len(title_parts) > 0:
                    title = title_parts[0].strip()
                    if len(title_parts) > 1:
                        content = title_parts[1].strip()
            
            # Extract main theme from lyrics for cover art
            try:
                theme_prompt = (
                    f"Analyze these song lyrics and extract the main visual theme or "
                    f"central imagery that could be used for cover art. "
                    f"First identify the song's emotional core, then describe 3-5 visual elements that represent it."
                    f"Format your response as a direct description, e.g., 'A sunrise over mountains symbolizing hope':\n\n{lyrics_raw}"
                )
                main_theme = self.ai_service.generate_text(theme_prompt, temperature=0.2, max_tokens=100)
                
                if not main_theme or not isinstance(main_theme, str):
                    main_theme = f"Visual representation of a {mood} {genre} song"
                    
            except Exception as e:
                print_debug(f"Error extracting main theme: {str(e)}")
                main_theme = f"Visual representation of a {mood} {genre} song"
            
            # Extract song sections
            sections = {}
            current_section = None
            section_text = []
            
            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                # Check if this is a section header
                if "VERSE" in line or "CHORUS" in line or "BRIDGE" in line or "PRE-CHORUS" in line or "OUTRO" in line:
                    # Save previous section if it exists
                    if current_section and section_text:
                        sections[current_section] = '\n'.join(section_text)
                        section_text = []
                    
                    current_section = line.strip(':')
                else:
                    # Add to current section
                    if current_section:
                        section_text.append(line)
            
            # Add final section
            if current_section and section_text:
                sections[current_section] = '\n'.join(section_text)
            
            # Build a structured response
            return {
                "success": True,
                "title": title,
                "content": content,
                "lyrics": content,  # Add the lyrics field with the same content
                "sections": sections,
                "genre": genre,
                "mood": mood,
                "purpose": purpose,
                "main_theme": main_theme,
                "custom_description": custom_description
            }
            
        except Exception as e:
            return {"success": False, "error": f"Error generating lyrics: {str(e)}"}
    
    def generate_cover_art(
        self, 
        lyrics_data: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Generate album cover art for the song.
        
        Args:
            lyrics_data: Dictionary containing song data including title, genre, mood, etc.
            
        Returns:
            Dict[str, Any]: Cover art generation results
        """
        # Enhanced validation of lyrics_data
        if not lyrics_data:
            return {"success": False, "error": "Lyrics data is not available. The generation process may be incomplete."}
            
        if not isinstance(lyrics_data, dict):
            return {"success": False, "error": f"Invalid lyrics data format: expected dictionary, got {type(lyrics_data)}"}
            
        if not lyrics_data.get("success", False):
            return {"success": False, "error": "Cannot generate cover art without valid lyrics data"}
        
        try:
            # Extract key information with fallbacks for all required fields
            title = lyrics_data.get("title", "Untitled")
            genre = lyrics_data.get("genre", "Pop")
            mood = lyrics_data.get("mood", "Happy")
            theme = lyrics_data.get("main_theme", "A colorful abstract representation of music")
            
            # Log the data we're working with for debugging
            print_debug(f"Generating cover art with: title={title}, genre={genre}, mood={mood}")
            
            # Generate cover art prompt using LLM
            detailed_prompt = self._generate_cover_art_prompt(title, genre, mood, theme)
            
            # Store the prompt for use in the HF API
            self.last_prompt = detailed_prompt
            
            # Use Hugging Face Inference API for generation
            if not self.use_hf_api:
                return {"success": False, "error": "Hugging Face API is not configured. Please set up API credentials."}
            
            try:
                # Generate image using Hugging Face API
                cover_art = self._generate_with_hf_api(genre, mood, theme, title)
                
                # Save the cover art
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename_base = f"{genre.lower()}_{mood.lower()}_{lyrics_data.get('purpose', '').lower().replace(' ', '_')}"
                cover_art_path = os.path.join(self.output_dir, f"{filename_base}_cover_{timestamp}.jpg")
                
                if cover_art:
                    cover_art.save(cover_art_path)
                    return {
                        "success": True,
                        "title": title,
                        "cover_art": cover_art,
                        "cover_art_path": cover_art_path,
                        "prompt": detailed_prompt
                    }
                else:
                    raise ValueError("Failed to generate cover art image")
            except Exception as e:
                return {"success": False, "error": f"Error generating cover art: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"Error in cover art generation: {str(e)}"}

    def _generate_cover_art_prompt(self, title: str, genre: str, mood: str, theme: str) -> str:
        """
        Generate a detailed prompt for cover art generation using LLM.
        
        Args:
            title: Song title
            genre: Music genre
            mood: Emotional mood
            theme: Visual theme from lyrics
            
        Returns:
            str: Detailed prompt for text-to-image model
        """
        # First, prepare a detailed instruction for the LLM
        llm_instruction = f"""
        Create a highly detailed prompt for an AI text-to-image model to generate album cover art for a {genre} song titled 
        "{title}" with a {mood} mood. The theme of the song is about: {theme}
        
        The prompt should:
        
        1. Include specific visual elements that represent the song's theme and mood
        2. Specify an artistic style appropriate for the {genre} genre of music
        3. Include details about composition, lighting, colors, and atmosphere
        4. Mention specific visual symbols or metaphors that connect to the song's theme
        5. Include technical specifications for high quality (like "high resolution", "detailed", etc.)
        
        The prompt should be highly detailed and descriptive, around 150-200 words, specifically optimized for 
        a text-to-image model (stabilityai/stable-diffusion-xl-base-1.0).
        IMPORTANT: Format as a direct prompt only, no explanations or meta-commentary.
        """
        
        # Generate prompt using LLM
        detailed_prompt = self.ai_service.generate_text(llm_instruction, temperature=0.7)
        
        # Add enhancement for image quality
        quality_enhancements = (
            "professional album cover design, typography, high quality, ultra detailed, 8k resolution, "
            "trending on artstation, professional photography, perfect composition"
        )
        
        # Add specific negative prompts to avoid typical issues
        self.last_negative_prompt = (
            "poor quality, blurry, distorted, disfigured, deformed, amateur, " 
            "text, watermark, signature, ugly, bad composition, multiple titles"
        )
        
        # Return the complete detailed prompt
        return f"{detailed_prompt} {quality_enhancements}"

    def _generate_with_hf_api(self, genre: str, mood: str, theme: str, title: str) -> Optional[Image.Image]:
        """
        Generate cover art using Hugging Face's Inference API.
        
        Args:
            genre: Music genre
            mood: Emotional mood
            theme: Visual theme from lyrics
            title: Song title
            
        Returns:
            Generated cover art image or None if generation failed
        """
        if not self.use_hf_api:
            raise ValueError("Hugging Face API is not configured. Please set up API credentials.")
        
        try:
            # Import necessary modules
            from PIL import ImageDraw, ImageFont
            
            # Prepare the model
            genre_info = self.genre_to_model.get(genre, self.genre_to_model["Pop"])
            model_id = genre_info["model"]
            
            # Use the LLM-generated prompt stored earlier
            prompt = self.last_prompt
            
            # Get the negative prompt
            negative_prompt = getattr(self, 'last_negative_prompt', 
                "poor quality, blurry, distorted, disfigured, bad composition")
            
            # Set up the client with the specified model
            self.hf_client.headers["x-use-cache"] = "0"  # Disable caching to get fresh results
            
            # Use text-to-image endpoint with our prompt
            result_image = self.hf_client.text_to_image(
                prompt=prompt,
                model=model_id,
                negative_prompt=negative_prompt,
                guidance_scale=8.0,  # Slightly higher for more prompt adherence
                num_inference_steps=40,  # More steps for higher quality
                width=1024,
                height=1024
            )
            
            # If the result is None, raise an error
            if result_image is None:
                raise ValueError("Hugging Face API returned None for cover art image")
                
            # Add title overlay if possible
            try:
                # Try to find a suitable font on the system
                font_paths = [
                    "arial.ttf", 
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
                    "/Library/Fonts/Arial Bold.ttf",
                    "C:\\Windows\\Fonts\\arialbd.ttf"
                ]
                
                font = None
                font_size = 60
                for path in font_paths:
                    try:
                        font = ImageFont.truetype(path, font_size)
                        break
                    except IOError:
                        continue
                
                # If we found a font, add the title
                if font:
                    # Create a copy of the image to draw on
                    img_with_title = result_image.copy()
                    draw = ImageDraw.Draw(img_with_title)
                    
                    # Determine text color based on image brightness
                    # Sample a few points to determine brightness
                    brightness_samples = []
                    for x in range(0, img_with_title.width, 100):
                        for y in range(0, img_with_title.height, 100):
                            try:
                                r, g, b = img_with_title.getpixel((x, y))[:3]
                                brightness = (r * 299 + g * 587 + b * 114) / 1000
                                brightness_samples.append(brightness)
                            except:
                                pass
                    
                    # Choose white for dark backgrounds, black for light backgrounds
                    avg_brightness = sum(brightness_samples) / len(brightness_samples) if brightness_samples else 128
                    text_color = (255, 255, 255) if avg_brightness < 128 else (0, 0, 0)
                    
                    # Add text shadow for readability
                    shadow_color = (0, 0, 0) if text_color == (255, 255, 255) else (255, 255, 255)
                    shadow_offset = 2
                    
                    # Position the text at the bottom
                    text_y = img_with_title.height - font_size * 2
                    
                    # Calculate text width to center it
                    text_width = draw.textlength(title, font=font)
                    text_x = (img_with_title.width - text_width) / 2
                    
                    # Draw shadow
                    draw.text((text_x + shadow_offset, text_y + shadow_offset), title, font=font, fill=shadow_color)
                    # Draw text
                    draw.text((text_x, text_y), title, font=font, fill=text_color)
                    
                    return img_with_title
            except Exception as e:
                print(f"Error adding title to cover art: {str(e)}. Using image without title.")
                
            return result_image
                
        except Exception as e:
            print(f"Error in Hugging Face API generation: {str(e)}")
            import traceback
            traceback.print_exc()
            raise  # Re-raise to be handled by the calling function
    
    def generate_song_package(
        self, 
        genre: str, 
        mood: str, 
        purpose: str, 
        custom_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate complete song package with lyrics and cover art.
        
        Args:
            genre: Music genre
            mood: Emotional mood
            purpose: Purpose or context for the song
            custom_description: Optional custom description for the lyrics generation
            
        Returns:
            Dict[str, Any]: Complete song package including lyrics and cover art
        """
        # Validate inputs
        if not genre or not isinstance(genre, str):
            return {"success": False, "error": "Genre is required and must be a string"}
            
        if not mood or not isinstance(mood, str):
            return {"success": False, "error": "Mood is required and must be a string"}
            
        if not purpose or not isinstance(purpose, str):
            return {"success": False, "error": "Purpose is required and must be a string"}
            
        # Convert inputs to title case for consistency
        genre = genre.title()
        mood = mood.title()
            
        if genre not in self.genres:
            return {"success": False, "error": f"Invalid genre. Supported genres: {', '.join(self.genres)}"}
        
        if mood not in self.moods:
            return {"success": False, "error": f"Invalid mood. Supported moods: {', '.join(self.moods)}"}
            
        if purpose not in self.purposes:
            return {"success": False, "error": f"Invalid purpose. Supported purposes: {', '.join(self.purposes)}"}
        
        try:
            # Generate lyrics
            print_debug(f"Generating lyrics with genre={genre}, mood={mood}, purpose={purpose}")
            lyrics_result = self.generate_lyrics(genre, mood, purpose, custom_description)
            
            if not lyrics_result:
                return {"success": False, "error": "Failed to generate lyrics: No result returned"}
                
            if not isinstance(lyrics_result, dict):
                return {"success": False, "error": f"Invalid lyrics result format: {type(lyrics_result)}"}
                
            if not lyrics_result.get("success", False):
                return lyrics_result
                
            # Ensure required fields exist in lyrics_result
            if "title" not in lyrics_result:
                lyrics_result["title"] = "Untitled"
            if "genre" not in lyrics_result:
                lyrics_result["genre"] = genre
            if "mood" not in lyrics_result:
                lyrics_result["mood"] = mood
            if "purpose" not in lyrics_result:
                lyrics_result["purpose"] = purpose
            if "content" in lyrics_result and "lyrics" not in lyrics_result:
                lyrics_result["lyrics"] = lyrics_result["content"]
                
            # Generate cover art
            cover_art_result = self.generate_cover_art(lyrics_result)
            if not cover_art_result.get("success", False):
                # If cover art fails, still return lyrics but include the error
                return {
                    "success": True,  # Overall success is still true if we have lyrics
                    "lyrics_data": lyrics_result,  # Include all lyrics data
                    "title": lyrics_result.get("title", "Untitled"),
                    "error_details": f"Failed to generate cover art: {cover_art_result.get('error', 'Unknown error')}",
                    "cover_art": None,
                    "cover_art_path": None,
                    "prompt": None
                }
                
            # Extract needed fields from nested results for a flat structure
            # This makes it easier for the frontend to access the data
            return {
                "success": True,
                "lyrics_data": lyrics_result,  # Include the entire lyrics result as lyrics_data
                "title": lyrics_result.get("title", "Untitled"),
                "cover_art": cover_art_result.get("cover_art"),  # Direct access to the PIL Image
                "cover_art_path": cover_art_result.get("cover_art_path", None),
                "prompt": cover_art_result.get("prompt", None)
            }
                
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            print_debug(f"Error in generate_song_package: {str(e)}\n{error_traceback}")
            return {
                "success": False,
                "error": f"Failed to generate song package: {str(e)}",
                "error_details": error_traceback
            } 