"""
Configuration module for the Codeway AI Suite.

This module contains shared settings and configuration for all the applications
in the suite.
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables from .env file
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# OpenAI configurations
OPENAI_MODEL = "gpt-4o-mini"  # The model specified for use
OPENAI_TEMPERATURE = 0.3  # Default temperature for deterministic outputs
OPENAI_MAX_TOKENS = 1000  # Default token limit

# Image Processing configurations
IMAGE_MAX_SIZE = (1024, 1024)  # Maximum image dimensions
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png"]

# Common API settings
API_SETTINGS = {
    "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
    "openai_api_base": os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
    "huggingface_api_token": os.environ.get("HUGGING_FACE_API_TOKEN", ""),
    "huggingface_use_api": os.environ.get("HUGGINGFACE_USE_API", "true").lower() in ["true", "1", "yes"],
    "timeout": 60,
    "max_retries": 3,
    "retry_delay": 5
}

# Hugging Face API settings
HUGGINGFACE_SETTINGS = {
    "text_to_image_models": {
        "default": "stabilityai/stable-diffusion-xl-base-1.0",
        "advanced": "black-forest-labs/FLUX.1-dev",
        "fast": "prompthero/openjourney-v4",
    },
    "image_to_image_models": {
        "default": "stabilityai/stable-diffusion-xl-refiner-1.0",
        "portrait": "runwayml/stable-diffusion-v1-5",
    },
    "rate_limits": {
        "free_tier": {
            "requests_per_minute": 5,
            "requests_per_day": 50,
        },
        "pro_tier": {
            "requests_per_minute": 30,
            "requests_per_day": 1000,
        }
    },
    "inference_steps": 30,
    "guidance_scale": 7.5,
}

# Logging settings
LOG_SETTINGS = {
    "log_level": "INFO",                      # Default log level
    "console_log_level": "INFO",              # Console output log level
    "file_log_level": "DEBUG",                # File output log level 
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log message format
    "date_format": "%Y-%m-%d %H:%M:%S",       # Timestamp format
    "log_dir": "logs",                        # Directory to store log files
    "app_log_file": "app.log",                # Main application log file
    "max_log_size_mb": 10,                    # Maximum log file size before rotation
    "backup_count": 5,                        # Number of backup logs to keep
    "module_specific_levels": {               # Module-specific log levels
        "nerd_ai": "INFO",
        "interior_design": "INFO",
        "music_generator": "INFO",
        "frontend": "INFO",
        "common": "INFO"
    }
}

# Nerd AI settings
NERD_AI_SETTINGS = {
    "supported_categories": ["algebra", "calculus", "geometry", "statistics", "linear_algebra", "physics"],
    "math_domains": ["algebra", "calculus", "geometry", "statistics", "trigonometry"],
    "ocr_confidence_threshold": 0.7,
    "show_execution_steps": True,
    "execution_timeout": 10,
    "safe_math_globals": {
        # Basic math functions
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "sorted": sorted
    }
}

INTERIOR_DESIGN_SETTINGS = {
    "supported_styles": ["Modern", "Soho", "Gothic"],
    "supported_rooms": ["Living Room", "Kitchen"],
    "image_formats": ["jpg", "jpeg", "png"],
    "max_image_size_mb": 10,
    "image_generation": {
        "resolution": (512, 512),
        "steps": 50,
        "guidance_scale": 7.5,
        "strength": 0.8,
    },
    "huggingface": {
        "models": {
            "default": "stabilityai/stable-diffusion-xl-base-1.0",
            "interior": "prompthero/openjourney-v4",
        },
        "use_api": API_SETTINGS["huggingface_use_api"],
    }
}

MUSIC_GENERATOR_SETTINGS = {
    "supported_genres": ["Pop", "Rock", "Jazz", "Hip Hop", "Classical"],
    "genres": ["Pop", "Rock", "Jazz", "Hip Hop", "Classical", "Electronic", "Country"],
    "supported_moods": ["Happy", "Sad", "Energetic", "Relaxing", "Romantic"],
    "moods": ["Happy", "Sad", "Energetic", "Calm", "Romantic", "Nostalgic", "Angry"],
    "supported_purposes": ["For celebration", "For motivation", "For my love", "For reflection", "For entertainment"],
    "purposes": ["For my love", "For my pet", "For my future self", "For celebration", "For reflection", "For motivation"],
    "lyrics_structure": ["verse", "chorus", "bridge", "outro"],
    "min_sections": 3,
    "max_sections": 8,
    "output_directory": "data/outputs/music",
    "cover_art": {
        "size": (800, 800),
        "format": "JPEG",
        "quality": 95
    },
    "huggingface": {
        "models": {
            "default": "stabilityai/stable-diffusion-xl-base-1.0",
            "album_cover": "prompthero/openjourney-v4",
        },
        "use_api": API_SETTINGS["huggingface_use_api"],
    }
} 