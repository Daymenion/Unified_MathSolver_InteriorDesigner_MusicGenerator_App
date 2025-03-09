"""
Common utilities package for the Codeway AI Suite.

This package contains shared utilities, configuration, and services used
across all applications in the suite.
"""

from .config import (
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS,
    IMAGE_MAX_SIZE, SUPPORTED_IMAGE_FORMATS,
    NERD_AI_SETTINGS, INTERIOR_DESIGN_SETTINGS, MUSIC_GENERATOR_SETTINGS
)
from .utils import (
    validate_image, preprocess_image, image_to_base64,
    format_timestamp, save_output
)
from .ai_service import OpenAIService

__all__ = [
    'OPENAI_API_KEY', 'OPENAI_MODEL', 'OPENAI_TEMPERATURE', 'OPENAI_MAX_TOKENS',
    'IMAGE_MAX_SIZE', 'SUPPORTED_IMAGE_FORMATS',
    'NERD_AI_SETTINGS', 'INTERIOR_DESIGN_SETTINGS', 'MUSIC_GENERATOR_SETTINGS',
    'validate_image', 'preprocess_image', 'image_to_base64',
    'format_timestamp', 'save_output',
    'OpenAIService'
] 