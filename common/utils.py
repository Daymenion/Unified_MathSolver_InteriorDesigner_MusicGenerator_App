"""
Utility functions shared across all applications in the Codeway AI Suite.
"""

import os
import base64
import re
from io import BytesIO
from datetime import datetime
from PIL import Image
import numpy as np
from typing import Union, Optional, Tuple, List, Dict, Any
import json
import shutil
import sys

from .config import SUPPORTED_IMAGE_FORMATS, IMAGE_MAX_SIZE

# Simple logging function to avoid circular imports
def print_debug(message):
    """Print a debug message to stderr."""
    print(f"DEBUG: {message}", file=sys.stderr)


def validate_image(image_file) -> Tuple[bool, str]:
    """
    Validate an uploaded image file.
    
    Args:
        image_file: File object or path to image
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    try:
        # Handle different types of image inputs
        if isinstance(image_file, str):
            # If it's a file path
            if not os.path.exists(image_file):
                return False, f"File not found: {image_file}"
            
            # Check file extension
            _, ext = os.path.splitext(image_file)
            if ext.lower() not in SUPPORTED_IMAGE_FORMATS:
                return False, f"Unsupported file format: {ext}. Supported formats: {', '.join(SUPPORTED_IMAGE_FORMATS)}"
            
            # Check file size
            file_size = os.path.getsize(image_file) / (1024 * 1024)  # Convert to MB
            if file_size > 10:
                return False, f"File too large: {file_size:.1f} MB. Maximum size: 10 MB"
            
            # Try to open the image
            try:
                img = Image.open(image_file)
                img.verify()  # Verify it's a valid image
                return True, ""
            except Exception as e:
                return False, f"Invalid image file: {str(e)}"
                
        elif hasattr(image_file, 'read'):
            # If it's a file-like object (e.g., from Streamlit)
            try:
                # Save the current position
                pos = image_file.tell()
                
                # Read the file content
                image_content = image_file.read()
                
                # Reset to the original position
                image_file.seek(pos)
                
                # Check file size
                file_size = len(image_content) / (1024 * 1024)  # Convert to MB
                if file_size > 10:
                    return False, f"File too large: {file_size:.1f} MB. Maximum size: 10 MB"
                
                # Try to open the image
                img = Image.open(BytesIO(image_content))
                img.verify()  # Verify it's a valid image
                return True, ""
            except Exception as e:
                return False, f"Invalid image file: {str(e)}"
        
        elif isinstance(image_file, Image.Image):
            # If it's already a PIL Image
            return True, ""
            
        else:
            return False, "Unsupported image input type"
            
    except Exception as e:
        return False, f"Error validating image: {str(e)}"


def preprocess_image(image_file) -> Image.Image:
    """
    Preprocess an image for AI processing.
    
    Args:
        image_file: File object or path to image
        
    Returns:
        PIL.Image.Image: Preprocessed image
    """
    # Handle different types of image inputs
    if isinstance(image_file, str):
        # If it's a file path
        img = Image.open(image_file)
    elif hasattr(image_file, 'read'):
        # If it's a file-like object
        image_content = image_file.read() if hasattr(image_file, 'read') else image_file
        img = Image.open(BytesIO(image_content))
    elif isinstance(image_file, Image.Image):
        # If it's already a PIL Image
        img = image_file
    else:
        raise ValueError("Unsupported image input type")
    
    # Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Resize if too large
    if img.width > IMAGE_MAX_SIZE[0] or img.height > IMAGE_MAX_SIZE[1]:
        img.thumbnail(IMAGE_MAX_SIZE)
    
    return img


def image_to_base64(image: Union[str, Image.Image]) -> str:
    """
    Convert an image to a base64 string.
    
    Args:
        image: PIL Image object or path to image file
        
    Returns:
        str: Base64 encoded image string
    """
    if isinstance(image, str):
        # If it's a file path
        with open(image, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    elif isinstance(image, Image.Image):
        # If it's a PIL Image
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        raise ValueError("Unsupported image input type")


def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """
    Format a timestamp for file naming.
    
    Args:
        timestamp: Datetime object (defaults to current time)
    
    Returns:
        str: Formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime("%Y%m%d_%H%M%S")


def save_output(output_data: Dict[str, Any], output_dir: str, prefix: str) -> Dict[str, str]:
    """
    Save output data to files.
    
    Args:
        output_data: Dictionary containing output data
        output_dir: Directory to save files
        prefix: Prefix for filenames
        
    Returns:
        Dict[str, str]: Dictionary mapping output types to file paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filenames directly
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize result dictionary
    result_files = {}
    
    # Save text output if present
    if "text" in output_data:
        text_path = os.path.join(output_dir, f"{prefix}_text_{timestamp}.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(output_data["text"])
        result_files["text"] = text_path
    
    # Save image output if present
    if "image" in output_data and output_data["image"] is not None:
        image_path = os.path.join(output_dir, f"{prefix}_image_{timestamp}.jpg")
        if isinstance(output_data["image"], Image.Image):
            output_data["image"].save(image_path)
        elif isinstance(output_data["image"], str):
            # If it's a base64 string
            img_data = base64.b64decode(output_data["image"])
            with open(image_path, "wb") as f:
                f.write(img_data)
        result_files["image"] = image_path
    
    # Save JSON data if present
    if "json" in output_data:
        json_path = os.path.join(output_dir, f"{prefix}_data_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_data["json"], f, indent=2)
        result_files["json"] = json_path
    
    return result_files


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to ensure it's valid across operating systems.
    
    Args:
        filename: Original filename
        
    Returns:
        str: Sanitized filename
    """
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[\\/*?:"<>|]', '_', filename)
    
    # Remove leading/trailing spaces and periods
    sanitized = sanitized.strip('. ')
    
    # Ensure the filename is not empty
    if not sanitized:
        sanitized = "unnamed_file"
    
    return sanitized


def create_directory(directory_path: str) -> str:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        str: Path to the created directory
    """
    # Create the directory if it doesn't exist
    os.makedirs(directory_path, exist_ok=True)
    return directory_path


def safe_get(data: Dict[str, Any], keys: Union[str, List[str]], default: Any = None) -> Any:
    """
    Safely access a value in a nested dictionary with a default fallback.
    
    Args:
        data: Dictionary to access
        keys: String key or list of keys for nested access
        default: Default value to return if key doesn't exist
        
    Returns:
        Any: The value if found, otherwise the default value
    """
    if not isinstance(data, dict):
        return default
        
    if isinstance(keys, str):
        return data.get(keys, default)
    
    # Handle nested keys
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    
    return current


def ensure_pil_image(image_data):
    """
    Ensure that the provided image data is converted to a PIL Image object.
    
    Args:
        image_data: Can be PIL Image, file path, bytes, BytesIO, numpy array, or dictionary with image data
        
    Returns:
        PIL.Image.Image or None: PIL Image object if conversion is successful, None otherwise
    """
    try:
        # If already a PIL Image, return it
        if isinstance(image_data, Image.Image):
            return image_data
            
        # If it's a dictionary, try to extract image data
        if isinstance(image_data, dict):
            # Common keys that might contain the image
            possible_keys = ['image', 'img', 'data', 'content', 'base64', 'bytes', 'file', 'pil_image']
            
            # Try each key
            for key in possible_keys:
                if key in image_data and image_data[key] is not None:
                    # Recursively try to convert this value to a PIL Image
                    try:
                        return ensure_pil_image(image_data[key])
                    except:
                        continue
                        
            # If we have a path, try loading it
            if 'path' in image_data and os.path.exists(image_data['path']):
                return Image.open(image_data['path'])
                
            # Log warning with available keys
            print_debug(f"Failed to extract image from dictionary with keys: {list(image_data.keys())}")
            return None
            
        # If it's a file path
        if isinstance(image_data, str) and os.path.exists(image_data):
            return Image.open(image_data)
            
        # If it's bytes or BytesIO
        if isinstance(image_data, (bytes, BytesIO)):
            if isinstance(image_data, bytes):
                image_data = BytesIO(image_data)
            return Image.open(image_data)
            
        # If it's a numpy array
        if 'numpy' in str(type(image_data)):
            try:
                import numpy as np
                if isinstance(image_data, np.ndarray):
                    return Image.fromarray(image_data)
            except (ImportError, ValueError) as e:
                print_debug(f"Error converting numpy array to PIL Image: {e}")
                
        # If all else fails, try a direct conversion
        try:
            return Image.open(image_data)
        except Exception as e:
            print_debug(f"Could not convert to PIL Image: {e}")
            return None
            
    except Exception as e:
        print_debug(f"Error in ensure_pil_image: {e}")
        return None


def format_math_notation(text):
    """
    Format mathematical notation for proper rendering in Streamlit.
    
    This function converts various math notation formats to proper LaTeX format
    that can be rendered correctly by Streamlit's markdown.
    
    Args:
        text (str): Text containing math notation
        
    Returns:
        str: Text with properly formatted LaTeX math notation
    """
    import re
    
    if text is None or not text:
        return ""
    
    # Clean up the text first
    # Replace problematic characters that can interfere with regex patterns
    text = text.replace('\\\\', '\\')  # Double backslashes to single
    
    # Handle display math expressions with square brackets
    text = re.sub(r'\[(.*?)\]', r'$$\1$$', text, flags=re.DOTALL)
    
    # Process square bracket notation differently if it contains math
    text = re.sub(r'\\int_\{([^}]+)\}\^\{([^}]+)\}', r'\\int_{\1}^{\2}', text)
    
    # Replace bad formatting in fractions
    text = re.sub(r'\\frac\{1\}\{([^}]+)\^([^}]+)\}', r'\\frac{1}{\\1^\2}', text)
    
    # Fix common math notation issues
    text = text.replace('x−2', 'x-2')
    text = text.replace('\\ ,', ',')
    text = text.replace('\\$', '$')
    
    # Ensure proper spacing around display math
    text = re.sub(r'([^\n])(\$\$)', r'\1\n\n\2', text)
    text = re.sub(r'(\$\$)([^\n])', r'\1\n\n\2', text)
    
    # Fix any instances where x - 2^{3} is improperly formatted
    text = re.sub(r'(x\s*-\s*2)\^([0-9]+)', r'\1^{\2}', text)
    
    return text


def format_math_display(solution_text):
    """
    Format a math solution output for better display in Streamlit.
    
    This function enhances a math solution by:
    1. Adding proper formatting for sections
    2. Ensuring math notation is properly styled
    3. Making the overall presentation more readable
    
    Args:
        solution_text (str): The raw solution text
        
    Returns:
        str: Formatted solution text ready for display
    """
    import re
    
    if not solution_text:
        return ""
    
    # Step 1: Protect all LaTeX blocks by replacing them with placeholders
    # so they won't be affected by later HTML formatting
    all_math = []
    display_math_pattern = r'(\$\$[^\$]+\$\$)'
    inline_math_pattern = r'(\$[^\$]+\$)'
    
    # Function to replace math with placeholders
    def protect_math(text, pattern, prefix):
        matches = re.findall(pattern, text, re.DOTALL)
        for i, match in enumerate(matches):
            placeholder = f"__{prefix}_MATH_{i}__"
            all_math.append((placeholder, match))
            text = text.replace(match, placeholder, 1)
        return text
    
    # Protect display math first, then inline math
    solution_text = protect_math(solution_text, display_math_pattern, "DISPLAY")
    solution_text = protect_math(solution_text, inline_math_pattern, "INLINE")
    
    # Step 2: Format the text with HTML
    # Format section headers
    solution_text = re.sub(
        r'(?m)^(#+)\s+(.+?)$',
        r'<div class="math-section-header" style="margin-top: 20px; margin-bottom: 10px; padding: 8px; background-color: #f0f0f0; border-radius: 5px; font-weight: bold;">\2</div>',
        solution_text
    )
    
    # Format steps - match patterns like "Step 1:" or "1."
    solution_text = re.sub(
        r'(?:Step\s*)?(\d+)[\.:]?\s+(.+?)(?=(?:\r?\n)+(?:Step\s*)?\d+[\.:]|\r?\n\r?\n|$)',
        r'<div class="math-step" style="margin-bottom: 15px; padding: 10px; border-left: 3px solid #4CAF50; background-color: #f9f9f9;"><span style="font-weight: bold;">Step \1:</span> \2</div>',
        solution_text,
        flags=re.DOTALL
    )
    
    # Handle final answer/conclusion section
    solution_text = re.sub(
        r'(?i)(Final Answer|Conclusion)([^\n]*(?:\n(?!\n)[^\n]*)*)',
        r'<div class="math-conclusion" style="margin-top: 25px; margin-bottom: 15px; padding: 15px; background-color: #e3f2fd; border-radius: 5px; border-left: 5px solid #2196F3;"><div style="font-weight: bold; margin-bottom: 8px;">\1</div>\2</div>',
        solution_text
    )
    
    # Step 3: Restore the math expressions
    for placeholder, math in all_math:
        solution_text = solution_text.replace(placeholder, math)
    
    # Step 4: Format the entire solution
    solution_text = f'<div class="math-solution" style="font-family: system-ui, -apple-system, sans-serif; line-height: 1.6;">{solution_text}</div>'
    
    return solution_text


def format_math_problem(problem_text):
    """
    Format a math problem for clear display in Streamlit.
    
    This function takes a raw math problem text and formats it
    for optimal display, ensuring proper LaTeX rendering.
    
    Args:
        problem_text (str): The raw problem text
        
    Returns:
        str: Formatted problem text ready for display
    """
    import re
    
    if not problem_text:
        return ""
    
    # First apply basic math notation formatting
    formatted_text = format_math_notation(problem_text)
    
    # If we don't have LaTeX delimiters, wrap the content in math delimiters
    if '$' not in formatted_text:
        # Check if it's likely to be a math expression
        if any(c in formatted_text for c in '+-*/=^_{}\\'):
            formatted_text = f"$${formatted_text}$$"
    
    # Wrap in a container with styling for problem statements
    formatted_text = f'''
    <div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px; border-left: 5px solid #673AB7; margin-bottom: 20px; font-family: system-ui, -apple-system, sans-serif;">
        {formatted_text}
    </div>
    '''
    
    return formatted_text 