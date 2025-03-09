"""
Path setup utilities to ensure consistent imports and file access.

This module provides functions to set up the Python path for proper imports
and to locate resources within the project structure.
"""

import os
import sys
from pathlib import Path
import logging
from typing import Optional, Union, List

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('path_setup')

def get_project_root() -> Path:
    """
    Get the absolute path to the project root directory.
    
    Returns:
        Path: The absolute path to the project root
    """
    # Start from the module's directory and go up until finding the project root
    current_dir = Path(__file__).parent
    while not (current_dir / '__init__.py').exists() or not (current_dir.parent / '__init__.py').exists():
        parent = current_dir.parent
        if parent == current_dir:  # Reached the filesystem root without finding project root
            # Fall back to the directory containing this file's package
            return Path(__file__).parents[1].absolute()
        current_dir = parent
    return current_dir.absolute()

def setup_python_path() -> bool:
    """
    Ensure the project root is in the Python path for imports.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        project_root = get_project_root()
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
            logger.info(f"Python path set to: {project_root}")
        return True
    except Exception as e:
        logger.error(f"Error setting Python path: {str(e)}")
        return False

def resolve_path(relative_path: Union[str, Path]) -> Path:
    """
    Resolve a path relative to the project root.
    
    Args:
        relative_path: Path relative to the project root
        
    Returns:
        Path: Absolute path
    """
    return get_project_root() / relative_path

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure
        
    Returns:
        Path: Absolute path to the directory
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory.absolute()

# Initialize the paths when imported
setup_python_path() 