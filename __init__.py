"""
Codeway AI Suite - A collection of AI-powered tools.

This package contains three main AI applications:
1. Nerd AI: Math problem solver with image scanning capabilities
2. Interior Design App: Transform room styles with AI
3. Music Generator: Create personalized song lyrics and cover art
"""

# Version info
__version__ = '1.0.0'
__author__ = 'Codeway AI Team'

# Package initialization
import os
import sys
from pathlib import Path

# Ensure the project root is in path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root)) 