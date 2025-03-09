"""
Frontend package for the Daymenion AI Suite.

This package contains the Streamlit UI components for all three tools:
1. Nerd AI: Math problem solver with image scanning capabilities
2. Interior Design App: Transform room styles with AI
3. Music Generator: Create personalized song lyrics and cover art
"""

# Individual modules are imported directly where needed
# to avoid circular imports and dependency issues

__all__ = [
    'home',
    'nerd_ai',
    'interior_design',
    'music_generator'
] 