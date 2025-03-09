"""
Home page for the Codeway AI Suite.

This module renders the home page with general information about the suite.
"""

import os
import streamlit as st
from pathlib import Path

def safe_image_display(image_path, use_column_width=True, caption=None):
    """
    Safely display an image with fallback for missing files.
    
    Args:
        image_path: Path to the image file
        use_column_width: Whether to use the column width
        caption: Optional caption for the image
    """
    # Ensure the path is resolved correctly
    if not os.path.isabs(image_path):
        # Try to find the file relative to the current directory
        paths_to_try = [
            image_path,  # As provided
            os.path.join(os.path.dirname(__file__), image_path),  # Relative to this file
            os.path.join(os.path.dirname(os.path.dirname(__file__)), image_path)  # Relative to project root
        ]
        
        for path in paths_to_try:
            if os.path.exists(path):
                image_path = path
                break
    
    # Check if the file exists
    if os.path.exists(image_path):
        try:
            st.image(image_path, use_column_width=use_column_width, caption=caption)
            return True
        except Exception as e:
            st.warning(f"Could not display image: {str(e)}")
            return False
    else:
        # Display a placeholder instead
        st.info(f"Image not found: {os.path.basename(image_path)}")
        # Create a colored rectangle as placeholder
        import numpy as np
        from PIL import Image, ImageDraw
        
        # Create a placeholder colored rectangle
        img = Image.new('RGB', (300, 200), color=(200, 200, 200))
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, 290, 190], fill=(240, 240, 240), outline=(180, 180, 180))
        draw.text((100, 90), "Image Missing", fill=(100, 100, 100))
        
        # Display the placeholder
        st.image(img, use_column_width=use_column_width, caption=caption or "Missing image")
        return False

def show():
    """
    Render the home page for the Codeway AI Suite.
    
    This page displays general information about the AI Suite and its components.
    """
    st.title("Welcome to Codeway AI Suite")
    
    st.markdown("""
    ## All-in-One AI Toolbox
    
    The Codeway AI Suite offers three powerful AI tools to help you with various tasks:
    
    1. **Nerd AI**: Take a photo of a math problem and get a step-by-step solution.
    
    2. **Interior Design**: Transform the style of any room with just a few clicks.
    
    3. **Music Generator**: Create custom lyrics and album art for different genres and moods.
    
    Use the sidebar to navigate between the different tools.
    """)
    
    # Display features in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Nerd AI")
        safe_image_display("frontend/assets/nerd_ai_preview.png", caption="Math Problem Solver")
        st.markdown("""
        - OCR technology to read math problems
        - Support for algebra, calculus, geometry and more
        - Step-by-step solutions with explanations
        """)
    
    with col2:
        st.subheader("Interior Design")
        safe_image_display("frontend/assets/interior_design_preview.png", caption="Style Transformer")
        st.markdown("""
        - Transform rooms into different design styles
        - Realistic style conversions
        - Support for Modern, Soho, and Gothic styles
        """)
    
    with col3:
        st.subheader("Music Generator")
        safe_image_display("frontend/assets/music_generator_preview.png", caption="Music Creator")
        st.markdown("""
        - Generate lyrics for various genres and moods
        - Create matching album cover art
        - Customize themes and purposes
        """)
    
    # Getting started section
    st.markdown("---")
    st.subheader("Getting Started")
    st.markdown("""
    To get started, simply select a tool from the sidebar on the left.
    Each tool has its own interface with specific instructions on how to use it.
    
    All generated outputs are saved automatically and can be downloaded directly
    from the respective tool pages.
    """)
    
    # Footer
    st.markdown("---")
    st.caption("Codeway AI Suite - Powered by advanced AI for everyday tasks")

# For testing the module directly
if __name__ == "__main__":
    show() 