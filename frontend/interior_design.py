"""
Interior Design frontend component for the Codeway AI Suite.
"""

import streamlit as st
import os
import tempfile
from PIL import Image
import base64
from io import BytesIO

# Use absolute imports instead of relative imports
from interior_design.style_transformer import StyleTransformer
from common.config import INTERIOR_DESIGN_SETTINGS
from common.utils import safe_get, ensure_pil_image
from common.logger import get_logger, log_exceptions

# Initialize logger
logger = get_logger("frontend.interior_design")


def get_image_download_link(img, filename, text):
    """Generate a download link for an image."""
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}.jpg">{text}</a>'
    return href


def render_interior_design():
    """Render the Interior Design application."""
    
    st.title("🏠 Interior Design Style Transformer")
    
    st.markdown("""
    ## Transform Your Space with AI
    
    Upload a photo of your room, and our AI will transform it into a different interior design style.
    Choose from a variety of styles to see how your space could look with a makeover.
    """)
    
    # Initialize the style transformer if not already in session state
    if "style_transformer" not in st.session_state:
        logger.debug("Initializing StyleTransformer in session state")
        st.session_state.style_transformer = StyleTransformer()
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Upload a photo of your room", 
        type=["jpg", "jpeg", "png"]
    )
    
    # Sample image selection
    st.markdown("### Or use a sample image")
    sample_images = ["living-room-classic.jpg", "kitchen-traditional.jpg"]
    sample_selection = st.selectbox("Select a sample room", ["None"] + sample_images)
    
    # Style selection
    st.markdown("### Choose a transformation style")
    style = st.selectbox(
        "Target style", 
        INTERIOR_DESIGN_SETTINGS["supported_styles"]
    )
    
    if uploaded_file is not None:
        logger.info(f"User uploaded file: {uploaded_file.name}")
        # Display the uploaded image
        st.image(uploaded_file, caption="Your Room", use_column_width=True)
        image_data = uploaded_file
        image_source = "upload"
    
    elif sample_selection != "None":
        logger.info(f"User selected sample image: {sample_selection}")
        # Display the sample image
        sample_path = os.path.join("frontend", "assets", sample_selection)
        if os.path.exists(sample_path):
            st.image(sample_path, caption="Sample Room", use_column_width=True)
            image_data = sample_path
            image_source = "sample"
        else:
            st.error(f"Sample image not found: {sample_path}")
            image_data = None
            image_source = None
    else:
        image_data = None
        image_source = None
    
    # Transform button
    if image_data is not None and st.button("Transform Style"):
        logger.info(f"User requested style transformation: {style}")
        with st.spinner(f"Transforming to {style} style..."):
            # Get the room type based on the image name
            if image_source == "sample":
                room_type = "Living Room" if "living" in os.path.basename(image_data).lower() else "Kitchen"
            else:
                # Default to Living Room for uploaded images
                room_type = "Living Room"
            
            # Perform the transformation
            result = st.session_state.style_transformer.transform_style(
                image_data, style, room_type
            )
            
            if result["success"]:
                logger.info("Style transformation successful")
                # Display the before and after images
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original")
                    st.image(image_data, use_column_width=True)
                
                with col2:
                    st.subheader(f"{style} Style")
                    # Safely convert to PIL Image if needed
                    transformed_image = ensure_pil_image(result["transformed_image"])
                    
                    if transformed_image is not None:
                        st.image(transformed_image, use_column_width=True)
                        
                        # Create a temporary file for downloading
                        temp_image_path = os.path.join(tempfile.gettempdir(), f"transformed_{style.lower()}.jpg")
                        transformed_image.save(temp_image_path, format="JPEG")
                        
                        with open(temp_image_path, "rb") as f:
                            st.download_button(
                                label="Download Transformed Image",
                                data=f,
                                file_name=f"transformed_{style.lower()}.jpg",
                                mime="image/jpeg"
                            )
                    else:
                        st.warning("Could not process the transformed image.")
                        logger.warning(f"Failed to convert transformed_image to PIL Image, type: {type(result['transformed_image'])}")
                
                # Transformation details
                with st.expander("See Transformation Details"):
                    st.subheader("Prompt Used")
                    st.text(safe_get(result, "prompt", "Prompt not available"))
                    
                    # Show parameters with robust error handling
                    st.subheader("Parameters")
                    
                    # Create a parameters dictionary with safe defaults
                    parameters = {
                        "style": style,
                        "room_type": safe_get(result, "room_type", "Unknown"),
                    }
                    
                    # Safely add image generation parameters using safe_get
                    parameters["guidance_scale"] = safe_get(INTERIOR_DESIGN_SETTINGS, ["image_generation", "guidance_scale"], 7.5)
                    parameters["steps"] = safe_get(INTERIOR_DESIGN_SETTINGS, ["image_generation", "steps"], 50)
                    parameters["strength"] = safe_get(INTERIOR_DESIGN_SETTINGS, ["image_generation", "strength"], 0.8)
                    
                    # Display the parameters
                    st.json(parameters)
            else:
                error_msg = safe_get(result, "error", "Unknown error occurred")
                logger.error(f"Style transformation failed: {error_msg}")
                st.error(f"Error: {error_msg}")
    
    # Example transformations
    st.markdown("---")
    st.subheader("Example Transformations")
    
    # Create tabs for different examples
    tabs = st.tabs(["Living Room Examples", "Kitchen Examples"])
    
    # Living Room Examples
    with tabs[0]:
        st.markdown("### Living Room Style Transformations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Original")
            st.image("https://placehold.co/600x400/EDEDE9/343A40?text=Living+Room", caption="Original Living Room")
            
        with col2:
            st.markdown("#### Modern Style")
            st.image("https://placehold.co/600x400/E9ECEF/343A40?text=Modern+Living+Room", caption="Modern Style")
            
        with col3:
            st.markdown("#### Gothic Style")
            st.image("https://placehold.co/600x400/212529/E9ECEF?text=Gothic+Living+Room", caption="Gothic Style")
    
    # Kitchen Examples
    with tabs[1]:
        st.markdown("### Kitchen Style Transformations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Original")
            st.image("https://placehold.co/600x400/EDEDE9/343A40?text=Kitchen", caption="Original Kitchen")
            
        with col2:
            st.markdown("#### Soho Style")
            st.image("https://placehold.co/600x400/DEB887/343A40?text=Soho+Kitchen", caption="Soho Style")
            
        with col3:
            st.markdown("#### Modern Style")
            st.image("https://placehold.co/600x400/E9ECEF/343A40?text=Modern+Kitchen", caption="Modern Style")
    
    # How it works section
    st.markdown("---")
    st.subheader("How it works")
    
    st.markdown("""
    Our Interior Design app uses advanced AI to transform your room while preserving its structure:
    
    1. **Image Analysis**: Your uploaded image is analyzed to identify the room type and key elements.
    2. **Style Definition**: Each style (Modern, Soho, Gothic) has specific characteristics defined in our system.
    3. **Prompt Generation**: A detailed prompt is created that describes how to transform your room while keeping its layout.
    4. **Style Transfer**: Our AI model transforms the image based on the prompt, applying the new style while maintaining the room's structure.
    5. **Image Enhancement**: The final image is optimized for quality and visual appeal.
    
    The transformation preserves the positions of furniture and major elements while changing colors, textures, materials, and decorative elements.
    """) 


def show():
    """
    Show the Interior Design interface.
    
    This function is called by the main app to display the Interior Design page.
    """
    render_interior_design() 