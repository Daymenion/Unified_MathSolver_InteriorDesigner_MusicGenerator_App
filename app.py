"""
Main application entry point for the Codeway AI Suite.

This application provides a Streamlit interface to three AI-powered tools:
1. Nerd AI: Math problem solver with image scanning capabilities
2. Interior Design App: Transform room styles with AI
3. Music Generator: Create personalized song lyrics and cover art
"""

import os
import sys
import traceback
from pathlib import Path

# Add project root to Python path for proper imports
try:
    project_root = Path(__file__).parent.absolute()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except Exception as e:
    print(f"Error setting Python path: {str(e)}")
    sys.exit(1)

# Import logger early
try:
    from common.logger import get_logger, log_exceptions
    logger = get_logger("app")
    logger.info("Application starting")
except Exception as e:
    print(f"Error initializing logger: {str(e)}")
    print(traceback.format_exc())
    sys.exit(1)

# Initialize Streamlit
import streamlit as st

# Set up error handling for the app
def handle_exception(exc_type, exc_value, exc_traceback):
    """Custom exception handler to log errors gracefully"""
    error_details = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    st.error(f"An error occurred in the application. Please contact support.")
    st.error(f"Error: {str(exc_value)}")
    
    # Log the error with our logging system
    logger.critical(
        f"Unhandled exception: {str(exc_value)}", 
        module="app", 
        exc_info=(exc_type, exc_value, exc_traceback)
    )
    return True

# Install the exception handler
sys.excepthook = handle_exception

# Try to import frontend modules
with log_exceptions("importing_frontend_modules", "app"):
    # Now we can import our modules
    from frontend import home
    from frontend import nerd_ai
    from frontend import interior_design 
    from frontend import music_generator
    
    logger.info("Frontend modules imported successfully")
    
    # Configure page settings
    st.set_page_config(
        page_title="Codeway AI Suite",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    logger.debug("Streamlit page configured")

    # Define pages
    PAGES = {
        "Home": home,
        "Nerd AI": nerd_ai,
        "Interior Design": interior_design,
        "Music Generator": music_generator
    }

    # Sidebar for navigation
    st.sidebar.title("Codeway AI Suite")
    
    # Safely load the logo with error handling
    with log_exceptions("loading_logo", "app"):
        logo_path = "frontend/assets/logo.png"
        if os.path.exists(logo_path):
            st.sidebar.image(logo_path, width=200)
            logger.debug(f"Logo loaded successfully from {logo_path}")
        else:
            st.sidebar.warning("Logo image not found")
            logger.warning(f"Logo image not found at path: {logo_path}")

    # Page selection
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    logger.info(f"User selected page: {selection}")

    # Footer for sidebar
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Created by Mehmet Ünlü\n\n"
        "© 2025 Daymenion"
    )

    # Display selected page with error handling
    with log_exceptions(f"rendering_page_{selection}", "app"):
        page = PAGES[selection]
        # Check for various possible render methods
        if hasattr(page, 'show'):
            logger.debug(f"Rendering page {selection} using show() method")
            page.show()
        # Fallback to render_* method if it exists
        elif hasattr(page, f'render_{selection.lower().replace(" ", "_")}'):
            method_name = f'render_{selection.lower().replace(" ", "_")}'
            logger.debug(f"Rendering page {selection} using {method_name}() method")
            render_method = getattr(page, method_name)
            render_method()
        else:
            error_msg = f"The selected page '{selection}' does not have a proper display method."
            st.error(error_msg)
            st.info("Each page module should have either a 'show()' method or a " 
                   f"'render_{selection.lower().replace(' ', '_')}()' method.")
            logger.error(error_msg, module="app")
            
logger.info("Application initialization complete") 