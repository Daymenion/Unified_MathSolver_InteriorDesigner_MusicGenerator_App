"""
Music Generator frontend component for the Daymenion AI Suite.
"""

import streamlit as st
import os
import tempfile
import base64
from PIL import Image
from io import BytesIO

# Use absolute imports instead of relative imports
from music_generator.generator import MusicGenerator
from common.config import MUSIC_GENERATOR_SETTINGS
from common.utils import safe_get, ensure_pil_image
from common.logger import get_logger, log_exceptions

# Initialize logger
logger = get_logger("frontend.music_generator")


def get_image_download_link(img, filename, text):
    """Generate a download link for an image."""
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}.jpg">{text}</a>'
    return href


def render_music_generator():
    """Render the Music Generator application."""
    
    st.title("🎵 Music Generator")
    
    st.markdown("""
    ## Create Personalized Song Lyrics and Cover Art
    
    Generate unique song lyrics and matching cover art based on your preferences.
    Customize the genre, mood, and purpose to create a song that's perfect for any occasion.
    """)
    
    # Initialize the generator if not already in session state
    if "music_generator" not in st.session_state:
        logger.debug("Initializing MusicGenerator in session state")
        st.session_state.music_generator = MusicGenerator()
    
    # Create a form for user inputs
    with st.form("song_generation_form"):
        # Genre selector
        genre = st.selectbox(
            "Choose a music genre",
            MUSIC_GENERATOR_SETTINGS["genres"]
        )
        
        # Mood selector
        mood = st.selectbox(
            "Choose a mood",
            MUSIC_GENERATOR_SETTINGS["moods"]
        )
        
        # Purpose selector
        purpose = st.selectbox(
            "What is this song for?",
            MUSIC_GENERATOR_SETTINGS["purposes"]
        )
        
        # Custom description
        custom_description = st.text_area(
            "Add a custom description (optional)",
            placeholder="E.g., A song about a journey through the mountains at sunset..."
        )
        
        # Submit button
        submit_button = st.form_submit_button("Generate Song")
    
    # Process the form submission
    if submit_button:
        logger.info(f"User requested song generation: genre={genre}, mood={mood}, purpose={purpose}")
        
        with st.spinner("Creating your personalized song..."):
            # Generate the song package
            result = st.session_state.music_generator.generate_song_package(
                genre, mood, purpose, custom_description
            )
            
            if result["success"]:
                logger.info("Song generated successfully")
                # Display the results
                st.success("Song successfully generated!")
                
                # Create tabs for lyrics and cover art
                lyrics_tab, cover_art_tab = st.tabs(["Lyrics", "Cover Art"])
                
                # Lyrics tab with robust error handling
                with lyrics_tab:
                    st.subheader(f"{genre} Song - {mood} Mood")
                    
                    # Check if lyrics_data exists in the result
                    if "lyrics_data" in result and isinstance(result["lyrics_data"], dict):
                        # Try to get lyrics from either the 'lyrics' or 'content' field
                        lyrics_text = None
                        
                        if "lyrics" in result["lyrics_data"]:
                            lyrics_text = result["lyrics_data"]["lyrics"]
                        elif "content" in result["lyrics_data"]:
                            lyrics_text = result["lyrics_data"]["content"]
                            
                        if lyrics_text:
                            # Display and prepare the lyrics for download
                            st.markdown(lyrics_text)
                            
                            # Create a filename for download
                            lyrics_filename = f"lyrics_{genre.lower()}_{mood.lower()}.txt"
                            
                            # Create a temporary file to download
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp:
                                tmp.write(lyrics_text.encode())
                                tmp_path = tmp.name
                            
                            with open(tmp_path, 'rb') as f:
                                st.download_button(
                                    label="Download Lyrics",
                                    data=f,
                                    file_name=lyrics_filename,
                                    mime="text/plain"
                                )
                        else:
                            # Handle missing lyrics text
                            st.warning("Lyrics text is not available in the expected format.")
                            
                            # Display any available information
                            if "title" in result["lyrics_data"]:
                                st.markdown(f"**Title:** {result['lyrics_data']['title']}")
                            if "sections" in result["lyrics_data"] and result["lyrics_data"]["sections"]:
                                st.markdown("### Song Sections:")
                                for section, text in result["lyrics_data"]["sections"].items():
                                    st.markdown(f"**{section}**")
                                    st.markdown(text)
                    else:
                        # Handle missing lyrics data
                        st.warning("Lyrics data is not available. The generation process may be incomplete.")
                        
                        # Check if we have any lyrics-related information to display
                        if "title" in result:
                            st.markdown(f"**Title:** {safe_get(result, 'title', 'Untitled')}")
                        if "error_details" in result:
                            st.error(f"Error details: {safe_get(result, 'error_details', 'No details available')}")
                            
                # Cover Art tab with error handling
                with cover_art_tab:
                    st.subheader("Album Cover Art")
                    
                    # Check if cover_art exists in the result
                    if "cover_art" in result and result["cover_art"] is not None:
                        # Convert to PIL Image if needed
                        cover_art_image = ensure_pil_image(result["cover_art"])
                        
                        if cover_art_image is not None:
                            # Display the PIL Image
                            st.image(cover_art_image, caption=f"{genre} {mood} Album Cover", use_column_width=True)
                            
                            # Save to a temporary file for download if needed
                            temp_image_path = os.path.join(tempfile.gettempdir(), f"cover_{genre.lower()}_{mood.lower()}.jpg")
                            cover_art_image.save(temp_image_path, format="JPEG")
                            
                            with open(temp_image_path, "rb") as f:
                                cover_bytes = f.read()
                                
                            st.download_button(
                                label="Download Cover Art",
                                data=cover_bytes,
                                file_name=f"cover_{genre.lower()}_{mood.lower()}.jpg",
                                mime="image/jpeg"
                            )
                        else:
                            st.warning("Could not process the cover art image.")
                            logger.warning(f"Failed to convert cover_art to PIL Image, type: {type(result['cover_art'])}")
                    else:
                        st.warning("Cover art is not available. The generation process may be incomplete.")
                        
                        # Show the prompt used if available
                        if "prompt" in result:
                            with st.expander("Show Cover Art Prompt"):
                                st.text(result["prompt"])
            else:
                # Show detailed error information
                logger.error(f"Song generation failed: {safe_get(result, 'error', 'Unknown error')}")
                st.error(f"Error: {safe_get(result, 'error', 'Unknown error occurred')}")
                
                # Show more detailed error information if available
                if "error_details" in result:
                    with st.expander("See Error Details"):
                        st.code(safe_get(result, "error_details", "No detailed error information available"))
    
    # Example songs
    st.markdown("---")
    st.subheader("Example Songs")
    
    # Create tabs for different examples
    tabs = st.tabs(["Pop - Happy", "Rock - Energetic"])
    
    # Example 1: Pop - Happy
    with tabs[0]:
        st.markdown("### Pop Song - Happy Mood")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Title: "Sunshine Day"**
            
            **Verse 1:**  
            Morning light breaks through my window  
            A brand new day is calling my name  
            I feel the rhythm in my heartbeat  
            Nothing's ever gonna be the same
            
            **Chorus:**  
            It's a sunshine day, oh yeah  
            Clouds are rolling away  
            Got that feeling inside  
            Like I'm ready to fly  
            On this sunshine day
            
            **Verse 2:**  
            People smiling as I'm walking  
            The world is painted in brighter hues  
            Every moment feels like magic  
            And I wanna share this joy with you
            
            **Chorus:**  
            It's a sunshine day, oh yeah  
            Clouds are rolling away  
            Got that feeling inside  
            Like I'm ready to fly  
            On this sunshine day
            
            **Bridge:**  
            Even when the rain comes pouring down  
            I'll keep this feeling safe and sound  
            'Cause I know that soon enough  
            The sun will shine again, no doubt
            
            **Chorus:**  
            It's a sunshine day, oh yeah  
            Clouds are rolling away  
            Got that feeling inside  
            Like I'm ready to fly  
            On this sunshine day
            """)
            
        with col2:
            st.image("https://placehold.co/400x400/FFD700/343A40?text=Sunshine+Day", caption="Cover Art")
    
    # Example 2: Rock - Energetic
    with tabs[1]:
        st.markdown("### Rock Song - Energetic Mood")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Title: "Breaking Free"**
            
            **Verse 1:**  
            Chains on my wrists, bars on my soul  
            Years of playing someone else's role  
            The pressure building, about to explode  
            Time to rewrite the story that's been told
            
            **Chorus:**  
            I'm breaking free, can't hold me down  
            Shattering these walls all around  
            Hear my voice, it's a powerful sound  
            I'm breaking free, no turning back now
            
            **Verse 2:**  
            Fire in my veins, lightning in my eyes  
            Done with all the limits and the lies  
            Standing on the edge, ready to rise  
            This is my moment, my time to fly
            
            **Chorus:**  
            I'm breaking free, can't hold me down  
            Shattering these walls all around  
            Hear my voice, it's a powerful sound  
            I'm breaking free, no turning back now
            
            **Bridge:**  
            Every step I take (Every step I take)  
            Is a chain I break (Is a chain I break)  
            No more living in fear  
            The real me is finally here
            
            **Chorus:**  
            I'm breaking free, can't hold me down  
            Shattering these walls all around  
            Hear my voice, it's a powerful sound  
            I'm breaking free, no turning back now
            """)
            
        with col2:
            st.image("https://placehold.co/400x400/FF4500/FFFFFF?text=Breaking+Free", caption="Cover Art")
    
    # How it works section
    st.markdown("---")
    st.subheader("How it works")
    
    st.markdown("""
    Our Music Generator creates personalized songs through a sophisticated process:
    
    1. **User Preferences**: We collect your preferences for genre, mood, purpose, and any custom descriptions.
    2. **Lyric Generation**: Our AI creates original lyrics that match your preferences, with proper song structure including verses, chorus, and bridge.
    3. **Theme Extraction**: Key themes and imagery are extracted from the lyrics to inform the cover art.
    4. **Style Definition**: An artistic style is determined based on the genre and mood of the song.
    5. **Cover Art Creation**: The cover art is generated to visually represent the song's theme in the appropriate artistic style.
    
    The result is a complete song package with lyrics and cover art that perfectly captures your vision!
    """) 


def show():
    """
    Show the Music Generator interface.
    
    This function is called by the main app to display the Music Generator page.
    """
    render_music_generator() 