"""
Showcase script for demonstrating the Daymenion AI Suite capabilities.

This script runs a demonstration of all three AI tools:
1. Nerd AI: Math problem solver with image scanning capabilities
2. Interior Design App: Transform room styles with AI
3. Music Generator: Create personalized song lyrics and cover art
"""

import os
import sys
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Ensure the project root is in the Python path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('showcase')

# Now import our modules
try:
    from nerd_ai.solver import MathSolver
    from interior_design.style_transformer import StyleTransformer
    from music_generator.generator import MusicGenerator
    from common.utils import create_directory, save_output
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    logger.error("Make sure you're running from the project root directory")
    sys.exit(1)
    
def setup_directories():
    """Ensure output directories exist."""
    os.makedirs("data/outputs/nerd_ai", exist_ok=True)
    os.makedirs("data/outputs/interior_design", exist_ok=True)
    os.makedirs("data/outputs/music", exist_ok=True)
    logger.info("Output directories confirmed")

def showcase_nerd_ai():
    """Demonstrate Nerd AI math problem solver."""
    logger.info("Starting Nerd AI showcase...")
    
    # Initialize solver
    solver = MathSolver()
    
    # Process each math problem image
    input_dir = "data/inputs/nerd_ai"
    math_problems = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not math_problems:
        logger.warning(f"No math problems found in {input_dir} directory")
        return
    
    # Create output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/outputs/nerd_ai/math_solutions_{timestamp}.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("NERD AI MATH PROBLEM SOLUTIONS\n")
        f.write("==============================\n\n")
        
        for problem_file in math_problems:
            logger.info(f"Processing math problem: {problem_file}")
            problem_path = os.path.join(input_dir, problem_file)
            
            try:
                # Process the image
                result = solver.solve_from_image(problem_path)
                
                # Write to output file
                f.write(f"Problem: {problem_file}\n")
                f.write(f"OCR Text: {result['problem_text']}\n")
                f.write(f"Problem Type: {result['problem_type']}\n")
                
                # Handle the solution structure correctly
                if isinstance(result['solution'], dict):
                    f.write(f"Solution Code:\n{result['solution']['code']}\n\n")
                    f.write(f"Explanation:\n{result['solution']['explanation']}\n\n")
                else:
                    f.write(f"Solution:\n{result['solution']}\n\n")
                    if 'code' in result:
                        f.write(f"Code:\n{result['code']}\n\n")
                
                f.write("-------------------------------\n\n")
                
                logger.info(f"Solved problem: {problem_file}")
            except Exception as e:
                logger.error(f"Error processing {problem_file}: {str(e)}")
                f.write(f"Error processing {problem_file}: {str(e)}\n\n")
    
    logger.info(f"Nerd AI showcase complete. Results saved to {output_file}")
    return output_file

def showcase_interior_design():
    """Demonstrate Interior Design room style transformer."""
    logger.info("Starting Interior Design showcase...")
    
    # Initialize transformer
    transformer = StyleTransformer()
    
    # Check for room images
    input_dir = "data/inputs/interior_design"
    room_images = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not room_images:
        logger.warning(f"No room images found in {input_dir} directory")
        return
    
    # Process each room with different styles
    styles = ["First-Ages", "Scifi", "Medieval", "Japanese"]
    results = []
    
    for room_file in room_images:
        room_path = os.path.join(input_dir, room_file)
        room_name = os.path.splitext(room_file)[0]
        
        for style in styles:
            try:
                logger.info(f"Transforming {room_file} to {style} style")
                
                # Transform the room
                result = transformer.transform_style(room_path, style)
                
                # Save the transformed image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"data/outputs/interior_design/{room_name}_{style.lower()}_{timestamp}.jpg"
                
                # Save the image
                result["transformed_image"].save(output_path)
                
                # Save the prompt for reference
                prompt_path = f"data/outputs/interior_design/{room_name}_{style.lower()}_{timestamp}_prompt.txt"
                with open(prompt_path, "w", encoding="utf-8") as f:
                    f.write(f"Room Type: {result['room_type']}\n")
                    f.write(f"Style: {style}\n\n")
                    f.write(f"Transformation Prompt:\n{result['prompt']}")
                
                results.append({
                    "room": room_file,
                    "style": style,
                    "output_path": output_path,
                    "prompt_path": prompt_path
                })
                
                logger.info(f"Transformed {room_file} to {style} style. Saved to {output_path}")
            except Exception as e:
                logger.error(f"Error transforming {room_file} to {style}: {str(e)}")
    
    logger.info(f"Interior Design showcase complete. Generated {len(results)} transformations")
    return results

def showcase_music_generator():
    """Demonstrate Music Generator for lyrics and cover art."""
    logger.info("Starting Music Generator showcase...")
    
    # Initialize generator
    generator = MusicGenerator()
    
    # Define different combinations to showcase
    combinations = [
        {"genre": "Pop", "mood": "Happy", "purpose": "For celebration", "description": "A graduation party"},
        {"genre": "Rock", "mood": "Energetic", "purpose": "For motivation", "description": "Overcoming challenges"},
        {"genre": "Jazz", "mood": "Romantic", "purpose": "For my love", "description": "Anniversary celebration"}
    ]
    
    results = []
    
    for combo in combinations:
        try:
            logger.info(f"Generating {combo['genre']} song with {combo['mood']} mood for {combo['purpose']}")
            
            # Generate the song package
            result = generator.generate_song_package(
                genre=combo["genre"],
                mood=combo["mood"],
                purpose=combo["purpose"],
                custom_description=combo["description"]
            )
            
            # Check if the operation was successful
            if not result.get("success", False):
                logger.error(f"Failed to generate song: {result.get('error', 'Unknown error')}")
                continue
                
            # Get the lyrics and cover art data from the new structured response
            lyrics = result.get("lyrics", {})
            cover_art = result.get("cover_art", {})
            metadata = result.get("metadata", {})
            
            # Create a descriptive name
            name_base = f"{combo['genre'].lower()}_{combo['mood'].lower()}_{combo['purpose'].replace(' ', '_').lower()}"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save the lyrics
            lyrics_path = f"data/outputs/music/{name_base}_lyrics_{timestamp}.txt"
            with open(lyrics_path, "w", encoding="utf-8") as f:
                f.write(f"SONG: {lyrics.get('title', 'Untitled')}\n")
                f.write(f"Genre: {combo['genre']}, Mood: {combo['mood']}, Purpose: {combo['purpose']}\n\n")
                
                # Write structured lyrics with sections if available
                if lyrics.get("sections"):
                    for section_name, section_content in lyrics.get("sections", {}).items():
                        f.write(f"{section_name}:\n{section_content}\n\n")
                else:
                    # Fallback to full content
                    f.write(lyrics.get("content", "No lyrics generated"))
            
            # Save the cover art if available
            cover_art_path = f"data/outputs/music/{name_base}_cover_{timestamp}.jpg"
            if cover_art.get("image"):
                cover_art.get("image").save(cover_art_path)
            
            # Save the theme description
            theme_path = f"data/outputs/music/{name_base}_theme_{timestamp}.txt"
            with open(theme_path, "w", encoding="utf-8") as f:
                f.write(f"SONG THEME: {lyrics.get('title', 'Untitled')}\n")
                f.write(f"Genre: {combo['genre']}, Mood: {combo['mood']}, Purpose: {combo['purpose']}\n\n")
                f.write(f"Theme: {cover_art.get('theme', 'No theme specified')}\n\n")
                f.write(f"Description: {combo['description']}\n\n")
                
                # Add extra metadata if available
                if metadata:
                    f.write(f"Package ID: {metadata.get('package_id', '')}\n")
                    f.write(f"Timestamp: {metadata.get('timestamp', '')}\n")
            
            results.append({
                "genre": combo["genre"],
                "mood": combo["mood"],
                "purpose": combo["purpose"],
                "lyrics_path": lyrics_path,
                "cover_art_path": cover_art_path,
                "theme_path": theme_path
            })
            
            logger.info(f"Generated {combo['genre']} song. Lyrics saved to {lyrics_path}, cover art to {cover_art_path}")
        except Exception as e:
            logger.error(f"Error generating {combo['genre']} song: {str(e)}")
    
    logger.info(f"Music Generator showcase complete. Generated {len(results)} songs")
    return results

def main():
    """Run the complete showcase."""
    logger.info("Starting Daymenion AI Suite showcase...")
    
    # Setup directories
    setup_directories()
    
    # Run showcases
    try:
        # Nerd AI
        math_output = showcase_nerd_ai()
        print(f"\nNerd AI showcase complete. Results saved to: {math_output}\n")
        
        # Interior Design
        design_outputs = showcase_interior_design()
        if design_outputs:
            print("\nInterior Design showcase complete. Transformed rooms:")
            for output in design_outputs:
                print(f"- {output['room']} to {output['style']} style: {output['output_path']}")
        print()
        
        # Music Generator
        music_outputs = showcase_music_generator()
        if music_outputs:
            print("\nMusic Generator showcase complete. Generated songs:")
            for output in music_outputs:
                print(f"- {output['genre']} {output['mood']} song for {output['purpose']}")
                print(f"  Lyrics: {output['lyrics_path']}")
                print(f"  Cover Art: {output['cover_art_path']}")
        print()
        
        logger.info("Showcase complete! All outputs saved to data/outputs directory")
        print("\nShowcase complete! All outputs saved to data/outputs directory")
    except Exception as e:
        logger.error(f"Error during showcase: {str(e)}")
        print(f"Error during showcase: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 