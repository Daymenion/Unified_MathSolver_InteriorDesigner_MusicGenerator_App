"""
Showcase script for demonstrating the Daymenion AI Suite capabilities.

This script runs a demonstration of all three AI tools:
1. Nerd AI: Math problem solver with image scanning capabilities
2. Interior Design App: Transform room styles with AI
3. Music Generator: Create personalized song lyrics and cover art

The script reads inputs from the data/inputs directory and writes outputs
to the data/outputs directory, with each application using its respective subdirectories.
"""

import os
import sys
import logging
import time
import json
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
    from common.utils import create_directory, save_output, print_debug
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    logger.error("Make sure you're running from the project root directory")
    sys.exit(1)

# Define input and output directories
INPUT_DIR = os.path.join(project_root, "data", "inputs")
OUTPUT_DIR = os.path.join(project_root, "data", "outputs")

def setup_directories():
    """Ensure input and output directories exist."""
    # Check input directories
    input_dirs = {
        'nerd_ai': os.path.join(INPUT_DIR, "nerd_ai"),
        'interior_design': os.path.join(INPUT_DIR, "interior_design"),
        'music': os.path.join(INPUT_DIR, "music")
    }
    
    # Check output directories
    output_dirs = {
        'nerd_ai': os.path.join(OUTPUT_DIR, "nerd_ai"),
        'interior_design': os.path.join(OUTPUT_DIR, "interior_design"),
        'music': os.path.join(OUTPUT_DIR, "music")
    }
    
    # Create missing directories
    for name, dir_path in input_dirs.items():
        os.makedirs(dir_path, exist_ok=True)
        if not os.path.exists(dir_path):
            logger.warning(f"Could not create input directory: {dir_path}")
    
    for name, dir_path in output_dirs.items():
        os.makedirs(dir_path, exist_ok=True)
        if not os.path.exists(dir_path):
            logger.warning(f"Could not create output directory: {dir_path}")
    
    logger.info("Input and output directories confirmed")
    return input_dirs, output_dirs

def showcase_nerd_ai(input_dir, output_dir):
    """Demonstrate Nerd AI math problem solver using input images."""
    logger.info("\n" + "=" * 50)
    logger.info("SHOWCASING NERD AI: MATH PROBLEM SOLVER")
    logger.info("=" * 50)
    
    # Initialize solver
    solver = MathSolver()
    
    # Process each math problem image
    math_problems = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not math_problems:
        logger.warning(f"No math problems found in {input_dir} directory")
        return None
    
    # Create output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"math_solutions_{timestamp}.txt")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("NERD AI MATH PROBLEM SOLUTIONS\n")
        f.write("==============================\n\n")
        
        for i, problem_file in enumerate(math_problems, 1):
            problem_path = os.path.join(input_dir, problem_file)
            f.write(f"Problem {i}: {problem_file}\n")
            f.write("-" * 50 + "\n\n")
            
            try:
                # Start timing
                start_time = time.time()
                
                # Process the image (OCR)
                logger.info(f"Processing problem {i}: {problem_file}")
                problem_text = solver.ocr_math_problem(problem_path)
                
                # Identify the type of problem
                problem_type = solver.identify_problem_type(problem_text)
                
                # Solve the problem
                solution = solver.generate_solution(problem_text, problem_type)
                
                # End timing
                elapsed_time = time.time() - start_time
                
                # Write to output file
                f.write(f"Problem Text: {problem_text}\n\n")
                f.write(f"Problem Type: {problem_type}\n\n")
                f.write(f"Solution Code:\n{solution['code']}\n\n")
                f.write(f"Explanation:\n{solution['explanation']}\n\n")
                f.write(f"Solved in {elapsed_time:.2f} seconds\n\n")
                f.write("=" * 50 + "\n\n")
                
                logger.info(f"Solved problem {i} ({problem_type}) in {elapsed_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Error processing {problem_file}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                f.write(f"Error: Could not solve problem - {str(e)}\n\n")
                f.write("=" * 50 + "\n\n")
    
    logger.info(f"Nerd AI showcase complete. Results saved to {output_file}")
    return output_file

def showcase_interior_design(input_dir, output_dir):
    """Demonstrate Interior Design room style transformer using input images."""
    logger.info("\n" + "=" * 50)
    logger.info("SHOWCASING INTERIOR DESIGN APP: STYLE TRANSFORMER")
    logger.info("=" * 50)
    
    # Initialize transformer
    transformer = StyleTransformer()
    
    # Check for room images
    room_images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not room_images:
        logger.warning(f"No room images found in {input_dir} directory")
        return []
    
    # Process each room with different styles
    styles = ["Modern", "Industrial", "Scandinavian", "Minimalist"]
    results = []
    
    for room_file in room_images:
        room_path = os.path.join(input_dir, room_file)
        room_name = os.path.splitext(room_file)[0]
        
        for style in styles:
            try:
                logger.info(f"Transforming {room_file} to {style} style")
                
                # Transform the room
                start_time = time.time()
                result = transformer.transform_style(room_path, style)
                elapsed_time = time.time() - start_time
                
                if result.get("success", False) and result.get("transformed_image"):
                    # Save the transformed image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_filename = f"{room_name}_{style.lower()}_{timestamp}.jpg"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # Save the image
                    result["transformed_image"].save(output_path)
                    
                    # Save the prompt for reference
                    prompt_filename = f"{room_name}_{style.lower()}_{timestamp}_prompt.txt"
                    prompt_path = os.path.join(output_dir, prompt_filename)
                    
                    with open(prompt_path, "w", encoding="utf-8") as f:
                        f.write(f"Room: {room_file}\n")
                        f.write(f"Style: {style}\n")
                        f.write(f"Prompt: {result.get('prompt', 'No prompt available')}\n")
                    
                    logger.info(f"Transformed {room_file} to {style} style in {elapsed_time:.2f} seconds")
                    
                    results.append({
                        "room": room_file,
                        "style": style,
                        "output_path": output_path,
                        "prompt_path": prompt_path,
                        "time_taken": elapsed_time
                    })
                else:
                    logger.error(f"Failed to transform {room_file} to {style} style: {result.get('error', 'Unknown error')}")
            
            except Exception as e:
                logger.error(f"Error transforming {room_file} to {style} style: {str(e)}")
    
    logger.info(f"Interior Design showcase complete. Generated {len(results)} transformations")
    return results

def showcase_music_generator(input_dir, output_dir):
    """Demonstrate Music Generator for lyrics and cover art using input descriptions."""
    logger.info("\n" + "=" * 50)
    logger.info("SHOWCASING MUSIC GENERATOR: LYRICS AND COVER ART")
    logger.info("=" * 50)
    
    # Initialize generator
    generator = MusicGenerator()
    
    # Get the supported purposes from the music generator
    supported_purposes = generator.purposes if hasattr(generator, 'purposes') else [
        "For celebration", "For motivation", "For my love", "For reflection", "For entertainment"
    ]
    
    # Look for input files with descriptions
    input_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.txt')]
    
    # Define default combinations to showcase if no inputs found
    combinations = []
    
    # Read input files for custom song descriptions
    if input_files:
        for input_file in input_files:
            try:
                with open(os.path.join(input_dir, input_file), 'r', encoding='utf-8') as f:
                    description = f.read().strip()
                
                # Use the first supported purpose
                purpose = supported_purposes[0] if supported_purposes else "For my love"
                
                # Create two variations for each description
                combinations.append({
                    "genre": "Pop", 
                    "mood": "Happy", 
                    "purpose": purpose, 
                    "description": description
                })
                
                combinations.append({
                    "genre": "Rock", 
                    "mood": "Energetic", 
                    "purpose": purpose, 
                    "description": description
                })
            except Exception as e:
                logger.error(f"Error reading input file {input_file}: {str(e)}")
    
    # Add default combinations if none were found
    if not combinations:
        logger.warning(f"No description files found in {input_dir} directory. Using default settings.")
        valid_purpose = supported_purposes[0] if supported_purposes else "For my love"
        combinations = [
            {"genre": "Pop", "mood": "Happy", "purpose": valid_purpose, "description": "A graduation party"},
            {"genre": "Rock", "mood": "Energetic", "purpose": valid_purpose, "description": "Overcoming challenges"},
            {"genre": "Jazz", "mood": "Romantic", "purpose": valid_purpose, "description": "Anniversary celebration"}
        ]
    
    results = []
    
    for i, combo in enumerate(combinations, 1):
        try:
            logger.info(f"Generating {combo['genre']} song with {combo['mood']} mood for '{combo['purpose']}'")
            
            # Generate the song package
            start_time = time.time()
            result = generator.generate_song_package(
                genre=combo["genre"],
                mood=combo["mood"],
                purpose=combo["purpose"],
                custom_description=combo["description"]
            )
            elapsed_time = time.time() - start_time
            
            # Check if the operation was successful
            if not result.get("success", False):
                logger.error(f"Failed to generate song: {result.get('error', 'Unknown error')}")
                continue
            
            # Create output filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{combo['genre'].lower()}_{combo['mood'].lower()}_{i}_{timestamp}"
            
            # Save lyrics
            lyrics_filename = f"{base_filename}_lyrics.txt"
            lyrics_path = os.path.join(output_dir, lyrics_filename)
            
            with open(lyrics_path, "w", encoding="utf-8") as f:
                f.write(f"Title: {result.get('title', 'Untitled')}\n\n")
                f.write(f"Genre: {combo['genre']}\n")
                f.write(f"Mood: {combo['mood']}\n")
                f.write(f"Purpose: {combo['purpose']}\n\n")
                
                if "lyrics_data" in result and "content" in result["lyrics_data"]:
                    f.write(result["lyrics_data"]["content"])
                elif "lyrics_data" in result and "lyrics" in result["lyrics_data"]:
                    f.write(result["lyrics_data"]["lyrics"])
                else:
                    f.write("No lyrics content available.")
            
            # Save cover art if available
            cover_art_path = None
            if "cover_art" in result and result["cover_art"]:
                cover_art_filename = f"{base_filename}_cover.jpg"
                cover_art_path = os.path.join(output_dir, cover_art_filename)
                result["cover_art"].save(cover_art_path)
            
            # Save theme/prompt information
            theme_filename = f"{base_filename}_theme.txt"
            theme_path = os.path.join(output_dir, theme_filename)
            
            with open(theme_path, "w", encoding="utf-8") as f:
                f.write(f"Song: {result.get('title', 'Untitled')}\n\n")
                f.write(f"Main Theme: {result['lyrics_data'].get('main_theme', 'No theme available')}\n\n")
                f.write(f"Prompt Used for Cover Art: {result.get('prompt', 'No prompt available')}\n")
            
            # Record the result
            results.append({
                "genre": combo["genre"],
                "mood": combo["mood"],
                "purpose": combo["purpose"],
                "elapsed_time": elapsed_time,
                "title": result.get("title", "Untitled"),
                "lyrics_path": lyrics_path,
                "cover_art_path": cover_art_path,
                "theme_path": theme_path
            })
            
            logger.info(f"Generated {combo['genre']} song '{result.get('title', 'Untitled')}' in {elapsed_time:.2f} seconds")
            logger.info(f"Lyrics saved to {lyrics_path}")
            if cover_art_path:
                logger.info(f"Cover art saved to {cover_art_path}")
        
        except Exception as e:
            logger.error(f"Error generating {combo['genre']} song: {str(e)}")
    
    logger.info(f"Music Generator showcase complete. Generated {len(results)} songs")
    return results

def main():
    """Run the complete showcase with inputs from data/inputs and outputs to data/outputs."""
    logger.info("Starting Daymenion AI Suite showcase...")
    
    # Setup directories and get paths
    input_dirs, output_dirs = setup_directories()
    
    # Create showcase summary file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(OUTPUT_DIR, f"showcase_summary_{timestamp}.txt")
    
    # Run showcases
    showcase_results = {
        "timestamp": timestamp,
        "nerd_ai": None,
        "interior_design": None,
        "music_generator": None
    }
    
    try:
        # Nerd AI
        logger.info("Running Nerd AI showcase...")
        math_output = showcase_nerd_ai(input_dirs['nerd_ai'], output_dirs['nerd_ai'])
        showcase_results["nerd_ai"] = math_output
        
        # Interior Design
        logger.info("Running Interior Design showcase...")
        design_outputs = showcase_interior_design(input_dirs['interior_design'], output_dirs['interior_design'])
        showcase_results["interior_design"] = design_outputs
        
        # Music Generator
        logger.info("Running Music Generator showcase...")
        music_outputs = showcase_music_generator(input_dirs['music'], output_dirs['music'])
        showcase_results["music_generator"] = music_outputs
        
        # Write summary file
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("DAYMENION AI SUITE SHOWCASE SUMMARY\n")
            f.write("==================================\n\n")
            f.write(f"Showcase run on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Nerd AI summary
            f.write("NERD AI SUMMARY\n")
            f.write("--------------\n")
            if math_output:
                f.write(f"Solutions file: {math_output}\n\n")
            else:
                f.write("No math problems processed\n\n")
            
            # Interior Design summary
            f.write("INTERIOR DESIGN SUMMARY\n")
            f.write("----------------------\n")
            if design_outputs:
                for i, output in enumerate(design_outputs, 1):
                    f.write(f"{i}. {output['room']} transformed to {output['style']} style\n")
                    f.write(f"   Output: {output['output_path']}\n")
                    f.write(f"   Prompt: {output['prompt_path']}\n")
                    f.write(f"   Time taken: {output['time_taken']:.2f} seconds\n\n")
            else:
                f.write("No room transformations processed\n\n")
            
            # Music Generator summary
            f.write("MUSIC GENERATOR SUMMARY\n")
            f.write("----------------------\n")
            if music_outputs:
                for i, output in enumerate(music_outputs, 1):
                    f.write(f"{i}. {output['genre']} song with {output['mood']} mood\n")
                    f.write(f"   Title: {output['title']}\n")
                    f.write(f"   Purpose: {output['purpose']}\n")
                    f.write(f"   Lyrics: {output['lyrics_path']}\n")
                    if output['cover_art_path']:
                        f.write(f"   Cover Art: {output['cover_art_path']}\n")
                    f.write(f"   Time taken: {output['elapsed_time']:.2f} seconds\n\n")
            else:
                f.write("No songs generated\n\n")
        
        # Save JSON summary for programmatic access
        json_summary = os.path.join(OUTPUT_DIR, f"showcase_summary_{timestamp}.json")
        with open(json_summary, "w", encoding="utf-8") as f:
            json.dump(showcase_results, f, default=str, indent=2)
        
        logger.info(f"Showcase complete! Summary saved to {summary_file}")
        print(f"\nShowcase complete! Summary saved to {summary_file}")
        
    except Exception as e:
        logger.error(f"Error during showcase: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"Error during showcase: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 