"""
Comprehensive test script for Codeway AI Suite.
This script verifies that all components of the three applications are functioning correctly:
1. Nerd AI - Math Problem Solver
2. Interior Design App - Room Style Transformer
3. Music Generator - Lyrics and Cover Art Generator
"""

import os
import sys
import json
import logging
import argparse
import tempfile
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from PIL import Image
import base64
from io import BytesIO
from dotenv import load_dotenv
from interior_design.style_transformer import StyleTransformer
from PIL import Image, ImageDraw, ImageFont

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Define log file path
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f'ai_suite_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# Configure logging with explicit file handler
logger = logging.getLogger("ai_suite_test")
logger.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create file handler
file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Log startup message
logger.info(f"Starting Codeway AI Suite test, log file: {LOG_FILE}")

# Load environment variables
load_dotenv()

#########################
# Common Tests
#########################

def test_openai_api_connectivity():
    """Test OpenAI API connectivity."""
    logger.info("Testing OpenAI API connectivity...")
    
    try:
        from common.ai_service import OpenAIService
        
        # Create OpenAI service
        api_service = OpenAIService()
        
        # Test text generation
        prompt = "Generate a simple greeting."
        result = api_service.generate_text(prompt, max_tokens=50)
        
        if result and len(result) > 0:
            logger.info("OpenAI API connection successful (text generation)")
            logger.info(f"Response: {result[:50]}...")
        else:
            logger.error("OpenAI API text generation failed")
            return False
        
        # Test with image analysis if possible (using a mock image)
        try:
            # Create a simple test image
            test_image = Image.new('RGB', (100, 100), color='red')
            
            prompt = "Describe what you see in this image."
            result = api_service.analyze_image(test_image, prompt)
            
            if result and len(result) > 0:
                logger.info("OpenAI API connection successful (image analysis)")
                logger.info(f"Response: {result[:50]}...")
            else:
                logger.warning("OpenAI API image analysis failed, but this might be expected if using a mock image")
        except Exception as e:
            logger.warning(f"OpenAI API image analysis test skipped or failed: {str(e)}")
            # This is not a critical failure, as not all versions of the API support this
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing OpenAI API: {str(e)}")
        return False


def test_environment_variables():
    """Test that necessary environment variables are set."""
    logger.info("Testing environment variables...")
    
    required_vars = ["OPENAI_API_KEY"]
    optional_vars = ["HUGGINGFACE_API_KEY"]
    
    missing_required = []
    missing_optional = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)
    
    for var in optional_vars:
        if not os.getenv(var):
            missing_optional.append(var)
    
    if missing_required:
        logger.error(f"Missing required environment variables: {', '.join(missing_required)}")
        return False
    
    if missing_optional:
        logger.warning(f"Missing optional environment variables: {', '.join(missing_optional)}")
    
    logger.info("All required environment variables are set")
    return True


def test_common_utilities():
    """Test common utility functions."""
    logger.info("Testing common utilities...")
    
    try:
        from common.utils import validate_image, preprocess_image, image_to_base64, format_timestamp
        
        # Test image validation
        test_image = Image.new('RGB', (100, 100), color='red')
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        test_image.save(temp_file.name)
        temp_file.close()
        
        class MockFile:
            def __init__(self, path):
                self.name = path
                self._file = open(path, 'rb')
                
            def read(self):
                return self._file.read()
                
            def seek(self, position):
                return self._file.seek(position)
                
            def tell(self):
                return self._file.tell()
                
            def close(self):
                self._file.close()
        
        mock_file = MockFile(temp_file.name)
        
        try:
            is_valid, _ = validate_image(mock_file)
            if not is_valid:
                logger.error("Image validation failed")
                return False
            
            # Test image preprocessing
            processed_image = preprocess_image(mock_file)
            if not isinstance(processed_image, Image.Image):
                logger.error("Image preprocessing failed")
                return False
            
            # Test base64 conversion
            base64_str = image_to_base64(processed_image)
            if not base64_str or len(base64_str) == 0:
                logger.error("Image to base64 conversion failed")
                return False
        finally:
            # Clean up
            mock_file.close()
            try:
                os.unlink(temp_file.name)
            except:
                pass
        
        # Test timestamp formatting
        timestamp = format_timestamp()
        if not timestamp or len(timestamp) == 0:
            logger.error("Timestamp formatting failed")
            return False
        
        logger.info("Common utilities test successful")
        return True
        
    except Exception as e:
        logger.error(f"Error testing common utilities: {str(e)}")
        return False


#########################
# Nerd AI Tests
#########################

def test_nerd_ai_ocr():
    """Test Nerd AI OCR functionality with a test image."""
    logger.info("Testing Nerd AI OCR...")
    
    try:
        from nerd_ai.solver import MathSolver
        
        # Create a test image with a simple math problem
        # (In a real test, you would use a pre-created test image file)
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a white image
        img = Image.new('RGB', (500, 200), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a standard font
        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except IOError:
            # Fallback to default
            font = ImageFont.load_default()
        
        # Draw a simple math problem
        draw.text((50, 50), "2x + 5 = 15", fill='black', font=font)
        
        # Create a MathSolver instance
        solver = MathSolver()
        
        # Test OCR using in-memory image
        extracted_text = solver.ocr_math_problem(img)
        
        if not extracted_text or len(extracted_text) == 0:
            logger.error("OCR extraction failed - no text extracted")
            return False
        
        logger.info(f"OCR extracted text: {extracted_text}")
        
        # Check if the extracted text contains any part of our equation
        if "2x" in extracted_text or "=" in extracted_text or "15" in extracted_text:
            logger.info("OCR test successful - found parts of the equation")
        else:
            logger.warning("OCR test partially successful - text extracted but didn't match expected equation")
            
        return True
        
    except Exception as e:
        logger.error(f"Error testing Nerd AI OCR: {str(e)}")
        return False


def test_nerd_ai_problem_classification():
    """Test Nerd AI problem classification."""
    logger.info("Testing Nerd AI problem classification...")
    
    try:
        from nerd_ai.solver import MathSolver
        
        # Create a MathSolver instance
        solver = MathSolver()
        
        # Test problem classification with different types
        test_cases = [
            ("Solve for x: 2x + 5 = 15", "algebra"),
            ("Find the derivative of f(x) = x^2 + 3x", "calculus"),
            ("Calculate the area of a circle with radius 5", "geometry"),
            ("Find the mean of 4, 7, 9, 12, 15", "statistics")
        ]
        
        success_count = 0
        
        for problem_text, expected_type in test_cases:
            try:
                problem_type = solver.identify_problem_type(problem_text)
                
                logger.info(f"Problem: {problem_text}")
                logger.info(f"Classified as: {problem_type}")
                logger.info(f"Expected: {expected_type}")
                
                # Consider it a success if the classification is reasonable
                # (exact matches aren't always necessary due to model variability)
                if problem_type == expected_type or problem_type in expected_type or expected_type in problem_type:
                    success_count += 1
                    logger.info("Classification successful")
                else:
                    logger.warning(f"Classification differs from expected: got '{problem_type}', expected '{expected_type}'")
            except Exception as e:
                logger.error(f"Error classifying problem: {str(e)}")
        
        success_rate = success_count / len(test_cases)
        logger.info(f"Problem classification success rate: {success_rate:.2f} ({success_count}/{len(test_cases)})")
        
        # Consider the test successful if at least 50% of classifications were reasonable
        return success_rate >= 0.5
        
    except Exception as e:
        logger.error(f"Error testing Nerd AI problem classification: {str(e)}")
        return False


def test_nerd_ai_solution_generation():
    """Test Nerd AI solution generation."""
    logger.info("Testing Nerd AI solution generation...")
    
    try:
        from nerd_ai.solver import MathSolver
        
        # Create a MathSolver instance
        solver = MathSolver()
        
        # Test with a simple algebra problem
        problem_text = "Solve for x: 2x + 5 = 15"
        problem_type = "algebra"
        
        logger.info(f"Generating solution for: {problem_text}")
        solution_data = solver.generate_solution(problem_text, problem_type)
        
        if (solution_data and "explanation" in solution_data and 
            solution_data["explanation"] and len(solution_data["explanation"]) > 0):
            logger.info("Solution generation successful")
            logger.info(f"Solution excerpt: {solution_data['explanation'][:100]}...")
            
            # Check if solution contains the expected answer (x = 5)
            if "x = 5" in solution_data["explanation"]:
                logger.info("Solution contains the correct answer")
            else:
                logger.warning("Solution might not contain the expected answer 'x = 5'")
            
            return True
        else:
            logger.error("Solution generation failed - empty or invalid solution")
            return False
        
    except Exception as e:
        logger.error(f"Error testing Nerd AI solution generation: {str(e)}")
        return False


def test_nerd_ai_end_to_end():
    """Test Nerd AI end-to-end workflow."""
    logger.info("Testing Nerd AI end-to-end workflow...")
    
    try:
        from nerd_ai.solver import MathSolver
        
        # Create a test image with a simple math problem
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a white image
        img = Image.new('RGB', (500, 200), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a standard font
        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except IOError:
            # Fallback to default
            font = ImageFont.load_default()
        
        # Draw a simple math problem
        draw.text((50, 50), "2x + 5 = 15", fill='black', font=font)
        
        # Save to a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img.save(temp_file.name)
        temp_file.close()
        
        # Create a mock file object
        class MockFile:
            def __init__(self, path):
                self.name = path
        
        mock_file = MockFile(temp_file.name)
        
        # Create a MathSolver instance
        solver = MathSolver()
        
        # Test the end-to-end workflow
        logger.info("Running end-to-end test with image of '2x + 5 = 15'")
        result = solver.solve_from_image(mock_file)
        
        # Clean up
        os.unlink(temp_file.name)
        
        if not result["success"]:
            logger.error(f"End-to-end test failed: {result.get('error', 'Unknown error')}")
            return False
        
        logger.info(f"Extracted problem: {result['problem_text']}")
        logger.info(f"Problem type: {result['problem_type']}")
        logger.info(f"Solution excerpt: {result['solution'][:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing Nerd AI end-to-end: {str(e)}")
        return False


#########################
# Interior Design Tests
#########################

def test_interior_design_room_detection():
    """Test Interior Design room type detection."""
    logger.info("Testing Interior Design room detection...")
    
    try:
        from interior_design.style_transformer import StyleTransformer
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a StyleTransformer instance
        transformer = StyleTransformer()
        
        # Create mock test images
        # (In a real test, you would use real room images)
        
        def create_mock_room_image(room_type):
            img = Image.new('RGB', (500, 300), color='white')
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except IOError:
                font = ImageFont.load_default()
            
            # Add text indicating room type
            draw.text((50, 50), f"This is a {room_type}", fill='black', font=font)
            
            # Add simple furniture outlines based on room type
            if room_type == "Living Room":
                # Draw a couch
                draw.rectangle((100, 150, 300, 200), outline='black')
                # Draw a coffee table
                draw.rectangle((150, 220, 250, 250), outline='black')
            elif room_type == "Kitchen":
                # Draw a counter
                draw.rectangle((100, 150, 400, 180), outline='black')
                # Draw a stove
                draw.rectangle((200, 150, 300, 180), outline='black', width=2)
            
            return img
        
        # Test with different room types
        test_rooms = ["Living Room", "Kitchen"]
        success_count = 0
        
        for expected_room_type in test_rooms:
            try:
                # Create a test image
                img = create_mock_room_image(expected_room_type)
                
                # Test room detection
                detected_room = transformer.identify_room_type(img)
                
                logger.info(f"Expected room type: {expected_room_type}")
                logger.info(f"Detected room type: {detected_room}")
                
                # Consider it a success if the detected room is reasonable
                if (detected_room.lower() == expected_room_type.lower() or 
                    detected_room.lower() in expected_room_type.lower() or 
                    expected_room_type.lower() in detected_room.lower()):
                    success_count += 1
                    logger.info("Room detection successful")
                else:
                    logger.warning(f"Room detection differs from expected: got '{detected_room}', expected '{expected_room_type}'")
            except Exception as e:
                logger.error(f"Error detecting room type: {str(e)}")
        
        success_rate = success_count / len(test_rooms)
        logger.info(f"Room detection success rate: {success_rate:.2f} ({success_count}/{len(test_rooms)})")
        
        # Consider the test successful if at least 50% of detections were reasonable
        return success_rate >= 0.5
        
    except Exception as e:
        logger.error(f"Error testing Interior Design room detection: {str(e)}")
        return False


def test_interior_design_prompt_generation():
    """Test Interior Design transformation prompt generation."""
    logger.info("Testing Interior Design prompt generation...")
    
    try:
        from interior_design.style_transformer import StyleTransformer
        
        # Create a StyleTransformer instance
        transformer = StyleTransformer()
        
        # Test prompt generation with different room types and styles
        test_cases = [
            ("Living Room", "Modern"),
            ("Living Room", "Gothic"),
            ("Kitchen", "Soho")
        ]
        
        for room_type, style in test_cases:
            prompt = transformer.generate_transformation_prompt(room_type, style)
            
            logger.info(f"Room type: {room_type}, Style: {style}")
            logger.info(f"Generated prompt: {prompt}")
            
            # Check that the prompt contains key elements
            if (room_type in prompt and style in prompt and 
                "maintain" in prompt.lower() and "layout" in prompt.lower()):
                logger.info("Prompt generation successful - contains all key elements")
            else:
                logger.warning("Prompt is missing some key elements")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing Interior Design prompt generation: {str(e)}")
        return False


def test_interior_design_style_transformation():
    """Test Interior Design style transformation."""
    logger.info("Testing Interior Design style transformation...")
    
    try:
        
        
        # Create a StyleTransformer instance
        transformer = StyleTransformer()
        
        # Create a test image
        img = Image.new('RGB', (500, 300), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            font = ImageFont.load_default()
        
        # Draw a simple living room
        draw.text((50, 50), "Living Room", fill='black', font=font)
        # Draw a couch
        draw.rectangle((100, 150, 300, 200), outline='black')
        # Draw a coffee table
        draw.rectangle((150, 220, 250, 250), outline='black')
        
        # Save to a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img.save(temp_file.name)
        temp_file.close()
        
        # Create a mock file object
        class MockFile:
            def __init__(self, path):
                self.name = path
        
        mock_file = MockFile(temp_file.name)
        
        # Test the transformation
        style = "Modern"
        room_type = "Living Room"
        
        logger.info(f"Transforming test image to {style} style...")
        result = transformer.transform_style(mock_file, style, room_type)
        
        # Clean up
        os.unlink(temp_file.name)
        
        if not result["success"]:
            logger.error(f"Style transformation failed: {result.get('error', 'Unknown error')}")
            return False
        
        logger.info("Style transformation successful")
        logger.info(f"Room type: {result['room_type']}")
        logger.info(f"Style: {result['style']}")
        
        # Check that the transformed image was created
        if not result["transformed_image"] or not isinstance(result["transformed_image"], Image.Image):
            logger.error("Transformation failed - no valid image returned")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing Interior Design style transformation: {str(e)}")
        return False


#########################
# Music Generator Tests
#########################

def test_music_generator_lyrics():
    """Test Music Generator lyrics generation."""
    logger.info("Testing Music Generator lyrics generation...")
    
    try:
        from music_generator.generator import MusicGenerator
        
        # Create a MusicGenerator instance
        generator = MusicGenerator()
        
        # Test lyrics generation with a simple request
        genre = "Pop"
        mood = "Happy"
        purpose = "For celebration"
        custom_description = "A song about achieving goals and succeeding"
        
        logger.info(f"Generating lyrics for {genre} song with {mood} mood...")
        result = generator.generate_lyrics(genre, mood, purpose, custom_description)
        
        if not result["success"]:
            logger.error(f"Lyrics generation failed: {result.get('error', 'Unknown error')}")
            return False
        
        logger.info("Lyrics generation successful")
        logger.info(f"Lyrics excerpt: {result['lyrics'][:100]}...")
        
        # Check that the lyrics contain key elements
        if ("Verse" in result["lyrics"] and 
            "Chorus" in result["lyrics"] and 
            len(result["lyrics"]) > 200):
            logger.info("Lyrics are properly structured with verses and chorus")
        else:
            logger.warning("Lyrics might not have proper song structure")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing Music Generator lyrics: {str(e)}")
        return False


def test_music_generator_cover_art():
    """Test Music Generator cover art generation."""
    logger.info("Testing Music Generator cover art generation...")
    
    try:
        from music_generator.generator import MusicGenerator
        
        # Create a MusicGenerator instance
        generator = MusicGenerator()
        
        # Create mock lyrics data
        lyrics_data = {
            "success": True,
            "lyrics": """
            Verse 1:
            Standing tall against the odds
            Breaking through the barriers
            Every step a victory
            Every moment counts
            
            Chorus:
            We're celebrating now
            Raising voices high
            The moment we've been waiting for
            Is finally here tonight
            """,
            "genre": "Pop",
            "mood": "Happy",
            "purpose": "For celebration",
            "main_theme": "Achievement and celebration",
            "custom_description": "A song about achieving goals and succeeding"
        }
        
        logger.info(f"Generating cover art for '{lyrics_data['main_theme']}' theme...")
        result = generator.generate_cover_art(lyrics_data)
        
        if not result["success"]:
            logger.error(f"Cover art generation failed: {result.get('error', 'Unknown error')}")
            return False
        
        logger.info("Cover art generation successful")
        
        # Check that a valid image was created
        if not result["cover_art"] or not isinstance(result["cover_art"], Image.Image):
            logger.error("Cover art generation failed - no valid image returned")
            return False
        
        logger.info(f"Cover art size: {result['cover_art'].size}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing Music Generator cover art: {str(e)}")
        return False


def test_music_generator_end_to_end():
    """Test Music Generator end-to-end workflow."""
    logger.info("Testing Music Generator end-to-end workflow...")
    
    try:
        from music_generator.generator import MusicGenerator
        
        # Create a MusicGenerator instance
        generator = MusicGenerator()
        
        # Test the end-to-end package generation
        genre = "Rock"
        mood = "Energetic"
        purpose = "For motivation"
        custom_description = "A song about overcoming challenges and finding inner strength"
        
        logger.info(f"Generating complete song package for {genre}/{mood} song...")
        result = generator.generate_song_package(genre, mood, purpose, custom_description)
        
        if not result["success"]:
            logger.error(f"Song package generation failed: {result.get('error', 'Unknown error')}")
            return False
        
        logger.info("Song package generation successful")
        
        # Check lyrics
        if not result["lyrics_data"] or not result["lyrics_data"]["success"]:
            logger.error("Lyrics generation failed as part of package")
            return False
        
        logger.info(f"Lyrics excerpt: {result['lyrics_data']['lyrics'][:100]}...")
        
        # Check cover art
        if not result["cover_art_data"] or not result["cover_art_data"]["success"]:
            logger.error("Cover art generation failed as part of package")
            return False
        
        logger.info("Cover art successfully generated")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing Music Generator end-to-end: {str(e)}")
        return False


#########################
# Main Test Runner
#########################

def run_tests(test_names=None, skip_slow=False):
    """
    Run specified tests or all tests.
    
    Args:
        test_names: List of test names to run, or None to run all tests
        skip_slow: Whether to skip slow tests
    
    Returns:
        Tuple of (passed_count, failed_count, test_results)
    """
    # Define all available tests
    all_tests = {
        # Common tests
        "environment": test_environment_variables,
        "openai_api": test_openai_api_connectivity,
        "utilities": test_common_utilities,
        
        # Nerd AI tests
        "nerd_ai_ocr": test_nerd_ai_ocr,
        "nerd_ai_classification": test_nerd_ai_problem_classification,
        "nerd_ai_solution": test_nerd_ai_solution_generation,
        "nerd_ai_e2e": test_nerd_ai_end_to_end,
        
        # Interior Design tests
        "interior_room_detection": test_interior_design_room_detection,
        "interior_prompt": test_interior_design_prompt_generation,
        "interior_transform": test_interior_design_style_transformation,
        
        # Music Generator tests
        "music_lyrics": test_music_generator_lyrics,
        "music_cover_art": test_music_generator_cover_art,
        "music_e2e": test_music_generator_end_to_end,
    }
    
    # Slow tests that might be skipped
    slow_tests = {
        "nerd_ai_e2e", 
        "interior_transform", 
        "music_cover_art", 
        "music_e2e"
    }
    
    # Determine which tests to run
    tests_to_run = {}
    
    if test_names and "all" not in test_names:
        # Run specific tests
        for name in test_names:
            if name in all_tests:
                tests_to_run[name] = all_tests[name]
            else:
                logger.warning(f"Unknown test: {name}")
    else:
        # Run all tests
        tests_to_run = all_tests
    
    # Skip slow tests if requested
    if skip_slow:
        for slow_test in slow_tests:
            if slow_test in tests_to_run:
                logger.info(f"Skipping slow test: {slow_test}")
                tests_to_run.pop(slow_test)
    
    # Run tests and collect results
    results = {}
    passed = 0
    failed = 0
    
    for name, test_func in tests_to_run.items():
        logger.info(f"\n========== Test: {name} ==========")
        
        try:
            success = test_func()
            results[name] = success
            
            if success:
                passed += 1
                logger.info(f"Test '{name}': PASSED")
            else:
                failed += 1
                logger.error(f"Test '{name}': FAILED")
        except Exception as e:
            results[name] = False
            failed += 1
            logger.error(f"Test '{name}': ERROR - {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    return passed, failed, results


def print_summary(passed, failed, results):
    """Print a summary of test results."""
    logger.info("\n========== Test Summary ==========")
    logger.info(f"Total Tests: {passed + failed}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    
    # Group by category
    categories = {
        "Common": ["environment", "openai_api", "utilities"],
        "Nerd AI": ["nerd_ai_ocr", "nerd_ai_classification", "nerd_ai_solution", "nerd_ai_e2e"],
        "Interior Design": ["interior_room_detection", "interior_prompt", "interior_transform"],
        "Music Generator": ["music_lyrics", "music_cover_art", "music_e2e"]
    }
    
    # Print results by category
    for category, tests in categories.items():
        logger.info(f"\n{category} Tests:")
        
        category_tests = [t for t in tests if t in results]
        if not category_tests:
            logger.info("  No tests run in this category")
            continue
            
        category_passed = sum(1 for t in category_tests if results[t])
        category_total = len(category_tests)
        
        logger.info(f"  {category_passed}/{category_total} tests passed")
        
        for test in category_tests:
            status = "PASSED" if results.get(test, False) else "FAILED"
            logger.info(f"  {test}: {status}")
    
    # Print overall pass/fail
    all_passed = failed == 0
    logger.info("\nOverall Result: {}".format("PASSED" if all_passed else "FAILED"))
    
    return all_passed


def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description="Test Codeway AI Suite components")
    parser.add_argument("--test", type=str, nargs="+", 
                         choices=["all", "environment", "openai_api", "utilities", 
                                 "nerd_ai_ocr", "nerd_ai_classification", "nerd_ai_solution", "nerd_ai_e2e", 
                                 "interior_room_detection", "interior_prompt", "interior_transform", 
                                 "music_lyrics", "music_cover_art", "music_e2e"],
                         default=["all"], help="Specific test(s) to run")
    parser.add_argument("--skip-slow", action="store_true", help="Skip slow tests")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                        default="INFO", help="Set logging level")
    
    args = parser.parse_args()
    
    # Set log level based on argument
    log_level = getattr(logging, args.log_level)
    logger.setLevel(log_level)
    console_handler.setLevel(log_level)
    file_handler.setLevel(log_level)
    
    # Run tests
    logger.info(f"Running tests: {', '.join(args.test)}")
    passed, failed, results = run_tests(args.test, args.skip_slow)
    
    # Print summary
    all_passed = print_summary(passed, failed, results)
    
    # Final message
    logger.info("Test suite completed. Check the log file for details.")
    print(f"Log file created at: {LOG_FILE}")
    
    # Return appropriate exit code
    return 0 if all_passed else 1


if __name__ == "__main__":
    # Make sure logs are properly flushed even if the program crashes
    try:
        sys.exit(main())
    finally:
        for handler in logger.handlers:
            handler.flush()
            handler.close()
        logging.shutdown() 