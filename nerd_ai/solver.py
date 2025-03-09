"""
Math Solver module for Nerd AI.

This module contains the core functionality for scanning and solving math problems.
"""

import os
from typing import Dict, Any, List, Optional, Union
from PIL import Image

from common.ai_service import OpenAIService
from common.utils import validate_image, preprocess_image
from common.config import NERD_AI_SETTINGS
from common.logger import get_logger, log_exceptions


class MathSolver:
    """
    Math problem solver for Nerd AI.
    
    This class handles the complete workflow for processing math problem images and
    generating solutions.
    """
    
    def __init__(self):
        """Initialize the math solver with required services."""
        self.logger = get_logger("nerd_ai.solver")
        self.logger.info("Initializing MathSolver")
        self.ai_service = OpenAIService()
        self.supported_domains = NERD_AI_SETTINGS["math_domains"]
        self.logger.debug(f"Supported math domains: {', '.join(self.supported_domains)}")
    
    def ocr_math_problem(self, image: Union[Image.Image, str]) -> str:
        """
        Extract math problem from an image using OCR.
        
        Args:
            image: PIL Image object or path to image file
            
        Returns:
            str: Extracted math problem text
        """
        self.logger.info("Extracting math problem using OCR")
        
        prompt = (
            "Extract the mathematical problem or equation from this image. "
            "Format the extracted problem maintaining all mathematical symbols and notation. "
            "Return ONLY the extracted problem text, nothing else. "
            "If there's a mixture of text and math, extract and format everything accurately."
        )
        
        with log_exceptions("ocr_text_extraction", "nerd_ai.solver"):
            extracted_text = self.ai_service.analyze_image(image, prompt)
            self.logger.debug(f"Extracted text: {extracted_text[:100]}{'...' if len(extracted_text) > 100 else ''}")
            return extracted_text.strip()
    
    def identify_problem_type(self, problem_text: str) -> str:
        """
        Identify the type/domain of the math problem.
        
        Args:
            problem_text: Extracted problem text
            
        Returns:
            str: Problem domain (e.g., 'algebra', 'calculus')
        """
        self.logger.info("Identifying math problem type")
        
        prompt = (
            f"Identify the mathematical domain of this problem:\n\n"
            f"{problem_text}\n\n"
            f"Choose one domain from this list: {', '.join(self.supported_domains)}.\n"
            f"Reply with ONLY the domain name, nothing else."
        )
        
        with log_exceptions("problem_type_identification", "nerd_ai.solver"):
            problem_type = self.ai_service.generate_text(prompt, temperature=0.1)
            problem_type = problem_type.strip().lower()
            self.logger.info(f"Identified problem type: {problem_type}")
            return problem_type
    
    def generate_solution(self, problem_text: str, problem_type: str) -> Dict[str, str]:
        """
        Generate a solution for a math problem.
        
        Args:
            problem_text: The text of the problem to solve
            problem_type: The type/domain of the problem
            
        Returns:
            Dict[str, str]: Solution data with code and explanation
        """
        self.logger.info(f"Generating solution for {problem_type} problem")
        
        # First, generate the code to solve the problem
        prompt = (
            f"Create a Python solution for this {problem_type} problem:\n\n"
            f"{problem_text}\n\n"
            f"Important requirements for your solution:\n"
            f"1. Use appropriate mathematical libraries (like sympy, numpy, etc.)\n"
            f"2. Include step-by-step comments explaining the solution process\n"
            f"3. Print the steps and final answer in a clear, readable format\n"
            f"4. Return only the Python code, nothing else\n"
            f"5. Don't use matplotlib or any visualization libraries\n"
            f"6. Use sympy for symbolic mathematics wherever appropriate\n"
            f"7. Make sure the code is executable without errors"
        )
        
        with log_exceptions("generating_solution_code", "nerd_ai.solver"):
            solution_code = self.ai_service.generate_text(prompt, temperature=0.1)
            self.logger.debug(f"Generated solution code of length: {len(solution_code)}")
        
        # Generate an explanation
        explanation_prompt = (
            f"Provide a step-by-step explanation for this {problem_type} problem:\n\n"
            f"{problem_text}\n\n"
            f"Format guidelines for your response:\n"
            f"1. Use LaTeX mathematical notation with PROPER DELIMITERS:\n"
            f"   - Use $...$ for inline math (expressions within a paragraph)\n"
            f"   - Use $$...$$ for display math (standalone expressions)\n"
            f"   - DO NOT use square brackets [ ] or parentheses ( ) to denote equations\n"
            f"2. Structure your explanation with clear step headings (e.g., Step 1, Step 2)\n"
            f"3. Include a 'Final Answer' or 'Conclusion' section at the end\n"
            f"4. Keep equations properly formatted - for complex fractions use \\frac{{}}{{}} not /\n"
            f"5. Use proper LaTeX for mathematical symbols (e.g., \\infty for infinity, \\int for integrals)\n"
            f"6. Break down the solution process into 3-6 clear steps, explaining the mathematical concepts\n"
            f"7. Provide the answer in its simplified form\n\n"
            f"Make your explanation educational and clear enough for a student to understand."
        )
        
        with log_exceptions("generating_explanation", "nerd_ai.solver"):
            explanation = self.ai_service.generate_text(explanation_prompt, temperature=0.1)
            self.logger.debug(f"Generated explanation of length: {len(explanation)}")
        
        self.logger.info("Solution generation complete")
        return {
            "code": solution_code,
            "explanation": explanation
        }
    
    def solve_from_image(self, image_file) -> Dict[str, Any]:
        """
        Complete workflow to solve a math problem from an image.
        
        Args:
            image_file: Uploaded image file object
            
        Returns:
            Dict[str, Any]: Complete solution data
        """
        self.logger.info("Starting complete math problem solving workflow")
        
        # Validate the image
        is_valid, error_message = validate_image(image_file)
        if not is_valid:
            self.logger.warning(f"Invalid image: {error_message}")
            return {"success": False, "error": error_message}
        
        try:
            # Preprocess the image
            with log_exceptions("preprocessing_image", "nerd_ai.solver"):
                processed_image = preprocess_image(image_file)
                self.logger.debug("Image preprocessing complete")
            
            # Extract the problem using OCR
            problem_text = self.ocr_math_problem(processed_image)
            if not problem_text:
                error_msg = "Could not extract a math problem from the image"
                self.logger.warning(error_msg)
                return {"success": False, "error": error_msg}
            
            # Identify the problem type
            problem_type = self.identify_problem_type(problem_text)
            
            # Generate the solution
            solution_data = self.generate_solution(problem_text, problem_type)
            
            self.logger.info("Math problem solved successfully")
            return {
                "success": True,
                "problem_text": problem_text,
                "problem_type": problem_type,
                "solution": solution_data["explanation"],
                "code": solution_data["code"]
            }
            
        except Exception as e:
            error_msg = f"Error solving problem: {str(e)}"
            self.logger.exception(error_msg, module="nerd_ai.solver")
            return {"success": False, "error": error_msg} 