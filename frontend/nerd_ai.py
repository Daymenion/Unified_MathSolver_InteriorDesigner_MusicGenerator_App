"""
Nerd AI frontend component for the Daymenion AI Suite.
"""

import streamlit as st
import os
import tempfile
from PIL import Image

# Use absolute imports instead of relative imports
from nerd_ai.solver import MathSolver
from common.config import NERD_AI_SETTINGS, API_SETTINGS
from common.utils import format_math_notation, format_math_display, format_math_problem
from common.logger import get_logger, log_exceptions

# Initialize logger for this module
logger = get_logger("frontend.nerd_ai")

def render_nerd_ai():
    """Render the Nerd AI application."""
    logger.info("Rendering Nerd AI frontend")
    
    # Add custom CSS for better math rendering
    st.markdown("""
    <style>
    /* Make math notation standout more */
    .katex { 
        font-size: 1.1em;
    }
    /* Add a subtle background to inline math */
    .katex-display {
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 5px;
        margin: 15px auto;
        overflow-x: auto;
    }
    /* Make sure math containers don't overflow on mobile */
    .math-solution, .math-step {
        overflow-x: auto;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🔢 Nerd AI - Math Problem Solver")
    
    st.markdown("""
    ## Scan & Solve Math Problems
    
    Upload an image of a math problem, and our AI will:
    1. Extract the problem using OCR
    2. Identify the type of math problem
    3. Generate a step-by-step solution
    
    Supported math domains: {}
    """.format(", ".join(NERD_AI_SETTINGS["math_domains"])))
    
    # Initialize the solver if not already in session state
    if "math_solver" not in st.session_state:
        logger.debug("Initializing MathSolver in session state")
        st.session_state.math_solver = MathSolver()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image of a math problem", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        logger.info(f"User uploaded file: {uploaded_file.name}")
        
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Math Problem", width=400)
        
        # Add a button to process the image
        if st.button("Solve Problem"):
            logger.info("User clicked 'Solve Problem' button")
            
            with st.spinner("Processing..."):
                # Process the image and get the solution
                with log_exceptions("solve_from_image", "frontend.nerd_ai"):
                    logger.debug("Calling MathSolver.solve_from_image")
                    result = st.session_state.math_solver.solve_from_image(uploaded_file)
                
                if result["success"]:
                    logger.info("Problem solved successfully")
                    # Display the extracted problem
                    st.success("Problem successfully extracted and solved!")
                    
                    st.subheader("Extracted Problem")
                    st.markdown(format_math_problem(result["problem_text"]), unsafe_allow_html=True)
                    
                    st.subheader("Problem Type")
                    st.write(result["problem_type"].capitalize())
                    
                    # Display the solution with enhanced formatting
                    st.subheader("Solution")
                    st.markdown(format_math_display(result["solution"]), unsafe_allow_html=True)
                    
                    # Option to show the code used to solve
                    with st.expander("Show Python Code"):
                        st.code(result["code"], language="python")
                else:
                    error_msg = result["error"]
                    logger.warning(f"Problem solving failed: {error_msg}")
                    st.error(f"Error: {error_msg}")
    
    # Example problems
    st.markdown("---")
    st.subheader("Or try one of our examples")
    
    col1, col2 = st.columns(2)
    
    # Example 1: Algebra
    with col1:
        st.markdown("### Algebra Example")
        st.markdown("Solve the quadratic equation: x² - 5x + 6 = 0")
        if st.button("Solve Algebra Example"):
            logger.info("User clicked 'Solve Algebra Example' button")
            
            with st.spinner("Solving..."):
                # Simulate processing for demo purposes
                logger.debug("Preparing algebra example solution")
                example_result = {
                    "success": True,
                    "problem_text": "x² - 5x + 6 = 0",
                    "problem_type": "algebra",
                    "solution": """
                    ### Step-by-step solution:
                    
                    We need to solve the quadratic equation $x^2 - 5x + 6 = 0$
                    
                    1. We can use the quadratic formula: $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$
                    2. In this equation, $a = 1$, $b = -5$, and $c = 6$
                    3. Substituting these values: $x = \\frac{5 \\pm \\sqrt{(-5)^2 - 4 \\cdot 1 \\cdot 6}}{2 \\cdot 1}$
                    4. Simplifying: $x = \\frac{5 \\pm \\sqrt{25 - 24}}{2} = \\frac{5 \\pm \\sqrt{1}}{2} = \\frac{5 \\pm 1}{2}$
                    5. So $x = \\frac{5 + 1}{2} = 3$ or $x = \\frac{5 - 1}{2} = 2$
                    
                    Therefore, the solutions are $x = 2$ and $x = 3$.
                    """,
                    "code": """
                    import sympy as sp

                    # Define the variable
                    x = sp.Symbol('x')

                    # Define the equation
                    equation = x**2 - 5*x + 6

                    # Solve the equation
                    solutions = sp.solve(equation, x)

                    # Print step-by-step solution
                    print("Step-by-step solution:")
                    print("1. We need to solve the quadratic equation x² - 5x + 6 = 0")
                    print("2. Using the quadratic formula: x = (-b ± √(b² - 4ac)) / (2a)")
                    print("3. For this equation, a = 1, b = -5, and c = 6")
                    print("4. Substituting: x = (5 ± √(25 - 24)) / 2 = (5 ± √1) / 2 = (5 ± 1) / 2")
                    print("5. So x = 3 or x = 2")
                    
                    # Print the final answer
                    print("\\nSolutions:", solutions)
                    """
                }
                
                # Display the results
                logger.info("Displaying algebra example solution")
                st.success("Problem successfully solved!")
                
                st.subheader("Problem")
                st.markdown(format_math_problem(example_result["problem_text"]), unsafe_allow_html=True)
                
                st.subheader("Problem Type")
                st.write(example_result["problem_type"].capitalize())
                
                st.subheader("Solution")
                st.markdown(format_math_display(example_result["solution"]), unsafe_allow_html=True)
                
                with st.expander("Show Python Code"):
                    st.code(example_result["code"], language="python")
    
    # Example 2: Calculus
    with col2:
        st.markdown("### Calculus Example")
        st.markdown("Find the derivative of: f(x) = x³ - 4x² + 2x")
        if st.button("Solve Calculus Example"):
            logger.info("User clicked 'Solve Calculus Example' button")
            
            with st.spinner("Solving..."):
                # Simulate processing for demo purposes
                logger.debug("Preparing calculus example solution")
                example_result = {
                    "success": True,
                    "problem_text": "Find the derivative of: f(x) = x³ - 4x² + 2x",
                    "problem_type": "calculus",
                    "solution": """
                    ### Step-by-step solution:
                    
                    We need to find the derivative of $f(x) = x^3 - 4x^2 + 2x$
                    
                    1. Apply the power rule for each term: $\\frac{d}{dx}(x^n) = n \\cdot x^{n-1}$
                    2. For the first term: $\\frac{d}{dx}(x^3) = 3x^2$
                    3. For the second term: $\\frac{d}{dx}(-4x^2) = -4 \\cdot 2x = -8x$
                    4. For the third term: $\\frac{d}{dx}(2x) = 2$
                    5. Combine all terms: $f'(x) = 3x^2 - 8x + 2$
                    
                    Therefore, the derivative is $f'(x) = 3x^2 - 8x + 2$.
                    """,
                    "code": """
                    import sympy as sp

                    # Define the variable
                    x = sp.Symbol('x')

                    # Define the function
                    f = x**3 - 4*x**2 + 2*x

                    # Find the derivative
                    f_prime = sp.diff(f, x)

                    # Print step-by-step solution
                    print("Step-by-step solution:")
                    print("1. We need to find the derivative of f(x) = x³ - 4x² + 2x")
                    print("2. Using the power rule: d/dx(x^n) = n·x^(n-1)")
                    print("3. For the first term: d/dx(x³) = 3x²")
                    print("4. For the second term: d/dx(-4x²) = -4·2x = -8x")
                    print("5. For the third term: d/dx(2x) = 2")
                    print("6. Combining all terms: f'(x) = 3x² - 8x + 2")
                    
                    # Print the final answer
                    print("\\nDerivative: f'(x) =", f_prime)
                    """
                }
                
                # Display the results
                logger.info("Displaying calculus example solution")
                st.success("Problem successfully solved!")
                
                st.subheader("Problem")
                st.markdown(format_math_problem(example_result["problem_text"]), unsafe_allow_html=True)
                
                st.subheader("Problem Type")
                st.write(example_result["problem_type"].capitalize())
                
                st.subheader("Solution")
                st.markdown(format_math_display(example_result["solution"]), unsafe_allow_html=True)
                
                with st.expander("Show Python Code"):
                    st.code(example_result["code"], language="python")
                    
    # How it works section
    st.markdown("---")
    st.subheader("How it works")
    
    st.markdown("""
    Nerd AI uses a sophisticated workflow to solve math problems:
    
    1. **Image Processing**: Your uploaded image is preprocessed to enhance readability.
    2. **OCR**: Our AI model extracts the mathematical text and symbols from the image.
    3. **Problem Classification**: The system identifies what type of math problem it is.
    4. **Solution Generation**: Using specialized mathematical libraries like SymPy, the AI generates Python code to solve the problem.
    5. **Explanation**: The system provides a detailed, step-by-step explanation of how to solve the problem.
    
    All of this happens in seconds, giving you not just answers, but understanding!
    """) 
    
    logger.debug("Nerd AI frontend rendering complete")


def show():
    """
    Show the Nerd AI interface.
    
    This function is called by the main app to display the Nerd AI page.
    """
    with log_exceptions("showing_nerd_ai_page", "frontend.nerd_ai"):
        render_nerd_ai() 