# Nerd AI: Math Problem Solver Workflow

## Overview

Nerd AI is an intelligent math problem solver that can analyze images of mathematical problems, identify their type, and generate step-by-step solutions with educational explanations. The system uses OCR to extract mathematical notation from images and leverages advanced language models to solve problems across various mathematical domains.

## Complete Workflow

1. **Problem Intake**
   - User uploads or takes a photo of a math problem
   - Image is validated for format, size, and quality
   - Supported formats: JPG, JPEG, PNG
   - Maximum file size: 10MB

2. **OCR Processing**
   - OCR extracts the mathematical notation and text from the image
   - Specialized prompting ensures accurate recognition of math symbols
   - Extracted text is normalized and formatted for processing
   - Special attention to mathematical symbols, subscripts, superscripts, and equations

3. **Problem Classification**
   - AI analyzes the problem text to identify the mathematical domain
   - Categories include: algebra, calculus, statistics, geometry, number theory, trigonometry
   - Classification determines specialized solving approach
   - High precision classification improves solution quality

4. **Solution Generation**
   - Two-step process:
     1. Code generation: Create Python code to solve the problem
     2. Educational explanation: Generate step-by-step explanation with proper mathematical notation
   - Solution is formatted with LaTeX for proper mathematical display
   - Both computational and conceptual approaches provided

5. **Result Presentation**
   - Complete solution with:
     - Original problem text
     - Problem type/domain
     - Solution code (for verification)
     - Step-by-step explanation
     - Final answer clearly marked
   - LaTeX formatting for proper mathematical rendering

## Prompts & Models

### OCR Prompt Template
```
Extract the mathematical problem from this image. 
Capture all mathematical notation precisely, including:
- Fractions, square roots, and other mathematical symbols
- Subscripts and superscripts
- Equations and inequalities
- Any text instructions that are part of the problem

Output only the extracted problem with no additional explanation or commentary.
```

### Problem Classification Prompt Template
```
Identify the mathematical domain of this problem:
"{extracted_problem_text}"

Classify it into ONE of the following categories:
- algebra: Equations, inequalities, functions, systems of equations
- calculus: Derivatives, integrals, limits, optimization
- statistics: Probability, distributions, hypothesis testing, data analysis
- geometry: Angles, shapes, areas, volumes, coordinate geometry
- number_theory: Number properties, divisibility, modular arithmetic
- trigonometry: Trig functions, identities, angles, triangles

Return only the category name with no explanation.
```

### Solution Code Generation Prompt Template
```
Create a Python solution for this {problem_type} problem:

{problem_text}

Important requirements for your solution:
1. Use appropriate mathematical libraries (like sympy, numpy, etc.)
2. Make sure the code is correct and executable
3. Use clear variable names and add comments to explain steps
4. Include imports at the beginning
5. Format complex mathematical expressions carefully
6. End with a clear print statement showing the final result
7. Make sure the solution is complete and correct
```

### Educational Explanation Prompt Template
```
Provide a clear, step-by-step explanation for this {problem_type} problem:

{problem_text}

Format your explanation following these guidelines:
1. Use properly formatted LaTeX math notation:
   - Use $...$ for inline math expressions
   - Use $$...$$ for display math (standalone expressions)
   - DO NOT use square brackets [ ] or parentheses ( ) to denote equations
2. Structure your explanation with clear step headings (e.g., Step 1, Step 2)
3. Include a 'Final Answer' or 'Conclusion' section at the end
4. Keep equations properly formatted - for complex fractions use \frac{}{} not /
5. Use proper LaTeX for mathematical symbols (e.g., \infty for infinity, \int for integrals)
6. Break down the solution process into 3-6 clear steps, explaining the mathematical concepts
7. Provide the answer in its simplified form

Make your explanation educational and clear enough for a student to understand.
```

### Model Selection Rationale

**Primary Model: GPT-4o-mini**
- Provides optimal balance of capability and efficiency for math reasoning
- Strong performance in OCR and mathematical notation recognition
- Excellent at multi-step problem solving and educational explanations
- Cost-effective for deployment at scale
- Superior handling of mathematical notation and formatting

**Parameter Recommendations:**
- OCR Processing: `temperature=0.1` (low creativity for accurate extraction)
- Problem Classification: `temperature=0.0` (deterministic classification)
- Solution Code Generation: `temperature=0.2` (slight creativity for problem-solving approaches)
- Educational Explanation: `temperature=0.3` (moderate creativity for clear explanations)

## Solution Examples

### Example 1: Algebraic Equation

**Original Problem Image:** Q1.jpeg
**Problem Text:** Solve for x: 3x + 5 = 17

**Generated Solution:**
```python
# Import libraries
import sympy as sp

# Define the variable
x = sp.Symbol('x')

# Define the equation: 3x + 5 = 17
equation = sp.Eq(3*x + 5, 17)

# Solve the equation
solution = sp.solve(equation, x)

# Print the result
print("Solution:")
print(f"x = {solution[0]}")
```

**Educational Explanation:**
```
Step 1: Understand the problem
We need to solve for the value of x in the equation: 
$$3x + 5 = 17$$

Step 2: Isolate the variable term
First, we subtract 5 from both sides of the equation:
$$3x + 5 - 5 = 17 - 5$$
$$3x = 12$$

Step 3: Divide both sides by the coefficient of x
Now, we divide both sides by 3:
$$\frac{3x}{3} = \frac{12}{3}$$
$$x = 4$$

Final Answer:
The solution to the equation 3x + 5 = 17 is x = 4.

Verification:
Let's substitute x = 4 back into the original equation:
$$3(4) + 5 = 17$$
$$12 + 5 = 17$$
$$17 = 17$$
The equation is satisfied, confirming our answer is correct.
```

### Example 2: Calculus Problem

**Original Problem Image:** Q2.jpeg
**Problem Text:** Find the derivative of f(x) = x³ - 4x² + 7x - 2

**Generated Solution:**
```python
# Import libraries
import sympy as sp

# Define the variable
x = sp.Symbol('x')

# Define the function f(x) = x³ - 4x² + 7x - 2
f = x**3 - 4*x**2 + 7*x - 2

# Calculate the derivative
f_prime = sp.diff(f, x)

# Simplify the result
f_prime_simplified = sp.simplify(f_prime)

# Print the results
print("Original function:")
print(f"f(x) = {f}")
print("\nDerivative:")
print(f"f'(x) = {f_prime_simplified}")
```

**Educational Explanation:**
```
Step 1: Identify the function
Given function: $$f(x) = x^3 - 4x^2 + 7x - 2$$

Step 2: Apply the derivative rules
We'll use the power rule and linearity of differentiation to find the derivative. The power rule states that for any term $x^n$, the derivative is $nx^{n-1}$.

For each term:
- $\frac{d}{dx}(x^3) = 3x^2$
- $\frac{d}{dx}(-4x^2) = -4 \cdot 2x = -8x$
- $\frac{d}{dx}(7x) = 7$
- $\frac{d}{dx}(-2) = 0$ (as the derivative of a constant is zero)

Step 3: Combine the derivatives of each term
$$f'(x) = 3x^2 - 8x + 7$$

Final Answer:
The derivative of $f(x) = x^3 - 4x^2 + 7x - 2$ is $f'(x) = 3x^2 - 8x + 7$
```

## Implementation Details

### Error Handling

The Nerd AI system includes robust error handling at each stage:

1. **Image Validation Errors:**
   - Image format not supported
   - File too large
   - Image not readable or corrupted

2. **OCR Extraction Errors:**
   - Text not recognized
   - Mathematical notation unclear
   - Multiple problems detected

3. **Solution Generation Errors:**
   - Problem type not supported
   - Problem too complex
   - Inconsistent or incomplete information

### Performance Considerations

- Average processing time: 5-15 seconds per problem
- OCR accuracy: >95% for clear images
- Solution accuracy: >90% across supported problem types
- Handling of complex notation may require additional processing time
- Resource usage optimized for parallel processing

## Future Enhancements

1. **Extended Problem Types:**
   - Support for more advanced mathematics (e.g., differential equations, linear algebra)
   - Multi-part problems
   - Word problems with mathematical components

2. **Interactive Solutions:**
   - Step-by-step interactive walkthroughs
   - Alternative solution methods
   - Practice problems generation

3. **Performance Improvements:**
   - Faster OCR processing
   - Specialized mathematical notation handling
   - Caching common problem types 