# Codeway AI Suite

A unified application suite featuring three AI-powered tools:

1. **Nerd AI**: Math problem solver with image scanning capabilities
2. **Interior Design App**: Transform room styles with AI
3. **Music Generator**: Create personalized song lyrics and cover art

## Project Structure

```
ai_suite/
├── common/              # Shared utilities and components
├── nerd_ai/             # Math solver application
├── interior_design/     # Interior design application
├── music_generator/     # Music generation application
├── frontend/            # Streamlit UI components
├── workflows/           # Business stakeholder workflow documentation
├── tests/               # Test suite for all components
├── app.py               # Main application entry point
├── run_tests.py         # Test runner script
├── requirements.txt     # Project dependencies
└── README.md            # This file
```

## Setup Instructions

1. **Clone the repository**
   ```
   git clone https://github.com/your-username/codeway-ai-suite.git
   cd codeway-ai-suite
   ```

2. **Create a virtual environment**
   ```
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - Windows:
     ```
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

5. **Set up environment variables**
   Create a `.env` file in the root directory with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   HUGGINGFACE_API_KEY=your_huggingface_api_key
   ```

6. **Run the application**
   ```
   streamlit run app.py
   ```

## Features

### Nerd AI
- Scan math problems using OCR
- Solve various types of mathematical problems
- Provide step-by-step explanations

### Interior Design App
- Transform room styles through AI
- Support for multiple design styles (Modern, Soho, Gothic)
- Maintain original room layout while changing style elements

### Music Generator
- Create personalized song lyrics based on genre, mood, and purpose
- Generate matching cover art
- Export lyrics and artwork

## Testing

Run the test suite to verify all components are working correctly:

```
python run_tests.py
```

To run specific tests:

```
python run_tests.py --test openai_api nerd_ai_ocr
```

To skip slow tests:

```
python run_tests.py --skip-slow
```

## Recent Updates

### Version 1.1.0 (March 2025)

- **OpenAI API Client Improvements**:
  - Replaced the old OpenAI client with a more robust httpx-based implementation
  - Added proper retry logic with exponential backoff
  - Enhanced error handling and reporting
  - Fixed proxy-related issues

- **Code Structure Improvements**:
  - Fixed import issues across all modules
  - Standardized on absolute imports for better reliability
  - Enhanced image validation to handle different file object types
  - Improved test suite with better file handling

- **Dependencies**:
  - Added httpx for more reliable API communication
  - Updated requirements.txt with specific versions

## Technologies Used

- **Frontend**: Streamlit
- **Backend**: Python with FastAPI
- **AI Models**: OpenAI GPT-4o-mini, Hugging Face models
- **Image Processing**: Pillow, scikit-image
- **Mathematical Libraries**: NumPy, SciPy, SymPy
- **HTTP Client**: httpx

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Hugging Face API Integration

The AI Suite now includes integration with Hugging Face's Inference API, which allows you to use powerful diffusion models for image generation without downloading large model files locally. This is implemented in both the Interior Design Style Transformer and Music Generator modules.

### Features

- **Text-to-Image Generation**: Create high-quality images from text descriptions using state-of-the-art models like Stable Diffusion XL.
- **Image-to-Image Transformation**: Transform existing images based on text prompts, particularly useful for the Interior Design module.
- **Fallback Mechanism**: If the Hugging Face API is not available or fails, the system gracefully falls back to the original mock implementations.

### Setup

1. Create a Hugging Face account at [huggingface.co](https://huggingface.co) if you don't already have one.
2. Generate an API token from your [Hugging Face profile settings](https://huggingface.co/settings/tokens).
3. Add your token to the `.env` file:
   ```
   HUGGING_FACE_API_TOKEN=your_token_here
   HUGGINGFACE_USE_API=true
   ```

### Rate Limits

The free tier of Hugging Face's Inference API has the following rate limits:
- Approximately 5 requests per minute
- Around 50 requests per day

For more intensive usage, consider upgrading to Hugging Face Pro or implementing your own model hosting.

### Testing

You can test the Hugging Face API integration using the included test script:

```bash
python test_hf_integration.py
```

This will run both text-to-image and image-to-image tests, saving the results to the `data/test_outputs` directory.

### Models Used

- **Interior Design**: Uses `stabilityai/stable-diffusion-xl-base-1.0` for room style transformations.
- **Music Generator**: Uses `stabilityai/stable-diffusion-xl-base-1.0` for album cover art generation.

These models can be configured in `common/config.py` if you prefer to use different models.

## Running the Application

To ensure proper module imports and environment setup, always use the `run.py` script to run the application:

```bash
# Run the Streamlit application
python run.py app

# Run the showcase demonstration
python run.py showcase

# Run the test suite
python run.py test

# Test the Hugging Face integration
python run.py test_hf
```

This script sets up the correct Python path environment to ensure all modules can be imported properly. 