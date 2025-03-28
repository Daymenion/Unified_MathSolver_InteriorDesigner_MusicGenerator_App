# Daymenion AI Suite

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
│   ├── annotator/       # ControlNet annotators (MLSD, etc.)
│   ├── utils/           # Interior design utilities
│   └── style_transformer.py  # Core transformation engine
├── music_generator/     # Music generation application
├── frontend/            # Streamlit UI components
├── tests/               # Test suite for all components
│   ├── test_suite.py    # Comprehensive test suite
│   ├── showcase.py      # Demo of all three applications
│   ├── test_hf_integration.py  # Hugging Face API integration tests
│   └── run_tests.py     # Test runner script
├── workflows/           # Business stakeholder workflow documentation
├── data/                # Data directory for samples and outputs
├── app.py               # Main application entry point
├── run.py               # Command-line runner
├── requirements.txt     # Project dependencies
└── README.md            # This file
```

## Setup Instructions

1. **Clone the repository**
   ```
   git clone https://github.com/Daymenion/Unified_MathSolver_InteriorDesign_MusicGenerator_App.git
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
   python run.py app
   ```

## Features

### Nerd AI
- Scan math problems using OCR
- Solve various types of mathematical problems (algebra, calculus, statistics, etc.)
- Provide step-by-step explanations with properly formatted mathematical notation
- Export solutions as formatted text

### Interior Design App
- Transform room styles using AI image generation
- Support for multiple design styles (Modern, Minimalist, Industrial, etc.)
- Maintain original room layout while changing style elements
- **NEW**: Now supports both API-based and local Stable Diffusion processing methods:
  - **API Method**: Uses Hugging Face Inference API with SDXL for high-quality transformations
  - **Local Method**: Uses Stable Diffusion with ControlNet MLSD for structure-preserving transformations on your own GPU
- High-quality image transformations with configurable parameters
- Enhanced UI with advanced settings for customization

### Music Generator
- Create personalized song lyrics based on genre, mood, and purpose
- Generate matching cover art with customizable themes
- Structured song output with verses, chorus, and other sections
- Export lyrics and artwork in various formats

## Testing

The testing framework has been organized for better maintainability and organization:

### Running Tests

Run the comprehensive test suite:

```
python tests/run_tests.py
```

To run specific tests:

```
python tests/run_tests.py --test openai_api nerd_ai_ocr
```

To skip slow tests:

```
python tests/run_tests.py --skip-slow
```

### Available Tests

- **Common Components**: `environment`, `openai_api`, `utilities`
- **Nerd AI**: `nerd_ai_ocr`, `nerd_ai_classification`, `nerd_ai_solution`
- **Interior Design**: `interior_room_detection`, `interior_prompt`, `interior_transform`
- **Music Generator**: `music_lyrics`, `music_cover_art`, `music_package`

### Showcase Demo

Run the showcase demo to see all three applications in action:

```
python tests/showcase.py
```

### Hugging Face API Integration Tests

Test the Hugging Face API integration for image generation:

```
python tests/test_hf_integration.py
```

## Workflows

The application supports the following key workflows:

1. **Nerd AI Workflow**:
   - Upload a math problem image or enter text
   - Problem is parsed using OCR (if image)
   - Problem type is automatically identified
   - Step-by-step solution is generated
   - Solution is displayed with properly formatted math notation

2. **Interior Design Workflow** (Updated):
   - Upload a room image or select a sample
   - Select desired style transformation
   - Choose processing method (API or Local Stable Diffusion)
   - Adjust advanced settings (strength, steps, resolution)
   - Room is analyzed and key elements are identified
   - AI generates a detailed transformation prompt
   - Transformed image is generated maintaining the original layout
   - Download the transformed image

3. **Music Generator Workflow**:
   - Select music genre, mood, and purpose
   - Add custom description (optional)
   - AI generates structured lyrics with title, verses, and chorus
   - Cover art is generated to match the song theme
   - Download lyrics and cover art

For detailed workflow documentation, see the `workflows/` directory.

## Interior Design App - Detailed Overview

The Interior Design app has been significantly enhanced with dual processing options and improved customization. Here's a detailed breakdown:

### Processing Methods

#### 1. Hugging Face API Method (Cloud-based)
- Uses SDXL Refiner model through Hugging Face's Inference API
- Requires an API key and internet connection
- Fast processing with minimal local resource usage
- Subject to API rate limits and quotas

#### 2. Local Stable Diffusion Method (GPU-based)
- Uses Stable Diffusion with ControlNet MLSD for structure preservation
- Runs entirely on your local GPU (CUDA) or CPU
- No internet connection required after initial model download
- Unlimited usage with no rate limits
- Requires more local resources (RAM, VRAM, etc.)

### Key Features

- **Room Type Detection**: Automatically identifies the type of room in your image
- **Multi-Style Support**: Transform your space into various styles including Modern, Soho, Gothic, and more
- **Style Preservation**: Maintains structural elements while completely transforming design aspects
- **Advanced Customization**:
  - Transformation strength (0.1-1.0)
  - Inference steps (20-100)
  - Image resolution (384-768)
- **Prompt Generation**: Uses AI to create detailed prompts for consistent style transformations
- **Downloadable Results**: Save your transformed room images locally

### System Requirements

For local processing:
- **GPU Mode**: NVIDIA GPU with CUDA support, 6GB+ VRAM recommended
- **CPU Mode**: Will work but processing is much slower
- **RAM**: 8GB minimum, 16GB+ recommended
- **Disk Space**: ~4GB for model files (downloaded on first use)

### Configuration Options

The Interior Design app can be configured through environment variables or the `common/config.py` file:

```python
INTERIOR_DESIGN_SETTINGS = {
    "supported_styles": ["Modern", "Soho", "Gothic", "Minimalist", "Industrial", "Vintage", "Contemporary"],
    "supported_rooms": ["Living Room", "Kitchen", "Bedroom", "Bathroom", "Dining Room", "Office"],
    "image_generation": {
        "guidance_scale": 8.0,
        "steps": 50,
        "strength": 0.8
    }
}
```

## Error Handling and Robustness

The latest version includes improved error handling across all components:

- Comprehensive input validation for all user inputs
- Graceful failure modes with informative error messages
- Fallback mechanisms when API services are unavailable
- Automatic handling of edge cases in content generation
- User-friendly error messages with troubleshooting suggestions

## Future Development

Planned enhancements for future versions:

- Additional math problem types and solution methods
- More interior design styles and room customization options
- Enhanced 3D visualization for interior design transformations
- Advanced music generation features (melody, chords, etc.)
- User accounts and saved project functionality
- Mobile-friendly responsive design

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Hugging Face API Integration

The AI Suite includes integration with Hugging Face's Inference API, which allows you to use powerful diffusion models for image generation without downloading large model files locally. This is implemented in both the Interior Design Style Transformer and Music Generator modules.

### Features

- **Text-to-Image Generation**: Create high-quality images from text descriptions using state-of-the-art models like Stable Diffusion XL.
- **Image-to-Image Transformation**: Transform existing images based on text prompts, particularly useful for the Interior Design module.
- **Fallback Mechanism**: If the Hugging Face API is not available or fails, the system gracefully falls back to local processing.

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

For more intensive usage, consider upgrading to Hugging Face Pro, using the local processing option, or implementing your own model hosting.

### Testing

You can test the Hugging Face API integration using the included test script:

```bash
python test_hf_integration.py
```

This will run both text-to-image and image-to-image tests, saving the results to the `data/test_outputs` directory.

### Models Used

- **Interior Design**: 
  - API Method: Uses `stabilityai/stable-diffusion-xl-refiner-1.0` for high-quality transformations
  - Local Method: Uses `runwayml/stable-diffusion-v1-5` with `lllyasviel/sd-controlnet-mlsd` for structure preservation

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