# Daymenion AI Suite Testing Framework

This directory contains comprehensive testing tools for the Daymenion AI Suite applications. The testing framework verifies the functionality of all three applications:

1. **Nerd AI** - Math Problem Solver
2. **Interior Design App** - Room Style Transformer
3. **Music Generator** - Lyrics and Cover Art Generator

## Test Structure

The testing framework includes:

- `test_suite.py` - Comprehensive suite of unit and integration tests
- `run_tests.py` - Command-line test runner with various options
- `showcase.py` - Demonstration script of all three applications
- `test_hf_integration.py` - Hugging Face API integration tests

## Running Tests

### Basic Usage

To run all tests:

```bash
python tests/run_tests.py
```

### Command Line Options

The test suite supports several command-line options:

```bash
# Run specific tests
python tests/run_tests.py --test openai_api nerd_ai_ocr

# Skip slow tests
python tests/run_tests.py --skip-slow

# Set log level
python tests/run_tests.py --log-level DEBUG
```

Available test options:
- Common: `environment`, `openai_api`, `utilities`
- Nerd AI: `nerd_ai_ocr`, `nerd_ai_classification`, `nerd_ai_solution`, `nerd_ai_e2e`
- Interior Design: `interior_room_detection`, `interior_prompt`, `interior_transform`
- Music Generator: `music_lyrics`, `music_cover_art`, `music_e2e`

## Showcase Demo

The showcase script demonstrates all three applications with sample inputs:

```bash
python tests/showcase.py
```

This script:
- Solves sample math problems using Nerd AI
- Transforms room images with the Interior Design app
- Generates lyrics and cover art with the Music Generator
- Saves all outputs to the data/showcase directory

## Hugging Face API Integration Tests

To test Hugging Face API integration for image generation:

```bash
python tests/test_hf_integration.py
```

This script tests:
- Text-to-image generation with multiple models
- Image-to-image transformation capabilities
- API error handling and robustness

**Note:** These tests require a valid Hugging Face API key in your environment variables.

## Logging

All test logs are saved in the `logs` directory with timestamps. Each test run generates a new log file with detailed information about test execution, including:

- Test results (pass/fail)
- Detailed error messages
- Performance metrics
- API responses

Example log file: `ai_suite_test_20230308_120145.log`

## Adding New Tests

When adding new functionality to any of the applications, corresponding tests should be added to maintain code quality and prevent regressions.

1. Add test functions to the appropriate section in `test_suite.py`
2. Include the new tests in the `all_tests` dictionary
3. Update the command-line argument choices in the `main()` function

## Best Practices

- Tests should be independent and not rely on the state from other tests
- Each test should have a clear purpose and test only one aspect of functionality
- Mock external dependencies when possible to ensure consistent test results
- Include both positive (expected success) and negative (expected failure) test cases
- Maintain test coverage for all critical application paths

## Error Handling Tests

The latest version includes robust error handling tests:

- Testing invalid inputs and edge cases
- Verifying appropriate error messages
- Ensuring graceful failure modes
- Testing API connection issues and retries
- Validating recovery mechanisms 