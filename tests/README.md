# Codeway AI Suite Testing Framework

This directory contains comprehensive testing tools for the Codeway AI Suite applications. The testing framework verifies the functionality of all three applications:

1. **Nerd AI** - Math Problem Solver
2. **Interior Design App** - Room Style Transformer
3. **Music Generator** - Lyrics and Cover Art Generator

## Test Structure

The testing framework includes:

- Individual unit tests for each component
- Integration tests for component interactions
- End-to-end tests for complete application workflows
- Performance and reliability tests

## Running Tests

### Basic Usage

To run all tests:

```bash
python tests/test_suite.py
```

### Command Line Options

The test suite supports several command-line options:

```bash
# Run specific tests
python tests/test_suite.py --test openai_api nerd_ai_ocr

# Skip slow tests
python tests/test_suite.py --skip-slow

# Set log level
python tests/test_suite.py --log-level DEBUG
```

Available test options:
- Common: `environment`, `openai_api`, `utilities`
- Nerd AI: `nerd_ai_ocr`, `nerd_ai_classification`, `nerd_ai_solution`, `nerd_ai_e2e`
- Interior Design: `interior_room_detection`, `interior_prompt`, `interior_transform`
- Music Generator: `music_lyrics`, `music_cover_art`, `music_e2e`

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