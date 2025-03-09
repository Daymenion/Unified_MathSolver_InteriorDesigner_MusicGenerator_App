#!/usr/bin/env python
"""
Test runner for the Codeway AI Suite.

This script runs tests for all components of the AI suite.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Ensure the project root is in the Python path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Codeway AI Suite Test Runner")
    parser.add_argument(
        "--test", 
        nargs="*", 
        help="Specific tests to run, e.g. --test environment openai_api"
    )
    parser.add_argument(
        "--skip-slow", 
        action="store_true", 
        help="Skip slow tests (e.g. those making many API calls)"
    )
    return parser.parse_args()

def run_tests(args):
    """Run the test suite."""
    # Create command to run the test suite
    command = ["python", "-m", "tests.test_suite"]
    
    # Add specific tests if provided
    if args.test:
        command.extend(["--test"] + args.test)
    
    # Skip slow tests if requested
    if args.skip_slow:
        command.append("--skip-slow")
    
    # Run the command
    result = subprocess.run(command, shell=False)
    return result.returncode

def main():
    """Main entry point."""
    args = parse_args()
    
    # Print header
    print("=" * 38)
    print("Codeway AI Suite Test Runner")
    print("=" * 38)
    
    # Print test parameters
    if args.test:
        print(f"Running tests with custom args: --test {' '.join(args.test)}")
    else:
        print("Running all tests with default settings (skipping slow tests)")
        print("To run specific tests, use: python run_tests.py --test openai_api nerd_ai_ocr")
    
    print("=" * 38)
    
    # Run the tests
    return run_tests(args)

if __name__ == "__main__":
    sys.exit(main()) 