#!/usr/bin/env python
"""
Test runner for the Daymenion AI Suite.

This script runs tests for all components of the AI suite.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Ensure the project root is in the Python path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Daymenion AI Suite Test Runner")
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
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
        default="INFO",
        help="Set logging level"
    )
    return parser.parse_args()

def run_tests(args):
    """Run the test suite with the specified arguments."""
    # Build the command to run the test suite
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "test_suite.py")]
    
    # Add test selection if specified
    if args.test:
        cmd.extend(["--test"] + args.test)
        
    # Add skip-slow flag if specified
    if args.skip_slow:
        cmd.append("--skip-slow")
        
    # Add log level
    cmd.extend(["--log-level", args.log_level])
    
    # Run the test suite
    return subprocess.call(cmd)

def main():
    """Main entry point."""
    args = parse_args()
    sys.exit(run_tests(args))

if __name__ == "__main__":
    main() 