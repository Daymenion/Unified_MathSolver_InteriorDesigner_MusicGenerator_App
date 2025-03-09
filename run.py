#!/usr/bin/env python3
"""
Unified run script for the Daymenion AI Suite.

This script provides a unified way to run the different components of the AI Suite:
- App: The main Streamlit application
- Showcase: Demonstration of AI capabilities
- Tests: Run the test suite
"""

import os
import sys
import argparse
import subprocess
import traceback
from pathlib import Path

# Ensure proper path setup
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Python path set to: {project_root}")

def run_app():
    """Run the Streamlit application."""
    try:
        print("Running Streamlit app...")
        import streamlit.web.cli as stcli
        
        # Use subprocess to avoid Streamlit's sys.argv manipulation affecting our script
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", 
             os.path.join(project_root, "app.py"),
             "--browser.serverAddress", "localhost",
             "--browser.gatherUsageStats", "false"],
            check=True
        )
        return True
    except ImportError:
        print("ERROR: Streamlit is not installed. Please install it with 'pip install streamlit'.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Streamlit app failed to start: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Failed to run app: {str(e)}")
        traceback.print_exc()
        return False

def run_showcase():
    """Run the showcase demonstration."""
    cmd = [sys.executable, os.path.join("tests", "showcase.py")]
    print("Running showcase demonstration...")
    return subprocess.run(cmd).returncode

def run_tests(args=None):
    """Run the test suite with optional arguments."""
    cmd = [sys.executable, os.path.join("tests", "run_tests.py")]
    
    # If additional args are provided, add them to the command
    if args:
        cmd.extend(args)
        
    print("Running test suite...")
    return subprocess.run(cmd).returncode

def run_hf_tests():
    """Run Hugging Face API integration tests."""
    cmd = [sys.executable, os.path.join("tests", "test_hf_integration.py")]
    print("Running Hugging Face API integration tests...")
    return subprocess.run(cmd).returncode

def install_dependencies():
    """Install required dependencies."""
    try:
        print("Installing dependencies...")
        requirements_file = os.path.join(project_root, "requirements.txt")
        
        if not os.path.exists(requirements_file):
            print("WARNING: requirements.txt not found. Creating basic requirements file.")
            with open(requirements_file, "w") as f:
                f.write("streamlit>=1.28.0\n")
                f.write("pillow>=9.0.0\n")
                f.write("numpy>=1.20.0\n")
                f.write("huggingface_hub>=0.18.0\n")
                f.write("python-dotenv>=0.20.0\n")
        
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", requirements_file],
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install dependencies: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Unexpected error during dependency installation: {str(e)}")
        traceback.print_exc()
        return False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Daymenion AI Suite Runner")
    
    # Define commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # App command
    app_parser = subparsers.add_parser("app", help="Run the Streamlit app")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--test", nargs="*", help="Specific tests to run")
    test_parser.add_argument("--skip-slow", action="store_true", help="Skip slow tests")
    test_parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                            default="INFO", help="Set logging level")
    
    # Showcase command
    subparsers.add_parser("showcase", help="Run the showcase demonstration")
    
    # HF tests command
    subparsers.add_parser("hf-test", help="Run Hugging Face API integration tests")
    
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_args()
    
    if args.command == "app":
        # Run the Streamlit app
        success = run_app()
    elif args.command == "test":
        # Run tests
        test_args = []
        if hasattr(args, "test") and args.test:
            test_args.extend(["--test"] + args.test)
        if hasattr(args, "skip_slow") and args.skip_slow:
            test_args.append("--skip-slow")
        if hasattr(args, "log_level"):
            test_args.extend(["--log-level", args.log_level])
            
        success = run_tests(test_args) == 0
    elif args.command == "showcase":
        # Run showcase
        success = run_showcase() == 0
    elif args.command == "hf-test":
        # Run Hugging Face tests
        success = run_hf_tests() == 0
    else:
        # No command specified, show help
        print("Please specify a command. Use --help for more information.")
        return 1
    
    return 0 if success else 1

if __name__ == "__main__":
    main() 