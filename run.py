#!/usr/bin/env python3
"""
Unified run script for the Codeway AI Suite.

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
    """Run the AI showcase demonstration."""
    try:
        print("Running showcase demonstration...")
        from showcase import run_showcase
        run_showcase()
        return True
    except ImportError:
        print("ERROR: Could not import showcase module.")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"ERROR: Showcase failed: {str(e)}")
        traceback.print_exc()
        return False

def run_tests():
    """Run the test suite."""
    try:
        print("Running test suite...")
        from tests.test_suite import run_tests
        success = run_tests()
        return success
    except ImportError:
        print("ERROR: Could not import test suite.")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"ERROR: Tests failed: {str(e)}")
        traceback.print_exc()
        return False

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

def main():
    """Main entry point for the run script."""
    parser = argparse.ArgumentParser(description="Run Codeway AI Suite components")
    parser.add_argument("mode", choices=["app", "showcase", "test", "install"],
                       help="Run mode: app (Streamlit UI), showcase (demonstration), test (run tests), install (dependencies)")
    
    args = parser.parse_args()
    
    if args.mode == "app":
        success = run_app()
    elif args.mode == "showcase":
        success = run_showcase()
    elif args.mode == "test":
        success = run_tests()
    elif args.mode == "install":
        success = install_dependencies()
    else:
        print(f"ERROR: Unknown mode: {args.mode}")
        success = False
    
    # Return appropriate exit code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 