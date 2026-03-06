#!/usr/bin/env python
"""
Quick startup script for Corn Cob Classification Web Server
Run this file to start the application
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("=" * 60)
    print("Corn Cob Classification System - Startup")
    print("=" * 60)
    
    project_dir = Path(__file__).parent
    
    print(f"\nProject directory: {project_dir}")
    print(f"Data directory: {project_dir / '20250319'}")
    
    # Check if data directory exists
    if not (project_dir / '20250319').exists():
        print("\n⚠ Warning: 20250319 directory not found!")
        print("Please ensure your radar data CSV files are in the 20250319 folder.")
    
    # Check if virtual environment should be used
    venv_dir = project_dir / 'venv'
    if venv_dir.exists():
        print("\n✓ Virtual environment found")
        print("Activating virtual environment...")
        if sys.platform == 'win32':
            activate_script = venv_dir / 'Scripts' / 'activate.bat'
        else:
            activate_script = venv_dir / 'bin' / 'activate'
    
    # Check requirements
    print("\nChecking dependencies...")
    try:
        import flask
        print("✓ Flask installed")
    except ImportError:
        print("✗ Flask not found. Run: pip install -r requirements.txt")
        sys.exit(1)
    
    try:
        import tensorflow
        print("✓ TensorFlow installed")
    except ImportError:
        print("✗ TensorFlow not found. Run: pip install -r requirements.txt")
        sys.exit(1)
    
    try:
        import pandas
        print("✓ Pandas installed")
    except ImportError:
        print("✗ Pandas not found. Run: pip install -r requirements.txt")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Starting Flask Web Server...")
    print("=" * 60)
    print("\nServer will start at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("\n")
    
    # Start Flask app
    os.chdir(project_dir)
    subprocess.run([sys.executable, 'app.py'])

if __name__ == '__main__':
    main()
