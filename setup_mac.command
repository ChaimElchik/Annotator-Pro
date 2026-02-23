#!/bin/bash
cd "$(dirname "$0")"

echo "=============================================="
echo "Video Annotator Pro - Mac/Linux Setup"
echo "=============================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed or not in your PATH."
    echo "Please download and install it from https://www.python.org/downloads/mac-osx/"
    echo "Or install via Homebrew: brew install python3"
    echo "Press any key to exit..."
    read -n 1
    exit 1
fi

echo "Creating virtual environment (.venv)..."
python3 -m venv .venv
if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment."
    echo "Press any key to exit..."
    read -n 1
    exit 1
fi

echo "Activating virtual environment and installing dependencies..."
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install some dependencies. Please check the error messages above."
    echo "Press any key to exit..."
    read -n 1
    exit 1
fi

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "You can now close this window and double-click 'run_mac.command' to start the application."
echo "=============================================="
