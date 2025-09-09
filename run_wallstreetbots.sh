#!/bin/bash
# WallStreetBots macOS/Linux Launcher
# This script acts like an executable to launch the trading system

set -e  # Exit on any error

echo ""
echo "========================================"
echo "   WallStreetBots macOS/Linux Launcher"
echo "========================================"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Python is available
if command_exists python3; then
    PYTHON_CMD="python3"
elif command_exists python; then
    PYTHON_CMD="python"
else
    echo "❌ ERROR: Python is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
echo "✅ Found Python $PYTHON_VERSION"

# Check if we're in the right directory
if [[ ! -f "manage.py" ]]; then
    echo "❌ ERROR: This script must be run from the WallStreetBots root directory"
    echo "Current directory: $(pwd)"
    echo "Please navigate to the WallStreetBots folder and try again"
    exit 1
fi

# Make sure the Python launcher is executable
chmod +x run_wallstreetbots.py

echo "🚀 Launching WallStreetBots..."
echo ""

# Run the Python launcher
$PYTHON_CMD run_wallstreetbots.py