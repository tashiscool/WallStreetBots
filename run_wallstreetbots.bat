@echo off
REM WallStreetBots Windows Launcher
REM This .bat file acts like a .exe to launch the trading system

title WallStreetBots Launcher

echo.
echo ========================================
echo    WallStreetBots Windows Launcher
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "manage.py" (
    echo ERROR: This script must be run from the WallStreetBots root directory
    echo Current directory: %CD%
    echo Please navigate to the WallStreetBots folder and try again
    pause
    exit /b 1
)

REM Run the Python launcher
python run_wallstreetbots.py

pause