#!/usr/bin/env python3
"""
Create executable for WallStreetBots
This script creates platform-specific executables and shortcuts
"""

import os
import sys
import platform
import shutil
from pathlib import Path

def create_windows_executable():
    """Create Windows .exe using PyInstaller"""
    try:
        import PyInstaller
    except ImportError:
        print("Installing PyInstaller...")
        os.system("pip install pyinstaller")
    
    print("Creating Windows executable...")
    
    # PyInstaller command
    cmd = [
        "pyinstaller",
        "--onefile",
        "--noconsole",
        "--name=WallStreetBots",
        "--icon=icon.ico" if Path("icon.ico").exists() else "",
        "run_wallstreetbots.py"
    ]
    
    cmd = [c for c in cmd if c]  # Remove empty strings
    
    os.system(" ".join(cmd))
    print("✅ Windows executable created in dist/ folder")

def create_macos_app():
    """Create macOS .app bundle"""
    try:
        import py2app
    except ImportError:
        print("Installing py2app...")
        os.system("pip install py2app")
    
    print("Creating macOS .app bundle...")
    
    # Create setup.py for py2app
    setup_py = '''
from setuptools import setup

APP = ['run_wallstreetbots.py']
DATA_FILES = []
OPTIONS = {
    'argv_emulation': True,
    'plist': {
        'CFBundleName': 'WallStreetBots',
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleVersion': '1.0.0',
        'CFBundleIdentifier': 'com.wallstreetbots.launcher',
    }
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
'''
    
    with open("setup_py2app.py", "w") as f:
        f.write(setup_py)
    
    os.system("python setup_py2app.py py2app")
    print("✅ macOS .app created in dist/ folder")

def create_desktop_shortcut():
    """Create desktop shortcut"""
    system = platform.system().lower()
    
    if system == "windows":
        # Windows .lnk shortcut
        try:
            import win32com.client
            
            desktop = Path.home() / "Desktop"
            shortcut_path = desktop / "WallStreetBots.lnk"
            
            shell = win32com.client.Dispatch("WScript.Shell")
            shortcut = shell.CreateShortCut(str(shortcut_path))
            shortcut.Targetpath = str(Path.cwd() / "run_wallstreetbots.bat")
            shortcut.WorkingDirectory = str(Path.cwd())
            shortcut.Description = "WallStreetBots Trading System"
            shortcut.save()
            
            print(f"✅ Desktop shortcut created: {shortcut_path}")
            
        except ImportError:
            print("⚠️ Could not create Windows shortcut (pywin32 not available)")
            print("   You can manually create a shortcut to run_wallstreetbots.bat")
    
    elif system == "darwin":  # macOS
        desktop = Path.home() / "Desktop"
        shortcut_path = desktop / "WallStreetBots.command"
        
        script_content = f'''#!/bin/bash
cd "{Path.cwd()}"
./run_wallstreetbots.sh
'''
        
        with open(shortcut_path, "w") as f:
            f.write(script_content)
        
        os.chmod(shortcut_path, 0o755)
        print(f"✅ Desktop shortcut created: {shortcut_path}")
    
    else:  # Linux
        desktop = Path.home() / "Desktop"
        shortcut_path = desktop / "WallStreetBots.desktop"
        
        desktop_entry = f'''[Desktop Entry]
Version=1.0
Type=Application
Name=WallStreetBots
Comment=WallStreetBots Trading System
Exec={Path.cwd() / "run_wallstreetbots.sh"}
Path={Path.cwd()}
Icon=utilities-terminal
Terminal=true
Categories=Office;Finance;
'''
        
        with open(shortcut_path, "w") as f:
            f.write(desktop_entry)
        
        os.chmod(shortcut_path, 0o755)
        print(f"✅ Desktop shortcut created: {shortcut_path}")

def main():
    """Main function"""
    print("🔧 WallStreetBots Executable Creator")
    print("="*40)
    
    system = platform.system().lower()
    
    print("What would you like to create?")
    print("1. Desktop shortcut (recommended)")
    print("2. Standalone executable (advanced)")
    print("3. Both")
    print("4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1" or choice == "3":
        create_desktop_shortcut()
    
    if choice == "2" or choice == "3":
        if system == "windows":
            create_windows_executable()
        elif system == "darwin":
            create_macos_app()
        else:
            print("⚠️ Standalone executables not supported on Linux")
            print("   Use the desktop shortcut or run ./run_wallstreetbots.sh directly")
    
    elif choice == "4":
        print("👋 Goodbye!")
        return
    
    print("\n✅ Done!")
    print("\nYou can now launch WallStreetBots using:")
    if system == "windows":
        print("  • Double-click the desktop shortcut")
        print("  • Double-click run_wallstreetbots.bat")
        print("  • Run 'python run_wallstreetbots.py' in terminal")
    else:
        print("  • Double-click the desktop shortcut")
        print("  • Run './run_wallstreetbots.sh' in terminal")
        print("  • Run 'python3 run_wallstreetbots.py' in terminal")

if __name__ == "__main__":
    main()