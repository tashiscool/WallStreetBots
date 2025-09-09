#!/usr/bin/env python3
"""
WallStreetBots Launcher - Cross-platform executable launcher
This script acts like a .exe/.bat file to launch the trading system
"""

import sys
import os
import subprocess
import platform
from pathlib import Path

class WallStreetBotsLauncher:
    def __init__(self):
        self.base_dir=Path(__file__).parent
        self.system=platform.system().lower()
        self.venv_path=self.base_dir / "venv"
        
    def get_python_executable(self):
        """Get the correct Python executable path for the platform"""
        if self.system== "windows":python_exe = self.venv_path / "Scripts" / "python.exe"
            if not python_exe.exists():
                python_exe=self.venv_path / "Scripts" / "python3.exe"
        else:  # macOS/Linux
            python_exe = self.venv_path / "bin" / "python"
            if not python_exe.exists():
                python_exe=self.venv_path / "bin" / "python3"
        
        # Fallback to system Python if venv doesn't exist
        if not python_exe.exists():
            return "python3" if self.system != "windows" else "python"
        
        return str(python_exe)
    
    def check_environment(self):
        """Check if the environment is properly set up"""
        print("🔍 Checking WallStreetBots Environment...")
        
        # Check if we're in the right directory
        required_files=["manage.py", "simple_bot.py", "requirements.txt"]
        missing_files=[f for f in required_files if not (self.base_dir / f).exists()]
        
        if missing_files:
            print(f"❌ Missing required files: {', '.join(missing_files)}")
            print(f"   Make sure you're running this from the WallStreetBots root directory")
            return False
        
        # Check Python
        python_exe=self.get_python_executable()
        try:
            result=subprocess.run([python_exe, "--version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode== 0:
                print(f"✅ Python: {result.stdout.strip()}")
            else:
                print(f"❌ Python check failed")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"❌ Python executable not found: {python_exe}")
            print(f"   Error: {e}")
            return False
        
        # Check .env file
        env_file=self.base_dir / ".env"
        if not env_file.exists():
            print("⚠️ .env file not found - you may need to configure API keys")
            print("   Copy .env.example to .env and add your credentials")
        else:
            print("✅ .env file found")
        
        return True
    
    def install_dependencies(self):
        """Install required dependencies"""
        print("\n📦 Installing dependencies...")
        python_exe=self.get_python_executable()
        
        try:
            # Upgrade pip first
            subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", "pip"], 
                          check=True, cwd=self.base_dir)
            
            # Install requirements
            subprocess.run([python_exe, "-m", "pip", "install", "-r", "requirements.txt"], 
                          check=True, cwd=self.base_dir)
            
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    def show_menu(self):
        """Display the main menu"""
        print("\n" + "="*50)
        print("🤖 WallStreetBots Control Center")
        print("="*50)
        print("1. 🚀 Start Simple Trading Bot (Paper Trading)")
        print("2. 💰 Start Simple Trading Bot (Real Money) [DANGER]")
        print("3. 🧪 Run Risk Model Tests")
        print("4. 📊 Run Advanced Feature Tests")
        print("5. 🔧 Django Admin Panel")
        print("6. 📈 Demo Risk Models")
        print("7. 🛠️ Setup/Install Dependencies")
        print("8. 🔍 System Status Check")
        print("9. ❌ Exit")
        print("="*50)
    
    def run_simple_bot(self, real_money=False):
        """Run the simple trading bot"""
        python_exe=self.get_python_executable()
        
        if real_money:
            print("⚠️  WARNING: REAL MONEY MODE!")
            print("⚠️  This will trade with real money!")
            confirm=input("Type 'YES I UNDERSTAND' to continue: ")
            if confirm != "YES I UNDERSTAND":print("❌ Cancelled for safety")
                return
        
        try:
            env=os.environ.copy()
            env["DJANGO_SETTINGS_MODULE"] = "backend.settings"
            
            if real_money:
                # Set environment variable to disable paper trading
                env["WALLSTREETBOTS_REAL_MONEY"] = "true"
            
            print(f"🚀 Starting {'Real Money' if real_money else 'Paper'} Trading Bot...")
            subprocess.run([python_exe, "simple_bot.py"], 
                          cwd=self.base_dir, env=env)
                          
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
        except subprocess.CalledProcessError as e:
            print(f"❌ Bot failed to start: {e}")
    
    def run_tests(self, test_type="risk"):
        """Run various test suites"""
        python_exe=self.get_python_executable()
        
        test_files={
            "risk":"test_month_5_6_advanced_features.py",
            "advanced":"test_integrated_advanced_risk_system.py",
            "complete":"test_complete_risk_bundle.py"
        }
        
        test_file=test_files.get(test_type, test_files["risk"])
        
        try:
            env=os.environ.copy()
            env["DJANGO_SETTINGS_MODULE"] = "backend.settings"
            
            print(f"🧪 Running {test_type} tests...")
            subprocess.run([python_exe, test_file], 
                          cwd=self.base_dir, env=env)
                          
        except subprocess.CalledProcessError as e:
            print(f"❌ Tests failed: {e}")
    
    def run_django_admin(self):
        """Start Django admin panel"""
        python_exe=self.get_python_executable()
        
        try:
            env=os.environ.copy()
            env["DJANGO_SETTINGS_MODULE"] = "backend.settings"
            
            print("🔧 Starting Django development server...")
            print("   Access admin at: http://localhost:8000/admin/")
            subprocess.run([python_exe, "manage.py", "runserver"], 
                          cwd=self.base_dir, env=env)
                          
        except KeyboardInterrupt:
            print("\n🛑 Django server stopped")
        except subprocess.CalledProcessError as e:
            print(f"❌ Django failed to start: {e}")
    
    def run_demo_risk_models(self):
        """Run the risk models demo"""
        python_exe=self.get_python_executable()
        
        try:
            env=os.environ.copy()
            env["DJANGO_SETTINGS_MODULE"] = "backend.settings"
            
            print("📊 Running Risk Models Demo...")
            subprocess.run([python_exe, "demo_risk_models.py"], 
                          cwd=self.base_dir, env=env)
                          
        except subprocess.CalledProcessError as e:
            print(f"❌ Demo failed: {e}")
    
    def system_status(self):
        """Show detailed system status"""
        print("\n🔍 System Status Check")
        print("="*30)
        
        # Platform info
        print(f"Platform: {platform.system()} {platform.release()}")
        print(f"Python: {sys.version}")
        
        # Directory check
        print(f"Working Directory: {self.base_dir}")
        
        # Virtual environment
        if self.venv_path.exists():
            print(f"✅ Virtual environment: {self.venv_path}")
        else:
            print(f"❌ Virtual environment not found: {self.venv_path}")
        
        # Key files
        key_files=[
            "manage.py", "simple_bot.py", "requirements.txt", 
            ".env", "README.md", "backend/settings.py"
        ]
        
        print("\nKey Files:")
        for file in key_files:
            path=self.base_dir / file
            status = "✅" if path.exists() else "❌"
            print(f"  {status} {file}")
        
        # Database files
        db_files=["db.sqlite3", "compliance.db", "risk_database.db"]
        print("\nDatabase Files:")
        for db in db_files:
            path=self.base_dir / db
            status = "✅" if path.exists() else "⚪"
            size=f" ({path.stat().st_size} bytes)" if path.exists() else ""
            print(f"  {status} {db}{size}")
    
    def run(self):
        """Main launcher loop"""
        print("🤖 WallStreetBots Launcher")
        print("="*30)
        
        # Check if running in interactive mode
        is_interactive=sys.stdin.isatty()
        
        # Initial environment check
        if not self.check_environment():
            print("\n❌ Environment check failed!")
            if is_interactive:
                try:
                    response=input("Would you like to try installing dependencies? (y/n): ")
                    if response.lower() == 'y':if not self.install_dependencies():
                            print("❌ Setup failed. Exiting.")
                            sys.exit(1)
                    else:
                        sys.exit(1)
                except (EOFError, KeyboardInterrupt):
                    print("\n❌ Setup required. Exiting.")
                    sys.exit(1)
            else:
                print("❌ Non-interactive mode: automatic dependency installation...")
                if not self.install_dependencies():
                    print("❌ Setup failed. Exiting.")
                    sys.exit(1)
        
        # Handle non-interactive mode
        if not is_interactive:
            print("\n🤖 Non-interactive mode detected")
            print("Available commands:")
            print("  --status    : Show system status")
            print("  --test      : Run risk tests")
            print("  --demo      : Run demo")
            print("  --help      : Show this help")
            
            if len(sys.argv) > 1:
                arg=sys.argv[1]
                if arg == "--status":self.system_status()
                elif arg== "--test":self.run_tests("risk")
                elif arg== "--demo":self.run_demo_risk_models()
                elif arg== "--help":print("\nUse the interactive launcher: python3 run_wallstreetbots.py")
                else:
                    print(f"Unknown argument: {arg}")
            else:
                print("\n✅ Environment check passed!")
                print("To use interactively, run in a terminal with input capability.")
            return
        
        # Main menu loop (interactive mode)
        while True:
            try:
                self.show_menu()
                choice=input("\nSelect option (1-9): ").strip()
                
                if choice== "1":self.run_simple_bot(real_money=False)
                elif choice== "2":self.run_simple_bot(real_money=True)
                elif choice== "3":self.run_tests("risk")
                elif choice== "4":self.run_tests("advanced")
                elif choice== "5":self.run_django_admin()
                elif choice== "6":self.run_demo_risk_models()
                elif choice== "7":self.install_dependencies()
                elif choice== "8":self.system_status()
                elif choice== "9":print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please select 1-9.")
                
                if choice in ["1", "2", "3", "4", "5", "6"]:
                    try:
                        input("\nPress Enter to return to menu...")
                    except (EOFError, KeyboardInterrupt):
                        break
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except EOFError:
                print("\n👋 Input ended. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                try:
                    input("Press Enter to continue...")
                except (EOFError, KeyboardInterrupt):
                    break

if __name__== "__main__":launcher = WallStreetBotsLauncher()
    launcher.run()