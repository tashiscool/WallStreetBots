#!/usr / bin / env python
"""
Test script to verify Django setup and dependencies before running migrations
"""
import sys
import os

def test_django_imports(): 
    """Test that all required Django packages can be imported"""
    print("Testing Django imports...")
    
    try: 
        import django
        print(f"✅ Django {django.get_version()} imported successfully")
    except ImportError as e: 
        print(f"❌ Django import failed: {e}")
        return False
    
    try: 
        import rest_framework
        print("✅ Django REST Framework imported successfully")
    except ImportError as e: 
        print(f"❌ Django REST Framework import failed: {e}")
        return False
    
    try: 
        import corsheaders
        print("✅ django - cors - headers imported successfully")
    except ImportError as e: 
        print(f"❌ django - cors - headers import failed: {e}")
        return False
    
    try: 
        import social_django
        print("✅ social - auth - app - django imported successfully")
    except ImportError as e: 
        print(f"❌ social - auth - app - django import failed: {e}")
        return False
    
    try: 
        import admin_interface
        print("✅ django - admin - interface imported successfully")
    except ImportError as e: 
        print(f"❌ django - admin - interface import failed: {e}")
        return False
    
    try: 
        import colorfield
        print("✅ django - colorfield imported successfully")
    except ImportError as e: 
        print(f"❌ django - colorfield import failed: {e}")
        return False
        
    return True

def test_other_dependencies(): 
    """Test other required dependencies"""
    print("\nTesting other dependencies...")
    
    dependencies = [
        ('apscheduler', 'APScheduler'),
        ('alpaca', 'Alpaca - py (Modern API)'),
        ('plotly', 'Plotly'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('yfinance', 'yfinance'),
        ('pytz', 'pytz'),
        ('requests', 'requests')
    ]
    
    all_good = True
    for module_name, display_name in dependencies: 
        try: 
            __import__(module_name)
            print(f"✅ {display_name} imported successfully")
        except ImportError as e: 
            print(f"❌ {display_name} import failed: {e}")
            all_good = False
    
    return all_good

def test_django_setup(): 
    """Test Django setup and configuration"""
    print("\nTesting Django setup...")
    
    # Set environment variables for testing
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
    os.environ.setdefault('SECRET_KEY', 'test_secret_key_for_setup_test')
    os.environ.setdefault('GITHUB_WORKFLOW', 'true')  # Use GitHub workflow DB config
    
    try: 
        import django
        django.setup()
        print("✅ Django setup completed successfully")
        
        # Test that we can access the database configuration
        from django.conf import settings
        db_config = settings.DATABASES['default']
        print(f"✅ Database configuration loaded: {db_config['ENGINE']}")
        
        return True
        
    except Exception as e: 
        print(f"❌ Django setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main(): 
    """Run all tests"""
    print("🧪 TESTING DJANGO SETUP FOR GITHUB ACTIONS")
    print(" = " * 60)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Django imports
    if test_django_imports(): 
        tests_passed += 1
    
    # Test 2: Other dependencies
    if test_other_dependencies(): 
        tests_passed += 1
    
    # Test 3: Django setup
    if test_django_setup(): 
        tests_passed += 1
    
    print("\n" + " = " * 60)
    print(f"RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed ==  total_tests: 
        print("✅ ALL TESTS PASSED - Django migrations should work!")
        return 0
    else: 
        print("❌ SOME TESTS FAILED - Fix dependencies before running migrations")
        return 1

if __name__ ==  "__main__": sys.exit(main())