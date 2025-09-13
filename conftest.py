"""
Pytest configuration for Django tests
"""

import os
import django
from django.conf import settings
from django.test.utils import get_runner

def pytest_configure():
    """Configure Django for pytest"""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.test_settings')
    django.setup()

def pytest_unconfigure():
    """Clean up after tests"""
    pass
