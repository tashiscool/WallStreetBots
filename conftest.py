"""
Pytest configuration for Django 4.2+ compatibility
"""
import os
import pytest
import django
from django.core import mail
from django.conf import settings

# Setup Django before importing models
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
if not settings.configured:
    django.setup()


@pytest.fixture(autouse=True)
def setup_django_mail():
    """Setup Django mail outbox for tests"""
    # Clear the outbox before each test
    mail.outbox = []
    yield
    # Clear the outbox after each test
    mail.outbox = []
