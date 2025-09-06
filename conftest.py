"""
Pytest configuration for Django 4.2+ compatibility
"""
import pytest
from django.core import mail


@pytest.fixture(autouse=True)
def setup_django_mail():
    """Setup Django mail outbox for tests"""
    # Clear the outbox before each test
    mail.outbox = []
    yield
    # Clear the outbox after each test
    mail.outbox = []
