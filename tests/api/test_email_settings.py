"""
Tests for email notification settings API endpoints.
"""

import json
import os
import pytest
from unittest.mock import patch, MagicMock

# Setup Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.test_settings')
import django
django.setup()

from django.test import TestCase, Client, RequestFactory
from django.contrib.auth.models import User


class TestEmailSettingsAPI(TestCase):
    """Test cases for email settings API endpoints."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = Client()
        self.factory = RequestFactory()
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        self.client.login(username='testuser', password='testpass123')

    def test_test_email_missing_fields(self):
        """Test that test_email returns error for missing required fields."""
        response = self.client.post(
            '/api/settings/email/test',
            data=json.dumps({
                'smtp_host': '',
                'email_from': '',
                'email_to': '',
            }),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data['status'], 'error')
        self.assertIn('required', data['message'].lower())

    @patch('smtplib.SMTP')
    def test_test_email_success(self, mock_smtp):
        """Test that test_email sends email successfully."""
        # Configure mock
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        response = self.client.post(
            '/api/settings/email/test',
            data=json.dumps({
                'smtp_host': 'smtp.test.com',
                'smtp_port': 587,
                'email_from': 'from@test.com',
                'email_to': 'to@test.com',
                'smtp_user': 'user',
                'smtp_pass': 'pass',
            }),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('sent successfully', data['message'])

    @patch('smtplib.SMTP')
    def test_test_email_auth_failure(self, mock_smtp):
        """Test that test_email handles authentication failure."""
        import smtplib
        mock_server = MagicMock()
        mock_server.login.side_effect = smtplib.SMTPAuthenticationError(535, 'Authentication failed')
        mock_smtp.return_value.__enter__.return_value = mock_server

        response = self.client.post(
            '/api/settings/email/test',
            data=json.dumps({
                'smtp_host': 'smtp.test.com',
                'smtp_port': 587,
                'email_from': 'from@test.com',
                'email_to': 'to@test.com',
                'smtp_user': 'user',
                'smtp_pass': 'wrongpass',
            }),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 401)
        data = response.json()
        self.assertEqual(data['status'], 'error')
        self.assertIn('authentication', data['message'].lower())

    def test_save_settings_email_config(self):
        """Test that save_settings saves email configuration."""
        response = self.client.post(
            '/api/settings/save',
            data=json.dumps({
                'email_enabled': True,
                'smtp_host': 'smtp.test.com',
                'smtp_port': 587,
                'email_from': 'from@test.com',
                'email_to': 'to@test.com',
                'smtp_user': 'user',
                'smtp_pass': 'pass',
                'email_alerts': {
                    'stop_loss': True,
                    'risk_alert': True,
                    'entry_signal': False,
                },
            }),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'success')

        # Verify environment variables were set
        self.assertEqual(os.environ.get('ALERT_EMAIL_SMTP_HOST'), 'smtp.test.com')
        self.assertEqual(os.environ.get('ALERT_EMAIL_SMTP_PORT'), '587')
        self.assertEqual(os.environ.get('ALERT_EMAIL_FROM'), 'from@test.com')
        self.assertEqual(os.environ.get('ALERT_EMAIL_TO'), 'to@test.com')

    def test_save_settings_all_options(self):
        """Test that save_settings saves all configuration options."""
        response = self.client.post(
            '/api/settings/save',
            data=json.dumps({
                'alpaca_api_key': '',  # Empty to skip credential save
                'alpaca_secret_key': '',
                'trading_mode': 'paper',
                'email_enabled': True,
                'smtp_host': 'smtp.gmail.com',
                'smtp_port': 587,
                'email_from': 'alerts@test.com',
                'email_to': 'user@test.com',
                'discord_webhook': 'https://discord.com/webhook/test',
                'slack_webhook': 'https://hooks.slack.com/test',
                'shadow_mode': True,
                'log_level': 'DEBUG',
                'timezone': 'America/Los_Angeles',
            }),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'success')

        # Verify settings were saved
        self.assertEqual(os.environ.get('ALERT_DISCORD_WEBHOOK'), 'https://discord.com/webhook/test')
        self.assertEqual(os.environ.get('ALERT_SLACK_WEBHOOK'), 'https://hooks.slack.com/test')
        self.assertEqual(os.environ.get('TRADING_SHADOW_MODE'), 'true')
        self.assertEqual(os.environ.get('LOG_LEVEL'), 'DEBUG')
        self.assertEqual(os.environ.get('TRADING_TIMEZONE'), 'America/Los_Angeles')

    def test_unauthenticated_access_denied(self):
        """Test that unauthenticated users cannot access email settings."""
        self.client.logout()

        response = self.client.post(
            '/api/settings/email/test',
            data=json.dumps({
                'smtp_host': 'smtp.test.com',
            }),
            content_type='application/json',
        )

        # Should redirect to login
        self.assertEqual(response.status_code, 302)

    def test_invalid_json_handling(self):
        """Test that invalid JSON is handled properly."""
        response = self.client.post(
            '/api/settings/save',
            data='not valid json {{{',
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data['status'], 'error')
        self.assertIn('JSON', data['message'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
