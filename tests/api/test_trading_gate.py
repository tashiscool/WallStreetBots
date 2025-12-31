"""
Tests for the Trading Gate (paper-to-live trading enforcement) system.
"""

import json
import os
from datetime import timedelta
from unittest.mock import patch, MagicMock

# Setup Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.test_settings')
import django
django.setup()

from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.utils import timezone


class TestTradingGateModel(TestCase):
    """Test TradingGate model functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )

    def test_create_trading_gate(self):
        """Test creating a TradingGate for a user."""
        from backend.auth0login.models import TradingGate

        gate = TradingGate.objects.create(
            user=self.user,
            paper_trading_days_required=14,
        )

        self.assertEqual(gate.user, self.user)
        self.assertEqual(gate.paper_trading_days_required, 14)
        self.assertFalse(gate.live_trading_approved)
        self.assertIsNone(gate.paper_trading_started_at)

    def test_days_in_paper_trading(self):
        """Test calculating days in paper trading."""
        from backend.auth0login.models import TradingGate

        gate = TradingGate.objects.create(
            user=self.user,
            paper_trading_started_at=timezone.now() - timedelta(days=10),
        )

        self.assertEqual(gate.days_in_paper_trading, 10)
        self.assertEqual(gate.days_remaining, 4)  # 14 - 10

    def test_paper_trading_duration_met(self):
        """Test checking if paper trading duration is met."""
        from backend.auth0login.models import TradingGate

        # Not met
        gate1 = TradingGate.objects.create(
            user=self.user,
            paper_trading_started_at=timezone.now() - timedelta(days=5),
        )
        self.assertFalse(gate1.paper_trading_duration_met)

        # Delete and create new
        gate1.delete()

        # Met
        gate2 = TradingGate.objects.create(
            user=self.user,
            paper_trading_started_at=timezone.now() - timedelta(days=20),
        )
        self.assertTrue(gate2.paper_trading_duration_met)

    def test_approve_live_trading(self):
        """Test approving live trading."""
        from backend.auth0login.models import TradingGate

        gate = TradingGate.objects.create(
            user=self.user,
            paper_trading_started_at=timezone.now() - timedelta(days=20),
        )

        gate.approve_live_trading(method='auto')

        self.assertTrue(gate.live_trading_approved)
        self.assertIsNotNone(gate.live_trading_approved_at)
        self.assertEqual(gate.approval_method, 'auto')


class TestTradingGateService(TestCase):
    """Test TradingGateService functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )

    def test_get_or_create_gate(self):
        """Test getting or creating a gate for a user."""
        from backend.auth0login.services.trading_gate import trading_gate_service

        gate = trading_gate_service.get_or_create_gate(self.user)

        self.assertIsNotNone(gate)
        self.assertEqual(gate.user, self.user)

        # Getting again should return the same gate
        gate2 = trading_gate_service.get_or_create_gate(self.user)
        self.assertEqual(gate.id, gate2.id)

    def test_start_paper_trading(self):
        """Test starting paper trading."""
        from backend.auth0login.services.trading_gate import trading_gate_service

        gate = trading_gate_service.start_paper_trading(self.user)

        self.assertIsNotNone(gate.paper_trading_started_at)

    def test_get_requirements(self):
        """Test getting requirements list."""
        from backend.auth0login.services.trading_gate import trading_gate_service

        trading_gate_service.start_paper_trading(self.user)
        requirements = trading_gate_service.get_requirements(self.user)

        self.assertEqual(len(requirements), 3)
        requirement_names = [r.name for r in requirements]
        self.assertIn('paper_trading_duration', requirement_names)
        self.assertIn('minimum_trades', requirement_names)
        self.assertIn('no_catastrophic_loss', requirement_names)

    def test_is_live_trading_allowed_not_approved(self):
        """Test that live trading is not allowed without approval."""
        from backend.auth0login.services.trading_gate import trading_gate_service

        trading_gate_service.start_paper_trading(self.user)
        is_allowed, reason = trading_gate_service.is_live_trading_allowed(self.user)

        self.assertFalse(is_allowed)
        self.assertIn('not yet approved', reason.lower())

    def test_is_live_trading_allowed_approved(self):
        """Test that live trading is allowed after approval."""
        from backend.auth0login.services.trading_gate import trading_gate_service
        from backend.auth0login.models import TradingGate

        gate = trading_gate_service.start_paper_trading(self.user)
        gate.approve_live_trading(method='override')

        is_allowed, reason = trading_gate_service.is_live_trading_allowed(self.user)

        self.assertTrue(is_allowed)

    def test_approve_with_force(self):
        """Test forcing approval (admin override)."""
        from backend.auth0login.services.trading_gate import trading_gate_service

        trading_gate_service.start_paper_trading(self.user)
        result = trading_gate_service.approve_live_trading(self.user, force=True)

        self.assertEqual(result['status'], 'approved')
        self.assertTrue(result['approved'])
        self.assertEqual(result['method'], 'override')


class TestTradingGateAPI(TestCase):
    """Test Trading Gate API endpoints."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = Client()
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        self.client.login(username='testuser', password='testpass123')

    def test_get_status(self):
        """Test getting trading gate status."""
        response = self.client.get('/api/trading-gate/status')

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('gate', data)
        self.assertIn('is_paper_trading', data['gate'])
        self.assertIn('requirements', data['gate'])

    def test_get_requirements(self):
        """Test getting requirements."""
        response = self.client.get('/api/trading-gate/requirements')

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('requirements', data)
        self.assertEqual(len(data['requirements']), 3)

    def test_start_paper_trading(self):
        """Test starting paper trading via API."""
        response = self.client.post('/api/trading-gate/start-paper')

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('paper_trading_started_at', data)

    def test_request_live_trading_not_ready(self):
        """Test requesting live trading when requirements not met."""
        # Start paper trading
        self.client.post('/api/trading-gate/start-paper')

        # Request live (should fail - requirements not met)
        response = self.client.post('/api/trading-gate/request-live')

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'success')
        # The result should show not approved
        self.assertFalse(data['result'].get('approved', True))

    def test_unauthenticated_access_denied(self):
        """Test that unauthenticated access is denied."""
        self.client.logout()

        response = self.client.get('/api/trading-gate/status')

        # Should redirect to login
        self.assertEqual(response.status_code, 302)


class TestLiveTradingDecorator(TestCase):
    """Test the @require_live_trading_approved decorator."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = Client()
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )

    def test_decorator_blocks_unapproved_user(self):
        """Test that decorator blocks users without live trading approval."""
        from django.http import HttpRequest, JsonResponse
        from backend.auth0login.decorators import require_live_trading_approved
        from backend.auth0login.services.trading_gate import trading_gate_service

        # Start paper trading for user
        trading_gate_service.start_paper_trading(self.user)

        # Create a mock view
        @require_live_trading_approved
        def mock_order_view(request):
            return JsonResponse({'status': 'order_placed'})

        # Create request with authenticated user (real User is always authenticated)
        request = MagicMock()
        request.user = self.user

        # Call the decorated view
        response = mock_order_view(request)

        self.assertEqual(response.status_code, 403)
        data = json.loads(response.content)
        self.assertEqual(data['error_code'], 'LIVE_TRADING_NOT_APPROVED')

    def test_decorator_allows_approved_user(self):
        """Test that decorator allows users with live trading approval."""
        from django.http import JsonResponse
        from backend.auth0login.decorators import require_live_trading_approved
        from backend.auth0login.services.trading_gate import trading_gate_service

        # Start paper trading and approve
        gate = trading_gate_service.start_paper_trading(self.user)
        gate.approve_live_trading(method='override')

        # Create a mock view
        @require_live_trading_approved
        def mock_order_view(request):
            return JsonResponse({'status': 'order_placed'})

        # Create request with authenticated user (real User is always authenticated)
        request = MagicMock()
        request.user = self.user

        # Call the decorated view
        response = mock_order_view(request)

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data['status'], 'order_placed')


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
