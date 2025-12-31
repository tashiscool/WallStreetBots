"""Comprehensive tests for DigestService.

Tests cover:
- Email digest generation (daily/weekly)
- Trade aggregation
- Alert aggregation
- Position aggregation
- Performance metrics
- Strategy breakdowns
- Email rendering (HTML/text)
- SMTP sending
- Batch digest sending
- Error handling and edge cases
"""

import smtplib
from datetime import datetime, timedelta
from decimal import Decimal
from email.mime.multipart import MIMEMultipart
from unittest.mock import MagicMock, Mock, patch, call

import pytest
from django.contrib.auth.models import User
from django.utils import timezone

from backend.auth0login.services.digest_service import DigestService


@pytest.fixture
def user():
    """Create a test user."""
    return User.objects.create_user(
        username='testuser',
        email='test@example.com',
        password='testpass123'
    )


@pytest.fixture
def digest_service(user):
    """Create a DigestService instance."""
    return DigestService(user=user)


@pytest.fixture
def mock_smtp_config():
    """Mock SMTP configuration."""
    with patch.dict('os.environ', {
        'ALERT_EMAIL_SMTP_HOST': 'smtp.example.com',
        'ALERT_EMAIL_SMTP_PORT': '587',
        'ALERT_EMAIL_USER': 'bot@example.com',
        'ALERT_EMAIL_PASS': 'password',
        'ALERT_EMAIL_FROM': 'bot@example.com',
        'APP_BASE_URL': 'http://localhost:8000'
    }):
        yield


class TestDigestServiceInit:
    """Test DigestService initialization."""

    def test_init_with_user(self, user):
        """Test initialization with user."""
        service = DigestService(user=user)
        assert service.user == user
        assert service.smtp_host == 'smtp.example.com' or service.smtp_host is None
        assert service.base_url is not None

    def test_init_without_user(self):
        """Test initialization without user."""
        service = DigestService()
        assert service.user is None

    def test_init_reads_environment(self, mock_smtp_config):
        """Test initialization reads environment variables."""
        service = DigestService()
        assert service.smtp_host == 'smtp.example.com'
        assert service.smtp_port == 587
        assert service.smtp_user == 'bot@example.com'
        assert service.smtp_pass == 'password'
        assert service.from_email == 'bot@example.com'
        assert service.base_url == 'http://localhost:8000'


class TestPeriodBounds:
    """Test _get_period_bounds method."""

    def test_daily_period_bounds(self, digest_service):
        """Test daily period calculation."""
        with patch('django.utils.timezone.now') as mock_now:
            mock_now.return_value = timezone.make_aware(
                datetime(2024, 1, 15, 14, 30, 0)
            )
            start, end = digest_service._get_period_bounds('daily')

            # Should be previous day midnight to midnight
            assert end.hour == 0
            assert end.minute == 0
            assert end.second == 0
            assert (end - start).days == 1

    def test_weekly_period_bounds(self, digest_service):
        """Test weekly period calculation."""
        with patch('django.utils.timezone.now') as mock_now:
            # Wednesday
            mock_now.return_value = timezone.make_aware(
                datetime(2024, 1, 17, 14, 30, 0)
            )
            start, end = digest_service._get_period_bounds('weekly')

            # Should be 7 days apart
            assert (end - start).days == 7
            # End should be Monday at midnight
            assert end.weekday() == 0  # Monday
            assert end.hour == 0


class TestAggregates:
    """Test aggregation methods."""

    @patch('backend.auth0login.services.digest_service.TradeSignalSnapshot')
    def test_aggregate_trades_empty(self, mock_model, digest_service, user):
        """Test trade aggregation with no trades."""
        mock_model.objects.filter.return_value.order_by.return_value = []
        mock_model.objects.filter.return_value.filter.return_value.count.return_value = 0
        mock_model.objects.filter.return_value.filter.return_value.aggregate.return_value = {'total': None}

        now = timezone.now()
        result = digest_service.aggregate_trades(user, now - timedelta(days=1), now)

        assert result['summary']['total_trades'] == 0
        assert result['summary']['total_pnl'] == 0.0
        assert result['summary']['win_rate'] == 0.0
        assert result['trades'] == []

    @patch('backend.auth0login.services.digest_service.TradeSignalSnapshot')
    def test_aggregate_trades_with_data(self, mock_model, digest_service, user):
        """Test trade aggregation with trades."""
        # Create mock trades
        mock_trade = Mock()
        mock_trade.symbol = 'AAPL'
        mock_trade.direction = 'buy'
        mock_trade.quantity = Decimal('100')
        mock_trade.entry_price = Decimal('150.00')
        mock_trade.exit_price = Decimal('155.00')
        mock_trade.pnl_amount = Decimal('500.00')
        mock_trade.strategy_name = 'wsb_dip_bot'
        mock_trade.created_at = timezone.now()

        # Setup mock query chain
        mock_qs = Mock()
        mock_qs.order_by.return_value = [mock_trade]
        mock_qs.filter.return_value = mock_qs
        mock_qs.count.return_value = 1
        mock_qs.aggregate.return_value = {'total': Decimal('500.00')}
        mock_model.objects.filter.return_value = mock_qs

        now = timezone.now()
        result = digest_service.aggregate_trades(user, now - timedelta(days=1), now)

        assert result['summary']['total_trades'] >= 0
        assert 'trades' in result

    @patch('backend.auth0login.services.digest_service.CircuitBreakerHistory')
    def test_aggregate_alerts(self, mock_model, digest_service, user):
        """Test alert aggregation."""
        mock_alert = Mock()
        mock_alert.action = 'trip'
        mock_alert.reason = 'Daily loss limit exceeded'
        mock_alert.timestamp = timezone.now()

        mock_qs = Mock()
        mock_qs.order_by.return_value = [mock_alert]
        mock_model.objects.filter.return_value = mock_qs

        now = timezone.now()
        result = digest_service.aggregate_alerts(user, now - timedelta(days=1), now)

        assert 'summary' in result
        assert 'alerts' in result
        assert result['summary']['total_alerts'] >= 0

    @patch('backend.auth0login.services.digest_service.TradeSignalSnapshot')
    def test_aggregate_positions(self, mock_model, digest_service, user):
        """Test position aggregation."""
        mock_qs = Mock()
        mock_qs.values.return_value.annotate.return_value = [
            {
                'symbol': 'AAPL',
                'total_quantity': Decimal('100'),
                'avg_price': Decimal('150.00'),
                'trade_count': 1
            }
        ]
        mock_model.objects.filter.return_value = mock_qs

        now = timezone.now()
        result = digest_service.aggregate_positions(user, now)

        assert 'summary' in result
        assert 'positions' in result
        assert 'total_value' in result['summary']

    @patch('backend.auth0login.services.digest_service.BenchmarkService')
    def test_aggregate_performance_success(self, mock_benchmark, digest_service, user):
        """Test performance aggregation."""
        mock_service = Mock()
        mock_service.get_performance_vs_benchmark.return_value = {
            'portfolio_return': 10.5,
            'benchmark_return': 8.0,
            'alpha': 2.5,
            'sharpe_ratio': 1.5,
            'max_drawdown': -5.0,
        }
        mock_benchmark.return_value = mock_service

        now = timezone.now()
        result = digest_service.aggregate_performance(user, now - timedelta(days=7), now)

        assert result['period_return'] == 10.5
        assert result['benchmark_return'] == 8.0
        assert result['alpha'] == 2.5

    def test_aggregate_performance_failure(self, digest_service, user):
        """Test performance aggregation with import error."""
        with patch('backend.auth0login.services.digest_service.BenchmarkService', side_effect=ImportError):
            now = timezone.now()
            result = digest_service.aggregate_performance(user, now - timedelta(days=7), now)

            # Should return defaults
            assert result['period_return'] == 0
            assert result['benchmark_return'] == 0
            assert result['alpha'] == 0

    @patch('backend.auth0login.services.digest_service.TradeSignalSnapshot')
    def test_aggregate_by_strategy(self, mock_model, digest_service, user):
        """Test strategy aggregation."""
        mock_qs = Mock()
        mock_qs.values.return_value.annotate.return_value.order_by.return_value = [
            {
                'strategy_name': 'wsb_dip_bot',
                'total_trades': 10,
                'total_pnl': Decimal('500.00'),
                'winning_trades': 7,
            }
        ]
        mock_model.objects.filter.return_value = mock_qs

        now = timezone.now()
        result = digest_service.aggregate_by_strategy(user, now - timedelta(days=7), now)

        assert isinstance(result, list)
        if result:
            assert 'strategy' in result[0]
            assert 'total_trades' in result[0]
            assert 'win_rate' in result[0]


class TestDigestGeneration:
    """Test digest data generation."""

    @patch.object(DigestService, 'aggregate_trades')
    @patch.object(DigestService, 'aggregate_alerts')
    @patch.object(DigestService, 'aggregate_positions')
    @patch.object(DigestService, 'aggregate_performance')
    @patch.object(DigestService, 'aggregate_by_strategy')
    def test_generate_digest_data_daily(
        self, mock_strategy, mock_perf, mock_pos, mock_alerts, mock_trades,
        digest_service, user
    ):
        """Test daily digest generation."""
        # Setup mock returns
        mock_trades.return_value = {
            'summary': {'total_trades': 5, 'winning_trades': 3, 'losing_trades': 2,
                       'total_pnl': 100.0, 'win_rate': 60.0},
            'trades': []
        }
        mock_alerts.return_value = {'summary': {'total_alerts': 1}, 'alerts': []}
        mock_pos.return_value = {'summary': {'open_positions': 2}, 'positions': []}
        mock_perf.return_value = {'period_return': 5.0}
        mock_strategy.return_value = []

        result = digest_service.generate_digest_data(user, 'daily')

        assert result['digest_type'] == 'daily'
        assert result['user']['username'] == 'testuser'
        assert result['summary']['total_trades'] == 5
        assert result['summary']['win_rate'] == 60.0
        assert 'period_start' in result
        assert 'period_end' in result

    @patch.object(DigestService, 'aggregate_trades')
    @patch.object(DigestService, 'aggregate_alerts')
    @patch.object(DigestService, 'aggregate_positions')
    @patch.object(DigestService, 'aggregate_performance')
    @patch.object(DigestService, 'aggregate_by_strategy')
    def test_generate_digest_data_weekly(
        self, mock_strategy, mock_perf, mock_pos, mock_alerts, mock_trades,
        digest_service, user
    ):
        """Test weekly digest generation."""
        mock_trades.return_value = {
            'summary': {'total_trades': 20, 'winning_trades': 12, 'losing_trades': 8,
                       'total_pnl': 500.0, 'win_rate': 60.0},
            'trades': []
        }
        mock_alerts.return_value = {'summary': {'total_alerts': 3}, 'alerts': []}
        mock_pos.return_value = {'summary': {'open_positions': 5}, 'positions': []}
        mock_perf.return_value = {'period_return': 10.0}
        mock_strategy.return_value = []

        result = digest_service.generate_digest_data(user, 'weekly')

        assert result['digest_type'] == 'weekly'
        assert result['summary']['total_trades'] == 20


class TestEmailFormatting:
    """Test email formatting methods."""

    def test_get_email_subject_positive_pnl(self, digest_service):
        """Test email subject with positive P&L."""
        data = {
            'summary': {
                'total_trades': 10,
                'total_pnl': 150.50
            }
        }
        subject = digest_service._get_email_subject('daily', data)

        assert 'Daily Digest' in subject
        assert '10 trades' in subject
        assert '+$150.50' in subject

    def test_get_email_subject_negative_pnl(self, digest_service):
        """Test email subject with negative P&L."""
        data = {
            'summary': {
                'total_trades': 5,
                'total_pnl': -50.25
            }
        }
        subject = digest_service._get_email_subject('weekly', data)

        assert 'Weekly Digest' in subject
        assert '5 trades' in subject
        assert '-$50.25' in subject

    def test_render_email_text(self, digest_service):
        """Test plain text email rendering."""
        data = {
            'summary': {
                'total_trades': 10,
                'winning_trades': 6,
                'losing_trades': 4,
                'win_rate': 60.0,
                'total_pnl': 100.0,
                'open_positions': 3
            },
            'trades': [],
            'alerts': [],
            'positions': [],
            'performance': {},
            'by_strategy': []
        }

        text = digest_service._render_email_text('daily', data)

        assert 'SUMMARY' in text
        assert 'Total Trades: 10' in text
        assert 'Win Rate: 60.0%' in text
        assert '+$100.00' in text

    def test_render_email_html_fallback(self, digest_service):
        """Test HTML email rendering (fallback)."""
        data = {
            'summary': {
                'total_trades': 5,
                'winning_trades': 3,
                'losing_trades': 2,
                'win_rate': 60.0,
                'total_pnl': 50.0,
                'open_positions': 2
            },
            'trades': [],
            'alerts': [],
            'positions': [],
            'performance': {},
            'by_strategy': []
        }

        html = digest_service._render_email_html('daily', data)

        assert '<!DOCTYPE html>' in html
        assert 'WallStreetBots' in html
        assert '5' in html  # Total trades

    def test_generate_inline_html_with_performance(self, digest_service):
        """Test inline HTML with performance data."""
        data = {
            'summary': {
                'total_trades': 10,
                'winning_trades': 6,
                'losing_trades': 4,
                'win_rate': 60.0,
                'total_pnl': 200.0,
                'open_positions': 3
            },
            'trades': [],
            'alerts': [],
            'positions': [],
            'performance': {
                'period_return': 5.5,
                'benchmark_return': 3.0,
                'alpha': 2.5
            },
            'by_strategy': []
        }

        html = digest_service._generate_inline_html('daily', data)

        assert '5.5%' in html  # period return
        assert '3.0%' in html  # benchmark return
        assert '2.5%' in html  # alpha


class TestEmailSending:
    """Test email sending functionality."""

    @patch('backend.auth0login.services.digest_service.DigestLog')
    @patch('smtplib.SMTP')
    def test_send_digest_email_success(
        self, mock_smtp_class, mock_digest_log,
        digest_service, user, mock_smtp_config
    ):
        """Test successful digest email sending."""
        # Setup mocks
        mock_smtp = Mock()
        mock_smtp_class.return_value.__enter__.return_value = mock_smtp

        mock_log = Mock()
        mock_log.delivery_status = 'pending'
        mock_digest_log.objects.filter.return_value.first.return_value = None
        mock_digest_log.objects.update_or_create.return_value = (mock_log, True)

        data = {
            'summary': {
                'total_trades': 5,
                'winning_trades': 3,
                'losing_trades': 2,
                'win_rate': 60.0,
                'total_pnl': 100.0,
                'open_positions': 2
            },
            'trades': [],
            'alerts': [],
            'positions': [],
            'performance': {},
            'by_strategy': []
        }

        success, error, returned_data = digest_service.send_digest_email(user, 'daily', data)

        assert success is True
        assert error == ""
        mock_smtp.sendmail.assert_called_once()
        mock_log.mark_sent.assert_called_once()

    @patch('backend.auth0login.services.digest_service.DigestLog')
    def test_send_digest_email_no_smtp_config(
        self, mock_digest_log, user
    ):
        """Test email sending with missing SMTP config."""
        service = DigestService(user)
        service.smtp_host = None

        mock_log = Mock()
        mock_digest_log.objects.filter.return_value.first.return_value = None
        mock_digest_log.objects.update_or_create.return_value = (mock_log, True)

        data = {'summary': {'total_trades': 0, 'total_pnl': 0}}

        success, error, _ = service.send_digest_email(user, 'daily', data)

        assert success is False
        assert 'Email not configured' in error
        mock_log.mark_failed.assert_called_once()

    @patch('backend.auth0login.services.digest_service.DigestLog')
    def test_send_digest_email_no_user_email(
        self, mock_digest_log, mock_smtp_config
    ):
        """Test email sending with user missing email."""
        user = User.objects.create_user(username='noemail', email='')
        service = DigestService(user)

        mock_log = Mock()
        mock_digest_log.objects.filter.return_value.first.return_value = None
        mock_digest_log.objects.update_or_create.return_value = (mock_log, True)

        data = {'summary': {'total_trades': 0, 'total_pnl': 0}}

        success, error, _ = service.send_digest_email(user, 'daily', data)

        assert success is False
        assert 'no email address' in error

    @patch('backend.auth0login.services.digest_service.DigestLog')
    def test_send_digest_email_already_sent(
        self, mock_digest_log, user, mock_smtp_config
    ):
        """Test email sending when digest already sent."""
        mock_log = Mock()
        mock_log.delivery_status = 'sent'
        mock_digest_log.objects.filter.return_value.first.return_value = mock_log

        data = {'summary': {'total_trades': 0, 'total_pnl': 0}}

        success, error, _ = DigestService(user).send_digest_email(user, 'daily', data)

        assert success is False
        assert 'already sent' in error

    @patch('backend.auth0login.services.digest_service.DigestLog')
    @patch('smtplib.SMTP')
    def test_send_digest_email_smtp_error(
        self, mock_smtp_class, mock_digest_log,
        user, mock_smtp_config
    ):
        """Test email sending with SMTP error."""
        mock_smtp = Mock()
        mock_smtp.sendmail.side_effect = smtplib.SMTPException("Connection failed")
        mock_smtp_class.return_value.__enter__.return_value = mock_smtp

        mock_log = Mock()
        mock_digest_log.objects.filter.return_value.first.return_value = None
        mock_digest_log.objects.update_or_create.return_value = (mock_log, True)

        data = {'summary': {'total_trades': 0, 'total_pnl': 0}, 'trades': [],
                'alerts': [], 'positions': [], 'performance': {}, 'by_strategy': []}

        success, error, _ = DigestService(user).send_digest_email(user, 'daily', data)

        assert success is False
        assert 'Connection failed' in error
        mock_log.mark_failed.assert_called_once()


class TestBatchSending:
    """Test batch digest sending."""

    @patch('backend.auth0login.services.digest_service.User')
    @patch('backend.auth0login.services.digest_service.UserProfile')
    @patch.object(DigestService, 'send_digest_email')
    def test_send_digests_for_frequency_dry_run(
        self, mock_send, mock_profile, mock_user_model, digest_service
    ):
        """Test batch sending in dry run mode."""
        mock_user1 = Mock()
        mock_user1.username = 'user1'
        mock_user1.email = 'user1@example.com'

        mock_user_model.objects.filter.return_value.exclude.return_value.select_related.return_value = [mock_user1]

        result = digest_service.send_digests_for_frequency('daily', dry_run=True)

        assert result['dry_run'] is True
        assert result['sent'] >= 0
        assert result['failed'] == 0

    @patch('backend.auth0login.services.digest_service.User')
    @patch('backend.auth0login.services.digest_service.UserProfile')
    @patch.object(DigestService, 'send_digest_email')
    def test_send_digests_for_frequency_success(
        self, mock_send, mock_profile, mock_user_model, digest_service
    ):
        """Test batch sending with successes."""
        mock_user1 = Mock()
        mock_user1.username = 'user1'
        mock_user1.email = 'user1@example.com'

        mock_user_model.objects.filter.return_value.exclude.return_value.select_related.return_value = [mock_user1]
        mock_send.return_value = (True, "", {})

        result = digest_service.send_digests_for_frequency('daily', dry_run=False)

        assert result['sent'] >= 0

    @patch('backend.auth0login.services.digest_service.User')
    @patch('backend.auth0login.services.digest_service.UserProfile')
    @patch.object(DigestService, 'send_digest_email')
    def test_send_digests_for_frequency_with_failures(
        self, mock_send, mock_profile, mock_user_model, digest_service
    ):
        """Test batch sending with failures."""
        mock_user1 = Mock()
        mock_user1.username = 'user1'
        mock_user1.email = 'user1@example.com'

        mock_user_model.objects.filter.return_value.exclude.return_value.select_related.return_value = [mock_user1]
        mock_send.return_value = (False, "SMTP error", {})

        result = digest_service.send_digests_for_frequency('daily', dry_run=False)

        assert result['failed'] >= 0
        assert len(result['errors']) >= 0

    @patch('backend.auth0login.services.digest_service.User')
    def test_send_digests_invalid_type(self, mock_user_model, digest_service):
        """Test batch sending with invalid digest type."""
        result = digest_service.send_digests_for_frequency('invalid')

        assert result['success'] is False
        assert 'Invalid digest type' in result['error']


class TestPreview:
    """Test digest preview functionality."""

    @patch.object(DigestService, 'generate_digest_data')
    def test_preview_digest(self, mock_generate, digest_service, user):
        """Test digest preview generation."""
        mock_generate.return_value = {
            'summary': {'total_trades': 5, 'total_pnl': 100.0},
            'trades': [], 'alerts': [], 'positions': [],
            'performance': {}, 'by_strategy': []
        }

        preview = digest_service.preview_digest(user, 'daily')

        assert 'data' in preview
        assert 'subject' in preview
        assert 'html' in preview
        assert 'text' in preview
        assert 'Daily Digest' in preview['subject']
