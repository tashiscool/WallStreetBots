"""
Comprehensive tests for backend/auth0login/views.py
Target: 80%+ coverage with all edge cases and error handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from django.test import RequestFactory
from django.contrib.auth.models import AnonymousUser, User
from django.http import HttpRequest


@pytest.fixture
def request_factory():
    """Django request factory."""
    return RequestFactory()


@pytest.fixture
def mock_authenticated_user():
    """Create authenticated user."""
    user = Mock(spec=User)
    user.is_authenticated = True
    user.first_name = "Test"
    user.last_name = "User"
    user.id = 1

    # Mock credential
    credential = Mock()
    credential.alpaca_id = "test_api_key"
    credential.alpaca_key = "test_secret_key"
    user.credential = credential

    # Mock portfolio
    portfolio = Mock()
    portfolio.cash = 10000.0
    portfolio.strategy = "momentum"
    portfolio.get_strategy_display = Mock(return_value="Momentum")
    user.portfolio = portfolio

    # Mock social_auth
    social_auth = Mock()
    social_auth.get = Mock(return_value=Mock())
    user.social_auth = social_auth

    return user


@pytest.fixture
def mock_anonymous_user():
    """Create anonymous user."""
    user = AnonymousUser()
    return user


class TestLogin:
    """Test login view."""

    @patch('backend.auth0login.views.render')
    @patch('backend.auth0login.views.redirect')
    def test_login_authenticated_user(self, mock_redirect, mock_render, request_factory, mock_authenticated_user):
        """Test login redirects authenticated users to dashboard."""
        from backend.auth0login.views import login

        request = request_factory.get('/login')
        request.user = mock_authenticated_user

        login(request)

        mock_redirect.assert_called_once()
        mock_render.assert_not_called()

    @patch('backend.auth0login.views.render')
    def test_login_anonymous_user(self, mock_render, request_factory, mock_anonymous_user):
        """Test login renders template for anonymous users."""
        from backend.auth0login.views import login

        request = request_factory.get('/login')
        request.user = mock_anonymous_user

        login(request)

        mock_render.assert_called_once()


class TestGetUserInformation:
    """Test get_user_information view."""

    @patch('backend.auth0login.views.sync_alpaca')
    def test_get_user_information_success(
        self,
        mock_sync,
        request_factory,
        mock_authenticated_user,
    ):
        """Test successful user information retrieval."""
        from backend.auth0login.views import get_user_information

        # Mock sync_alpaca return value
        user_details = {
            "equity": "100000.00",
            "buy_power": "50000.00",
            "cash": "25000.00",
            "usable_cash": "23000.00",
            "currency": "USD",
            "long_portfolio_value": "75000.00",
            "short_portfolio_value": "0.00",
            "display_portfolio": [],
            "orders": [],
            "strategy": "Momentum",
            "portfolio_percent_change": "5.0",
            "portfolio_dollar_change": "5000.00",
            "portfolio_change_direction": "positive",
        }
        mock_sync.return_value = user_details

        request = request_factory.get('/')
        request.user = mock_authenticated_user

        user, userdata, auth0user, details = get_user_information(request)

        assert user == mock_authenticated_user
        assert userdata["name"] == "Test"
        assert userdata["total_equity"] == "100000.00"
        assert "alpaca_key" in userdata

    @patch('backend.auth0login.views.sync_alpaca')
    def test_get_user_information_no_credentials(
        self,
        mock_sync,
        request_factory,
        mock_authenticated_user,
    ):
        """Test user information when sync returns None."""
        mock_sync.return_value = None
        delattr(mock_authenticated_user, 'credential')

        request = request_factory.get('/')
        request.user = mock_authenticated_user

        user, userdata, auth0user, details = get_user_information(request)

        assert "error" in userdata
        assert userdata["alpaca_id"] == "no alpaca id"


class TestDashboard:
    """Test dashboard view."""

    @patch('backend.auth0login.views.get_user_information')
    @patch('backend.auth0login.views.get_portfolio_chart')
    @patch('backend.auth0login.views.render')
    def test_dashboard_get(
        self,
        mock_render,
        mock_chart,
        mock_get_info,
        request_factory,
        mock_authenticated_user,
    ):
        """Test dashboard GET request."""
        from backend.auth0login.views import dashboard

        mock_get_info.return_value = (
            mock_authenticated_user,
            {"name": "Test"},
            Mock(),
            {"equity": "100000"},
        )
        mock_chart.return_value = "<div>Chart</div>"

        request = request_factory.get('/dashboard')
        request.user = mock_authenticated_user
        request.method = 'GET'

        dashboard(request)

        mock_render.assert_called_once()

    @patch('backend.auth0login.views.get_user_information')
    @patch('backend.auth0login.views.CredentialForm')
    @patch('backend.auth0login.views.HttpResponseRedirect')
    def test_dashboard_post_credential(
        self,
        mock_redirect,
        mock_form_class,
        mock_get_info,
        request_factory,
        mock_authenticated_user,
    ):
        """Test dashboard POST credential update."""
        from backend.auth0login.views import dashboard

        mock_get_info.return_value = (
            mock_authenticated_user,
            {"name": "Test"},
            Mock(),
            {"equity": "100000"},
        )

        # Mock form
        mock_form = Mock()
        mock_form.is_valid.return_value = True
        mock_form.get_id.return_value = "new_id"
        mock_form.get_key.return_value = "new_key"
        mock_form_class.return_value = mock_form

        request = request_factory.post('/dashboard', {"submit_credential": "1"})
        request.user = mock_authenticated_user
        request.method = 'POST'

        dashboard(request)

        mock_redirect.assert_called_once()

    @patch('backend.auth0login.views.get_user_information')
    @patch('backend.auth0login.views.OrderForm')
    @patch('backend.auth0login.views.render')
    def test_dashboard_post_order(
        self,
        mock_render,
        mock_form_class,
        mock_get_info,
        request_factory,
        mock_authenticated_user,
    ):
        """Test dashboard POST order submission."""
        from backend.auth0login.views import dashboard

        user_details = {"equity": "100000", "orders": []}
        mock_get_info.return_value = (
            mock_authenticated_user,
            {"name": "Test", "orders": []},
            Mock(),
            user_details,
        )

        # Mock form
        mock_form = Mock()
        mock_form.is_valid.return_value = True
        mock_form.place_order.return_value = "Order placed successfully"
        mock_form_class.return_value = mock_form

        request = request_factory.post('/dashboard', {"submit_order": "1"})
        request.user = mock_authenticated_user
        request.method = 'POST'

        with patch('backend.auth0login.views.Order') as mock_order_class:
            mock_order_class.objects.filter.return_value.order_by.return_value.iterator.return_value = []

            dashboard(request)

            mock_render.assert_called_once()

    @patch('backend.auth0login.views.get_user_information')
    @patch('backend.auth0login.views.StrategyForm')
    @patch('backend.auth0login.views.HttpResponseRedirect')
    def test_dashboard_post_strategy(
        self,
        mock_redirect,
        mock_form_class,
        mock_get_info,
        request_factory,
        mock_authenticated_user,
    ):
        """Test dashboard POST strategy update."""
        from backend.auth0login.views import dashboard

        mock_get_info.return_value = (
            mock_authenticated_user,
            {"name": "Test"},
            Mock(),
            {"equity": "100000"},
        )

        # Mock form
        mock_form = Mock()
        mock_form.is_valid.return_value = True
        mock_form.cleaned_data = "momentum"
        mock_form_class.return_value = mock_form

        request = request_factory.post('/dashboard', {"submit_strategy": "1"})
        request.user = mock_authenticated_user
        request.method = 'POST'

        dashboard(request)

        mock_redirect.assert_called_once()


class TestGetPortfolioChart:
    """Test get_portfolio_chart view."""

    @patch('backend.auth0login.views.get_user_information')
    @patch('backend.auth0login.views.api')
    def test_get_portfolio_chart_success(
        self,
        mock_api,
        mock_get_info,
        request_factory,
        mock_authenticated_user,
    ):
        """Test successful portfolio chart generation."""
        from backend.auth0login.views import get_portfolio_chart
        import pandas as pd

        mock_get_info.return_value = (
            mock_authenticated_user,
            {},
            Mock(),
            {"equity": "100000"},
        )

        # Mock Alpaca API
        mock_rest = Mock()
        mock_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10),
            'equity': [100000 + i * 100 for i in range(10)],
        })
        mock_portfolio_history = Mock()
        mock_portfolio_history.df = mock_df
        mock_rest.get_portfolio_history.return_value = mock_portfolio_history
        mock_api.REST.return_value = mock_rest

        request = request_factory.get('/chart')
        request.user = mock_authenticated_user

        result = get_portfolio_chart(request)

        assert result is not None
        assert isinstance(result, str)

    @patch('backend.auth0login.views.get_user_information')
    def test_get_portfolio_chart_no_details(
        self,
        mock_get_info,
        request_factory,
        mock_authenticated_user,
    ):
        """Test portfolio chart when user details are None."""
        from backend.auth0login.views import get_portfolio_chart

        mock_get_info.return_value = (
            mock_authenticated_user,
            {},
            Mock(),
            None,
        )

        request = request_factory.get('/chart')
        request.user = mock_authenticated_user

        result = get_portfolio_chart(request)

        assert result is None


class TestOrders:
    """Test orders view."""

    @patch('backend.auth0login.views.get_user_information')
    @patch('backend.auth0login.views.render')
    def test_orders_get(
        self,
        mock_render,
        mock_get_info,
        request_factory,
        mock_authenticated_user,
    ):
        """Test orders GET request."""
        from backend.auth0login.views import orders

        mock_get_info.return_value = (
            mock_authenticated_user,
            {"name": "Test", "orders": []},
            Mock(),
            {"equity": "100000"},
        )

        request = request_factory.get('/orders')
        request.user = mock_authenticated_user
        request.method = 'GET'

        orders(request)

        mock_render.assert_called_once()


class TestPositions:
    """Test positions view."""

    @patch('backend.auth0login.views.get_user_information')
    @patch('backend.auth0login.views.render')
    def test_positions_get(
        self,
        mock_render,
        mock_get_info,
        request_factory,
        mock_authenticated_user,
    ):
        """Test positions GET request."""
        from backend.auth0login.views import positions

        mock_get_info.return_value = (
            mock_authenticated_user,
            {"name": "Test"},
            Mock(),
            {"equity": "100000"},
        )

        request = request_factory.get('/positions')
        request.user = mock_authenticated_user
        request.method = 'GET'

        positions(request)

        mock_render.assert_called_once()

    @patch('backend.auth0login.views.get_user_information')
    @patch('backend.auth0login.views.WatchListForm')
    @patch('backend.auth0login.views.render')
    def test_positions_post_watchlist(
        self,
        mock_render,
        mock_form_class,
        mock_get_info,
        request_factory,
        mock_authenticated_user,
    ):
        """Test positions POST watchlist addition."""
        from backend.auth0login.views import positions

        mock_get_info.return_value = (
            mock_authenticated_user,
            {"name": "Test"},
            Mock(),
            {"equity": "100000"},
        )

        # Mock form
        mock_form = Mock()
        mock_form.is_valid.return_value = True
        mock_form.add_to_watchlist.return_value = "Added to watchlist"
        mock_form_class.return_value = mock_form

        request = request_factory.post('/positions', {"add_to_watchlist": "1"})
        request.user = mock_authenticated_user
        request.method = 'POST'

        positions(request)

        mock_render.assert_called_once()


class TestMachineLearning:
    """Test machine_learning view."""

    @patch('backend.auth0login.views.dashboard_service')
    @patch('backend.auth0login.views.render')
    def test_machine_learning(
        self,
        mock_render,
        mock_service,
        request_factory,
        mock_authenticated_user,
    ):
        """Test machine learning page."""
        from backend.auth0login.views import machine_learning

        mock_service.get_ml_dashboard.return_value = {}
        mock_service.get_market_regime.return_value = {}

        request = request_factory.get('/ml')
        request.user = mock_authenticated_user

        machine_learning(request)

        mock_render.assert_called_once()


class TestStrategies:
    """Test strategies view."""

    @patch('backend.auth0login.views.dashboard_service')
    @patch('backend.auth0login.views.render')
    def test_strategies(
        self,
        mock_render,
        mock_service,
        request_factory,
        mock_authenticated_user,
    ):
        """Test strategies page."""
        from backend.auth0login.views import strategies

        mock_strategies = [
            Mock(enabled=True),
            Mock(enabled=True),
            Mock(enabled=False),
        ]
        mock_service.get_all_strategies.return_value = mock_strategies
        mock_service.get_market_regime.return_value = {}

        request = request_factory.get('/strategies')
        request.user = mock_authenticated_user

        strategies(request)

        mock_render.assert_called_once()


class TestStrategyWSBDipBot:
    """Test strategy_wsb_dip_bot view."""

    @patch('backend.auth0login.views.dashboard_service')
    @patch('backend.auth0login.views.render')
    def test_strategy_wsb_dip_bot_get(
        self,
        mock_render,
        mock_service,
        request_factory,
        mock_authenticated_user,
    ):
        """Test WSB Dip Bot strategy GET."""
        from backend.auth0login.views import strategy_wsb_dip_bot

        mock_service.get_strategy_config.return_value = Mock()

        request = request_factory.get('/strategy/wsb')
        request.user = mock_authenticated_user

        strategy_wsb_dip_bot(request)

        mock_render.assert_called_once()

    @patch('backend.auth0login.views.dashboard_service')
    @patch('backend.auth0login.views.render')
    def test_strategy_wsb_dip_bot_post(
        self,
        mock_render,
        mock_service,
        request_factory,
        mock_authenticated_user,
    ):
        """Test WSB Dip Bot strategy POST."""
        from backend.auth0login.views import strategy_wsb_dip_bot

        mock_service.get_strategy_config.return_value = Mock()
        mock_service.save_strategy_config.return_value = {
            "status": "success",
            "message": "Config saved",
        }

        request = request_factory.post('/strategy/wsb', {
            "action": "save_config",
            "run_lookback": "7",
            "run_threshold": "8",
            "dip_threshold": "-2",
            "wsb_weight": "30",
            "target_dte": "21",
            "otm_pct": "3",
            "delta_target": "50",
            "profit_multiplier": "25",
            "position_size": "3",
            "stop_loss": "50",
            "max_positions": "5",
            "watchlist": "AAPL,MSFT",
            "scan_interval": "300",
            "strategy_enabled": "on",
        })
        request.user = mock_authenticated_user
        request.method = 'POST'

        strategy_wsb_dip_bot(request)

        mock_render.assert_called_once()


class TestBacktesting:
    """Test backtesting view."""

    @patch('backend.auth0login.views.dashboard_service')
    @patch('backend.auth0login.views.render')
    def test_backtesting(
        self,
        mock_render,
        mock_service,
        request_factory,
        mock_authenticated_user,
    ):
        """Test backtesting page."""
        from backend.auth0login.views import backtesting

        mock_service.get_backtesting_status.return_value = {
            "is_available": True,
            "strategies": [],
        }

        request = request_factory.get('/backtesting')
        request.user = mock_authenticated_user

        backtesting(request)

        mock_render.assert_called_once()


class TestRiskManagement:
    """Test risk_management view."""

    @patch('backend.auth0login.views.dashboard_service')
    @patch('backend.auth0login.views.render')
    def test_risk_management(
        self,
        mock_render,
        mock_service,
        request_factory,
        mock_authenticated_user,
    ):
        """Test risk management page."""
        from backend.auth0login.views import risk_management

        mock_service.get_risk_metrics.return_value = {}
        mock_service.get_market_regime.return_value = {}

        request = request_factory.get('/risk')
        request.user = mock_authenticated_user

        risk_management(request)

        mock_render.assert_called_once()


class TestAnalytics:
    """Test analytics view."""

    @patch('backend.auth0login.views.dashboard_service')
    @patch('backend.auth0login.views.render')
    def test_analytics(
        self,
        mock_render,
        mock_service,
        request_factory,
        mock_authenticated_user,
    ):
        """Test analytics page."""
        from backend.auth0login.views import analytics

        mock_service.get_analytics_metrics.return_value = {}
        mock_service.get_market_regime.return_value = {}

        request = request_factory.get('/analytics')
        request.user = mock_authenticated_user

        analytics(request)

        mock_render.assert_called_once()


class TestAlerts:
    """Test alerts view."""

    @patch('backend.auth0login.views.dashboard_service')
    @patch('backend.auth0login.views.render')
    def test_alerts(
        self,
        mock_render,
        mock_service,
        request_factory,
        mock_authenticated_user,
    ):
        """Test alerts page."""
        from backend.auth0login.views import alerts

        mock_service.get_recent_alerts.return_value = []
        mock_service.get_unread_alert_count.return_value = 0

        request = request_factory.get('/alerts')
        request.user = mock_authenticated_user

        alerts(request)

        mock_render.assert_called_once()


class TestSystemStatus:
    """Test system_status view."""

    @patch('backend.auth0login.views.dashboard_service')
    @patch('backend.auth0login.views.render')
    def test_system_status(
        self,
        mock_render,
        mock_service,
        request_factory,
        mock_authenticated_user,
    ):
        """Test system status page."""
        from backend.auth0login.views import system_status

        mock_service.get_system_status.return_value = {}

        request = request_factory.get('/status')
        request.user = mock_authenticated_user

        system_status(request)

        mock_render.assert_called_once()


class TestCryptoTrading:
    """Test crypto_trading view."""

    @patch('backend.auth0login.views.dashboard_service')
    @patch('backend.auth0login.views.render')
    def test_crypto_trading(
        self,
        mock_render,
        mock_service,
        request_factory,
        mock_authenticated_user,
    ):
        """Test crypto trading page."""
        from backend.auth0login.views import crypto_trading

        mock_service.get_crypto_dashboard.return_value = {}

        request = request_factory.get('/crypto')
        request.user = mock_authenticated_user

        crypto_trading(request)

        mock_render.assert_called_once()


class TestExtendedHours:
    """Test extended_hours view."""

    @patch('backend.auth0login.views.dashboard_service')
    @patch('backend.auth0login.views.render')
    def test_extended_hours_get(
        self,
        mock_render,
        mock_service,
        request_factory,
        mock_authenticated_user,
    ):
        """Test extended hours GET."""
        from backend.auth0login.views import extended_hours

        mock_service.get_extended_hours.return_value = {}

        request = request_factory.get('/extended-hours')
        request.user = mock_authenticated_user
        request.method = 'GET'

        extended_hours(request)

        mock_render.assert_called_once()

    @patch('backend.auth0login.views.dashboard_service')
    @patch('backend.auth0login.views.render')
    def test_extended_hours_post(
        self,
        mock_render,
        mock_service,
        request_factory,
        mock_authenticated_user,
    ):
        """Test extended hours POST."""
        from backend.auth0login.views import extended_hours

        mock_service.get_extended_hours.return_value = {}
        mock_service.update_extended_hours_settings.return_value = {
            "status": "success",
            "message": "Settings updated",
        }

        request = request_factory.post('/extended-hours', {
            "action": "update_settings",
            "pre_market_enabled": "on",
            "after_hours_enabled": "on",
        })
        request.user = mock_authenticated_user
        request.method = 'POST'

        extended_hours(request)

        mock_render.assert_called_once()


class TestMarginBorrow:
    """Test margin_borrow view."""

    @patch('backend.auth0login.views.dashboard_service')
    @patch('backend.auth0login.views.render')
    def test_margin_borrow(
        self,
        mock_render,
        mock_service,
        request_factory,
        mock_authenticated_user,
    ):
        """Test margin borrow page."""
        from backend.auth0login.views import margin_borrow

        mock_service.get_margin_borrow.return_value = {}

        request = request_factory.get('/margin')
        request.user = mock_authenticated_user

        margin_borrow(request)

        mock_render.assert_called_once()


class TestExoticSpreads:
    """Test exotic_spreads view."""

    @patch('backend.auth0login.views.dashboard_service')
    @patch('backend.auth0login.views.render')
    def test_exotic_spreads(
        self,
        mock_render,
        mock_service,
        request_factory,
        mock_authenticated_user,
    ):
        """Test exotic spreads page."""
        from backend.auth0login.views import exotic_spreads

        mock_service.get_exotic_spreads.return_value = {}

        request = request_factory.get('/spreads')
        request.user = mock_authenticated_user

        exotic_spreads(request)

        mock_render.assert_called_once()


class TestMLTraining:
    """Test ml_training view."""

    @patch('backend.auth0login.views.dashboard_service')
    @patch('backend.auth0login.views.render')
    def test_ml_training_get(
        self,
        mock_render,
        mock_service,
        request_factory,
        mock_authenticated_user,
    ):
        """Test ML training GET."""
        from backend.auth0login.views import ml_training

        mock_service.get_ml_models_status.return_value = {}
        mock_service.get_feature_availability.return_value = {}

        request = request_factory.get('/ml-training')
        request.user = mock_authenticated_user
        request.method = 'GET'

        ml_training(request)

        mock_render.assert_called_once()

    @patch('backend.auth0login.views.dashboard_service')
    @patch('backend.auth0login.views.render')
    def test_ml_training_post_start_training(
        self,
        mock_render,
        mock_service,
        request_factory,
        mock_authenticated_user,
    ):
        """Test ML training POST start_training."""
        from backend.auth0login.views import ml_training

        mock_service.get_ml_models_status.return_value = {}
        mock_service.get_feature_availability.return_value = {}
        mock_service.start_ml_training.return_value = {
            "status": "success",
            "message": "Training started",
        }

        request = request_factory.post('/ml-training', {
            "action": "start_training",
            "model_type": "lstm",
            "epochs": "100",
            "batch_size": "32",
            "learning_rate": "0.001",
            "data_period": "2y",
            "validation_split": "0.2",
            "symbols": "nasdaq100",
        })
        request.user = mock_authenticated_user
        request.method = 'POST'

        ml_training(request)

        mock_render.assert_called_once()


class TestTaxOptimization:
    """Test tax_optimization view."""

    @patch('backend.auth0login.views.get_tax_optimizer_service')
    @patch('backend.auth0login.views.render')
    def test_tax_optimization(
        self,
        mock_render,
        mock_get_service,
        request_factory,
        mock_authenticated_user,
    ):
        """Test tax optimization page."""
        from backend.auth0login.views import tax_optimization

        mock_service = Mock()
        mock_service.get_year_summary.return_value = {}
        mock_service.get_harvesting_opportunities.return_value = []
        mock_service.get_all_lots.return_value = []
        mock_get_service.return_value = mock_service

        request = request_factory.get('/tax')
        request.user = mock_authenticated_user

        tax_optimization(request)

        mock_render.assert_called_once()


class TestStrategyLeaderboard:
    """Test strategy_leaderboard view."""

    @patch('backend.auth0login.views.LeaderboardService')
    @patch('backend.auth0login.views.render')
    def test_strategy_leaderboard(
        self,
        mock_render,
        mock_service_class,
        request_factory,
        mock_authenticated_user,
    ):
        """Test strategy leaderboard page."""
        from backend.auth0login.views import strategy_leaderboard

        mock_service = Mock()
        mock_service.get_leaderboard.return_value = {}
        mock_service.get_top_performers.return_value = []
        mock_service.get_all_strategies.return_value = []
        mock_service_class.return_value = mock_service

        request = request_factory.get('/leaderboard')
        request.user = mock_authenticated_user

        strategy_leaderboard(request)

        mock_render.assert_called_once()


class TestStrategyBuilder:
    """Test strategy_builder view."""

    @patch('backend.auth0login.views.CustomStrategy')
    @patch('backend.auth0login.views.CustomStrategyRunner')
    @patch('backend.auth0login.views.get_strategy_templates')
    @patch('backend.auth0login.views.render')
    def test_strategy_builder(
        self,
        mock_render,
        mock_get_templates,
        mock_runner_class,
        mock_strategy_class,
        request_factory,
        mock_authenticated_user,
    ):
        """Test strategy builder page."""
        from backend.auth0login.views import strategy_builder

        mock_strategy_class.objects.filter.return_value = []
        mock_runner_class.get_available_indicators.return_value = {}
        mock_runner_class.get_available_operators.return_value = {}
        mock_runner_class.get_exit_types.return_value = {}
        mock_get_templates.return_value = []

        request = request_factory.get('/builder')
        request.user = mock_authenticated_user

        strategy_builder(request)

        mock_render.assert_called_once()


class TestLogout:
    """Test logout view."""

    @patch('backend.auth0login.views.HttpResponseRedirect')
    @patch('backend.auth0login.views.log_out')
    def test_logout(
        self,
        mock_logout,
        mock_redirect,
        request_factory,
        mock_authenticated_user,
    ):
        """Test logout."""
        from backend.auth0login.views import logout

        with patch('backend.auth0login.views.settings') as mock_settings:
            mock_settings.SOCIAL_AUTH_AUTH0_DOMAIN = "test.auth0.com"
            mock_settings.SOCIAL_AUTH_AUTH0_KEY = "test_client_id"

            request = request_factory.get('/logout')
            request.user = mock_authenticated_user
            request.build_absolute_uri = Mock(return_value="http://localhost/")

            logout(request)

            mock_logout.assert_called_once_with(request)
            mock_redirect.assert_called_once()
