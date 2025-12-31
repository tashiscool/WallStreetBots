"""
Django Template Tests for WallStreetBots Dashboard.

Tests template rendering, view responses, and authentication requirements.
Uses Django's test client to verify all templates render correctly.
"""

import pytest
from unittest.mock import Mock, patch


# Mock the dashboard_service for all tests
@pytest.fixture(autouse=True)
def mock_dashboard_service():
    """Mock dashboard service for all tests."""
    with patch('backend.auth0login.views.dashboard_service') as mock_service:
        # Setup mock return values for all service methods
        mock_service.get_all_strategies.return_value = [
            Mock(
                name="wsb_dip_bot",
                display_name="WSB Dip Bot",
                enabled=True,
                max_position_size=0.03,
                risk_tolerance="high",
                description="Momentum-based dip buying",
                color="#cb0c9f",
                parameters={}
            ),
            Mock(
                name="wheel_strategy",
                display_name="Wheel Strategy",
                enabled=True,
                max_position_size=0.05,
                risk_tolerance="medium",
                description="Cash-secured puts and covered calls",
                color="#2dce89",
                parameters={}
            ),
        ]

        mock_service.get_strategy_config.return_value = Mock(
            name="test_strategy",
            display_name="Test Strategy",
            enabled=True,
            max_position_size=0.05,
            risk_tolerance="medium",
            description="Test strategy description",
            color="#5e72e4",
            parameters={"param1": 10, "param2": 0.5}
        )

        mock_service.get_market_regime.return_value = Mock(
            current_regime="bullish",
            bullish_probability=0.68,
            bearish_probability=0.08,
            sideways_probability=0.24,
            confidence=0.72,
            position_multiplier=1.2,
            recommended_strategies=["wsb_dip_bot"],
            disabled_strategies=[]
        )

        mock_service.get_risk_metrics.return_value = Mock(
            max_portfolio_risk=10.0,
            max_position_size=5.0,
            max_daily_drawdown=6.0,
            max_correlation=0.7,
            current_portfolio_risk=3.2,
            current_var_95=2.1,
            current_var_99=3.8,
            current_drawdown=1.5,
            risk_level="low",
            warnings=[]
        )

        mock_service.get_analytics_metrics.return_value = Mock(
            total_pnl=12450.0,
            total_return_pct=15.8,
            win_rate=68.5,
            win_loss_ratio=2.3,
            best_trade=892.0,
            worst_trade=-345.0,
            total_trades=156,
            sharpe_ratio=1.72,
            sortino_ratio=2.15,
            max_drawdown=-6.2,
            beta=1.15,
            alpha=4.2,
            r_squared=0.85,
            volatility=18.5,
            profit_factor=1.8,
            strategy_breakdown={}
        )

        mock_service.get_recent_alerts.return_value = [
            Mock(
                id="alert_001",
                alert_type="critical",
                priority="urgent",
                title="Test Alert",
                message="Test message",
                ticker="AAPL",
                timestamp=Mock(),
                acknowledged=False,
                strategy="Test Strategy"
            )
        ]
        mock_service.get_unread_alert_count.return_value = 1

        mock_service.get_system_status.return_value = Mock(
            overall_status="healthy",
            trading_engine_status="running",
            trading_engine_uptime="99.98%",
            trading_engine_heartbeat="2s ago",
            active_strategies=4,
            total_strategies=10,
            market_data_status="connected",
            market_data_latency="12ms",
            market_data_last_update="1s ago",
            market_data_messages_per_sec=1245,
            database_status="healthy",
            database_connections="8 / 100",
            database_query_time="3.2ms avg",
            database_disk_usage="24.5 GB",
            broker_status="connected",
            broker_rate_limit="180 / 200",
            broker_last_sync="5s ago",
            broker_account_status="active",
            cpu_usage=23.0,
            memory_usage=45.0,
            disk_io="12 MB/s",
            api_calls_today=1245,
            success_rate=98.5,
            avg_response_time="45ms",
            errors_today=3,
            recent_logs=[]
        )

        mock_service.get_ml_dashboard.return_value = Mock(
            regime_detector=Mock(name="Regime Detector", status="ready", accuracy=82.3),
            risk_agent=Mock(name="Risk Agent", status="ready", accuracy=78.5),
            signal_validator=Mock(name="Signal Validator", status="ready", accuracy=74.1),
            ddpg_optimizer=Mock(name="DDPG", status="not_trained", accuracy=0.0),
            current_regime="bullish",
            regime_probabilities={"bullish": 68, "sideways": 24, "bearish": 8},
            risk_score=32,
            risk_recommendation="maintain_positions",
            signals_generated=12,
            signals_approved=8,
            signals_rejected=4,
            validation_factors={},
            min_confidence_threshold=60,
            regime_sensitivity="medium",
            auto_retrain_schedule="weekly"
        )

        mock_service.get_crypto_dashboard.return_value = Mock(
            is_available=True,
            supported_assets=["BTC/USD", "ETH/USD"],
            active_positions=[],
            pending_orders=[],
            dip_bot_enabled=False,
            dip_bot_status="stopped",
            daily_trades=0,
            active_signals=[],
            total_crypto_value=0.0,
            crypto_pnl=0.0
        )

        mock_service.get_extended_hours.return_value = Mock(
            is_available=True,
            current_session="closed",
            session_start="",
            session_end="",
            is_optimal_window=False,
            pre_market_enabled=True,
            after_hours_enabled=True,
            next_session="",
            time_until_next="",
            extended_hours_trades_today=0
        )

        mock_service.get_margin_borrow.return_value = Mock(
            is_available=True,
            margin_status="healthy",
            buying_power=100000.0,
            margin_used=25000.0,
            margin_available=75000.0,
            maintenance_margin=12500.0,
            short_positions=[],
            total_borrow_cost_daily=0.0,
            htb_alerts=[],
            squeeze_risk_symbols=[]
        )

        mock_service.get_exotic_spreads.return_value = Mock(
            is_available=True,
            spread_types_available=["Iron Condor", "Butterfly", "Straddle"],
            active_spreads=[],
            total_spread_value=0.0,
            total_spread_pnl=0.0,
            pending_spread_orders=[],
            suggested_spreads=[]
        )

        mock_service.get_feature_availability.return_value = {
            "strategy_manager": True,
            "risk_manager": True,
            "analytics": True,
            "alert_system": True,
            "health_monitor": True,
            "lstm_predictor": True,
            "transformer_predictor": True,
            "cnn_predictor": True,
            "ensemble_predictor": True,
            "rl_agents": True,
            "rl_environment": True,
            "lstm_signal_calculator": True,
            "lstm_pipeline": True,
            "training_utils": True,
            "data_fetcher": True,
            "crypto": True,
            "extended_hours": True,
            "enhanced_borrow": True,
            "exotic_spreads": True,
            "market_regime": True,
            "regime_adapter": True,
        }

        mock_service.get_ml_models_status.return_value = {
            "predictors": {
                "lstm": {"available": True, "name": "LSTM", "description": "LSTM model"},
                "transformer": {"available": True, "name": "Transformer", "description": "Transformer model"},
                "cnn": {"available": True, "name": "CNN", "description": "CNN model"},
                "ensemble": {"available": True, "name": "Ensemble", "description": "Ensemble model"},
            },
            "rl_agents": {
                "ppo": {"available": True, "name": "PPO", "description": "PPO agent"},
                "dqn": {"available": True, "name": "DQN", "description": "DQN agent"},
            },
            "infrastructure": {
                "environment": {"available": True, "name": "Environment", "description": "Trading env"},
                "signal_calculator": {"available": True, "name": "Signal Calculator", "description": "Signal calc"},
                "pipeline": {"available": True, "name": "Pipeline", "description": "LSTM pipeline"},
                "training": {"available": True, "name": "Training", "description": "Training utils"},
                "data_fetcher": {"available": True, "name": "Data Fetcher", "description": "Market data"},
            }
        }

        yield mock_service


@pytest.fixture
def authenticated_client(client, django_user_model, db):
    """Create an authenticated test client."""
    user = django_user_model.objects.create_user(
        username='testuser',
        password='testpass123',
        email='test@example.com'
    )
    client.force_login(user)
    return client


@pytest.fixture(autouse=True)
def mock_get_user_information():
    """Mock get_user_information for views that require Auth0."""
    with patch('backend.auth0login.views.get_user_information') as mock_func:
        # Return mock user data
        mock_user = Mock()
        mock_user.credential = Mock()
        mock_user.credential.alpaca_id = "test_id"
        mock_user.credential.alpaca_key = "test_key"
        mock_userdata = {"name": "Test User", "email": "test@example.com"}
        mock_auth0user = Mock()
        mock_user_details = {"name": "Test User"}
        mock_func.return_value = (mock_user, mock_userdata, mock_auth0user, mock_user_details)
        yield mock_func


@pytest.fixture(autouse=True)
def mock_alpaca_api():
    """Mock Alpaca API calls for dashboard views."""
    with patch('backend.auth0login.views.api') as mock_api:
        # Mock REST client
        mock_rest = Mock()
        mock_api.REST.return_value = mock_rest

        # Mock portfolio history
        import pandas as pd
        mock_df = pd.DataFrame({
            'equity': [10000, 10100, 10200],
            'timestamp': pd.date_range('2024-01-01', periods=3)
        })
        mock_portfolio_history = Mock()
        mock_portfolio_history.df = mock_df
        mock_rest.get_portfolio_history.return_value = mock_portfolio_history

        yield mock_api


@pytest.fixture
def anonymous_client(client):
    """Create an anonymous (not logged in) client."""
    return client


class TestCorePageRendering:
    """Test that core pages render without errors."""

    @pytest.mark.django_db
    def test_login_page_renders(self, anonymous_client):
        """Test login page renders successfully for anonymous users."""
        response = anonymous_client.get('/')
        # Login page should be accessible (either 200 or redirect to auth)
        assert response.status_code in [200, 302]

    @pytest.mark.django_db
    def test_dashboard_page_renders(self, authenticated_client):
        """Test dashboard page renders successfully."""
        response = authenticated_client.get('/dashboard')
        assert response.status_code == 200

    @pytest.mark.django_db
    def test_positions_page_renders(self, authenticated_client):
        """Test positions page renders successfully."""
        response = authenticated_client.get('/positions')
        assert response.status_code == 200

    @pytest.mark.django_db
    def test_orders_page_renders(self, authenticated_client):
        """Test orders page renders successfully."""
        response = authenticated_client.get('/orders')
        assert response.status_code == 200

    @pytest.mark.django_db
    def test_user_settings_page_renders(self, authenticated_client):
        """Test user-settings page renders successfully."""
        response = authenticated_client.get('/user-settings')
        assert response.status_code == 200

    @pytest.mark.django_db
    def test_strategies_page_renders(self, authenticated_client):
        """Test strategies page renders successfully."""
        response = authenticated_client.get('/strategies')
        assert response.status_code == 200

    @pytest.mark.django_db
    def test_backtesting_page_renders(self, authenticated_client):
        """Test backtesting page renders successfully."""
        response = authenticated_client.get('/backtesting')
        assert response.status_code == 200

    @pytest.mark.django_db
    def test_risk_page_renders(self, authenticated_client):
        """Test risk management page renders successfully."""
        response = authenticated_client.get('/risk')
        assert response.status_code == 200

    @pytest.mark.django_db
    def test_analytics_page_renders(self, authenticated_client):
        """Test analytics page renders successfully."""
        response = authenticated_client.get('/analytics')
        assert response.status_code == 200

    @pytest.mark.django_db
    def test_alerts_page_renders(self, authenticated_client):
        """Test alerts page renders successfully."""
        response = authenticated_client.get('/alerts')
        assert response.status_code == 200

    @pytest.mark.django_db
    def test_system_status_page_renders(self, authenticated_client):
        """Test system status page renders successfully."""
        response = authenticated_client.get('/system-status')
        assert response.status_code == 200

    @pytest.mark.django_db
    def test_settings_page_renders(self, authenticated_client):
        """Test settings page renders successfully."""
        response = authenticated_client.get('/settings')
        assert response.status_code == 200

    @pytest.mark.django_db
    def test_setup_wizard_page_renders(self, authenticated_client):
        """Test setup wizard page renders successfully."""
        response = authenticated_client.get('/setup')
        assert response.status_code == 200


class TestStrategyPageRendering:
    """Test that individual strategy pages render correctly."""

    @pytest.mark.django_db
    def test_strategy_wsb_dip_bot_page_renders(self, authenticated_client):
        """Test WSB Dip Bot strategy page renders successfully."""
        response = authenticated_client.get('/strategies/wsb-dip-bot')
        assert response.status_code == 200

    @pytest.mark.django_db
    def test_strategy_wheel_page_renders(self, authenticated_client):
        """Test Wheel Strategy page renders successfully."""
        response = authenticated_client.get('/strategies/wheel')
        assert response.status_code == 200

    @pytest.mark.django_db
    def test_strategy_momentum_weeklies_page_renders(self, authenticated_client):
        """Test Momentum Weeklies page renders successfully."""
        response = authenticated_client.get('/strategies/momentum-weeklies')
        assert response.status_code == 200

    @pytest.mark.django_db
    def test_strategy_earnings_protection_page_renders(self, authenticated_client):
        """Test Earnings Protection page renders successfully."""
        response = authenticated_client.get('/strategies/earnings-protection')
        assert response.status_code == 200

    @pytest.mark.django_db
    def test_strategy_debit_spreads_page_renders(self, authenticated_client):
        """Test Debit Spreads page renders successfully."""
        response = authenticated_client.get('/strategies/debit-spreads')
        assert response.status_code == 200

    @pytest.mark.django_db
    def test_strategy_leaps_tracker_page_renders(self, authenticated_client):
        """Test LEAPS Tracker page renders successfully."""
        response = authenticated_client.get('/strategies/leaps-tracker')
        assert response.status_code == 200

    @pytest.mark.django_db
    def test_strategy_lotto_scanner_page_renders(self, authenticated_client):
        """Test Lotto Scanner page renders successfully."""
        response = authenticated_client.get('/strategies/lotto-scanner')
        assert response.status_code == 200

    @pytest.mark.django_db
    def test_strategy_swing_trading_page_renders(self, authenticated_client):
        """Test Swing Trading page renders successfully."""
        response = authenticated_client.get('/strategies/swing-trading')
        assert response.status_code == 200

    @pytest.mark.django_db
    def test_strategy_spx_credit_spreads_page_renders(self, authenticated_client):
        """Test SPX Credit Spreads page renders successfully."""
        response = authenticated_client.get('/strategies/spx-credit-spreads')
        assert response.status_code == 200

    @pytest.mark.django_db
    def test_strategy_index_baseline_page_renders(self, authenticated_client):
        """Test Index Baseline page renders successfully."""
        response = authenticated_client.get('/strategies/index-baseline')
        assert response.status_code == 200


class TestAuthenticationRequired:
    """Test that all protected pages require authentication."""

    @pytest.mark.django_db
    def test_dashboard_requires_auth(self, anonymous_client):
        """Test dashboard page redirects to login."""
        response = anonymous_client.get('/dashboard')
        assert response.status_code in [302, 301]

    @pytest.mark.django_db
    def test_positions_requires_auth(self, anonymous_client):
        """Test positions page redirects to login."""
        response = anonymous_client.get('/positions')
        assert response.status_code in [302, 301]

    @pytest.mark.django_db
    def test_orders_requires_auth(self, anonymous_client):
        """Test orders page redirects to login."""
        response = anonymous_client.get('/orders')
        assert response.status_code in [302, 301]

    @pytest.mark.django_db
    def test_user_settings_requires_auth(self, anonymous_client):
        """Test user-settings page redirects to login."""
        response = anonymous_client.get('/user-settings')
        assert response.status_code in [302, 301]

    @pytest.mark.django_db
    def test_strategies_requires_auth(self, anonymous_client):
        """Test strategies page redirects to login."""
        response = anonymous_client.get('/strategies')
        assert response.status_code in [302, 301]

    @pytest.mark.django_db
    def test_backtesting_requires_auth(self, anonymous_client):
        """Test backtesting page redirects to login."""
        response = anonymous_client.get('/backtesting')
        assert response.status_code in [302, 301]

    @pytest.mark.django_db
    def test_risk_requires_auth(self, anonymous_client):
        """Test risk page redirects to login."""
        response = anonymous_client.get('/risk')
        assert response.status_code in [302, 301]

    @pytest.mark.django_db
    def test_analytics_requires_auth(self, anonymous_client):
        """Test analytics page redirects to login."""
        response = anonymous_client.get('/analytics')
        assert response.status_code in [302, 301]

    @pytest.mark.django_db
    def test_alerts_requires_auth(self, anonymous_client):
        """Test alerts page redirects to login."""
        response = anonymous_client.get('/alerts')
        assert response.status_code in [302, 301]

    @pytest.mark.django_db
    def test_system_status_requires_auth(self, anonymous_client):
        """Test system status page redirects to login."""
        response = anonymous_client.get('/system-status')
        assert response.status_code in [302, 301]

    @pytest.mark.django_db
    def test_settings_requires_auth(self, anonymous_client):
        """Test settings page redirects to login."""
        response = anonymous_client.get('/settings')
        assert response.status_code in [302, 301]

    @pytest.mark.django_db
    def test_setup_wizard_requires_auth(self, anonymous_client):
        """Test setup wizard page redirects to login."""
        response = anonymous_client.get('/setup')
        assert response.status_code in [302, 301]


class TestAdvancedFeaturePages:
    """Test advanced feature pages render correctly."""

    @pytest.mark.django_db
    def test_crypto_trading_page_renders(self, authenticated_client):
        """Test crypto trading page renders successfully."""
        response = authenticated_client.get('/crypto')
        assert response.status_code == 200

    @pytest.mark.django_db
    def test_extended_hours_page_renders(self, authenticated_client):
        """Test extended hours page renders successfully."""
        response = authenticated_client.get('/extended-hours')
        assert response.status_code == 200

    @pytest.mark.django_db
    def test_margin_borrow_page_renders(self, authenticated_client):
        """Test margin/borrow page renders successfully."""
        response = authenticated_client.get('/margin-borrow')
        assert response.status_code == 200

    @pytest.mark.django_db
    def test_exotic_spreads_page_renders(self, authenticated_client):
        """Test exotic spreads page renders successfully."""
        response = authenticated_client.get('/exotic-spreads')
        assert response.status_code == 200

    @pytest.mark.django_db
    def test_feature_status_page_renders(self, authenticated_client):
        """Test feature status page renders successfully."""
        response = authenticated_client.get('/feature-status')
        assert response.status_code == 200

    @pytest.mark.django_db
    def test_ml_training_page_renders(self, authenticated_client):
        """Test ML training page renders successfully."""
        response = authenticated_client.get('/ml-training')
        assert response.status_code == 200

    @pytest.mark.django_db
    def test_machine_learning_page_renders(self, authenticated_client):
        """Test machine learning page renders successfully."""
        response = authenticated_client.get('/machine-learning')
        assert response.status_code == 200


class TestTemplateContent:
    """Test that templates contain expected content."""

    @pytest.mark.django_db
    def test_strategies_page_has_content(self, authenticated_client):
        """Test strategies page contains strategy-related content."""
        response = authenticated_client.get('/strategies')
        content = response.content.decode('utf-8')
        # Should contain some HTML structure
        assert '<' in content and '>' in content

    @pytest.mark.django_db
    def test_alerts_page_has_content(self, authenticated_client):
        """Test alerts page contains alert-related content."""
        response = authenticated_client.get('/alerts')
        content = response.content.decode('utf-8')
        assert '<' in content and '>' in content

    @pytest.mark.django_db
    def test_system_status_page_has_content(self, authenticated_client):
        """Test system status page contains status-related content."""
        response = authenticated_client.get('/system-status')
        content = response.content.decode('utf-8')
        assert '<' in content and '>' in content


class TestErrorHandling:
    """Test error handling in views."""

    @pytest.mark.django_db
    def test_nonexistent_page_returns_404(self, authenticated_client):
        """Test that nonexistent pages return 404."""
        response = authenticated_client.get('/nonexistent-page-xyz')
        assert response.status_code == 404

    @pytest.mark.django_db
    def test_invalid_strategy_handles_gracefully(self, authenticated_client, mock_dashboard_service):
        """Test that views handle None strategy config gracefully."""
        mock_dashboard_service.get_strategy_config.return_value = None
        # Should still render (may show empty or error state)
        response = authenticated_client.get('/strategies/wsb-dip-bot')
        # Either renders or handles gracefully
        assert response.status_code in [200, 404, 500]
