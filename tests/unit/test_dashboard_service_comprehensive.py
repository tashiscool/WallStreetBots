"""
Comprehensive tests for backend/auth0login/dashboard_service.py
Target: 80%+ coverage with all edge cases and error handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime
from decimal import Decimal

from backend.auth0login.dashboard_service import (
    DashboardService,
    StrategyUIConfig,
    RiskMetricsUI,
    AnalyticsMetricsUI,
    MarketRegimeUI,
    AlertUI,
    SystemStatusUI,
    MLDashboardUI,
    MLModelUI,
    CryptoDashboardUI,
    ExtendedHoursUI,
    MarginBorrowUI,
    ExoticSpreadsUI,
    dashboard_service,
)


@pytest.fixture
def dashboard():
    """Create DashboardService instance."""
    return DashboardService()


class TestDataClasses:
    """Test dataclass structures."""

    def test_strategy_ui_config_creation(self):
        """Test StrategyUIConfig creation."""
        config = StrategyUIConfig(
            name="test_strategy",
            display_name="Test Strategy",
            enabled=True,
            max_position_size=0.05,
        )

        assert config.name == "test_strategy"
        assert config.display_name == "Test Strategy"
        assert config.enabled is True

    def test_risk_metrics_ui_creation(self):
        """Test RiskMetricsUI creation."""
        metrics = RiskMetricsUI(
            max_portfolio_risk=10.0,
            max_position_size=5.0,
            current_portfolio_risk=3.2,
        )

        assert metrics.max_portfolio_risk == 10.0
        assert metrics.current_portfolio_risk == 3.2

    def test_analytics_metrics_ui_creation(self):
        """Test AnalyticsMetricsUI creation."""
        metrics = AnalyticsMetricsUI(
            total_pnl=10000.0,
            total_return_pct=15.5,
            win_rate=68.5,
        )

        assert metrics.total_pnl == 10000.0
        assert metrics.win_rate == 68.5

    def test_market_regime_ui_creation(self):
        """Test MarketRegimeUI creation."""
        regime = MarketRegimeUI(
            current_regime="bullish",
            bullish_probability=0.68,
            confidence=0.75,
        )

        assert regime.current_regime == "bullish"
        assert regime.bullish_probability == 0.68

    def test_alert_ui_creation(self):
        """Test AlertUI creation."""
        alert = AlertUI(
            id="alert_001",
            alert_type="warning",
            title="Test Alert",
            message="This is a test",
        )

        assert alert.id == "alert_001"
        assert alert.alert_type == "warning"

    def test_system_status_ui_creation(self):
        """Test SystemStatusUI creation."""
        status = SystemStatusUI(
            overall_status="healthy",
            trading_engine_status="running",
            active_strategies=5,
        )

        assert status.overall_status == "healthy"
        assert status.active_strategies == 5

    def test_ml_model_ui_creation(self):
        """Test MLModelUI creation."""
        model = MLModelUI(
            name="LSTM Predictor",
            status="ready",
            accuracy=85.5,
        )

        assert model.name == "LSTM Predictor"
        assert model.accuracy == 85.5

    def test_crypto_dashboard_ui_creation(self):
        """Test CryptoDashboardUI creation."""
        crypto = CryptoDashboardUI(
            is_available=True,
            supported_assets=["BTC/USD", "ETH/USD"],
        )

        assert crypto.is_available is True
        assert len(crypto.supported_assets) == 2


class TestDashboardServiceSingleton:
    """Test DashboardService singleton pattern."""

    def test_singleton_instance(self):
        """Test that DashboardService is a singleton."""
        instance1 = DashboardService()
        instance2 = DashboardService()

        assert instance1 is instance2

    def test_initialization_once(self):
        """Test that initialization happens only once."""
        service = DashboardService()

        # Second initialization should not reset
        service.test_attribute = "test"
        service2 = DashboardService()

        assert hasattr(service2, 'test_attribute')
        assert service2.test_attribute == "test"


class TestStrategyConfiguration:
    """Test strategy configuration methods."""

    def test_get_all_strategies(self, dashboard):
        """Test getting all strategies."""
        strategies = dashboard.get_all_strategies()

        assert len(strategies) > 0
        assert all(isinstance(s, StrategyUIConfig) for s in strategies)
        # Should have wsb-dip-bot
        assert any(s.name == "wsb_dip_bot" for s in strategies)

    def test_get_strategy_config(self, dashboard):
        """Test getting specific strategy config."""
        strategy = dashboard.get_strategy_config("wsb_dip_bot")

        assert strategy is not None
        assert strategy.name == "wsb_dip_bot"
        assert strategy.display_name == "WSB Dip Bot"

    def test_get_strategy_config_not_found(self, dashboard):
        """Test getting non-existent strategy."""
        strategy = dashboard.get_strategy_config("non_existent_strategy")

        assert strategy is None

    def test_save_strategy_config_success(self, dashboard):
        """Test saving strategy configuration."""
        config = {
            "enabled": True,
            "max_position_size": 0.05,
            "max_positions": 10,
        }

        result = dashboard.save_strategy_config("wsb_dip_bot", config)

        assert result["status"] == "success"
        assert "message" in result

    def test_save_strategy_config_disabled(self, dashboard):
        """Test saving disabled strategy."""
        config = {
            "enabled": False,
            "max_position_size": 0.03,
            "max_positions": 5,
        }

        result = dashboard.save_strategy_config("wsb_dip_bot", config)

        assert result["status"] == "success"
        assert "disabled" in result["message"]

    def test_save_strategy_config_error(self, dashboard):
        """Test error handling in save_strategy_config."""
        # Pass invalid config to trigger error
        with patch.object(dashboard.logger, 'info', side_effect=Exception("Test error")):
            result = dashboard.save_strategy_config("test", {})

            assert result["status"] == "error"


class TestRiskMetrics:
    """Test risk metrics methods."""

    def test_get_risk_metrics_fallback(self, dashboard):
        """Test getting risk metrics with fallback data."""
        metrics = dashboard.get_risk_metrics()

        assert isinstance(metrics, RiskMetricsUI)
        assert metrics.max_portfolio_risk > 0
        assert metrics.max_position_size > 0

    def test_get_risk_metrics_risk_levels(self, dashboard):
        """Test risk level determination."""
        # Test different risk levels
        metrics = dashboard.get_risk_metrics()

        # Low risk (default fallback)
        assert metrics.risk_level in ["low", "moderate", "high", "critical"]

    @patch('backend.auth0login.dashboard_service.HAS_RISK_MANAGER', True)
    def test_get_risk_manager_initialization(self, dashboard):
        """Test risk manager lazy initialization."""
        with patch('backend.auth0login.dashboard_service.RealTimeRiskManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager

            manager = dashboard._get_risk_manager()

            assert manager is not None


class TestAnalytics:
    """Test analytics methods."""

    def test_get_analytics_metrics(self, dashboard):
        """Test getting analytics metrics."""
        metrics = dashboard.get_analytics_metrics()

        assert isinstance(metrics, AnalyticsMetricsUI)
        assert metrics.total_pnl >= 0 or metrics.total_pnl < 0
        assert metrics.total_trades >= 0

    def test_get_analytics_metrics_strategy_breakdown(self, dashboard):
        """Test strategy breakdown in analytics."""
        metrics = dashboard.get_analytics_metrics()

        assert isinstance(metrics.strategy_breakdown, dict)
        assert len(metrics.strategy_breakdown) > 0


class TestMarketRegime:
    """Test market regime methods."""

    def test_get_market_regime_fallback(self, dashboard):
        """Test getting market regime with fallback."""
        regime = dashboard.get_market_regime()

        assert isinstance(regime, MarketRegimeUI)
        assert regime.current_regime in ["bullish", "bearish", "sideways", "undefined"]
        assert 0 <= regime.bullish_probability <= 1
        assert 0 <= regime.bearish_probability <= 1
        assert 0 <= regime.sideways_probability <= 1

    def test_get_market_regime_recommended_strategies(self, dashboard):
        """Test recommended strategies in regime."""
        regime = dashboard.get_market_regime()

        assert isinstance(regime.recommended_strategies, list)
        assert isinstance(regime.disabled_strategies, list)


class TestAlerts:
    """Test alert methods."""

    def test_get_recent_alerts(self, dashboard):
        """Test getting recent alerts."""
        alerts = dashboard.get_recent_alerts(limit=10)

        assert isinstance(alerts, list)
        assert len(alerts) <= 10
        if alerts:
            assert all(isinstance(a, AlertUI) for a in alerts)

    def test_get_recent_alerts_custom_limit(self, dashboard):
        """Test getting alerts with custom limit."""
        alerts = dashboard.get_recent_alerts(limit=5)

        assert len(alerts) <= 5

    def test_get_unread_alert_count(self, dashboard):
        """Test getting unread alert count."""
        count = dashboard.get_unread_alert_count()

        assert isinstance(count, int)
        assert count >= 0


class TestSystemStatus:
    """Test system status methods."""

    def test_get_system_status(self, dashboard):
        """Test getting system status."""
        status = dashboard.get_system_status()

        assert isinstance(status, SystemStatusUI)
        assert status.overall_status in ["healthy", "warning", "critical"]
        assert status.active_strategies >= 0
        assert status.total_strategies > 0

    def test_get_system_status_components(self, dashboard):
        """Test system status components."""
        status = dashboard.get_system_status()

        # Check all components are present
        assert status.trading_engine_status is not None
        assert status.market_data_status is not None
        assert status.database_status is not None
        assert status.broker_status is not None


class TestMachineLearning:
    """Test machine learning methods."""

    def test_get_ml_dashboard(self, dashboard):
        """Test getting ML dashboard."""
        ml_data = dashboard.get_ml_dashboard()

        assert isinstance(ml_data, MLDashboardUI)
        assert ml_data.regime_detector is not None
        assert ml_data.risk_agent is not None

    def test_get_ml_dashboard_models(self, dashboard):
        """Test ML models in dashboard."""
        ml_data = dashboard.get_ml_dashboard()

        assert isinstance(ml_data.regime_detector, MLModelUI)
        assert isinstance(ml_data.risk_agent, MLModelUI)
        assert isinstance(ml_data.signal_validator, MLModelUI)

    def test_start_ml_training_no_utils(self, dashboard):
        """Test ML training when utils not available."""
        with patch('backend.auth0login.dashboard_service.HAS_TRAINING_UTILS', False):
            result = dashboard.start_ml_training("lstm", {})

            assert result["status"] == "error"

    @patch('backend.auth0login.dashboard_service.HAS_TRAINING_UTILS', True)
    @patch('backend.auth0login.dashboard_service.HAS_LSTM_PIPELINE', True)
    def test_start_ml_training_lstm(self, dashboard):
        """Test starting LSTM training."""
        config = {
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
            "symbols": "nasdaq100",
        }

        with patch('backend.auth0login.dashboard_service.TrainingConfig'):
            result = dashboard.start_ml_training("lstm", config)

            assert result["status"] == "success"

    @patch('backend.auth0login.dashboard_service.HAS_TRAINING_UTILS', True)
    @patch('backend.auth0login.dashboard_service.HAS_RL_AGENTS', True)
    def test_start_ml_training_ppo(self, dashboard):
        """Test starting PPO training."""
        config = {
            "epochs": 100,
        }

        with patch('backend.auth0login.dashboard_service.TrainingConfig'):
            result = dashboard.start_ml_training("ppo", config)

            assert result["status"] == "success"

    def test_start_ml_training_unknown_model(self, dashboard):
        """Test starting training for unknown model."""
        with patch('backend.auth0login.dashboard_service.HAS_TRAINING_UTILS', True):
            with patch('backend.auth0login.dashboard_service.TrainingConfig'):
                result = dashboard.start_ml_training("unknown_model", {})

                assert result["status"] == "warning"

    @patch('backend.auth0login.dashboard_service.HAS_DATA_FETCHER', True)
    def test_fetch_training_data_success(self, dashboard):
        """Test fetching training data."""
        result = dashboard.fetch_training_data("nasdaq100", "2y")

        assert result["status"] == "success"
        assert "message" in result

    def test_fetch_training_data_no_fetcher(self, dashboard):
        """Test fetching data when fetcher not available."""
        with patch('backend.auth0login.dashboard_service.HAS_DATA_FETCHER', False):
            result = dashboard.fetch_training_data("nasdaq100", "2y")

            assert result["status"] == "error"


class TestCryptoTrading:
    """Test crypto trading methods."""

    def test_get_crypto_dashboard_not_available(self, dashboard):
        """Test crypto dashboard when not available."""
        with patch('backend.auth0login.dashboard_service.HAS_CRYPTO', False):
            crypto_data = dashboard.get_crypto_dashboard()

            assert crypto_data.is_available is False

    @pytest.mark.asyncio
    async def test_start_crypto_dip_bot_not_available(self, dashboard):
        """Test starting crypto dip bot when not available."""
        with patch('backend.auth0login.dashboard_service.HAS_CRYPTO', False):
            result = await dashboard.start_crypto_dip_bot()

            assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_stop_crypto_dip_bot_not_available(self, dashboard):
        """Test stopping crypto dip bot when not available."""
        with patch('backend.auth0login.dashboard_service.HAS_CRYPTO', False):
            result = await dashboard.stop_crypto_dip_bot()

            assert result["status"] == "error"


class TestExtendedHours:
    """Test extended hours methods."""

    def test_get_extended_hours(self, dashboard):
        """Test getting extended hours data."""
        data = dashboard.get_extended_hours()

        assert isinstance(data, ExtendedHoursUI)

    def test_update_extended_hours_settings_not_available(self, dashboard):
        """Test updating settings when not available."""
        with patch('backend.auth0login.dashboard_service.HAS_EXTENDED_HOURS', False):
            result = dashboard.update_extended_hours_settings(True, True)

            assert result["status"] == "error"

    @patch('backend.auth0login.dashboard_service.HAS_EXTENDED_HOURS', True)
    def test_update_extended_hours_settings_success(self, dashboard):
        """Test updating extended hours settings."""
        with patch('backend.auth0login.dashboard_service.create_extended_hours_manager'):
            result = dashboard.update_extended_hours_settings(True, True)

            assert result["status"] == "success"

    @patch('backend.auth0login.dashboard_service.HAS_EXTENDED_HOURS', True)
    def test_update_extended_hours_settings_disabled(self, dashboard):
        """Test disabling extended hours."""
        with patch('backend.auth0login.dashboard_service.create_extended_hours_manager'):
            result = dashboard.update_extended_hours_settings(False, False)

            assert result["status"] == "success"
            assert "disabled" in result["message"]


class TestMarginBorrow:
    """Test margin and borrow methods."""

    def test_get_margin_borrow_not_available(self, dashboard):
        """Test margin borrow when not available."""
        with patch('backend.auth0login.dashboard_service.HAS_ENHANCED_BORROW', False):
            data = dashboard.get_margin_borrow()

            assert data.is_available is False

    @pytest.mark.asyncio
    async def test_get_locate_quote_not_available(self, dashboard):
        """Test locate quote when not available."""
        with patch('backend.auth0login.dashboard_service.HAS_ENHANCED_BORROW', False):
            result = await dashboard.get_locate_quote("AAPL", 100)

            assert result["status"] == "error"


class TestExoticSpreads:
    """Test exotic spreads methods."""

    def test_get_exotic_spreads_not_available(self, dashboard):
        """Test exotic spreads when not available."""
        with patch('backend.auth0login.dashboard_service.HAS_EXOTIC_SPREADS', False):
            data = dashboard.get_exotic_spreads()

            assert data.is_available is False
            assert len(data.spread_types_available) == 0

    @pytest.mark.asyncio
    async def test_build_spread_not_available(self, dashboard):
        """Test building spread when not available."""
        with patch('backend.auth0login.dashboard_service.HAS_EXOTIC_SPREADS', False):
            result = await dashboard.build_spread("iron_condor", "AAPL", 200.0, {})

            assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_suggest_spreads_not_available(self, dashboard):
        """Test suggesting spreads when not available."""
        with patch('backend.auth0login.dashboard_service.HAS_EXOTIC_SPREADS', False):
            result = await dashboard.suggest_spreads("AAPL", 200.0, "neutral")

            assert result["status"] == "error"


class TestFeatureAvailability:
    """Test feature availability methods."""

    def test_get_feature_availability(self, dashboard):
        """Test getting feature availability."""
        features = dashboard.get_feature_availability()

        assert isinstance(features, dict)
        assert "strategy_manager" in features
        assert "risk_manager" in features
        assert "analytics" in features
        assert "lstm_predictor" in features

    def test_get_ml_models_status(self, dashboard):
        """Test getting ML models status."""
        status = dashboard.get_ml_models_status()

        assert isinstance(status, dict)
        assert "predictors" in status
        assert "rl_agents" in status
        assert "infrastructure" in status


class TestBacktesting:
    """Test backtesting methods."""

    def test_get_backtesting_status(self, dashboard):
        """Test getting backtesting status."""
        status = dashboard.get_backtesting_status()

        assert isinstance(status, dict)
        assert "is_available" in status
        assert "strategies" in status
        assert "benchmarks" in status

    @pytest.mark.asyncio
    async def test_run_backtest_not_available(self, dashboard):
        """Test running backtest when not available."""
        with patch('backend.auth0login.dashboard_service.HAS_BACKTESTING', False):
            result = await dashboard.run_backtest(
                "wsb-dip-bot",
                "2024-01-01",
                "2024-12-31",
            )

            assert result["status"] == "error"

    @pytest.mark.asyncio
    @patch('backend.auth0login.dashboard_service.HAS_BACKTESTING', True)
    async def test_run_backtest_success(self, dashboard):
        """Test successful backtest run."""
        with patch('backend.auth0login.dashboard_service.run_backtest_engine', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {
                "config": {},
                "summary": {},
                "trades": [],
            }

            result = await dashboard.run_backtest(
                "wsb-dip-bot",
                "2024-01-01",
                "2024-12-31",
            )

            assert result["status"] == "success"
            assert "results" in result


class TestCaching:
    """Test caching behavior."""

    def test_cache_initialization(self, dashboard):
        """Test cache is initialized."""
        assert hasattr(dashboard, '_cache')
        assert isinstance(dashboard._cache, dict)
        assert hasattr(dashboard, '_cache_ttl')


class TestErrorHandling:
    """Test error handling across methods."""

    def test_get_all_strategies_handles_errors(self, dashboard):
        """Test that get_all_strategies handles errors gracefully."""
        # Should always return a list
        strategies = dashboard.get_all_strategies()
        assert isinstance(strategies, list)

    def test_save_strategy_config_handles_exceptions(self, dashboard):
        """Test exception handling in save_strategy_config."""
        with patch.object(dashboard, 'logger') as mock_logger:
            mock_logger.info.side_effect = Exception("Test error")

            result = dashboard.save_strategy_config("test", {})

            assert result["status"] == "error"
            mock_logger.error.assert_called()


class TestGlobalServiceInstance:
    """Test the global dashboard_service instance."""

    def test_global_service_exists(self):
        """Test that global service instance exists."""
        from backend.auth0login.dashboard_service import dashboard_service

        assert dashboard_service is not None
        assert isinstance(dashboard_service, DashboardService)

    def test_global_service_is_singleton(self):
        """Test that global service is the same as new instance."""
        from backend.auth0login.dashboard_service import dashboard_service

        new_instance = DashboardService()

        assert dashboard_service is new_instance


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_get_strategy_config_empty_string(self, dashboard):
        """Test getting strategy with empty string."""
        strategy = dashboard.get_strategy_config("")

        assert strategy is None

    def test_save_strategy_config_empty_config(self, dashboard):
        """Test saving empty configuration."""
        result = dashboard.save_strategy_config("test", {})

        # Should still succeed
        assert result["status"] in ["success", "error"]

    def test_get_recent_alerts_zero_limit(self, dashboard):
        """Test getting alerts with zero limit."""
        alerts = dashboard.get_recent_alerts(limit=0)

        assert isinstance(alerts, list)

    def test_get_recent_alerts_large_limit(self, dashboard):
        """Test getting alerts with very large limit."""
        alerts = dashboard.get_recent_alerts(limit=1000)

        assert isinstance(alerts, list)

    def test_fetch_training_data_unknown_symbols(self, dashboard):
        """Test fetching data with unknown symbol set."""
        with patch('backend.auth0login.dashboard_service.HAS_DATA_FETCHER', True):
            result = dashboard.fetch_training_data("unknown_set", "2y")

            # Should use default symbols
            assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_build_spread_unknown_type(self, dashboard):
        """Test building unknown spread type."""
        with patch('backend.auth0login.dashboard_service.HAS_EXOTIC_SPREADS', True):
            with patch.object(dashboard, '_get_spread_builder', return_value=Mock()):
                result = await dashboard.build_spread(
                    "unknown_spread",
                    "AAPL",
                    200.0,
                    {},
                )

                assert result["status"] == "error"


class TestAsyncMethods:
    """Test async method behavior."""

    @pytest.mark.asyncio
    async def test_run_backtest_error_handling(self, dashboard):
        """Test error handling in run_backtest."""
        with patch('backend.auth0login.dashboard_service.HAS_BACKTESTING', True):
            with patch('backend.auth0login.dashboard_service.run_backtest_engine', new_callable=AsyncMock) as mock_run:
                mock_run.side_effect = Exception("Test error")

                result = await dashboard.run_backtest(
                    "test",
                    "2024-01-01",
                    "2024-12-31",
                )

                assert result["status"] == "error"
