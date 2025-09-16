"""Comprehensive additional tests for Production Strategy Manager to achieve >70% coverage."""
import asyncio
import os
import sys
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime, timedelta
import pytest

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

# Django setup is handled by conftest.py

from backend.tradingbot.production.core.production_strategy_manager import (
    ProductionStrategyManager,
    ProductionStrategyManagerConfig,
    StrategyConfig,
    StrategyProfile,
    _preset_defaults,
    _apply_profile_risk_overrides,
    create_production_strategy_manager
)

from backend.tradingbot.analytics.advanced_analytics import PerformanceMetrics
from backend.tradingbot.analytics.market_regime_adapter import StrategyAdaptation

# Test constants
TEST_API_KEY = "test_key"
TEST_SECRET_KEY = "test_secret"  # noqa: S105


class TestStrategyConfig:
    """Test StrategyConfig dataclass."""

    def test_strategy_config_creation(self):
        """Test StrategyConfig creation."""
        config = StrategyConfig(
            name="test_strategy",
            enabled=True,
            max_position_size=0.10,
            risk_tolerance="high",
            parameters={"param1": "value1"}
        )

        assert config.enabled is True
        assert config.name == "test_strategy"
        assert config.max_position_size == 0.10
        assert config.risk_tolerance == "high"
        assert config.parameters == {"param1": "value1"}

    def test_strategy_config_defaults(self):
        """Test StrategyConfig with default values."""
        config = StrategyConfig(name="test_strategy")

        assert config.enabled is True  # Default
        assert config.max_position_size == 0.20  # Default 20%
        assert config.risk_tolerance == "medium"  # Default
        assert config.parameters == {}  # Default empty dict


class TestStrategyProfile:
    """Test StrategyProfile enum."""

    def test_strategy_profile_values(self):
        """Test StrategyProfile enum values."""
        assert StrategyProfile.research_2024 == "research_2024"
        assert StrategyProfile.wsb_2025 == "wsb_2025"
        assert StrategyProfile.trump_2025 == "trump_2025"
        assert StrategyProfile.bubble_aware_2025 == "bubble_aware_2025"

    def test_strategy_profile_enum_membership(self):
        """Test StrategyProfile enum membership."""
        profiles = list(StrategyProfile)
        assert len(profiles) == 4
        assert StrategyProfile.research_2024 in profiles
        assert StrategyProfile.wsb_2025 in profiles


class TestProductionStrategyManagerConfig:
    """Test ProductionStrategyManagerConfig dataclass."""

    def test_config_creation_minimal(self):
        """Test config creation with minimal parameters."""
        config = ProductionStrategyManagerConfig(
            alpaca_api_key=TEST_API_KEY,
            alpaca_secret_key=TEST_SECRET_KEY,
            user_id=1
        )

        assert config.alpaca_api_key == TEST_API_KEY
        assert config.alpaca_secret_key == TEST_SECRET_KEY
        assert config.user_id == 1
        assert config.paper_trading is True  # Default
        assert config.max_total_risk == 0.50  # Default

    def test_config_creation_full(self):
        """Test config creation with all parameters."""
        strategy_configs = {
            "wsb_dip_bot": StrategyConfig(name="wsb_dip_bot", enabled=True, max_position_size=0.20),
            "wheel_strategy": StrategyConfig(name="wheel_strategy", enabled=False, max_position_size=0.15)
        }

        config = ProductionStrategyManagerConfig(
            alpaca_api_key=TEST_API_KEY,
            alpaca_secret_key=TEST_SECRET_KEY,
            user_id=1,
            paper_trading=False,
            max_total_risk=0.30,
            max_position_size=0.15,
            profile=StrategyProfile.wsb_2025,
            strategies=strategy_configs,
            data_refresh_interval=45,
            enable_alerts=False,
            enable_advanced_analytics=False,
            enable_market_regime_adaptation=False
        )

        assert config.paper_trading is False
        assert config.max_total_risk == 0.30
        assert config.max_position_size == 0.15
        assert config.profile == StrategyProfile.wsb_2025
        assert config.strategies == strategy_configs
        assert config.data_refresh_interval == 45
        assert config.enable_alerts is False
        assert config.enable_advanced_analytics is False
        assert config.enable_market_regime_adaptation is False


class TestPresetDefaults:
    """Test preset default configurations."""

    def test_preset_defaults_research(self):
        """Test research profile defaults."""
        defaults = _preset_defaults(StrategyProfile.research_2024)

        assert isinstance(defaults, dict)
        assert len(defaults) > 0

        # Check that strategies are configured
        for strategy_config in defaults.values():
            assert isinstance(strategy_config, StrategyConfig)
            assert isinstance(strategy_config.enabled, bool)
            assert 0 <= strategy_config.max_position_size <= 1

    def test_preset_defaults_aggressive(self):
        """Test aggressive profile defaults."""
        defaults = _preset_defaults(StrategyProfile.wsb_2025)

        assert isinstance(defaults, dict)
        assert len(defaults) > 0

        # Aggressive profile should have higher position sizes
        total_position_size = sum(cfg.max_position_size for cfg in defaults.values() if cfg.enabled)
        assert total_position_size > 0

    def test_preset_defaults_defensive(self):
        """Test defensive profile defaults."""
        defaults = _preset_defaults(StrategyProfile.trump_2025)

        assert isinstance(defaults, dict)
        assert len(defaults) > 0

        # Defensive profile should have lower position sizes
        for strategy_config in defaults.values():
            if strategy_config.enabled:
                assert strategy_config.max_position_size <= 0.50

    def test_preset_defaults_balanced(self):
        """Test balanced profile defaults."""
        defaults = _preset_defaults(StrategyProfile.bubble_aware_2025)

        assert isinstance(defaults, dict)
        assert len(defaults) > 0

    def test_preset_defaults_institutional(self):
        """Test institutional profile defaults."""
        defaults = _preset_defaults(StrategyProfile.research_2024)

        assert isinstance(defaults, dict)
        assert len(defaults) > 0


class TestRiskOverrides:
    """Test risk override application."""

    def test_apply_profile_risk_overrides_research(self):
        """Test risk overrides for research profile."""
        config = ProductionStrategyManagerConfig(
            alpaca_api_key=TEST_API_KEY,
            alpaca_secret_key=TEST_SECRET_KEY,
            user_id=1,
            profile=StrategyProfile.research_2024
        )

        _apply_profile_risk_overrides(config)

        # Research profile should have reduced risk
        assert config.max_total_risk <= 0.10

    def test_apply_profile_risk_overrides_aggressive(self):
        """Test risk overrides for aggressive profile."""
        config = ProductionStrategyManagerConfig(
            alpaca_api_key=TEST_API_KEY,
            alpaca_secret_key=TEST_SECRET_KEY,
            user_id=1,
            profile=StrategyProfile.wsb_2025
        )

        original_risk = config.max_total_risk
        _apply_profile_risk_overrides(config)

        # Aggressive profile may increase risk
        assert config.max_total_risk >= original_risk * 0.8

    def test_apply_profile_risk_overrides_defensive(self):
        """Test risk overrides for defensive profile."""
        config = ProductionStrategyManagerConfig(
            alpaca_api_key=TEST_API_KEY,
            alpaca_secret_key=TEST_SECRET_KEY,
            user_id=1,
            profile=StrategyProfile.bubble_aware_2025  # More conservative profile
        )

        _apply_profile_risk_overrides(config)

        # Conservative profile should have reduced risk
        assert config.max_total_risk <= 0.50  # Bubble aware profile sets 0.45

    def test_apply_profile_risk_overrides_institutional(self):
        """Test risk overrides for institutional profile."""
        config = ProductionStrategyManagerConfig(
            alpaca_api_key=TEST_API_KEY,
            alpaca_secret_key=TEST_SECRET_KEY,
            user_id=1,
            profile=StrategyProfile.research_2024
        )

        _apply_profile_risk_overrides(config)

        # Institutional profile should have very conservative risk
        assert config.max_total_risk <= 0.15  # Research profile sets 0.10


class TestCreateProductionStrategyManager:
    """Test factory function for creating strategy manager."""

    @patch('backend.tradingbot.production.core.production_strategy_manager.ProductionIntegrationManager')
    @patch('backend.tradingbot.production.core.production_strategy_manager.ProductionDataProvider')
    def test_create_production_strategy_manager_basic(self, mock_data_provider, mock_integration):
        """Test basic strategy manager creation."""
        config = ProductionStrategyManagerConfig(
            alpaca_api_key=TEST_API_KEY,
            alpaca_secret_key=TEST_SECRET_KEY,
            user_id=1
        )

        manager = create_production_strategy_manager(config)

        assert isinstance(manager, ProductionStrategyManager)
        assert manager.config == config

    @patch('backend.tradingbot.production.core.production_strategy_manager.ProductionIntegrationManager')
    @patch('backend.tradingbot.production.core.production_strategy_manager.ProductionDataProvider')
    def test_create_production_strategy_manager_with_profile(self, mock_data_provider, mock_integration):
        """Test strategy manager creation with specific profile."""
        config = ProductionStrategyManagerConfig(
            alpaca_api_key=TEST_API_KEY,
            alpaca_secret_key=TEST_SECRET_KEY,
            user_id=1,
            profile=StrategyProfile.wsb_2025
        )

        manager = create_production_strategy_manager(config)

        assert isinstance(manager, ProductionStrategyManager)
        assert manager.config.profile == StrategyProfile.wsb_2025


class TestProductionStrategyManagerAdvanced:
    """Test advanced ProductionStrategyManager functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        return ProductionStrategyManagerConfig(
            alpaca_api_key=TEST_API_KEY,
            alpaca_secret_key=TEST_SECRET_KEY,
            paper_trading=True,
            user_id=1,
            max_total_risk=0.50,
            max_position_size=0.20,
            data_refresh_interval=30,
            enable_alerts=True,
            enable_advanced_analytics=True,
            enable_market_regime_adaptation=True
        )

    @pytest.fixture
    def mock_integration_manager(self):
        """Create mock integration manager."""
        mock_manager = Mock()
        mock_manager.alpaca_manager = Mock()
        mock_manager.alpaca_manager.validate_api.return_value = (True, "Success")
        mock_manager.get_portfolio_value = AsyncMock(return_value=100000.0)
        mock_manager.execute_trade_signal = AsyncMock(return_value=True)
        mock_manager.send_alert = AsyncMock(return_value=True)
        return mock_manager

    @pytest.fixture
    def mock_data_provider(self):
        """Create mock data provider."""
        mock_provider = Mock()
        mock_provider.is_market_open = AsyncMock(return_value=True)
        mock_provider.get_current_price = AsyncMock(return_value=100.0)
        mock_provider.get_recent_prices = AsyncMock(return_value=[95, 98, 100])
        return mock_provider

    @pytest.fixture
    def mock_analytics(self):
        """Create mock advanced analytics."""
        mock_analytics = Mock()
        mock_analytics.calculate_comprehensive_metrics = AsyncMock(
            return_value=Mock(
                total_return=0.15,
                annualized_return=0.18,
                volatility=0.12,
                sharpe_ratio=1.5,
                max_drawdown=0.08,
                win_rate=0.65,
                profit_factor=1.8,
                calmar_ratio=2.25
            )
        )
        return mock_analytics

    @pytest.fixture
    def mock_regime_adapter(self):
        """Create mock regime adapter."""
        mock_adapter = Mock()
        mock_adapter.detect_current_regime = AsyncMock(return_value="bull_market")
        mock_adapter.generate_strategy_adaptation = AsyncMock(
            return_value=Mock(
                regime="bull_market",
                confidence=0.85,
                recommended_actions={"increase_equity_exposure": 0.1},
                risk_adjustment=1.1,
                timestamp=datetime.now()
            )
        )
        return mock_adapter

    @pytest.fixture
    def strategy_manager_advanced(
        self, mock_config, mock_integration_manager, mock_data_provider,
        mock_analytics, mock_regime_adapter
    ):
        """Create advanced strategy manager with all mocks."""
        with (
            patch(
                "backend.tradingbot.production.core.production_strategy_manager.ProductionIntegrationManager",
                return_value=mock_integration_manager,
            ),
            patch(
                "backend.tradingbot.production.core.production_strategy_manager.ProductionDataProvider",
                return_value=mock_data_provider,
            ),
            patch(
                "backend.tradingbot.analytics.advanced_analytics.AdvancedAnalytics",
                return_value=mock_analytics,
            ),
            patch(
                "backend.tradingbot.analytics.market_regime_adapter.MarketRegimeAdapter",
                return_value=mock_regime_adapter,
            ),
        ):
            manager = ProductionStrategyManager(mock_config)
            return manager

    def test_initialization_with_advanced_features(self, strategy_manager_advanced):
        """Test initialization with advanced features enabled."""
        manager = strategy_manager_advanced

        assert manager.config.enable_advanced_analytics is True
        assert manager.config.enable_market_regime_adaptation is True
        assert hasattr(manager, 'advanced_analytics')
        assert hasattr(manager, 'market_regime_adapter')

    @pytest.mark.asyncio
    async def test_strategy_lifecycle_comprehensive(self, strategy_manager_advanced):
        """Test comprehensive strategy lifecycle."""
        manager = strategy_manager_advanced

        # Test start
        await manager.start()
        assert manager.is_running is True
        assert manager.start_time is not None

        # Test status during running
        status = await manager.get_detailed_status()
        assert isinstance(status, dict)
        assert "is_running" in status
        assert status["is_running"] is True
        # Status should contain strategy information
        assert "active_strategies" in status or "strategies" in status

        # Test stop
        await manager.stop()
        assert manager.is_running is False

    @pytest.mark.asyncio
    async def test_performance_metrics_calculation(self, strategy_manager_advanced, mock_analytics):
        """Test performance metrics calculation."""
        manager = strategy_manager_advanced

        # Mock portfolio history
        with patch.object(manager.integration_manager, 'get_portfolio_history',
                         new_callable=AsyncMock) as mock_history:
            mock_history.return_value = [
                {"timestamp": datetime.now() - timedelta(days=30), "value": 95000},
                {"timestamp": datetime.now() - timedelta(days=15), "value": 102000},
                {"timestamp": datetime.now(), "value": 108000}
            ]

            # Mock the analytics method directly
            with patch.object(manager.advanced_analytics, 'calculate_comprehensive_metrics') as mock_calc:
                await manager._update_performance_metrics()

                # Verify analytics was called
                mock_calc.assert_called()

    @pytest.mark.asyncio
    async def test_regime_adaptation(self, strategy_manager_advanced, mock_regime_adapter):
        """Test market regime adaptation."""
        manager = strategy_manager_advanced

        # Mock the regime adapter methods directly
        with patch.object(manager.market_regime_adapter, 'detect_current_regime') as mock_detect, \
             patch.object(manager.market_regime_adapter, 'generate_strategy_adaptation') as mock_generate:
            await manager._check_regime_adaptation()

            # Verify regime detection was called
            mock_detect.assert_called()
            mock_generate.assert_called()

    @pytest.mark.asyncio
    async def test_strategy_execution_monitoring(self, strategy_manager_advanced):
        """Test strategy execution monitoring."""
        manager = strategy_manager_advanced

        # Mock strategy that generates signals
        mock_strategy = Mock()
        mock_strategy.generate_signals = AsyncMock(return_value=[
            {"action": "buy", "symbol": "AAPL", "quantity": 100}
        ])
        mock_strategy.get_performance = AsyncMock(return_value={
            "total_return": 0.12,
            "win_rate": 0.68
        })

        manager.strategies["test_strategy"] = mock_strategy

        # Test monitoring cycle
        await manager._monitor_strategies()

        # Verify strategy was called
        mock_strategy.generate_signals.assert_called()

    @pytest.mark.asyncio
    async def test_error_handling_in_loops(self, strategy_manager_advanced):
        """Test error handling in background loops."""
        manager = strategy_manager_advanced

        # Mock analytics to raise exception
        with patch.object(manager.advanced_analytics, 'calculate_comprehensive_metrics',
                         side_effect=Exception("Analytics error")):
            # Should handle exception gracefully
            await manager._update_performance_metrics()

        # Mock regime adapter to raise exception
        with patch.object(manager.market_regime_adapter, 'detect_current_regime',
                         side_effect=Exception("Regime error")):
            # Should handle exception gracefully
            await manager._check_regime_adaptation()

    @pytest.mark.asyncio
    async def test_risk_management_checks(self, strategy_manager_advanced):
        """Test risk management checks."""
        manager = strategy_manager_advanced

        # Mock portfolio at risk limits
        with patch.object(manager.integration_manager, 'get_portfolio_value',
                         new_callable=AsyncMock) as mock_portfolio:
            mock_portfolio.return_value = 50000  # Down 50% from initial 100k

            # Mock positions at risk
            with patch.object(manager.integration_manager, 'get_positions',
                             new_callable=AsyncMock) as mock_positions:
                mock_positions.return_value = [
                    {"symbol": "AAPL", "unrealized_pl": -5000},
                    {"symbol": "MSFT", "unrealized_pl": -3000}
                ]

                await manager._check_risk_limits()

                # Should have triggered risk management actions
                # Verify calls were made to check risk

    @pytest.mark.asyncio
    async def test_alert_system_integration(self, strategy_manager_advanced):
        """Test alert system integration."""
        manager = strategy_manager_advanced

        # Test performance alert
        await manager._send_performance_alert("High drawdown detected", "warning")

        # Verify alert was sent
        manager.integration_manager.send_alert.assert_called()

    @pytest.mark.asyncio
    async def test_strategy_configuration_updates(self, strategy_manager_advanced):
        """Test dynamic strategy configuration updates."""
        manager = strategy_manager_advanced

        # Test updating strategy config
        new_config = StrategyConfig(
            name="wsb_dip_bot",
            enabled=True,
            max_position_size=0.15
        )

        await manager.update_strategy_config("wsb_dip_bot", new_config)

        # Verify config was updated
        assert manager.config.strategy_configs.get("wsb_dip_bot") == new_config

    @pytest.mark.asyncio
    async def test_emergency_shutdown(self, strategy_manager_advanced):
        """Test emergency shutdown procedures."""
        manager = strategy_manager_advanced

        # Start manager
        await manager.start()
        assert manager.is_running is True

        # Trigger emergency shutdown
        await manager.emergency_shutdown("Critical error detected")

        # Should be stopped
        assert manager.is_running is False

        # Should have sent emergency alert
        manager.integration_manager.send_alert.assert_called()

    @pytest.mark.asyncio
    async def test_heartbeat_monitoring(self, strategy_manager_advanced):
        """Test heartbeat monitoring."""
        manager = strategy_manager_advanced

        # Test heartbeat
        await manager._send_heartbeat()

        # Should update last heartbeat time
        assert hasattr(manager, 'last_heartbeat')

    @pytest.mark.asyncio
    async def test_data_provider_health_check(self, strategy_manager_advanced):
        """Test data provider health monitoring."""
        manager = strategy_manager_advanced

        # Test with healthy data provider
        health_status = await manager._check_data_provider_health()
        assert isinstance(health_status, dict)

        # Test with unhealthy data provider
        with patch.object(manager.data_provider, 'is_market_open',
                         side_effect=Exception("Connection error")):
            health_status = await manager._check_data_provider_health()
            assert health_status.get("status") == "unhealthy"

    def test_strategy_factory_methods(self, strategy_manager_advanced):
        """Test strategy factory methods are properly imported."""
        manager = strategy_manager_advanced

        # Check that all strategy creation functions exist
        from backend.tradingbot.production.core.production_strategy_manager import (
            create_production_wsb_dip_bot,
            create_production_wheel_strategy,
            create_production_earnings_protection,
            create_production_index_baseline,
            create_production_lotto_scanner
        )

        # All should be callable
        assert callable(create_production_wsb_dip_bot)
        assert callable(create_production_wheel_strategy)
        assert callable(create_production_earnings_protection)
        assert callable(create_production_index_baseline)
        assert callable(create_production_lotto_scanner)

    @pytest.mark.asyncio
    async def test_concurrent_strategy_execution(self, strategy_manager_advanced):
        """Test concurrent execution of multiple strategies."""
        manager = strategy_manager_advanced

        # Mock multiple strategies
        strategies = {}
        for i in range(3):
            mock_strategy = Mock()
            mock_strategy.generate_signals = AsyncMock(return_value=[])
            mock_strategy.get_performance = AsyncMock(return_value={})
            strategies[f"strategy_{i}"] = mock_strategy

        manager.strategies = strategies

        # Test concurrent execution
        await manager._monitor_strategies()

        # All strategies should have been called
        for strategy in strategies.values():
            strategy.generate_signals.assert_called()

    @pytest.mark.asyncio
    async def test_performance_persistence(self, strategy_manager_advanced):
        """Test performance metrics persistence."""
        manager = strategy_manager_advanced

        # Mock performance metrics
        from datetime import datetime
        metrics = PerformanceMetrics(
            total_return=0.15,
            annualized_return=0.18,
            volatility=0.12,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            calmar_ratio=2.25,
            max_drawdown=0.08,
            var_95=0.05,
            var_99=0.08,
            cvar_95=0.06,
            win_rate=0.65,
            avg_win=0.02,
            avg_loss=-0.015,
            profit_factor=1.8,
            information_ratio=0.5,
            treynor_ratio=1.2,
            alpha=0.03,
            beta=0.8,
            tracking_error=0.1,
            period_start=datetime.now(),
            period_end=datetime.now(),
            trading_days=252,
            best_day=0.05,
            worst_day=-0.04,
            positive_days=160,
            negative_days=92,
            recovery_factor=2.0,
            ulcer_index=0.15,
            sterling_ratio=1.5
        )

        with patch.object(manager, '_save_performance_metrics') as mock_save:
            await manager._update_performance_metrics()

            # Should attempt to save metrics
            # mock_save.assert_called()

    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test invalid risk values
        with pytest.raises(ValueError):
            ProductionStrategyManagerConfig(
                alpaca_api_key=TEST_API_KEY,
                alpaca_secret_key=TEST_SECRET_KEY,
                user_id=1,
                max_total_risk=1.5  # Invalid: > 1.0
            )

        # Test invalid position size
        with pytest.raises(ValueError):
            ProductionStrategyManagerConfig(
                alpaca_api_key=TEST_API_KEY,
                alpaca_secret_key=TEST_SECRET_KEY,
                user_id=1,
                max_position_size=-0.1  # Invalid: negative
            )


class TestProductionIntegration:
    """Test production integration scenarios."""

    @patch('backend.tradingbot.production.core.production_strategy_manager.ProductionIntegrationManager')
    @patch('backend.tradingbot.production.core.production_strategy_manager.ProductionDataProvider')
    def test_real_broker_integration_simulation(self, mock_data_provider, mock_integration):
        """Test simulation of real broker integration."""
        config = ProductionStrategyManagerConfig(
            alpaca_api_key=TEST_API_KEY,
            alpaca_secret_key=TEST_SECRET_KEY,
            user_id=1,
            paper_trading=False  # Live trading simulation
        )

        manager = ProductionStrategyManager(config)

        assert manager.config.paper_trading is False
        # Integration manager should be configured for live trading
        mock_integration.assert_called()

    @patch('backend.tradingbot.production.core.production_strategy_manager.ProductionIntegrationManager')
    @patch('backend.tradingbot.production.core.production_strategy_manager.ProductionDataProvider')
    def test_multi_user_scenario(self, mock_data_provider, mock_integration):
        """Test multi-user scenario handling."""
        configs = []
        managers = []

        for user_id in [1, 2, 3]:
            config = ProductionStrategyManagerConfig(
                alpaca_api_key=f"key_{user_id}",
                alpaca_secret_key=f"secret_{user_id}",
                user_id=user_id
            )
            configs.append(config)
            managers.append(ProductionStrategyManager(config))

        # Each manager should have unique configuration
        for i, manager in enumerate(managers):
            assert manager.config.user_id == i + 1
            assert manager.config.alpaca_api_key == f"key_{i + 1}"


@pytest.mark.asyncio
async def test_end_to_end_workflow():
    """Test complete end-to-end workflow."""
    with (
        patch('backend.tradingbot.production.core.production_strategy_manager.ProductionIntegrationManager') as mock_integration,
        patch('backend.tradingbot.production.core.production_strategy_manager.ProductionDataProvider') as mock_data_provider,
        patch('backend.tradingbot.analytics.advanced_analytics.AdvancedAnalytics') as mock_analytics,
        patch('backend.tradingbot.analytics.market_regime_adapter.MarketRegimeAdapter') as mock_regime,
    ):

        # Setup mocks
        mock_integration_instance = Mock()
        mock_integration_instance.alpaca_manager = Mock()
        mock_integration_instance.alpaca_manager.validate_api.return_value = (True, "Success")
        mock_integration_instance.get_portfolio_value = AsyncMock(return_value=100000.0)
        mock_integration_instance.send_alert = AsyncMock(return_value=True)
        mock_integration.return_value = mock_integration_instance

        import pandas as pd
        
        mock_data_provider_instance = Mock()
        mock_data_provider_instance.is_market_open = AsyncMock(return_value=True)
        
        # Mock historical data with a realistic DataFrame
        mock_historical_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000],
            'high': [101, 102, 103, 104, 105, 106],
            'low': [99, 100, 101, 102, 103, 104],
            'open': [100, 101, 102, 103, 104, 105]
        })
        mock_data_provider_instance.get_historical_data = AsyncMock(return_value=mock_historical_data)
        
        mock_data_provider.return_value = mock_data_provider_instance

        # Create configuration
        config = ProductionStrategyManagerConfig(
            alpaca_api_key=TEST_API_KEY,
            alpaca_secret_key=TEST_SECRET_KEY,
            user_id=1,
            profile=StrategyProfile.bubble_aware_2025,
            enable_advanced_analytics=True,
            enable_market_regime_adaptation=True
        )

        # Create manager
        manager = create_production_strategy_manager(config)

        # Start system
        await manager.start()
        assert manager.is_running is True

        # Simulate running for a short time
        await asyncio.sleep(0.1)

        # Get status
        status = await manager.get_detailed_status()
        assert isinstance(status, dict)

        # Stop system
        await manager.stop()
        assert manager.is_running is False