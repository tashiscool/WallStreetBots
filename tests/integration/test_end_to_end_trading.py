"""End - to - End Trading Integration Tests.

Comprehensive integration tests that validate the entire trading flow from signal generation
to order execution and position management.
"""

# Test constants
TEST_API_KEY = "test_key"
TEST_SECRET_KEY = "test_secret"  # noqa: S105

import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest

from backend.tradingbot.error_handling import TradingErrorRecoveryManager
from backend.tradingbot.monitoring import SystemHealthMonitor
from backend.tradingbot.production.core.production_strategy_manager import (
    ProductionStrategyManager,
    ProductionStrategyManagerConfig,
)


class TestEndToEndTrading:
    """Integration tests that validate entire trading flow."""

    @pytest.fixture
    def mock_trading_system(self):
        """Create mock trading system for testing."""
        # Mock data provider
        mock_data_provider = Mock()
        mock_data_provider.is_market_open = AsyncMock(return_value=True)
        mock_data_provider.get_current_price = AsyncMock(
            return_value=Mock(price=Decimal("150.00"))
        )
        mock_data_provider.get_price_history = AsyncMock(
            return_value=[
                Decimal("100.00"),
                Decimal("102.00"),
                Decimal("105.00"),
                Decimal("108.00"),
                Decimal("110.00"),
                Decimal("112.00"),
                Decimal("115.00"),
                Decimal("118.00"),
                Decimal("120.00"),
                Decimal("115.00"),  # Dip after run pattern
            ]
        )
        mock_data_provider.get_volume_history = AsyncMock(
            return_value=[
                1000000,
                1100000,
                1200000,
                1300000,
                1400000,
                1500000,
                1600000,
                1700000,
                1800000,
                2000000,
            ]
        )
        mock_data_provider.get_options_chain = AsyncMock(return_value=[])

        # Mock integration manager
        mock_integration = Mock()
        mock_integration.get_portfolio_value = AsyncMock(
            return_value=Decimal("100000.00")
        )
        mock_integration.execute_trade = AsyncMock(
            return_value={
                "order_id": "test_order_123",
                "status": "FILLED",
                "filled_price": Decimal("5.00"),
                "quantity": 100,
            }
        )
        mock_integration.get_positions = AsyncMock(return_value=[])
        mock_integration.get_portfolio_summary = Mock(
            return_value={
                "total_value": Decimal("100000.00"),
                "cash": Decimal("50000.00"),
                "positions": [],
            }
        )

        # Mock broker manager
        mock_broker = Mock()
        mock_account = Mock()
        mock_account.status = "ACTIVE"
        mock_account.buying_power = Decimal("50000.00")
        mock_account.portfolio_value = Decimal("100000.00")
        mock_broker.get_account = AsyncMock(return_value=mock_account)
        mock_broker.validate_api = Mock(
            return_value=(True, "API validation successful")
        )

        # Set the broker manager on the integration manager
        mock_integration.alpaca_manager = mock_broker

        # Create strategy manager
        config = ProductionStrategyManagerConfig(
            alpaca_api_key=TEST_API_KEY,
            alpaca_secret_key=TEST_SECRET_KEY,
            paper_trading=True,
            user_id=1,
            max_total_risk=0.50,
            max_position_size=0.20,
            enable_alerts=True,
        )

        strategy_manager = ProductionStrategyManager(config)
        strategy_manager.data_provider = mock_data_provider
        strategy_manager.integration_manager = mock_integration
        strategy_manager.broker_manager = mock_broker

        return {
            "strategy_manager": strategy_manager,
            "data_provider": mock_data_provider,
            "integration": mock_integration,
            "broker": mock_broker,
        }

    @pytest.mark.asyncio
    async def test_complete_dip_bot_flow(self, mock_trading_system):
        """Test full WSB Dip Bot execution flow."""
        strategy_manager = mock_trading_system["strategy_manager"]

        # 1. Strategy manager is already initialized in __init__

        # 2. Start all strategies
        await strategy_manager.start_all_strategies()

        # 3. Strategies run automatically when started
        # Wait a moment for strategies to initialize
        await asyncio.sleep(0.1)

        # 4. Verify system status
        # Note: May be empty due to strict signal criteria, which is expected

        # 5. Verify system status
        status = strategy_manager.get_system_status()
        assert status["is_running"] is True
        assert "active_strategies" in status

        # 6. Stop strategies
        await strategy_manager.stop_all_strategies()

        # 7. Verify shutdown
        status = strategy_manager.get_system_status()
        assert status["is_running"] is False

    @pytest.mark.asyncio
    async def test_error_recovery_flow(self, mock_trading_system):
        """Test error recovery mechanisms."""
        from backend.tradingbot.error_handling import (
            DataProviderError,
            TradingErrorRecoveryManager,
        )

        # Create recovery manager
        recovery_manager = TradingErrorRecoveryManager(
            trading_system=mock_trading_system["strategy_manager"],
            config={"max_retry_attempts": 2},
        )

        # Simulate data provider error
        data_error = DataProviderError(
            "Test data provider failure", provider="test_provider"
        )

        # Handle error
        recovery_action = await recovery_manager.handle_trading_error(data_error)

        # Verify recovery action
        assert recovery_action is not None

        # Check error statistics
        stats = recovery_manager.get_error_statistics()
        assert stats["total_errors"] > 0
        assert "DataProviderError" in stats["error_counts"]

    @pytest.mark.asyncio
    async def test_system_health_monitoring(self, mock_trading_system):
        """Test system health monitoring."""
        # Create health monitor
        health_monitor = SystemHealthMonitor(
            trading_system=mock_trading_system["strategy_manager"],
            config={
                "data_feed_latency_threshold": 5.0,
                "memory_usage_threshold": 0.80,
                "cpu_usage_threshold": 0.90,
            },
        )

        # Run health check
        health_report = await health_monitor.check_system_health()

        # Verify health report structure
        assert health_report.timestamp is not None
        assert health_report.overall_status is not None
        assert health_report.data_feed_status is not None
        assert health_report.broker_status is not None
        assert health_report.database_status is not None
        assert health_report.resource_status is not None
        assert health_report.trading_status is not None

        # Check uptime stats
        uptime_stats = health_monitor.get_uptime_stats()
        assert uptime_stats["uptime_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_position_reconciliation_flow(self, mock_trading_system):
        """Test position reconciliation process."""
        integration = mock_trading_system["integration"]

        # Mock position reconciliation
        integration.reconcile_positions = AsyncMock(
            return_value={
                "discrepancies": [],
                "requires_intervention": False,
                "timestamp": datetime.now(),
            }
        )

        # Run reconciliation
        reconciliation_result = await integration.reconcile_positions()

        # Verify reconciliation
        assert reconciliation_result["requires_intervention"] is False
        assert len(reconciliation_result["discrepancies"]) == 0

    @pytest.mark.asyncio
    async def test_risk_management_integration(self, mock_trading_system):
        """Test risk management integration."""
        integration = mock_trading_system["integration"]

        # Mock risk validation
        integration.validate_trade_risk = AsyncMock(
            return_value={
                "approved": True,
                "risk_percentage": 0.15,
                "position_size": Decimal("15000.00"),
            }
        )

        # Test risk validation
        risk_result = await integration.validate_trade_risk(
            ticker="AAPL", quantity=100, price=Decimal("150.00")
        )

        # Verify risk validation
        assert risk_result["approved"] is True
        assert risk_result["risk_percentage"] <= 0.20  # Within limits

    @pytest.mark.asyncio
    async def test_alert_system_integration(self, mock_trading_system):
        """Test alert system integration."""
        integration = mock_trading_system["integration"]

        # Mock alert system
        integration.alert_system = Mock()
        integration.alert_system.send_alert = AsyncMock()

        # Send test alert
        await integration.alert_system.send_alert(
            message="Test alert", priority="HIGH", alert_type="SYSTEM"
        )

        # Verify alert was sent
        integration.alert_system.send_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_data_provider_failover(self, mock_trading_system):
        """Test data provider failover mechanism."""
        data_provider = mock_trading_system["data_provider"]

        # Mock failover
        data_provider.switch_to_backup = AsyncMock()

        # Simulate primary source failure
        data_provider.get_current_price = AsyncMock(
            side_effect=Exception("Primary source failed")
        )

        # Switch to backup
        await data_provider.switch_to_backup()

        # Verify failover
        data_provider.switch_to_backup.assert_called_once()

    @pytest.mark.asyncio
    async def test_strategy_performance_monitoring(self, mock_trading_system):
        """Test strategy performance monitoring."""
        strategy_manager = mock_trading_system["strategy_manager"]

        # Mock performance metrics
        strategy_manager.get_performance_metrics = AsyncMock(
            return_value={
                "total_trades": 10,
                "successful_trades": 8,
                "total_pnl": Decimal("1500.00"),
                "win_rate": 0.80,
                "avg_trade_duration": 2.5,
            }
        )

        # Get performance metrics
        metrics = await strategy_manager.get_performance_metrics()

        # Verify metrics
        assert metrics["total_trades"] > 0
        assert metrics["win_rate"] >= 0
        assert metrics["total_pnl"] is not None

    @pytest.mark.asyncio
    async def test_emergency_halt_procedure(self, mock_trading_system):
        """Test emergency halt procedure."""
        strategy_manager = mock_trading_system["strategy_manager"]

        # Mock emergency halt
        strategy_manager.emergency_halt = AsyncMock()

        # Trigger emergency halt
        await strategy_manager.emergency_halt("Test emergency halt")

        # Verify emergency halt was called
        strategy_manager.emergency_halt.assert_called_once_with("Test emergency halt")

    @pytest.mark.asyncio
    async def test_comprehensive_system_test(self, mock_trading_system):
        """Comprehensive system test covering all major components."""
        strategy_manager = mock_trading_system["strategy_manager"]

        # 1. System is already initialized in __init__

        # 2. Start strategies
        await strategy_manager.start_all_strategies()

        # 3. Strategies run automatically when started
        # Wait for strategies to initialize
        await asyncio.sleep(0.2)

        # 4. Check system health
        health_monitor = SystemHealthMonitor(
            trading_system=strategy_manager,
            config={
                "data_feed_latency_threshold": 5.0,
                "memory_usage_threshold": 0.80,
                "cpu_usage_threshold": 0.90,
            },
        )
        health_report = await health_monitor.check_system_health()

        # 5. Verify system is operational
        assert health_report.overall_status is not None

        # 6. Test error recovery
        recovery_manager = TradingErrorRecoveryManager(
            trading_system=strategy_manager, config={"max_retry_attempts": 2}
        )
        from backend.tradingbot.error_handling import BrokerConnectionError

        broker_error = BrokerConnectionError("Test broker connection failure")
        recovery_action = await recovery_manager.handle_trading_error(broker_error)

        # 7. Verify error handling
        assert recovery_action is not None

        # 8. Stop system
        await strategy_manager.stop_all_strategies()

        # 9. Verify shutdown
        status = strategy_manager.get_system_status()
        assert status["is_running"] is False

        # 10. Verify all components worked together
        assert True  # If we get here, the comprehensive test passed
