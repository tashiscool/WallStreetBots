"""Test shadow execution client for canary testing."""
import pytest
from unittest.mock import Mock, patch

from backend.tradingbot.execution.shadow_client import ShadowExecutionClient, CanaryExecutionClient
from backend.tradingbot.execution.interfaces import ExecutionClient, OrderRequest, OrderAck


class TestShadowExecutionClient:
    """Test shadow execution client."""

    def test_shadow_client_initialization(self):
        """Test shadow client initializes correctly."""
        real_client = Mock(spec=ExecutionClient)
        shadow_client = ShadowExecutionClient(real=real_client, log_all_orders=True)

        assert shadow_client.real == real_client
        assert shadow_client.log_all_orders is True
        assert len(shadow_client.order_log) == 0

    def test_validate_connection_delegates_to_real(self):
        """Test validate_connection delegates to real client."""
        real_client = Mock(spec=ExecutionClient)
        real_client.validate_connection.return_value = True

        shadow_client = ShadowExecutionClient(real=real_client)
        result = shadow_client.validate_connection()

        assert result is True
        real_client.validate_connection.assert_called_once()

    def test_place_order_logs_but_does_not_submit(self):
        """Test place_order logs order but doesn't submit to real client."""
        real_client = Mock(spec=ExecutionClient)
        shadow_client = ShadowExecutionClient(real=real_client, log_all_orders=True)

        order_request = OrderRequest(
            client_order_id="test_order_123",
            symbol="AAPL",
            qty=100,
            side="buy",
            type="market"
        )

        with patch.object(shadow_client, '_get_timestamp', return_value="2024-01-15T10:30:00Z"):
            result = shadow_client.place_order(order_request)

        # Should return mock acknowledgment
        assert isinstance(result, OrderAck)
        assert result.client_order_id == "test_order_123"
        assert result.broker_order_id is None
        assert result.accepted is True
        assert result.reason == "shadow_mode"

        # Should not call real client
        real_client.place_order.assert_not_called()

        # Should log the order
        assert len(shadow_client.order_log) == 1
        logged_order = shadow_client.order_log[0]
        assert logged_order["client_order_id"] == "test_order_123"
        assert logged_order["symbol"] == "AAPL"
        assert logged_order["side"] == "buy"
        assert logged_order["quantity"] == 100

    def test_get_order_returns_empty_dict(self):
        """Test get_order returns empty dict in shadow mode."""
        real_client = Mock(spec=ExecutionClient)
        shadow_client = ShadowExecutionClient(real=real_client)

        result = shadow_client.get_order("broker_order_123")

        assert result == {}
        real_client.get_order.assert_not_called()

    def test_list_open_orders_returns_empty_list(self):
        """Test list_open_orders returns empty list in shadow mode."""
        real_client = Mock(spec=ExecutionClient)
        shadow_client = ShadowExecutionClient(real=real_client)

        result = shadow_client.list_open_orders()

        assert result == []
        real_client.list_open_orders.assert_not_called()

    def test_cancel_order_returns_true(self):
        """Test cancel_order returns True in shadow mode."""
        real_client = Mock(spec=ExecutionClient)
        shadow_client = ShadowExecutionClient(real=real_client)

        result = shadow_client.cancel_order("broker_order_123")

        assert result is True
        real_client.cancel_order.assert_not_called()

    def test_reconcile_returns_none(self):
        """Test reconcile returns None in shadow mode."""
        real_client = Mock(spec=ExecutionClient)
        shadow_client = ShadowExecutionClient(real=real_client)

        result = shadow_client.reconcile("client_order_123")

        assert result is None
        real_client.reconcile.assert_not_called()

    def test_get_order_log(self):
        """Test getting order log."""
        real_client = Mock(spec=ExecutionClient)
        shadow_client = ShadowExecutionClient(real=real_client)

        # Place a test order
        order_request = OrderRequest("test_123", "MSFT", 50, "sell", "limit", limit_price=300.0)
        shadow_client.place_order(order_request)

        order_log = shadow_client.get_order_log()

        assert len(order_log) == 1
        assert order_log[0]["client_order_id"] == "test_123"
        assert order_log[0]["symbol"] == "MSFT"

        # Should return a copy, not the original
        order_log.clear()
        assert len(shadow_client.order_log) == 1

    def test_clear_order_log(self):
        """Test clearing order log."""
        real_client = Mock(spec=ExecutionClient)
        shadow_client = ShadowExecutionClient(real=real_client)

        # Place a test order
        order_request = OrderRequest("test_456", "GOOGL", 10, "buy", "market")
        shadow_client.place_order(order_request)

        assert len(shadow_client.order_log) == 1

        shadow_client.clear_order_log()

        assert len(shadow_client.order_log) == 0

    def test_get_order_count(self):
        """Test getting order count."""
        real_client = Mock(spec=ExecutionClient)
        shadow_client = ShadowExecutionClient(real=real_client)

        assert shadow_client.get_order_count() == 0

        # Place orders
        order1 = OrderRequest("order1", "AAPL", 100, "buy", "market")
        order2 = OrderRequest("order2", "MSFT", 50, "sell", "limit", limit_price=300.0)

        shadow_client.place_order(order1)
        shadow_client.place_order(order2)

        assert shadow_client.get_order_count() == 2

    def test_status(self):
        """Test getting shadow client status."""
        real_client = Mock(spec=ExecutionClient)
        real_client.validate_connection.return_value = True

        shadow_client = ShadowExecutionClient(real=real_client, log_all_orders=False)

        status = shadow_client.status()

        assert status["mode"] == "shadow"
        assert status["real_client_connected"] is True
        assert status["orders_logged"] == 0
        assert status["log_all_orders"] is False

    def test_shadow_without_logging(self):
        """Test shadow client with logging disabled."""
        real_client = Mock(spec=ExecutionClient)
        shadow_client = ShadowExecutionClient(real=real_client, log_all_orders=False)

        order_request = OrderRequest("no_log_test", "TSLA", 25, "buy", "market")

        with patch('backend.tradingbot.execution.shadow_client.log') as mock_log:
            shadow_client.place_order(order_request)

            # Should not log info message when log_all_orders is False
            mock_log.info.assert_not_called()

        # Should still record in order_log
        assert len(shadow_client.order_log) == 1


class TestCanaryExecutionClient:
    """Test canary execution client."""

    def test_canary_client_initialization(self):
        """Test canary client initializes correctly."""
        real_client = Mock(spec=ExecutionClient)
        canary_client = CanaryExecutionClient(
            real=real_client,
            canary_allocation_pct=0.05,
            max_daily_orders=5
        )

        assert canary_client.real == real_client
        assert canary_client.canary_allocation_pct == 0.05
        assert canary_client.max_daily_orders == 5
        assert canary_client.daily_order_count == 0

    def test_canary_daily_limit_enforcement(self):
        """Test canary client enforces daily order limits."""
        real_client = Mock(spec=ExecutionClient)
        canary_client = CanaryExecutionClient(
            real=real_client,
            max_daily_orders=2
        )

        order1 = OrderRequest("canary1", "AAPL", 100, "buy", "market")
        order2 = OrderRequest("canary2", "MSFT", 50, "sell", "limit", limit_price=300.0)
        order3 = OrderRequest("canary3", "GOOGL", 10, "buy", "market")

        # First two orders should succeed
        with patch.object(canary_client, '_get_current_date', return_value="2024-01-15"):
            result1 = canary_client.place_order(order1)
            result2 = canary_client.place_order(order2)

            assert result1.accepted is True
            assert result2.accepted is True
            assert canary_client.daily_order_count == 2

            # Third order should be rejected due to daily limit
            result3 = canary_client.place_order(order3)

            assert result3.accepted is False
            assert result3.reason == "canary_daily_limit_exceeded"
            assert canary_client.daily_order_count == 2  # Should not increment

    def test_canary_daily_reset(self):
        """Test canary client resets daily count on new day."""
        real_client = Mock(spec=ExecutionClient)
        canary_client = CanaryExecutionClient(
            real=real_client,
            max_daily_orders=1
        )

        order1 = OrderRequest("day1_order", "AAPL", 100, "buy", "market")
        order2 = OrderRequest("day2_order", "MSFT", 50, "sell", "limit", limit_price=300.0)

        # Place order on day 1
        with patch.object(canary_client, '_get_current_date', return_value="2024-01-15"):
            result1 = canary_client.place_order(order1)
            assert result1.accepted is True
            assert canary_client.daily_order_count == 1

        # Place order on day 2 - should reset counter
        with patch.object(canary_client, '_get_current_date', return_value="2024-01-16"):
            result2 = canary_client.place_order(order2)
            assert result2.accepted is True
            assert canary_client.daily_order_count == 1  # Reset and incremented
            assert canary_client.last_reset_date == "2024-01-16"

    def test_canary_status(self):
        """Test canary client status includes canary-specific info."""
        real_client = Mock(spec=ExecutionClient)
        real_client.validate_connection.return_value = True

        canary_client = CanaryExecutionClient(
            real=real_client,
            canary_allocation_pct=0.1,
            max_daily_orders=10
        )

        status = canary_client.status()

        assert status["mode"] == "shadow"  # Inherits from shadow
        assert status["canary_allocation_pct"] == 0.1
        assert status["max_daily_orders"] == 10
        assert status["daily_order_count"] == 0
        assert "last_reset_date" in status

    def test_shadow_client_comprehensive_edge_cases(self):
        """Test shadow client with comprehensive edge cases and stress scenarios."""
        real_client = Mock(spec=ExecutionClient)
        shadow_client = ShadowExecutionClient(real=real_client, log_all_orders=True)

        # Test invalid order requests
        invalid_order = OrderRequest("", "AAPL", 100, "buy", "market")  # Empty client order ID
        result = shadow_client.place_order(invalid_order)
        assert result.accepted is True  # Shadow mode accepts everything
        assert result.reason == "shadow_mode"

        # Test very large order
        large_order = OrderRequest("large_order", "AAPL", 1000000, "buy", "market")
        result = shadow_client.place_order(large_order)
        assert result.accepted is True

        # Test order with special characters
        special_order = OrderRequest(
            "order@#$%^&*()_+-={}[]|\\:;\"'<>?,./'",
            "SYMBOL-WITH.CHARS_123",
            50,
            "sell",
            "limit",
            limit_price=999.99
        )
        result = shadow_client.place_order(special_order)
        assert result.accepted is True

        # Test rapid-fire orders (performance)
        import time
        start_time = time.time()
        for i in range(1000):
            order = OrderRequest(f"rapid_{i}", "SPY", 100, "buy", "market")
            shadow_client.place_order(order)
        end_time = time.time()

        assert end_time - start_time < 2.0  # Should handle 1000 orders in under 2 seconds
        assert shadow_client.get_order_count() >= 1003  # Previous orders + 1000 new ones

        # Test order log memory management
        initial_count = shadow_client.get_order_count()
        shadow_client.clear_order_log()
        assert shadow_client.get_order_count() == 0

        # Test concurrent access simulation (though not truly concurrent)
        orders = []
        for i in range(50):
            order = OrderRequest(f"concurrent_{i}", "QQQ", 10 + i, "buy", "market")
            result = shadow_client.place_order(order)
            orders.append(result)

        # Verify all orders were processed
        assert len(orders) == 50
        assert all(o.accepted for o in orders)
        assert shadow_client.get_order_count() == 50

        # Test order log integrity
        order_log = shadow_client.get_order_log()
        assert len(order_log) == 50
        symbols = [entry["symbol"] for entry in order_log]
        assert all(symbol == "QQQ" for symbol in symbols)

    def test_canary_client_advanced_scenarios(self):
        """Test canary client with advanced allocation and risk scenarios."""
        real_client = Mock(spec=ExecutionClient)
        canary_client = CanaryExecutionClient(
            real=real_client,
            canary_allocation_pct=0.05,  # Very conservative 5%
            max_daily_orders=3
        )

        # Test edge case allocation percentages (no validation in current implementation)
        edge_client = CanaryExecutionClient(real_client, canary_allocation_pct=1.5)
        assert edge_client.canary_allocation_pct == 1.5  # Implementation accepts any value

        # Test negative allocation percentage (no validation in current implementation)
        negative_client = CanaryExecutionClient(real_client, canary_allocation_pct=-0.1)
        assert negative_client.canary_allocation_pct == -0.1  # Implementation accepts negative values

        # Test zero max daily orders (no validation in current implementation)
        zero_orders_client = CanaryExecutionClient(real_client, max_daily_orders=0)
        assert zero_orders_client.max_daily_orders == 0  # Implementation accepts zero

        # Test negative max daily orders (no validation in current implementation)
        negative_orders_client = CanaryExecutionClient(real_client, max_daily_orders=-5)
        assert negative_orders_client.max_daily_orders == -5  # Implementation accepts negative values

        # Test boundary conditions for daily limits
        order1 = OrderRequest("canary_boundary_1", "AAPL", 100, "buy", "market")
        order2 = OrderRequest("canary_boundary_2", "MSFT", 50, "sell", "limit", limit_price=300.0)
        order3 = OrderRequest("canary_boundary_3", "GOOGL", 10, "buy", "market")
        order4 = OrderRequest("canary_boundary_4", "TSLA", 25, "sell", "market")

        with patch.object(canary_client, '_get_current_date', return_value="2024-03-15"):
            # First 3 orders should succeed
            result1 = canary_client.place_order(order1)
            result2 = canary_client.place_order(order2)
            result3 = canary_client.place_order(order3)

            assert result1.accepted is True
            assert result2.accepted is True
            assert result3.accepted is True
            assert canary_client.daily_order_count == 3

            # 4th order should be rejected
            result4 = canary_client.place_order(order4)
            assert result4.accepted is False
            assert result4.reason == "canary_daily_limit_exceeded"
            assert canary_client.daily_order_count == 3  # Should not increment

        # Test date rollover with microsecond precision
        with patch.object(canary_client, '_get_current_date', return_value="2024-03-16"):
            # Should reset and allow new orders
            order5 = OrderRequest("canary_new_day", "NVDA", 15, "buy", "market")
            result5 = canary_client.place_order(order5)

            assert result5.accepted is True
            assert canary_client.daily_order_count == 1
            assert canary_client.last_reset_date == "2024-03-16"

        # Test extremely high allocation percentage
        high_allocation_client = CanaryExecutionClient(
            real=real_client,
            canary_allocation_pct=0.99,  # 99% allocation
            max_daily_orders=1000
        )

        status = high_allocation_client.status()
        assert status["canary_allocation_pct"] == 0.99
        assert status["max_daily_orders"] == 1000

        # Test zero allocation (edge case)
        zero_allocation_client = CanaryExecutionClient(
            real=real_client,
            canary_allocation_pct=0.0,  # 0% allocation
            max_daily_orders=1
        )

        zero_order = OrderRequest("zero_allocation", "SPY", 100, "buy", "market")
        with patch.object(zero_allocation_client, '_get_current_date', return_value="2024-03-17"):
            result = zero_allocation_client.place_order(zero_order)
            assert result.accepted is True  # Should still work, just logs differently

    def test_shadow_error_handling_robustness(self):
        """Test shadow client error handling and robustness."""
        real_client = Mock(spec=ExecutionClient)
        shadow_client = ShadowExecutionClient(real=real_client)

        # Test when real client validation fails
        real_client.validate_connection.side_effect = Exception("Network error")
        with pytest.raises(Exception, match="Network error"):
            shadow_client.validate_connection()

        # Reset real client
        real_client.validate_connection.side_effect = None
        real_client.validate_connection.return_value = False

        # Test with invalid timestamp generation
        with patch.object(shadow_client, '_get_timestamp', side_effect=Exception("Time error")):
            order = OrderRequest("error_order", "AAPL", 100, "buy", "market")
            with pytest.raises(Exception, match="Time error"):
                shadow_client.place_order(order)

        # Test log overflow scenario
        with patch.object(shadow_client, '_get_timestamp', return_value="2024-01-01T00:00:00Z"):
            # Add many orders to test memory usage
            for i in range(10000):
                order = OrderRequest(f"overflow_order_{i}", "SPY", 1, "buy", "market")
                shadow_client.place_order(order)

            assert shadow_client.get_order_count() == 10000
            order_log = shadow_client.get_order_log()
            assert len(order_log) == 10000

            # Verify log entries are properly formatted
            first_entry = order_log[0]
            assert "timestamp" in first_entry
            assert "action" in first_entry
            assert "client_order_id" in first_entry
            assert first_entry["timestamp"] == "2024-01-01T00:00:00Z"