"""Test execution layer interfaces."""
import pytest
from backend.tradingbot.execution.interfaces import (
    OrderRequest,
    OrderAck,
    OrderFill,
    ExecutionClient
)


class TestOrderRequest:
    """Test order request data structure."""

    def test_market_order_creation(self):
        """Test creating a market order request."""
        order = OrderRequest(
            client_order_id="test_order_123",
            symbol="AAPL",
            qty=100,
            side="buy",
            type="market"
        )

        assert order.client_order_id == "test_order_123"
        assert order.symbol == "AAPL"
        assert order.qty == 100
        assert order.side == "buy"
        assert order.type == "market"
        assert order.limit_price is None
        assert order.time_in_force == "day"  # default

    def test_limit_order_creation(self):
        """Test creating a limit order request."""
        order = OrderRequest(
            client_order_id="limit_order_456",
            symbol="MSFT",
            qty=50,
            side="sell",
            type="limit",
            limit_price=300.50
        )

        assert order.symbol == "MSFT"
        assert order.side == "sell"
        assert order.qty == 50
        assert order.type == "limit"
        assert order.limit_price == 300.50

    def test_order_with_custom_time_in_force(self):
        """Test order with custom time in force."""
        order = OrderRequest(
            client_order_id="gtc_order_789",
            symbol="GOOGL",
            qty=10,
            side="buy",
            type="limit",
            limit_price=2500.00,
            time_in_force="gtc"
        )

        assert order.time_in_force == "gtc"

    def test_order_request_immutability(self):
        """Test that order request is immutable."""
        order = OrderRequest(
            client_order_id="immutable_test",
            symbol="TSLA",
            qty=25,
            side="buy",
            type="market"
        )

        # Should not be able to modify frozen dataclass
        with pytest.raises(AttributeError):
            order.qty = 50


class TestOrderAck:
    """Test order acknowledgment data structure."""

    def test_order_ack_accepted(self):
        """Test creating an accepted order ack."""
        ack = OrderAck(
            client_order_id="client_123",
            broker_order_id="broker_456",
            accepted=True
        )

        assert ack.client_order_id == "client_123"
        assert ack.broker_order_id == "broker_456"
        assert ack.accepted is True
        assert ack.reason is None

    def test_order_ack_rejected(self):
        """Test creating a rejected order ack."""
        ack = OrderAck(
            client_order_id="client_789",
            broker_order_id=None,
            accepted=False,
            reason="Insufficient buying power"
        )

        assert ack.client_order_id == "client_789"
        assert ack.broker_order_id is None
        assert ack.accepted is False
        assert ack.reason == "Insufficient buying power"

    def test_order_ack_immutability(self):
        """Test that order ack is immutable."""
        ack = OrderAck(
            client_order_id="immutable_test",
            broker_order_id="broker_123",
            accepted=True
        )

        # Should not be able to modify frozen dataclass
        with pytest.raises(AttributeError):
            ack.accepted = False


class TestOrderFill:
    """Test order fill data structure."""

    def test_order_fill_creation(self):
        """Test creating an order fill."""
        fill = OrderFill(
            broker_order_id="broker_999",
            symbol="NVDA",
            avg_price=450.25,
            filled_qty=15,
            status="filled"
        )

        assert fill.broker_order_id == "broker_999"
        assert fill.symbol == "NVDA"
        assert fill.avg_price == 450.25
        assert fill.filled_qty == 15
        assert fill.status == "filled"

    def test_partial_fill(self):
        """Test creating a partial fill."""
        fill = OrderFill(
            broker_order_id="broker_888",
            symbol="AAPL",
            avg_price=150.75,
            filled_qty=75,
            status="partially_filled"
        )

        assert fill.status == "partially_filled"
        assert fill.filled_qty == 75

    def test_fill_immutability(self):
        """Test that order fill is immutable."""
        fill = OrderFill(
            broker_order_id="immutable_test",
            symbol="MSFT",
            avg_price=300.00,
            filled_qty=50,
            status="filled"
        )

        # Should not be able to modify frozen dataclass
        with pytest.raises(AttributeError):
            fill.filled_qty = 100


class TestExecutionClient:
    """Test execution client interface."""

    def test_execution_client_is_abstract(self):
        """Test that ExecutionClient is abstract and methods raise NotImplementedError."""
        client = ExecutionClient()

        with pytest.raises(NotImplementedError):
            client.validate_connection()

        with pytest.raises(NotImplementedError):
            client.place_order(OrderRequest("test", "AAPL", 100, "buy", "market"))

        with pytest.raises(NotImplementedError):
            client.get_order("broker_123")

        with pytest.raises(NotImplementedError):
            client.list_open_orders()

        with pytest.raises(NotImplementedError):
            client.cancel_order("broker_123")

        with pytest.raises(NotImplementedError):
            client.reconcile("client_123")

    def test_execution_client_inheritance(self):
        """Test that ExecutionClient can be inherited."""

        class MockExecutionClient(ExecutionClient):
            def validate_connection(self) -> bool:
                return True

            def place_order(self, req: OrderRequest) -> OrderAck:
                return OrderAck(req.client_order_id, "mock_broker_id", True)

            def get_order(self, broker_order_id: str) -> dict:
                return {"status": "filled"}

            def list_open_orders(self) -> dict:
                return {}

            def cancel_order(self, broker_order_id: str) -> bool:
                return True

            def reconcile(self, client_order_id: str):
                return None

        # Should be able to instantiate and use
        mock_client = MockExecutionClient()
        assert mock_client.validate_connection() is True

        order_req = OrderRequest("test_123", "AAPL", 100, "buy", "market")
        ack = mock_client.place_order(order_req)
        assert ack.accepted is True
        assert ack.client_order_id == "test_123"