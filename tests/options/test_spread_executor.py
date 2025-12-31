"""
Tests for Spread Executor

Tests cover:
- OrderStatus and ExecutionStrategy enums
- LegOrder and SpreadOrder dataclasses
- ExecutionResult dataclass
- SpreadExecutor execution methods
"""

import pytest
from datetime import datetime, date, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch

from backend.tradingbot.options.spread_executor import (
    OrderStatus,
    ExecutionStrategy,
    LegOrder,
    SpreadOrder,
    ExecutionResult,
    SpreadExecutor,
)
from backend.tradingbot.options.exotic_spreads import (
    OptionSpread,
    SpreadLeg,
    SpreadType,
    LegType,
)


class TestOrderStatusEnum:
    """Tests for OrderStatus enum."""

    def test_enum_values(self):
        """Test all enum values exist."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.SUBMITTED.value == "submitted"
        assert OrderStatus.PARTIAL.value == "partial"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"
        assert OrderStatus.EXPIRED.value == "expired"
        assert OrderStatus.FAILED.value == "failed"


class TestExecutionStrategyEnum:
    """Tests for ExecutionStrategy enum."""

    def test_enum_values(self):
        """Test all enum values exist."""
        assert ExecutionStrategy.ATOMIC.value == "atomic"
        assert ExecutionStrategy.LEGGED_IN.value == "legged_in"
        assert ExecutionStrategy.WINGS_FIRST.value == "wings_first"


class TestLegOrder:
    """Tests for LegOrder dataclass."""

    def test_leg_order_creation(self):
        """Test creating a leg order."""
        expiry = date.today() + timedelta(days=30)
        leg = SpreadLeg(
            leg_type=LegType.LONG_CALL,
            strike=Decimal("150"),
            expiry=expiry,
            contracts=1,
        )
        leg_order = LegOrder(leg=leg)

        assert leg_order.status == OrderStatus.PENDING
        assert leg_order.order_id is None
        assert leg_order.filled_qty == 0

    def test_is_complete_pending(self):
        """Test is_complete returns False for pending."""
        expiry = date.today() + timedelta(days=30)
        leg = SpreadLeg(
            leg_type=LegType.LONG_CALL,
            strike=Decimal("150"),
            expiry=expiry,
            contracts=1,
        )
        leg_order = LegOrder(leg=leg)

        assert leg_order.is_complete is False

    def test_is_complete_filled(self):
        """Test is_complete returns True for filled."""
        expiry = date.today() + timedelta(days=30)
        leg = SpreadLeg(
            leg_type=LegType.LONG_CALL,
            strike=Decimal("150"),
            expiry=expiry,
            contracts=1,
        )
        leg_order = LegOrder(leg=leg, status=OrderStatus.FILLED)

        assert leg_order.is_complete is True

    def test_is_complete_cancelled(self):
        """Test is_complete returns True for cancelled."""
        expiry = date.today() + timedelta(days=30)
        leg = SpreadLeg(
            leg_type=LegType.LONG_CALL,
            strike=Decimal("150"),
            expiry=expiry,
            contracts=1,
        )
        leg_order = LegOrder(leg=leg, status=OrderStatus.CANCELLED)

        assert leg_order.is_complete is True


class TestSpreadOrder:
    """Tests for SpreadOrder dataclass."""

    @pytest.fixture
    def sample_spread(self):
        """Create a sample spread."""
        expiry = date.today() + timedelta(days=30)
        legs = [
            SpreadLeg(LegType.LONG_CALL, Decimal("150"), expiry, 1),
            SpreadLeg(LegType.SHORT_CALL, Decimal("160"), expiry, -1),
        ]
        return OptionSpread(
            spread_type=SpreadType.VERTICAL_CALL,
            ticker="AAPL",
            legs=legs,
        )

    def test_spread_order_creation(self, sample_spread):
        """Test creating a spread order."""
        spread_order = SpreadOrder(spread=sample_spread)

        assert spread_order.status == OrderStatus.PENDING
        assert spread_order.execution_strategy == ExecutionStrategy.ATOMIC

    def test_is_complete_pending(self, sample_spread):
        """Test is_complete returns False for pending."""
        spread_order = SpreadOrder(spread=sample_spread)
        assert spread_order.is_complete is False

    def test_is_complete_filled(self, sample_spread):
        """Test is_complete returns True for filled."""
        spread_order = SpreadOrder(spread=sample_spread, status=OrderStatus.FILLED)
        assert spread_order.is_complete is True

    def test_all_legs_filled_true(self, sample_spread):
        """Test all_legs_filled when all filled."""
        spread_order = SpreadOrder(spread=sample_spread)
        spread_order.leg_orders = [
            LegOrder(leg=spread_order.spread.legs[0], status=OrderStatus.FILLED),
            LegOrder(leg=spread_order.spread.legs[1], status=OrderStatus.FILLED),
        ]

        assert spread_order.all_legs_filled is True

    def test_all_legs_filled_false(self, sample_spread):
        """Test all_legs_filled when not all filled."""
        spread_order = SpreadOrder(spread=sample_spread)
        spread_order.leg_orders = [
            LegOrder(leg=spread_order.spread.legs[0], status=OrderStatus.FILLED),
            LegOrder(leg=spread_order.spread.legs[1], status=OrderStatus.PENDING),
        ]

        assert spread_order.all_legs_filled is False

    def test_some_legs_filled_true(self, sample_spread):
        """Test some_legs_filled when some filled."""
        spread_order = SpreadOrder(spread=sample_spread)
        spread_order.leg_orders = [
            LegOrder(leg=spread_order.spread.legs[0], status=OrderStatus.FILLED),
            LegOrder(leg=spread_order.spread.legs[1], status=OrderStatus.PENDING),
        ]

        assert spread_order.some_legs_filled is True

    def test_some_legs_filled_false(self, sample_spread):
        """Test some_legs_filled when none filled."""
        spread_order = SpreadOrder(spread=sample_spread)
        spread_order.leg_orders = [
            LegOrder(leg=spread_order.spread.legs[0], status=OrderStatus.PENDING),
            LegOrder(leg=spread_order.spread.legs[1], status=OrderStatus.PENDING),
        ]

        assert spread_order.some_legs_filled is False


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_result_success(self):
        """Test successful execution result."""
        spread = Mock()
        spread_order = SpreadOrder(spread=spread, status=OrderStatus.FILLED)

        result = ExecutionResult(
            success=True,
            spread_order=spread_order,
            execution_time_ms=150.0,
            filled_premium=Decimal("2.50"),
        )

        assert result.success is True
        assert result.execution_time_ms == 150.0
        assert result.filled_premium == Decimal("2.50")

    def test_result_failure(self):
        """Test failed execution result."""
        spread = Mock()
        spread_order = SpreadOrder(spread=spread, status=OrderStatus.FAILED)

        result = ExecutionResult(
            success=False,
            spread_order=spread_order,
            error_message="Broker rejected order",
        )

        assert result.success is False
        assert result.error_message == "Broker rejected order"


class TestSpreadExecutor:
    """Tests for SpreadExecutor."""

    @pytest.fixture
    def mock_broker(self):
        """Create a mock broker client."""
        broker = Mock()
        broker.get_account = AsyncMock(return_value={"equity": 100000})
        broker.submit_multi_leg_order = AsyncMock(return_value={"id": "order123"})
        broker.submit_option_order = AsyncMock(return_value={"id": "order456"})
        broker.submit_order = AsyncMock(return_value={"id": "order789"})
        broker.get_order = AsyncMock(return_value={"status": "filled", "filled_avg_price": 2.50})
        broker.cancel_order = AsyncMock(return_value=True)
        return broker

    @pytest.fixture
    def sample_spread(self):
        """Create a sample spread."""
        expiry = date.today() + timedelta(days=30)
        legs = [
            SpreadLeg(LegType.LONG_CALL, Decimal("150"), expiry, 1, premium=Decimal("5.00")),
            SpreadLeg(LegType.SHORT_CALL, Decimal("160"), expiry, -1, premium=Decimal("2.00")),
        ]
        return OptionSpread(
            spread_type=SpreadType.VERTICAL_CALL,
            ticker="AAPL",
            legs=legs,
        )

    def test_executor_creation(self, mock_broker):
        """Test executor creation."""
        executor = SpreadExecutor(mock_broker)
        assert executor.broker is mock_broker
        assert executor.max_slippage_pct == 2.0
        assert executor.auto_rollback is True

    def test_build_option_symbol(self, mock_broker):
        """Test building OCC option symbol."""
        executor = SpreadExecutor(mock_broker)
        symbol = executor._build_option_symbol(
            "AAPL",
            Decimal("150"),
            date(2024, 12, 20),
            "call",
        )

        # Format: TICKER YYMMDD C/P STRIKE
        assert symbol == "AAPL241220C00150000"

    def test_build_option_symbol_put(self, mock_broker):
        """Test building OCC option symbol for put."""
        executor = SpreadExecutor(mock_broker)
        symbol = executor._build_option_symbol(
            "AAPL",
            Decimal("145.50"),
            date(2024, 12, 20),
            "put",
        )

        assert symbol == "AAPL241220P00145500"

    @pytest.mark.asyncio
    async def test_validate_spread_valid(self, mock_broker, sample_spread):
        """Test spread validation for valid spread."""
        executor = SpreadExecutor(mock_broker)
        error = await executor._validate_spread(sample_spread)
        assert error is None

    @pytest.mark.asyncio
    async def test_validate_spread_too_few_legs(self, mock_broker):
        """Test spread validation rejects single leg."""
        executor = SpreadExecutor(mock_broker)
        spread = OptionSpread(
            spread_type=SpreadType.VERTICAL_CALL,
            ticker="AAPL",
            legs=[SpreadLeg(LegType.LONG_CALL, Decimal("150"), date.today(), 1)],
        )
        error = await executor._validate_spread(spread)
        assert error == "Spread must have at least 2 legs"

    @pytest.mark.asyncio
    async def test_validate_spread_zero_contracts(self, mock_broker):
        """Test spread validation rejects zero contracts."""
        executor = SpreadExecutor(mock_broker)
        spread = OptionSpread(
            spread_type=SpreadType.VERTICAL_CALL,
            ticker="AAPL",
            legs=[
                SpreadLeg(LegType.LONG_CALL, Decimal("150"), date.today(), 0),
                SpreadLeg(LegType.SHORT_CALL, Decimal("160"), date.today(), -1),
            ],
        )
        error = await executor._validate_spread(spread)
        assert "zero contracts" in error

    @pytest.mark.asyncio
    async def test_validate_spread_invalid_strike(self, mock_broker):
        """Test spread validation rejects invalid strike."""
        executor = SpreadExecutor(mock_broker)
        spread = OptionSpread(
            spread_type=SpreadType.VERTICAL_CALL,
            ticker="AAPL",
            legs=[
                SpreadLeg(LegType.LONG_CALL, Decimal("-150"), date.today(), 1),
                SpreadLeg(LegType.SHORT_CALL, Decimal("160"), date.today(), -1),
            ],
        )
        error = await executor._validate_spread(spread)
        assert "Invalid strike" in error

    @pytest.mark.asyncio
    async def test_validate_spread_broker_error(self, mock_broker):
        """Test spread validation handles broker errors."""
        mock_broker.get_account = AsyncMock(side_effect=Exception("Connection error"))
        executor = SpreadExecutor(mock_broker)

        expiry = date.today() + timedelta(days=30)
        spread = OptionSpread(
            spread_type=SpreadType.VERTICAL_CALL,
            ticker="AAPL",
            legs=[
                SpreadLeg(LegType.LONG_CALL, Decimal("150"), expiry, 1),
                SpreadLeg(LegType.SHORT_CALL, Decimal("160"), expiry, -1),
            ],
        )
        error = await executor._validate_spread(spread)
        assert "Broker connection error" in error

    @pytest.mark.asyncio
    async def test_execute_spread_atomic_success(self, mock_broker, sample_spread):
        """Test successful atomic execution."""
        executor = SpreadExecutor(mock_broker)

        result = await executor.execute_spread(
            sample_spread,
            execution_strategy=ExecutionStrategy.ATOMIC,
        )

        assert result.success is True
        assert result.spread_order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_execute_spread_rejected(self, mock_broker, sample_spread):
        """Test execution rejected by validation."""
        mock_broker.get_account = AsyncMock(side_effect=Exception("Auth error"))
        executor = SpreadExecutor(mock_broker)

        result = await executor.execute_spread(
            sample_spread,
            execution_strategy=ExecutionStrategy.ATOMIC,
        )

        assert result.success is False
        assert result.spread_order.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_execute_legged_in(self, mock_broker, sample_spread):
        """Test legged-in execution."""
        executor = SpreadExecutor(mock_broker)

        result = await executor.execute_spread(
            sample_spread,
            execution_strategy=ExecutionStrategy.LEGGED_IN,
        )

        # Should succeed with mock broker
        assert result is not None
        assert result.spread_order is not None

    @pytest.mark.asyncio
    async def test_execute_wings_first(self, mock_broker, sample_spread):
        """Test wings-first execution."""
        executor = SpreadExecutor(mock_broker)

        result = await executor.execute_spread(
            sample_spread,
            execution_strategy=ExecutionStrategy.WINGS_FIRST,
        )

        assert result is not None
        assert result.spread_order is not None

    @pytest.mark.asyncio
    async def test_calculate_filled_premium(self, mock_broker, sample_spread):
        """Test filled premium calculation."""
        executor = SpreadExecutor(mock_broker)

        spread_order = SpreadOrder(
            spread=sample_spread,
            leg_orders=[
                LegOrder(
                    leg=sample_spread.legs[0],
                    status=OrderStatus.FILLED,
                    filled_price=Decimal("5.00"),
                    filled_qty=1,
                ),
                LegOrder(
                    leg=sample_spread.legs[1],
                    status=OrderStatus.FILLED,
                    filled_price=Decimal("2.00"),
                    filled_qty=1,
                ),
            ],
        )

        executor._calculate_filled_premium(spread_order)

        # Long pays, short receives: -5.00 + 2.00 = -3.00
        assert spread_order.filled_premium == Decimal("-3.00")

    @pytest.mark.asyncio
    async def test_close_spread(self, mock_broker, sample_spread):
        """Test closing a spread."""
        executor = SpreadExecutor(mock_broker)

        result = await executor.close_spread(sample_spread)

        assert result is not None
        # Close creates reverse legs
        close_spread = result.spread_order.spread
        assert len(close_spread.legs) == 2

    @pytest.mark.asyncio
    async def test_rollback_partial_fill(self, mock_broker, sample_spread):
        """Test rollback of partial fill."""
        executor = SpreadExecutor(mock_broker)

        spread_order = SpreadOrder(
            spread=sample_spread,
            leg_orders=[
                LegOrder(
                    leg=sample_spread.legs[0],
                    status=OrderStatus.FILLED,
                    filled_price=Decimal("5.00"),
                    filled_qty=1,
                ),
                LegOrder(
                    leg=sample_spread.legs[1],
                    status=OrderStatus.PENDING,
                ),
            ],
        )

        await executor._rollback_partial_fill(spread_order)

        # Should have called submit_order to reverse the filled leg
        assert mock_broker.submit_order.called

    @pytest.mark.asyncio
    async def test_wait_for_fill_timeout(self, mock_broker, sample_spread):
        """Test wait for fill times out."""
        mock_broker.get_order = AsyncMock(return_value={"status": "pending"})
        executor = SpreadExecutor(mock_broker, order_timeout_seconds=1)

        spread_order = SpreadOrder(spread=sample_spread, order_id="order123")

        result = await executor._wait_for_fill(spread_order)
        assert result is False
