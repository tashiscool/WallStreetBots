"""
Execution Module for WallStreetBots.

Provides order execution, state management, and position adjustment.
"""

from .interfaces import (
    ExecutionClient,
    OrderRequest,
    OrderAck,
    OrderFill,
    OrderSide,
    OrderType,
    TimeInForce,
)

from .order_state_machine import (
    OrderStatus,
    OrderEvent,
    OrderStateMachine,
    OrderManager,
    ORDER_STATE_TABLE,
)

from .position_adjustment import (
    PositionAdjuster,
    PositionState,
    AdjustmentConfig,
    AdjustmentOrder,
    AdjustmentType,
    AdjustmentTrigger,
    AdjustmentStrategies,
)

from .replay_guard import ReplayGuard

from .shadow_client import ShadowExecutionClient, CanaryExecutionClient

# Multi-leg execution (ported from Polymarket-Kalshi arbitrage bot)
from .multi_leg import (
    LegSide,
    LegStatus,
    OrderLeg,
    MultiLegOrder,
    ExecutionResult,
    MultiLegExecutor,
    ArbitrageExecutor,
    SpreadExecutor,
)

# Position tracking (ported from Polymarket-Kalshi arbitrage bot)
from .position_tracker import (
    PositionSide,
    FillRecord,
    Position,
    PositionTrackerState,
    PositionChannel,
    PositionTracker,
    get_position_tracker,
)

# Arbitrage detection (ported from Polymarket-Kalshi arbitrage bot)
from .arbitrage_detector import (
    ArbitrageType,
    OpportunityStatus,
    MarketQuote,
    ArbitrageOpportunity,
    DetectorConfig,
    ArbitrageDetector,
    ArbitrageScanner,
    get_arbitrage_detector,
    configure_arbitrage_detector,
)

__all__ = [
    # Interfaces
    "ExecutionClient",
    "OrderAck",
    "OrderFill",
    "OrderRequest",
    "OrderSide",
    "OrderType",
    "TimeInForce",
    # Order state machine
    "OrderStatus",
    "OrderEvent",
    "OrderStateMachine",
    "OrderManager",
    "ORDER_STATE_TABLE",
    # Position adjustment
    "PositionAdjuster",
    "PositionState",
    "AdjustmentConfig",
    "AdjustmentOrder",
    "AdjustmentType",
    "AdjustmentTrigger",
    "AdjustmentStrategies",
    # Guards
    "ReplayGuard",
    "ShadowExecutionClient",
    "CanaryExecutionClient",
    # Multi-leg execution (from Polymarket-Kalshi arbitrage bot)
    "LegSide",
    "LegStatus",
    "OrderLeg",
    "MultiLegOrder",
    "ExecutionResult",
    "MultiLegExecutor",
    "ArbitrageExecutor",
    "SpreadExecutor",
    # Position tracking (from Polymarket-Kalshi arbitrage bot)
    "PositionSide",
    "FillRecord",
    "Position",
    "PositionTrackerState",
    "PositionChannel",
    "PositionTracker",
    "get_position_tracker",
    # Arbitrage detection (from Polymarket-Kalshi arbitrage bot)
    "ArbitrageType",
    "OpportunityStatus",
    "MarketQuote",
    "ArbitrageOpportunity",
    "DetectorConfig",
    "ArbitrageDetector",
    "ArbitrageScanner",
    "get_arbitrage_detector",
    "configure_arbitrage_detector",
]
