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

# L2 Order Book (market microstructure)
from .order_book import (
    OrderBook,
    OrderBookFeatures,
    OrderBookLevel,
    OrderBookManager,
    OrderBookSnapshot,
)

# Adverse selection & toxic flow detection
from .adverse_selection import (
    AdverseSelectionMetrics,
    ToxicFlowDetector,
    TradeRecord,
    VPINCalculator,
    VPINResult,
)

# Maker/taker fee model & execution optimization
from .fee_model import (
    AssetClass,
    FeeEstimate,
    FeeModel,
    FeeOptimizer,
    FeeSchedule,
    FeeType,
    OrderTypeRecommendation,
    ALPACA_EQUITY,
    ALPACA_CRYPTO,
    IBKR_TIERED_EQUITY,
    IBKR_FIXED_EQUITY,
    IBKR_OPTIONS,
    KNOWN_SCHEDULES,
)

# Pre-trade compliance
from .pre_trade_compliance import (
    ComplianceLimits,
    ComplianceResult,
    ComplianceRule,
    PreTradeComplianceService,
)

# Maker-checker (four-eyes) approval workflow
from .maker_checker import (
    ApprovalRequest,
    ApprovalStatus,
    ApprovalThresholds,
    MakerCheckerService,
)

# FIX protocol adapter
from .fix_adapter import (
    FIXExecutionClient,
    FIXMessage,
    FIXMessageBuilder,
    FIXMsgType,
    FIXOrdType,
    FIXSessionConfig,
    FIXSide,
    FIXTimeInForce,
    FIXVersion,
)

# Order Management System
from .oms import (
    ManagedOrder,
    OrderManagementSystem,
)

__all__ = [
    "ALPACA_CRYPTO",
    "ALPACA_EQUITY",
    "IBKR_FIXED_EQUITY",
    "IBKR_OPTIONS",
    "IBKR_TIERED_EQUITY",
    "KNOWN_SCHEDULES",
    "ORDER_STATE_TABLE",
    "AdjustmentConfig",
    "AdjustmentOrder",
    "AdjustmentStrategies",
    "AdjustmentTrigger",
    "AdjustmentType",
    # Adverse selection & toxic flow detection
    "AdverseSelectionMetrics",
    # Maker-checker (four-eyes) approval
    "ApprovalRequest",
    "ApprovalStatus",
    "ApprovalThresholds",
    "ArbitrageDetector",
    "ArbitrageExecutor",
    "ArbitrageOpportunity",
    "ArbitrageScanner",
    # Arbitrage detection (from Polymarket-Kalshi arbitrage bot)
    "ArbitrageType",
    # Maker/taker fee model & execution optimization
    "AssetClass",
    "CanaryExecutionClient",
    # Pre-trade compliance
    "ComplianceLimits",
    "ComplianceResult",
    "ComplianceRule",
    "DetectorConfig",
    # Interfaces
    "ExecutionClient",
    "ExecutionResult",
    # FIX protocol adapter
    "FIXExecutionClient",
    "FIXMessage",
    "FIXMessageBuilder",
    "FIXMsgType",
    "FIXOrdType",
    "FIXSessionConfig",
    "FIXSide",
    "FIXTimeInForce",
    "FIXVersion",
    "FeeEstimate",
    "FeeModel",
    "FeeOptimizer",
    "FeeSchedule",
    "FeeType",
    "FillRecord",
    # Multi-leg execution (from Polymarket-Kalshi arbitrage bot)
    "LegSide",
    "LegStatus",
    "MakerCheckerService",
    # Order Management System
    "ManagedOrder",
    "MarketQuote",
    "MultiLegExecutor",
    "MultiLegOrder",
    "OpportunityStatus",
    "OrderAck",
    # L2 Order Book (market microstructure)
    "OrderBook",
    "OrderBookFeatures",
    "OrderBookLevel",
    "OrderBookManager",
    "OrderBookSnapshot",
    "OrderEvent",
    "OrderFill",
    "OrderLeg",
    "OrderManagementSystem",
    "OrderManager",
    "OrderRequest",
    "OrderSide",
    "OrderStateMachine",
    # Order state machine
    "OrderStatus",
    "OrderType",
    "OrderTypeRecommendation",
    "Position",
    # Position adjustment
    "PositionAdjuster",
    "PositionChannel",
    # Position tracking (from Polymarket-Kalshi arbitrage bot)
    "PositionSide",
    "PositionState",
    "PositionTracker",
    "PositionTrackerState",
    "PreTradeComplianceService",
    # Guards
    "ReplayGuard",
    "ShadowExecutionClient",
    "SpreadExecutor",
    "TimeInForce",
    "ToxicFlowDetector",
    "TradeRecord",
    "VPINCalculator",
    "VPINResult",
    "configure_arbitrage_detector",
    "get_arbitrage_detector",
    "get_position_tracker",
]
