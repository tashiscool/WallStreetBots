"""
Prediction Market Arbitrage Module.

Best-of-breed synthesis from 8 open-source arbitrage bots:
- terauss/Polymarket-Kalshi-Arbitrage-bot
- CarlosIbCu/polymarket-kalshi-btc-arbitrage-bot
- dexorynLabs/polymarket-kalshi-arbitrage-trading-bot-v1
- ImMike/polymarket-arbitrage
- RichardFeynmanEnthusiast/kalshi-polymarket-arbitrage-bot
- jtdoherty/arb-bot
- carmentacollective/antevorta
- AntonyKoshy/prophet-arbitrage-bot

Key features:
- Event-driven architecture with MessageBus
- Domain-driven design with pure business logic
- Tiered fee calculation (Polymarket + Kalshi quadratic)
- Cross-platform market matching
- Multiple position sizing strategies (Kelly, sqrt, percentage)
- Emergency unwind mechanism
- Cool-down cycles after trades
"""

# Platform Client Abstraction
from .platform_client import (
    # Enums
    Platform,
    Outcome,
    Side,
    Environment,
    # Data classes
    PriceLevel,
    OrderBook,
    MarketState,
    OrderRequest,
    OrderResponse,
    # Abstract client
    PlatformClient,
    # Concrete implementations
    PolymarketClient,
    KalshiClient,
    # Factory
    PlatformClientFactory,
)

# Fee Calculator
from .fee_calculator import (
    FeeType,
    FeeResult,
    FeeCalculator,
    GasCostEstimator,
    get_fee_calculator,
)

# Market Matcher
from .market_matcher import (
    MarketCategory,
    MarketInfo,
    MarketPair,
    EntityExtractor,
    MarketCategorizer,
    MarketMatcher,
    MarketSlugGenerator,
    get_market_matcher,
)

# Position Sizing
from .position_sizing import (
    SizingStrategy,
    WalletBalance,
    SizingContext,
    SizingResult,
    PositionSizer,
    FixedSizer,
    PercentageSizer,
    KellySizer,
    SqrtSizer,
    ProportionalSizer,
    CompositePositionSizer,
    create_position_sizer,
    create_conservative_sizer,
)

# Arbitrage Engine (core)
from .arbitrage_engine import (
    # Domain Events
    DomainEvent,
    OrderBookUpdated,
    ArbitrageOpportunityDetected,
    TradeExecutionRequested,
    TradeExecutionCompleted,
    TradeExecutionFailed,
    UnwindRequested,
    SystemShutdownRequested,
    # Message Bus
    MessageBus,
    # Opportunity
    ArbitrageStrategy,
    ArbitrageOpportunity,
    # Core components
    ArbitrageDetector,
    TradeExecutor,
    ArbitrageEngine,
    # Factory
    create_arbitrage_engine,
)

# Dashboard Server (WebSocket + REST API)
from .dashboard_server import (
    DashboardState,
    ConnectionManager,
    DashboardIntegration,
    create_dashboard_app,
    run_dashboard_server,
)

# Logging Configuration (JSON structured logging)
from .logging_config import (
    LoggingConfig,
    ArbitrageJsonFormatter,
    ColoredFormatter,
    setup_logging,
    TradeLogger,
    OpportunityLogger,
    PerformanceLogger,
    get_trade_logger,
    get_opportunity_logger,
    get_performance_logger,
)

# Opportunity Analytics (timing and statistics)
from .opportunity_analytics import (
    OpportunityTiming,
    ArbStats,
    ExecutionStats,
    OpportunityAnalytics,
    LatencyTracker,
    Timer,
    get_opportunity_analytics,
    get_latency_tracker,
)

# CLI Diagnostic Printer (terminal visualization)
from .diagnostic_printer import (
    DiagnosticConfig,
    Colors,
    DiagnosticPrinter,
    run_diagnostic_printer,
    print_order_book_snapshot,
    print_arbitrage_summary,
)

# Trade Storage (database persistence)
from .trade_storage import (
    ArbitrageTradeRecord,
    TradeStorageBackend,
    SQLiteStorageBackend,
    TradeStorageService,
    init_trade_storage,
    get_trade_storage,
)

# Matching Progress Tracker
from .matching_progress import (
    MatchingStatus,
    MatchingProgress,
    MatchingProgressTracker,
    AsyncMatchingProgressTracker,
    format_progress_bar,
    print_matching_progress,
    get_matching_progress_tracker,
)

# Performance Profiling
from .profiling import (
    CProfiler,
    YappiProfiler,
    run_cprofile_session,
    run_yappi_session,
    profile_function,
)

__all__ = [
    "ArbStats",
    "ArbitrageDetector",
    "ArbitrageEngine",
    "ArbitrageJsonFormatter",
    "ArbitrageOpportunity",
    "ArbitrageOpportunityDetected",
    "ArbitrageStrategy",
    # Trade Storage
    "ArbitrageTradeRecord",
    "AsyncMatchingProgressTracker",
    # Performance Profiling
    "CProfiler",
    "ColoredFormatter",
    "Colors",
    "CompositePositionSizer",
    "ConnectionManager",
    "DashboardIntegration",
    # Dashboard Server
    "DashboardState",
    # Diagnostic Printer
    "DiagnosticConfig",
    "DiagnosticPrinter",
    # Arbitrage Engine
    "DomainEvent",
    "EntityExtractor",
    "Environment",
    "ExecutionStats",
    "FeeCalculator",
    "FeeResult",
    # Fee Calculator
    "FeeType",
    "FixedSizer",
    "GasCostEstimator",
    "KalshiClient",
    "KellySizer",
    "LatencyTracker",
    # Logging Configuration
    "LoggingConfig",
    "MarketCategorizer",
    # Market Matcher
    "MarketCategory",
    "MarketInfo",
    "MarketMatcher",
    "MarketPair",
    "MarketSlugGenerator",
    "MarketState",
    "MatchingProgress",
    "MatchingProgressTracker",
    # Matching Progress
    "MatchingStatus",
    "MessageBus",
    "OpportunityAnalytics",
    "OpportunityLogger",
    # Opportunity Analytics
    "OpportunityTiming",
    "OrderBook",
    "OrderBookUpdated",
    "OrderRequest",
    "OrderResponse",
    "Outcome",
    "PercentageSizer",
    "PerformanceLogger",
    # Platform Client
    "Platform",
    "PlatformClient",
    "PlatformClientFactory",
    "PolymarketClient",
    "PositionSizer",
    "PriceLevel",
    "ProportionalSizer",
    "SQLiteStorageBackend",
    "Side",
    "SizingContext",
    "SizingResult",
    # Position Sizing
    "SizingStrategy",
    "SqrtSizer",
    "SystemShutdownRequested",
    "Timer",
    "TradeExecutionCompleted",
    "TradeExecutionFailed",
    "TradeExecutionRequested",
    "TradeExecutor",
    "TradeLogger",
    "TradeStorageBackend",
    "TradeStorageService",
    "UnwindRequested",
    "WalletBalance",
    "YappiProfiler",
    "create_arbitrage_engine",
    "create_conservative_sizer",
    "create_dashboard_app",
    "create_position_sizer",
    "format_progress_bar",
    "get_fee_calculator",
    "get_latency_tracker",
    "get_market_matcher",
    "get_matching_progress_tracker",
    "get_opportunity_analytics",
    "get_opportunity_logger",
    "get_performance_logger",
    "get_trade_logger",
    "get_trade_storage",
    "init_trade_storage",
    "print_arbitrage_summary",
    "print_matching_progress",
    "print_order_book_snapshot",
    "profile_function",
    "run_cprofile_session",
    "run_dashboard_server",
    "run_diagnostic_printer",
    "run_yappi_session",
    "setup_logging",
]
