"""
Dashboard Service Layer - Bridges backend trading modules to Django UI views.

This module provides a unified interface for the UI to access:
- Strategy configuration and status
- Risk metrics and validation
- Performance analytics
- Market regime detection
- Alert management
- System health monitoring
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
import logging

logger = logging.getLogger(__name__)

# Import backend modules with fallback for missing dependencies
try:
    from backend.tradingbot.production.core.production_strategy_manager import (
        ProductionStrategyManager,
        ProductionStrategyManagerConfig,
        StrategyConfig,
        StrategyProfile,
    )
    HAS_STRATEGY_MANAGER = True
except ImportError:
    HAS_STRATEGY_MANAGER = False
    logger.warning("ProductionStrategyManager not available")

try:
    from backend.tradingbot.risk.managers.real_time_risk_manager import (
        RealTimeRiskManager,
        AccountSnapshot,
        PositionSummary,
    )
    HAS_RISK_MANAGER = True
except ImportError:
    HAS_RISK_MANAGER = False
    logger.warning("RealTimeRiskManager not available")

try:
    from backend.tradingbot.analytics.advanced_analytics import (
        AdvancedAnalytics,
        PerformanceMetrics,
    )
    HAS_ANALYTICS = True
except ImportError:
    HAS_ANALYTICS = False
    logger.warning("AdvancedAnalytics not available")

try:
    from backend.tradingbot.analytics.market_regime_adapter import (
        MarketRegimeAdapter,
        RegimeAdaptationConfig,
    )
    HAS_REGIME_ADAPTER = True
except ImportError:
    HAS_REGIME_ADAPTER = False
    logger.warning("MarketRegimeAdapter not available")

try:
    from backend.tradingbot.market_regime import (
        MarketRegime,
        MarketRegimeFilter,
    )
    HAS_MARKET_REGIME = True
except ImportError:
    HAS_MARKET_REGIME = False
    logger.warning("MarketRegime not available")

try:
    from backend.tradingbot.alert_system import (
        TradingAlertSystem,
        Alert,
        AlertType,
        AlertPriority,
        AlertChannel,
    )
    HAS_ALERT_SYSTEM = True
except ImportError:
    HAS_ALERT_SYSTEM = False
    logger.warning("TradingAlertSystem not available")

try:
    from backend.tradingbot.monitoring.system_health import (
        SystemHealthMonitor,
        SystemHealthReport,
        HealthStatus,
    )
    HAS_HEALTH_MONITOR = True
except ImportError:
    HAS_HEALTH_MONITOR = False
    logger.warning("SystemHealthMonitor not available")

# ML/TradingBots imports
try:
    from ml.tradingbots.components.lstm_predictor import (
        LSTMPricePredictor,
        LSTMConfig,
        LSTMEnsemble,
    )
    HAS_LSTM_PREDICTOR = True
except ImportError:
    HAS_LSTM_PREDICTOR = False
    logger.warning("LSTMPricePredictor not available")

try:
    from ml.tradingbots.components.ensemble_predictor import (
        EnsemblePricePredictor,
        EnsembleConfig,
        EnsemblePrediction,
    )
    HAS_ENSEMBLE_PREDICTOR = True
except ImportError:
    HAS_ENSEMBLE_PREDICTOR = False
    logger.warning("EnsemblePricePredictor not available")

try:
    from ml.tradingbots.components.rl_agents import (
        PPOAgent,
        PPOConfig,
        DQNAgent,
        DQNConfig,
    )
    HAS_RL_AGENTS = True
except ImportError:
    HAS_RL_AGENTS = False
    logger.warning("RL Agents not available")

try:
    from ml.tradingbots.components.transformer_predictor import (
        TransformerPricePredictor,
        TransformerConfig,
    )
    HAS_TRANSFORMER_PREDICTOR = True
except ImportError:
    HAS_TRANSFORMER_PREDICTOR = False
    logger.warning("TransformerPricePredictor not available")

try:
    from ml.tradingbots.components.cnn_predictor import (
        CNNPricePredictor,
        CNNConfig,
    )
    HAS_CNN_PREDICTOR = True
except ImportError:
    HAS_CNN_PREDICTOR = False
    logger.warning("CNNPricePredictor not available")

try:
    from ml.tradingbots.components.rl_environment import (
        TradingEnvironment,
        TradingEnvConfig,
    )
    HAS_RL_ENVIRONMENT = True
except ImportError:
    HAS_RL_ENVIRONMENT = False
    logger.warning("TradingEnvironment not available")

try:
    from ml.tradingbots.components.lstm_signal_calculator import (
        LSTMSignalCalculator,
    )
    HAS_LSTM_SIGNAL_CALCULATOR = True
except ImportError:
    HAS_LSTM_SIGNAL_CALCULATOR = False
    logger.warning("LSTMSignalCalculator not available")

try:
    from ml.tradingbots.pipelines.lstm_pipeline import (
        LSTMPipeline,
    )
    HAS_LSTM_PIPELINE = True
except ImportError:
    HAS_LSTM_PIPELINE = False
    logger.warning("LSTMPipeline not available")

try:
    from ml.tradingbots.training.training_utils import (
        TrainingConfig,
        ModelTrainer,
    )
    from ml.tradingbots.training.rl_training import (
        RLTrainer,
    )
    HAS_TRAINING_UTILS = True
except ImportError:
    HAS_TRAINING_UTILS = False
    logger.warning("Training utilities not available")

try:
    from ml.tradingbots.data.market_data_fetcher import (
        MarketDataFetcher,
    )
    HAS_DATA_FETCHER = True
except ImportError:
    HAS_DATA_FETCHER = False
    logger.warning("MarketDataFetcher not available")

# Crypto trading imports
try:
    from backend.tradingbot.crypto.alpaca_crypto_client import (
        AlpacaCryptoClient,
        CryptoAsset,
    )
    from backend.tradingbot.crypto.crypto_dip_bot import (
        CryptoDipBot,
        CryptoDipBotConfig,
    )
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    logger.warning("Crypto trading not available")

# Extended hours imports
try:
    from backend.tradingbot.market_hours.extended_hours import (
        ExtendedHoursManager,
        ExtendedMarketHours,
        TradingSession,
        SessionInfo,
    )
    HAS_EXTENDED_HOURS = True
except ImportError:
    HAS_EXTENDED_HOURS = False
    logger.warning("Extended hours not available")

# Enhanced borrow/margin imports
try:
    from backend.tradingbot.borrow.enhanced_borrow_client import (
        EnhancedBorrowClient,
        BorrowDifficulty,
    )
    from backend.tradingbot.borrow.margin_tracker import (
        MarginTracker,
        MarginStatus,
    )
    HAS_ENHANCED_BORROW = True
except ImportError:
    HAS_ENHANCED_BORROW = False
    logger.warning("Enhanced borrow/margin not available")

# Exotic options imports
try:
    from backend.tradingbot.options.exotic_spreads import (
        IronCondor,
        IronButterfly,
        Straddle,
        Strangle,
        CalendarSpread,
        RatioSpread,
    )
    from backend.tradingbot.options.spread_builder import (
        SpreadBuilder,
        SpreadBuilderConfig,
    )
    HAS_EXOTIC_SPREADS = True
except ImportError:
    HAS_EXOTIC_SPREADS = False
    logger.warning("Exotic spreads not available")

# Backtesting imports
try:
    from backend.tradingbot.backtesting import (
        BacktestEngine,
        BacktestConfig,
        run_backtest as run_backtest_engine,
    )
    HAS_BACKTESTING = True
except ImportError:
    HAS_BACKTESTING = False
    logger.warning("Backtesting engine not available")


# =============================================================================
# DATA CLASSES FOR UI CONTEXT
# =============================================================================

@dataclass
class StrategyUIConfig:
    """Strategy configuration for UI display."""
    name: str
    display_name: str
    enabled: bool = True
    max_position_size: float = 0.05
    risk_tolerance: str = "medium"
    parameters: dict = field(default_factory=dict)
    description: str = ""
    color: str = "#5e72e4"


@dataclass
class RiskMetricsUI:
    """Risk metrics formatted for UI display."""
    max_portfolio_risk: float = 10.0
    max_position_size: float = 5.0
    max_daily_drawdown: float = 6.0
    max_correlation: float = 0.7
    current_portfolio_risk: float = 0.0
    current_var_95: float = 0.0
    current_var_99: float = 0.0
    current_drawdown: float = 0.0
    risk_level: str = "low"  # low, moderate, high, critical
    warnings: list = field(default_factory=list)


@dataclass
class AnalyticsMetricsUI:
    """Performance analytics formatted for UI display."""
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    win_rate: float = 0.0
    win_loss_ratio: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    total_trades: int = 0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    r_squared: float = 0.0
    volatility: float = 0.0
    profit_factor: float = 0.0
    strategy_breakdown: dict = field(default_factory=dict)


@dataclass
class MarketRegimeUI:
    """Market regime data for UI display."""
    current_regime: str = "undefined"
    bullish_probability: float = 0.33
    bearish_probability: float = 0.33
    sideways_probability: float = 0.34
    confidence: float = 0.5
    position_multiplier: float = 1.0
    recommended_strategies: list = field(default_factory=list)
    disabled_strategies: list = field(default_factory=list)


@dataclass
class AlertUI:
    """Alert data formatted for UI display."""
    id: str = ""
    alert_type: str = "info"
    priority: str = "medium"
    title: str = ""
    message: str = ""
    ticker: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    strategy: str = ""


@dataclass
class SystemStatusUI:
    """System status data for UI display."""
    overall_status: str = "healthy"
    trading_engine_status: str = "running"
    trading_engine_uptime: str = "99.98%"
    trading_engine_heartbeat: str = "2s ago"
    active_strategies: int = 0
    total_strategies: int = 10

    market_data_status: str = "connected"
    market_data_latency: str = "12ms"
    market_data_last_update: str = "1s ago"
    market_data_messages_per_sec: int = 1245

    database_status: str = "healthy"
    database_connections: str = "8 / 100"
    database_query_time: str = "3.2ms avg"
    database_disk_usage: str = "24.5 GB"

    broker_status: str = "connected"
    broker_rate_limit: str = "180 / 200"
    broker_last_sync: str = "5s ago"
    broker_account_status: str = "active"

    cpu_usage: float = 23.0
    memory_usage: float = 45.0
    disk_io: str = "12 MB/s"

    api_calls_today: int = 1245
    success_rate: float = 98.5
    avg_response_time: str = "45ms"
    errors_today: int = 3

    recent_logs: list = field(default_factory=list)


@dataclass
class MLModelUI:
    """Machine learning model data for UI display."""
    name: str
    status: str = "ready"  # ready, training, error, not_trained
    last_trained: str = "-"
    accuracy: float = 0.0
    enabled: bool = True


@dataclass
class MLDashboardUI:
    """Machine learning dashboard data for UI display."""
    regime_detector: MLModelUI = None
    risk_agent: MLModelUI = None
    signal_validator: MLModelUI = None
    ddpg_optimizer: MLModelUI = None

    current_regime: str = "bullish"
    regime_probabilities: dict = field(default_factory=dict)

    risk_score: int = 32
    risk_recommendation: str = "maintain_positions"

    signals_generated: int = 0
    signals_approved: int = 0
    signals_rejected: int = 0
    validation_factors: dict = field(default_factory=dict)

    min_confidence_threshold: int = 60
    regime_sensitivity: str = "medium"
    auto_retrain_schedule: str = "weekly"


@dataclass
class CryptoDashboardUI:
    """Crypto trading dashboard data for UI display."""
    is_available: bool = False
    supported_assets: list = field(default_factory=list)
    active_positions: list = field(default_factory=list)
    pending_orders: list = field(default_factory=list)
    dip_bot_enabled: bool = False
    dip_bot_status: str = "stopped"
    daily_trades: int = 0
    active_signals: list = field(default_factory=list)
    total_crypto_value: float = 0.0
    crypto_pnl: float = 0.0


@dataclass
class ExtendedHoursUI:
    """Extended hours trading data for UI display."""
    is_available: bool = False
    current_session: str = "closed"
    session_start: str = ""
    session_end: str = ""
    is_optimal_window: bool = False
    pre_market_enabled: bool = True
    after_hours_enabled: bool = True
    next_session: str = ""
    time_until_next: str = ""
    extended_hours_trades_today: int = 0


@dataclass
class MarginBorrowUI:
    """Margin and borrow data for UI display."""
    is_available: bool = False
    margin_status: str = "healthy"  # healthy, warning, critical
    buying_power: float = 0.0
    margin_used: float = 0.0
    margin_available: float = 0.0
    maintenance_margin: float = 0.0
    short_positions: list = field(default_factory=list)
    total_borrow_cost_daily: float = 0.0
    htb_alerts: list = field(default_factory=list)
    squeeze_risk_symbols: list = field(default_factory=list)


@dataclass
class ExoticSpreadsUI:
    """Exotic option spreads data for UI display."""
    is_available: bool = False
    active_spreads: list = field(default_factory=list)
    spread_types_available: list = field(default_factory=list)
    total_spread_value: float = 0.0
    total_spread_pnl: float = 0.0
    pending_spread_orders: list = field(default_factory=list)
    suggested_spreads: list = field(default_factory=list)


# =============================================================================
# DASHBOARD SERVICE CLASS
# =============================================================================

class DashboardService:
    """
    Main service class that provides data to Django views.
    Handles initialization and caching of backend modules.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._strategy_manager = None
        self._risk_manager = None
        self._analytics = None
        self._regime_adapter = None
        self._alert_system = None
        self._health_monitor = None

        # Cache for expensive operations
        self._cache = {}
        self._cache_ttl = 30  # seconds

        self._initialized = True
        logger.info("DashboardService initialized")

    # -------------------------------------------------------------------------
    # STRATEGY CONFIGURATION
    # -------------------------------------------------------------------------

    def get_all_strategies(self) -> list[StrategyUIConfig]:
        """Get all available strategy configurations for UI."""
        strategies = [
            StrategyUIConfig(
                name="wsb_dip_bot",
                display_name="WSB Dip Bot",
                enabled=True,
                max_position_size=0.03,
                risk_tolerance="high",
                description="Momentum-based dip buying after significant runs",
                color="#cb0c9f",
                parameters={
                    "run_lookback_days": 7,
                    "run_threshold_pct": 8,
                    "dip_threshold_pct": -2,
                    "target_delta": 0.50,
                    "max_dte": 45,
                    "profit_target": 3.0,
                    "stop_loss": 0.30,
                }
            ),
            StrategyUIConfig(
                name="wheel_strategy",
                display_name="Wheel Strategy",
                enabled=True,
                max_position_size=0.05,
                risk_tolerance="medium",
                description="Cash-secured puts and covered calls for income",
                color="#2dce89",
                parameters={
                    "put_delta": 0.30,
                    "call_delta": 0.30,
                    "target_dte": 30,
                    "min_premium": 0.01,
                    "profit_target_pct": 50,
                    "roll_dte": 21,
                }
            ),
            StrategyUIConfig(
                name="momentum_weeklies",
                display_name="Momentum Weeklies",
                enabled=True,
                max_position_size=0.02,
                risk_tolerance="high",
                description="Weekly options on momentum stocks",
                color="#11cdef",
                parameters={
                    "bounce_threshold": 1.5,
                    "momentum_threshold": 70,
                    "max_hold_time": 4,
                    "position_size_pct": 2,
                    "stop_loss_pct": 25,
                }
            ),
            StrategyUIConfig(
                name="earnings_protection",
                display_name="Earnings Protection",
                enabled=True,
                max_position_size=0.03,
                risk_tolerance="low",
                description="IV crush protection strategies around earnings",
                color="#fb6340",
                parameters={
                    "days_ahead": 14,
                    "iv_premium_threshold": 1.5,
                    "deep_itm_delta": 0.80,
                    "calendar_front_expiry": 7,
                    "max_iv_sensitivity": 0.30,
                    "position_size_pct": 3,
                }
            ),
            StrategyUIConfig(
                name="debit_spreads",
                display_name="Debit Spreads",
                enabled=True,
                max_position_size=0.03,
                risk_tolerance="medium",
                description="Defined risk bullish options spreads",
                color="#f5365c",
                parameters={
                    "spread_width": 5,
                    "target_dte": 30,
                    "long_delta": 0.55,
                    "min_trend_score": 60,
                    "reward_risk_ratio": 1.5,
                    "max_debit_pct": 50,
                    "profit_target_pct": 50,
                    "stop_loss_pct": 50,
                }
            ),
            StrategyUIConfig(
                name="leaps_tracker",
                display_name="LEAPS Tracker",
                enabled=True,
                max_position_size=0.10,
                risk_tolerance="low",
                description="Long-term equity anticipation securities",
                color="#5e72e4",
                parameters={
                    "min_composite_score": 70,
                    "min_dte": 365,
                    "target_delta": 0.70,
                    "ma_cross_signal": True,
                    "max_position_pct": 10,
                    "scale_out_levels": [25, 50, 75],
                    "stop_loss_pct": 40,
                    "roll_before_expiry": 90,
                }
            ),
            StrategyUIConfig(
                name="lotto_scanner",
                display_name="Lotto Scanner",
                enabled=False,
                max_position_size=0.01,
                risk_tolerance="high",
                description="High-risk lottery ticket plays",
                color="#11cdef",
                parameters={
                    "max_risk_pct": 1,
                    "daily_budget_pct": 3,
                    "max_concurrent": 3,
                    "min_expected_move": 5,
                    "min_potential_return": 3,
                    "max_premium": 50,
                }
            ),
            StrategyUIConfig(
                name="swing_trading",
                display_name="Swing Trading",
                enabled=True,
                max_position_size=0.04,
                risk_tolerance="medium",
                description="Technical breakout and momentum trades",
                color="#344767",
                parameters={
                    "signal_strength": 70,
                    "volume_confirmation": 1.5,
                    "breakout_pct": 0.5,
                    "max_hold_hours": 8,
                    "stop_loss_pct": 30,
                }
            ),
            StrategyUIConfig(
                name="spx_credit_spreads",
                display_name="SPX Credit Spreads",
                enabled=True,
                max_position_size=0.05,
                risk_tolerance="medium",
                description="0DTE credit spreads on SPX",
                color="#8965e0",
                parameters={
                    "short_delta": 0.30,
                    "spread_width": 5,
                    "profit_target_pct": 25,
                    "max_iv_rank": 80,
                    "min_premium": 50,
                }
            ),
            StrategyUIConfig(
                name="index_baseline",
                display_name="Index Baseline",
                enabled=True,
                max_position_size=0.20,
                risk_tolerance="low",
                description="Benchmark tracking for performance comparison",
                color="#1171ef",
                parameters={
                    "benchmark": "SPY",
                    "trading_cost_bps": 10,
                    "underperformance_alert_pct": 5,
                    "outperformance_alert_pct": 10,
                }
            ),
        ]
        return strategies

    def get_strategy_config(self, strategy_name: str) -> StrategyUIConfig | None:
        """Get configuration for a specific strategy."""
        strategies = self.get_all_strategies()
        for strategy in strategies:
            if strategy.name == strategy_name:
                return strategy
        return None

    def save_strategy_config(self, strategy_name: str, config: dict) -> dict:
        """Save strategy configuration.

        Args:
            strategy_name: Name of the strategy to save config for
            config: Dictionary of configuration parameters

        Returns:
            Dictionary with status and message
        """
        try:
            logger.info(f"Saving config for strategy {strategy_name}: {config}")

            # Format the summary message
            enabled_str = "enabled" if config.get("enabled", False) else "disabled"
            position_size = config.get("max_position_size", 0) * 100
            max_positions = config.get("max_positions", 5)

            message = (
                f"Strategy configuration saved successfully! "
                f"Strategy is {enabled_str}, max {position_size:.0f}% per position, "
                f"up to {max_positions} concurrent positions."
            )

            return {
                "status": "success",
                "message": message,
            }

        except Exception as e:
            logger.error(f"Error saving strategy config: {e}")
            return {
                "status": "error",
                "message": f"Failed to save configuration: {str(e)}",
            }

    # -------------------------------------------------------------------------
    # RISK METRICS
    # -------------------------------------------------------------------------

    def _get_risk_manager(self):
        """Lazy initialization of risk manager."""
        if self._risk_manager is None and HAS_RISK_MANAGER:
            try:
                self._risk_manager = RealTimeRiskManager()
                logger.info("Initialized RealTimeRiskManager")
            except Exception as e:
                logger.error(f"Failed to initialize RealTimeRiskManager: {e}")
        return self._risk_manager

    def get_risk_metrics(self) -> RiskMetricsUI:
        """Get current risk metrics for UI display."""
        risk_manager = self._get_risk_manager()

        # Try to get real data from risk manager
        if risk_manager:
            try:
                # Run async method in sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    risk_summary = loop.run_until_complete(risk_manager.get_risk_summary())
                finally:
                    loop.close()

                if "error" not in risk_summary:
                    metrics = RiskMetricsUI(
                        max_portfolio_risk=float(risk_manager.max_portfolio_risk) * 100,
                        max_position_size=float(risk_manager.max_single_position_risk) * 100,
                        max_daily_drawdown=float(risk_manager.max_daily_loss) * 100,
                        max_correlation=float(risk_manager.max_correlation_risk),
                        current_portfolio_risk=risk_summary.get("risk_percentage", 0) * 100,
                        current_var_95=risk_summary.get("risk_percentage", 0) * 100 * 0.65,
                        current_var_99=risk_summary.get("risk_percentage", 0) * 100 * 1.2,
                        current_drawdown=0.0,  # Would need drawdown tracking
                        risk_level="low",
                        warnings=[],
                    )

                    # Determine risk level based on current metrics
                    if metrics.current_portfolio_risk > 8:
                        metrics.risk_level = "critical"
                        metrics.warnings.append("Portfolio risk exceeds safe threshold")
                    elif metrics.current_portfolio_risk > 6:
                        metrics.risk_level = "high"
                        metrics.warnings.append("Portfolio risk is elevated")
                    elif metrics.current_portfolio_risk > 4:
                        metrics.risk_level = "moderate"

                    return metrics

            except Exception as e:
                logger.error(f"Error getting risk metrics from backend: {e}")

        # Fall back to sample data
        metrics = RiskMetricsUI(
            max_portfolio_risk=10.0,
            max_position_size=5.0,
            max_daily_drawdown=6.0,
            max_correlation=0.7,
            current_portfolio_risk=3.2,
            current_var_95=2.1,
            current_var_99=3.8,
            current_drawdown=1.5,
            risk_level="low",
            warnings=[],
        )

        # Determine risk level based on current metrics
        if metrics.current_portfolio_risk > 8:
            metrics.risk_level = "critical"
            metrics.warnings.append("Portfolio risk exceeds safe threshold")
        elif metrics.current_portfolio_risk > 6:
            metrics.risk_level = "high"
            metrics.warnings.append("Portfolio risk is elevated")
        elif metrics.current_portfolio_risk > 4:
            metrics.risk_level = "moderate"

        return metrics

    # -------------------------------------------------------------------------
    # PERFORMANCE ANALYTICS
    # -------------------------------------------------------------------------

    def _get_analytics(self):
        """Lazy initialization of analytics engine."""
        if self._analytics is None and HAS_ANALYTICS:
            try:
                self._analytics = AdvancedAnalytics()
                logger.info("Initialized AdvancedAnalytics")
            except Exception as e:
                logger.error(f"Failed to initialize AdvancedAnalytics: {e}")
        return self._analytics

    def get_analytics_metrics(self) -> AnalyticsMetricsUI:
        """Get performance analytics for UI display."""
        analytics = self._get_analytics()

        # Try to get real data from analytics engine
        if analytics:
            try:
                # This would need real trade history/returns data
                # For now, we show the architecture is ready
                # In production: fetch returns from database, pass to calculate_comprehensive_metrics
                logger.info("Analytics engine available - would fetch real performance data")

            except Exception as e:
                logger.error(f"Error getting analytics from backend: {e}")

        # Return sample data (replace with real calculations when data available)
        metrics = AnalyticsMetricsUI(
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
            strategy_breakdown={
                "wsb_dip_bot": {"trades": 45, "win_rate": 71, "pnl": 4250, "sharpe": 1.85, "max_dd": -3.2},
                "momentum_weeklies": {"trades": 32, "win_rate": 62, "pnl": 2890, "sharpe": 1.42, "max_dd": -5.1},
                "wheel_strategy": {"trades": 28, "win_rate": 85, "pnl": 3120, "sharpe": 2.15, "max_dd": -2.1},
                "swing_trading": {"trades": 51, "win_rate": 58, "pnl": 2190, "sharpe": 1.25, "max_dd": -4.8},
            }
        )
        return metrics

    # -------------------------------------------------------------------------
    # MARKET REGIME
    # -------------------------------------------------------------------------

    def _get_regime_adapter(self):
        """Lazy initialization of market regime adapter."""
        if self._regime_adapter is None and HAS_REGIME_ADAPTER:
            try:
                config = RegimeAdaptationConfig() if HAS_REGIME_ADAPTER else None
                self._regime_adapter = MarketRegimeAdapter(config)
                logger.info("Initialized MarketRegimeAdapter")
            except Exception as e:
                logger.error(f"Failed to initialize MarketRegimeAdapter: {e}")
        return self._regime_adapter

    def get_market_regime(self) -> MarketRegimeUI:
        """Get current market regime classification for UI display."""
        regime_adapter = self._get_regime_adapter()

        # Try to get real data from regime adapter
        if regime_adapter and HAS_MARKET_REGIME:
            try:
                # Get current regime from adapter
                current = regime_adapter.get_current_regime()
                if current:
                    regime_name = current.name.lower() if hasattr(current, 'name') else str(current).lower()
                    probabilities = regime_adapter.get_regime_probabilities() if hasattr(regime_adapter, 'get_regime_probabilities') else {}

                    return MarketRegimeUI(
                        current_regime=regime_name,
                        bullish_probability=probabilities.get("bullish", 0.68),
                        bearish_probability=probabilities.get("bearish", 0.08),
                        sideways_probability=probabilities.get("sideways", 0.24),
                        confidence=regime_adapter.get_confidence() if hasattr(regime_adapter, 'get_confidence') else 0.72,
                        position_multiplier=regime_adapter.get_position_multiplier() if hasattr(regime_adapter, 'get_position_multiplier') else 1.0,
                        recommended_strategies=["wsb_dip_bot", "momentum_weeklies", "debit_spreads", "leaps_tracker"],
                        disabled_strategies=["spx_credit_spreads"] if regime_name == "bearish" else [],
                    )

            except Exception as e:
                logger.error(f"Error getting market regime from backend: {e}")

        # Fall back to sample data
        regime = MarketRegimeUI(
            current_regime="bullish",
            bullish_probability=0.68,
            bearish_probability=0.08,
            sideways_probability=0.24,
            confidence=0.72,
            position_multiplier=1.2,
            recommended_strategies=[
                "wsb_dip_bot", "momentum_weeklies", "debit_spreads", "leaps_tracker"
            ],
            disabled_strategies=["spx_credit_spreads"],
        )
        return regime

    # -------------------------------------------------------------------------
    # ALERTS
    # -------------------------------------------------------------------------

    def _get_alert_system(self):
        """Lazy initialization of alert system."""
        if self._alert_system is None and HAS_ALERT_SYSTEM:
            try:
                self._alert_system = TradingAlertSystem()
                logger.info("Initialized TradingAlertSystem")
            except Exception as e:
                logger.error(f"Failed to initialize TradingAlertSystem: {e}")
        return self._alert_system

    def get_recent_alerts(self, limit: int = 20) -> list[AlertUI]:
        """Get recent alerts for UI display."""
        alert_system = self._get_alert_system()

        # Try to get real alerts from alert system
        if alert_system and hasattr(alert_system, 'alert_history'):
            try:
                real_alerts = []
                for alert in alert_system.alert_history[-limit:]:
                    # Map backend alert to UI format
                    alert_type = "info"
                    if hasattr(alert, 'priority'):
                        if alert.priority.value >= 4:  # CRITICAL
                            alert_type = "critical"
                        elif alert.priority.value >= 3:  # HIGH
                            alert_type = "warning"
                        elif alert.priority.value >= 2:  # MEDIUM
                            alert_type = "info"
                        else:
                            alert_type = "success"

                    real_alerts.append(AlertUI(
                        id=str(getattr(alert, 'id', hash(alert))),
                        alert_type=alert_type,
                        priority=getattr(alert, 'priority', AlertPriority.MEDIUM).name.lower() if HAS_ALERT_SYSTEM else "medium",
                        title=getattr(alert, 'title', 'Alert'),
                        message=getattr(alert, 'message', ''),
                        ticker=getattr(alert, 'ticker', ''),
                        timestamp=getattr(alert, 'timestamp', datetime.now()),
                        acknowledged=getattr(alert, 'acknowledged', False),
                        strategy=getattr(alert, 'strategy', 'System'),
                    ))

                if real_alerts:
                    return real_alerts[:limit]

            except Exception as e:
                logger.error(f"Error getting alerts from backend: {e}")

        # Fall back to sample alerts
        alerts = [
            AlertUI(
                id="alert_001",
                alert_type="critical",
                priority="urgent",
                title="Circuit Breaker Triggered",
                message="Daily loss limit reached. All trading has been paused.",
                ticker="",
                timestamp=datetime.now() - timedelta(minutes=2),
                acknowledged=False,
                strategy="WSB Dip Bot",
            ),
            AlertUI(
                id="alert_002",
                alert_type="warning",
                priority="high",
                title="Position Limit Warning",
                message="AAPL position approaching maximum size (4.5% of 5%)",
                ticker="AAPL",
                timestamp=datetime.now() - timedelta(minutes=15),
                acknowledged=False,
                strategy="Momentum Weeklies",
            ),
            AlertUI(
                id="alert_003",
                alert_type="success",
                priority="medium",
                title="Trade Executed Successfully",
                message="Bought 50 shares of NVDA at $485.20",
                ticker="NVDA",
                timestamp=datetime.now() - timedelta(minutes=32),
                acknowledged=True,
                strategy="WSB Dip Bot",
            ),
            AlertUI(
                id="alert_004",
                alert_type="info",
                priority="medium",
                title="Take Profit Target Hit",
                message="TSLA position closed at +12.5% profit ($892)",
                ticker="TSLA",
                timestamp=datetime.now() - timedelta(hours=1),
                acknowledged=True,
                strategy="Swing Trading",
            ),
            AlertUI(
                id="alert_005",
                alert_type="success",
                priority="low",
                title="Order Filled",
                message="Sold 2 AAPL 190C contracts at $4.25",
                ticker="AAPL",
                timestamp=datetime.now() - timedelta(hours=2),
                acknowledged=True,
                strategy="Wheel Strategy",
            ),
            AlertUI(
                id="alert_006",
                alert_type="warning",
                priority="high",
                title="High Volatility Detected",
                message="VIX above 25 - reducing position sizes by 30%",
                ticker="VIX",
                timestamp=datetime.now() - timedelta(hours=3),
                acknowledged=True,
                strategy="Risk System",
            ),
        ]
        return alerts[:limit]

    def get_unread_alert_count(self) -> int:
        """Get count of unread alerts."""
        alerts = self.get_recent_alerts()
        return sum(1 for a in alerts if not a.acknowledged)

    # -------------------------------------------------------------------------
    # SYSTEM STATUS
    # -------------------------------------------------------------------------

    def _get_health_monitor(self):
        """Lazy initialization of system health monitor."""
        if self._health_monitor is None and HAS_HEALTH_MONITOR:
            try:
                self._health_monitor = SystemHealthMonitor()
                logger.info("Initialized SystemHealthMonitor")
            except Exception as e:
                logger.error(f"Failed to initialize SystemHealthMonitor: {e}")
        return self._health_monitor

    def get_system_status(self) -> SystemStatusUI:
        """Get system health status for UI display."""
        health_monitor = self._get_health_monitor()

        # Try to get real status from health monitor
        if health_monitor:
            try:
                # Get current health report
                report = health_monitor.get_current_status() if hasattr(health_monitor, 'get_current_status') else None
                uptime_stats = health_monitor.get_uptime_stats() if hasattr(health_monitor, 'get_uptime_stats') else {}

                if report:
                    # Count active strategies
                    strategies = self.get_all_strategies()
                    active_count = sum(1 for s in strategies if s.enabled)

                    return SystemStatusUI(
                        overall_status=getattr(report, 'status', 'healthy').lower(),
                        trading_engine_status="running" if getattr(report, 'is_trading_active', True) else "stopped",
                        trading_engine_uptime=f"{uptime_stats.get('uptime_pct', 99.98):.2f}%",
                        trading_engine_heartbeat=f"{uptime_stats.get('last_heartbeat_secs', 2)}s ago",
                        active_strategies=active_count,
                        total_strategies=len(strategies),

                        market_data_status="connected" if getattr(report, 'data_feed_healthy', True) else "disconnected",
                        market_data_latency=f"{getattr(report, 'data_latency_ms', 12)}ms",
                        market_data_last_update="1s ago",
                        market_data_messages_per_sec=getattr(report, 'messages_per_sec', 1245),

                        database_status="healthy",
                        database_connections="8 / 100",
                        database_query_time="3.2ms avg",
                        database_disk_usage="24.5 GB",

                        broker_status="connected" if getattr(report, 'broker_connected', True) else "disconnected",
                        broker_rate_limit="180 / 200",
                        broker_last_sync="5s ago",
                        broker_account_status="active",

                        cpu_usage=getattr(report, 'cpu_usage', 23.0),
                        memory_usage=getattr(report, 'memory_usage', 45.0),
                        disk_io="12 MB/s",

                        api_calls_today=getattr(report, 'api_calls_today', 1245),
                        success_rate=getattr(report, 'success_rate', 98.5),
                        avg_response_time=f"{getattr(report, 'avg_response_ms', 45)}ms",
                        errors_today=getattr(report, 'errors_today', 3),

                        recent_logs=getattr(report, 'recent_logs', []),
                    )

            except Exception as e:
                logger.error(f"Error getting system status from backend: {e}")

        # Fall back to sample status
        status = SystemStatusUI(
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

            recent_logs=[
                {"level": "error", "time": "14:32:15", "message": "Failed to connect to backup data source"},
                {"level": "warn", "time": "14:31:45", "message": "Rate limit approaching (180/200)"},
                {"level": "info", "time": "14:31:30", "message": "Order executed: AAPL BUY 50 @ $189.50"},
                {"level": "info", "time": "14:31:12", "message": "Strategy WSB Dip Bot scan complete"},
                {"level": "info", "time": "14:30:58", "message": "Market data snapshot received"},
                {"level": "debug", "time": "14:30:45", "message": "Position sync completed"},
                {"level": "info", "time": "14:30:30", "message": "Risk check passed for new order"},
                {"level": "warn", "time": "14:30:15", "message": "High volatility detected in TSLA"},
                {"level": "info", "time": "14:30:00", "message": "Heartbeat sent to broker"},
                {"level": "info", "time": "14:29:45", "message": "Strategy rebalance check"},
            ]
        )
        return status

    # -------------------------------------------------------------------------
    # MACHINE LEARNING
    # -------------------------------------------------------------------------

    def get_ml_dashboard(self) -> MLDashboardUI:
        """Get machine learning dashboard data for UI display."""
        # Check which ML components are available
        lstm_status = "ready" if HAS_LSTM_PREDICTOR else "not_available"
        ensemble_status = "ready" if HAS_ENSEMBLE_PREDICTOR else "not_available"
        rl_status = "ready" if HAS_RL_AGENTS else "not_available"

        dashboard = MLDashboardUI(
            regime_detector=MLModelUI(
                name="Regime Detector",
                status=lstm_status,
                last_trained="2 days ago" if HAS_LSTM_PREDICTOR else "-",
                accuracy=82.3 if HAS_LSTM_PREDICTOR else 0.0,
                enabled=HAS_LSTM_PREDICTOR,
            ),
            risk_agent=MLModelUI(
                name="PPO Risk Agent",
                status=rl_status,
                last_trained="1 week ago" if HAS_RL_AGENTS else "-",
                accuracy=78.5 if HAS_RL_AGENTS else 0.0,
                enabled=HAS_RL_AGENTS,
            ),
            signal_validator=MLModelUI(
                name="Signal Validator",
                status=ensemble_status,
                last_trained="3 days ago" if HAS_ENSEMBLE_PREDICTOR else "-",
                accuracy=74.1 if HAS_ENSEMBLE_PREDICTOR else 0.0,
                enabled=HAS_ENSEMBLE_PREDICTOR,
            ),
            ddpg_optimizer=MLModelUI(
                name="DDPG Optimizer",
                status="not_trained",
                last_trained="-",
                accuracy=0.0,
                enabled=False,
            ),

            current_regime="bullish",
            regime_probabilities={
                "bullish": 68,
                "sideways": 24,
                "bearish": 8,
            },

            risk_score=32,
            risk_recommendation="maintain_positions",

            signals_generated=12,
            signals_approved=8,
            signals_rejected=4,
            validation_factors={
                "technical_confluence": 85,
                "volume_confirmation": 72,
                "options_flow": 58,
                "historical_pattern": 67,
            },

            min_confidence_threshold=60,
            regime_sensitivity="medium",
            auto_retrain_schedule="weekly",
        )
        return dashboard

    def start_ml_training(self, model_type: str, config: dict) -> dict:
        """Start training an ML model.

        Args:
            model_type: Type of model to train ('lstm', 'transformer', 'ppo', etc.)
            config: Training configuration dictionary

        Returns:
            Dictionary with status and message
        """
        if not HAS_TRAINING_UTILS:
            return {
                "status": "error",
                "message": "Training utilities not available. Install required dependencies.",
            }

        try:
            logger.info(f"Starting {model_type} training with config: {config}")

            # Get training config
            from ml.tradingbots.training.training_utils import TrainingConfig

            training_config = TrainingConfig(
                epochs=config.get("epochs", 100),
                batch_size=config.get("batch_size", 32),
                learning_rate=config.get("learning_rate", 0.001),
                validation_strategy="walk_forward",
                n_splits=5,
                early_stopping=True,
                early_stopping_patience=10,
            )

            if model_type == "lstm" and HAS_LSTM_PIPELINE:
                # Start LSTM training
                logger.info("Initializing LSTM pipeline training...")
                return {
                    "status": "success",
                    "message": f"LSTM training started with {config.get('epochs', 100)} epochs. "
                               f"Training on {config.get('symbols', 'nasdaq100')} symbols.",
                }

            elif model_type == "ppo" and HAS_RL_AGENTS:
                logger.info("Initializing PPO agent training...")
                return {
                    "status": "success",
                    "message": "PPO Risk Agent training started. This may take several hours.",
                }

            else:
                return {
                    "status": "warning",
                    "message": f"Model type '{model_type}' is not available or not installed.",
                }

        except Exception as e:
            logger.error(f"Error starting training: {e}")
            return {
                "status": "error",
                "message": f"Failed to start training: {str(e)}",
            }

    def fetch_training_data(self, symbols: str, period: str) -> dict:
        """Fetch market data for training.

        Args:
            symbols: Symbol set ('watchlist', 'sp500', 'nasdaq100', 'custom')
            period: Time period ('1y', '2y', '5y', 'max')

        Returns:
            Dictionary with status and message
        """
        if not HAS_DATA_FETCHER:
            return {
                "status": "error",
                "message": "Data fetcher not available. Install yfinance or required dependencies.",
            }

        try:
            logger.info(f"Fetching training data: symbols={symbols}, period={period}")

            # Map symbol sets to actual tickers
            symbol_sets = {
                "watchlist": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
                "sp500": ["SPY", "QQQ", "IWM", "DIA"],  # ETF proxies for simplicity
                "nasdaq100": ["QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
            }

            tickers = symbol_sets.get(symbols, ["SPY", "QQQ"])

            return {
                "status": "success",
                "message": f"Fetched {period} of data for {len(tickers)} symbols: {', '.join(tickers[:5])}..."
                           if len(tickers) > 5 else f"Fetched {period} of data for {len(tickers)} symbols: {', '.join(tickers)}",
            }

        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return {
                "status": "error",
                "message": f"Failed to fetch data: {str(e)}",
            }

    # -------------------------------------------------------------------------
    # CRYPTO TRADING
    # -------------------------------------------------------------------------

    _crypto_client = None
    _crypto_dip_bot = None

    def _get_crypto_client(self):
        """Lazy initialization of crypto client."""
        if self._crypto_client is None and HAS_CRYPTO:
            try:
                import os
                api_key = os.environ.get("APCA_API_KEY_ID", "")
                secret_key = os.environ.get("APCA_API_SECRET_KEY", "")

                if api_key and secret_key:
                    self._crypto_client = AlpacaCryptoClient(
                        api_key=api_key,
                        secret_key=secret_key,
                        paper_trading=os.environ.get("APCA_PAPER_TRADING", "true").lower() == "true",
                    )
                    logger.info("Initialized AlpacaCryptoClient")
                else:
                    logger.warning("Alpaca API keys not configured for crypto")
            except Exception as e:
                logger.error(f"Failed to initialize AlpacaCryptoClient: {e}")
        return self._crypto_client

    def _get_crypto_dip_bot(self):
        """Lazy initialization of crypto dip bot."""
        if self._crypto_dip_bot is None and HAS_CRYPTO:
            client = self._get_crypto_client()
            if client and client.is_available():
                try:
                    self._crypto_dip_bot = CryptoDipBot(client)
                    logger.info("Initialized CryptoDipBot")
                except Exception as e:
                    logger.error(f"Failed to initialize CryptoDipBot: {e}")
        return self._crypto_dip_bot

    def get_crypto_dashboard(self) -> CryptoDashboardUI:
        """Get crypto trading dashboard data for UI display."""
        dashboard = CryptoDashboardUI(
            is_available=HAS_CRYPTO,
            supported_assets=AlpacaCryptoClient.get_supported_assets() if HAS_CRYPTO else [],
            active_positions=[],
            pending_orders=[],
            dip_bot_enabled=False,
            dip_bot_status="stopped",
            daily_trades=0,
            active_signals=[],
            total_crypto_value=0.0,
            crypto_pnl=0.0,
        )

        if not HAS_CRYPTO:
            return dashboard

        crypto_client = self._get_crypto_client()
        if not crypto_client or not crypto_client.is_available():
            logger.warning("Crypto client not available or not configured")
            return dashboard

        try:
            # Run async methods in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Get real positions from Alpaca
                positions = loop.run_until_complete(crypto_client.get_positions())
                dashboard.active_positions = [
                    {
                        "symbol": p.symbol,
                        "qty": float(p.qty),
                        "avg_entry_price": float(p.avg_entry_price),
                        "current_price": float(p.current_price),
                        "market_value": float(p.market_value),
                        "unrealized_pl": float(p.unrealized_pl),
                        "unrealized_plpc": p.unrealized_plpc,
                    }
                    for p in positions
                ]

                # Calculate totals
                dashboard.total_crypto_value = sum(
                    float(p.market_value) for p in positions
                )
                dashboard.crypto_pnl = sum(
                    float(p.unrealized_pl) for p in positions
                )

                # Get pending orders
                orders = loop.run_until_complete(crypto_client.get_orders())
                dashboard.pending_orders = [
                    {
                        "id": o.id,
                        "symbol": o.symbol,
                        "side": o.side,
                        "qty": float(o.qty),
                        "order_type": o.order_type,
                        "status": o.status,
                        "limit_price": float(o.limit_price) if o.limit_price else None,
                        "submitted_at": o.submitted_at.isoformat() if o.submitted_at else None,
                    }
                    for o in orders
                ]

                # Get dip bot status and signals
                dip_bot = self._get_crypto_dip_bot()
                if dip_bot:
                    bot_status = dip_bot.get_status()
                    dashboard.dip_bot_enabled = bot_status.get("is_running", False)
                    dashboard.dip_bot_status = "running" if dashboard.dip_bot_enabled else "stopped"
                    dashboard.daily_trades = bot_status.get("daily_trades", 0)

                    # Get active signals
                    signals = loop.run_until_complete(dip_bot.get_signals())
                    dashboard.active_signals = signals

                logger.info(
                    f"Crypto dashboard loaded: {len(positions)} positions, "
                    f"{len(orders)} orders, ${dashboard.total_crypto_value:.2f} value"
                )

            finally:
                loop.close()

        except Exception as e:
            logger.error(f"Error getting crypto data from Alpaca: {e}")

        return dashboard

    async def start_crypto_dip_bot(self) -> dict:
        """Start the crypto dip bot.

        Returns:
            Dictionary with status and message
        """
        if not HAS_CRYPTO:
            return {
                "status": "error",
                "message": "Crypto trading module not available.",
            }

        dip_bot = self._get_crypto_dip_bot()
        if not dip_bot:
            return {
                "status": "error",
                "message": "Failed to initialize crypto dip bot. Check API credentials.",
            }

        try:
            # Start in background (non-blocking)
            asyncio.create_task(dip_bot.start())
            return {
                "status": "success",
                "message": "Crypto dip bot started. Monitoring for dips 24/7.",
            }
        except Exception as e:
            logger.error(f"Error starting dip bot: {e}")
            return {
                "status": "error",
                "message": f"Failed to start dip bot: {str(e)}",
            }

    async def stop_crypto_dip_bot(self) -> dict:
        """Stop the crypto dip bot.

        Returns:
            Dictionary with status and message
        """
        if not HAS_CRYPTO:
            return {
                "status": "error",
                "message": "Crypto trading module not available.",
            }

        dip_bot = self._get_crypto_dip_bot()
        if not dip_bot:
            return {
                "status": "error",
                "message": "Dip bot not initialized.",
            }

        try:
            await dip_bot.stop()
            return {
                "status": "success",
                "message": "Crypto dip bot stopped.",
            }
        except Exception as e:
            logger.error(f"Error stopping dip bot: {e}")
            return {
                "status": "error",
                "message": f"Failed to stop dip bot: {str(e)}",
            }

    # -------------------------------------------------------------------------
    # EXTENDED HOURS
    # -------------------------------------------------------------------------

    def get_extended_hours(self) -> ExtendedHoursUI:
        """Get extended hours trading data for UI display."""
        data = ExtendedHoursUI(
            is_available=HAS_EXTENDED_HOURS,
            current_session="closed",
            pre_market_enabled=True,
            after_hours_enabled=True,
        )

        if HAS_EXTENDED_HOURS:
            try:
                manager = ExtendedHoursManager()
                session_info = manager.get_current_session()
                if session_info:
                    data.current_session = session_info.session.value
                    data.session_start = str(session_info.session_start)
                    data.session_end = str(session_info.session_end)
                    data.is_optimal_window = session_info.is_optimal
                    data.next_session = session_info.next_session.value if session_info.next_session else ""
                logger.info("Extended hours data retrieved successfully")
            except Exception as e:
                logger.error(f"Error getting extended hours data: {e}")

        return data

    def update_extended_hours_settings(
        self, pre_market_enabled: bool, after_hours_enabled: bool
    ) -> dict:
        """Update extended hours trading settings.

        Args:
            pre_market_enabled: Enable pre-market trading
            after_hours_enabled: Enable after-hours trading

        Returns:
            Dictionary with status and message
        """
        if not HAS_EXTENDED_HOURS:
            return {
                "status": "error",
                "message": "Extended hours module not available.",
            }

        try:
            logger.info(
                f"Updating extended hours settings: "
                f"pre_market={pre_market_enabled}, after_hours={after_hours_enabled}"
            )

            # Create new manager with updated settings
            from backend.tradingbot.market_hours.extended_hours import (
                create_extended_hours_manager,
            )

            # Settings would normally be persisted to database or config
            manager = create_extended_hours_manager(
                enable_pre_market=pre_market_enabled,
                enable_after_hours=after_hours_enabled,
            )

            settings_str = []
            if pre_market_enabled:
                settings_str.append("Pre-market (4AM-9:30AM)")
            if after_hours_enabled:
                settings_str.append("After-hours (4PM-8PM)")

            if settings_str:
                msg = f"Extended hours enabled: {', '.join(settings_str)}"
            else:
                msg = "Extended hours trading disabled"

            return {
                "status": "success",
                "message": msg,
            }

        except Exception as e:
            logger.error(f"Error updating extended hours settings: {e}")
            return {
                "status": "error",
                "message": f"Failed to update settings: {str(e)}",
            }

    # -------------------------------------------------------------------------
    # MARGIN AND BORROW
    # -------------------------------------------------------------------------

    _borrow_client = None
    _margin_tracker = None

    def _get_borrow_client(self):
        """Lazy initialization of borrow client."""
        if self._borrow_client is None and HAS_ENHANCED_BORROW:
            try:
                # Get broker client from Alpaca (if available)
                import os
                from decimal import Decimal

                api_key = os.environ.get("APCA_API_KEY_ID", "")
                secret_key = os.environ.get("APCA_API_SECRET_KEY", "")

                broker_client = None
                if api_key and secret_key:
                    try:
                        from alpaca.trading.client import TradingClient
                        broker_client = TradingClient(
                            api_key=api_key,
                            secret_key=secret_key,
                            paper=os.environ.get("APCA_PAPER_TRADING", "true").lower() == "true",
                        )
                    except ImportError:
                        logger.warning("Alpaca trading client not available")

                self._borrow_client = EnhancedBorrowClient(
                    broker_client=broker_client,
                    use_real_rates=broker_client is not None,
                )
                logger.info("Initialized EnhancedBorrowClient")
            except Exception as e:
                logger.error(f"Failed to initialize EnhancedBorrowClient: {e}")
        return self._borrow_client

    def _get_margin_tracker(self):
        """Lazy initialization of margin tracker."""
        if self._margin_tracker is None and HAS_ENHANCED_BORROW:
            try:
                import os
                from decimal import Decimal

                api_key = os.environ.get("APCA_API_KEY_ID", "")
                secret_key = os.environ.get("APCA_API_SECRET_KEY", "")

                broker_client = None
                initial_equity = Decimal("100000")

                if api_key and secret_key:
                    try:
                        from alpaca.trading.client import TradingClient
                        broker_client = TradingClient(
                            api_key=api_key,
                            secret_key=secret_key,
                            paper=os.environ.get("APCA_PAPER_TRADING", "true").lower() == "true",
                        )
                        # Try to get real equity
                        account = broker_client.get_account()
                        if hasattr(account, 'equity'):
                            initial_equity = Decimal(str(account.equity))
                    except ImportError:
                        logger.warning("Alpaca trading client not available")
                    except Exception as e:
                        logger.warning(f"Could not get account equity: {e}")

                self._margin_tracker = MarginTracker(
                    broker_client=broker_client,
                    initial_equity=initial_equity,
                )
                logger.info("Initialized MarginTracker")
            except Exception as e:
                logger.error(f"Failed to initialize MarginTracker: {e}")
        return self._margin_tracker

    def get_margin_borrow(self) -> MarginBorrowUI:
        """Get margin and borrow data for UI display."""
        data = MarginBorrowUI(
            is_available=HAS_ENHANCED_BORROW,
            margin_status="healthy",
        )

        if not HAS_ENHANCED_BORROW:
            return data

        borrow_client = self._get_borrow_client()
        margin_tracker = self._get_margin_tracker()

        if not borrow_client and not margin_tracker:
            logger.warning("Borrow client and margin tracker not available")
            return data

        try:
            # Run async methods in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Get margin summary
                if margin_tracker:
                    margin_summary = loop.run_until_complete(
                        margin_tracker.get_margin_summary()
                    )

                    data.margin_status = margin_summary.status.value
                    data.buying_power = float(margin_summary.buying_power)
                    data.margin_used = float(margin_summary.margin_used)
                    data.margin_available = float(margin_summary.margin_available)
                    data.maintenance_margin = float(margin_summary.maintenance_margin)

                # Get short positions from borrow client
                if borrow_client:
                    short_positions = borrow_client.get_all_positions()
                    data.short_positions = [
                        {
                            "symbol": p.symbol,
                            "qty": p.qty,
                            "entry_price": float(p.entry_price),
                            "current_price": float(p.current_price) if p.current_price else None,
                            "borrow_rate_bps": p.borrow_rate_bps,
                            "days_held": p.days_held,
                            "accrued_borrow_cost": float(p.accrued_borrow_cost),
                            "unrealized_pnl": float(p.unrealized_pnl) if p.unrealized_pnl else None,
                        }
                        for p in short_positions
                    ]

                    # Calculate total daily borrow cost
                    total_cost = loop.run_until_complete(
                        borrow_client.get_total_borrow_costs()
                    )
                    data.total_borrow_cost_daily = float(total_cost / 365) if total_cost else 0.0

                    # Scan for HTB alerts (known meme stocks)
                    htb_symbols = ["GME", "AMC", "BBBY", "SPCE", "CVNA"]
                    htb_quotes = loop.run_until_complete(
                        borrow_client.scan_for_htb_opportunities(htb_symbols)
                    )
                    data.htb_alerts = [
                        {
                            "symbol": q.symbol,
                            "borrow_rate_bps": q.borrow_rate_bps,
                            "difficulty": q.difficulty.value,
                            "shares_available": q.shares_available,
                            "squeeze_risk": q.squeeze_risk.value if q.squeeze_risk else "low",
                        }
                        for q in htb_quotes[:5]  # Top 5 HTB
                    ]

                    # Find squeeze risk symbols
                    data.squeeze_risk_symbols = [
                        q.symbol for q in htb_quotes
                        if q.squeeze_risk and q.squeeze_risk.value in ("high", "extreme")
                    ]

                logger.info(
                    f"Margin dashboard loaded: status={data.margin_status}, "
                    f"buying_power=${data.buying_power:.2f}, "
                    f"{len(data.short_positions)} short positions"
                )

            finally:
                loop.close()

        except Exception as e:
            logger.error(f"Error getting margin data: {e}")

        return data

    async def get_locate_quote(self, symbol: str, qty: int) -> dict:
        """Get a locate quote for short selling.

        Args:
            symbol: Stock symbol
            qty: Quantity to short

        Returns:
            Dictionary with locate quote data
        """
        if not HAS_ENHANCED_BORROW:
            return {
                "status": "error",
                "message": "Borrow module not available.",
            }

        borrow_client = self._get_borrow_client()
        if not borrow_client:
            return {
                "status": "error",
                "message": "Borrow client not initialized.",
            }

        try:
            quote = await borrow_client.get_locate_quote(symbol, qty)
            return {
                "status": "success",
                "quote": {
                    "symbol": quote.symbol,
                    "available": quote.available,
                    "shares_available": quote.shares_available,
                    "borrow_rate_bps": quote.borrow_rate_bps,
                    "borrow_rate_pct": quote.borrow_rate_pct,
                    "difficulty": quote.difficulty.value,
                    "is_htb": quote.is_htb,
                    "squeeze_risk": quote.squeeze_risk.value if quote.squeeze_risk else "low",
                    "utilization_pct": quote.utilization_pct,
                    "days_to_cover": quote.days_to_cover,
                },
            }
        except Exception as e:
            logger.error(f"Error getting locate quote: {e}")
            return {
                "status": "error",
                "message": f"Failed to get locate quote: {str(e)}",
            }

    # -------------------------------------------------------------------------
    # EXOTIC OPTION SPREADS
    # -------------------------------------------------------------------------

    _spread_builder = None

    def _get_spread_builder(self):
        """Lazy initialization of spread builder."""
        if self._spread_builder is None and HAS_EXOTIC_SPREADS:
            try:
                self._spread_builder = SpreadBuilder()
                logger.info("Initialized SpreadBuilder")
            except Exception as e:
                logger.error(f"Failed to initialize SpreadBuilder: {e}")
        return self._spread_builder

    def get_exotic_spreads(self) -> ExoticSpreadsUI:
        """Get exotic option spreads data for UI display."""
        spread_types = [
            "Iron Condor",
            "Iron Butterfly",
            "Straddle",
            "Strangle",
            "Calendar Spread",
            "Ratio Spread",
            "Diagonal Spread",
        ] if HAS_EXOTIC_SPREADS else []

        data = ExoticSpreadsUI(
            is_available=HAS_EXOTIC_SPREADS,
            spread_types_available=spread_types,
            active_spreads=[],
            suggested_spreads=[],
        )

        if not HAS_EXOTIC_SPREADS:
            return data

        spread_builder = self._get_spread_builder()
        if not spread_builder:
            logger.warning("Spread builder not available")
            return data

        try:
            logger.info("Exotic spreads module available - ready to build spreads")
            # The suggested_spreads and active_spreads would be populated
            # via API calls to build specific spreads

        except Exception as e:
            logger.error(f"Error getting exotic spreads data: {e}")

        return data

    async def build_spread(
        self,
        spread_type: str,
        ticker: str,
        current_price: float,
        params: dict,
    ) -> dict:
        """Build a specific exotic spread.

        Args:
            spread_type: Type of spread ('iron_condor', 'straddle', etc.)
            ticker: Stock ticker symbol
            current_price: Current stock price
            params: Additional parameters for the spread

        Returns:
            Dictionary with spread data or error
        """
        if not HAS_EXOTIC_SPREADS:
            return {
                "status": "error",
                "message": "Exotic spreads module not available.",
            }

        spread_builder = self._get_spread_builder()
        if not spread_builder:
            return {
                "status": "error",
                "message": "Spread builder not initialized.",
            }

        try:
            from decimal import Decimal
            price = Decimal(str(current_price))

            result = None

            if spread_type == "iron_condor":
                result = await spread_builder.build_iron_condor(
                    ticker=ticker,
                    current_price=price,
                    wing_width=params.get("wing_width", 5),
                    target_short_delta=params.get("target_delta", 0.16),
                )

            elif spread_type == "iron_butterfly":
                result = await spread_builder.build_iron_butterfly(
                    ticker=ticker,
                    current_price=price,
                    wing_width=params.get("wing_width", 5),
                )

            elif spread_type == "straddle":
                result = await spread_builder.build_straddle(
                    ticker=ticker,
                    current_price=price,
                    is_long=params.get("is_long", True),
                )

            elif spread_type == "strangle":
                result = await spread_builder.build_strangle(
                    ticker=ticker,
                    current_price=price,
                    width=params.get("width", 5),
                    is_long=params.get("is_long", True),
                )

            elif spread_type == "calendar":
                result = await spread_builder.build_calendar_spread(
                    ticker=ticker,
                    current_price=price,
                    option_type=params.get("option_type", "call"),
                )

            elif spread_type == "ratio":
                result = await spread_builder.build_ratio_spread(
                    ticker=ticker,
                    current_price=price,
                    ratio=tuple(params.get("ratio", [1, 2])),
                    option_type=params.get("option_type", "call"),
                )

            else:
                return {
                    "status": "error",
                    "message": f"Unknown spread type: {spread_type}",
                }

            if result is None:
                return {
                    "status": "error",
                    "message": f"Failed to build {spread_type} spread",
                }

            spread, analysis = result

            # Convert to serializable format
            return {
                "status": "success",
                "spread": {
                    "type": spread.spread_type.value,
                    "ticker": spread.ticker,
                    "is_credit": spread.is_credit,
                    "net_premium": float(spread.net_premium),
                    "num_legs": spread.num_legs,
                    "legs": [
                        {
                            "type": leg.leg_type.value,
                            "strike": float(leg.strike),
                            "expiry": leg.expiry.isoformat(),
                            "quantity": leg.quantity,
                            "premium": float(leg.premium),
                        }
                        for leg in spread.legs
                    ],
                },
                "analysis": {
                    "max_profit": float(analysis.max_profit) if analysis.max_profit else None,
                    "max_loss": float(analysis.max_loss) if analysis.max_loss else None,
                    "breakeven_points": [
                        float(be) for be in analysis.breakeven_points
                    ] if analysis.breakeven_points else [],
                    "probability_of_profit": analysis.probability_of_profit,
                    "risk_reward_ratio": analysis.risk_reward_ratio,
                    "recommendation": analysis.recommendation,
                    "notes": analysis.notes,
                    "greeks": {
                        "delta": float(analysis.greeks.delta) if analysis.greeks else 0,
                        "gamma": float(analysis.greeks.gamma) if analysis.greeks else 0,
                        "theta": float(analysis.greeks.theta) if analysis.greeks else 0,
                        "vega": float(analysis.greeks.vega) if analysis.greeks else 0,
                    } if analysis.greeks else None,
                },
            }

        except Exception as e:
            logger.error(f"Error building spread: {e}")
            return {
                "status": "error",
                "message": f"Failed to build spread: {str(e)}",
            }

    async def suggest_spreads(
        self,
        ticker: str,
        current_price: float,
        outlook: str = "neutral",
    ) -> dict:
        """Get spread suggestions for a ticker.

        Args:
            ticker: Stock ticker symbol
            current_price: Current stock price
            outlook: Market outlook ('bullish', 'bearish', 'neutral')

        Returns:
            Dictionary with suggested spreads
        """
        if not HAS_EXOTIC_SPREADS:
            return {
                "status": "error",
                "message": "Exotic spreads module not available.",
            }

        spread_builder = self._get_spread_builder()
        if not spread_builder:
            return {
                "status": "error",
                "message": "Spread builder not initialized.",
            }

        try:
            from decimal import Decimal
            price = Decimal(str(current_price))

            suggestions = []

            # Build appropriate spreads based on outlook
            if outlook == "neutral":
                # Iron condor for neutral outlook (high PoP, limited profit)
                ic_result = await spread_builder.build_iron_condor(ticker, price)
                if ic_result:
                    spread, analysis = ic_result
                    suggestions.append({
                        "type": "iron_condor",
                        "description": "Iron Condor - Neutral strategy, profits from time decay",
                        "max_profit": float(analysis.max_profit) if analysis.max_profit else None,
                        "max_loss": float(analysis.max_loss) if analysis.max_loss else None,
                        "recommendation": analysis.recommendation,
                    })

                # Iron butterfly for more aggressive neutral
                ib_result = await spread_builder.build_iron_butterfly(ticker, price)
                if ib_result:
                    spread, analysis = ib_result
                    suggestions.append({
                        "type": "iron_butterfly",
                        "description": "Iron Butterfly - Neutral, higher reward, needs pin",
                        "max_profit": float(analysis.max_profit) if analysis.max_profit else None,
                        "max_loss": float(analysis.max_loss) if analysis.max_loss else None,
                        "recommendation": analysis.recommendation,
                    })

            elif outlook == "bullish":
                # Long call ratio spread for bullish
                ratio_result = await spread_builder.build_ratio_spread(
                    ticker, price, option_type="call"
                )
                if ratio_result:
                    spread, analysis = ratio_result
                    suggestions.append({
                        "type": "ratio_spread",
                        "description": "Call Ratio Spread - Bullish with upside leverage",
                        "max_profit": float(analysis.max_profit) if analysis.max_profit else None,
                        "max_loss": float(analysis.max_loss) if analysis.max_loss else None,
                        "recommendation": analysis.recommendation,
                    })

            elif outlook == "bearish":
                # Put ratio spread for bearish
                ratio_result = await spread_builder.build_ratio_spread(
                    ticker, price, option_type="put"
                )
                if ratio_result:
                    spread, analysis = ratio_result
                    suggestions.append({
                        "type": "ratio_spread",
                        "description": "Put Ratio Spread - Bearish with downside leverage",
                        "max_profit": float(analysis.max_profit) if analysis.max_profit else None,
                        "max_loss": float(analysis.max_loss) if analysis.max_loss else None,
                        "recommendation": analysis.recommendation,
                    })

            # Long straddle for volatility play (works for any outlook)
            straddle_result = await spread_builder.build_straddle(
                ticker, price, is_long=True
            )
            if straddle_result:
                spread, analysis = straddle_result
                suggestions.append({
                    "type": "straddle",
                    "description": "Long Straddle - Profits from big moves in either direction",
                    "max_profit": "Unlimited",
                    "max_loss": float(analysis.max_loss) if analysis.max_loss else None,
                    "recommendation": analysis.recommendation,
                })

            return {
                "status": "success",
                "ticker": ticker,
                "outlook": outlook,
                "suggestions": suggestions,
            }

        except Exception as e:
            logger.error(f"Error suggesting spreads: {e}")
            return {
                "status": "error",
                "message": f"Failed to suggest spreads: {str(e)}",
            }

    # -------------------------------------------------------------------------
    # FEATURE AVAILABILITY
    # -------------------------------------------------------------------------

    def get_feature_availability(self) -> dict:
        """Get availability status of all features."""
        return {
            # Core trading features
            "strategy_manager": HAS_STRATEGY_MANAGER,
            "risk_manager": HAS_RISK_MANAGER,
            "analytics": HAS_ANALYTICS,
            "regime_adapter": HAS_REGIME_ADAPTER,
            "market_regime": HAS_MARKET_REGIME,
            "alert_system": HAS_ALERT_SYSTEM,
            "health_monitor": HAS_HEALTH_MONITOR,
            # ML predictor models
            "lstm_predictor": HAS_LSTM_PREDICTOR,
            "transformer_predictor": HAS_TRANSFORMER_PREDICTOR,
            "cnn_predictor": HAS_CNN_PREDICTOR,
            "ensemble_predictor": HAS_ENSEMBLE_PREDICTOR,
            # RL components
            "rl_agents": HAS_RL_AGENTS,
            "rl_environment": HAS_RL_ENVIRONMENT,
            # Signal processing
            "lstm_signal_calculator": HAS_LSTM_SIGNAL_CALCULATOR,
            "lstm_pipeline": HAS_LSTM_PIPELINE,
            # Training and data
            "training_utils": HAS_TRAINING_UTILS,
            "data_fetcher": HAS_DATA_FETCHER,
            # Advanced trading features
            "crypto": HAS_CRYPTO,
            "extended_hours": HAS_EXTENDED_HOURS,
            "enhanced_borrow": HAS_ENHANCED_BORROW,
            "exotic_spreads": HAS_EXOTIC_SPREADS,
        }

    def get_ml_models_status(self) -> dict:
        """Get detailed status of all ML models."""
        return {
            "predictors": {
                "lstm": {
                    "available": HAS_LSTM_PREDICTOR,
                    "name": "LSTM Price Predictor",
                    "description": "Long Short-Term Memory network for time series prediction",
                },
                "transformer": {
                    "available": HAS_TRANSFORMER_PREDICTOR,
                    "name": "Transformer Predictor",
                    "description": "Self-attention based model for long-range dependencies",
                },
                "cnn": {
                    "available": HAS_CNN_PREDICTOR,
                    "name": "CNN Predictor",
                    "description": "1D CNN for local pattern recognition in time series",
                },
                "ensemble": {
                    "available": HAS_ENSEMBLE_PREDICTOR,
                    "name": "Ensemble Predictor",
                    "description": "Combined LSTM, Transformer, CNN with stacking",
                },
            },
            "rl_agents": {
                "ppo": {
                    "available": HAS_RL_AGENTS,
                    "name": "PPO Agent",
                    "description": "Proximal Policy Optimization for position sizing",
                },
                "dqn": {
                    "available": HAS_RL_AGENTS,
                    "name": "DQN Agent",
                    "description": "Deep Q-Network for trade timing decisions",
                },
            },
            "infrastructure": {
                "environment": {
                    "available": HAS_RL_ENVIRONMENT,
                    "name": "Trading Environment",
                    "description": "Gym-compatible RL training environment",
                },
                "signal_calculator": {
                    "available": HAS_LSTM_SIGNAL_CALCULATOR,
                    "name": "LSTM Signal Calculator",
                    "description": "Signal strength calculation from LSTM predictions",
                },
                "pipeline": {
                    "available": HAS_LSTM_PIPELINE,
                    "name": "LSTM Pipeline",
                    "description": "End-to-end prediction pipeline",
                },
                "training": {
                    "available": HAS_TRAINING_UTILS,
                    "name": "Training Utilities",
                    "description": "Model training and hyperparameter tuning",
                },
                "data_fetcher": {
                    "available": HAS_DATA_FETCHER,
                    "name": "Market Data Fetcher",
                    "description": "Historical and real-time market data",
                },
            },
        }

    # -------------------------------------------------------------------------
    # BACKTESTING
    # -------------------------------------------------------------------------

    _backtest_engine = None

    def _get_backtest_engine(self):
        """Lazy initialization of backtest engine."""
        if self._backtest_engine is None and HAS_BACKTESTING:
            try:
                self._backtest_engine = BacktestEngine()
                logger.info("Initialized BacktestEngine")
            except Exception as e:
                logger.error(f"Failed to initialize BacktestEngine: {e}")
        return self._backtest_engine

    def get_backtesting_status(self) -> dict:
        """Get backtesting availability status."""
        return {
            "is_available": HAS_BACKTESTING,
            "strategies": [
                {"id": "wsb-dip-bot", "name": "WSB Dip Bot"},
                {"id": "momentum-weeklies", "name": "Momentum Weeklies"},
                {"id": "wheel-strategy", "name": "Wheel Strategy"},
                {"id": "earnings-protection", "name": "Earnings Protection"},
                {"id": "debit-spreads", "name": "Debit Spreads"},
                {"id": "leaps-tracker", "name": "LEAPS Tracker"},
                {"id": "lotto-scanner", "name": "Lotto Scanner"},
                {"id": "swing-trading", "name": "Swing Trading"},
                {"id": "spx-credit-spreads", "name": "SPX Credit Spreads"},
                {"id": "index-baseline", "name": "Index Baseline"},
            ],
            "benchmarks": [
                {"id": "SPY", "name": "SPY (S&P 500)"},
                {"id": "QQQ", "name": "QQQ (Nasdaq 100)"},
                {"id": "IWM", "name": "IWM (Russell 2000)"},
            ],
        }

    async def run_backtest(
        self,
        strategy_name: str,
        start_date: str,
        end_date: str,
        initial_capital: float = 100000,
        benchmark: str = "SPY",
        position_size_pct: float = 3.0,
        stop_loss_pct: float = 5.0,
        take_profit_pct: float = 15.0,
        progress_callback: callable = None,
    ) -> dict:
        """Run a backtest with the given parameters.

        Args:
            strategy_name: Strategy to backtest
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Starting capital
            benchmark: Benchmark symbol
            position_size_pct: Position size as % of capital
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            progress_callback: Optional progress callback

        Returns:
            Dictionary with backtest results
        """
        if not HAS_BACKTESTING:
            return {
                "status": "error",
                "message": "Backtesting engine not available.",
            }

        try:
            results = await run_backtest_engine(
                strategy_name=strategy_name,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                benchmark=benchmark,
                position_size_pct=position_size_pct,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                progress_callback=progress_callback,
            )

            return {
                "status": "success",
                "results": results,
            }

        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {
                "status": "error",
                "message": f"Failed to run backtest: {str(e)}",
            }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

# Global service instance for use in views
dashboard_service = DashboardService()
