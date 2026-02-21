"""Advanced Risk Management Module-2025 Implementation
Sophisticated risk models for algorithmic trading.
"""

# Core imports that are always available
try:
    from .database_schema import RiskDatabaseManager
except ImportError:
    RiskDatabaseManager = None

# Circuit breaker (ported from Polymarket-Kalshi arbitrage bot)
# Note: There are two circuit breaker implementations:
# - risk.circuit_breaker.CircuitBreaker: Basic, thread-safe, position/loss/error tracking
# - risk.monitoring.circuit_breaker.CircuitBreaker: Production-grade with VIX, DB persistence
# Both expose can_trade() for compatibility. Use MonitoringCircuitBreaker for production.
try:
    from .circuit_breaker import (
        TripReason,
        CircuitBreakerConfig,
        CircuitBreakerState,
        CircuitBreaker,
        get_circuit_breaker,
        configure_circuit_breaker,
    )
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    TripReason = CircuitBreakerConfig = CircuitBreakerState = None
    CircuitBreaker = get_circuit_breaker = configure_circuit_breaker = None
    CIRCUIT_BREAKER_AVAILABLE = False

try:
    from .monitoring.circuit_breaker import (
        CircuitBreaker as MonitoringCircuitBreaker,
        BreakerLimits,
        VIXTripLevel,
        VIXBreakerConfig,
    )
    MONITORING_BREAKER_AVAILABLE = True
except ImportError:
    MonitoringCircuitBreaker = BreakerLimits = VIXTripLevel = VIXBreakerConfig = None
    MONITORING_BREAKER_AVAILABLE = False

try:
    from .ml_risk_predictor import MLRiskPredictor, RiskPrediction, VolatilityForecast
except ImportError:
    MLRiskPredictor = RiskPrediction = VolatilityForecast = None

# Optional imports with fallbacks
try:
    from .engines.advanced_var_engine import AdvancedVaREngine, VaRResult, VaRSuite
    from .engines import engine
except ImportError:
    AdvancedVaREngine = VaRResult = VaRSuite = None
    engine = None

try:
    from .monitoring.risk_dashboard import RiskAlert, RiskDashboard2025, RiskSummary
except ImportError:
    RiskAlert = RiskDashboard2025 = RiskSummary = None

try:
    from .engines.stress_testing_engine import StressScenario, StressTesting2025, StressTestReport
except ImportError:
    StressScenario = StressTesting2025 = StressTestReport = None

# Risk integration manager
RiskIntegrationManager = RiskLimits = RiskMetrics = None
try:
    from .managers.risk_integration_manager import RiskIntegrationManager, RiskLimits, RiskMetrics
except ImportError:
    pass

# Import complete risk engine utilities
try:
    from .engines.risk_engine_complete import RiskEngine
    from .engines.risk_engine_complete import RiskMetrics as CompleteRiskMetrics
except ImportError:
    # Fallback if complete engine not available
    pass

# Import production-ready risk engine
try:
    from .engines.engine import (
        RiskEngine as ProductionRiskEngine,
        RiskLimits as ProductionRiskLimits,
    )

    PRODUCTION_RISK_AVAILABLE = True
except ImportError:
    PRODUCTION_RISK_AVAILABLE = False

# Month 5 - 6: Advanced Features and Automation
try:
    from .advanced_ml_risk_agents import (
        DDPGRiskAgent,
        MultiAgentRiskCoordinator,
        PPORiskAgent,
        RiskActionType,
        RiskEnvironment,
        RiskState,
    )
    from .managers.multi_asset_risk_manager import AssetClass, MultiAssetRiskManager, RiskFactor
    from .compliance.regulatory_compliance_manager import (
        ComplianceRule,
        ComplianceStatus,
        RegulatoryAuthority,
        RegulatoryComplianceManager,
    )

    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False

__all__ = [
    "AdvancedVaREngine",
    "MLRiskPredictor",
    "RiskAlert",
    "RiskDashboard2025",
    "RiskDatabaseManager",
    "RiskIntegrationManager",
    "RiskLimits",
    "RiskMetrics",
    "RiskPrediction",
    "RiskSummary",
    "StressScenario",
    "StressTestReport",
    "StressTesting2025",
    "VaRResult",
    "VaRSuite",
    "VolatilityForecast",
]

# Add circuit breaker if available (from Polymarket-Kalshi arbitrage bot)
if CIRCUIT_BREAKER_AVAILABLE:
    __all__.extend([
        "CircuitBreaker",
        "CircuitBreakerConfig",
        "CircuitBreakerState",
        "TripReason",
        "configure_circuit_breaker",
        "get_circuit_breaker",
    ])

# Add advanced features if available
if ADVANCED_FEATURES_AVAILABLE:
    __all__.extend(
        [
            "AssetClass",
            "ComplianceRule",
            "ComplianceStatus",
            "DDPGRiskAgent",
            "MultiAgentRiskCoordinator",
            "MultiAssetRiskManager",
            "PPORiskAgent",
            "RegulatoryAuthority",
            "RegulatoryComplianceManager",
            "RiskActionType",
            "RiskEnvironment",
            "RiskFactor",
            "RiskState",
        ]
    )

# Pre-trade validation engine (ported from Nautilus Trader)
try:
    from .validation_engine import (
        RiskValidationEngine,
        TradingState,
        ValidationResult,
    )
    VALIDATION_ENGINE_AVAILABLE = True
except ImportError:
    RiskValidationEngine = TradingState = ValidationResult = None
    VALIDATION_ENGINE_AVAILABLE = False

# Import portfolio construction rules
try:
    from .portfolio_rules import (
        sector_cap_check,
        simple_corr_guard,
    )

    PORTFOLIO_RULES_AVAILABLE = True
except ImportError:
    PORTFOLIO_RULES_AVAILABLE = False

# Add portfolio rules if available
if PORTFOLIO_RULES_AVAILABLE:
    __all__.extend(
        [
            "sector_cap_check",
            "simple_corr_guard",
        ]
    )

# Version information
__version__ = "2025.1.0"
__author__ = "WallStreetBots Risk Management Team"
__description__ = "Advanced risk management models for algorithmic trading"
