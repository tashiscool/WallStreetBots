"""Advanced Risk Management Module-2025 Implementation
Sophisticated risk models for algorithmic trading.
"""

from .advanced_var_engine import AdvancedVaREngine, VaRResult, VaRSuite
from .database_schema import RiskDatabaseManager
from .ml_risk_predictor import MLRiskPredictor, RiskPrediction, VolatilityForecast
from .risk_dashboard import RiskAlert, RiskDashboard2025, RiskSummary
from .risk_integration_manager import RiskIntegrationManager, RiskLimits, RiskMetrics
from .stress_testing_engine import StressScenario, StressTesting2025, StressTestReport

# Import complete risk engine utilities
try:
    from .risk_engine_complete import RiskEngine
    from .risk_engine_complete import RiskMetrics as CompleteRiskMetrics
except ImportError:
    # Fallback if complete engine not available
    pass

# Import production-ready risk engine
try:
    from .engine import (
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
    from .multi_asset_risk_manager import AssetClass, MultiAssetRiskManager, RiskFactor
    from .regulatory_compliance_manager import (
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
