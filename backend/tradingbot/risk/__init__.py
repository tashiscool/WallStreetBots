"""
Advanced Risk Management Module - 2025 Implementation
Sophisticated risk models for algorithmic trading
"""

from .advanced_var_engine import AdvancedVaREngine, VaRResult, VaRSuite
from .stress_testing_engine import StressTesting2025, StressTestReport, StressScenario
from .ml_risk_predictor import MLRiskPredictor, RiskPrediction, VolatilityForecast
from .risk_dashboard import RiskDashboard2025, RiskSummary, RiskAlert
from .risk_integration_manager import RiskIntegrationManager, RiskLimits, RiskMetrics
from .database_schema import RiskDatabaseManager

# Import complete risk engine utilities
try: 
    from .risk_engine_complete import RiskEngine, RiskMetrics as CompleteRiskMetrics
except ImportError: 
    # Fallback if complete engine not available
    pass

# Month 5 - 6: Advanced Features and Automation
try: 
    from .advanced_ml_risk_agents import (
        MultiAgentRiskCoordinator, RiskEnvironment, RiskState, RiskActionType,
        PPORiskAgent, DDPGRiskAgent
    )
    from .multi_asset_risk_manager import (
        MultiAssetRiskManager, AssetClass, RiskFactor
    )
    from .regulatory_compliance_manager import (
        RegulatoryComplianceManager, RegulatoryAuthority, ComplianceStatus, ComplianceRule
    )
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError: 
    ADVANCED_FEATURES_AVAILABLE = False

__all__ = [
    'AdvancedVaREngine',
    'VaRResult', 
    'VaRSuite',
    'StressTesting2025',
    'StressTestReport',
    'StressScenario',
    'MLRiskPredictor',
    'RiskPrediction',
    'VolatilityForecast',
    'RiskDashboard2025',
    'RiskSummary',
    'RiskAlert',
    'RiskIntegrationManager',
    'RiskLimits',
    'RiskMetrics',
    'RiskDatabaseManager'
]

# Add advanced features if available
if ADVANCED_FEATURES_AVAILABLE: 
    __all__.extend([
        'MultiAgentRiskCoordinator',
        'RiskEnvironment',
        'RiskState', 
        'RiskActionType',
        'PPORiskAgent',
        'DDPGRiskAgent',
        'MultiAssetRiskManager',
        'AssetClass',
        'RiskFactor',
        'RegulatoryComplianceManager',
        'RegulatoryAuthority',
        'ComplianceStatus',
        'ComplianceRule'
    ])

# Version information
__version__ = "2025.1.0"
__author__ = "WallStreetBots Risk Management Team"
__description__ = "Advanced risk management models for algorithmic trading"