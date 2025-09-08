"""
Advanced Risk Management Module - 2025 Implementation
Sophisticated risk models for algorithmic trading
"""

from .advanced_var_engine import AdvancedVaREngine, VaRResult, VaRSuite
from .stress_testing_engine import StressTesting2025, StressTestReport, StressScenario
from .ml_risk_predictor import MLRiskPredictor, RiskPrediction, VolatilityForecast
from .risk_dashboard import RiskDashboard2025, RiskSummary, RiskAlert

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
    'RiskAlert'
]

# Version information
__version__ = "2025.1.0"
__author__ = "WallStreetBots Risk Management Team"
__description__ = "Advanced risk management models for algorithmic trading"