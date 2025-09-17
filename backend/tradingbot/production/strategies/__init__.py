"""Production strategy implementations."""

# Import strategies with fallbacks for any that might not exist
try:
    from ...strategies.production.production_wsb_dip_bot import ProductionWSBDipBot
except ImportError:
    ProductionWSBDipBot = None

try:
    from ...strategies.production.production_wheel_strategy import ProductionWheelStrategy
except ImportError:
    ProductionWheelStrategy = None

try:
    from ...strategies.production.production_swing_trading import ProductionSwingTrading
except ImportError:
    ProductionSwingTrading = None

try:
    from ...strategies.production.production_momentum_weeklies import ProductionMomentumWeeklies
except ImportError:
    ProductionMomentumWeeklies = None

try:
    from ...strategies.production.production_lotto_scanner import ProductionLottoScanner
except ImportError:
    ProductionLottoScanner = None

try:
    from ...strategies.production.production_leaps_tracker import ProductionLEAPSTracker
except ImportError:
    ProductionLEAPSTracker = None

try:
    from ...strategies.production.production_index_baseline import ProductionIndexBaseline
except ImportError:
    ProductionIndexBaseline = None

try:
    from ...strategies.production.production_earnings_protection import ProductionEarningsProtection
except ImportError:
    ProductionEarningsProtection = None

try:
    from ...strategies.production.production_debit_spreads import ProductionDebitSpreads
except ImportError:
    ProductionDebitSpreads = None

try:
    from ...strategies.production.production_spx_credit_spreads import ProductionSPXCreditSpreads
except ImportError:
    ProductionSPXCreditSpreads = None

__all__ = [
    "ProductionDebitSpreads",
    "ProductionEarningsProtection",
    "ProductionIndexBaseline",
    "ProductionLEAPSTracker",
    "ProductionLottoScanner",
    "ProductionMomentumWeeklies",
    "ProductionSPXCreditSpreads",
    "ProductionSwingTrading",
    "ProductionWSBDipBot",
    "ProductionWheelStrategy",
]