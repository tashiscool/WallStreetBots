"""
Production Trading Strategies

Production-ready implementations of WSB-style trading strategies:
- ProductionWSBDipBot: Dip-after-run strategy with live execution
- ProductionEarningsProtection: Earnings protection strategies
- ProductionIndexBaseline: Baseline performance tracking
- ProductionWheelStrategy: Automated premium selling wheel strategy
- ProductionMomentumWeeklies: Intraday momentum and reversal patterns for weekly options
- ProductionDebitSpreads: Call spreads with reduced theta/IV risk
- ProductionLEAPSTracker: Long-term secular growth trends with systematic profit-taking
- ProductionSwingTrading: Fast profit-taking swing trades with same-day exit discipline
- ProductionSPXCreditSpreads: WSB-style 0DTE/short-term credit spreads with defined risk
"""

from .production_wsb_dip_bot import ProductionWSBDipBot
from .production_earnings_protection import ProductionEarningsProtection
from .production_index_baseline import ProductionIndexBaseline
from .production_wheel_strategy import ProductionWheelStrategy
from .production_momentum_weeklies import ProductionMomentumWeeklies
from .production_debit_spreads import ProductionDebitSpreads
from .production_leaps_tracker import ProductionLEAPSTracker
from .production_swing_trading import ProductionSwingTrading
from .production_spx_credit_spreads import ProductionSPXCreditSpreads

__all__ = [
    'ProductionWSBDipBot',
    'ProductionEarningsProtection',
    'ProductionIndexBaseline',
    'ProductionWheelStrategy',
    'ProductionMomentumWeeklies',
    'ProductionDebitSpreads',
    'ProductionLEAPSTracker',
    'ProductionSwingTrading',
    'ProductionSPXCreditSpreads',
]
