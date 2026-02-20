"""
Protection Plugins

Circuit breaker protections inspired by freqtrade.
Automatically stop or reduce trading based on various conditions.
"""

from .base import IProtection, ProtectionConfig
from .cooldown import CooldownProtection
from .stoploss_guard import StoplossGuardProtection
from .max_drawdown import MaxDrawdownProtection
from .low_profit_pairs import LowProfitPairsProtection

__all__ = [
    'CooldownProtection',
    'IProtection',
    'LowProfitPairsProtection',
    'MaxDrawdownProtection',
    'ProtectionConfig',
    'StoplossGuardProtection',
]
