"""Borrow and locate module for WallStreetBots.

Provides short selling support:
- Locate quotes and borrow rates
- Dynamic HTB tracking
- Margin requirement calculation
- Short squeeze risk detection
"""

from .client import (
    BorrowClient,
    LocateQuote,
    guard_can_short,
)
from .enhanced_borrow_client import (
    EnhancedBorrowClient,
    EnhancedLocateQuote,
    BorrowDifficulty,
    ShortSqueezeRisk,
    ShortPosition,
    BorrowRateHistory,
    create_enhanced_borrow_client,
)
from .margin_tracker import (
    MarginTracker,
    MarginSummary,
    MarginRequirement,
    MarginStatus,
    MarginCallInfo,
    PositionType,
    create_margin_tracker,
)

__all__ = [
    # Basic client
    "BorrowClient",
    "LocateQuote",
    "guard_can_short",
    # Enhanced borrow
    "EnhancedBorrowClient",
    "EnhancedLocateQuote",
    "BorrowDifficulty",
    "ShortSqueezeRisk",
    "ShortPosition",
    "BorrowRateHistory",
    "create_enhanced_borrow_client",
    # Margin tracking
    "MarginTracker",
    "MarginSummary",
    "MarginRequirement",
    "MarginStatus",
    "MarginCallInfo",
    "PositionType",
    "create_margin_tracker",
]
