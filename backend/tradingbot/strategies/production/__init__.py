"""Production strategy wrappers.

Submodules are lazy-loaded so importing one strategy does not require every
production strategy dependency to be importable.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "production_debit_spreads",
    "production_earnings_protection",
    "production_index_baseline",
    "production_leaps_tracker",
    "production_lotto_scanner",
    "production_momentum_weeklies",
    "production_narrative_rerate",
    "production_spx_credit_spreads",
    "production_swing_trading",
    "production_wheel_strategy",
    "production_wsb_dip_bot",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
