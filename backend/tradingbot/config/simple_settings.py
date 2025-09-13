# backend/tradingbot/config/simple_settings.py
"""
Simple configuration without external dependencies
Fallback for when pydantic is not available
"""

from __future__ import annotations
import os
from typing import Literal

StrategyProfile = Literal["research_2024", "wsb_2025"]


class AppSettings:
    def __init__(self):
        # Broker/API
        self.alpaca_api_key = os.getenv("ALPACA_API_KEY", "")
        self.alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        self.alpaca_paper = os.getenv("ALPACA_PAPER", "True").lower() == "true"

        # Profile
        self.profile = os.getenv("WSB_PROFILE", "research_2024")

        # Risk
        self.max_total_risk = float(os.getenv("MAX_TOTAL_RISK", "0.30"))
        self.max_position_size = float(os.getenv("MAX_POSITION_SIZE", "0.10"))

        # Analytics/Regime
        self.enable_advanced_analytics = (
            os.getenv("ENABLE_ADV_ANALYTICS", "True").lower() == "true"
        )
        self.enable_market_regime_adaptation = (
            os.getenv("ENABLE_REGIME_ADAPT", "True").lower() == "true"
        )
        self.analytics_update_interval = int(
            os.getenv("ANALYTICS_UPDATE_INTERVAL", "3600")
        )
        self.regime_adaptation_interval = int(
            os.getenv("REGIME_ADAPTATION_INTERVAL", "1800")
        )

        # Data
        self.use_cache = os.getenv("DATA_USE_CACHE", "True").lower() == "true"
        self.data_cache_path = os.getenv("DATA_CACHE_PATH", "./.cache")

        # Safety
        self.dry_run = os.getenv("DRY_RUN", "True").lower() == "true"

        # Validate critical settings
        self._validate()

    def _validate(self):
        """Basic validation without pydantic"""
        errors = []

        if not self.alpaca_api_key or "your_" in self.alpaca_api_key.lower():
            errors.append("ALPACA_API_KEY is missing or looks like a placeholder")

        if not self.alpaca_secret_key or "your_" in self.alpaca_secret_key.lower():
            errors.append("ALPACA_SECRET_KEY is missing or looks like a placeholder")

        if self.profile not in ["research_2024", "wsb_2025"]:
            errors.append(f"Invalid profile: {self.profile}")

        if not (0.0 <= self.max_total_risk <= 1.0):
            errors.append(
                f"max_total_risk must be between 0 and 1, got {self.max_total_risk}"
            )

        if not (0.0 <= self.max_position_size <= 1.0):
            errors.append(
                f"max_position_size must be between 0 and 1, got {self.max_position_size}"
            )

        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")


def load_settings() -> AppSettings:
    return AppSettings()
