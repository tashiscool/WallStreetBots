# backend/tradingbot/config/settings.py
from __future__ import annotations
import os
from typing import Literal, Optional

try:
    from pydantic_settings import BaseSettings
    from pydantic import Field, field_validator

    PYDANTIC_V2 = True
except ImportError:
    try:
        from pydantic import BaseSettings, Field, validator as field_validator

        PYDANTIC_V2 = False
    except ImportError:
        # Fallback - simple settings without validation
        class BaseSettings:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        def Field(default=None, **kwargs):
            return default

        def field_validator(*args, **kwargs):
            def decorator(func):
                return func

            return decorator

        PYDANTIC_V2 = False

StrategyProfile = Literal["research_2024", "wsb_2025"]


class AppSettings(BaseSettings):
    # Broker/API
    alpaca_api_key: str = Field(..., env="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(..., env="ALPACA_SECRET_KEY")
    alpaca_paper: bool = Field(True, env="ALPACA_PAPER")  # ALWAYS true in non-prod

    # Profile selects default risk knobs (still overridable)
    profile: StrategyProfile = Field("research_2024", env="WSB_PROFILE")

    # Risk
    max_total_risk: float = Field(0.30, ge=0.0, le=1.0)
    max_position_size: float = Field(0.10, ge=0.0, le=1.0)

    # Analytics/Regime
    enable_advanced_analytics: bool = Field(True, env="ENABLE_ADV_ANALYTICS")
    enable_market_regime_adaptation: bool = Field(True, env="ENABLE_REGIME_ADAPT")
    analytics_update_interval: int = Field(3600, ge=10)
    regime_adaptation_interval: int = Field(1800, ge=10)

    # Data
    use_cache: bool = Field(True, env="DATA_USE_CACHE")
    data_cache_path: str = Field("./.cache", env="DATA_CACHE_PATH")

    # Safety
    dry_run: bool = Field(True, env="DRY_RUN")  # blocks live orders if True

    class Config:
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"

    if PYDANTIC_V2:

        @field_validator("alpaca_api_key", "alpaca_secret_key")
        @classmethod
        def _no_placeholders(cls, v: str) -> str:
            if not v or "your_" in v.lower() or "test" in v.lower():
                raise ValueError("API keys missing or look like placeholders.")
            return v
    else:

        @field_validator("alpaca_api_key", "alpaca_secret_key")
        def _no_placeholders(cls, v: str) -> str:
            if not v or "your_" in v.lower() or "test" in v.lower():
                raise ValueError("API keys missing or look like placeholders.")
            return v


def load_settings() -> AppSettings:
    return AppSettings()
