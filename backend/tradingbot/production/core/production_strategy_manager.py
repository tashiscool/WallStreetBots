"""Production Strategy Manager
Orchestrates all production - ready strategies with real broker integration.

This module provides a unified interface for managing all production strategies:
- WSB Dip Bot (production version)
- Earnings Protection (production version)
- Index Baseline (production version)
- Additional strategies as they become production - ready

All strategies use:
- Real market data from Alpaca
- Live broker integration
- Django model persistence
- Comprehensive risk management
"""

import asyncio
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

# Import advanced analytics and market regime adaptation
from ...analytics.advanced_analytics import AdvancedAnalytics, PerformanceMetrics
from ...analytics.market_regime_adapter import (
    MarketRegimeAdapter,
    RegimeAdaptationConfig,
    StrategyAdaptation,
)
from ..data.production_data_integration import (
    ReliableDataProvider as ProductionDataProvider,
)
# Strategy creation functions will be imported lazily to avoid circular imports
from .production_integration import ProductionIntegrationManager

# Import signal validation components
from backend.validation import (
    AlphaValidationGate, ValidationCriteria, ValidationStateAdapter, 
    ValidationReporter, ParameterRegistry
)

# Import Django models for persistence
from ...models.models import (
    ValidationRun, SignalValidationMetrics, DataQualityMetrics, 
    ValidationParameterRegistry
)


class StrategyProfile(str, Enum):
    research_2024 = "research_2024"
    wsb_2025 = "wsb_2025"
    trump_2025 = "trump_2025"
    bubble_aware_2025 = "bubble_aware_2025"


@dataclass
class StrategyConfig:
    """Configuration for individual strategy."""

    name: str
    enabled: bool = True
    max_position_size: float = 0.20
    risk_tolerance: str = "medium"  # "low", "medium", "high"
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProductionStrategyManagerConfig:
    """Configuration for production strategy manager."""

    alpaca_api_key: str
    alpaca_secret_key: str
    paper_trading: bool = True
    user_id: int = 1

    # Strategy configurations
    strategies: dict[str, StrategyConfig] = field(default_factory=dict)
    strategy_configs: dict[str, StrategyConfig] = field(default_factory=dict)

    # Risk management
    max_total_risk: float = 0.50  # 50% max total risk
    max_position_size: float = 0.20  # 20% max per position

    def __post_init__(self):
        """Validate configuration values."""
        if self.max_total_risk > 1.0:
            raise ValueError(f"max_total_risk must be <= 1.0, got {self.max_total_risk}")
        if self.max_position_size > 1.0:
            raise ValueError(f"max_position_size must be <= 1.0, got {self.max_position_size}")
        if self.max_total_risk < 0:
            raise ValueError(f"max_total_risk must be >= 0, got {self.max_total_risk}")
        if self.max_position_size < 0:
            raise ValueError(f"max_position_size must be >= 0, got {self.max_position_size}")

    # Data settings
    data_refresh_interval: int = 30  # seconds

    # Alert settings
    enable_alerts: bool = True

    # NEW: choose a preset
    profile: StrategyProfile = StrategyProfile.research_2024

    # Advanced analytics settings
    enable_advanced_analytics: bool = True
    enable_market_regime_adaptation: bool = True
    analytics_update_interval: int = 3600  # seconds
    regime_adaptation_interval: int = 1800  # seconds

    # Signal validation settings
    enable_signal_validation: bool = True
    validation_criteria: ValidationCriteria | None = None
    validation_update_interval: int = 300  # seconds
    validation_reporting_enabled: bool = True
    validation_persistence_enabled: bool = True


def _preset_defaults(profile: StrategyProfile) -> dict[str, StrategyConfig]:
    """Return default strategy configs for a given profile."""
    if profile == StrategyProfile.research_2024:
        # === RESEARCH 2024 PRESET (Conservative) ===
        return {
            "wsb_dip_bot": StrategyConfig(
                "wsb_dip_bot",
                True,
                0.25,
                "high",
                {
                    "run_lookback_days": 7,
                    "run_threshold": 0.08,
                    "dip_threshold": -0.02,
                    "target_dte_days": 21,
                    "otm_percentage": 0.03,
                    "target_multiplier": 2.5,
                    "delta_target": 0.50,
                    "wsb_sentiment_weight": 0.3,
                },
            ),
            "earnings_protection": StrategyConfig(
                "earnings_protection",
                True,
                0.06,
                "high",
                {
                    "iv_percentile_threshold": 70,
                    "min_implied_move": 0.04,
                    "max_days_to_earnings": 30,
                    "min_days_to_earnings": 7,
                    "preferred_strategies": ["long_straddle", "reverse_iron_condor"],
                    "delta_range": (0.25, 0.30),
                    "profit_target": 0.50,
                    "iv_crush_protection": True,
                    "straddle_vs_strangle": "strangle",
                    "atm_for_low_price": True,
                    "ric_for_high_price": True,
                    "avoid_low_iv": True,
                    "exit_before_announcement": False,
                },
            ),
            "index_baseline": StrategyConfig(
                "index_baseline",
                True,
                0.60,
                "medium",
                {
                    "benchmarks": ["SPY", "QQQ", "IWM", "VTI", "ARKK"],
                    "target_allocation": 0.60,
                    "rebalance_threshold": 0.03,
                    "tax_loss_threshold": -0.05,
                    "momentum_factor": 0.2,
                    "volatility_target": 0.20,
                },
            ),
            "wheel_strategy": StrategyConfig(
                "wheel_strategy",
                True,
                0.20,
                "medium",
                {
                    "target_iv_rank": 50,
                    "target_dte_range": (30, 45),
                    "target_delta_range": (0.15, 0.30),
                    "max_positions": 8,
                    "min_premium_dollars": 50,
                    "profit_target": 0.50,
                    "roll_at_dte": 21,
                    "assignment_acceptance": True,
                    "covered_call_delta": 0.20,
                    "portfolio_allocation": 0.40,
                    "diversified_watchlist": [
                        "SPY",
                        "QQQ",
                        "AAPL",
                        "MSFT",
                        "GOOGL",
                        "AMZN",
                    ],
                    "avoid_earnings": True,
                    "min_liquidity_score": 0.7,
                    "fundamental_screen": True,
                },
            ),
            "momentum_weeklies": StrategyConfig(
                "momentum_weeklies",
                True,
                0.05,
                "high",
                {
                    "watchlist": ["SPY", "QQQ", "IWM"],
                    "max_positions": 3,
                    "min_volume_spike": 1.5,
                    "min_momentum_threshold": 0.015,
                    "target_dte_range": (0, 2),
                    "target_delta_range": (0.05, 0.15),
                    "min_premium": 1.00,
                    "profit_target": 0.50,
                    "stop_loss": 2.0,
                    "preferred_day": "monday",
                    "entry_time_after": "10: 00",
                    "avg_hold_hours": 2,
                    "wing_width": 30,
                },
            ),
            "debit_spreads": StrategyConfig(
                "debit_spreads",
                True,
                0.15,
                "high",
                {
                    "watchlist": [
                        "TSLA",
                        "NVDA",
                        "AMD",
                        "PLTR",
                        "GME",
                        "AMC",
                        "MSTR",
                        "COIN",
                        "AAPL",
                        "MSFT",
                        "GOOGL",
                        "META",
                        "AMZN",
                        "NFLX",
                        "SNOW",
                        "UBER",
                        "ROKU",
                        "SQ",
                        "PYPL",
                    ],
                    "max_positions": 12,
                    "min_dte": 10,
                    "max_dte": 35,
                    "min_risk_reward": 1.2,
                    "min_trend_strength": 0.4,
                    "max_iv_rank": 90,
                    "min_volume_score": 0.2,
                    "profit_target": 0.40,
                    "stop_loss": 0.70,
                    "time_exit_dte": 3,
                    "momentum_multiplier": 1.5,
                },
            ),
            "leaps_tracker": StrategyConfig(
                "leaps_tracker",
                True,
                0.03,
                "medium",
                {
                    "max_positions": 5,
                    "max_total_allocation": 0.15,
                    "min_dte": 365,
                    "max_dte": 730,
                    "delta_strategy": "mixed",
                    "high_delta_range": (0.70, 0.80),
                    "low_delta_range": (0.10, 0.30),
                    "profit_levels": [100, 200, 300],
                    "scale_out_percentage": 25,
                    "stop_loss": 1.0,
                    "time_exit_dte": 90,
                    "entry_staging": True,
                    "staging_periods": 3,
                    "focus_sectors": ["technology", "growth"],
                    "min_premium_percentage": 0.10,
                },
            ),
            "swing_trading": StrategyConfig(
                "swing_trading",
                True,
                0.05,
                "high",
                {
                    "watchlist": [
                        "TSLA",
                        "NVDA",
                        "AMD",
                        "PLTR",
                        "GME",
                        "AMC",
                        "MSTR",
                        "COIN",
                        "SPY",
                        "QQQ",
                        "AAPL",
                        "MSFT",
                        "GOOGL",
                        "META",
                        "NFLX",
                        "ARKK",
                        "TQQQ",
                        "SOXL",
                        "SPXL",
                        "XLK",
                    ],
                    "max_positions": 8,
                    "max_expiry_days": 35,
                    "min_strength_score": 45.0,
                    "min_volume_multiple": 1.5,
                    "min_breakout_strength": 0.001,
                    "min_premium": 0.15,
                    "profit_targets": [30, 60, 150],
                    "stop_loss_pct": 50,
                    "max_hold_hours": 24,
                    "end_of_day_exit_hour": 16,
                    "meme_stock_multiplier": 1.5,
                    "wsb_momentum_factor": 0.3,
                },
            ),
            "spx_credit_spreads": StrategyConfig(
                "spx_credit_spreads",
                True,
                0.04,
                "medium",
                {
                    "strategy_type": "iron_condor",
                    "target_short_delta": 0.15,
                    "target_dte_range": (28, 35),
                    "profit_target_pct": 0.50,
                    "stop_loss_multiple": 2.2,
                    "max_dte": 45,
                    "min_credit": 1.00,
                    "max_spread_width": 50,
                    "max_positions": 3,
                    "long_delta": 0.05,
                    "entry_time_preference": "morning",
                    "roll_at_dte": 21,
                    "min_option_volume": 100,
                    "min_option_oi": 50,
                    "double_stop_loss_protection": True,
                    "vix_filter": 25,
                    "avoid_earnings_days": True,
                },
            ),
            "lotto_scanner": StrategyConfig(
                "lotto_scanner",
                True,
                0.01,
                "extreme",
                {
                    "max_risk_pct": 0.5,
                    "max_concurrent_positions": 3,
                    "profit_targets": [500, 1000, 2000],
                    "stop_loss_pct": 1.0,
                    "min_win_probability": 0.05,
                    "max_dte": 2,
                    "catalyst_required": True,
                    "volume_spike_min": 5.0,
                    "focus_tickers": ["SPY", "QQQ", "TSLA", "NVDA"],
                    "earnings_window_only": True,
                    "otm_threshold": 0.10,
                    "iv_spike_required": True,
                    "max_premium_cost": 2.00,
                },
            ),
        }

    elif profile == StrategyProfile.bubble_aware_2025:
        # === BUBBLE - AWARE 2025 PRESET (Enhanced with M & A Scanner) ===
        # Based on validated market data: PLTR P / S  > 100, OpenAI $300B/$14B loss projection,
        # 64% US VC to AI, $364B Big Tech capex, MIT ROI study, M & A deregulation

        # AI Infrastructure (beneficiaries of $364B Big Tech capex)
        ai_infra_core = [
            "SPY",
            "QQQ",
            "SMH",
            "SOXX",
            "NVDA",
            "AVGO",
            "AMAT",
            "LRCX",
            "INTC",
            "MU",
        ]
        # M & A targets with high $/employee potential (Ferguson FTC, competition EO revoked)
        ma_targets = ["XLF", "KRE", "IBB", "JETS", "XLE", "XLU", "XLRE"]
        # Israeli tech (13.4B exits, $300M median, 45% premiums)
        israeli_tech = ["CYBR", "S", "CHKP", "NICE", "MNDY", "WIX", "FROG"]
        # Bubble candidates (P / S  > 35 threshold, overvaluation indicators)
        bubble_watch = [
            "PLTR",
            "SMCI",
            "ARM",
            "COIN",
            "MSTR",
        ]  # PLTR P / S  > 100 validated

        return {
            "wsb_dip_bot": StrategyConfig(
                "wsb_dip_bot",
                True,
                0.20,
                "high",
                {
                    "run_lookback_days": 5,
                    "run_threshold": 0.07,
                    "dip_threshold": -0.018,
                    "target_dte_days": 14,
                    "otm_percentage": 0.02,
                    "target_multiplier": 2.2,  # Reduced from 3.0 due to bubble risk
                    "delta_target": 0.45,
                    "wsb_sentiment_weight": 0.25,
                    # Bubble-aware controls
                    "ai_exposure_limit": 0.15,  # Cap AI exposure at 15%
                    "avoid_post_gap_hours": 2,  # Skip euphoric gaps
                    "news_resolution_required": True,  # Wait for policy clarity
                    "overvaluation_filter": True,  # Skip P / S  > 35 names
                    "bubble_watch_list": bubble_watch,
                },
            ),
            "earnings_protection": StrategyConfig(
                "earnings_protection",
                True,
                0.05,
                "high",
                {
                    "iv_percentile_threshold": 70,  # Be choosy with rich vol
                    "min_implied_move": 0.04,
                    "max_days_to_earnings": 30,
                    "min_days_to_earnings": 5,  # Avoid pure day - of IV crush
                    "preferred_strategies": ["reverse_iron_condor", "long_strangle"],
                    "delta_range": (0.25, 0.30),
                    "profit_target": 0.50,
                    "iv_crush_protection": True,
                    "straddle_vs_strangle": "strangle",
                    "atm_for_low_price": True,
                    "ric_for_high_price": True,
                    "avoid_low_iv": True,
                    "exit_before_announcement": True,  # Take IV run when extreme
                    # Bubble hedging
                    "ai_bubble_hedge": True,
                    "overvaluation_threshold": 35,  # P / S ratio trigger
                    "bubble_indicators": [
                        "insider_selling",
                        "margin_debt",
                        "options_skew",
                    ],
                },
            ),
            "index_baseline": StrategyConfig(
                "index_baseline",
                True,
                0.55,
                "medium",
                {
                    "benchmarks": ["SPY", "QQQ", "IWM", "VTI", "ARKK"],
                    "target_allocation": 0.55,
                    "rebalance_threshold": 0.03,
                    "tax_loss_threshold": -0.05,
                    "momentum_factor": 0.20,
                    "volatility_target": 0.18,  # Lower for AI dispersion
                    # Sector bias for policy environment
                    "trump_sector_bias": {
                        "financials": 1.4,  # Ferguson FTC, competition EO revoked
                        "energy": 1.3,  # Deregulation focus
                        "defense": 1.2,  # Geopolitical tensions
                        "ai_stocks": 0.7,  # Reduce overvalued exposure
                    },
                },
            ),
            "wheel_strategy": StrategyConfig(
                "wheel_strategy",
                True,
                0.18,
                "medium",
                {
                    "target_iv_rank": 50,
                    "target_dte_range": (30, 45),
                    "target_delta_range": (0.15, 0.25),
                    "max_positions": 8,
                    "min_premium_dollars": 50,
                    "profit_target": 0.50,
                    "roll_at_dte": 21,
                    "assignment_acceptance": True,
                    "covered_call_delta": 0.20,
                    "portfolio_allocation": 0.35,
                    "diversified_watchlist": [*ai_infra_core, "AAPL", "MSFT", "GOOGL"],
                    "avoid_earnings": True,
                    "min_liquidity_score": 0.7,
                    "fundamental_screen": True,
                    # Avoid bubble names for wheel
                    "exclude_overvalued": bubble_watch,
                },
            ),
            "momentum_weeklies": StrategyConfig(
                "momentum_weeklies",
                True,
                0.04,
                "high",
                {
                    "watchlist": ai_infra_core,  # AI infra momentum from $364B capex
                    "max_positions": 3,
                    "min_volume_spike": 1.5,
                    "min_momentum_threshold": 0.015,
                    "target_dte_range": (0, 2),  # 0DTE focus (62% of SPX volume)
                    "target_delta_range": (0.05, 0.15),
                    "min_premium": 1.00,
                    "profit_target": 0.50,  # 50% per tasty research
                    "stop_loss": 2.0,  # ~2x stop
                    "preferred_day": "monday",
                    "entry_time_after": "10: 00",  # After policy announcements
                    "avg_hold_hours": 2,
                    "wing_width": 30,
                    # M & A speculation scanner
                    "ma_scanner_enabled": True,
                    "price_per_employee_threshold": 5000000,  # $5M+ per employee filter
                    "ma_premium_target": 0.25,  # 25% typical premium
                },
            ),
            "debit_spreads": StrategyConfig(
                "debit_spreads",
                True,
                0.12,
                "high",
                {
                    "watchlist": ai_infra_core
                    + ma_targets,  # Infra focus, avoid pure AI apps
                    "max_positions": 10,
                    "min_dte": 10,
                    "max_dte": 35,
                    "min_risk_reward": 1.2,
                    "min_trend_strength": 0.45,  # Higher confirmation threshold
                    "max_iv_rank": 90,
                    "min_volume_score": 0.3,
                    "profit_target": 0.40,
                    "stop_loss": 0.70,
                    "time_exit_dte": 3,
                    "momentum_multiplier": 1.4,
                    # Policy tailwinds
                    "policy_tailwind_filter": True,
                    "avoid_bubble_names": bubble_watch,
                },
            ),
            "leaps_tracker": StrategyConfig(
                "leaps_tracker",
                True,
                0.03,
                "medium",
                {
                    "max_positions": 5,
                    "max_total_allocation": 0.12,  # Tighter allocation
                    "min_dte": 365,
                    "max_dte": 730,
                    "delta_strategy": "mixed",
                    "high_delta_range": (0.70, 0.80),
                    "low_delta_range": (0.10, 0.30),
                    "profit_levels": [100, 200, 300],
                    "scale_out_percentage": 25,
                    "stop_loss": 1.0,
                    "time_exit_dte": 90,
                    "entry_staging": True,  # Stagger entries
                    "staging_periods": 3,
                    # Focus on infra over pure AI apps
                    "focus_sectors": ["semicap", "power", "datacenters"],
                    "min_premium_percentage": 0.10,
                    "avoid_ai_apps": True,  # Skip app - layer names
                },
            ),
            "swing_trading": StrategyConfig(
                "swing_trading",
                True,
                0.04,
                "high",
                {
                    "watchlist": ai_infra_core
                    + israeli_tech,  # AI infra + Israeli M & A targets
                    "max_positions": 6,
                    "max_expiry_days": 25,
                    "min_strength_score": 48.0,
                    "min_volume_multiple": 1.7,  # Higher volume requirement
                    "min_breakout_strength": 0.001,
                    "min_premium": 0.20,
                    "profit_targets": [30, 60, 120],  # Quicker profit taking
                    "stop_loss_pct": 45,
                    "max_hold_hours": 24,  # Overnight headline risk
                    "end_of_day_exit_hour": 16,
                    "meme_stock_multiplier": 1.3,
                    "wsb_momentum_factor": 0.25,
                    # Policy momentum
                    "trump_policy_momentum": 1.4,
                    "israeli_tech_premium_scanner": True,  # M & A speculation
                },
            ),
            "spx_credit_spreads": StrategyConfig(
                "spx_credit_spreads",
                True,
                0.04,
                "medium",
                {
                    "strategy_type": "iron_condor",
                    "target_short_delta": 0.12,  # 10 - 15 delta sweet spot
                    "target_dte_range": (28, 35),  # 28 - 35 DTE per research
                    "profit_target_pct": 0.50,  # 50% take profit
                    "stop_loss_multiple": 2.2,  # 2.2x stop loss
                    "min_credit": 1.00,
                    "max_spread_width": 50,
                    "max_positions": 3,
                    "long_delta": 0.05,
                    "entry_time_preference": "morning",  # AM entries avoid policy risk
                    "roll_at_dte": 21,
                    "min_option_volume": 100,
                    "min_option_oi": 50,
                    "double_stop_loss_protection": True,
                    "vix_filter": 25,  # Skip regime-shift days
                    "avoid_earnings_days": True,
                    # Volatility harvesting overlay
                    "trump_volatility_mode": True,
                    "policy_uncertainty_multiplier": 1.8,
                    "tariff_announcement_hedge": True,
                },
            ),
            "lotto_scanner": StrategyConfig(
                "lotto_scanner",
                True,
                0.01,
                "extreme",
                {
                    "max_risk_pct": 0.5,  # Tiny position sizes
                    "max_concurrent_positions": 3,
                    "profit_targets": [500, 1000, 2000],
                    "stop_loss_pct": 1.0,
                    "min_win_probability": 0.05,
                    "max_dte": 2,
                    "catalyst_required": True,
                    "volume_spike_min": 5.0,
                    "focus_tickers": ["SPY", "QQQ", "TSLA", "NVDA"],
                    "earnings_window_only": True,
                    "otm_threshold": 0.10,
                    "iv_spike_required": True,
                    "news_catalyst_weight": 0.8,
                    "max_premium_cost": 2.00,
                    # M & A speculation overlay
                    "ma_rumor_scanner": True,
                    "israeli_tech_focus": israeli_tech,
                },
            ),
        }
    elif profile == StrategyProfile.trump_2025:
        # === TRUMP 2025 PRESET (Policy - Aware) ===
        # AI infrastructure focus (deregulation + domestic fab incentives)
        ai_infra_core = [
            "SPY",
            "QQQ",
            "SMH",
            "SOXX",
            "INTC",
            "MU",
            "AMAT",
            "LRCX",
            "ACLS",
            "NVDA",
            "AVGO",
            "QCOM",
            "TXN",
            "ADI",
        ]
        # M & A beneficiaries (deregulation + antitrust relaxation)
        ma_targets = [
            "XLF",
            "KRE",
            "IBB",
            "JETS",
            "XLE",
            "XLU",
            "XLRE",
        ]  # Financials, biotech, energy, utilities, REITs
        # Israeli tech scanner (high - value M & A activity)
        israeli_tech = ["CYBR", "S", "CHKP", "NICE", "MNDY", "WIX", "FROG"]

        return {
            "spx_credit_spreads": StrategyConfig(
                "spx_credit_spreads",
                True,
                0.15,
                "medium",
                {
                    "strategy_type": "iron_condor",
                    "target_short_delta": 0.12,  # 10 - 15 delta per analysis
                    "target_dte_range": (28, 35),  # Longer DTE for policy volatility
                    "profit_target_pct": 0.50,  # 50% take profit
                    "stop_loss_multiple": 2.2,  # 2.2x stop per source
                    "vix_filter": 25,  # Skip regime-shift days
                    "entry_time_preference": "morning",  # Morning entries preferred
                    "roll_at_dte": 21,
                    "max_positions": 8,  # Reduced from WSB profile
                    "avoid_policy_announcement_days": True,
                },
            ),
            "momentum_weeklies": StrategyConfig(
                "momentum_weeklies",
                True,
                0.12,
                "high",
                {
                    "watchlist": ai_infra_core,  # AI - infra focus per analysis
                    "entry_time_after": "10: 00",  # After 10: 00 ET per source
                    "target_dte_range": (0, 2),  # 0 - 2 DTE only
                    "min_volume_spike": 1.5,
                    "profit_target": 0.50,  # 50% profit target
                    "stop_loss": 2.0,  # ~2x stop
                    "max_positions": 6,
                    "news_catalyst_required": True,
                    "policy_momentum_weight": 1.4,
                },
            ),
            "wheel_strategy": StrategyConfig(
                "wheel_strategy",
                True,
                0.30,
                "medium",
                {
                    "target_dte_range": (30, 45),  # 30 - 45 DTE per analysis
                    "target_delta_range": (0.15, 0.25),  # 15 - 25 delta per source
                    "avoid_earnings": True,
                    "diversified_watchlist": ai_infra_core
                    + ma_targets[:4],  # AI infra + megacaps
                    "profit_target": 0.50,
                    "max_positions": 12,
                    "avoid_tariff_decision_weeks": True,
                    "policy_beneficiary_weight": 1.3,
                },
            ),
            "earnings_protection": StrategyConfig(
                "earnings_protection",
                True,
                0.08,
                "medium",
                {
                    "iv_percentile_threshold": 60,  # Higher threshold - be choosy
                    "min_implied_move": 0.035,
                    "max_days_to_earnings": 14,
                    "min_days_to_earnings": 2,
                    "preferred_strategies": [
                        "reverse_iron_condor",
                        "tight_strangle",
                    ],  # Tight RICs when IV high
                    "prefer_strangle_when_iv_high": True,
                    "exit_before_announcement": True,  # Take profits pre-announce if IV extreme
                    "guidance_risk_filter": True,  # Only on material guidance risk
                    "supply_chain_exposure_weight": 1.5,  # Focus on supply - chain / policy exposure
                },
            ),
            "debit_spreads": StrategyConfig(
                "debit_spreads",
                True,
                0.15,
                "high",
                {
                    "watchlist": ai_infra_core
                    + ma_targets,  # Infra / equipment focus, avoid pure AI apps
                    "max_positions": 10,
                    "min_dte": 14,
                    "max_dte": 35,
                    "min_risk_reward": 1.2,
                    "min_trend_strength": 0.40,  # Require trend confirmation
                    "profit_target": 0.40,  # ~40% gains
                    "stop_loss": 0.70,
                    "momentum_multiplier": 1.3,
                    "policy_tailwind_filter": True,  # Favor policy winners
                },
            ),
            "leaps_tracker": StrategyConfig(
                "leaps_tracker",
                True,
                0.08,
                "medium",
                {  # Reduced allocation
                    "max_positions": 4,  # Constrained per analysis
                    "max_total_allocation": 0.20,  # Cap allocation tightly
                    "min_dte": 365,
                    "max_dte": 730,
                    "focus_sectors": [
                        "us_fab",
                        "data_center",
                        "infrastructure",
                    ],  # US - fab and infra only
                    "profit_levels": [50, 100, 200, 400],
                    "scale_out_percentage": 25,
                    "stop_loss": 0.75,
                    "entry_staging": True,  # Stagger entries
                    "staging_periods": 4,
                    "avoid_pure_ai_apps": True,  # Avoid app - layer names
                },
            ),
            "swing_trading": StrategyConfig(
                "swing_trading",
                True,
                0.05,
                "high",
                {
                    "watchlist": ai_infra_core + israeli_tech,
                    "max_positions": 6,
                    "max_expiry_days": 1,  # Day - only or sub - 24h holds
                    "min_strength_score": 50.0,
                    "min_volume_multiple": 2.0,  # Higher volume requirement
                    "min_breakout_strength": 0.001,
                    "profit_targets": [25, 50, 100],  # Quicker profit taking
                    "stop_loss_pct": 60,  # Wider stops with tiny size
                    "max_hold_hours": 18,  # Sub - 24h holds
                    "news_catalyst_required": True,  # Require volume+catalyst
                    "overnight_headline_risk_filter": True,
                },
            ),
            "wsb_dip_bot": StrategyConfig(
                "wsb_dip_bot",
                True,
                0.15,
                "medium",
                {  # Reduced from WSB
                    "run_lookback_days": 6,  # 5 - 7d lookback per analysis - FIXED TYPO
                    "run_threshold": 0.07,
                    "dip_threshold": -0.018,
                    "target_dte_days": 18,
                    "otm_percentage": 0.025,  # Smaller OTM
                    "target_multiplier": 2.5,
                    "delta_target": 0.50,
                    "wsb_sentiment_weight": 0.30,
                    "news_resolution_required": True,  # Wait for details to clarify - CONSISTENT NAMING
                    "policy_headline_filter": True,  # Filter out policy shock days
                    "index_heavyweights_only": True,  # Most liquid leaders only
                },
            ),
            "index_baseline": StrategyConfig(
                "index_baseline",
                True,
                0.45,
                "medium",
                {
                    "benchmarks": [
                        "SPY",
                        "QQQ",
                        "IWM",
                        "VTI",
                        "SMH",
                        "SOXX",
                        "XLF",
                        "XLE",
                    ],  # Add AI - infra adjacency
                    "target_allocation": 0.45,
                    "rebalance_threshold": 0.03,
                    "tax_loss_threshold": -0.05,
                    "momentum_factor": 0.20,
                    "volatility_target": 0.22,
                    "ai_infrastructure_tilt": 0.15,  # Modest tilt to AI infra
                },
            ),
            "lotto_scanner": StrategyConfig(
                "lotto_scanner",
                False,
                0.01,
                "extreme",
                {  # DISABLED per analysis
                    "max_risk_pct": 1.0,  # Max 1% per trade if enabled
                    "max_concurrent_positions": 2,
                    "profit_targets": [200, 400, 800],
                    "stop_loss_pct": 1.0,  # Accept full loss profile
                    "catalyst_required": True,  # News - catalyst required
                    "policy_shock_filter": True,  # Bimodal outcomes hard to handicap
                },
            ),
        }

    # === WSB 2025 PRESET (Aggressive) ===
    meme_core = [
        "NVDA",
        "TSLA",
        "SMCI",
        "ARM",
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "COIN",
        "MSTR",
        "PLTR",
        "AMD",
        "SPY",
        "QQQ",
        "IWM",
        "GME",
        "AMC",
    ]
    return {
        "wsb_dip_bot": StrategyConfig(
            "wsb_dip_bot",
            True,
            0.28,
            "high",
            {
                "run_lookback_days": 5,
                "run_threshold": 0.06,
                "dip_threshold": -0.015,
                "target_dte_days": 14,
                "otm_percentage": 0.02,
                "target_multiplier": 3.0,
                "delta_target": 0.55,
                "wsb_sentiment_weight": 0.40,
                "use_intraday_confirm": True,
                "min_option_volume": 5000,
            },
        ),
        "earnings_protection": StrategyConfig(
            "earnings_protection",
            True,
            0.22,
            "high",
            {
                "iv_percentile_threshold": 40,
                "min_implied_move": 0.025,
                "max_days_to_earnings": 10,
                "min_days_to_earnings": 0,
                "preferred_strategies": [
                    "long_straddle",
                    "long_strangle",
                    "calendar_spread",
                    "protective_hedge",
                ],
                "earnings_momentum_weight": 0.50,
                "wsb_sentiment_multiplier": 1.5,
                "watchlist": [t for t in meme_core if t not in ("SPY", "QQQ", "IWM")],
            },
        ),
        "index_baseline": StrategyConfig(
            "index_baseline",
            True,
            0.55,
            "medium",
            {
                "benchmarks": ["SPY", "QQQ", "IWM", "VTI", "ARKK", "SMH", "SOXX"],
                "target_allocation": 0.55,
                "rebalance_threshold": 0.03,
                "tax_loss_threshold": -0.05,
                "momentum_factor": 0.25,
                "volatility_target": 0.25,
            },
        ),
        "wheel_strategy": StrategyConfig(
            "wheel_strategy",
            True,
            0.42,
            "high",
            {
                "target_iv_rank": 25,
                "target_dte_range": (14, 28),
                "target_delta_range": (0.25, 0.45),
                "max_positions": 20,
                "min_premium_dollars": 20,
                "profit_target": 0.50,
                "max_loss_pct": 0.75,
                "assignment_buffer_days": 2,
                "gamma_squeeze_factor": 0.30,
                "watchlist": [t for t in meme_core if t not in ("SPY", "QQQ", "IWM")],
            },
        ),
        "momentum_weeklies": StrategyConfig(
            "momentum_weeklies",
            True,
            0.10,
            "high",
            {
                "watchlist": meme_core,
                "max_positions": 8,
                "min_volume_spike": 1.8,
                "min_momentum_threshold": 0.008,
                "target_dte_range": (0, 5),
                "otm_range": (0.01, 0.08),
                "min_premium": 0.20,
                "profit_target": 0.15,
                "stop_loss": 0.80,
                "time_exit_hours": 6,
                "use_0dte_priority": True,
            },
        ),
        "debit_spreads": StrategyConfig(
            "debit_spreads",
            True,
            0.18,
            "high",
            {
                "watchlist": [t for t in meme_core if t not in ("SPY", "QQQ", "IWM")],
                "max_positions": 12,
                "min_dte": 7,
                "max_dte": 21,
                "min_risk_reward": 1.0,
                "min_trend_strength": 0.35,
                "max_iv_rank": 92,
                "min_volume_score": 0.2,
                "profit_target": 0.45,
                "stop_loss": 0.70,
                "time_exit_dte": 2,
                "momentum_multiplier": 1.6,
            },
        ),
        "leaps_tracker": StrategyConfig(
            "leaps_tracker",
            True,
            0.16,
            "high",
            {
                "max_positions": 8,
                "max_total_allocation": 0.40,
                "min_dte": 180,
                "max_dte": 540,
                "min_composite_score": 40,
                "min_entry_timing_score": 30,
                "max_exit_timing_score": 85,
                "profit_levels": [50, 100, 300, 600],
                "scale_out_percentage": 20,
                "stop_loss": 0.70,
                "time_exit_dte": 45,
                "meme_stock_bonus": 25,
                "wsb_sentiment_weight": 0.35,
                "watchlist": [t for t in meme_core if t not in ("SPY", "QQQ", "IWM")],
            },
        ),
        "swing_trading": StrategyConfig(
            "swing_trading",
            True,
            0.06,
            "high",
            {
                "watchlist": meme_core,
                "max_positions": 10,
                "max_expiry_days": 35,
                "min_strength_score": 42.0,
                "min_volume_multiple": 1.4,
                "min_breakout_strength": 0.0008,
                "min_premium": 0.15,
                "profit_targets": [30, 60, 150],
                "stop_loss_pct": 50,
                "max_hold_hours": 36,
                "end_of_day_exit_hour": 16,
                "meme_stock_multiplier": 1.6,
                "wsb_momentum_factor": 0.35,
            },
        ),
        "spx_credit_spreads": StrategyConfig(
            "spx_credit_spreads",
            True,
            0.12,
            "high",
            {
                "use_0dte_priority": True,
                "target_short_delta": 0.20,
                "profit_target_pct": 0.50,
                "stop_loss_pct": 3.0,
                "max_dte": 2,
                "min_credit": 0.40,
                "max_spread_width": 100,
                "max_positions": 12,
                "risk_free_rate": 0.05,
                "target_iv_percentile": 18,
                "min_option_volume": 100,
                "gamma_squeeze_factor": 0.30,
                "vix_momentum_weight": 0.25,
                "market_regime_filter": False,
                "entry_time_window_et": (10, 15.5),
            },
        ),
        "lotto_scanner": StrategyConfig(
            "lotto_scanner",
            True,
            0.04,
            "extreme",
            {
                "0dte_only": True,
                "max_risk_pct": 2.5,
                "max_concurrent_positions": 10,
                "profit_targets": [150, 250, 400],
                "stop_loss_pct": 0.90,
                "max_dte": 1,
            },
        ),
    }


def _apply_profile_risk_overrides(cfg: ProductionStrategyManagerConfig):
    """Tighten / loosen top - level risk based on profile."""
    if cfg.profile == StrategyProfile.research_2024:
        cfg.max_total_risk = 0.10  # Very conservative for research
        cfg.max_position_size = 0.05  # Small position sizes
        cfg.data_refresh_interval = 60  # Less frequent updates
    elif cfg.profile == StrategyProfile.wsb_2025:
        cfg.max_total_risk = 0.65
        cfg.max_position_size = 0.30
        cfg.data_refresh_interval = 10
    elif cfg.profile == StrategyProfile.trump_2025:
        cfg.max_total_risk = 0.55  # Moderate risk for policy volatility
        cfg.max_position_size = 0.25  # Slightly higher than research
        cfg.data_refresh_interval = 20  # Between WSB and research
    elif cfg.profile == StrategyProfile.bubble_aware_2025:
        cfg.max_total_risk = 0.45  # Conservative due to bubble risk
        cfg.max_position_size = 0.20  # Keep position sizes controlled
        cfg.data_refresh_interval = 30  # Standard refresh rate
    else:
        cfg.max_total_risk = 0.50
        cfg.max_position_size = 0.20
        cfg.data_refresh_interval = 30


class ProductionStrategyManager:
    """Production Strategy Manager.

    Orchestrates all production - ready strategies:
    - Manages strategy lifecycle (start / stop / monitor)
    - Coordinates risk management across strategies
    - Provides unified performance tracking
    - Handles strategy communication and alerts
    """

    def __init__(self, config: ProductionStrategyManagerConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Apply profile-specific risk overrides
        _apply_profile_risk_overrides(self.config)

        # Initialize core components
        self.integration_manager = ProductionIntegrationManager(
            config.alpaca_api_key,
            config.alpaca_secret_key,
            config.paper_trading,
            config.user_id,
        )

        # Initialize strategies (will be done after data provider is set up)
        self.strategies: dict[str, Any] = {}

        # System state
        self.is_running = False
        self.start_time: datetime | None = None
        self.last_heartbeat: datetime | None = None

        # Task management
        self.tasks: list[asyncio.Task] = []

        # Performance tracking
        self.performance_metrics: dict[str, Any] = {}

        # Advanced analytics and market regime adaptation
        self.advanced_analytics = None
        self.market_regime_adapter = None
        self.current_regime_adaptation: StrategyAdaptation | None = None
        self.analytics_history: list[PerformanceMetrics] = []

        if self.config.enable_advanced_analytics:
            self.advanced_analytics = AdvancedAnalytics(risk_free_rate=0.02)
            self.logger.info("Advanced analytics enabled")

        if self.config.enable_market_regime_adaptation:
            regime_config = RegimeAdaptationConfig()
            self.market_regime_adapter = MarketRegimeAdapter(regime_config)
            self.logger.info("Market regime adaptation enabled")

        # Signal validation components
        self.validation_gate = None
        self.validation_state_adapter = None
        self.validation_reporter = None
        self.parameter_registry = None
        self.validation_history: list[dict] = []
        self.current_validation_state = "HEALTHY"

        if self.config.enable_signal_validation:
            self.validation_gate = AlphaValidationGate(self.config.validation_criteria)
            self.validation_state_adapter = ValidationStateAdapter()
            self.validation_reporter = ValidationReporter()
            self.parameter_registry = ParameterRegistry()
            self.logger.info("Signal validation enabled")

        # Initialize data provider with validation state adapter
        self.data_provider = ProductionDataProvider(
            config.alpaca_api_key, config.alpaca_secret_key,
            validation_state_adapter=getattr(self, 'validation_state_adapter', None)
        )

        # Initialize strategies now that data provider is available
        self._initialize_strategies()

        # Bubble-aware and M & A overlays (optional for strategies to read)
        self.bubble_aware_adjustments = {
            "ai_exposure_limit": 0.15,
            "overvaluation_short_bias": 0.05,
            "ma_speculation_boost": 1.3,
            "volatility_harvest_mode": True,
        }

        self.ma_speculation = {
            "regulatory_relaxation_weight": 1.5,
            "antitrust_probability": 0.30,
            "deal_premium_target": 0.25,
            "sectors": ["fintech", "biotech", "israel_tech"],
            "price_per_employee_threshold": 5000000,  # $5M+ per employee
        }

        self.logger.info(
            f"ProductionStrategyManager initialized with profile: {config.profile}"
        )

    def _initialize_strategies(self):
        """Initialize all enabled strategies."""
        try:
            # Get preset defaults based on configured profile
            default_configs = _preset_defaults(self.config.profile)

            # Merge with user configurations
            strategy_configs = {**default_configs, **self.config.strategies}

            # Initialize strategies
            for strategy_name, strategy_config in strategy_configs.items():
                if strategy_config.enabled:
                    try:
                        # Sanitize parameters to prevent runtime issues
                        sanitized_config = StrategyConfig(
                            name=strategy_config.name,
                            enabled=strategy_config.enabled,
                            max_position_size=strategy_config.max_position_size,
                            risk_tolerance=strategy_config.risk_tolerance,
                            parameters=self._sanitize_parameters(
                                strategy_config.parameters
                            ),
                        )
                        strategy = self._create_strategy(
                            strategy_name, sanitized_config
                        )
                        if strategy:
                            self.strategies[strategy_name] = strategy
                            self.logger.info(f"Initialized strategy: {strategy_name}")
                    except Exception as e:
                        self.logger.error(
                            f"Failed to initialize strategy {strategy_name}: {e}"
                        )

            self.logger.info(f"Initialized {len(self.strategies)} strategies")

        except Exception as e:
            self.logger.error(f"Error initializing strategies: {e}")

    def _create_strategy(
        self, strategy_name: str, config: StrategyConfig
    ) -> Any | None:
        """Create individual strategy instance."""
        try:
            if strategy_name == "wsb_dip_bot":
                from ...strategies.production.production_wsb_dip_bot import create_production_wsb_dip_bot
                return create_production_wsb_dip_bot(
                    self.integration_manager, self.data_provider, config.parameters
                )
            elif strategy_name == "earnings_protection":
                from ...strategies.production.production_earnings_protection import create_production_earnings_protection
                return create_production_earnings_protection(
                    self.integration_manager, self.data_provider, config.parameters
                )
            elif strategy_name == "index_baseline":
                from ...strategies.production.production_index_baseline import create_production_index_baseline
                return create_production_index_baseline(
                    self.integration_manager, self.data_provider, config.parameters
                )
            elif strategy_name == "wheel_strategy":
                from ...strategies.production.production_wheel_strategy import create_production_wheel_strategy
                return create_production_wheel_strategy(
                    self.integration_manager, self.data_provider, config.parameters
                )
            elif strategy_name == "momentum_weeklies":
                from ...strategies.production.production_momentum_weeklies import create_production_momentum_weeklies
                return create_production_momentum_weeklies(
                    self.integration_manager, self.data_provider, config.parameters
                )
            elif strategy_name == "debit_spreads":
                from ...strategies.production.production_debit_spreads import create_production_debit_spreads
                return create_production_debit_spreads(
                    self.integration_manager, self.data_provider, config.parameters
                )
            elif strategy_name == "leaps_tracker":
                from ...strategies.production.production_leaps_tracker import create_production_leaps_tracker
                return create_production_leaps_tracker(
                    self.integration_manager, self.data_provider, config.parameters
                )
            elif strategy_name == "swing_trading":
                from ...strategies.production.production_swing_trading import create_production_swing_trading
                return create_production_swing_trading(
                    self.integration_manager, self.data_provider, config.parameters
                )
            elif strategy_name == "spx_credit_spreads":
                from ...strategies.production.production_spx_credit_spreads import create_production_spx_credit_spreads
                return create_production_spx_credit_spreads(
                    self.integration_manager, self.data_provider, config.parameters
                )
            elif strategy_name == "lotto_scanner":
                from ...strategies.production.production_lotto_scanner import create_production_lotto_scanner
                return create_production_lotto_scanner(
                    self.integration_manager, self.data_provider, config.parameters
                )
            else:
                self.logger.warning(f"Unknown strategy: {strategy_name}")
                return None

        except Exception as e:
            self.logger.error(f"Error creating strategy {strategy_name}: {e}")
            return None

    def _validate_range(self, name: str, val: float, lo: float, hi: float) -> float:
        """Validate parameter ranges with clamping and logging."""
        if not isinstance(val, int | float) or not (lo <= float(val) <= hi):
            self.logger.warning(
                f"Parameter {name} out of range [{lo},{hi}]: {val}. Clamping."
            )
            return max(lo, min(hi, float(val))) if isinstance(val, int | float) else lo
        return float(val)

    def _sanitize_parameters(self, params: dict[str, Any]) -> dict[str, Any]:
        """Sanitize strategy parameters to avoid runtime blowups."""
        out = dict(params)
        # Explicit bounds for critical parameters
        if "target_short_delta" in out:
            out["target_short_delta"] = self._validate_range(
                "target_short_delta", out["target_short_delta"], 0.03, 0.30
            )
        if "profit_target_pct" in out:
            out["profit_target_pct"] = self._validate_range(
                "profit_target_pct", out["profit_target_pct"], 0.10, 0.80
            )
        if "stop_loss_multiple" in out:
            out["stop_loss_multiple"] = self._validate_range(
                "stop_loss_multiple", out["stop_loss_multiple"], 1.0, 5.0
            )
        if "vix_filter" in out:
            out["vix_filter"] = self._validate_range(
                "vix_filter", out["vix_filter"], 10, 60
            )
        if "ai_exposure_limit" in out:
            out["ai_exposure_limit"] = self._validate_range(
                "ai_exposure_limit", out["ai_exposure_limit"], 0.0, 0.50
            )
        if "overvaluation_threshold" in out:
            out["overvaluation_threshold"] = self._validate_range(
                "overvaluation_threshold", out["overvaluation_threshold"], 5, 200
            )
        if "price_per_employee_threshold" in out:
            out["price_per_employee_threshold"] = self._validate_range(
                "price_per_employee_threshold",
                out["price_per_employee_threshold"],
                100000,
                50000000,
            )
        return out

    async def start_all_strategies(self) -> bool:
        """Start all enabled strategies."""
        try:
            self.logger.info("Starting all production strategies")

            # Validate system state
            if not await self._validate_system_state():
                return False

            # Start strategies
            started_count = 0
            for strategy_name, strategy in self.strategies.items():
                try:
                    # Start strategy in background
                    task = asyncio.create_task(strategy.run_strategy())
                    self.tasks.append(task)
                    started_count += 1
                    self.logger.info(f"Started strategy: {strategy_name}")
                except Exception as e:
                    self.logger.error(f"Failed to start strategy {strategy_name}: {e}")

            if started_count > 0:
                self.is_running = True
                self.start_time = datetime.now()

                # Start monitoring tasks
                task = asyncio.create_task(self._monitoring_loop())
                self.tasks.append(task)
                task = asyncio.create_task(self._heartbeat_loop())
                self.tasks.append(task)
                task = asyncio.create_task(self._performance_tracking_loop())
                self.tasks.append(task)

                # Start advanced analytics and regime adaptation tasks
                if self.config.enable_advanced_analytics:
                    task = asyncio.create_task(self._analytics_loop())
                    self.tasks.append(task)
                if self.config.enable_market_regime_adaptation:
                    task = asyncio.create_task(self._regime_adaptation_loop())
                    self.tasks.append(task)
                if self.config.enable_signal_validation:
                    task = asyncio.create_task(self._validation_loop())
                    self.tasks.append(task)

                self.logger.info(f"Started {started_count} strategies successfully")
                return True
            else:
                self.logger.error("No strategies started successfully")
                return False

        except Exception as e:
            self.logger.error(f"Error starting strategies: {e}")
            return False

    async def stop_all_strategies(self):
        """Stop all strategies."""
        try:
            self.logger.info("Stopping all production strategies")

            self.is_running = False

            # Strategies will stop when their run loops exit
            # In a more sophisticated implementation, we would send stop signals

            self.logger.info("All strategies stopped")

        except Exception as e:
            self.logger.error(f"Error stopping strategies: {e}")

    async def _validate_system_state(self) -> bool:
        """Validate system state before starting."""
        try:
            # Validate Alpaca connection
            success, message = self.integration_manager.alpaca_manager.validate_api()
            if not success:
                self.logger.error(f"Alpaca validation failed: {message}")
                return False

            # Validate account size
            portfolio_value = await self.integration_manager.get_portfolio_value()
            if portfolio_value < 1000:  # Minimum $1000
                self.logger.error(f"Account size {portfolio_value} below minimum")
                return False

            # Validate market hours (optional - strategies can run outside market hours)
            market_open = await self.data_provider.is_market_open()
            if not market_open:
                self.logger.warning(
                    "Market is closed - strategies will wait for market open"
                )

            self.logger.info("System state validation passed")
            return True

        except Exception as e:
            self.logger.error(f"System state validation error: {e}")
            return False

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Monitor positions across all strategies
                await self._monitor_all_positions()

                # Check risk limits with validation-aware sizing
                await self._check_risk_limits_with_validation()

                # Monitor signal validation performance
                await self._monitor_signal_validation_health()

                # Update data cache
                await self._refresh_data_cache()

                # Wait for next cycle
                await asyncio.sleep(self.config.data_refresh_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def _heartbeat_loop(self):
        """Heartbeat loop for system monitoring."""
        while self.is_running:
            try:
                self.last_heartbeat = datetime.now()

                # Send heartbeat alert
                if self.config.enable_alerts:
                    await self.integration_manager.alert_system.send_alert(
                        "SYSTEM_HEARTBEAT",
                        "LOW",
                        f"Production Strategy Manager heartbeat - {len(self.strategies)} strategies active",
                    )

                # Wait 5 minutes between heartbeats
                await asyncio.sleep(300)

            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(60)

    async def _performance_tracking_loop(self):
        """Performance tracking loop."""
        while self.is_running:
            try:
                # Update performance metrics
                await self._update_performance_metrics()

                # Wait 1 hour between updates
                await asyncio.sleep(3600)

            except Exception as e:
                self.logger.error(f"Error in performance tracking loop: {e}")
                await asyncio.sleep(300)

    async def _monitor_all_positions(self):
        """Monitor positions across all strategies."""
        try:
            for strategy_name, strategy in self.strategies.items():
                try:
                    if hasattr(strategy, "monitor_positions"):
                        await strategy.monitor_positions()
                except Exception as e:
                    self.logger.error(
                        f"Error monitoring positions for {strategy_name}: {e}"
                    )

        except Exception as e:
            self.logger.error(f"Error in monitor_all_positions: {e}")

    async def _check_risk_limits(self):
        """Check risk limits across all strategies."""
        try:
            # Get total portfolio risk
            total_risk = await self.integration_manager.get_total_risk()
            portfolio_value = await self.integration_manager.get_portfolio_value()

            if portfolio_value > 0:
                risk_percentage = float(total_risk / portfolio_value)

                if risk_percentage > self.config.max_total_risk:
                    await self.integration_manager.alert_system.send_alert(
                        "RISK_ALERT",
                        "HIGH",
                        f"Total risk {risk_percentage: .1%} exceeds limit {self.config.max_total_risk: .1%}",
                    )

        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")

    async def _check_risk_limits_with_validation(self):
        """Check risk limits with validation-aware dynamic position sizing."""
        try:
            # Get total portfolio risk
            total_risk = await self.integration_manager.get_total_risk()
            portfolio_value = await self.integration_manager.get_portfolio_value()

            if portfolio_value > 0:
                base_risk_percentage = float(total_risk / portfolio_value)
                
                # Apply validation-based risk adjustment
                validation_multiplier = await self._get_validation_risk_multiplier()
                adjusted_risk_limit = self.config.max_total_risk * validation_multiplier
                
                if base_risk_percentage > adjusted_risk_limit:
                    await self.integration_manager.alert_system.send_alert(
                        "VALIDATION_RISK_ALERT",
                        "HIGH",
                        f"Risk {base_risk_percentage:.1%} exceeds validation-adjusted limit {adjusted_risk_limit:.1%} "
                        f"(validation multiplier: {validation_multiplier:.2f})",
                    )
                    
                    # Apply dynamic position sizing based on validation state
                    await self._apply_validation_based_position_sizing()

        except Exception as e:
            self.logger.error(f"Error checking validation-aware risk limits: {e}")

    async def _get_validation_risk_multiplier(self) -> float:
        """Get risk multiplier based on validation state."""
        try:
            if not self.validation_state_adapter:
                return 1.0
                
            state = self.validation_state_adapter.get_state()
            
            # Adjust risk limits based on validation state
            if state.value == 'HEALTHY':
                return 1.0  # Full risk allowed
            elif state.value == 'THROTTLE':
                return 0.7  # Reduce risk by 30%
            else:  # HALT
                return 0.3  # Reduce risk by 70%
                
        except Exception as e:
            self.logger.error(f"Error getting validation risk multiplier: {e}")
            return 0.5  # Conservative fallback

    async def _apply_validation_based_position_sizing(self):
        """Apply dynamic position sizing based on validation state."""
        try:
            if not self.validation_state_adapter:
                return
                
            state = self.validation_state_adapter.get_state()
            
            # Adjust position sizes for all strategies based on validation state
            for strategy_name, strategy in self.strategies.items():
                if hasattr(strategy, 'adjust_position_sizing'):
                    sizing_multiplier = self._get_position_sizing_multiplier(state.value)
                    await strategy.adjust_position_sizing(sizing_multiplier)
                    
                    self.logger.info(f"Adjusted {strategy_name} position sizing by {sizing_multiplier:.2f}x due to validation state: {state.value}")
                    
        except Exception as e:
            self.logger.error(f"Error applying validation-based position sizing: {e}")

    def _get_position_sizing_multiplier(self, validation_state: str) -> float:
        """Get position sizing multiplier based on validation state."""
        multipliers = {
            'HEALTHY': 1.0,
            'THROTTLE': 0.6,  # Reduce position sizes by 40%
            'HALT': 0.2       # Reduce position sizes by 80%
        }
        return multipliers.get(validation_state, 0.5)

    async def _refresh_data_cache(self):
        """Refresh data cache."""
        try:
            # Clear old cache entries
            self.data_provider.clear_cache()

        except Exception as e:
            self.logger.error(f"Error refreshing data cache: {e}")

    async def _update_performance_metrics(self):
        """Update performance metrics."""
        try:
            # Get portfolio summary
            portfolio_summary = self.integration_manager.get_portfolio_summary()

            # Get strategy performance
            strategy_performance = {}
            for strategy_name, strategy in self.strategies.items():
                try:
                    if hasattr(strategy, "get_strategy_status"):
                        strategy_performance[strategy_name] = (
                            strategy.get_strategy_status()
                        )
                except Exception as e:
                    self.logger.error(f"Error getting status for {strategy_name}: {e}")

            # Update metrics
            self.performance_metrics = {
                "timestamp": datetime.now().isoformat(),
                "system_uptime": (datetime.now() - self.start_time).total_seconds()
                if self.start_time
                else 0,
                "portfolio": portfolio_summary,
                "strategies": strategy_performance,
                "data_cache_stats": self.data_provider.get_cache_stats(),
            }
            
            # Calculate comprehensive analytics if enabled
            if self.advanced_analytics and hasattr(self.integration_manager, 'get_portfolio_history'):
                try:
                    portfolio_history = await self.integration_manager.get_portfolio_history()
                    if portfolio_history and len(portfolio_history) > 1:
                        # Convert portfolio history to returns
                        portfolio_values = [entry.get('value', 0) for entry in portfolio_history]
                        returns = []
                        for i in range(1, len(portfolio_values)):
                            if portfolio_values[i-1] != 0:
                                returns.append((portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1])
                        
                        if returns:
                            analytics_result = self.advanced_analytics.calculate_comprehensive_metrics(
                                returns=returns,
                                portfolio_values=portfolio_values
                            )
                            self.performance_metrics["analytics"] = analytics_result
                except Exception as e:
                    self.logger.error(f"Error calculating analytics: {e}")

        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")

    async def _check_regime_adaptation(self):
        """Check market regime and adapt strategies if needed."""
        if not self.market_regime_adapter:
            return
            
        try:
            # Get market data for regime detection
            market_data = self.data_provider.get_market_data()
            current_regime = self.market_regime_adapter.detect_current_regime(market_data)
            adaptation = self.market_regime_adapter.generate_strategy_adaptation(current_regime)
            
            if adaptation:
                self.current_regime_adaptation = adaptation
                self.logger.info(f"Market regime adaptation: {adaptation}")
                
        except Exception as e:
            self.logger.error(f"Error checking regime adaptation: {e}")

    async def _send_performance_alert(self, message: str, level: str = "info"):
        """Send performance alert."""
        try:
            await self.integration_manager.send_alert(message, level)
        except Exception as e:
            self.logger.error(f"Error sending alert: {e}")

    async def update_strategy_config(self, strategy_name: str, config: StrategyConfig):
        """Update strategy configuration."""
        try:
            if strategy_name in self.strategies:
                strategy = self.strategies[strategy_name]
                if hasattr(strategy, 'update_config'):
                    await strategy.update_config(config)
                # Update the config in the manager's config
                self.config.strategy_configs[strategy_name] = config
                self.logger.info(f"Updated config for strategy: {strategy_name}")
            else:
                self.logger.warning(f"Strategy not found: {strategy_name}")
        except Exception as e:
            self.logger.error(f"Error updating strategy config: {e}")

    async def emergency_shutdown(self, reason: str):
        """Emergency shutdown of the system."""
        try:
            self.logger.critical(f"EMERGENCY SHUTDOWN: {reason}")
            await self._send_performance_alert(f"Emergency shutdown: {reason}", "critical")
            await self.stop()
        except Exception as e:
            self.logger.error(f"Error during emergency shutdown: {e}")

    async def _send_heartbeat(self):
        """Send heartbeat signal."""
        try:
            self.last_heartbeat = datetime.now()
            await self._send_performance_alert("System heartbeat", "debug")
        except Exception as e:
            self.logger.error(f"Error sending heartbeat: {e}")

    async def _check_data_provider_health(self):
        """Check data provider health status."""
        try:
            # Test basic functionality
            is_open = await self.data_provider.is_market_open()
            return {
                "status": "healthy",
                "market_open": is_open,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Data provider health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _monitor_strategies(self):
        """Monitor strategy execution and performance."""
        try:
            for strategy_name, strategy in self.strategies.items():
                try:
                    # Generate signals if method exists
                    if hasattr(strategy, 'generate_signals'):
                        signals = await strategy.generate_signals()
                        self.logger.debug(f"Strategy {strategy_name} generated {len(signals)} signals")
                    
                    # Get performance if method exists
                    if hasattr(strategy, 'get_performance'):
                        performance = await strategy.get_performance()
                        self.logger.debug(f"Strategy {strategy_name} performance: {performance}")
                except Exception as e:
                    self.logger.error(f"Error monitoring strategy {strategy_name}: {e}")
        except Exception as e:
            self.logger.error(f"Error in strategy monitoring: {e}")

    async def _save_performance_metrics(self, metrics: PerformanceMetrics):
        """Save performance metrics to persistent storage."""
        try:
            self.analytics_history.append(metrics)
            self.logger.debug(f"Saved performance metrics: {metrics.total_return:.2%}")
        except Exception as e:
            self.logger.error(f"Error saving performance metrics: {e}")

    async def _analytics_loop(self):
        """Advanced analytics calculation loop."""
        while self.is_running:
            try:
                await self._calculate_advanced_analytics()
                await asyncio.sleep(self.config.analytics_update_interval)
            except Exception as e:
                self.logger.error(f"Error in analytics loop: {e}")
                await asyncio.sleep(300)

    async def _regime_adaptation_loop(self):
        """Market regime adaptation loop."""
        while self.is_running:
            try:
                await self._update_regime_adaptation()
                await asyncio.sleep(self.config.regime_adaptation_interval)
            except Exception as e:
                self.logger.error(f"Error in regime adaptation loop: {e}")
                await asyncio.sleep(300)

    async def _validation_loop(self):
        """Signal validation monitoring loop."""
        while self.is_running:
            try:
                await self._run_validation_cycle()
                await asyncio.sleep(self.config.validation_update_interval)
            except Exception as e:
                self.logger.error(f"Error in validation loop: {e}")
                await asyncio.sleep(300)

    async def _calculate_advanced_analytics(self):
        """Calculate comprehensive performance analytics."""
        try:
            if not self.advanced_analytics:
                return

            # Get portfolio returns data
            portfolio_returns = await self._get_portfolio_returns()
            benchmark_returns = await self._get_benchmark_returns()

            if len(portfolio_returns) < 30:  # Need minimum data
                self.logger.info("Insufficient data for analytics calculation")
                return

            # Calculate comprehensive metrics
            metrics = self.advanced_analytics.calculate_comprehensive_metrics(
                returns=portfolio_returns,
                benchmark_returns=benchmark_returns,
                start_date=self.start_time,
                end_date=datetime.now(),
            )

            # Store in history
            self.analytics_history.append(metrics)
            if len(self.analytics_history) > 100:  # Keep last 100 calculations
                self.analytics_history = self.analytics_history[-100:]

            # Generate and log analytics report
            self.advanced_analytics.generate_analytics_report(metrics)
            self.logger.info("Advanced analytics updated")

            # Send alert for significant changes
            if len(self.analytics_history) > 1:
                await self._check_analytics_alerts(metrics)

        except Exception as e:
            self.logger.error(f"Error calculating advanced analytics: {e}")

    async def _update_regime_adaptation(self):
        """Update market regime adaptation."""
        try:
            if not self.market_regime_adapter:
                return

            # Get current market data
            market_data = await self._get_market_data_for_regime()
            current_positions = await self._get_current_positions()

            # Generate strategy adaptation
            adaptation = await self.market_regime_adapter.generate_strategy_adaptation(
                market_data=market_data, current_positions=current_positions
            )

            # Check if adaptation changed
            if (
                not self.current_regime_adaptation
                or adaptation.regime != self.current_regime_adaptation.regime
                or abs(
                    adaptation.confidence - self.current_regime_adaptation.confidence
                )
                > 0.1
            ):
                self.current_regime_adaptation = adaptation
                await self._apply_regime_adaptation(adaptation)

                # Send regime change alert
                await self.integration_manager.alert_system.send_alert(
                    "REGIME_CHANGE",
                    "MEDIUM",
                    f"Market regime: {adaptation.regime.value} "
                    f"(confidence: {adaptation.confidence:.1%})",
                )

                self.logger.info(
                    f"Applied regime adaptation: {adaptation.regime.value}"
                )

        except Exception as e:
            self.logger.error(f"Error updating regime adaptation: {e}")

    async def _get_portfolio_returns(self) -> list[float]:
        """Get portfolio returns for analytics."""
        try:
            # Get portfolio value history from integration manager
            # This would need to be implemented in integration manager
            portfolio_history = await self.integration_manager.get_portfolio_history(
                days=180
            )

            if len(portfolio_history) < 2:
                return []

            # Calculate daily returns
            returns = []
            for i in range(1, len(portfolio_history)):
                prev_value = portfolio_history[i - 1]["value"]
                curr_value = portfolio_history[i]["value"]
                if prev_value > 0:
                    daily_return = (curr_value - prev_value) / prev_value
                    returns.append(daily_return)

            return returns

        except Exception as e:
            self.logger.error(f"Error getting portfolio returns: {e}")
            return []

    async def _get_benchmark_returns(self) -> list[float]:
        """Get benchmark returns (SPY) for comparison."""
        try:
            # Get SPY data for benchmark comparison
            spy_data = await self.data_provider.get_historical_data("SPY", days=180)

            if len(spy_data) < 2:
                return []

            # Calculate daily returns
            returns = []
            for i in range(1, len(spy_data)):
                prev_price = spy_data[i - 1].price
                curr_price = spy_data[i].price
                if prev_price > 0:
                    daily_return = (curr_price - prev_price) / prev_price
                    returns.append(daily_return)

            return returns

        except Exception as e:
            self.logger.error(f"Error getting benchmark returns: {e}")
            return []

    async def _get_market_data_for_regime(self) -> dict[str, Any]:
        """Get market data for regime detection."""
        try:
            # Get key market indicators
            market_data = {}

            # Get SPY data (primary indicator)
            spy_data = await self.data_provider.get_current_price("SPY")
            if spy_data:
                market_data["SPY"] = {
                    "price": spy_data.price,
                    "volume": getattr(spy_data, "volume", 1000000),
                    "high": getattr(spy_data, "high", spy_data.price * 1.01),
                    "low": getattr(spy_data, "low", spy_data.price * 0.99),
                }

            # Add volatility indicator
            vix_data = await self.data_provider.get_current_price("VIX")
            if vix_data:
                market_data["volatility"] = float(vix_data.price) / 100.0

            return market_data

        except Exception as e:
            self.logger.error(f"Error getting market data for regime: {e}")
            return {}

    async def _get_current_positions(self) -> dict[str, Any]:
        """Get current positions for regime adaptation."""
        try:
            return await self.integration_manager.get_all_positions()
        except Exception as e:
            self.logger.error(f"Error getting current positions: {e}")
            return {}

    async def _apply_regime_adaptation(self, adaptation: StrategyAdaptation):
        """Apply regime adaptation to strategies."""
        try:
            # Update strategy parameters based on regime adaptation
            for strategy_name, strategy in self.strategies.items():
                try:
                    # Check if strategy should be enabled / disabled
                    if strategy_name in adaptation.disabled_strategies:
                        if hasattr(strategy, "set_enabled"):
                            await strategy.set_enabled(False)
                        self.logger.info(
                            f"Disabled strategy {strategy_name} for regime {adaptation.regime.value}"
                        )

                    elif strategy_name in adaptation.recommended_strategies:
                        if hasattr(strategy, "set_enabled"):
                            await strategy.set_enabled(True)

                        # Apply parameter adjustments
                        if hasattr(strategy, "update_parameters"):
                            regime_params = {
                                "position_multiplier": adaptation.position_size_multiplier,
                                "max_risk": adaptation.max_risk_per_trade,
                                "stop_loss_adjustment": adaptation.stop_loss_adjustment,
                                "take_profit_adjustment": adaptation.take_profit_adjustment,
                                **adaptation.parameter_adjustments,
                            }
                            await strategy.update_parameters(regime_params)

                        self.logger.info(
                            f"Updated strategy {strategy_name} for regime {adaptation.regime.value}"
                        )

                except Exception as e:
                    self.logger.error(
                        f"Error applying adaptation to {strategy_name}: {e}"
                    )

        except Exception as e:
            self.logger.error(f"Error applying regime adaptation: {e}")

    async def _check_analytics_alerts(self, metrics: PerformanceMetrics):
        """Check for analytics - based alerts."""
        try:
            # Check for significant performance changes
            if len(self.analytics_history) > 1:
                prev_metrics = self.analytics_history[-2]

                # Check for significant drawdown increase
                if metrics.max_drawdown > prev_metrics.max_drawdown * 1.5:
                    await self.integration_manager.alert_system.send_alert(
                        "DRAWDOWN_ALERT",
                        "HIGH",
                        f"Max drawdown increased to {metrics.max_drawdown: .2%}",
                    )

                # Check for Sharpe ratio degradation
                if metrics.sharpe_ratio < prev_metrics.sharpe_ratio * 0.7:
                    await self.integration_manager.alert_system.send_alert(
                        "PERFORMANCE_ALERT",
                        "MEDIUM",
                        f"Sharpe ratio declined to {metrics.sharpe_ratio: .2f}",
                    )

        except Exception as e:
            self.logger.error(f"Error checking analytics alerts: {e}")

    async def _run_validation_cycle(self):
        """Run a complete validation cycle."""
        try:
            if not self.validation_gate:
                return

            # Collect validation results from all strategies
            validation_results = await self._collect_validation_results()
            
            # Run validation gate evaluation
            gate_result = self.validation_gate.evaluate_go_no_go(validation_results)
            
            # Update validation state based on results
            await self._update_validation_state(gate_result)
            
            # Store validation history
            self.validation_history.append({
                'timestamp': datetime.now(),
                'results': validation_results,
                'gate_result': gate_result,
                'state': self.current_validation_state
            })
            
            # Generate validation report if enabled
            if self.config.validation_reporting_enabled:
                await self._generate_validation_report(gate_result)
                
            # Persist validation data if enabled
            if self.config.validation_persistence_enabled:
                await self._persist_validation_data(gate_result)

        except Exception as e:
            self.logger.error(f"Error in validation cycle: {e}")

    async def _collect_validation_results(self) -> dict[str, dict[str, float]]:
        """Collect validation results from all strategies."""
        try:
            results = {
                'risk_adjusted_returns': {},
                'alpha_validation': {},
                'cross_market_validation': {},
                'execution_quality': {},
                'capital_efficiency': {}
            }
            
            # Collect metrics from each strategy
            for strategy in self.strategies.values():
                if hasattr(strategy, 'get_validation_metrics'):
                    strategy_metrics = await strategy.get_validation_metrics()
                    self._aggregate_validation_metrics(results, strategy_metrics)
            
            # Add system-level metrics
            await self._add_system_validation_metrics(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error collecting validation results: {e}")
            return {}

    def _aggregate_validation_metrics(self, results: dict, strategy_metrics: dict):
        """Aggregate validation metrics from individual strategies."""
        try:
            # Aggregate risk-adjusted returns
            if 'sharpe_ratio' in strategy_metrics:
                current_sharpe = results['risk_adjusted_returns'].get('min_sharpe_ratio', 0)
                results['risk_adjusted_returns']['min_sharpe_ratio'] = min(current_sharpe, strategy_metrics['sharpe_ratio'])
            
            if 'max_drawdown' in strategy_metrics:
                current_drawdown = results['risk_adjusted_returns'].get('max_drawdown', 0)
                results['risk_adjusted_returns']['max_drawdown'] = max(current_drawdown, strategy_metrics['max_drawdown'])
            
            # Aggregate other metrics similarly
            for category in ['alpha_validation', 'cross_market_validation', 'execution_quality', 'capital_efficiency']:
                if category in strategy_metrics:
                    for metric, value in strategy_metrics[category].items():
                        if metric not in results[category]:
                            results[category][metric] = value
                        else:
                            # Use appropriate aggregation method (min, max, avg)
                            if 'min_' in metric:
                                results[category][metric] = min(results[category][metric], value)
                            elif 'max_' in metric:
                                results[category][metric] = max(results[category][metric], value)
                            else:
                                # Average for other metrics
                                results[category][metric] = (results[category][metric] + value) / 2
                                
        except Exception as e:
            self.logger.error(f"Error aggregating validation metrics: {e}")

    async def _add_system_validation_metrics(self, results: dict):
        """Add system-level validation metrics."""
        try:
            # Get portfolio metrics
            portfolio_value = await self.integration_manager.get_portfolio_value()
            total_risk = await self.integration_manager.get_total_risk()
            
            if portfolio_value > 0:
                risk_percentage = total_risk / portfolio_value
                results['capital_efficiency']['max_margin_calls'] = 0  # Placeholder
                results['capital_efficiency']['min_return_on_capital'] = 0.15  # Placeholder
                
            # Add execution quality metrics
            results['execution_quality']['max_slippage_variance'] = 0.05  # Placeholder
            results['execution_quality']['min_fill_rate'] = 0.98  # Placeholder
            
        except Exception as e:
            self.logger.error(f"Error adding system validation metrics: {e}")

    async def _update_validation_state(self, gate_result: dict):
        """Update validation state based on gate results."""
        try:
            if not self.validation_state_adapter:
                return
                
            recommendation = gate_result.get('overall_recommendation', 'NO-GO')
            readiness_score = gate_result.get('deployment_readiness_score', 0.0)
            
            if recommendation == 'GO' and readiness_score >= 0.8:
                self.validation_state_adapter.set_state('HEALTHY', 'Validation passed')
                self.current_validation_state = 'HEALTHY'
            elif recommendation == 'GO' and readiness_score >= 0.6:
                self.validation_state_adapter.set_state('THROTTLE', 'Validation marginal')
                self.current_validation_state = 'THROTTLE'
            else:
                self.validation_state_adapter.set_state('HALT', 'Validation failed')
                self.current_validation_state = 'HALT'
                
            # Send alert if state changed
            if self.current_validation_state != getattr(self, '_last_validation_state', 'HEALTHY'):
                await self._send_validation_state_alert(self.current_validation_state, gate_result)
                self._last_validation_state = self.current_validation_state
                
        except Exception as e:
            self.logger.error(f"Error updating validation state: {e}")

    async def _send_validation_state_alert(self, state: str, gate_result: dict):
        """Send alert for validation state change."""
        try:
            readiness_score = gate_result.get('deployment_readiness_score', 0.0)
            failing_criteria = gate_result.get('failing_criteria', [])
            
            message = (f"Validation state changed to {state}\n"
                      f"Readiness score: {readiness_score:.1%}\n"
                      f"Failing criteria: {', '.join(failing_criteria) if failing_criteria else 'None'}")
            
            alert_level = "HIGH" if state == "HALT" else "MEDIUM" if state == "THROTTLE" else "LOW"
            
            await self.integration_manager.alert_system.send_alert(
                "VALIDATION_STATE_CHANGE",
                alert_level,
                message
            )
            
        except Exception as e:
            self.logger.error(f"Error sending validation state alert: {e}")

    async def _generate_validation_report(self, gate_result: dict):
        """Generate validation report."""
        try:
            if not self.validation_reporter:
                return
                
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'gate_result': gate_result,
                'validation_state': self.current_validation_state,
                'strategy_count': len(self.strategies),
                'validation_history_count': len(self.validation_history)
            }
            
            self.validation_reporter.write_json('validation_report', report_data)
            
        except Exception as e:
            self.logger.error(f"Error generating validation report: {e}")

    async def _persist_validation_data(self, gate_result: dict):
        """Persist validation data to database using Django models."""
        try:
            # Create validation run record
            run_id = f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            validation_run = ValidationRun.objects.create(
                run_id=run_id,
                strategy_name="production_strategy_manager",
                validation_state=self.current_validation_state,
                overall_recommendation=gate_result.get('overall_recommendation', 'NO-GO'),
                deployment_readiness_score=gate_result.get('deployment_readiness_score', 0.0),
                validation_results=self.validation_history[-1].get('results', {}) if self.validation_history else {},
                gate_result=gate_result,
                failing_criteria=gate_result.get('failing_criteria', [])
            )
            
            self.logger.info(f"Persisted validation run {run_id}: {gate_result.get('overall_recommendation')}")
            
        except Exception as e:
            self.logger.error(f"Error persisting validation data: {e}")

    async def _persist_signal_validation_metrics(self, metrics: dict, analysis: dict):
        """Persist signal validation metrics to database."""
        try:
            overall_metrics = metrics.get('overall_metrics', {})
            
            signal_metrics = SignalValidationMetrics.objects.create(
                strategy_name="production_strategy_manager",
                total_signals=overall_metrics.get('total_signals', 0),
                validated_signals=overall_metrics.get('validated_signals', 0),
                rejected_signals=overall_metrics.get('rejected_signals', 0),
                average_validation_score=overall_metrics.get('average_validation_score', 0.0),
                validation_latency_ms=overall_metrics.get('validation_latency_ms', 0.0),
                false_positive_rate=overall_metrics.get('false_positive_rate', 0.0),
                false_negative_rate=overall_metrics.get('false_negative_rate', 0.0),
                precision=0.0,  # Would be calculated from historical data
                recall=0.0,     # Would be calculated from historical data
                f1_score=0.0,    # Would be calculated from historical data
                overall_health=analysis.get('overall_health', 'UNKNOWN'),
                issues=analysis.get('issues', []),
                recommendations=analysis.get('recommendations', [])
            )
            
            self.logger.debug(f"Persisted signal validation metrics: {analysis.get('overall_health')}")
            
        except Exception as e:
            self.logger.error(f"Error persisting signal validation metrics: {e}")

    async def _persist_data_quality_metrics(self, quality_score: dict):
        """Persist data quality metrics to database."""
        try:
            data_quality = DataQualityMetrics.objects.create(
                validation_state=quality_score.get('validation_state', 'UNKNOWN'),
                overall_score=quality_score.get('overall_score', 0.0),
                source_scores=quality_score.get('source_scores', {}),
                recommendations=quality_score.get('recommendations', []),
                data_age_seconds=0.0,  # Would be calculated from actual data
                stale_data_count=0     # Would be calculated from actual data
            )
            
            self.logger.debug(f"Persisted data quality metrics: {quality_score.get('overall_score'):.2%}")
            
        except Exception as e:
            self.logger.error(f"Error persisting data quality metrics: {e}")

    async def _persist_frozen_parameters(self, strategy_name: str, params: dict, random_seed: int):
        """Persist frozen parameters to database."""
        try:
            # Get git commit hash
            import subprocess
            import asyncio
            try:
                process = await asyncio.create_subprocess_exec(
                    'git', 'rev-parse', 'HEAD',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                git_commit = stdout.decode().strip() if stdout else 'unknown'
            except Exception:
                git_commit = 'unknown'
            
            # Get requirements hash
            import hashlib
            import aiofiles
            try:
                async with aiofiles.open('requirements.txt', 'rb') as f:
                    content = await f.read()
                    requirements_hash = hashlib.sha256(content).hexdigest()
            except Exception:
                requirements_hash = 'unknown'
            
            frozen_params = ValidationParameterRegistry.objects.create(
                strategy_name=strategy_name,
                frozen_params=params,
                random_seed=random_seed,
                requirements_sha256=requirements_hash,
                python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                git_commit=git_commit,
                file_path=f"reports/params/{strategy_name}_frozen.json"
            )
            
            self.logger.info(f"Persisted frozen parameters for {strategy_name}")
            
        except Exception as e:
            self.logger.error(f"Error persisting frozen parameters: {e}")

    def get_advanced_analytics_summary(self) -> dict[str, Any]:
        """Get advanced analytics summary."""
        if not self.analytics_history:
            return {"status": "no_data"}

        latest_metrics = self.analytics_history[-1]
        return {
            "status": "active",
            "total_return": latest_metrics.total_return,
            "annualized_return": latest_metrics.annualized_return,
            "sharpe_ratio": latest_metrics.sharpe_ratio,
            "max_drawdown": latest_metrics.max_drawdown,
            "volatility": latest_metrics.volatility,
            "win_rate": latest_metrics.win_rate,
            "var_95": latest_metrics.var_95,
            "last_updated": latest_metrics.period_end.isoformat(),
            "trading_days": latest_metrics.trading_days,
        }

    def get_regime_adaptation_summary(self) -> dict[str, Any]:
        """Get regime adaptation summary."""
        if not self.market_regime_adapter:
            return {"status": "disabled"}

        return self.market_regime_adapter.get_adaptation_summary()

    def get_signal_validation_summary(self) -> dict[str, Any]:
        """Get signal validation summary."""
        if not self.validation_gate:
            return {"status": "disabled"}
            
        try:
            return {
                "status": "active",
                "current_state": self.current_validation_state,
                "validation_history_count": len(self.validation_history),
                "last_validation": self.validation_history[-1] if self.validation_history else None,
                "validation_gate_enabled": self.validation_gate is not None,
                "state_adapter_enabled": self.validation_state_adapter is not None,
                "reporting_enabled": self.config.validation_reporting_enabled,
                "persistence_enabled": self.config.validation_persistence_enabled,
            }
        except Exception as e:
            self.logger.error(f"Error getting signal validation summary: {e}")
            return {"status": "error", "error": str(e)}

    def get_system_status(self) -> dict[str, Any]:
        """Get current system status."""
        return {
            "is_running": self.is_running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "last_heartbeat": self.last_heartbeat.isoformat()
            if self.last_heartbeat
            else None,
            "active_strategies": len(self.strategies),
            "strategy_status": {
                name: strategy.get_strategy_status()
                if hasattr(strategy, "get_strategy_status")
                else {}
                for name, strategy in self.strategies.items()
            },
            "performance_metrics": self.performance_metrics,
            "configuration": {
                "paper_trading": self.config.paper_trading,
                "max_total_risk": self.config.max_total_risk,
                "max_position_size": self.config.max_position_size,
                "enabled_strategies": list(self.strategies.keys()),
                "advanced_analytics_enabled": self.config.enable_advanced_analytics,
                "market_regime_adaptation_enabled": self.config.enable_market_regime_adaptation,
                "signal_validation_enabled": self.config.enable_signal_validation,
            },
            "advanced_analytics": self.get_advanced_analytics_summary(),
            "market_regime": self.get_regime_adaptation_summary(),
            "signal_validation": self.get_signal_validation_summary(),
        }

    def get_strategy_performance(self, strategy_name: str) -> dict[str, Any] | None:
        """Get performance for specific strategy."""
        try:
            if strategy_name in self.strategies:
                strategy = self.strategies[strategy_name]
                if hasattr(strategy, "get_strategy_status"):
                    return strategy.get_strategy_status()
            return None
        except Exception as e:
            self.logger.error(f"Error getting performance for {strategy_name}: {e}")
            return None

    # Alias methods for compatibility with tests
    async def start(self) -> bool:
        """Start the strategy manager (alias for start_all_strategies)."""
        return await self.start_all_strategies()

    async def stop(self):
        """Stop the strategy manager (alias for stop_all_strategies)."""
        await self.stop_all_strategies()

    async def get_status(self) -> dict[str, Any]:
        """Get basic status (alias for get_system_status)."""
        return self.get_system_status()

    async def get_detailed_status(self) -> dict[str, Any]:
        """Get detailed status including analytics and regime adaptation."""
        status = self.get_system_status()
        
        # Add advanced analytics summary
        if self.advanced_analytics:
            status["advanced_analytics"] = self.get_advanced_analytics_summary()
        
        # Add regime adaptation summary
        if self.market_regime_adapter:
            status["regime_adaptation"] = self.get_regime_adaptation_summary()
        
        return status


    # ================== SIGNAL VALIDATION INTEGRATION ==================

    async def _monitor_signal_validation_health(self):
        """Monitor signal validation performance across all strategies."""
        try:
            # Collect comprehensive signal validation metrics
            signal_metrics = await self._collect_signal_validation_metrics()
            
            # Analyze signal validation performance
            performance_analysis = await self._analyze_signal_validation_performance(signal_metrics)
            
            # Check for performance degradation
            await self._check_signal_validation_degradation(performance_analysis)
            
            # Update signal validation alerts
            await self._update_signal_validation_alerts(performance_analysis)
            
            # Store performance history
            self._store_signal_validation_performance(performance_analysis)
            
            # Persist metrics to database
            await self._persist_signal_validation_metrics(signal_metrics, performance_analysis)
            
            # Apply performance feedback for each strategy
            for strategy_name, strategy_metrics in signal_metrics.get('strategies', {}).items():
                accuracy_metrics = strategy_metrics.get('accuracy_metrics', {})
                if 'performance_feedback' in accuracy_metrics:
                    await self._apply_performance_feedback(strategy_name, accuracy_metrics['performance_feedback'])

        except Exception as e:
            self.logger.error(f"Error monitoring signal validation health: {e}")

    async def _collect_signal_validation_metrics(self) -> dict[str, Any]:
        """Collect comprehensive signal validation metrics from all strategies."""
        try:
            metrics = {
                'timestamp': datetime.now(),
                'strategies': {},
                'overall_metrics': {
                    'total_signals': 0,
                    'validated_signals': 0,
                    'rejected_signals': 0,
                    'average_validation_score': 0.0,
                    'validation_latency_ms': 0.0,
                    'false_positive_rate': 0.0,
                    'false_negative_rate': 0.0
                }
            }
            
            total_score = 0
            strategy_count = 0
            
            for strategy_name, strategy in self.strategies.items():
                if hasattr(strategy, 'get_strategy_signal_summary'):
                    strategy_metrics = await self._get_strategy_signal_metrics(strategy_name, strategy)
                    metrics['strategies'][strategy_name] = strategy_metrics
                    
                    # Aggregate overall metrics
                    metrics['overall_metrics']['total_signals'] += strategy_metrics.get('total_signals', 0)
                    metrics['overall_metrics']['validated_signals'] += strategy_metrics.get('validated_signals', 0)
                    metrics['overall_metrics']['rejected_signals'] += strategy_metrics.get('rejected_signals', 0)
                    
                    strategy_score = strategy_metrics.get('average_validation_score', 0)
                    if strategy_score > 0:
                        total_score += strategy_score
                        strategy_count += 1
            
            # Calculate overall average score
            if strategy_count > 0:
                metrics['overall_metrics']['average_validation_score'] = total_score / strategy_count
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting signal validation metrics: {e}")
            return {'error': str(e)}

    async def _get_strategy_signal_metrics(self, strategy_name: str, strategy) -> dict[str, Any]:
        """Get detailed signal metrics for a specific strategy."""
        try:
            # Get basic signal summary
            summary = strategy.get_strategy_signal_summary()
            
            # Calculate additional metrics
            total_signals = summary.get('total_signals_validated', 0)
            validated_signals = summary.get('signals_by_action', {}).get('accept', 0)
            rejected_signals = summary.get('signals_by_action', {}).get('reject', 0)
            
            # Calculate validation accuracy if historical data is available
            accuracy_metrics = await self._calculate_signal_accuracy(strategy_name, summary)
            
            return {
                'strategy_name': strategy_name,
                'total_signals': total_signals,
                'validated_signals': validated_signals,
                'rejected_signals': rejected_signals,
                'validation_rate': validated_signals / max(total_signals, 1),
                'average_validation_score': summary.get('average_strength_score', 0),
                'validation_latency_ms': summary.get('average_validation_latency_ms', 0),
                'accuracy_metrics': accuracy_metrics,
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting signal metrics for {strategy_name}: {e}")
            return {'error': str(e)}

    async def _calculate_signal_accuracy(self, strategy_name: str, summary: dict) -> dict[str, Any]:
        """Calculate signal validation accuracy metrics with historical data."""
        try:
            # Get historical signal validation metrics from database
            historical_metrics = await self._get_historical_signal_metrics(strategy_name)
            
            if not historical_metrics:
                # Return placeholder metrics if no historical data
                return {
                    'true_positive_rate': 0.75,  # Placeholder
                    'false_positive_rate': 0.15,  # Placeholder
                    'false_negative_rate': 0.10,  # Placeholder
                    'precision': 0.80,  # Placeholder
                    'recall': 0.85,  # Placeholder
                    'f1_score': 0.82,  # Placeholder
                    'sample_size': summary.get('total_signals_validated', 0),
                    'confidence_level': 'LOW'
                }
            
            # Calculate accuracy metrics from historical data
            accuracy_metrics = await self._compute_accuracy_from_history(historical_metrics, summary)
            
            # Add performance feedback
            feedback = await self._generate_performance_feedback(strategy_name, accuracy_metrics)
            accuracy_metrics['performance_feedback'] = feedback
            
            return accuracy_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating signal accuracy for {strategy_name}: {e}")
            return {'error': str(e)}

    async def _get_historical_signal_metrics(self, strategy_name: str) -> list[dict]:
        """Get historical signal validation metrics from database."""
        try:
            # Query recent signal validation metrics from database
            from django.utils import timezone
            from datetime import timedelta
            
            cutoff_date = timezone.now() - timedelta(days=30)  # Last 30 days
            
            historical_metrics = SignalValidationMetrics.objects.filter(
                strategy_name=strategy_name,
                timestamp__gte=cutoff_date
            ).order_by('-timestamp')[:100]  # Last 100 records
            
            return [
                {
                    'timestamp': metric.timestamp,
                    'total_signals': metric.total_signals,
                    'validated_signals': metric.validated_signals,
                    'rejected_signals': metric.rejected_signals,
                    'precision': metric.precision,
                    'recall': metric.recall,
                    'f1_score': metric.f1_score,
                    'overall_health': metric.overall_health
                }
                for metric in historical_metrics
            ]
            
        except Exception as e:
            self.logger.error(f"Error getting historical signal metrics: {e}")
            return []

    async def _compute_accuracy_from_history(self, historical_metrics: list[dict], current_summary: dict) -> dict[str, Any]:
        """Compute accuracy metrics from historical data."""
        try:
            if not historical_metrics:
                return {'error': 'No historical data available'}
            
            # Calculate weighted averages based on sample sizes
            total_weighted_precision = 0
            total_weighted_recall = 0
            total_weighted_f1 = 0
            total_weight = 0
            
            for metric in historical_metrics:
                weight = metric.get('total_signals', 1)
                total_weight += weight
                total_weighted_precision += metric.get('precision', 0) * weight
                total_weighted_recall += metric.get('recall', 0) * weight
                total_weighted_f1 += metric.get('f1_score', 0) * weight
            
            if total_weight == 0:
                return {'error': 'No valid historical data'}
            
            # Calculate weighted averages
            avg_precision = total_weighted_precision / total_weight
            avg_recall = total_weighted_recall / total_weight
            avg_f1 = total_weighted_f1 / total_weight
            
            # Calculate true/false positive/negative rates
            total_signals = sum(m.get('total_signals', 0) for m in historical_metrics)
            total_validated = sum(m.get('validated_signals', 0) for m in historical_metrics)
            total_rejected = sum(m.get('rejected_signals', 0) for m in historical_metrics)
            
            if total_signals == 0:
                return {'error': 'No signals in historical data'}
            
            # Estimate rates (these would be more accurate with actual trade outcome data)
            validation_rate = total_validated / total_signals
            rejection_rate = total_rejected / total_signals
            
            # Estimate true/false positive rates based on historical performance
            # This is a simplified calculation - in practice, you'd need actual trade outcomes
            estimated_true_positive_rate = avg_precision * validation_rate
            estimated_false_positive_rate = (1 - avg_precision) * validation_rate
            estimated_false_negative_rate = (1 - avg_recall) * rejection_rate
            
            return {
                'true_positive_rate': estimated_true_positive_rate,
                'false_positive_rate': estimated_false_positive_rate,
                'false_negative_rate': estimated_false_negative_rate,
                'precision': avg_precision,
                'recall': avg_recall,
                'f1_score': avg_f1,
                'sample_size': total_signals,
                'confidence_level': 'HIGH' if total_signals > 100 else 'MEDIUM' if total_signals > 50 else 'LOW',
                'historical_period_days': 30,
                'validation_rate_trend': await self._calculate_validation_rate_trend(historical_metrics)
            }
            
        except Exception as e:
            self.logger.error(f"Error computing accuracy from history: {e}")
            return {'error': str(e)}

    async def _calculate_validation_rate_trend(self, historical_metrics: list[dict]) -> str:
        """Calculate trend in validation rate over time."""
        try:
            if len(historical_metrics) < 3:
                return 'INSUFFICIENT_DATA'
            
            # Get recent validation rates
            recent_rates = []
            for metric in historical_metrics[:5]:  # Last 5 records
                total = metric.get('total_signals', 0)
                validated = metric.get('validated_signals', 0)
                if total > 0:
                    recent_rates.append(validated / total)
            
            if len(recent_rates) < 3:
                return 'INSUFFICIENT_DATA'
            
            # Calculate trend
            first_half = sum(recent_rates[:len(recent_rates)//2]) / (len(recent_rates)//2)
            second_half = sum(recent_rates[len(recent_rates)//2:]) / (len(recent_rates) - len(recent_rates)//2)
            
            if second_half > first_half * 1.05:  # 5% increase
                return 'IMPROVING'
            elif second_half < first_half * 0.95:  # 5% decrease
                return 'DEGRADING'
            else:
                return 'STABLE'
                
        except Exception as e:
            self.logger.error(f"Error calculating validation rate trend: {e}")
            return 'ERROR'

    async def _generate_performance_feedback(self, strategy_name: str, accuracy_metrics: dict) -> dict[str, Any]:
        """Generate performance feedback based on accuracy metrics."""
        try:
            feedback = {
                'strategy_name': strategy_name,
                'timestamp': datetime.now(),
                'recommendations': [],
                'warnings': [],
                'performance_score': 0.0,
                'action_items': []
            }
            
            precision = accuracy_metrics.get('precision', 0)
            recall = accuracy_metrics.get('recall', 0)
            f1_score = accuracy_metrics.get('f1_score', 0)
            sample_size = accuracy_metrics.get('sample_size', 0)
            confidence_level = accuracy_metrics.get('confidence_level', 'LOW')
            
            # Calculate overall performance score
            performance_score = (precision * 0.4 + recall * 0.4 + f1_score * 0.2)
            feedback['performance_score'] = performance_score
            
            # Generate recommendations based on metrics
            if precision < 0.7:
                feedback['recommendations'].append('Consider tightening validation criteria to improve precision')
                feedback['action_items'].append('Review false positive signals and adjust validation thresholds')
            
            if recall < 0.7:
                feedback['recommendations'].append('Consider relaxing validation criteria to improve recall')
                feedback['action_items'].append('Review false negative signals and adjust validation thresholds')
            
            if f1_score < 0.75:
                feedback['warnings'].append('Overall signal validation performance is below optimal')
                feedback['action_items'].append('Conduct comprehensive validation criteria review')
            
            if sample_size < 50:
                feedback['warnings'].append('Limited sample size - accuracy metrics may not be reliable')
                feedback['recommendations'].append('Collect more validation data before making significant changes')
            
            if confidence_level == 'LOW':
                feedback['warnings'].append('Low confidence in accuracy metrics due to limited data')
                feedback['action_items'].append('Continue collecting validation data to improve confidence')
            
            # Check validation rate trend
            trend = accuracy_metrics.get('validation_rate_trend', 'UNKNOWN')
            if trend == 'DEGRADING':
                feedback['warnings'].append('Validation rate is degrading over time')
                feedback['action_items'].append('Investigate causes of validation rate degradation')
            elif trend == 'IMPROVING':
                feedback['recommendations'].append('Validation rate is improving - continue current approach')
            
            return feedback
            
        except Exception as e:
            self.logger.error(f"Error generating performance feedback: {e}")
            return {'error': str(e)}

    async def _apply_performance_feedback(self, strategy_name: str, feedback: dict):
        """Apply performance feedback to improve validation."""
        try:
            action_items = feedback.get('action_items', [])
            warnings = feedback.get('warnings', [])
            
            # Log feedback
            self.logger.info(f"Performance feedback for {strategy_name}:")
            for recommendation in feedback.get('recommendations', []):
                self.logger.info(f"  Recommendation: {recommendation}")
            for warning in warnings:
                self.logger.warning(f"  Warning: {warning}")
            for action in action_items:
                self.logger.info(f"  Action: {action}")
            
            # Apply automatic adjustments based on feedback
            performance_score = feedback.get('performance_score', 0)
            if performance_score < 0.6:
                # Low performance - consider reducing strategy allocation
                if strategy_name in self.strategies:
                    strategy = self.strategies[strategy_name]
                    if hasattr(strategy, 'adjust_validation_thresholds'):
                        await strategy.adjust_validation_thresholds({'tighten': True})
                        self.logger.info(f"Applied automatic validation threshold adjustment for {strategy_name}")
            
        except Exception as e:
            self.logger.error(f"Error applying performance feedback: {e}")

    async def _analyze_signal_validation_performance(self, metrics: dict) -> dict[str, Any]:
        """Analyze signal validation performance and identify issues."""
        try:
            analysis = {
                'timestamp': datetime.now(),
                'overall_health': 'HEALTHY',
                'issues': [],
                'recommendations': [],
                'performance_trend': 'STABLE'
            }
            
            overall_metrics = metrics.get('overall_metrics', {})
            
            # Check validation rate
            total_signals = overall_metrics.get('total_signals', 0)
            validated_signals = overall_metrics.get('validated_signals', 0)
            
            if total_signals > 0:
                validation_rate = validated_signals / total_signals
                
                if validation_rate < 0.3:
                    analysis['issues'].append('Low validation rate - too many signals rejected')
                    analysis['recommendations'].append('Review validation criteria - may be too strict')
                    analysis['overall_health'] = 'DEGRADED'
                elif validation_rate > 0.9:
                    analysis['issues'].append('High validation rate - validation may be too permissive')
                    analysis['recommendations'].append('Review validation criteria - may be too lenient')
            
            # Check average validation score
            avg_score = overall_metrics.get('average_validation_score', 0)
            if avg_score < 50:
                analysis['issues'].append('Low average validation score')
                analysis['recommendations'].append('Investigate signal quality degradation')
                analysis['overall_health'] = 'DEGRADED'
            
            # Check validation latency
            avg_latency = overall_metrics.get('validation_latency_ms', 0)
            if avg_latency > 1000:  # 1 second
                analysis['issues'].append('High validation latency')
                analysis['recommendations'].append('Optimize validation performance')
                analysis['overall_health'] = 'DEGRADED'
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing signal validation performance: {e}")
            return {'error': str(e)}

    async def _check_signal_validation_degradation(self, analysis: dict):
        """Check for signal validation performance degradation."""
        try:
            if analysis.get('overall_health') == 'DEGRADED':
                issues = analysis.get('issues', [])
                recommendations = analysis.get('recommendations', [])
                
                message = (f"Signal validation performance degraded:\n"
                          f"Issues: {', '.join(issues)}\n"
                          f"Recommendations: {', '.join(recommendations)}")
                
                await self.integration_manager.alert_system.send_alert(
                    "SIGNAL_VALIDATION_DEGRADATION",
                    "HIGH",
                    message
                )
                
        except Exception as e:
            self.logger.error(f"Error checking signal validation degradation: {e}")

    async def _update_signal_validation_alerts(self, analysis: dict):
        """Update signal validation alerts based on performance analysis."""
        try:
            # Store analysis for trend tracking
            if not hasattr(self, '_signal_validation_history'):
                self._signal_validation_history = []
            
            self._signal_validation_history.append(analysis)
            
            # Keep only recent history (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self._signal_validation_history = [
                entry for entry in self._signal_validation_history
                if entry.get('timestamp', datetime.now()) > cutoff_time
            ]
            
            # Check for trend changes
            if len(self._signal_validation_history) >= 3:
                await self._check_validation_trend_changes()
                
        except Exception as e:
            self.logger.error(f"Error updating signal validation alerts: {e}")

    async def _check_validation_trend_changes(self):
        """Check for significant trend changes in validation performance."""
        try:
            if len(self._signal_validation_history) < 3:
                return
                
            recent_health = [entry.get('overall_health', 'UNKNOWN') for entry in self._signal_validation_history[-3:]]
            
            # Check for health degradation trend
            if recent_health.count('DEGRADED') >= 2:
                await self.integration_manager.alert_system.send_alert(
                    "SIGNAL_VALIDATION_TREND_DEGRADATION",
                    "MEDIUM",
                    "Signal validation performance showing consistent degradation trend"
                )
            
            # Check for health improvement trend
            elif recent_health.count('HEALTHY') >= 2 and recent_health[0] != 'HEALTHY':
                await self.integration_manager.alert_system.send_alert(
                    "SIGNAL_VALIDATION_TREND_IMPROVEMENT",
                    "LOW",
                    "Signal validation performance showing improvement trend"
                )
                
        except Exception as e:
            self.logger.error(f"Error checking validation trend changes: {e}")

    def _store_signal_validation_performance(self, analysis: dict):
        """Store signal validation performance data."""
        try:
            # This would integrate with database storage
            # For now, just log the performance data
            self.logger.debug(f"Signal validation performance: {analysis.get('overall_health')}")
            
        except Exception as e:
            self.logger.error(f"Error storing signal validation performance: {e}")

    async def _check_strategy_signal_health(self, strategy_name: str, strategy):
        """Check signal health for a specific strategy."""
        try:
            summary = strategy.get_strategy_signal_summary()

            # Define thresholds for signal health
            thresholds = {
                'minimum_average_score': 50.0,
                'minimum_signals_validated': 5,
                'maximum_rejection_rate': 0.8
            }

            # Check average signal strength
            avg_score = summary.get('average_strength_score', 0)
            if avg_score < thresholds['minimum_average_score']:
                await self._handle_poor_signal_quality(strategy_name, avg_score, "Low average signal strength")

            # Check signal volume
            total_signals = summary.get('total_signals_validated', 0)
            if total_signals < thresholds['minimum_signals_validated']:
                self.logger.warning(f"Strategy {strategy_name} has low signal volume: {total_signals}")

            # Check rejection rate
            rejected_signals = summary.get('signals_by_action', {}).get('reject', 0)
            if total_signals > 0:
                rejection_rate = rejected_signals / total_signals
                if rejection_rate > thresholds['maximum_rejection_rate']:
                    await self._handle_high_rejection_rate(strategy_name, rejection_rate)

        except Exception as e:
            self.logger.error(f"Error checking signal health for {strategy_name}: {e}")

    async def _handle_poor_signal_quality(self, strategy_name: str, avg_score: float, reason: str):
        """Handle poor signal quality by adjusting strategy allocation."""
        try:
            self.logger.warning(f"Poor signal quality for {strategy_name}: {reason} (score: {avg_score:.1f})")

            # Reduce strategy allocation based on signal quality
            if strategy_name in self.strategy_configs:
                current_allocation = self.strategy_configs[strategy_name].allocation

                # Calculate reduced allocation based on signal strength
                quality_factor = max(0.1, avg_score / 100.0)  # Min 10% allocation
                new_allocation = current_allocation * quality_factor

                # Update allocation
                self.strategy_configs[strategy_name].allocation = new_allocation

                self.logger.info(f"Reduced {strategy_name} allocation from {current_allocation:.1%} to {new_allocation:.1%}")

                # Send alert if significant reduction
                if new_allocation < current_allocation * 0.5:
                    await self._send_signal_quality_alert(strategy_name, avg_score, current_allocation, new_allocation)

        except Exception as e:
            self.logger.error(f"Error handling poor signal quality for {strategy_name}: {e}")

    async def _handle_high_rejection_rate(self, strategy_name: str, rejection_rate: float):
        """Handle high signal rejection rate."""
        try:
            self.logger.warning(f"High rejection rate for {strategy_name}: {rejection_rate:.1%}")

            # Temporarily pause strategy if rejection rate is extremely high
            if rejection_rate > 0.95:
                await self._pause_strategy_temporarily(strategy_name, f"Extreme rejection rate: {rejection_rate:.1%}")

        except Exception as e:
            self.logger.error(f"Error handling high rejection rate for {strategy_name}: {e}")

    async def _pause_strategy_temporarily(self, strategy_name: str, reason: str):
        """Temporarily pause a strategy due to signal validation issues."""
        try:
            if strategy_name in self.strategy_configs:
                # Mark strategy as temporarily disabled
                self.strategy_configs[strategy_name].enabled = False
                setattr(self.strategy_configs[strategy_name], 'pause_reason', reason)
                setattr(self.strategy_configs[strategy_name], 'paused_at', datetime.now())

                self.logger.error(f"Temporarily paused strategy {strategy_name}: {reason}")

                # Send critical alert
                await self._send_strategy_pause_alert(strategy_name, reason)

        except Exception as e:
            self.logger.error(f"Error pausing strategy {strategy_name}: {e}")

    async def _send_signal_quality_alert(self, strategy_name: str, avg_score: float, old_allocation: float, new_allocation: float):
        """Send alert for signal quality degradation."""
        try:
            message = (f"Signal quality degradation for {strategy_name}:\n"
                      f"Average signal score: {avg_score:.1f}\n"
                      f"Allocation reduced: {old_allocation:.1%} → {new_allocation:.1%}")

            self.logger.warning(f"Signal quality alert: {message}")

        except Exception as e:
            self.logger.error(f"Error sending signal quality alert: {e}")

    async def _send_strategy_pause_alert(self, strategy_name: str, reason: str):
        """Send critical alert for strategy pause."""
        try:
            message = f"CRITICAL: Strategy {strategy_name} automatically paused - {reason}"
            self.logger.error(f"Critical alert: {message}")

        except Exception as e:
            self.logger.error(f"Error sending strategy pause alert: {e}")

    async def _get_signal_validation_summary(self) -> dict:
        """Get comprehensive signal validation summary across all strategies."""
        try:
            summary = {
                'strategies_with_validation': 0,
                'total_signals_validated': 0,
                'average_signal_strength': 0.0,
                'strategy_summaries': {}
            }

            total_score = 0
            strategies_count = 0

            for strategy_name, strategy in self.strategies.items():
                if hasattr(strategy, 'get_strategy_signal_summary'):
                    strategy_summary = strategy.get_strategy_signal_summary()
                    summary['strategy_summaries'][strategy_name] = strategy_summary
                    summary['strategies_with_validation'] += 1

                    # Aggregate metrics
                    strategy_signals = strategy_summary.get('total_signals_validated', 0)
                    summary['total_signals_validated'] += strategy_signals

                    strategy_score = strategy_summary.get('average_strength_score', 0)
                    if strategy_score > 0:
                        total_score += strategy_score
                        strategies_count += 1

            # Calculate overall average
            if strategies_count > 0:
                summary['average_signal_strength'] = total_score / strategies_count

            return summary

        except Exception as e:
            self.logger.error(f"Error getting signal validation summary: {e}")
            return {'error': str(e)}

    def get_signal_validation_performance_report(self) -> dict:
        """Get detailed signal validation performance report."""
        try:
            # Use asyncio to get the summary
            loop = asyncio.get_event_loop()
            validation_summary = loop.run_until_complete(self._get_signal_validation_summary())

            return {
                'timestamp': datetime.now(),
                'validation_summary': validation_summary,
                'strategy_configs': {name: {
                    'allocation': config.allocation,
                    'enabled': config.enabled,
                    'pause_reason': getattr(config, 'pause_reason', None),
                    'paused_at': getattr(config, 'paused_at', None)
                } for name, config in self.strategy_configs.items()}
            }

        except Exception as e:
            self.logger.error(f"Error generating signal validation report: {e}")
            return {'error': str(e)}


# Factory function
def create_production_strategy_manager(
    config: ProductionStrategyManagerConfig,
) -> ProductionStrategyManager:
    """Create ProductionStrategyManager instance."""
    return ProductionStrategyManager(config)


# ===================== STRATEGY CREATION FUNCTIONS =====================

def create_production_wsb_dip_bot(trading_client, data_provider, config, logger):
    """Create production WSB dip bot strategy."""
    from ...strategies.production.production_wsb_dip_bot import ProductionWSBDipBot
    return ProductionWSBDipBot(trading_client, data_provider, config, logger)


def create_production_wheel_strategy(trading_client, data_provider, config, logger):
    """Create production wheel strategy."""
    from ...strategies.production.production_wheel_strategy import ProductionWheelStrategy
    return ProductionWheelStrategy(trading_client, data_provider, config, logger)


def create_production_earnings_protection(trading_client, data_provider, config, logger):
    """Create production earnings protection strategy."""
    from ...strategies.production.production_earnings_protection import ProductionEarningsProtection
    return ProductionEarningsProtection(trading_client, data_provider, config, logger)


def create_production_index_baseline(trading_client, data_provider, config, logger):
    """Create production index baseline strategy."""
    from ...strategies.production.production_index_baseline import ProductionIndexBaseline
    return ProductionIndexBaseline(trading_client, data_provider, config, logger)


def create_production_lotto_scanner(trading_client, data_provider, config, logger):
    """Create production lotto scanner strategy."""
    from ...strategies.production.production_lotto_scanner import ProductionLottoScanner
    return ProductionLottoScanner(trading_client, data_provider, config, logger)
