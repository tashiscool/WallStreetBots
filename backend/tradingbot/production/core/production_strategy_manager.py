"""
Production Strategy Manager
Orchestrates all production-ready strategies with real broker integration

This module provides a unified interface for managing all production strategies:
- WSB Dip Bot (production version)
- Earnings Protection (production version)  
- Index Baseline (production version)
- Additional strategies as they become production-ready

All strategies use:
- Real market data from Alpaca
- Live broker integration
- Django model persistence
- Comprehensive risk management
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from .production_integration import ProductionIntegrationManager
from ..data.production_data_integration import ReliableDataProvider as ProductionDataProvider
from ..strategies.production_wsb_dip_bot import ProductionWSBDipBot, create_production_wsb_dip_bot
from ..strategies.production_earnings_protection import ProductionEarningsProtection, create_production_earnings_protection
from ..strategies.production_index_baseline import ProductionIndexBaseline, create_production_index_baseline
from ..strategies.production_wheel_strategy import ProductionWheelStrategy, create_production_wheel_strategy
from ..strategies.production_momentum_weeklies import ProductionMomentumWeeklies, create_production_momentum_weeklies
from ..strategies.production_debit_spreads import ProductionDebitSpreads, create_production_debit_spreads
from ..strategies.production_leaps_tracker import ProductionLEAPSTracker, create_production_leaps_tracker
from ..strategies.production_swing_trading import ProductionSwingTrading, create_production_swing_trading
from ..strategies.production_spx_credit_spreads import ProductionSPXCreditSpreads, create_production_spx_credit_spreads
from ..strategies.production_lotto_scanner import ProductionLottoScanner, create_production_lotto_scanner

# Import advanced analytics and market regime adaptation
from ...analytics.advanced_analytics import AdvancedAnalytics, PerformanceMetrics
from ...analytics.market_regime_adapter import MarketRegimeAdapter, RegimeAdaptationConfig, StrategyAdaptation


class StrategyProfile(str, Enum):
    research_2024="research_2024"
    wsb_2025 = "wsb_2025"
    trump_2025 = "trump_2025"
    bubble_aware_2025 = "bubble_aware_2025"


@dataclass
class StrategyConfig:
    """Configuration for individual strategy"""
    name: str
    enabled: bool = True
    max_position_size: float = 0.20
    risk_tolerance: str = "medium"  # "low", "medium", "high"
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProductionStrategyManagerConfig:
    """Configuration for production strategy manager"""
    alpaca_api_key: str
    alpaca_secret_key: str
    paper_trading: bool=True
    user_id: int = 1
    
    # Strategy configurations
    strategies: Dict[str, StrategyConfig] = field(default_factory=dict)
    
    # Risk management
    max_total_risk: float=0.50  # 50% max total risk
    max_position_size: float = 0.20  # 20% max per position
    
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


def _preset_defaults(profile: StrategyProfile) -> Dict[str, StrategyConfig]:
    """Return default strategy configs for a given profile."""
    if profile== StrategyProfile.research_2024:
        # === RESEARCH 2024 PRESET (Conservative) ===
        return {
            'wsb_dip_bot':StrategyConfig('wsb_dip_bot', True, 0.25, 'high', {
                'run_lookback_days':7, 'run_threshold':0.08, 'dip_threshold':-0.02,
                'target_dte_days':21, 'otm_percentage':0.03, 'target_multiplier':2.5,
                'delta_target':0.50, 'wsb_sentiment_weight':0.3
            }),
            'earnings_protection':StrategyConfig('earnings_protection', True, 0.06, 'high', {
                'iv_percentile_threshold':70, 'min_implied_move':0.04,
                'max_days_to_earnings':30, 'min_days_to_earnings':7,
                'preferred_strategies':['long_straddle','reverse_iron_condor'],
                'delta_range':(0.25, 0.30), 'profit_target':0.50,
                'iv_crush_protection':True, 'straddle_vs_strangle':'strangle',
                'atm_for_low_price':True, 'ric_for_high_price':True,
                'avoid_low_iv':True, 'exit_before_announcement':False
            }),
            'index_baseline':StrategyConfig('index_baseline', True, 0.60, 'medium', {
                'benchmarks':['SPY','QQQ','IWM','VTI','ARKK'],
                'target_allocation':0.60, 'rebalance_threshold':0.03,
                'tax_loss_threshold':-0.05, 'momentum_factor':0.2, 'volatility_target':0.20
            }),
            'wheel_strategy':StrategyConfig('wheel_strategy', True, 0.20, 'medium', {
                'target_iv_rank':50, 'target_dte_range':(30,45),
                'target_delta_range':(0.15,0.30), 'max_positions':8,
                'min_premium_dollars':50, 'profit_target':0.50, 'roll_at_dte':21,
                'assignment_acceptance':True, 'covered_call_delta':0.20,
                'portfolio_allocation':0.40,
                'diversified_watchlist':['SPY','QQQ','AAPL','MSFT','GOOGL','AMZN'],
                'avoid_earnings':True, 'min_liquidity_score':0.7, 'fundamental_screen':True
            }),
            'momentum_weeklies':StrategyConfig('momentum_weeklies', True, 0.05, 'high', {
                'watchlist':['SPY','QQQ','IWM'], 'max_positions':3,
                'min_volume_spike':1.5, 'min_momentum_threshold':0.015,
                'target_dte_range':(0,2), 'target_delta_range':(0.05,0.15),
                'min_premium':1.00, 'profit_target':0.50, 'stop_loss':2.0,
                'preferred_day':'monday', 'entry_time_after':'10:00', 'avg_hold_hours':2,
                'wing_width':30
            }),
            'debit_spreads':StrategyConfig('debit_spreads', True, 0.15, 'high', {
                'watchlist':['TSLA','NVDA','AMD','PLTR','GME','AMC','MSTR','COIN','AAPL','MSFT','GOOGL','META','AMZN','NFLX','SNOW','UBER','ROKU','SQ','PYPL'],
                'max_positions':12, 'min_dte':10, 'max_dte':35, 'min_risk_reward':1.2,
                'min_trend_strength':0.4, 'max_iv_rank':90, 'min_volume_score':0.2,
                'profit_target':0.40, 'stop_loss':0.70, 'time_exit_dte':3, 'momentum_multiplier':1.5
            }),
            'leaps_tracker':StrategyConfig('leaps_tracker', True, 0.03, 'medium', {
                'max_positions':5, 'max_total_allocation':0.15,
                'min_dte':365, 'max_dte':730, 'delta_strategy':'mixed',
                'high_delta_range':(0.70,0.80), 'low_delta_range':(0.10,0.30),
                'profit_levels':[100,200,300], 'scale_out_percentage':25,
                'stop_loss':1.0, 'time_exit_dte':90, 'entry_staging':True, 'staging_periods':3,
                'focus_sectors':['technology','growth'], 'min_premium_percentage':0.10
            }),
            'swing_trading':StrategyConfig('swing_trading', True, 0.05, 'high', {
                'watchlist':['TSLA','NVDA','AMD','PLTR','GME','AMC','MSTR','COIN','SPY','QQQ','AAPL','MSFT','GOOGL','META','NFLX','ARKK','TQQQ','SOXL','SPXL','XLK'],
                'max_positions':8, 'max_expiry_days':35, 'min_strength_score':45.0,
                'min_volume_multiple':1.5, 'min_breakout_strength':0.001,
                'min_premium':0.15, 'profit_targets':[30,60,150], 'stop_loss_pct':50,
                'max_hold_hours':24, 'end_of_day_exit_hour':16, 'meme_stock_multiplier':1.5,
                'wsb_momentum_factor':0.3
            }),
            'spx_credit_spreads':StrategyConfig('spx_credit_spreads', True, 0.04, 'medium', {
                'strategy_type':'iron_condor', 'target_short_delta':0.15,
                'target_dte_range':(28,35), 'profit_target_pct':0.50, 'stop_loss_multiple':2.2,
                'max_dte':45, 'min_credit':1.00, 'max_spread_width':50, 'max_positions':3,
                'long_delta':0.05, 'entry_time_preference':'morning', 'roll_at_dte':21,
                'min_option_volume':100, 'min_option_oi':50, 'double_stop_loss_protection':True,
                'vix_filter':25, 'avoid_earnings_days':True
            }),
            'lotto_scanner':StrategyConfig('lotto_scanner', True, 0.01, 'extreme', {
                'max_risk_pct':0.5, 'max_concurrent_positions':3,
                'profit_targets':[500,1000,2000], 'stop_loss_pct':1.0,
                'min_win_probability':0.05, 'max_dte':2, 'catalyst_required':True,
                'volume_spike_min':5.0, 'focus_tickers':['SPY','QQQ','TSLA','NVDA'],
                'earnings_window_only':True, 'otm_threshold':0.10, 'iv_spike_required':True,
                'max_premium_cost':2.00
            })
        }
    
    elif profile== StrategyProfile.bubble_aware_2025:
        # === BUBBLE-AWARE 2025 PRESET (Enhanced with M&A Scanner) ===
        # Based on validated market data: PLTR P/S >100, OpenAI $300B/$14B loss projection,
        # 64% US VC to AI, $364B Big Tech capex, MIT ROI study, M&A deregulation
        
        # AI Infrastructure (beneficiaries of $364B Big Tech capex)
        ai_infra_core=['SPY','QQQ','SMH','SOXX','NVDA','AVGO','AMAT','LRCX','INTC','MU']
        # M&A targets with high $/employee potential (Ferguson FTC, competition EO revoked)
        ma_targets=['XLF','KRE','IBB','JETS','XLE','XLU','XLRE']  
        # Israeli tech (13.4B exits, $300M median, 45% premiums)
        israeli_tech=['CYBR','S','CHKP','NICE','MNDY','WIX','FROG']
        # Bubble candidates (P/S >35 threshold, overvaluation indicators)
        bubble_watch=['PLTR','SMCI','ARM','COIN','MSTR']  # PLTR P/S >100 validated
        
        return {
            'wsb_dip_bot':StrategyConfig('wsb_dip_bot', True, 0.20, 'high', {
                'run_lookback_days':5,
                'run_threshold':0.07,
                'dip_threshold':-0.018,
                'target_dte_days':14,
                'otm_percentage':0.02,
                'target_multiplier':2.2,                    # Reduced from 3.0 due to bubble risk
                'delta_target':0.45,
                'wsb_sentiment_weight':0.25,
                # Bubble-aware controls
                'ai_exposure_limit':0.15,                   # Cap AI exposure at 15%
                'avoid_post_gap_hours':2,                   # Skip euphoric gaps
                'news_resolution_required':True,            # Wait for policy clarity
                'overvaluation_filter':True,                # Skip P/S >35 names
                'bubble_watch_list':bubble_watch
            }),
            'earnings_protection':StrategyConfig('earnings_protection', True, 0.05, 'high', {
                'iv_percentile_threshold':70,               # Be choosy with rich vol
                'min_implied_move':0.04,
                'max_days_to_earnings':30,
                'min_days_to_earnings':5,                   # Avoid pure day-of IV crush
                'preferred_strategies':['reverse_iron_condor','long_strangle'],
                'delta_range':(0.25, 0.30),
                'profit_target':0.50,
                'iv_crush_protection':True,
                'straddle_vs_strangle':'strangle',
                'atm_for_low_price':True,
                'ric_for_high_price':True,
                'avoid_low_iv':True,
                'exit_before_announcement':True,            # Take IV run when extreme
                # Bubble hedging
                'ai_bubble_hedge':True,
                'overvaluation_threshold':35,               # P/S ratio trigger  
                'bubble_indicators':['insider_selling', 'margin_debt', 'options_skew']
            }),
            'index_baseline':StrategyConfig('index_baseline', True, 0.55, 'medium', {
                'benchmarks':['SPY','QQQ','IWM','VTI','ARKK'],
                'target_allocation':0.55,
                'rebalance_threshold':0.03,
                'tax_loss_threshold':-0.05,
                'momentum_factor':0.20,
                'volatility_target':0.18,                  # Lower for AI dispersion
                # Sector bias for policy environment
                'trump_sector_bias':{
                    'financials':1.4,      # Ferguson FTC, competition EO revoked
                    'energy':1.3,          # Deregulation focus
                    'defense':1.2,         # Geopolitical tensions  
                    'ai_stocks':0.7        # Reduce overvalued exposure
                }
            }),
            'wheel_strategy':StrategyConfig('wheel_strategy', True, 0.18, 'medium', {
                'target_iv_rank':50,
                'target_dte_range':(30,45),
                'target_delta_range':(0.15,0.25),
                'max_positions':8,
                'min_premium_dollars':50,
                'profit_target':0.50,
                'roll_at_dte':21,
                'assignment_acceptance':True,
                'covered_call_delta':0.20,
                'portfolio_allocation':0.35,
                'diversified_watchlist':ai_infra_core + ['AAPL','MSFT','GOOGL'],
                'avoid_earnings':True,
                'min_liquidity_score':0.7,
                'fundamental_screen':True,
                # Avoid bubble names for wheel
                'exclude_overvalued':bubble_watch
            }),
            'momentum_weeklies':StrategyConfig('momentum_weeklies', True, 0.04, 'high', {
                'watchlist':ai_infra_core,                  # AI infra momentum from $364B capex
                'max_positions':3,
                'min_volume_spike':1.5,
                'min_momentum_threshold':0.015,
                'target_dte_range':(0,2),                   # 0DTE focus (62% of SPX volume)
                'target_delta_range':(0.05,0.15),
                'min_premium':1.00,
                'profit_target':0.50,                      # 50% per tasty research
                'stop_loss':2.0,                           # ~2x stop
                'preferred_day':'monday',
                'entry_time_after':'10:00',                # After policy announcements
                'avg_hold_hours':2,
                'wing_width':30,
                # M&A speculation scanner
                'ma_scanner_enabled':True,
                'price_per_employee_threshold':5000000,    # $5M+ per employee filter
                'ma_premium_target':0.25                   # 25% typical premium
            }),
            'debit_spreads':StrategyConfig('debit_spreads', True, 0.12, 'high', {
                'watchlist':ai_infra_core + ma_targets,     # Infra focus, avoid pure AI apps
                'max_positions':10,
                'min_dte':10,
                'max_dte':35,
                'min_risk_reward':1.2,
                'min_trend_strength':0.45,                 # Higher confirmation threshold
                'max_iv_rank':90,
                'min_volume_score':0.3,
                'profit_target':0.40,
                'stop_loss':0.70,
                'time_exit_dte':3,
                'momentum_multiplier':1.4,
                # Policy tailwinds
                'policy_tailwind_filter':True,
                'avoid_bubble_names':bubble_watch
            }),
            'leaps_tracker':StrategyConfig('leaps_tracker', True, 0.03, 'medium', {
                'max_positions':5,
                'max_total_allocation':0.12,               # Tighter allocation
                'min_dte':365,
                'max_dte':730,
                'delta_strategy':'mixed',
                'high_delta_range':(0.70,0.80),
                'low_delta_range':(0.10,0.30),
                'profit_levels':[100,200,300],
                'scale_out_percentage':25,
                'stop_loss':1.0,
                'time_exit_dte':90,
                'entry_staging':True,                      # Stagger entries
                'staging_periods':3,
                # Focus on infra over pure AI apps
                'focus_sectors':['semicap','power','datacenters'],
                'min_premium_percentage':0.10,
                'avoid_ai_apps':True                       # Skip app-layer names
            }),
            'swing_trading':StrategyConfig('swing_trading', True, 0.04, 'high', {
                'watchlist':ai_infra_core + israeli_tech,   # AI infra + Israeli M&A targets
                'max_positions':6,
                'max_expiry_days':25,
                'min_strength_score':48.0,
                'min_volume_multiple':1.7,                 # Higher volume requirement
                'min_breakout_strength':0.001,
                'min_premium':0.20,
                'profit_targets':[30,60,120],              # Quicker profit taking
                'stop_loss_pct':45,
                'max_hold_hours':24,                       # Overnight headline risk
                'end_of_day_exit_hour':16,
                'meme_stock_multiplier':1.3,
                'wsb_momentum_factor':0.25,
                # Policy momentum
                'trump_policy_momentum':1.4,
                'israeli_tech_premium_scanner':True        # M&A speculation
            }),
            'spx_credit_spreads':StrategyConfig('spx_credit_spreads', True, 0.04, 'medium', {
                'strategy_type':'iron_condor',
                'target_short_delta':0.12,                 # 10-15 delta sweet spot
                'target_dte_range':(28,35),                # 28-35 DTE per research
                'profit_target_pct':0.50,                  # 50% take profit
                'stop_loss_multiple':2.2,                  # 2.2x stop loss
                'min_credit':1.00,
                'max_spread_width':50,
                'max_positions':3,
                'long_delta':0.05,
                'entry_time_preference':'morning',         # AM entries avoid policy risk
                'roll_at_dte':21,
                'min_option_volume':100,
                'min_option_oi':50,
                'double_stop_loss_protection':True,
                'vix_filter':25,                           # Skip regime-shift days
                'avoid_earnings_days':True,
                # Volatility harvesting overlay
                'trump_volatility_mode':True,
                'policy_uncertainty_multiplier':1.8,
                'tariff_announcement_hedge':True
            }),
            'lotto_scanner':StrategyConfig('lotto_scanner', True, 0.01, 'extreme', {
                'max_risk_pct':0.5,                        # Tiny position sizes
                'max_concurrent_positions':3,
                'profit_targets':[500,1000,2000],
                'stop_loss_pct':1.0,
                'min_win_probability':0.05,
                'max_dte':2,
                'catalyst_required':True,
                'volume_spike_min':5.0,
                'focus_tickers':['SPY','QQQ','TSLA','NVDA'],
                'earnings_window_only':True,
                'otm_threshold':0.10,
                'iv_spike_required':True,
                'news_catalyst_weight':0.8,
                'max_premium_cost':2.00,
                # M&A speculation overlay
                'ma_rumor_scanner':True,
                'israeli_tech_focus':israeli_tech
            })
        }
    elif profile== StrategyProfile.trump_2025:
        # === TRUMP 2025 PRESET (Policy-Aware) ===
        # AI infrastructure focus (deregulation + domestic fab incentives)
        ai_infra_core=['SPY','QQQ','SMH','SOXX','INTC','MU','AMAT','LRCX','ACLS','NVDA','AVGO','QCOM','TXN','ADI']
        # M&A beneficiaries (deregulation + antitrust relaxation)
        ma_targets=['XLF','KRE','IBB','JETS','XLE','XLU','XLRE']  # Financials, biotech, energy, utilities, REITs
        # Israeli tech scanner (high-value M&A activity)
        israeli_tech=['CYBR','S','CHKP','NICE','MNDY','WIX','FROG']
        
        return {
            'spx_credit_spreads':StrategyConfig('spx_credit_spreads', True, 0.15, 'medium', {
                'strategy_type':'iron_condor',
                'target_short_delta':0.12,           # 10-15 delta per analysis
                'target_dte_range':(28,35),          # Longer DTE for policy volatility
                'profit_target_pct':0.50,            # 50% take profit
                'stop_loss_multiple':2.2,            # 2.2x stop per source
                'vix_filter':25,                     # Skip regime-shift days
                'entry_time_preference':'morning',    # Morning entries preferred
                'roll_at_dte':21,
                'max_positions':8,                   # Reduced from WSB profile
                'avoid_policy_announcement_days':True
            }),
            'momentum_weeklies':StrategyConfig('momentum_weeklies', True, 0.12, 'high', {
                'watchlist':ai_infra_core,           # AI-infra focus per analysis
                'entry_time_after':'10:00',         # After 10:00 ET per source
                'target_dte_range':(0,2),           # 0-2 DTE only
                'min_volume_spike':1.5,
                'profit_target':0.50,               # 50% profit target
                'stop_loss':2.0,                    # ~2x stop
                'max_positions':6,
                'news_catalyst_required':True,
                'policy_momentum_weight':1.4
            }),
            'wheel_strategy':StrategyConfig('wheel_strategy', True, 0.30, 'medium', {
                'target_dte_range':(30,45),         # 30-45 DTE per analysis
                'target_delta_range':(0.15,0.25),   # 15-25 delta per source
                'avoid_earnings':True,
                'diversified_watchlist':ai_infra_core + ma_targets[:4],  # AI infra + megacaps
                'profit_target':0.50,
                'max_positions':12,
                'avoid_tariff_decision_weeks':True,
                'policy_beneficiary_weight':1.3
            }),
            'earnings_protection':StrategyConfig('earnings_protection', True, 0.08, 'medium', {
                'iv_percentile_threshold':60,        # Higher threshold - be choosy
                'min_implied_move':0.035,
                'max_days_to_earnings':14,
                'min_days_to_earnings':2,
                'preferred_strategies':['reverse_iron_condor','tight_strangle'],  # Tight RICs when IV high
                'prefer_strangle_when_iv_high':True,
                'exit_before_announcement':True,     # Take profits pre-announce if IV extreme
                'guidance_risk_filter':True,         # Only on material guidance risk
                'supply_chain_exposure_weight':1.5   # Focus on supply-chain/policy exposure
            }),
            'debit_spreads':StrategyConfig('debit_spreads', True, 0.15, 'high', {
                'watchlist':ai_infra_core + ma_targets,  # Infra/equipment focus, avoid pure AI apps
                'max_positions':10,
                'min_dte':14, 'max_dte':35,
                'min_risk_reward':1.2,
                'min_trend_strength':0.40,           # Require trend confirmation
                'profit_target':0.40,               # ~40% gains
                'stop_loss':0.70,
                'momentum_multiplier':1.3,
                'policy_tailwind_filter':True       # Favor policy winners
            }),
            'leaps_tracker':StrategyConfig('leaps_tracker', True, 0.08, 'medium', {  # Reduced allocation
                'max_positions':4,                  # Constrained per analysis
                'max_total_allocation':0.20,        # Cap allocation tightly
                'min_dte':365, 'max_dte':730,
                'focus_sectors':['us_fab','data_center','infrastructure'],  # US-fab and infra only
                'profit_levels':[50,100,200,400],
                'scale_out_percentage':25,
                'stop_loss':0.75,
                'entry_staging':True,               # Stagger entries
                'staging_periods':4,
                'avoid_pure_ai_apps':True           # Avoid app-layer names
            }),
            'swing_trading':StrategyConfig('swing_trading', True, 0.05, 'high', {
                'watchlist':ai_infra_core + israeli_tech,
                'max_positions':6,
                'max_expiry_days':1,                # Day-only or sub-24h holds
                'min_strength_score':50.0,
                'min_volume_multiple':2.0,          # Higher volume requirement
                'min_breakout_strength':0.001,
                'profit_targets':[25,50,100],       # Quicker profit taking
                'stop_loss_pct':60,                 # Wider stops with tiny size
                'max_hold_hours':18,                # Sub-24h holds
                'news_catalyst_required':True,      # Require volume + catalyst
                'overnight_headline_risk_filter':True
            }),
            'wsb_dip_bot':StrategyConfig('wsb_dip_bot', True, 0.15, 'medium', {  # Reduced from WSB
                'run_lookback_days':6,              # 5-7d lookback per analysis - FIXED TYPO
                'run_threshold':0.07,
                'dip_threshold':-0.018,
                'target_dte_days':18,
                'otm_percentage':0.025,             # Smaller OTM
                'target_multiplier':2.5,
                'delta_target':0.50,
                'wsb_sentiment_weight':0.30,
                'news_resolution_required':True,    # Wait for details to clarify - CONSISTENT NAMING
                'policy_headline_filter':True,     # Filter out policy shock days
                'index_heavyweights_only':True     # Most liquid leaders only
            }),
            'index_baseline':StrategyConfig('index_baseline', True, 0.45, 'medium', {
                'benchmarks':['SPY','QQQ','IWM','VTI','SMH','SOXX','XLF','XLE'],  # Add AI-infra adjacency
                'target_allocation':0.45,
                'rebalance_threshold':0.03,
                'tax_loss_threshold':-0.05,
                'momentum_factor':0.20,
                'volatility_target':0.22,
                'ai_infrastructure_tilt':0.15      # Modest tilt to AI infra
            }),
            'lotto_scanner':StrategyConfig('lotto_scanner', False, 0.01, 'extreme', {  # DISABLED per analysis
                'max_risk_pct':1.0,                # Max 1% per trade if enabled
                'max_concurrent_positions':2,
                'profit_targets':[200,400,800],
                'stop_loss_pct':1.0,               # Accept full loss profile  
                'catalyst_required':True,          # News-catalyst required
                'policy_shock_filter':True        # Bimodal outcomes hard to handicap
            })
        }
    
    # === WSB 2025 PRESET (Aggressive) ===
    meme_core=['NVDA','TSLA','SMCI','ARM','AAPL','MSFT','GOOGL','AMZN','META','COIN','MSTR','PLTR','AMD','SPY','QQQ','IWM','GME','AMC']
    return {
        'wsb_dip_bot':StrategyConfig('wsb_dip_bot', True, 0.28, 'high', {
            'run_lookback_days':5,'run_threshold':0.06,'dip_threshold':-0.015,
            'target_dte_days':14,'otm_percentage':0.02,'target_multiplier':3.0,
            'delta_target':0.55,'wsb_sentiment_weight':0.40,'use_intraday_confirm':True,'min_option_volume':5000
        }),
        'earnings_protection':StrategyConfig('earnings_protection', True, 0.22, 'high', {
            'iv_percentile_threshold':40,'min_implied_move':0.025,'max_days_to_earnings':10,'min_days_to_earnings':0,
            'preferred_strategies':['long_straddle','long_strangle','calendar_spread','protective_hedge'],
            'earnings_momentum_weight':0.50,'wsb_sentiment_multiplier':1.5,
            'watchlist':[t for t in meme_core if t not in ('SPY','QQQ','IWM')]
        }),
        'index_baseline':StrategyConfig('index_baseline', True, 0.55, 'medium', {
            'benchmarks':['SPY','QQQ','IWM','VTI','ARKK','SMH','SOXX'],
            'target_allocation':0.55,'rebalance_threshold':0.03,'tax_loss_threshold':-0.05,
            'momentum_factor':0.25,'volatility_target':0.25
        }),
        'wheel_strategy':StrategyConfig('wheel_strategy', True, 0.42, 'high', {
            'target_iv_rank':25,'target_dte_range':(14,28),'target_delta_range':(0.25,0.45),
            'max_positions':20,'min_premium_dollars':20,'profit_target':0.50,'max_loss_pct':0.75,
            'assignment_buffer_days':2,'gamma_squeeze_factor':0.30,
            'watchlist':[t for t in meme_core if t not in ('SPY','QQQ','IWM')]
        }),
        'momentum_weeklies':StrategyConfig('momentum_weeklies', True, 0.10, 'high', {
            'watchlist':meme_core,'max_positions':8,'min_volume_spike':1.8,'min_momentum_threshold':0.008,
            'target_dte_range':(0,5),'otm_range':(0.01,0.08),'min_premium':0.20,
            'profit_target':0.15,'stop_loss':0.80,'time_exit_hours':6,'use_0dte_priority':True
        }),
        'debit_spreads':StrategyConfig('debit_spreads', True, 0.18, 'high', {
            'watchlist':[t for t in meme_core if t not in ('SPY','QQQ','IWM')],
            'max_positions':12,'min_dte':7,'max_dte':21,'min_risk_reward':1.0,
            'min_trend_strength':0.35,'max_iv_rank':92,'min_volume_score':0.2,
            'profit_target':0.45,'stop_loss':0.70,'time_exit_dte':2,'momentum_multiplier':1.6
        }),
        'leaps_tracker':StrategyConfig('leaps_tracker', True, 0.16, 'high', {
            'max_positions':8,'max_total_allocation':0.40,'min_dte':180,'max_dte':540,
            'min_composite_score':40,'min_entry_timing_score':30,'max_exit_timing_score':85,
            'profit_levels':[50,100,300,600],'scale_out_percentage':20,'stop_loss':0.70,'time_exit_dte':45,
            'meme_stock_bonus':25,'wsb_sentiment_weight':0.35,
            'watchlist':[t for t in meme_core if t not in ('SPY','QQQ','IWM')]
        }),
        'swing_trading':StrategyConfig('swing_trading', True, 0.06, 'high', {
            'watchlist':meme_core,'max_positions':10,'max_expiry_days':35,'min_strength_score':42.0,
            'min_volume_multiple':1.4,'min_breakout_strength':0.0008,'min_premium':0.15,
            'profit_targets':[30,60,150],'stop_loss_pct':50,'max_hold_hours':36,'end_of_day_exit_hour':16,
            'meme_stock_multiplier':1.6,'wsb_momentum_factor':0.35
        }),
        'spx_credit_spreads':StrategyConfig('spx_credit_spreads', True, 0.12, 'high', {
            'use_0dte_priority':True,'target_short_delta':0.20,'profit_target_pct':0.50,'stop_loss_pct':3.0,
            'max_dte':2,'min_credit':0.40,'max_spread_width':100,'max_positions':12,'risk_free_rate':0.05,
            'target_iv_percentile':18,'min_option_volume':100,'gamma_squeeze_factor':0.30,
            'vix_momentum_weight':0.25,'market_regime_filter':False,'entry_time_window_et':(10,15.5)
        }),
        'lotto_scanner':StrategyConfig('lotto_scanner', True, 0.04, 'extreme', {
            '0dte_only':True,'max_risk_pct':2.5,'max_concurrent_positions':10,
            'profit_targets':[150,250,400],'stop_loss_pct':0.90,'max_dte':1
        })
    }


def _apply_profile_risk_overrides(cfg: ProductionStrategyManagerConfig):
    """Tighten/loosen top-level risk based on profile."""
    if cfg.profile== StrategyProfile.wsb_2025:
        cfg.max_total_risk = 0.65
        cfg.max_position_size = 0.30
        cfg.data_refresh_interval = 10
    elif cfg.profile == StrategyProfile.trump_2025:
        cfg.max_total_risk = 0.55      # Moderate risk for policy volatility
        cfg.max_position_size = 0.25   # Slightly higher than research
        cfg.data_refresh_interval = 20  # Between WSB and research
    elif cfg.profile == StrategyProfile.bubble_aware_2025:
        cfg.max_total_risk = 0.45      # Conservative due to bubble risk
        cfg.max_position_size = 0.20   # Keep position sizes controlled
        cfg.data_refresh_interval = 30  # Standard refresh rate
    else:
        cfg.max_total_risk = 0.50
        cfg.max_position_size = 0.20
        cfg.data_refresh_interval = 30


class ProductionStrategyManager:
    """
    Production Strategy Manager
    
    Orchestrates all production-ready strategies:
    - Manages strategy lifecycle (start/stop/monitor)
    - Coordinates risk management across strategies
    - Provides unified performance tracking
    - Handles strategy communication and alerts
    """
    
    def __init__(self, config: ProductionStrategyManagerConfig):
        self.config=config
        self.logger = logging.getLogger(__name__)
        
        # Apply profile-specific risk overrides
        _apply_profile_risk_overrides(self.config)
        
        # Initialize core components
        self.integration_manager=ProductionIntegrationManager(
            config.alpaca_api_key,
            config.alpaca_secret_key,
            config.paper_trading,
            config.user_id
        )
        
        self.data_provider=ProductionDataProvider(
            config.alpaca_api_key,
            config.alpaca_secret_key
        )
        
        # Initialize strategies
        self.strategies: Dict[str, Any] = {}
        self._initialize_strategies()
        
        # System state
        self.is_running=False
        self.start_time: Optional[datetime] = None
        self.last_heartbeat: Optional[datetime] = None
        
        # Performance tracking
        self.performance_metrics: Dict[str, Any] = {}

        # Advanced analytics and market regime adaptation
        self.advanced_analytics = None
        self.market_regime_adapter = None
        self.current_regime_adaptation: Optional[StrategyAdaptation] = None
        self.analytics_history: List[PerformanceMetrics] = []

        if self.config.enable_advanced_analytics:
            self.advanced_analytics = AdvancedAnalytics(risk_free_rate=0.02)
            self.logger.info("Advanced analytics enabled")

        if self.config.enable_market_regime_adaptation:
            regime_config = RegimeAdaptationConfig()
            self.market_regime_adapter = MarketRegimeAdapter(regime_config)
            self.logger.info("Market regime adaptation enabled")
        
        # Bubble-aware and M&A overlays (optional for strategies to read)
        self.bubble_aware_adjustments={
            'ai_exposure_limit':0.15,
            'overvaluation_short_bias':0.05,
            'ma_speculation_boost':1.3,
            'volatility_harvest_mode':True
        }
        
        self.ma_speculation={
            'regulatory_relaxation_weight':1.5,
            'antitrust_probability':0.30,
            'deal_premium_target':0.25,
            'sectors':['fintech', 'biotech', 'israel_tech'],
            'price_per_employee_threshold':5000000  # $5M+ per employee
        }
        
        self.logger.info(f"ProductionStrategyManager initialized with profile: {config.profile}")
    
    def _initialize_strategies(self):
        """Initialize all enabled strategies"""
        try:
            # Get preset defaults based on configured profile
            default_configs=_preset_defaults(self.config.profile)
            
            
            # Merge with user configurations
            strategy_configs={**default_configs, **self.config.strategies}
            
            # Initialize strategies
            for strategy_name, strategy_config in strategy_configs.items():
                if strategy_config.enabled:
                    try:
                        # Sanitize parameters to prevent runtime issues
                        sanitized_config=StrategyConfig(
                            name=strategy_config.name,
                            enabled=strategy_config.enabled,
                            max_position_size=strategy_config.max_position_size,
                            risk_tolerance=strategy_config.risk_tolerance,
                            parameters=self._sanitize_parameters(strategy_config.parameters)
                        )
                        strategy=self._create_strategy(strategy_name, sanitized_config)
                        if strategy:
                            self.strategies[strategy_name] = strategy
                            self.logger.info(f"Initialized strategy: {strategy_name}")
                    except Exception as e:
                        self.logger.error(f"Failed to initialize strategy {strategy_name}: {e}")
            
            self.logger.info(f"Initialized {len(self.strategies)} strategies")
            
        except Exception as e:
            self.logger.error(f"Error initializing strategies: {e}")
    
    def _create_strategy(self, strategy_name: str, config: StrategyConfig) -> Optional[Any]:
        """Create individual strategy instance"""
        try:
            if strategy_name== 'wsb_dip_bot':return create_production_wsb_dip_bot(
                    self.integration_manager,
                    self.data_provider,
                    config.parameters
                )
            elif strategy_name== 'earnings_protection':return create_production_earnings_protection(
                    self.integration_manager,
                    self.data_provider,
                    config.parameters
                )
            elif strategy_name== 'index_baseline':return create_production_index_baseline(
                    self.integration_manager,
                    self.data_provider,
                    config.parameters
                )
            elif strategy_name== 'wheel_strategy':return create_production_wheel_strategy(
                    self.integration_manager,
                    self.data_provider,
                    config.parameters
                )
            elif strategy_name== 'momentum_weeklies':return create_production_momentum_weeklies(
                    self.integration_manager,
                    self.data_provider,
                    config.parameters
                )
            elif strategy_name== 'debit_spreads':return create_production_debit_spreads(
                    self.integration_manager,
                    self.data_provider,
                    config.parameters
                )
            elif strategy_name== 'leaps_tracker':return create_production_leaps_tracker(
                    self.integration_manager,
                    self.data_provider,
                    config.parameters
                )
            elif strategy_name== 'swing_trading':return create_production_swing_trading(
                    self.integration_manager,
                    self.data_provider,
                    config.parameters
                )
            elif strategy_name== 'spx_credit_spreads':return create_production_spx_credit_spreads(
                    self.integration_manager,
                    self.data_provider,
                    config.parameters
                )
            elif strategy_name== 'lotto_scanner':return create_production_lotto_scanner(
                    self.integration_manager,
                    self.data_provider,
                    config.parameters
                )
            else:
                self.logger.warning(f"Unknown strategy: {strategy_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating strategy {strategy_name}: {e}")
            return None
    
    def _validate_range(self, name: str, val: float, lo: float, hi: float) -> float:
        """Validate parameter ranges with clamping and logging."""
        if not isinstance(val, (int, float)) or not (lo <= float(val) <= hi):
            self.logger.warning(f"Parameter {name} out of range [{lo},{hi}]: {val}. Clamping.")
            return max(lo, min(hi, float(val))) if isinstance(val, (int, float)) else lo
        return float(val)
    
    def _sanitize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize strategy parameters to avoid runtime blowups."""
        out=dict(params)
        # Explicit bounds for critical parameters
        if 'target_short_delta' in out:
            out['target_short_delta'] = self._validate_range('target_short_delta', out['target_short_delta'], 0.03, 0.30)
        if 'profit_target_pct' in out:
            out['profit_target_pct'] = self._validate_range('profit_target_pct', out['profit_target_pct'], 0.10, 0.80)
        if 'stop_loss_multiple' in out:
            out['stop_loss_multiple'] = self._validate_range('stop_loss_multiple', out['stop_loss_multiple'], 1.0, 5.0)
        if 'vix_filter' in out:
            out['vix_filter'] = self._validate_range('vix_filter', out['vix_filter'], 10, 60)
        if 'ai_exposure_limit' in out:
            out['ai_exposure_limit'] = self._validate_range('ai_exposure_limit', out['ai_exposure_limit'], 0.0, 0.50)
        if 'overvaluation_threshold' in out:
            out['overvaluation_threshold'] = self._validate_range('overvaluation_threshold', out['overvaluation_threshold'], 5, 200)
        if 'price_per_employee_threshold' in out:
            out['price_per_employee_threshold'] = self._validate_range('price_per_employee_threshold', out['price_per_employee_threshold'], 100000, 50000000)
        return out
    
    async def start_all_strategies(self) -> bool:
        """Start all enabled strategies"""
        try:
            self.logger.info("Starting all production strategies")
            
            # Validate system state
            if not await self._validate_system_state():
                return False
            
            # Start strategies
            started_count=0
            for strategy_name, strategy in self.strategies.items():
                try:
                    # Start strategy in background
                    asyncio.create_task(strategy.run_strategy())
                    started_count += 1
                    self.logger.info(f"Started strategy: {strategy_name}")
                except Exception as e:
                    self.logger.error(f"Failed to start strategy {strategy_name}: {e}")
            
            if started_count > 0:
                self.is_running=True
                self.start_time = datetime.now()
                
                # Start monitoring tasks
                asyncio.create_task(self._monitoring_loop())
                asyncio.create_task(self._heartbeat_loop())
                asyncio.create_task(self._performance_tracking_loop())

                # Start advanced analytics and regime adaptation tasks
                if self.config.enable_advanced_analytics:
                    asyncio.create_task(self._analytics_loop())
                if self.config.enable_market_regime_adaptation:
                    asyncio.create_task(self._regime_adaptation_loop())
                
                self.logger.info(f"Started {started_count} strategies successfully")
                return True
            else:
                self.logger.error("No strategies started successfully")
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting strategies: {e}")
            return False
    
    async def stop_all_strategies(self):
        """Stop all strategies"""
        try:
            self.logger.info("Stopping all production strategies")
            
            self.is_running=False
            
            # Strategies will stop when their run loops exit
            # In a more sophisticated implementation, we would send stop signals
            
            self.logger.info("All strategies stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping strategies: {e}")
    
    async def _validate_system_state(self) -> bool:
        """Validate system state before starting"""
        try:
            # Validate Alpaca connection
            success, message=self.integration_manager.alpaca_manager.validate_api()
            if not success:
                self.logger.error(f"Alpaca validation failed: {message}")
                return False
            
            # Validate account size
            portfolio_value=await self.integration_manager.get_portfolio_value()
            if portfolio_value < 1000:  # Minimum $1000
                self.logger.error(f"Account size {portfolio_value} below minimum")
                return False
            
            # Validate market hours (optional - strategies can run outside market hours)
            market_open=await self.data_provider.is_market_open()
            if not market_open:
                self.logger.warning("Market is closed - strategies will wait for market open")
            
            self.logger.info("System state validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"System state validation error: {e}")
            return False
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Monitor positions across all strategies
                await self._monitor_all_positions()
                
                # Check risk limits
                await self._check_risk_limits()
                
                # Update data cache
                await self._refresh_data_cache()
                
                # Wait for next cycle
                await asyncio.sleep(self.config.data_refresh_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _heartbeat_loop(self):
        """Heartbeat loop for system monitoring"""
        while self.is_running:
            try:
                self.last_heartbeat=datetime.now()
                
                # Send heartbeat alert
                if self.config.enable_alerts:
                    await self.integration_manager.alert_system.send_alert(
                        "SYSTEM_HEARTBEAT",
                        "LOW",
                        f"Production Strategy Manager heartbeat - {len(self.strategies)} strategies active"
                    )
                
                # Wait 5 minutes between heartbeats
                await asyncio.sleep(300)
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(60)
    
    async def _performance_tracking_loop(self):
        """Performance tracking loop"""
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
        """Monitor positions across all strategies"""
        try:
            for strategy_name, strategy in self.strategies.items():
                try:
                    if hasattr(strategy, 'monitor_positions'):
                        await strategy.monitor_positions()
                except Exception as e:
                    self.logger.error(f"Error monitoring positions for {strategy_name}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error in monitor_all_positions: {e}")
    
    async def _check_risk_limits(self):
        """Check risk limits across all strategies"""
        try:
            # Get total portfolio risk
            total_risk=await self.integration_manager.get_total_risk()
            portfolio_value=await self.integration_manager.get_portfolio_value()
            
            if portfolio_value > 0:
                risk_percentage=float(total_risk / portfolio_value)
                
                if risk_percentage > self.config.max_total_risk:
                    await self.integration_manager.alert_system.send_alert(
                        "RISK_ALERT",
                        "HIGH",
                        f"Total risk {risk_percentage:.1%} exceeds limit {self.config.max_total_risk:.1%}"
                    )
                    
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
    
    async def _refresh_data_cache(self):
        """Refresh data cache"""
        try:
            # Clear old cache entries
            self.data_provider.clear_cache()
            
        except Exception as e:
            self.logger.error(f"Error refreshing data cache: {e}")
    
    async def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Get portfolio summary
            portfolio_summary=self.integration_manager.get_portfolio_summary()
            
            # Get strategy performance
            strategy_performance={}
            for strategy_name, strategy in self.strategies.items():
                try:
                    if hasattr(strategy, 'get_strategy_status'):
                        strategy_performance[strategy_name] = strategy.get_strategy_status()
                except Exception as e:
                    self.logger.error(f"Error getting status for {strategy_name}: {e}")
            
            # Update metrics
            self.performance_metrics={
                'timestamp':datetime.now().isoformat(),
                'system_uptime':(datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                'portfolio':portfolio_summary,
                'strategies':strategy_performance,
                'data_cache_stats':self.data_provider.get_cache_stats()
            }
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")

    async def _analytics_loop(self):
        """Advanced analytics calculation loop"""
        while self.is_running:
            try:
                await self._calculate_advanced_analytics()
                await asyncio.sleep(self.config.analytics_update_interval)
            except Exception as e:
                self.logger.error(f"Error in analytics loop: {e}")
                await asyncio.sleep(300)

    async def _regime_adaptation_loop(self):
        """Market regime adaptation loop"""
        while self.is_running:
            try:
                await self._update_regime_adaptation()
                await asyncio.sleep(self.config.regime_adaptation_interval)
            except Exception as e:
                self.logger.error(f"Error in regime adaptation loop: {e}")
                await asyncio.sleep(300)

    async def _calculate_advanced_analytics(self):
        """Calculate comprehensive performance analytics"""
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
                end_date=datetime.now()
            )

            # Store in history
            self.analytics_history.append(metrics)
            if len(self.analytics_history) > 100:  # Keep last 100 calculations
                self.analytics_history = self.analytics_history[-100:]

            # Generate and log analytics report
            report = self.advanced_analytics.generate_analytics_report(metrics)
            self.logger.info("Advanced analytics updated")

            # Send alert for significant changes
            if len(self.analytics_history) > 1:
                await self._check_analytics_alerts(metrics)

        except Exception as e:
            self.logger.error(f"Error calculating advanced analytics: {e}")

    async def _update_regime_adaptation(self):
        """Update market regime adaptation"""
        try:
            if not self.market_regime_adapter:
                return

            # Get current market data
            market_data = await self._get_market_data_for_regime()
            current_positions = await self._get_current_positions()

            # Generate strategy adaptation
            adaptation = await self.market_regime_adapter.generate_strategy_adaptation(
                market_data=market_data,
                current_positions=current_positions
            )

            # Check if adaptation changed
            if (not self.current_regime_adaptation or
                adaptation.regime != self.current_regime_adaptation.regime or
                abs(adaptation.confidence - self.current_regime_adaptation.confidence) > 0.1):

                self.current_regime_adaptation = adaptation
                await self._apply_regime_adaptation(adaptation)

                # Send regime change alert
                await self.integration_manager.alert_system.send_alert(
                    "REGIME_CHANGE",
                    "MEDIUM",
                    f"Market regime: {adaptation.regime.value} "
                    f"(confidence: {adaptation.confidence:.1%})"
                )

                self.logger.info(f"Applied regime adaptation: {adaptation.regime.value}")

        except Exception as e:
            self.logger.error(f"Error updating regime adaptation: {e}")

    async def _get_portfolio_returns(self) -> List[float]:
        """Get portfolio returns for analytics"""
        try:
            # Get portfolio value history from integration manager
            # This would need to be implemented in integration manager
            portfolio_history = await self.integration_manager.get_portfolio_history(days=180)

            if len(portfolio_history) < 2:
                return []

            # Calculate daily returns
            returns = []
            for i in range(1, len(portfolio_history)):
                prev_value = portfolio_history[i-1]['value']
                curr_value = portfolio_history[i]['value']
                if prev_value > 0:
                    daily_return = (curr_value - prev_value) / prev_value
                    returns.append(daily_return)

            return returns

        except Exception as e:
            self.logger.error(f"Error getting portfolio returns: {e}")
            return []

    async def _get_benchmark_returns(self) -> List[float]:
        """Get benchmark returns (SPY) for comparison"""
        try:
            # Get SPY data for benchmark comparison
            spy_data = await self.data_provider.get_historical_data('SPY', days=180)

            if len(spy_data) < 2:
                return []

            # Calculate daily returns
            returns = []
            for i in range(1, len(spy_data)):
                prev_price = spy_data[i-1].price
                curr_price = spy_data[i].price
                if prev_price > 0:
                    daily_return = (curr_price - prev_price) / prev_price
                    returns.append(daily_return)

            return returns

        except Exception as e:
            self.logger.error(f"Error getting benchmark returns: {e}")
            return []

    async def _get_market_data_for_regime(self) -> Dict[str, Any]:
        """Get market data for regime detection"""
        try:
            # Get key market indicators
            market_data = {}

            # Get SPY data (primary indicator)
            spy_data = await self.data_provider.get_current_price('SPY')
            if spy_data:
                market_data['SPY'] = {
                    'price': spy_data.price,
                    'volume': getattr(spy_data, 'volume', 1000000),
                    'high': getattr(spy_data, 'high', spy_data.price * 1.01),
                    'low': getattr(spy_data, 'low', spy_data.price * 0.99)
                }

            # Add volatility indicator
            vix_data = await self.data_provider.get_current_price('VIX')
            if vix_data:
                market_data['volatility'] = float(vix_data.price) / 100.0

            return market_data

        except Exception as e:
            self.logger.error(f"Error getting market data for regime: {e}")
            return {}

    async def _get_current_positions(self) -> Dict[str, Any]:
        """Get current positions for regime adaptation"""
        try:
            return await self.integration_manager.get_all_positions()
        except Exception as e:
            self.logger.error(f"Error getting current positions: {e}")
            return {}

    async def _apply_regime_adaptation(self, adaptation: StrategyAdaptation):
        """Apply regime adaptation to strategies"""
        try:
            # Update strategy parameters based on regime adaptation
            for strategy_name, strategy in self.strategies.items():
                try:
                    # Check if strategy should be enabled/disabled
                    if strategy_name in adaptation.disabled_strategies:
                        if hasattr(strategy, 'set_enabled'):
                            await strategy.set_enabled(False)
                        self.logger.info(f"Disabled strategy {strategy_name} for regime {adaptation.regime.value}")

                    elif strategy_name in adaptation.recommended_strategies:
                        if hasattr(strategy, 'set_enabled'):
                            await strategy.set_enabled(True)

                        # Apply parameter adjustments
                        if hasattr(strategy, 'update_parameters'):
                            regime_params = {
                                'position_multiplier': adaptation.position_size_multiplier,
                                'max_risk': adaptation.max_risk_per_trade,
                                'stop_loss_adjustment': adaptation.stop_loss_adjustment,
                                'take_profit_adjustment': adaptation.take_profit_adjustment,
                                **adaptation.parameter_adjustments
                            }
                            await strategy.update_parameters(regime_params)

                        self.logger.info(f"Updated strategy {strategy_name} for regime {adaptation.regime.value}")

                except Exception as e:
                    self.logger.error(f"Error applying adaptation to {strategy_name}: {e}")

        except Exception as e:
            self.logger.error(f"Error applying regime adaptation: {e}")

    async def _check_analytics_alerts(self, metrics: PerformanceMetrics):
        """Check for analytics-based alerts"""
        try:
            # Check for significant performance changes
            if len(self.analytics_history) > 1:
                prev_metrics = self.analytics_history[-2]

                # Check for significant drawdown increase
                if metrics.max_drawdown > prev_metrics.max_drawdown * 1.5:
                    await self.integration_manager.alert_system.send_alert(
                        "DRAWDOWN_ALERT",
                        "HIGH",
                        f"Max drawdown increased to {metrics.max_drawdown:.2%}"
                    )

                # Check for Sharpe ratio degradation
                if metrics.sharpe_ratio < prev_metrics.sharpe_ratio * 0.7:
                    await self.integration_manager.alert_system.send_alert(
                        "PERFORMANCE_ALERT",
                        "MEDIUM",
                        f"Sharpe ratio declined to {metrics.sharpe_ratio:.2f}"
                    )

        except Exception as e:
            self.logger.error(f"Error checking analytics alerts: {e}")

    def get_advanced_analytics_summary(self) -> Dict[str, Any]:
        """Get advanced analytics summary"""
        if not self.analytics_history:
            return {'status': 'no_data'}

        latest_metrics = self.analytics_history[-1]
        return {
            'status': 'active',
            'total_return': latest_metrics.total_return,
            'annualized_return': latest_metrics.annualized_return,
            'sharpe_ratio': latest_metrics.sharpe_ratio,
            'max_drawdown': latest_metrics.max_drawdown,
            'volatility': latest_metrics.volatility,
            'win_rate': latest_metrics.win_rate,
            'var_95': latest_metrics.var_95,
            'last_updated': latest_metrics.period_end.isoformat(),
            'trading_days': latest_metrics.trading_days
        }

    def get_regime_adaptation_summary(self) -> Dict[str, Any]:
        """Get regime adaptation summary"""
        if not self.market_regime_adapter:
            return {'status': 'disabled'}

        return self.market_regime_adapter.get_adaptation_summary()

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'is_running':self.is_running,
            'start_time':self.start_time.isoformat() if self.start_time else None,
            'last_heartbeat':self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            'active_strategies':len(self.strategies),
            'strategy_status':{
                name: strategy.get_strategy_status() if hasattr(strategy, 'get_strategy_status') else {}
                for name, strategy in self.strategies.items()
            },
            'performance_metrics':self.performance_metrics,
            'configuration':{
                'paper_trading':self.config.paper_trading,
                'max_total_risk':self.config.max_total_risk,
                'max_position_size':self.config.max_position_size,
                'enabled_strategies':list(self.strategies.keys()),
                'advanced_analytics_enabled': self.config.enable_advanced_analytics,
                'market_regime_adaptation_enabled': self.config.enable_market_regime_adaptation
            },
            'advanced_analytics': self.get_advanced_analytics_summary(),
            'market_regime': self.get_regime_adaptation_summary()
        }
    
    def get_strategy_performance(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Get performance for specific strategy"""
        try:
            if strategy_name in self.strategies:
                strategy=self.strategies[strategy_name]
                if hasattr(strategy, 'get_strategy_status'):
                    return strategy.get_strategy_status()
            return None
        except Exception as e:
            self.logger.error(f"Error getting performance for {strategy_name}: {e}")
            return None


# Factory function
def create_production_strategy_manager(config: ProductionStrategyManagerConfig) -> ProductionStrategyManager:
    """Create ProductionStrategyManager instance"""
    return ProductionStrategyManager(config)
