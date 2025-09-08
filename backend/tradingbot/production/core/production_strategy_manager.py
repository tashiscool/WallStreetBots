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


class StrategyProfile(str, Enum):
    research_2024 = "research_2024"
    wsb_2025 = "wsb_2025"


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
    paper_trading: bool = True
    user_id: int = 1
    
    # Strategy configurations
    strategies: Dict[str, StrategyConfig] = field(default_factory=dict)
    
    # Risk management
    max_total_risk: float = 0.50  # 50% max total risk
    max_position_size: float = 0.20  # 20% max per position
    
    # Data settings
    data_refresh_interval: int = 30  # seconds
    
    # Alert settings
    enable_alerts: bool = True
    
    # NEW: choose a preset
    profile: StrategyProfile = StrategyProfile.research_2024


def _preset_defaults(profile: StrategyProfile) -> Dict[str, StrategyConfig]:
    """Return default strategy configs for a given profile."""
    if profile == StrategyProfile.research_2024:
        # === RESEARCH 2024 PRESET (Conservative) ===
        return {
            'wsb_dip_bot': StrategyConfig('wsb_dip_bot', True, 0.25, 'high', {
                'run_lookback_days': 7, 'run_threshold': 0.08, 'dip_threshold': -0.02,
                'target_dte_days': 21, 'otm_percentage': 0.03, 'target_multiplier': 2.5,
                'delta_target': 0.50, 'wsb_sentiment_weight': 0.3
            }),
            'earnings_protection': StrategyConfig('earnings_protection', True, 0.06, 'high', {
                'iv_percentile_threshold': 70, 'min_implied_move': 0.04,
                'max_days_to_earnings': 30, 'min_days_to_earnings': 7,
                'preferred_strategies': ['long_straddle','reverse_iron_condor'],
                'delta_range': (0.25, 0.30), 'profit_target': 0.50,
                'iv_crush_protection': True, 'straddle_vs_strangle': 'strangle',
                'atm_for_low_price': True, 'ric_for_high_price': True,
                'avoid_low_iv': True, 'exit_before_announcement': False
            }),
            'index_baseline': StrategyConfig('index_baseline', True, 0.60, 'medium', {
                'benchmarks': ['SPY','QQQ','IWM','VTI','ARKK'],
                'target_allocation': 0.60, 'rebalance_threshold': 0.03,
                'tax_loss_threshold': -0.05, 'momentum_factor': 0.2, 'volatility_target': 0.20
            }),
            'wheel_strategy': StrategyConfig('wheel_strategy', True, 0.20, 'medium', {
                'target_iv_rank': 50, 'target_dte_range': (30,45),
                'target_delta_range': (0.15,0.30), 'max_positions': 8,
                'min_premium_dollars': 50, 'profit_target': 0.50, 'roll_at_dte': 21,
                'assignment_acceptance': True, 'covered_call_delta': 0.20,
                'portfolio_allocation': 0.40,
                'diversified_watchlist': ['SPY','QQQ','AAPL','MSFT','GOOGL','AMZN'],
                'avoid_earnings': True, 'min_liquidity_score': 0.7, 'fundamental_screen': True
            }),
            'momentum_weeklies': StrategyConfig('momentum_weeklies', True, 0.05, 'high', {
                'watchlist': ['SPY','QQQ','IWM'], 'max_positions': 3,
                'min_volume_spike': 1.5, 'min_momentum_threshold': 0.015,
                'target_dte_range': (0,2), 'target_delta_range': (0.05,0.15),
                'min_premium': 1.00, 'profit_target': 0.50, 'stop_loss': 2.0,
                'preferred_day': 'monday', 'entry_time_after': '10:00', 'avg_hold_hours': 2,
                'wing_width': 30
            }),
            'debit_spreads': StrategyConfig('debit_spreads', True, 0.15, 'high', {
                'watchlist': ['TSLA','NVDA','AMD','PLTR','GME','AMC','MSTR','COIN','AAPL','MSFT','GOOGL','META','AMZN','NFLX','SNOW','UBER','ROKU','SQ','PYPL'],
                'max_positions': 12, 'min_dte': 10, 'max_dte': 35, 'min_risk_reward': 1.2,
                'min_trend_strength': 0.4, 'max_iv_rank': 90, 'min_volume_score': 0.2,
                'profit_target': 0.40, 'stop_loss': 0.70, 'time_exit_dte': 3, 'momentum_multiplier': 1.5
            }),
            'leaps_tracker': StrategyConfig('leaps_tracker', True, 0.03, 'medium', {
                'max_positions': 5, 'max_total_allocation': 0.15,
                'min_dte': 365, 'max_dte': 730, 'delta_strategy': 'mixed',
                'high_delta_range': (0.70,0.80), 'low_delta_range': (0.10,0.30),
                'profit_levels': [100,200,300], 'scale_out_percentage': 25,
                'stop_loss': 1.0, 'time_exit_dte': 90, 'entry_staging': True, 'staging_periods': 3,
                'focus_sectors': ['technology','growth'], 'min_premium_percentage': 0.10
            }),
            'swing_trading': StrategyConfig('swing_trading', True, 0.05, 'high', {
                'watchlist': ['TSLA','NVDA','AMD','PLTR','GME','AMC','MSTR','COIN','SPY','QQQ','AAPL','MSFT','GOOGL','META','NFLX','ARKK','TQQQ','SOXL','SPXL','XLK'],
                'max_positions': 8, 'max_expiry_days': 35, 'min_strength_score': 45.0,
                'min_volume_multiple': 1.5, 'min_breakout_strength': 0.001,
                'min_premium': 0.15, 'profit_targets': [30,60,150], 'stop_loss_pct': 50,
                'max_hold_hours': 24, 'end_of_day_exit_hour': 16, 'meme_stock_multiplier': 1.5,
                'wsb_momentum_factor': 0.3
            }),
            'spx_credit_spreads': StrategyConfig('spx_credit_spreads', True, 0.04, 'medium', {
                'strategy_type': 'iron_condor', 'target_short_delta': 0.15,
                'target_dte_range': (28,35), 'profit_target_pct': 0.50, 'stop_loss_multiple': 2.2,
                'max_dte': 45, 'min_credit': 1.00, 'max_spread_width': 50, 'max_positions': 3,
                'long_delta': 0.05, 'entry_time_preference': 'morning', 'roll_at_dte': 21,
                'min_option_volume': 100, 'min_option_oi': 50, 'double_stop_loss_protection': True,
                'vix_filter': 25, 'avoid_earnings_days': True
            }),
            'lotto_scanner': StrategyConfig('lotto_scanner', True, 0.01, 'extreme', {
                'max_risk_pct': 0.5, 'max_concurrent_positions': 3,
                'profit_targets': [500,1000,2000], 'stop_loss_pct': 1.0,
                'min_win_probability': 0.05, 'max_dte': 2, 'catalyst_required': True,
                'volume_spike_min': 5.0, 'focus_tickers': ['SPY','QQQ','TSLA','NVDA'],
                'earnings_window_only': True, 'otm_threshold': 0.10, 'iv_spike_required': True,
                'max_premium_cost': 2.00
            })
        }
    
    # === WSB 2025 PRESET (Aggressive) ===
    meme_core = ['NVDA','TSLA','SMCI','ARM','AAPL','MSFT','GOOGL','AMZN','META','COIN','MSTR','PLTR','AMD','SPY','QQQ','IWM','GME','AMC']
    return {
        'wsb_dip_bot': StrategyConfig('wsb_dip_bot', True, 0.28, 'high', {
            'run_lookback_days': 5,'run_threshold': 0.06,'dip_threshold': -0.015,
            'target_dte_days': 14,'otm_percentage': 0.02,'target_multiplier': 3.0,
            'delta_target': 0.55,'wsb_sentiment_weight': 0.40,'use_intraday_confirm': True,'min_option_volume': 5000
        }),
        'earnings_protection': StrategyConfig('earnings_protection', True, 0.22, 'high', {
            'iv_percentile_threshold': 40,'min_implied_move': 0.025,'max_days_to_earnings': 10,'min_days_to_earnings': 0,
            'preferred_strategies': ['long_straddle','long_strangle','calendar_spread','protective_hedge'],
            'earnings_momentum_weight': 0.50,'wsb_sentiment_multiplier': 1.5,
            'watchlist': [t for t in meme_core if t not in ('SPY','QQQ','IWM')]
        }),
        'index_baseline': StrategyConfig('index_baseline', True, 0.55, 'medium', {
            'benchmarks': ['SPY','QQQ','IWM','VTI','ARKK','SMH','SOXX'],
            'target_allocation': 0.55,'rebalance_threshold': 0.03,'tax_loss_threshold': -0.05,
            'momentum_factor': 0.25,'volatility_target': 0.25
        }),
        'wheel_strategy': StrategyConfig('wheel_strategy', True, 0.42, 'high', {
            'target_iv_rank': 25,'target_dte_range': (14,28),'target_delta_range': (0.25,0.45),
            'max_positions': 20,'min_premium_dollars': 20,'profit_target': 0.50,'max_loss_pct': 0.75,
            'assignment_buffer_days': 2,'gamma_squeeze_factor': 0.30,
            'watchlist': [t for t in meme_core if t not in ('SPY','QQQ','IWM')]
        }),
        'momentum_weeklies': StrategyConfig('momentum_weeklies', True, 0.10, 'high', {
            'watchlist': meme_core,'max_positions': 8,'min_volume_spike': 1.8,'min_momentum_threshold': 0.008,
            'target_dte_range': (0,5),'otm_range': (0.01,0.08),'min_premium': 0.20,
            'profit_target': 0.15,'stop_loss': 0.80,'time_exit_hours': 6,'use_0dte_priority': True
        }),
        'debit_spreads': StrategyConfig('debit_spreads', True, 0.18, 'high', {
            'watchlist': [t for t in meme_core if t not in ('SPY','QQQ','IWM')],
            'max_positions': 12,'min_dte': 7,'max_dte': 21,'min_risk_reward': 1.0,
            'min_trend_strength': 0.35,'max_iv_rank': 92,'min_volume_score': 0.2,
            'profit_target': 0.45,'stop_loss': 0.70,'time_exit_dte': 2,'momentum_multiplier': 1.6
        }),
        'leaps_tracker': StrategyConfig('leaps_tracker', True, 0.16, 'high', {
            'max_positions': 8,'max_total_allocation': 0.40,'min_dte': 180,'max_dte': 540,
            'min_composite_score': 40,'min_entry_timing_score': 30,'max_exit_timing_score': 85,
            'profit_levels': [50,100,300,600],'scale_out_percentage': 20,'stop_loss': 0.70,'time_exit_dte': 45,
            'meme_stock_bonus': 25,'wsb_sentiment_weight': 0.35,
            'watchlist': [t for t in meme_core if t not in ('SPY','QQQ','IWM')]
        }),
        'swing_trading': StrategyConfig('swing_trading', True, 0.06, 'high', {
            'watchlist': meme_core,'max_positions': 10,'max_expiry_days': 35,'min_strength_score': 42.0,
            'min_volume_multiple': 1.4,'min_breakout_strength': 0.0008,'min_premium': 0.15,
            'profit_targets': [30,60,150],'stop_loss_pct': 50,'max_hold_hours': 36,'end_of_day_exit_hour': 16,
            'meme_stock_multiplier': 1.6,'wsb_momentum_factor': 0.35
        }),
        'spx_credit_spreads': StrategyConfig('spx_credit_spreads', True, 0.12, 'high', {
            'use_0dte_priority': True,'target_short_delta': 0.20,'profit_target_pct': 0.50,'stop_loss_pct': 3.0,
            'max_dte': 2,'min_credit': 0.40,'max_spread_width': 100,'max_positions': 12,'risk_free_rate': 0.05,
            'target_iv_percentile': 18,'min_option_volume': 100,'gamma_squeeze_factor': 0.30,
            'vix_momentum_weight': 0.25,'market_regime_filter': False,'entry_time_window_et': (10,15.5)
        }),
        'lotto_scanner': StrategyConfig('lotto_scanner', True, 0.04, 'extreme', {
            '0dte_only': True,'max_risk_pct': 2.5,'max_concurrent_positions': 10,
            'profit_targets': [150,250,400],'stop_loss_pct': 0.90,'max_dte': 1
        })
    }


def _apply_profile_risk_overrides(cfg: ProductionStrategyManagerConfig):
    """Tighten/loosen top-level risk based on profile."""
    if cfg.profile == StrategyProfile.wsb_2025:
        cfg.max_total_risk = 0.65
        cfg.max_position_size = 0.30
        cfg.data_refresh_interval = 10
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
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Apply profile-specific risk overrides
        _apply_profile_risk_overrides(self.config)
        
        # Initialize core components
        self.integration_manager = ProductionIntegrationManager(
            config.alpaca_api_key,
            config.alpaca_secret_key,
            config.paper_trading,
            config.user_id
        )
        
        self.data_provider = ProductionDataProvider(
            config.alpaca_api_key,
            config.alpaca_secret_key
        )
        
        # Initialize strategies
        self.strategies: Dict[str, Any] = {}
        self._initialize_strategies()
        
        # System state
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.last_heartbeat: Optional[datetime] = None
        
        # Performance tracking
        self.performance_metrics: Dict[str, Any] = {}
        
        self.logger.info(f"ProductionStrategyManager initialized with profile: {config.profile}")
    
    def _initialize_strategies(self):
        """Initialize all enabled strategies"""
        try:
            # Get preset defaults based on configured profile
            default_configs = _preset_defaults(self.config.profile)
            
            
            # Merge with user configurations
            strategy_configs = {**default_configs, **self.config.strategies}
            
            # Initialize strategies
            for strategy_name, strategy_config in strategy_configs.items():
                if strategy_config.enabled:
                    try:
                        strategy = self._create_strategy(strategy_name, strategy_config)
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
            if strategy_name == 'wsb_dip_bot':
                return create_production_wsb_dip_bot(
                    self.integration_manager,
                    self.data_provider,
                    config.parameters
                )
            elif strategy_name == 'earnings_protection':
                return create_production_earnings_protection(
                    self.integration_manager,
                    self.data_provider,
                    config.parameters
                )
            elif strategy_name == 'index_baseline':
                return create_production_index_baseline(
                    self.integration_manager,
                    self.data_provider,
                    config.parameters
                )
            elif strategy_name == 'wheel_strategy':
                return create_production_wheel_strategy(
                    self.integration_manager,
                    self.data_provider,
                    config.parameters
                )
            elif strategy_name == 'momentum_weeklies':
                return create_production_momentum_weeklies(
                    self.integration_manager,
                    self.data_provider,
                    config.parameters
                )
            elif strategy_name == 'debit_spreads':
                return create_production_debit_spreads(
                    self.integration_manager,
                    self.data_provider,
                    config.parameters
                )
            elif strategy_name == 'leaps_tracker':
                return create_production_leaps_tracker(
                    self.integration_manager,
                    self.data_provider,
                    config.parameters
                )
            elif strategy_name == 'swing_trading':
                return create_production_swing_trading(
                    self.integration_manager,
                    self.data_provider,
                    config.parameters
                )
            elif strategy_name == 'spx_credit_spreads':
                return create_production_spx_credit_spreads(
                    self.integration_manager,
                    self.data_provider,
                    config.parameters
                )
            elif strategy_name == 'lotto_scanner':
                return create_production_lotto_scanner(
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
    
    async def start_all_strategies(self) -> bool:
        """Start all enabled strategies"""
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
                    asyncio.create_task(strategy.run_strategy())
                    started_count += 1
                    self.logger.info(f"Started strategy: {strategy_name}")
                except Exception as e:
                    self.logger.error(f"Failed to start strategy {strategy_name}: {e}")
            
            if started_count > 0:
                self.is_running = True
                self.start_time = datetime.now()
                
                # Start monitoring tasks
                asyncio.create_task(self._monitoring_loop())
                asyncio.create_task(self._heartbeat_loop())
                asyncio.create_task(self._performance_tracking_loop())
                
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
            
            self.is_running = False
            
            # Strategies will stop when their run loops exit
            # In a more sophisticated implementation, we would send stop signals
            
            self.logger.info("All strategies stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping strategies: {e}")
    
    async def _validate_system_state(self) -> bool:
        """Validate system state before starting"""
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
                self.last_heartbeat = datetime.now()
                
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
            total_risk = await self.integration_manager.get_total_risk()
            portfolio_value = await self.integration_manager.get_portfolio_value()
            
            if portfolio_value > 0:
                risk_percentage = float(total_risk / portfolio_value)
                
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
            portfolio_summary = self.integration_manager.get_portfolio_summary()
            
            # Get strategy performance
            strategy_performance = {}
            for strategy_name, strategy in self.strategies.items():
                try:
                    if hasattr(strategy, 'get_strategy_status'):
                        strategy_performance[strategy_name] = strategy.get_strategy_status()
                except Exception as e:
                    self.logger.error(f"Error getting status for {strategy_name}: {e}")
            
            # Update metrics
            self.performance_metrics = {
                'timestamp': datetime.now().isoformat(),
                'system_uptime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                'portfolio': portfolio_summary,
                'strategies': strategy_performance,
                'data_cache_stats': self.data_provider.get_cache_stats()
            }
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            'active_strategies': len(self.strategies),
            'strategy_status': {
                name: strategy.get_strategy_status() if hasattr(strategy, 'get_strategy_status') else {}
                for name, strategy in self.strategies.items()
            },
            'performance_metrics': self.performance_metrics,
            'configuration': {
                'paper_trading': self.config.paper_trading,
                'max_total_risk': self.config.max_total_risk,
                'max_position_size': self.config.max_position_size,
                'enabled_strategies': list(self.strategies.keys())
            }
        }
    
    def get_strategy_performance(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Get performance for specific strategy"""
        try:
            if strategy_name in self.strategies:
                strategy = self.strategies[strategy_name]
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
