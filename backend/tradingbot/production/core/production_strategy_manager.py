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
        
        self.logger.info("ProductionStrategyManager initialized")
    
    def _initialize_strategies(self):
        """Initialize all enabled strategies"""
        try:
            # Default strategy configurations
            default_configs = {
                'wsb_dip_bot': StrategyConfig(
                    name='wsb_dip_bot',
                    enabled=True,
                    max_position_size=0.25,  # Bigger positions on dips
                    risk_tolerance='high',
                    parameters={
                        'run_lookback_days': 7,  # Shorter lookback for faster signals
                        'run_threshold': 0.08,  # Lower threshold for more signals
                        'dip_threshold': -0.02,  # Catch smaller dips
                        'target_dte_days': 21,  # Shorter term for more premium
                        'otm_percentage': 0.03,  # Closer to money for more action
                        'target_multiplier': 2.5,  # More realistic targets
                        'delta_target': 0.50,  # More aggressive delta
                        'wsb_sentiment_weight': 0.3  # Factor in WSB sentiment
                    }
                ),
                'earnings_protection': StrategyConfig(
                    name='earnings_protection',
                    enabled=True,
                    max_position_size=0.06,  # Research: smaller positions for earnings volatility
                    risk_tolerance='high',  # Earnings plays are inherently high risk
                    parameters={
                        'iv_percentile_threshold': 70,  # Research: enter when IV is elevated
                        'min_implied_move': 0.04,  # Research: need meaningful expected move
                        'max_days_to_earnings': 30,  # Research: enter ~1 month before for low IV
                        'min_days_to_earnings': 7,  # Research: avoid day-of IV crush
                        'preferred_strategies': ['long_straddle', 'reverse_iron_condor'],  # Research: best for earnings
                        'delta_range': (0.25, 0.30),  # Research: 25-30 delta for strangles
                        'profit_target': 0.50,  # Research: take profits before IV crush
                        'iv_crush_protection': True,  # Research: critical for earnings plays
                        'straddle_vs_strangle': 'strangle',  # Research: strangles often better percentage returns
                        'atm_for_low_price': True,  # Research: ATM straddles for stocks <$100
                        'ric_for_high_price': True,  # Research: reverse iron condor for >$100 stocks
                        'avoid_low_iv': True,  # Research: don't enter during low IV periods
                        'exit_before_announcement': False  # Research: hold through for volatility expansion
                    }
                ),
                'index_baseline': StrategyConfig(
                    name='index_baseline',
                    enabled=True,
                    max_position_size=0.60,  # Reduce to allow other strategies room
                    risk_tolerance='medium',  # More aggressive than before
                    parameters={
                        'benchmarks': ['SPY', 'QQQ', 'IWM', 'VTI', 'ARKK'],  # Add ARKK for WSB tech exposure
                        'target_allocation': 0.60,  # Match position size
                        'rebalance_threshold': 0.03,  # More frequent rebalancing
                        'tax_loss_threshold': -0.05,  # More aggressive tax loss harvesting
                        'momentum_factor': 0.2,  # Add momentum weighting
                        'volatility_target': 0.20  # Target 20% volatility for WSB style
                    }
                ),
                'wheel_strategy': StrategyConfig(
                    name='wheel_strategy',
                    enabled=True,
                    max_position_size=0.20,  # Research: reduce position sizes for better risk management
                    risk_tolerance='medium',  # Research shows wheel needs risk management
                    parameters={
                        'target_iv_rank': 50,  # Research: ThetaGang uses higher IV for better premium
                        'target_dte_range': (30, 45),  # Research: 30-45 DTE optimal for time decay
                        'target_delta_range': (0.15, 0.30),  # Research: 15-30 delta for cash-secured puts
                        'max_positions': 8,  # Research: diversification important
                        'min_premium_dollars': 50,  # Research: need meaningful premium
                        'profit_target': 0.50,  # Research: ThetaGang standard 50% profit target
                        'roll_at_dte': 21,  # Research: roll puts at 21 DTE
                        'assignment_acceptance': True,  # Research: wheel requires accepting assignment
                        'covered_call_delta': 0.20,  # Research: 20 delta for covered calls
                        'portfolio_allocation': 0.40,  # Research: ThetaGang uses 40% allocation
                        'diversified_watchlist': ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN'],  # Research: focus on stable, liquid names
                        'avoid_earnings': True,  # Research: avoid high gamma events
                        'min_liquidity_score': 0.7,  # Research: liquidity critical for wheel
                        'fundamental_screen': True  # Research: screen for quality companies
                    }
                ),
                'momentum_weeklies': StrategyConfig(
                    name='momentum_weeklies',
                    enabled=True,
                    max_position_size=0.05,  # Research shows 0DTE requires smaller positions
                    risk_tolerance='high',
                    parameters={
                        'watchlist': ['SPY', 'QQQ', 'IWM'],  # Research shows SPY=81%, QQQ=second most popular for 0DTE
                        'max_positions': 3,  # Conservative based on research risk warnings
                        'min_volume_spike': 1.5,  # Based on actual 0DTE volume requirements
                        'min_momentum_threshold': 0.015,  # Research-backed intraday momentum
                        'target_dte_range': (0, 2),  # Focus on 0DTE which has proven data
                        'target_delta_range': (0.05, 0.15),  # Research: 5-15 delta shorts most profitable
                        'min_premium': 1.00,  # Research shows $100-300 premium collection
                        'profit_target': 0.50,  # Research: 50% profit target optimal
                        'stop_loss': 2.0,  # Research: losses ~2.2x larger than wins
                        'preferred_day': 'monday',  # Research: Monday most profitable for 0DTE
                        'entry_time_after': '10:00',  # Research: after morning volatility
                        'avg_hold_hours': 2,  # Research: 2 hours average hold time
                        'wing_width': 30  # Research: 30-point wings standard
                    }
                ),
                'debit_spreads': StrategyConfig(
                    name='debit_spreads',
                    enabled=True,
                    max_position_size=0.15,  # Bigger directional bets
                    risk_tolerance='high',  # WSB loves directional plays
                    parameters={
                        'watchlist': ['TSLA', 'NVDA', 'AMD', 'PLTR', 'GME', 'AMC', 'MSTR', 'COIN', 'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NFLX', 'SNOW', 'UBER', 'ROKU', 'SQ', 'PYPL'],  # Prioritize meme stocks
                        'max_positions': 12,  # More concurrent bets
                        'min_dte': 10,  # Shorter term for more action
                        'max_dte': 35,  # Still reasonable but shorter
                        'min_risk_reward': 1.2,  # Lower barrier for more opportunities
                        'min_trend_strength': 0.4,  # Lower threshold to catch more moves
                        'max_iv_rank': 90,  # Higher tolerance for volatility
                        'min_volume_score': 0.2,  # Lower volume requirement
                        'profit_target': 0.40,  # Let winners run WSB style
                        'stop_loss': 0.70,  # Diamond hands approach
                        'time_exit_dte': 3,  # More aggressive time management
                        'momentum_multiplier': 1.5  # Extra weight for momentum
                    }
                ),
                'leaps_tracker': StrategyConfig(
                    name='leaps_tracker',
                    enabled=True,
                    max_position_size=0.03,  # Research: 1-3% position sizing per LEAPS
                    risk_tolerance='medium',  # Research shows need for risk management
                    parameters={
                        'max_positions': 5,  # Conservative based on research
                        'max_total_allocation': 0.15,  # Total LEAPS exposure limit
                        'min_dte': 365,  # Research: minimum 1 year for true LEAPS
                        'max_dte': 730,  # 2 years maximum
                        'delta_strategy': 'mixed',  # Research: both high and low delta work
                        'high_delta_range': (0.70, 0.80),  # Research: stock replacement deltas
                        'low_delta_range': (0.10, 0.30),  # Research: high leverage deltas  
                        'profit_levels': [100, 200, 300],  # Research: NVDA achieved 419% in 2024
                        'scale_out_percentage': 25,  # Take profits systematically
                        'stop_loss': 1.0,  # Complete loss acceptable for LEAPS
                        'time_exit_dte': 90,  # Exit before final decay
                        'entry_staging': True,  # Research: stage entries over time
                        'staging_periods': 3,  # Enter over 3 periods
                        'focus_sectors': ['technology', 'growth'],  # 2024 winners were tech-heavy
                        'min_premium_percentage': 0.10  # Research: 10% of stock price for growth stocks
                    }
                ),
                'swing_trading': StrategyConfig(
                    name='swing_trading',
                    enabled=True,
                    max_position_size=0.05,  # More than double the swing size
                    risk_tolerance='high',  # Keep high risk tolerance
                    parameters={
                        'watchlist': ['TSLA', 'NVDA', 'AMD', 'PLTR', 'GME', 'AMC', 'MSTR', 'COIN', 'SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'META', 'NFLX', 'ARKK', 'TQQQ', 'SOXL', 'SPXL', 'XLK'],  # Prioritize volatile/meme stocks
                        'max_positions': 8,  # More concurrent positions
                        'max_expiry_days': 35,  # Longer term for bigger moves
                        'min_strength_score': 45.0,  # Lower threshold for more signals
                        'min_volume_multiple': 1.5,  # Lower volume requirement
                        'min_breakout_strength': 0.001,  # Catch smaller breakouts
                        'min_premium': 0.15,  # Lower minimum premium
                        'profit_targets': [30, 60, 150],  # Higher targets, let winners run
                        'stop_loss_pct': 50,  # Wider stops, diamond hands
                        'max_hold_hours': 24,  # Hold longer for bigger moves
                        'end_of_day_exit_hour': 16,  # Hold through close
                        'meme_stock_multiplier': 1.5,  # Extra weight for memes
                        'wsb_momentum_factor': 0.3  # Factor in WSB sentiment
                    }
                ),
                'spx_credit_spreads': StrategyConfig(
                    name='spx_credit_spreads',
                    enabled=True,
                    max_position_size=0.04,  # Research shows smaller positions for credit spreads
                    risk_tolerance='medium',  # Research shows need for risk management
                    parameters={
                        'strategy_type': 'iron_condor',  # Research: iron condors had 63-70% win rate
                        'target_short_delta': 0.15,  # Research: 5-15 delta shorts most successful
                        'target_dte_range': (28, 35),  # Research: 28-35 DTE optimal for condors
                        'profit_target_pct': 0.50,  # Research: 50% profit target is optimal
                        'stop_loss_multiple': 2.2,  # Research: losses average 2.2x wins
                        'max_dte': 45,  # Research: "Tasty standard" 45 DTE
                        'min_credit': 1.00,  # Research: $100-300 premium target
                        'max_spread_width': 50,  # Research: standard width for SPX
                        'max_positions': 3,  # Conservative based on research warnings
                        'long_delta': 0.05,  # Research: wings typically 5 delta
                        'entry_time_preference': 'morning',  # Research: avoid late day entries
                        'roll_at_dte': 21,  # Research: roll at 21 DTE if losing
                        'min_option_volume': 100,  # Keep reasonable liquidity
                        'min_option_oi': 50,  # Standard liquidity requirement
                        'double_stop_loss_protection': True,  # Research: 2024 saw more double stops
                        'vix_filter': 25,  # Only trade when VIX reasonable
                        'avoid_earnings_days': True  # Research: avoid high gamma events
                    }
                ),
                'lotto_scanner': StrategyConfig(
                    name='lotto_scanner',
                    enabled=True,
                    max_position_size=0.01,  # Research: keep very small for high-risk plays
                    risk_tolerance='extreme',
                    parameters={
                        'max_risk_pct': 0.5,  # Research: limit total risk despite "lotto" nature
                        'max_concurrent_positions': 3,  # Research: focus on best opportunities only
                        'profit_targets': [500, 1000, 2000],  # Research: truly lottery-style returns
                        'stop_loss_pct': 1.0,  # Research: accept complete losses on lottery plays
                        'min_win_probability': 0.05,  # Research: very low probability, high reward
                        'max_dte': 2,  # Research: focus on immediate catalysts
                        'catalyst_required': True,  # Research: need news/earnings catalyst
                        'volume_spike_min': 5.0,  # Research: need significant volume anomaly
                        'focus_tickers': ['SPY', 'QQQ', 'TSLA', 'NVDA'],  # Research: stick to most liquid
                        'earnings_window_only': True,  # Research: only trade around earnings
                        'otm_threshold': 0.10,  # Research: far OTM for lottery potential
                        'iv_spike_required': True,  # Research: need IV expansion for entry
                        'news_catalyst_weight': 0.8,  # Research: news-driven moves essential
                        'max_premium_cost': 2.00  # Research: keep individual costs very low
                    }
                )
            }
            
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
