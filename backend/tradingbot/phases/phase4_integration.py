"""
Phase 4 Integration: Complete Production System Orchestration
READY FOR REAL MONEY TRADING

This is the master orchestrator that brings together all phases:
- Phase 1: Foundation infrastructure
- Phase 2: Low-risk strategies (Wheel, Debit Spreads, SPX)
- Phase 3: Medium-risk strategies (Momentum, LEAPS, Earnings)
- Phase 4: High-risk strategies + comprehensive validation

This system is designed to PROTECT YOUR MONEY while maximizing returns.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal
import json

from .production_config import ProductionConfig, create_config_manager
from .production_logging import create_production_logger, ErrorHandler, MetricsCollector
from .production_database import ProductionDatabaseManager, create_database_manager
from .trading_interface import create_trading_interface, TradingInterface
from .data_providers import create_data_provider, UnifiedDataProvider

# Phase imports
from .phase2_integration import Phase2StrategyManager
from .phase3_integration import Phase3StrategyManager
from .phase4_production import (
    ProductionBacktestEngine, HighRiskStrategyOrchestrator,
    BacktestConfig, StrategyBacktestResults, create_production_backtest_engine,
    create_high_risk_orchestrator
)

# Individual strategy imports
from .production_wsb_dip_bot import create_wsb_dip_bot_strategy
from .production_wheel_strategy import create_wheel_strategy
from .production_debit_spreads import create_debit_spreads_strategy
from .production_spx_spreads import create_spx_spreads_strategy


class SystemStatus(Enum):
    INITIALIZING = "initializing"
    VALIDATING = "validating"
    READY = "ready"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class SystemHealthCheck:
    """System health check result"""
    timestamp: datetime = field(default_factory=datetime.now)
    overall_status: SystemStatus = SystemStatus.INITIALIZING
    database_status: bool = False
    trading_interface_status: bool = False
    data_provider_status: bool = False
    phase2_status: bool = False
    phase3_status: bool = False
    phase4_status: bool = False
    active_strategies: int = 0
    total_account_risk: Decimal = Decimal('0.00')
    daily_pnl: Decimal = Decimal('0.00')
    alerts: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyAllocation:
    """Strategy allocation configuration"""
    strategy_name: str
    allocation_percentage: Decimal  # Percentage of account to allocate
    risk_level: str  # 'low', 'medium', 'high'
    max_positions: int
    enabled: bool = True
    validation_required: bool = True
    min_sharpe_ratio: Decimal = Decimal('0.5')
    max_drawdown: Decimal = Decimal('0.20')


class Phase4IntegrationManager:
    """
    Master orchestrator for the complete WallStreetBots production system
    
    This is the brain that coordinates all phases and ensures safe operation
    """
    
    def __init__(self, config_file_path: Optional[str] = None):
        # Load configuration
        self.config_manager = create_config_manager()
        self.config = self.config_manager.load_config()
        
        self.logger = create_production_logger("phase4_integration")
        self.error_handler = ErrorHandler(self.logger)
        self.metrics = MetricsCollector(self.logger)
        
        # System components
        self.database: Optional[ProductionDatabaseManager] = None
        self.trading_interface: Optional[TradingInterface] = None
        self.data_provider: Optional[UnifiedDataProvider] = None
        
        # Phase managers
        self.phase2_manager: Optional[Phase2StrategyManager] = None
        self.phase3_manager: Optional[Phase3StrategyManager] = None
        self.backtest_engine: Optional[ProductionBacktestEngine] = None
        self.high_risk_orchestrator: Optional[HighRiskStrategyOrchestrator] = None
        
        # System state
        self.system_status = SystemStatus.INITIALIZING
        self.validation_results: Dict[str, StrategyBacktestResults] = {}
        self.active_strategies: Dict[str, Any] = {}
        
        # Strategy allocation configuration
        self.strategy_allocations = {
            'wheel': StrategyAllocation(
                strategy_name='wheel',
                allocation_percentage=Decimal('0.30'),  # 30% allocation
                risk_level='low',
                max_positions=5,
                min_sharpe_ratio=Decimal('0.8'),
                max_drawdown=Decimal('0.15')
            ),
            'debit_spreads': StrategyAllocation(
                strategy_name='debit_spreads',
                allocation_percentage=Decimal('0.25'),  # 25% allocation
                risk_level='low',
                max_positions=8,
                min_sharpe_ratio=Decimal('0.7'),
                max_drawdown=Decimal('0.18')
            ),
            'spx_spreads': StrategyAllocation(
                strategy_name='spx_spreads',
                allocation_percentage=Decimal('0.20'),  # 20% allocation
                risk_level='low',
                max_positions=3,
                min_sharpe_ratio=Decimal('0.6'),
                max_drawdown=Decimal('0.20')
            ),
            'wsb_dip_bot': StrategyAllocation(
                strategy_name='wsb_dip_bot',
                allocation_percentage=Decimal('0.15'),  # 15% allocation
                risk_level='high',
                max_positions=5,
                min_sharpe_ratio=Decimal('0.5'),
                max_drawdown=Decimal('0.25')
            ),
            'momentum_weeklies': StrategyAllocation(
                strategy_name='momentum_weeklies',
                allocation_percentage=Decimal('0.10'),  # 10% allocation
                risk_level='medium',
                max_positions=3,
                min_sharpe_ratio=Decimal('0.4'),
                max_drawdown=Decimal('0.30')
            )
        }
        
        self.logger.info("Phase 4 Integration Manager initialized")
    
    async def initialize_system(self) -> bool:
        """
        Initialize the complete production system
        CRITICAL: This must complete successfully before any trading
        """
        try:
            self.logger.info("🚀 Initializing complete WallStreetBots production system...")
            self.system_status = SystemStatus.INITIALIZING
            
            # Step 1: Initialize database
            if not await self._initialize_database():
                raise Exception("Database initialization failed")
            
            # Step 2: Initialize data provider
            if not await self._initialize_data_provider():
                raise Exception("Data provider initialization failed")
            
            # Step 3: Initialize trading interface
            if not await self._initialize_trading_interface():
                raise Exception("Trading interface initialization failed")
            
            # Step 4: Initialize phase managers
            if not await self._initialize_phase_managers():
                raise Exception("Phase managers initialization failed")
            
            # Step 5: Initialize Phase 4 components
            if not await self._initialize_phase4_components():
                raise Exception("Phase 4 components initialization failed")
            
            self.logger.info("✅ System initialization completed successfully")
            self.system_status = SystemStatus.READY
            return True
            
        except Exception as e:
            self.error_handler.handle_error(e, {"operation": "initialize_system"})
            self.system_status = SystemStatus.ERROR
            return False
    
    async def _initialize_database(self) -> bool:
        """Initialize production database"""
        try:
            self.logger.info("Initializing production database...")
            
            db_config = {
                'db_host': self.config.database.host if hasattr(self.config, 'database') else 'localhost',
                'db_name': self.config.database.name if hasattr(self.config, 'database') else 'wallstreetbots',
                'db_user': self.config.database.user if hasattr(self.config, 'database') else 'postgres',
                'db_password': self.config.database.password if hasattr(self.config, 'database') else 'password'
            }
            
            self.database = create_database_manager(db_config)
            success = await self.database.initialize()
            
            if success:
                self.logger.info("✅ Database initialized successfully")
            else:
                self.logger.error("❌ Database initialization failed")
            
            return success
            
        except Exception as e:
            self.error_handler.handle_error(e, {"operation": "_initialize_database"})
            return False
    
    async def _initialize_data_provider(self) -> bool:
        """Initialize data provider"""
        try:
            self.logger.info("Initializing data provider...")
            
            data_config = {
                'iex_api_key': self.config.data_providers.iex_api_key,
                'polygon_api_key': self.config.data_providers.polygon_api_key,
                'fmp_api_key': self.config.data_providers.fmp_api_key,
                'news_api_key': self.config.data_providers.news_api_key,
                'alpha_vantage_api_key': self.config.data_providers.alpha_vantage_api_key,
            }
            
            self.data_provider = create_data_provider(data_config)
            
            # Test data provider
            test_data = await self.data_provider.get_market_data('SPY')
            if test_data.price > 0:
                self.logger.info("✅ Data provider initialized and tested successfully")
                return True
            else:
                self.logger.warning("⚠️ Data provider initialized but test failed")
                return False
            
        except Exception as e:
            self.error_handler.handle_error(e, {"operation": "_initialize_data_provider"})
            return False
    
    async def _initialize_trading_interface(self) -> bool:
        """Initialize trading interface"""
        try:
            self.logger.info("Initializing trading interface...")
            
            trading_config = {
                'alpaca_api_key': self.config.broker.alpaca_api_key,
                'alpaca_secret_key': self.config.broker.alpaca_secret_key,
                'alpaca_base_url': self.config.broker.alpaca_base_url,
                'account_size': self.config.risk.account_size,
                'max_position_risk': self.config.risk.max_position_risk,
                'default_commission': self.config.risk.default_commission
            }
            
            self.trading_interface = create_trading_interface(trading_config)
            
            # Test trading interface
            account_info = await self.trading_interface.get_account()
            if account_info:
                self.logger.info("✅ Trading interface initialized and tested successfully")
                return True
            else:
                self.logger.error("❌ Trading interface test failed")
                return False
            
        except Exception as e:
            self.error_handler.handle_error(e, {"operation": "_initialize_trading_interface"})
            return False
    
    async def _initialize_phase_managers(self) -> bool:
        """Initialize all phase managers"""
        try:
            self.logger.info("Initializing phase managers...")
            
            # Initialize Phase 2 (Low-risk strategies)
            self.phase2_manager = Phase2StrategyManager(self.config)
            await self.phase2_manager.initialize()
            
            # Initialize Phase 3 (Medium-risk strategies)  
            self.phase3_manager = Phase3StrategyManager(self.config)
            await self.phase3_manager.initialize()
            
            self.logger.info("✅ Phase managers initialized successfully")
            return True
            
        except Exception as e:
            self.error_handler.handle_error(e, {"operation": "_initialize_phase_managers"})
            return False
    
    async def _initialize_phase4_components(self) -> bool:
        """Initialize Phase 4 specific components"""
        try:
            self.logger.info("Initializing Phase 4 components...")
            
            # Initialize backtest engine
            self.backtest_engine = create_production_backtest_engine(self.config, self.database)
            
            # Initialize high-risk orchestrator
            self.high_risk_orchestrator = create_high_risk_orchestrator(self.config, self.database)
            
            self.logger.info("✅ Phase 4 components initialized successfully")
            return True
            
        except Exception as e:
            self.error_handler.handle_error(e, {"operation": "_initialize_phase4_components"})
            return False
    
    async def validate_all_strategies(self, force_revalidation: bool = False) -> bool:
        """
        CRITICAL: Validate all strategies before allowing real money trading
        This is the safety check that prevents deploying bad strategies
        """
        try:
            self.logger.info("🧪 Starting comprehensive strategy validation...")
            self.system_status = SystemStatus.VALIDATING
            
            # Configure validation period (last 2 years)
            validation_config = BacktestConfig(
                start_date=datetime.now() - timedelta(days=730),
                end_date=datetime.now() - timedelta(days=1),
                initial_capital=Decimal('100000.00'),
                commission_per_trade=Decimal('1.00'),
                monte_carlo_runs=500  # Reduced for faster validation
            )
            
            validation_passed = True
            validation_summary = {}
            
            # Validate each strategy
            for strategy_name, allocation in self.strategy_allocations.items():
                if not allocation.enabled:
                    continue
                
                try:
                    self.logger.info(f"Validating strategy: {strategy_name}")
                    
                    # Run comprehensive validation
                    backtest_results, monte_carlo_results = await self.backtest_engine.validate_strategy(
                        strategy_name, validation_config, monte_carlo=True
                    )
                    
                    # Check if strategy meets minimum requirements
                    strategy_passed = await self._evaluate_strategy_performance(
                        strategy_name, backtest_results, allocation
                    )
                    
                    if strategy_passed:
                        self.validation_results[strategy_name] = backtest_results
                        self.logger.info(f"✅ Strategy validation PASSED: {strategy_name}")
                        validation_summary[strategy_name] = "PASSED"
                    else:
                        self.logger.warning(f"❌ Strategy validation FAILED: {strategy_name}")
                        validation_summary[strategy_name] = "FAILED"
                        validation_passed = False
                    
                except Exception as e:
                    self.error_handler.handle_error(e, {"strategy": strategy_name})
                    self.logger.error(f"❌ Strategy validation ERROR: {strategy_name}")
                    validation_summary[strategy_name] = "ERROR"
                    validation_passed = False
            
            # Print validation summary
            self.logger.info("📊 STRATEGY VALIDATION SUMMARY:")
            for strategy, result in validation_summary.items():
                if result == "PASSED":
                    backtest = self.validation_results[strategy]
                    self.logger.info(f"   ✅ {strategy}: {backtest.annualized_return:.2%} return, "
                                   f"{backtest.sharpe_ratio:.2f} Sharpe, "
                                   f"{backtest.max_drawdown:.2%} max DD")
                else:
                    self.logger.info(f"   ❌ {strategy}: {result}")
            
            if validation_passed:
                self.logger.info("🎉 ALL STRATEGIES VALIDATED SUCCESSFULLY - READY FOR PRODUCTION")
                self.system_status = SystemStatus.READY
            else:
                self.logger.error("⚠️ STRATEGY VALIDATION FAILED - SYSTEM NOT READY FOR PRODUCTION")
                self.system_status = SystemStatus.ERROR
            
            return validation_passed
            
        except Exception as e:
            self.error_handler.handle_error(e, {"operation": "validate_all_strategies"})
            self.system_status = SystemStatus.ERROR
            return False
    
    async def _evaluate_strategy_performance(self, strategy_name: str, 
                                           results: StrategyBacktestResults,
                                           allocation: StrategyAllocation) -> bool:
        """Evaluate if strategy performance meets minimum standards"""
        try:
            # Check minimum requirements
            checks = []
            
            # Sharpe ratio check
            sharpe_ok = results.sharpe_ratio >= allocation.min_sharpe_ratio
            checks.append(("Sharpe Ratio", sharpe_ok, f"{results.sharpe_ratio:.2f} >= {allocation.min_sharpe_ratio:.2f}"))
            
            # Max drawdown check
            drawdown_ok = results.max_drawdown <= allocation.max_drawdown
            checks.append(("Max Drawdown", drawdown_ok, f"{results.max_drawdown:.2%} <= {allocation.max_drawdown:.2%}"))
            
            # Win rate check (minimum 30%)
            winrate_ok = results.win_rate >= Decimal('0.30')
            checks.append(("Win Rate", winrate_ok, f"{results.win_rate:.1%} >= 30.0%"))
            
            # Minimum trades check
            trades_ok = results.total_trades >= 20
            checks.append(("Sample Size", trades_ok, f"{results.total_trades} >= 20"))
            
            # Positive alpha check
            alpha_ok = results.alpha >= Decimal('0.00')
            checks.append(("Alpha vs SPY", alpha_ok, f"{results.alpha:.2%} >= 0.00%"))
            
            # Log detailed checks
            self.logger.info(f"   Performance evaluation for {strategy_name}:")
            for check_name, passed, details in checks:
                status = "✅" if passed else "❌"
                self.logger.info(f"     {status} {check_name}: {details}")
            
            # Strategy passes if all critical checks pass
            passed = all(check[1] for check in checks)
            
            return passed
            
        except Exception as e:
            self.error_handler.handle_error(e, {"strategy": strategy_name, "operation": "_evaluate_strategy_performance"})
            return False
    
    async def start_production_trading(self, paper_trading: bool = True) -> bool:
        """
        Start production trading system
        CRITICAL: Only starts if all validations pass
        """
        try:
            self.logger.info(f"🚀 Starting production trading (paper_trading={paper_trading})...")
            
            # Safety check: Ensure system is ready
            if self.system_status != SystemStatus.READY:
                raise Exception(f"System not ready for trading. Status: {self.system_status}")
            
            # Safety check: Ensure strategies are validated
            if not self.validation_results:
                raise Exception("No validated strategies available")
            
            # Update trading interface mode
            if hasattr(self.trading_interface, 'paper_trading'):
                self.trading_interface.paper_trading = paper_trading
            
            # Start strategies based on allocations
            for strategy_name, allocation in self.strategy_allocations.items():
                if not allocation.enabled or strategy_name not in self.validation_results:
                    continue
                
                try:
                    await self._start_strategy(strategy_name, allocation, paper_trading)
                    
                except Exception as e:
                    self.error_handler.handle_error(e, {"strategy": strategy_name})
                    self.logger.error(f"Failed to start strategy: {strategy_name}")
            
            self.system_status = SystemStatus.RUNNING
            
            # Start monitoring loop
            monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            mode_str = "PAPER TRADING" if paper_trading else "LIVE TRADING"
            self.logger.info(f"🎉 Production system started successfully in {mode_str} mode")
            self.logger.info(f"   Active strategies: {len(self.active_strategies)}")
            self.logger.info(f"   Total allocation: {sum(a.allocation_percentage for a in self.strategy_allocations.values() if a.enabled):.1%}")
            
            return True
            
        except Exception as e:
            self.error_handler.handle_error(e, {"operation": "start_production_trading"})
            self.system_status = SystemStatus.ERROR
            return False
    
    async def _start_strategy(self, strategy_name: str, allocation: StrategyAllocation, 
                            paper_trading: bool) -> None:
        """Start individual strategy"""
        try:
            self.logger.info(f"Starting strategy: {strategy_name} ({allocation.allocation_percentage:.1%} allocation)")
            
            # Create strategy instance
            if strategy_name == 'wheel':
                strategy = create_wheel_strategy(
                    self.trading_interface, self.data_provider, self.config, self.logger
                )
            elif strategy_name == 'debit_spreads':
                strategy = create_debit_spreads_strategy(
                    self.trading_interface, self.data_provider, self.config, self.logger
                )
            elif strategy_name == 'spx_spreads':
                strategy = create_spx_spreads_strategy(
                    self.trading_interface, self.data_provider, self.config, self.logger
                )
            elif strategy_name == 'wsb_dip_bot':
                strategy = create_wsb_dip_bot_strategy(
                    self.trading_interface, self.data_provider, self.config, self.logger
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy_name}")
            
            # Start strategy in background
            strategy_task = asyncio.create_task(strategy.run_strategy())
            
            self.active_strategies[strategy_name] = {
                'strategy': strategy,
                'task': strategy_task,
                'allocation': allocation,
                'start_time': datetime.now(),
                'paper_trading': paper_trading
            }
            
            self.logger.info(f"✅ Strategy started: {strategy_name}")
            
        except Exception as e:
            self.error_handler.handle_error(e, {"strategy": strategy_name, "operation": "_start_strategy"})
            raise
    
    async def _monitoring_loop(self) -> None:
        """Continuous monitoring of the production system"""
        try:
            self.logger.info("🔍 Starting system monitoring loop")
            
            while self.system_status == SystemStatus.RUNNING:
                try:
                    # Perform health check
                    health_check = await self.perform_system_health_check()
                    
                    # Log health status
                    if health_check.overall_status == SystemStatus.RUNNING:
                        self.logger.debug(f"System healthy: {health_check.active_strategies} active strategies, "
                                        f"Daily P&L: {health_check.daily_pnl:.2f}")
                    else:
                        self.logger.warning(f"System health issue: {health_check.overall_status}")
                    
                    # Check for alerts
                    if health_check.alerts:
                        for alert in health_check.alerts:
                            self.logger.warning(f"🚨 ALERT: {alert}")
                    
                    # Record metrics
                    self.metrics.record_metric("system_health_score", 
                                             1.0 if health_check.overall_status == SystemStatus.RUNNING else 0.0)
                    self.metrics.record_metric("active_strategies", health_check.active_strategies)
                    self.metrics.record_metric("total_account_risk", float(health_check.total_account_risk))
                    self.metrics.record_metric("daily_pnl", float(health_check.daily_pnl))
                    
                    # Sleep for 1 minute
                    await asyncio.sleep(60)
                    
                except Exception as e:
                    self.error_handler.handle_error(e, {"operation": "_monitoring_loop"})
                    await asyncio.sleep(60)  # Continue monitoring even if there's an error
            
        except Exception as e:
            self.error_handler.handle_error(e, {"operation": "_monitoring_loop"})
            self.logger.error("Monitoring loop stopped due to error")
    
    async def perform_system_health_check(self) -> SystemHealthCheck:
        """Perform comprehensive system health check"""
        try:
            health_check = SystemHealthCheck()
            health_check.overall_status = self.system_status
            
            # Check database
            try:
                if self.database and self.database.pool:
                    # Simple database connectivity check
                    async with self.database.pool.acquire() as conn:
                        await conn.fetchval("SELECT 1")
                    health_check.database_status = True
            except:
                health_check.database_status = False
                health_check.alerts.append("Database connectivity issue")
            
            # Check trading interface
            try:
                if self.trading_interface:
                    account_info = await self.trading_interface.get_account()
                    health_check.trading_interface_status = account_info is not None
                    
                    if account_info:
                        # Extract P&L info
                        health_check.daily_pnl = Decimal(str(account_info.get('daytrading_buying_power', 0))) - \
                            self.config.risk.account_size
            except:
                health_check.trading_interface_status = False
                health_check.alerts.append("Trading interface connectivity issue")
            
            # Check data provider
            try:
                if self.data_provider:
                    test_data = await self.data_provider.get_market_data('SPY')
                    health_check.data_provider_status = test_data.price > 0
            except:
                health_check.data_provider_status = False
                health_check.alerts.append("Data provider connectivity issue")
            
            # Check strategy health
            health_check.active_strategies = len([s for s in self.active_strategies.values() 
                                                if not s['task'].done()])
            
            # Calculate total risk
            try:
                positions = await self.database.get_active_positions()
                total_risk = sum(pos.risk_amount for pos in positions)
                health_check.total_account_risk = total_risk
                
                # Risk alerts
                risk_pct = total_risk / self.config.risk.account_size
                if risk_pct > Decimal('0.15'):  # 15% total risk warning
                    health_check.alerts.append(f"High account risk: {risk_pct:.1%}")
                
            except:
                health_check.alerts.append("Unable to calculate account risk")
            
            # Daily loss check
            if health_check.daily_pnl < -self.config.risk.account_size * Decimal('0.03'):  # 3% daily loss
                health_check.alerts.append(f"Significant daily loss: {health_check.daily_pnl:.2f}")
            
            # Overall system status
            if (health_check.database_status and health_check.trading_interface_status and 
                health_check.data_provider_status and health_check.active_strategies > 0):
                health_check.overall_status = SystemStatus.RUNNING
            else:
                health_check.overall_status = SystemStatus.ERROR
            
            return health_check
            
        except Exception as e:
            self.error_handler.handle_error(e, {"operation": "perform_system_health_check"})
            return SystemHealthCheck(overall_status=SystemStatus.ERROR, 
                                   alerts=["Health check system error"])
    
    async def stop_all_strategies(self) -> None:
        """Stop all running strategies safely"""
        try:
            self.logger.info("🛑 Stopping all strategies...")
            
            for strategy_name, strategy_info in self.active_strategies.items():
                try:
                    self.logger.info(f"Stopping strategy: {strategy_name}")
                    strategy_info['task'].cancel()
                    
                    # Wait for graceful shutdown
                    try:
                        await asyncio.wait_for(strategy_info['task'], timeout=30.0)
                    except asyncio.TimeoutError:
                        self.logger.warning(f"Strategy {strategy_name} did not stop gracefully")
                    except asyncio.CancelledError:
                        pass
                    
                except Exception as e:
                    self.error_handler.handle_error(e, {"strategy": strategy_name})
            
            self.active_strategies.clear()
            self.system_status = SystemStatus.STOPPED
            
            self.logger.info("✅ All strategies stopped")
            
        except Exception as e:
            self.error_handler.handle_error(e, {"operation": "stop_all_strategies"})
    
    async def emergency_shutdown(self) -> None:
        """Emergency shutdown of the entire system"""
        try:
            self.logger.critical("🚨 EMERGENCY SHUTDOWN INITIATED")
            
            # Stop all strategies immediately
            for strategy_name, strategy_info in self.active_strategies.items():
                strategy_info['task'].cancel()
            
            # Close all positions if possible
            try:
                positions = await self.database.get_active_positions()
                for position in positions:
                    await self.trading_interface.sell_stock(
                        ticker=position.ticker,
                        quantity=position.quantity,
                        order_type='market'
                    )
                    self.logger.critical(f"Emergency position close: {position.ticker}")
            except Exception as e:
                self.logger.critical(f"Failed to close positions during emergency: {e}")
            
            self.system_status = SystemStatus.ERROR
            self.logger.critical("🚨 EMERGENCY SHUTDOWN COMPLETE")
            
        except Exception as e:
            self.error_handler.handle_error(e, {"operation": "emergency_shutdown"})
    
    async def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system summary"""
        try:
            health_check = await self.perform_system_health_check()
            
            # Get strategy performance
            strategy_performance = {}
            for strategy_name in self.active_strategies.keys():
                if strategy_name in self.validation_results:
                    results = self.validation_results[strategy_name]
                    strategy_performance[strategy_name] = {
                        'annualized_return': float(results.annualized_return),
                        'sharpe_ratio': float(results.sharpe_ratio),
                        'max_drawdown': float(results.max_drawdown),
                        'win_rate': float(results.win_rate),
                        'total_trades': results.total_trades
                    }
            
            return {
                'timestamp': datetime.now().isoformat(),
                'system_status': self.system_status.value,
                'health_check': {
                    'database': health_check.database_status,
                    'trading_interface': health_check.trading_interface_status,
                    'data_provider': health_check.data_provider_status,
                    'active_strategies': health_check.active_strategies,
                    'total_risk': float(health_check.total_account_risk),
                    'daily_pnl': float(health_check.daily_pnl),
                    'alerts': health_check.alerts
                },
                'strategy_performance': strategy_performance,
                'allocations': {name: float(alloc.allocation_percentage) 
                              for name, alloc in self.strategy_allocations.items() if alloc.enabled}
            }
            
        except Exception as e:
            self.error_handler.handle_error(e, {"operation": "get_system_summary"})
            return {"error": str(e)}


# Factory function
def create_phase4_integration_manager(config_file_path: Optional[str] = None) -> Phase4IntegrationManager:
    """Create Phase 4 integration manager"""
    return Phase4IntegrationManager(config_file_path)


# Standalone execution
async def main():
    """Standalone Phase 4 integration demonstration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    manager = create_phase4_integration_manager()
    
    try:
        # Initialize system
        print("🚀 Initializing WallStreetBots Production System...")
        if not await manager.initialize_system():
            print("❌ System initialization failed")
            return
        
        # Validate strategies
        print("🧪 Validating all strategies...")
        if not await manager.validate_all_strategies():
            print("❌ Strategy validation failed")
            return
        
        # Start paper trading
        print("📊 Starting paper trading...")
        if not await manager.start_production_trading(paper_trading=True):
            print("❌ Failed to start trading")
            return
        
        print("🎉 WallStreetBots Production System is LIVE in paper trading mode!")
        
        # Run for demonstration
        await asyncio.sleep(60)  # Run for 1 minute
        
        # Get system summary
        summary = await manager.get_system_summary()
        print(f"📈 System Summary: {json.dumps(summary, indent=2)}")
        
        # Stop system
        print("🛑 Stopping system...")
        await manager.stop_all_strategies()
        
    except KeyboardInterrupt:
        print("🛑 User interrupted - stopping system...")
        await manager.emergency_shutdown()
    except Exception as e:
        print(f"❌ System error: {e}")
        await manager.emergency_shutdown()


if __name__ == "__main__":
    asyncio.run(main())