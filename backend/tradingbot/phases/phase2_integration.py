"""
Phase 2 Integration Script
Integrate all low - risk strategies with Phase 1 infrastructure
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any

from ..core.trading_interface import create_trading_interface
from ..core.data_providers import create_data_provider
from ..core.production_config import create_config_manager, ProductionConfig
from ..core.production_logging import create_production_logger, ErrorHandler, MetricsCollector
from ..core.production_wheel_strategy import create_wheel_strategy
from ..core.production_debit_spreads import create_debit_spreads_strategy
from ..core.production_spx_spreads import create_spx_spreads_strategy
from ..core.production_index_baseline import create_index_baseline_strategy


class Phase2StrategyManager: 
    """Phase 2 Strategy Manager - Orchestrates all low - risk strategies"""
    
    def __init__(self, config: ProductionConfig):
        self.config=config
        self.logger=create_production_logger("phase2_manager")
        self.error_handler=ErrorHandler(self.logger)
        self.metrics=MetricsCollector(self.logger)
        
        # Initialize core components
        self.trading_interface=None
        self.data_provider=None
        
        # Initialize strategies
        self.wheel_strategy=None
        self.debit_spreads=None
        self.spx_spreads=None
        self.index_baseline=None
        
        # Strategy status
        self.strategy_status: Dict[str, bool]={
            'wheel': False,
            'debit_spreads': False,
            'spx_spreads': False,
            'index_baseline': False
        }
        
        self.logger.info("Phase 2 Strategy Manager initialized")
    
    async def initialize(self): 
        """Initialize all Phase 2 components"""
        self.logger.info("Initializing Phase 2 components")
        
        try: 
            # Initialize core components
            self.data_provider=create_data_provider(self.config.data_providers.__dict__)
            
            # Create flat config for trading interface
            trading_config={
                'alpaca_api_key': self.config.broker.alpaca_api_key,
                'alpaca_secret_key': self.config.broker.alpaca_secret_key,
                'alpaca_base_url': self.config.broker.alpaca_base_url,
                'account_size': self.config.risk.account_size,
                'max_position_risk': self.config.risk.max_position_risk,
                'default_commission': self.config.risk.default_commission
            }
            self.trading_interface=create_trading_interface(trading_config)
            
            # Initialize strategies
            self.wheel_strategy=create_wheel_strategy(
                self.trading_interface, self.data_provider, self.config, self.logger
            )
            
            self.debit_spreads=create_debit_spreads_strategy(
                self.trading_interface, self.data_provider, self.config, self.logger
            )
            
            self.spx_spreads=create_spx_spreads_strategy(
                self.trading_interface, self.data_provider, self.config, self.logger
            )
            
            self.index_baseline=create_index_baseline_strategy(
                self.trading_interface, self.data_provider, self.config, self.logger
            )
            
            self.logger.info("Phase 2 components initialized successfully")
            
        except Exception as e: 
            self.error_handler.handle_error(e, {"operation": "initialize"})
            raise
    
    async def start_strategy(self, strategy_name: str) -> bool:
        """Start a specific strategy"""
        try: 
            self.logger.info(f"Starting strategy: {strategy_name}")
            
            if strategy_name== 'wheel' and self.wheel_strategy: 
                # Start wheel strategy in background
                asyncio.create_task(self.wheel_strategy.run_strategy())
                self.strategy_status['wheel']=True
                
            elif strategy_name== 'debit_spreads' and self.debit_spreads: 
                # Start debit spreads in background
                asyncio.create_task(self.debit_spreads.run_strategy())
                self.strategy_status['debit_spreads']=True
                
            elif strategy_name== 'spx_spreads' and self.spx_spreads: 
                # Start SPX spreads in background
                asyncio.create_task(self.spx_spreads.run_strategy())
                self.strategy_status['spx_spreads']=True
                
            elif strategy_name== 'index_baseline' and self.index_baseline: 
                # Start index baseline in background
                asyncio.create_task(self.index_baseline.run_baseline_tracking())
                self.strategy_status['index_baseline']=True
                
            else: 
                self.logger.error(f"Unknown strategy: {strategy_name}")
                return False
            
            self.logger.info(f"Strategy {strategy_name} started successfully")
            self.metrics.record_metric("strategy_started", 1, {"strategy": strategy_name})
            
            return True
            
        except Exception as e: 
            self.error_handler.handle_error(e, {"strategy": strategy_name, "operation": "start_strategy"})
            return False
    
    async def stop_strategy(self, strategy_name: str) -> bool:
        """Stop a specific strategy"""
        try: 
            self.logger.info(f"Stopping strategy: {strategy_name}")
            
            # In production, would implement proper strategy stopping
            self.strategy_status[strategy_name]=False
            
            self.logger.info(f"Strategy {strategy_name} stopped")
            self.metrics.record_metric("strategy_stopped", 1, {"strategy": strategy_name})
            
            return True
            
        except Exception as e: 
            self.error_handler.handle_error(e, {"strategy": strategy_name, "operation": "stop_strategy"})
            return False
    
    async def get_strategy_status(self) -> Dict[str, Any]: 
        """Get status of all strategies"""
        status={
            "timestamp": datetime.now().isoformat(),
            "strategies": self.strategy_status.copy(),
            "active_count": sum(1 for active in self.strategy_status.values() if active)
        }
        
        return status
    
    async def get_portfolio_summary(self) -> Dict[str, Any]: 
        """Get comprehensive portfolio summary"""
        summary={
            "timestamp": datetime.now().isoformat(),
            "strategies": {}
        }
        
        try: 
            # Get wheel strategy summary
            if self.wheel_strategy: 
                wheel_summary=await self.wheel_strategy.get_portfolio_summary()
                summary["strategies"]["wheel"]=wheel_summary
            
            # Get debit spreads summary
            if self.debit_spreads: 
                debit_summary=await self.debit_spreads.get_portfolio_summary()
                summary["strategies"]["debit_spreads"]=debit_summary
            
            # Get SPX spreads summary
            if self.spx_spreads: 
                spx_summary=await self.spx_spreads.get_portfolio_summary()
                summary["strategies"]["spx_spreads"]=spx_summary
            
            # Get index baseline summary
            if self.index_baseline: 
                baseline_summary=await self.index_baseline.get_performance_report()
                summary["strategies"]["index_baseline"]=baseline_summary
            
        except Exception as e: 
            self.error_handler.handle_error(e, {"operation": "get_portfolio_summary"})
        
        return summary
    
    async def run_strategies(self, strategy_names: List[str]):
        """Run specified strategies - MISSING METHOD THAT PRODUCTION RUNNER NEEDS"""
        self.logger.info(f"Running Phase 2 strategies: {strategy_names}")
        
        try: 
            # Start each requested strategy
            for strategy_name in strategy_names: 
                if strategy_name in self.strategy_status: 
                    success=await self.start_strategy(strategy_name)
                    if success: 
                        self.logger.info(f"Successfully started {strategy_name}")
                    else: 
                        self.logger.error(f"Failed to start {strategy_name}")
                else: 
                    self.logger.warning(f"Unknown strategy: {strategy_name}")
            
            # Keep strategies running
            self.logger.info("Strategies running. Press Ctrl + C to stop.")
            while True: 
                # Check strategy health every 60 seconds
                await asyncio.sleep(60)
                status=await self.get_strategy_status()
                self.logger.debug(f"Strategy status: {status}")
                
        except KeyboardInterrupt: 
            self.logger.info("Stopping strategies due to user interrupt")
            
            # Stop all running strategies
            for strategy_name in self.strategy_status: 
                if self.strategy_status[strategy_name]: 
                    await self.stop_strategy(strategy_name)
                    
        except Exception as e: 
            self.error_handler.handle_error(e, {"operation": "run_strategies"})
            raise
    
    async def run_phase2_demo(self): 
        """Run Phase 2 demonstration"""
        self.logger.info("Starting Phase 2 demonstration")
        
        try: 
            # Initialize components
            await self.initialize()
            
            # Start index baseline (always running)
            await self.start_strategy('index_baseline')
            
            # Demonstrate wheel strategy
            self.logger.info("Demonstrating Wheel Strategy")
            if self.wheel_strategy: 
                candidates=await self.wheel_strategy.scan_for_opportunities()
                self.logger.info(f"Found {len(candidates)} wheel candidates")
                
                # Show top candidate
                if candidates: 
                    top_candidate=candidates[0]
                    self.logger.info(f"Top wheel candidate: {top_candidate.ticker}",
                                   score=top_candidate.wheel_score,
                                   iv_rank=top_candidate.iv_rank)
            
            # Demonstrate debit spreads
            self.logger.info("Demonstrating Debit Spreads")
            if self.debit_spreads: 
                candidates=await self.debit_spreads.scan_for_opportunities()
                self.logger.info(f"Found {len(candidates)} debit spread candidates")
                
                # Show top candidate
                if candidates: 
                    top_candidate=candidates[0]
                    self.logger.info(f"Top debit spread: {top_candidate.ticker}",
                                   score=top_candidate.spread_score,
                                   profit_loss_ratio=top_candidate.profit_loss_ratio)
            
            # Demonstrate SPX spreads
            self.logger.info("Demonstrating SPX Spreads")
            if self.spx_spreads: 
                candidates=await self.spx_spreads.scan_for_opportunities()
                self.logger.info(f"Found {len(candidates)} SPX spread candidates")
                
                # Show top candidate
                if candidates: 
                    top_candidate=candidates[0]
                    self.logger.info(f"Top SPX spread",
                                   score=top_candidate.spread_score,
                                   profit_loss_ratio=top_candidate.profit_loss_ratio)
            
            # Get portfolio summary
            summary=await self.get_portfolio_summary()
            self.logger.info("Phase 2 portfolio summary", **summary)
            
            self.logger.info("Phase 2 demonstration completed successfully")
            
        except Exception as e: 
            self.error_handler.handle_error(e, {"operation": "run_phase2_demo"})
            raise


async def main(): 
    """Main Phase 2 integration function"""
    # Load configuration
    config_manager=create_config_manager()
    config=config_manager.load_config()
    
    # Create strategy manager
    manager=Phase2StrategyManager(config)
    
    # Run demonstration
    await manager.run_phase2_demo()


if __name__== "__main__": # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run Phase 2 integration
    asyncio.run(main())
