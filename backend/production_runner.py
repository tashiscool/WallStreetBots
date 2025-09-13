#!/usr / bin/env python3
"""
Production Trading System Runner
READY FOR REAL MONEY TRADING

This script connects all the production components and runs live trading strategies.

CRITICAL SETUP REQUIRED: 
1. Copy backend/.env.example to backend/.env 
2. Fill in ALL API keys (Alpaca, IEX, Polygon, etc.)
3. Set PAPER_TRADING_MODE = false for real money
4. Start with small position sizes

Usage: 
    # Paper trading (safe)
    python production_runner.py --paper

    # Real money trading (DANGEROUS - use small sizes)
    python production_runner.py --live --strategies wheel,debit_spreads --max - account-risk 0.02

    # Run specific phase
    python production_runner.py --phase 2 --live
"""

import asyncio
import argparse
import logging
import sys
import os
from typing import List
from datetime import datetime

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'tradingbot'))

from backend.tradingbot.production_config import create_config_manager, ProductionConfig  # noqa: E402
from backend.tradingbot.trading_interface import create_trading_interface  # noqa: E402
from backend.tradingbot.data_providers import create_data_provider  # noqa: E402
from backend.tradingbot.production_logging import create_production_logger  # noqa: E402

# Phase imports
from backend.tradingbot.phase2_integration import Phase2StrategyManager  # noqa: E402
from backend.tradingbot.phase3_integration import Phase3StrategyManager  # noqa: E402

# Individual strategy imports
from backend.tradingbot.production_wheel_strategy import create_wheel_strategy  # noqa: E402
from backend.tradingbot.production_debit_spreads import create_debit_spreads_strategy  # noqa: E402
from backend.tradingbot.production_spx_spreads import create_spx_spreads_strategy  # noqa: E402
from backend.tradingbot.production_momentum_weeklies import create_momentum_weeklies_strategy  # noqa: E402
from backend.tradingbot.production_wsb_dip_bot import create_wsb_dip_bot_strategy  # noqa: E402
from backend.tradingbot.production_lotto_scanner import create_lotto_scanner_strategy  # noqa: E402


class ProductionTradingRunner: 
    """Production trading system runner with safety controls"""
    
    def __init__(self, args): 
        self.args = args
        self.logger = create_production_logger("production_runner")
        
        # Load configuration
        self.config_manager = create_config_manager()
        self.config = self.config_manager.load_config()
        
        # Validate configuration
        self._validate_config()
        
        # Create core components
        self.data_provider = create_data_provider(self._get_data_config())
        self.trading_interface = create_trading_interface(self.config)
        
        self.logger.info("Production trading system initialized")
    
    def _validate_config(self): 
        """Validate critical configuration for real money trading"""
        errors = []
        
        # Validate data providers
        data_errors = self.config.data_providers.validate()
        if data_errors: 
            errors.extend([f"Data: {e}" for e in data_errors])
        
        # Validate broker
        broker_errors = self.config.broker.validate()
        if broker_errors: 
            errors.extend([f"Broker: {e}" for e in broker_errors])
        
        # Validate risk settings for live trading
        if not self.args.paper: 
            if self.config.risk.max_position_risk  >  0.05:  # 5% max
                errors.append("Risk: max_position_risk  >  5% for live trading")
            
            if self.config.risk.account_size  <=  0: 
                errors.append("Risk: account_size must be set for live trading")
        
        if errors: 
            self.logger.error("Configuration validation failed: ")
            for error in errors: 
                self.logger.error(f"  - {error}")
            raise ValueError("Invalid configuration - cannot start trading")
        
        self.logger.info("Configuration validation passed")
    
    def _get_data_config(self): 
        """Get data provider configuration"""
        return {
            'iex_api_key': self.config.data_providers.iex_api_key,
            'polygon_api_key': self.config.data_providers.polygon_api_key,
            'fmp_api_key': self.config.data_providers.fmp_api_key,
            'news_api_key': self.config.data_providers.news_api_key,
            'alpha_vantage_api_key': self.config.data_providers.alpha_vantage_api_key,
        }
    
    async def run_phase(self, phase: int):
        """Run specific phase strategies"""
        if phase ==  2: 
            await self._run_phase_2()
        elif phase ==  3: 
            await self._run_phase_3()
        elif phase ==  4: 
            await self._run_phase_4()
        else: 
            raise ValueError(f"Invalid phase: {phase}")
    
    async def _run_phase_2(self): 
        """Run Phase 2 - Low - risk strategies"""
        self.logger.info("Starting Phase 2 - Low - risk strategies")
        
        manager = Phase2StrategyManager(self.config)
        await manager.initialize()
        
        strategies = []
        if 'wheel' in self.args.strategies: 
            strategies.append('wheel')
        if 'debit_spreads' in self.args.strategies: 
            strategies.append('debit_spreads')
        if 'spx_spreads' in self.args.strategies: 
            strategies.append('spx_spreads')
        
        await manager.run_strategies(strategies)
    
    async def _run_phase_3(self): 
        """Run Phase 3 - Medium - risk strategies"""
        self.logger.info("Starting Phase 3 - Medium - risk strategies")
        
        from backend.tradingbot.phase3_integration import Phase3StrategyManager
        
        manager = Phase3StrategyManager(self.config)
        await manager.initialize()
        
        # Run available Phase 3 strategies
        strategies = ['earnings_protection', 'swing_trading', 'momentum_weeklies']
        
        # Start each strategy
        for strategy_name in strategies: 
            try: 
                success = await manager.start_strategy(strategy_name)
                if success: 
                    self.logger.info(f"Successfully started {strategy_name}")
                else: 
                    self.logger.error(f"Failed to start {strategy_name}")
            except Exception as e: 
                self.logger.error(f"Error starting {strategy_name}: {e}")
        
        # Keep strategies running
        self.logger.info("Phase 3 strategies running. Press Ctrl + C to stop.")
        try: 
            while True: 
                await asyncio.sleep(60)  # Monitor every minute
                status = await manager.get_strategy_status()
                self.logger.debug(f"Phase 3 status: {status}")
        except KeyboardInterrupt: 
            self.logger.info("Stopping Phase 3 strategies...")
            for strategy_name in strategies: 
                await manager.stop_strategy(strategy_name)
    
    async def _run_phase_4(self): 
        """Run Phase 4 - Complete system with validation"""
        self.logger.info("Starting Phase 4 - Complete validated system")
        
        from backend.tradingbot.phase4_integration import create_phase4_integration_manager
        
        # Create Phase 4 integration manager
        phase4_manager = create_phase4_integration_manager()
        
        try: 
            # Initialize complete system
            self.logger.info("Initializing complete production system...")
            if not await phase4_manager.initialize_system(): 
                raise ValueError("Phase 4 system initialization failed")
            
            # Validate all strategies
            self.logger.info("Validating all strategies...")
            if not await phase4_manager.validate_all_strategies(): 
                raise ValueError("Strategy validation failed - system not safe for trading")
            
            # Start production trading
            paper_trading = self.args.paper
            self.logger.info(f"Starting production trading (paper = {paper_trading})...")
            if not await phase4_manager.start_production_trading(paper_trading = paper_trading): 
                raise ValueError("Failed to start production trading")
            
            # Run until interrupted
            self.logger.info("Phase 4 complete system running. Press Ctrl + C to stop.")
            try: 
                while True: 
                    await asyncio.sleep(60)
                    summary = await phase4_manager.get_system_summary()
                    self.logger.info(f"System status: {summary['system_status']}, "
                                   f"Active strategies: {summary['health_check']['active_strategies']}, "
                                   f"Daily P & L: {summary['health_check']['daily_pnl']:.2f}")
            except KeyboardInterrupt: 
                self.logger.info("Stopping Phase 4 system...")
                await phase4_manager.stop_all_strategies()
                
        except Exception as e: 
            self.logger.error(f"Phase 4 error: {e}")
            await phase4_manager.emergency_shutdown()
            raise
    
    async def run_individual_strategies(self, strategy_names: List[str]):
        """Run individual strategies"""
        for strategy_name in strategy_names: 
            self.logger.info(f"Starting strategy: {strategy_name}")
            
            if strategy_name ==  'wheel': 
                strategy = create_wheel_strategy(
                    self.trading_interface, 
                    self.data_provider, 
                    self.config, 
                    self.logger
                )
                await strategy.run()
            
            elif strategy_name ==  'debit_spreads': 
                strategy = create_debit_spreads_strategy(
                self.trading_interface,
                self.data_provider,
                self.config,
                self.logger
            )
                await strategy.run()
            
            elif strategy_name ==  'spx_spreads': 
                strategy = create_spx_spreads_strategy(
                    self.trading_interface,
                    self.data_provider,
                    self.config,
                    self.logger
                )
                await strategy.run()
            
            elif strategy_name ==  'wsb_dip_bot': 
                strategy = create_wsb_dip_bot_strategy(
                    self.trading_interface,
                    self.data_provider,
                    self.config,
                    self.logger
                )
                await strategy.run()
            
            else: 
                self.logger.error(f"Unknown strategy: {strategy_name}")
    
    def print_safety_warning(self): 
        """Print critical safety warning"""
        print("\n" + " = " * 80)
        print("🚨 CRITICAL SAFETY WARNING 🚨")
        print(" = " * 80)
        
        if self.args.paper: 
            print("✅ PAPER TRADING MODE - No real money at risk")
        else: 
            print("💰 LIVE TRADING MODE - REAL MONEY AT RISK")
            print("⚠️  Max position risk: {:.1%}".format(self.config.risk.max_position_risk))
            print("⚠️  Account size: ${:,.2f}".format(self.config.risk.account_size))
            print("⚠️  Max total risk: ${:,.2f}".format(
                self.config.risk.account_size * self.config.risk.max_total_risk
            ))
        
        print("\n📊 STRATEGIES TO RUN: ")
        for strategy in self.args.strategies: 
            print(f"   - {strategy}")
        
        print("\n⏰ Starting in 10 seconds... (Ctrl + C to cancel)")
        print(" = " * 80)


def parse_args(): 
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description = "Production Trading System - Ready for Real Money",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = """
Examples: 
    # Safe paper trading
    python production_runner.py --paper --strategies wheel,debit_spreads
    
    # Live trading with conservative risk
    python production_runner.py --live --strategies wheel --max - account-risk 0.01
    
    # Run Phase 2 strategies
    python production_runner.py --phase 2 --paper
    
    # EXTREME CAUTION: High - risk strategies
    python production_runner.py --live --strategies wsb_dip_bot --max - account-risk 0.02
        """
    )
    
    # Trading mode
    mode_group = parser.add_mutually_exclusive_group(required  =  True)
    mode_group.add_argument('--paper', action = 'store_true',
                           help = 'Run in paper trading mode (safe)')
    mode_group.add_argument('--live', action = 'store_true',
                           help = 'Run with real money (DANGEROUS)')
    
    # Strategy selection
    strategy_group = parser.add_mutually_exclusive_group(required  =  True)
    strategy_group.add_argument('--strategies', type = str,
                               help = 'Comma - separated list of strategies: wheel,debit_spreads,spx_spreads (wsb_dip_bot not yet in production)')
    strategy_group.add_argument('--phase', type = int, choices = [2, 3, 4],
                               help = 'Run all strategies in a phase (2 = low - risk, 3 = medium - risk, 4 = high - risk with validation)')
    
    # Risk controls
    parser.add_argument('--max - account-risk', type = float, default = 0.10,
                       help = 'Maximum account risk percentage (default: 0.10 = 10%%)')
    parser.add_argument('--max - position-risk', type = float, default = 0.02,
                       help = 'Maximum single position risk (default: 0.02 = 2%%)')
    
    # Operational
    parser.add_argument('--config', type = str,
                       help = 'Configuration file path')
    parser.add_argument('--log - level', type = str, default = 'INFO',
                       choices = ['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help = 'Logging level')
    parser.add_argument('--dry - run', action = 'store_true',
                       help = 'Show what would be done without executing')
    
    args = parser.parse_args()
    
    # Process strategy list
    if args.strategies: 
        args.strategies = [s.strip() for s in args.strategies.split(',')]
    
    return args


async def main(): 
    """Main entry point"""
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level = getattr(logging, args.log_level),
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try: 
        # Initialize runner
        runner = ProductionTradingRunner(args)
        
        # Show safety warning
        runner.print_safety_warning()
        
        # Wait for confirmation (unless dry run)
        if not args.dry_run: 
            await asyncio.sleep(10)
        
        print("\n🚀 Starting trading system...\n")
        
        # Run strategies
        if args.phase: 
            await runner.run_phase(args.phase)
        else: 
            await runner.run_individual_strategies(args.strategies)
        
    except KeyboardInterrupt: 
        print("\n⛔ Trading system stopped by user")
    except Exception as e: 
        print(f"\n❌ Trading system failed: {e}")
        raise


if __name__ ==  "__main__": asyncio.run(main())