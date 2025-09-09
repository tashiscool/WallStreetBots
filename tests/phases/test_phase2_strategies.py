"""
Phase 2 Strategy Tests
Test all low-risk strategy implementations
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import tempfile
import json
import os

# Test Phase 2 components
from backend.tradingbot.core.production_wheel_strategy import (
    ProductionWheelStrategy, WheelPosition, WheelCandidate, WheelStage, WheelStatus
)
from backend.tradingbot.core.production_debit_spreads import (
    ProductionDebitSpreads, SpreadPosition, SpreadCandidate, SpreadType, SpreadStatus,
    QuantLibPricer
)
from backend.tradingbot.core.production_spx_spreads import (
    ProductionSPXSpreads, SPXSpreadPosition, SPXSpreadCandidate, SPXSpreadType, SPXSpreadStatus,
    CMEDataProvider
)
from backend.tradingbot.core.production_index_baseline import (
    ProductionIndexBaseline, BenchmarkData, StrategyPerformance, PerformanceComparison,
    BenchmarkType, PerformanceCalculator
)


class TestWheelStrategy(unittest.TestCase):
    """Test Wheel Strategy implementation"""
    
    def setUp(self):
        """Setup test environment"""
        self.mock_trading=Mock()
        self.mock_data=Mock()
        self.mock_config=Mock()
        self.mock_logger=Mock()
        
        # Setup mock config
        self.mock_config.trading.max_concurrent_trades=5
        self.mock_config.risk.max_position_risk = 0.10
        self.mock_config.risk.account_size = 100000.0
        self.mock_config.trading.universe = ["AAPL", "MSFT", "GOOGL"]
        
        self.wheel_strategy=ProductionWheelStrategy(
            self.mock_trading, self.mock_data, self.mock_config, self.mock_logger
        )
    
    def test_wheel_position_creation(self):
        """Test wheel position creation"""
        position=WheelPosition(
            ticker="AAPL",
            stage=WheelStage.CASH_SECURED_PUT,
            status=WheelStatus.ACTIVE,
            quantity=100,
            entry_price=150.0,
            current_price=155.0,
            unrealized_pnl=500.0,
            option_type="put",
            strike_price=145.0,
            expiry_date=datetime.now() + timedelta(days=30),
            premium_received=200.0
        )
        
        self.assertEqual(position.ticker, "AAPL")
        self.assertEqual(position.stage, WheelStage.CASH_SECURED_PUT)
        self.assertEqual(position.status, WheelStatus.ACTIVE)
        self.assertEqual(position.quantity, 100)
        self.assertEqual(position.unrealized_pnl, 500.0)
    
    def test_wheel_candidate_scoring(self):
        """Test wheel candidate scoring"""
        candidate=WheelCandidate(
            ticker="AAPL",
            current_price=150.0,
            volatility_rank=0.7,
            iv_rank=0.6,
            put_premium=3.0,
            earnings_risk=0.1,
            rsi=45.0
        )
        
        score=candidate.calculate_wheel_score()
        
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertEqual(candidate.wheel_score, score)
    
    def test_wheel_position_pnl_calculation(self):
        """Test wheel position P&L calculation"""
        # Cash secured put - profitable
        position=WheelPosition(
            ticker="AAPL",
            stage=WheelStage.CASH_SECURED_PUT,
            status=WheelStatus.ACTIVE,
            quantity=100,
            entry_price=150.0,
            current_price=155.0,  # Stock above strike
            unrealized_pnl=0.0,
            option_type="put",
            strike_price=145.0,
            expiry_date=datetime.now() + timedelta(days=30),
            premium_received=200.0
        )
        
        pnl=position.calculate_unrealized_pnl()
        self.assertEqual(pnl, 200.0)  # Full premium if stock stays above strike
        
        # Cash secured put - loss scenario
        position.current_price=140.0  # Stock below strike
        pnl = position.calculate_unrealized_pnl()
        expected_loss=(145.0 - 140.0) * 100  # $500 loss
        expected_pnl=200.0 - expected_loss  # Premium - loss
        self.assertEqual(pnl, expected_pnl)


class TestDebitSpreads(unittest.TestCase):
    """Test Debit Spreads implementation"""
    
    def setUp(self):
        """Setup test environment"""
        self.mock_trading=Mock()
        self.mock_data=Mock()
        self.mock_config=Mock()
        self.mock_logger=Mock()
        
        # Setup mock config
        self.mock_config.trading.max_concurrent_trades=5
        self.mock_config.risk.max_position_risk = 0.10
        self.mock_config.risk.account_size = 100000.0
        self.mock_config.trading.universe = ["AAPL", "MSFT", "GOOGL"]
        
        self.debit_spreads=ProductionDebitSpreads(
            self.mock_trading, self.mock_data, self.mock_config, self.mock_logger
        )
    
    def test_spread_position_creation(self):
        """Test spread position creation"""
        position=SpreadPosition(
            ticker="AAPL",
            spread_type=SpreadType.BULL_CALL_SPREAD,
            status=SpreadStatus.ACTIVE,
            long_strike=145.0,
            short_strike=150.0,
            quantity=10,
            net_debit=2.0,
            max_profit=3.0,
            max_loss=2.0,
            long_option={"strike":145.0, "premium":3.0},
            short_option={"strike":150.0, "premium":1.0}
        )
        
        self.assertEqual(position.ticker, "AAPL")
        self.assertEqual(position.spread_type, SpreadType.BULL_CALL_SPREAD)
        self.assertEqual(position.long_strike, 145.0)
        self.assertEqual(position.short_strike, 150.0)
        self.assertEqual(position.net_debit, 2.0)
    
    def test_spread_candidate_scoring(self):
        """Test spread candidate scoring"""
        candidate=SpreadCandidate(
            ticker="AAPL",
            current_price=150.0,
            spread_type=SpreadType.BULL_CALL_SPREAD,
            long_strike=145.0,
            short_strike=150.0,
            long_premium=3.0,
            short_premium=1.0,
            net_debit=2.0,
            max_profit=3.0,
            max_loss=2.0,
            profit_loss_ratio=1.5,
            net_delta=0.3,
            net_theta=-0.1,
            net_vega=0.05
        )
        
        score=candidate.calculate_spread_score()
        
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertEqual(candidate.spread_score, score)
    
    def test_quantlib_pricer(self):
        """Test QuantLib pricer"""
        pricer=QuantLibPricer()
        
        # Test Black-Scholes calculation
        result=pricer.calculate_black_scholes(
            spot_price=100.0,
            strike_price=100.0,
            risk_free_rate=0.02,
            volatility=0.20,
            time_to_expiry=0.25,  # 3 months
            option_type="call"
        )
        
        self.assertIn('price', result)
        self.assertIn('delta', result)
        self.assertIn('gamma', result)
        self.assertIn('theta', result)
        self.assertIn('vega', result)
        
        # Price should be positive
        self.assertGreater(result['price'], 0.0)
        
        # Delta should be between 0 and 1 for call
        self.assertGreaterEqual(result['delta'], 0.0)
        self.assertLessEqual(result['delta'], 1.0)


class TestSPXSpreads(unittest.TestCase):
    """Test SPX Spreads implementation"""
    
    def setUp(self):
        """Setup test environment"""
        self.mock_trading=Mock()
        self.mock_data=Mock()
        self.mock_config=Mock()
        self.mock_logger=Mock()
        
        # Setup mock config
        self.mock_config.trading.max_concurrent_trades=5
        self.mock_config.risk.max_position_risk = 0.10
        self.mock_config.risk.account_size = 100000.0
        
        self.spx_spreads = ProductionSPXSpreads(
            self.mock_trading, self.mock_data, self.mock_config, self.mock_logger
        )
    
    def test_spx_spread_position_creation(self):
        """Test SPX spread position creation"""
        position=SPXSpreadPosition(
            spread_type=SPXSpreadType.PUT_CREDIT_SPREAD,
            status=SPXSpreadStatus.ACTIVE,
            long_strike=4400.0,
            short_strike=4450.0,
            quantity=1,
            net_credit=2.0,
            max_profit=2.0,
            max_loss=48.0,
            long_option={"strike":4400.0, "premium":1.0},
            short_option={"strike":4450.0, "premium":3.0}
        )
        
        self.assertEqual(position.spread_type, SPXSpreadType.PUT_CREDIT_SPREAD)
        self.assertEqual(position.long_strike, 4400.0)
        self.assertEqual(position.short_strike, 4450.0)
        self.assertEqual(position.net_credit, 2.0)
    
    def test_spx_spread_candidate_scoring(self):
        """Test SPX spread candidate scoring"""
        candidate=SPXSpreadCandidate(
            spread_type=SPXSpreadType.PUT_CREDIT_SPREAD,
            long_strike=4400.0,
            short_strike=4450.0,
            long_premium=1.0,
            short_premium=3.0,
            net_credit=2.0,
            max_profit=2.0,
            max_loss=48.0,
            profit_loss_ratio=0.04,
            net_delta=-0.1,
            net_theta=0.05,
            net_vega=-0.02,
            spx_price=4500.0,
            vix_level=20.0,
            market_regime="bull"
        )
        
        score=candidate.calculate_spread_score()
        
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertEqual(candidate.spread_score, score)
    
    def test_cme_data_provider(self):
        """Test CME data provider"""
        logger=Mock()
        cme_provider=CMEDataProvider(logger)
        
        # Test VIX level
        vix_level=asyncio.run(cme_provider.get_vix_level())
        self.assertGreater(vix_level, 0.0)
        
        # Test market regime
        regime=asyncio.run(cme_provider.get_market_regime())
        self.assertIn(regime, ["bull", "bear", "neutral"])


class TestIndexBaseline(unittest.TestCase):
    """Test Index Baseline implementation"""
    
    def setUp(self):
        """Setup test environment"""
        self.mock_trading=Mock()
        self.mock_data=Mock()
        self.mock_config=Mock()
        self.mock_logger=Mock()
        
        self.index_baseline=ProductionIndexBaseline(
            self.mock_trading, self.mock_data, self.mock_config, self.mock_logger
        )
    
    def test_benchmark_data_creation(self):
        """Test benchmark data creation"""
        benchmark=BenchmarkData(
            ticker="SPY",
            benchmark_type=BenchmarkType.SPY,
            current_price=450.0,
            daily_return=0.01,
            weekly_return=0.02,
            monthly_return=0.05,
            ytd_return=0.15,
            annual_return=0.20,
            volatility=0.15,
            sharpe_ratio=1.2,
            max_drawdown=0.08
        )
        
        self.assertEqual(benchmark.ticker, "SPY")
        self.assertEqual(benchmark.benchmark_type, BenchmarkType.SPY)
        self.assertEqual(benchmark.current_price, 450.0)
        self.assertEqual(benchmark.sharpe_ratio, 1.2)
    
    def test_strategy_performance_creation(self):
        """Test strategy performance creation"""
        performance=StrategyPerformance(
            strategy_name="Wheel Strategy",
            total_return=0.12,
            daily_return=0.0005,
            weekly_return=0.002,
            monthly_return=0.01,
            ytd_return=0.12,
            annual_return=0.12,
            volatility=0.18,
            sharpe_ratio=0.8,
            max_drawdown=0.12,
            win_rate=0.65,
            total_trades=100,
            winning_trades=65,
            losing_trades=35,
            avg_win=150.0,
            avg_loss=75.0,
            profit_factor=1.3
        )
        
        self.assertEqual(performance.strategy_name, "Wheel Strategy")
        self.assertEqual(performance.total_return, 0.12)
        self.assertEqual(performance.win_rate, 0.65)
        self.assertEqual(performance.profit_factor, 1.3)
    
    def test_performance_calculator(self):
        """Test performance calculator"""
        calculator=PerformanceCalculator(Mock())
        
        # Test returns calculation
        prices=[100.0, 101.0, 102.0, 101.5, 103.0]
        returns=calculator.calculate_returns(prices)
        
        self.assertIn('daily_return', returns)
        self.assertIn('weekly_return', returns)
        self.assertIn('monthly_return', returns)
        self.assertIn('ytd_return', returns)
        self.assertIn('annual_return', returns)
        
        # Test volatility calculation
        returns_list=[0.01, 0.02, -0.01, 0.015, 0.005]
        volatility=calculator.calculate_volatility(returns_list)
        self.assertGreater(volatility, 0.0)
        
        # Test Sharpe ratio calculation
        sharpe=calculator.calculate_sharpe_ratio(returns_list)
        self.assertIsInstance(sharpe, float)
        
        # Test max drawdown calculation
        prices=[100.0, 105.0, 110.0, 108.0, 115.0, 112.0, 120.0]
        max_dd=calculator.calculate_max_drawdown(prices)
        self.assertGreaterEqual(max_dd, 0.0)
    
    def test_performance_comparison_creation(self):
        """Test performance comparison creation"""
        comparison=PerformanceComparison(
            strategy_name="Wheel Strategy",
            benchmark_ticker="SPY",
            strategy_return=0.12,
            benchmark_return=0.10,
            alpha=0.02,
            beta=0.8,
            strategy_volatility=0.18,
            benchmark_volatility=0.15,
            information_ratio=0.11,
            strategy_sharpe=0.8,
            benchmark_sharpe=0.9
        )
        
        self.assertEqual(comparison.strategy_name, "Wheel Strategy")
        self.assertEqual(comparison.benchmark_ticker, "SPY")
        self.assertEqual(comparison.alpha, 0.02)
        self.assertEqual(comparison.beta, 0.8)


class TestPhase2Integration(unittest.TestCase):
    """Test Phase 2 integration"""
    
    def setUp(self):
        """Setup test environment"""
        self.temp_dir=tempfile.mkdtemp()
        self.config_file=os.path.join(self.temp_dir, "test_config.json")
        
        # Create test configuration
        test_config={
            "data_providers":{
                "iex_api_key":"test_key",
                "polygon_api_key":"test_key"
            },
            "broker":{
                "alpaca_api_key":"test_key",
                "alpaca_secret_key":"test_secret"
            },
            "risk":{
                "max_position_risk":0.10,
                "account_size":100000.0
            },
            "trading":{
                "universe":["AAPL", "MSFT", "GOOGL"],
                "max_concurrent_trades":5
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(test_config, f)
    
    def tearDown(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_phase2_components_creation(self):
        """Test Phase 2 components can be created"""
        from backend.tradingbot.core.production_config import ConfigManager
        from backend.tradingbot.core.production_logging import ProductionLogger
        
        # Load configuration
        config_manager=ConfigManager(self.config_file)
        config=config_manager.load_config()
        
        # Create logger
        logger=ProductionLogger("test_phase2")
        
        # Test that all components can be created
        self.assertIsNotNone(config)
        self.assertIsNotNone(logger)
        
        # Test configuration values
        self.assertEqual(config.risk.max_position_risk, 0.10)
        self.assertEqual(config.risk.account_size, 100000.0)
        self.assertEqual(len(config.trading.universe), 3)


if __name__== "__main__":# Run tests
    unittest.main()
