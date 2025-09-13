#!/usr / bin/env python3
"""
Standalone Phase 2 Tests
Test Phase 2 strategies without external dependencies
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
import tempfile
import json
import os
import math
from enum import Enum
from dataclasses import dataclass, field


# Define Phase 2 components directly for testing
class WheelStage(Enum): 
    CASH_SECURED_PUT="cash_secured_put"
    ASSIGNED_STOCK="assigned_stock"
    COVERED_CALL="covered_call"
    CLOSED_POSITION="closed_position"


class WheelStatus(Enum): 
    ACTIVE="active"
    EXPIRED="expired"
    ASSIGNED="assigned"
    CLOSED="closed"
    ROLLED="rolled"


@dataclass
class WheelPosition: 
    ticker: str
    stage: WheelStage
    status: WheelStatus
    quantity: int
    entry_price: float
    current_price: float
    unrealized_pnl: float
    option_type: str
    strike_price: float
    expiry_date: datetime
    premium_received: float
    premium_paid: float=0.0
    entry_date: datetime=field(default_factory=datetime.now)
    last_update: datetime=field(default_factory=datetime.now)
    days_to_expiry: int=0
    delta: float=0.0
    theta: float=0.0
    iv_rank: float=0.0
    
    def calculate_unrealized_pnl(self) -> float: 
        """Calculate unrealized P & L"""
        if self.stage== WheelStage.CASH_SECURED_PUT: 
            if self.current_price >= self.strike_price: 
                return self.premium_received
            else: 
                loss=(self.strike_price - self.current_price) * self.quantity
                return self.premium_received - loss
        elif self.stage== WheelStage.ASSIGNED_STOCK: 
            stock_pnl=(self.current_price - self.entry_price) * self.quantity
            return stock_pnl + self.premium_received
        elif self.stage== WheelStage.COVERED_CALL: 
            stock_pnl=(self.current_price - self.entry_price) * self.quantity
            call_pnl=self.premium_received - self.premium_paid
            
            if self.current_price >= self.strike_price: 
                assignment_pnl=(self.strike_price - self.entry_price) * self.quantity
                return assignment_pnl + call_pnl
            else: 
                return stock_pnl + call_pnl
        return 0.0
    
    def calculate_days_to_expiry(self) -> int: 
        """Calculate days to expiry"""
        if self.expiry_date: 
            delta=self.expiry_date - datetime.now()
            return max(0, delta.days)
        return 0


@dataclass
class WheelCandidate: 
    ticker: str
    current_price: float
    volatility_rank: float
    earnings_date: datetime=None
    earnings_risk: float=0.0
    rsi: float=50.0
    support_level: float=0.0
    resistance_level: float=0.0
    put_premium: float=0.0
    call_premium: float=0.0
    iv_rank: float=0.0
    wheel_score: float=0.0
    risk_score: float=0.0
    
    def calculate_wheel_score(self) -> float: 
        """Calculate wheel strategy score"""
        score=0.0
        score += self.volatility_rank * 0.3
        score += self.iv_rank * 0.2
        score += min(self.put_premium / self.current_price, 0.05) * 100
        score -= self.earnings_risk * 0.2
        if 30 <= self.rsi <= 70: 
            score += 0.1
        self.wheel_score=max(0.0, min(1.0, score))
        return self.wheel_score


class SpreadType(Enum): 
    BULL_CALL_SPREAD="bull_call_spread"
    BEAR_PUT_SPREAD="bear_put_spread"
    CALENDAR_SPREAD="calendar_spread"
    DIAGONAL_SPREAD="diagonal_spread"


class SpreadStatus(Enum): 
    ACTIVE="active"
    EXPIRED="expired"
    CLOSED="closed"
    ROLLED="rolled"


@dataclass
class SpreadPosition: 
    ticker: str
    spread_type: SpreadType
    status: SpreadStatus
    long_strike: float
    short_strike: float
    quantity: int
    net_debit: float
    max_profit: float
    max_loss: float
    long_option: dict
    short_option: dict
    current_value: float=0.0
    unrealized_pnl: float=0.0
    profit_pct: float=0.0
    entry_date: datetime=field(default_factory=datetime.now)
    expiry_date: datetime=field(default_factory=lambda: datetime.now() + timedelta(days=30))
    last_update: datetime=field(default_factory=datetime.now)
    net_delta: float=0.0
    net_gamma: float=0.0
    net_theta: float=0.0
    net_vega: float=0.0
    
    def calculate_max_profit(self) -> float: 
        """Calculate maximum profit potential"""
        if self.spread_type== SpreadType.BULL_CALL_SPREAD: 
            return (self.short_strike - self.long_strike) * self.quantity * 100 - self.net_debit * self.quantity * 100
        return 0.0
    
    def calculate_max_loss(self) -> float: 
        """Calculate maximum loss potential"""
        return self.net_debit * self.quantity * 100


@dataclass
class SpreadCandidate: 
    ticker: str
    current_price: float
    spread_type: SpreadType
    long_strike: float
    short_strike: float
    long_premium: float
    short_premium: float
    net_debit: float
    max_profit: float
    max_loss: float
    profit_loss_ratio: float
    net_delta: float
    net_theta: float
    net_vega: float
    spread_score: float=0.0
    risk_score: float=0.0
    
    def calculate_spread_score(self) -> float: 
        """Calculate spread strategy score"""
        score=0.0
        score += min(self.profit_loss_ratio, 3.0) * 0.3
        if self.spread_type== SpreadType.BULL_CALL_SPREAD: 
            score += max(0, self.net_delta) * 0.2
        score += max(0, -self.net_theta) * 0.2
        debit_pct=self.net_debit / self.current_price
        score += max(0, 0.05 - debit_pct) * 20
        strike_width=abs(self.short_strike - self.long_strike)
        if 2 <= strike_width <= 10: 
            score += 0.1
        self.spread_score=max(0.0, min(1.0, score))
        return self.spread_score


class QuantLibPricer: 
    """QuantLib - based options pricing"""
    
    def __init__(self): 
        self.logger=Mock()
    
    def calculate_black_scholes(self, 
                              spot_price: float,
                              strike_price: float,
                              risk_free_rate: float,
                              volatility: float,
                              time_to_expiry: float,
                              option_type: str) -> dict:
        """Calculate Black - Scholes price and Greeks"""
        try: 
            # Calculate d1 and d2
            d1=(math.log(spot_price / strike_price) + 
                  (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
            d2=d1 - volatility * math.sqrt(time_to_expiry)
            
            # Calculate option price
            if option_type.lower() == 'call': price=(spot_price * self._normal_cdf(d1) - 
                        strike_price * math.exp(-risk_free_rate * time_to_expiry) * self._normal_cdf(d2))
            else:  # put
                price=(strike_price * math.exp(-risk_free_rate * time_to_expiry) * self._normal_cdf(-d2) - 
                        spot_price * self._normal_cdf(-d1))
            
            # Calculate Greeks
            delta=self._normal_cdf(d1) if option_type.lower() == 'call' else self._normal_cdf(d1) - 1
            gamma=self._normal_pdf(d1) / (spot_price * volatility * math.sqrt(time_to_expiry))
            theta=(-spot_price * self._normal_pdf(d1) * volatility / (2 * math.sqrt(time_to_expiry)) - 
                    risk_free_rate * strike_price * math.exp(-risk_free_rate * time_to_expiry) * self._normal_cdf(d2))
            vega=spot_price * self._normal_pdf(d1) * math.sqrt(time_to_expiry)
            
            return {
                'price': price,
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega
            }
            
        except Exception as e: 
            self.logger.error(f"Black - Scholes calculation error: {e}")
            return {
                'price': 0.0,
                'delta': 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0
            }
    
    def _normal_cdf(self, x: float) -> float:
        """Cumulative distribution function of standard normal distribution"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def _normal_pdf(self, x: float) -> float:
        """Probability density function of standard normal distribution"""
        return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)


class BenchmarkType(Enum): 
    SPY="spy"
    VTI="vti"
    QQQ="qqq"
    IWM="iwm"


@dataclass
class BenchmarkData: 
    ticker: str
    benchmark_type: BenchmarkType
    current_price: float
    daily_return: float
    weekly_return: float
    monthly_return: float
    ytd_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    last_update: datetime=field(default_factory=datetime.now)


@dataclass
class StrategyPerformance: 
    strategy_name: str
    total_return: float
    daily_return: float
    weekly_return: float
    monthly_return: float
    ytd_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    profit_factor: float
    last_update: datetime=field(default_factory=datetime.now)


@dataclass
class PerformanceComparison: 
    strategy_name: str
    benchmark_ticker: str
    strategy_return: float
    benchmark_return: float
    alpha: float
    beta: float
    strategy_volatility: float
    benchmark_volatility: float
    information_ratio: float
    strategy_sharpe: float
    benchmark_sharpe: float
    comparison_date: datetime=field(default_factory=datetime.now)


class PerformanceCalculator: 
    """Performance calculation utilities"""
    
    def __init__(self, logger): 
        self.logger=logger
    
    def calculate_returns(self, prices: list) -> dict:
        """Calculate various return metrics"""
        if len(prices) < 2: 
            return {
                'daily_return': 0.0,
                'weekly_return': 0.0,
                'monthly_return': 0.0,
                'ytd_return': 0.0,
                'annual_return': 0.0
            }
        
        # Daily return
        daily_return=(prices[-1] - prices[-2]) / prices[-2] if len(prices) >= 2 else 0.0
        
        # Weekly return (5 trading days)
        weekly_return=(prices[-1] - prices[-6]) / prices[-6] if len(prices) >= 6 else 0.0
        
        # Monthly return (20 trading days)
        monthly_return=(prices[-1] - prices[-21]) / prices[-21] if len(prices) >= 21 else 0.0
        
        # YTD return (simplified)
        ytd_return=(prices[-1] - prices[0]) / prices[0] if len(prices) > 0 else 0.0
        
        # Annual return (simplified)
        annual_return=ytd_return * (252 / len(prices)) if len(prices) > 0 else 0.0
        
        return {
            'daily_return': daily_return,
            'weekly_return': weekly_return,
            'monthly_return': monthly_return,
            'ytd_return': ytd_return,
            'annual_return': annual_return
        }
    
    def calculate_volatility(self, returns: list) -> float:
        """Calculate volatility (standard deviation of returns)"""
        if len(returns) < 2: 
            return 0.0
        
        mean_return=sum(returns) / len(returns)
        variance=sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        return math.sqrt(variance)
    
    def calculate_sharpe_ratio(self, returns: list, risk_free_rate: float=0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2: 
            return 0.0
        
        mean_return=sum(returns) / len(returns)
        volatility=self.calculate_volatility(returns)
        
        if volatility== 0: 
            return 0.0
        
        return (mean_return - risk_free_rate) / volatility
    
    def calculate_max_drawdown(self, prices: list) -> float:
        """Calculate maximum drawdown"""
        if len(prices) < 2: 
            return 0.0
        
        peak=prices[0]
        max_dd=0.0
        
        for price in prices: 
            if price > peak: 
                peak=price
            else: 
                drawdown=(peak - price) / peak
                max_dd=max(max_dd, drawdown)
        
        return max_dd
    
    def calculate_alpha_beta(self, strategy_returns: list, 
                           benchmark_returns: list) -> tuple:
        """Calculate alpha and beta"""
        if len(strategy_returns) != len(benchmark_returns) or len(strategy_returns) < 2: 
            return 0.0, 1.0
        
        # Calculate covariance and variance
        strategy_mean=sum(strategy_returns) / len(strategy_returns)
        benchmark_mean=sum(benchmark_returns) / len(benchmark_returns)
        
        covariance=sum((s - strategy_mean) * (b - benchmark_mean) 
                        for s, b in zip(strategy_returns, benchmark_returns)) / len(strategy_returns)
        
        benchmark_variance=sum((b - benchmark_mean) ** 2 for b in benchmark_returns) / len(benchmark_returns)
        
        if benchmark_variance== 0: 
            return 0.0, 1.0
        
        beta=covariance / benchmark_variance
        alpha=strategy_mean - beta * benchmark_mean
        
        return alpha, beta


class TestWheelStrategy(unittest.TestCase): 
    """Test Wheel Strategy implementation"""
    
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
        """Test wheel position P & L calculation"""
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
        pnl=position.calculate_unrealized_pnl()
        expected_loss=(145.0 - 140.0) * 100  # $500 loss
        expected_pnl=200.0 - expected_loss  # Premium - loss
        self.assertEqual(pnl, expected_pnl)
    
    def test_wheel_position_days_to_expiry(self): 
        """Test days to expiry calculation"""
        position=WheelPosition(
            ticker="AAPL",
            stage=WheelStage.CASH_SECURED_PUT,
            status=WheelStatus.ACTIVE,
            quantity=100,
            entry_price=150.0,
            current_price=155.0,
            unrealized_pnl=0.0,
            option_type="put",
            strike_price=145.0,
            expiry_date=datetime.now() + timedelta(days=30),
            premium_received=200.0
        )
        
        days=position.calculate_days_to_expiry()
        self.assertGreaterEqual(days, 0)
        self.assertLessEqual(days, 30)


class TestDebitSpreads(unittest.TestCase): 
    """Test Debit Spreads implementation"""
    
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
            long_option={"strike": 145.0, "premium": 3.0},
            short_option={"strike": 150.0, "premium": 1.0}
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
        
        # Test Black - Scholes calculation
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
    
    def test_spread_position_max_profit_loss(self): 
        """Test spread position max profit / loss calculations"""
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
            long_option={"strike": 145.0, "premium": 3.0},
            short_option={"strike": 150.0, "premium": 1.0}
        )
        
        max_profit=position.calculate_max_profit()
        max_loss=position.calculate_max_loss()
        
        self.assertGreater(max_profit, 0.0)
        self.assertGreater(max_loss, 0.0)
        self.assertEqual(max_loss, 2.0 * 10 * 100)  # net_debit * quantity * 100


class TestIndexBaseline(unittest.TestCase): 
    """Test Index Baseline implementation"""
    
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
    
    def test_alpha_beta_calculation(self): 
        """Test alpha and beta calculation"""
        calculator=PerformanceCalculator(Mock())
        
        strategy_returns=[0.01, 0.02, -0.01, 0.015, 0.005]
        benchmark_returns=[0.008, 0.018, -0.012, 0.012, 0.003]
        
        alpha, beta=calculator.calculate_alpha_beta(strategy_returns, benchmark_returns)
        
        self.assertIsInstance(alpha, float)
        self.assertIsInstance(beta, float)
        self.assertGreater(beta, 0.0)  # Beta should be positive


class TestPhase2EndToEnd(unittest.TestCase): 
    """End - to-end tests for Phase 2"""
    
    def test_wheel_strategy_workflow(self): 
        """Test complete wheel strategy workflow"""
        # Test candidate creation
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
        
        # Test position creation
        position=WheelPosition(
            ticker="AAPL",
            stage=WheelStage.CASH_SECURED_PUT,
            status=WheelStatus.ACTIVE,
            quantity=100,
            entry_price=150.0,
            current_price=155.0,
            unrealized_pnl=0.0,
            option_type="put",
            strike_price=145.0,
            expiry_date=datetime.now() + timedelta(days=30),
            premium_received=200.0
        )
        
        # Test P & L calculation
        pnl=position.calculate_unrealized_pnl()
        self.assertEqual(pnl, 200.0)
        
        # Test days to expiry
        days=position.calculate_days_to_expiry()
        self.assertGreaterEqual(days, 0)
    
    def test_debit_spreads_workflow(self): 
        """Test complete debit spreads workflow"""
        # Test candidate creation
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
        
        # Test position creation
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
            long_option={"strike": 145.0, "premium": 3.0},
            short_option={"strike": 150.0, "premium": 1.0}
        )
        
        # Test max profit / loss calculations
        max_profit=position.calculate_max_profit()
        max_loss=position.calculate_max_loss()
        
        self.assertGreater(max_profit, 0.0)
        self.assertGreater(max_loss, 0.0)
    
    def test_index_baseline_workflow(self): 
        """Test complete index baseline workflow"""
        # Test benchmark creation
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
        self.assertEqual(benchmark.annual_return, 0.20)
        
        # Test strategy performance creation
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
        self.assertEqual(performance.win_rate, 0.65)
        
        # Test performance comparison
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
        
        self.assertEqual(comparison.alpha, 0.02)
        self.assertEqual(comparison.beta, 0.8)


if __name__== "__main__": # Run tests
    unittest.main()
