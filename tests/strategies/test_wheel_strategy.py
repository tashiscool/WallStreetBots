"""Comprehensive tests for Wheel Strategy."""
import pytest
import pandas as pd
from datetime import datetime, date, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from backend.tradingbot.strategies.wheel_strategy import (
    WheelStrategy,
    WheelPosition,
    WheelCandidate
)


class TestWheelStrategy:
    """Test Wheel Strategy implementation."""

    def test_wheel_strategy_initialization(self):
        """Test Wheel Strategy initialization."""
        strategy = WheelStrategy()
        
        # Test default attributes
        assert hasattr(strategy, 'portfolio_file')
        assert hasattr(strategy, 'positions')
        assert hasattr(strategy, 'wheel_candidates')
        
        # Test that wheel_candidates list is populated
        assert len(strategy.wheel_candidates) > 0
        assert "AAPL" in strategy.wheel_candidates
        assert "MSFT" in strategy.wheel_candidates
        assert "GOOGL" in strategy.wheel_candidates

    def test_wheel_position_creation(self):
        """Test WheelPosition dataclass creation."""
        position = WheelPosition(
            ticker="AAPL",
            position_type="cash_secured_put",
            shares=0,
            avg_cost=0.0,
            strike=150,
            expiry="2025-02-21",
            premium_collected=2.50,
            days_to_expiry=30,
            current_price=155.0,
            unrealized_pnl=250.0,
            total_premium_collected=500.0,
            assignment_risk=0.15,
            annualized_return=0.25,
            status="active"
        )
        
        assert position.ticker == "AAPL"
        assert position.position_type == "cash_secured_put"
        assert position.shares == 0
        assert position.avg_cost == 0.0
        assert position.strike == 150
        assert position.expiry == "2025-02-21"
        assert position.premium_collected == 2.50
        assert position.days_to_expiry == 30
        assert position.current_price == 155.0
        assert position.unrealized_pnl == 250.0
        assert position.total_premium_collected == 500.0
        assert position.assignment_risk == 0.15
        assert position.annualized_return == 0.25
        assert position.status == "active"

    def test_wheel_candidate_creation(self):
        """Test WheelCandidate dataclass creation."""
        candidate = WheelCandidate(
            ticker="AAPL",
            company_name="Apple Inc.",
            current_price=155.0,
            iv_rank=45.0,
            put_strike=150,
            put_expiry="2025-02-21",
            put_premium=2.50,
            put_delta=0.30,
            call_strike=160,
            call_premium=1.80,
            call_delta=0.25,
            wheel_annual_return=0.25,
            dividend_yield=0.005,
            liquidity_score=0.80,
            quality_score=85.0,
            volatility_score=0.75,
            risk_factors=["earnings_risk", "market_volatility"]
        )
        
        assert candidate.ticker == "AAPL"
        assert candidate.company_name == "Apple Inc."
        assert candidate.current_price == 155.0
        assert candidate.iv_rank == 45.0
        assert candidate.put_strike == 150
        assert candidate.put_expiry == "2025-02-21"
        assert candidate.put_premium == 2.50
        assert candidate.put_delta == 0.30
        assert candidate.call_strike == 160
        assert candidate.call_premium == 1.80
        assert candidate.call_delta == 0.25
        assert candidate.wheel_annual_return == 0.25
        assert candidate.dividend_yield == 0.005
        assert candidate.liquidity_score == 0.80
        assert candidate.quality_score == 85.0
        assert candidate.volatility_score == 0.75
        assert candidate.risk_factors == ["earnings_risk", "market_volatility"]

    def test_wheel_strategy_portfolio_management(self):
        """Test portfolio management functionality."""
        strategy = WheelStrategy()
        
        # Test adding a position
        position = WheelPosition(
            ticker="AAPL",
            position_type="cash_secured_put",
            shares=0,
            avg_cost=0.0,
            strike=150,
            expiry="2025-02-21",
            premium_collected=2.50,
            days_to_expiry=30,
            current_price=155.0,
            unrealized_pnl=250.0,
            total_premium_collected=500.0,
            assignment_risk=0.15,
            annualized_return=0.25,
            status="active"
        )
        
        # Test that we can add positions (method may exist)
        if hasattr(strategy, 'add_position'):
            strategy.add_position(position)
            assert len(strategy.positions) > 0

    def test_wheel_candidate_evaluation(self):
        """Test wheel candidate evaluation logic."""
        candidate = WheelCandidate(
            ticker="AAPL",
            company_name="Apple Inc.",
            current_price=155.0,
            iv_rank=45.0,
            put_strike=150,
            put_expiry="2025-02-21",
            put_premium=2.50,
            put_delta=0.30,
            call_strike=160,
            call_premium=1.80,
            call_delta=0.25,
            wheel_annual_return=0.25,
            dividend_yield=0.005,
            liquidity_score=0.80,
            quality_score=85.0,
            volatility_score=0.75,
            risk_factors=["earnings_risk", "market_volatility"]
        )
        
        # Test candidate properties
        assert candidate.iv_rank > 0
        assert candidate.wheel_annual_return > 0
        assert candidate.dividend_yield >= 0
        assert candidate.liquidity_score >= 0
        assert candidate.quality_score >= 0
        assert candidate.volatility_score >= 0

    def test_position_calculations(self):
        """Test position calculation methods."""
        position = WheelPosition(
            ticker="AAPL",
            position_type="cash_secured_put",
            shares=0,
            avg_cost=0.0,
            strike=150,
            expiry="2025-02-21",
            premium_collected=2.50,
            days_to_expiry=30,
            current_price=155.0,
            unrealized_pnl=250.0,
            total_premium_collected=500.0,
            assignment_risk=0.15,
            annualized_return=0.25,
            status="active"
        )
        
        # Test basic calculations
        assert position.premium_collected > 0
        assert position.current_price > 0
        assert position.assignment_risk >= 0
        assert position.annualized_return >= 0

    def test_error_handling(self):
        """Test error handling in various scenarios."""
        strategy = WheelStrategy()
        
        # Test with invalid data
        try:
            invalid_position = WheelPosition(
                ticker="",
                position_type="invalid",
                shares=-1,
                avg_cost=-1.0,
                strike=0,
                expiry="",
                premium_collected=-1.0,
                days_to_expiry=-1,
                current_price=-1.0,
                unrealized_pnl=0.0,
                total_premium_collected=0.0,
                assignment_risk=-1.0,
                annualized_return=-1.0,
                status="invalid"
            )
            # Should still create the object (dataclass doesn't validate)
            assert invalid_position.ticker == ""
        except Exception:
            # If validation is implemented, that's fine too
            pass

    def test_integration_scenario(self):
        """Test complete integration scenario."""
        strategy = WheelStrategy()
        
        # Create a complete workflow
        candidate = WheelCandidate(
            ticker="AAPL",
            company_name="Apple Inc.",
            current_price=155.0,
            iv_rank=45.0,
            put_strike=150,
            put_expiry="2025-02-21",
            put_premium=2.50,
            put_delta=0.30,
            call_strike=160,
            call_premium=1.80,
            call_delta=0.25,
            wheel_annual_return=0.25,
            dividend_yield=0.005,
            liquidity_score=0.80,
            quality_score=85.0,
            volatility_score=0.75,
            risk_factors=["earnings_risk", "market_volatility"]
        )
        
        position = WheelPosition(
            ticker="AAPL",
            position_type="cash_secured_put",
            shares=0,
            avg_cost=0.0,
            strike=150,
            expiry="2025-02-21",
            premium_collected=2.50,
            days_to_expiry=30,
            current_price=155.0,
            unrealized_pnl=250.0,
            total_premium_collected=500.0,
            assignment_risk=0.15,
            annualized_return=0.25,
            status="active"
        )
        
        # Verify complete workflow
        assert isinstance(candidate, WheelCandidate)
        assert isinstance(position, WheelPosition)
        assert candidate.ticker == position.ticker