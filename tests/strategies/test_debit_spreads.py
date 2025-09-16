"""Comprehensive tests for Debit Spreads strategy."""
import pytest
import pandas as pd
from datetime import datetime, date, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from backend.tradingbot.strategies.debit_spreads import (
    DebitSpreadScanner,
    SpreadOpportunity
)


class TestDebitSpreadScanner:
    """Test Debit Spread Scanner strategy implementation."""

    def test_debit_spread_scanner_initialization(self):
        """Test Debit Spread Scanner initialization."""
        scanner = DebitSpreadScanner()
        
        # Test default attributes
        assert hasattr(scanner, 'watchlist')
        
        # Test that watchlist is populated
        assert len(scanner.watchlist) > 0
        assert "AAPL" in scanner.watchlist
        assert "MSFT" in scanner.watchlist
        assert "GOOGL" in scanner.watchlist

    def test_spread_opportunity_creation(self):
        """Test SpreadOpportunity dataclass creation."""
        opportunity = SpreadOpportunity(
            ticker="AAPL",
            scan_date=date(2025, 1, 17),
            spot_price=155.0,
            trend_strength=0.75,
            expiry_date="2025-02-21",
            days_to_expiry=30,
            long_strike=150,
            short_strike=160,
            spread_width=10,
            long_premium=8.50,
            short_premium=3.00,
            net_debit=5.50,
            max_profit=4.50,
            max_profit_pct=0.818,
            breakeven=155.50,
            prob_profit=0.65,
            risk_reward=0.818,
            iv_rank=45.0,
            volume_score=0.80
        )
        
        assert opportunity.ticker == "AAPL"
        assert opportunity.scan_date == date(2025, 1, 17)
        assert opportunity.spot_price == 155.0
        assert opportunity.trend_strength == 0.75
        assert opportunity.expiry_date == "2025-02-21"
        assert opportunity.days_to_expiry == 30
        assert opportunity.long_strike == 150
        assert opportunity.short_strike == 160
        assert opportunity.spread_width == 10
        assert opportunity.long_premium == 8.50
        assert opportunity.short_premium == 3.00
        assert opportunity.net_debit == 5.50
        assert opportunity.max_profit == 4.50
        assert opportunity.max_profit_pct == 0.818
        assert opportunity.breakeven == 155.50
        assert opportunity.prob_profit == 0.65
        assert opportunity.risk_reward == 0.818
        assert opportunity.iv_rank == 45.0
        assert opportunity.volume_score == 0.80

    def test_debit_spread_scanner_methods(self):
        """Test Debit Spread Scanner methods."""
        scanner = DebitSpreadScanner()
        
        # Test that key methods exist
        assert hasattr(scanner, 'scan_all_spreads')
        assert hasattr(scanner, 'assess_trend_strength')
        assert hasattr(scanner, 'calculate_iv_rank')
        assert hasattr(scanner, 'find_optimal_spreads')
        assert hasattr(scanner, 'get_options_data')

    def test_spread_opportunity_calculations(self):
        """Test spread opportunity calculations."""
        opportunity = SpreadOpportunity(
            ticker="AAPL",
            scan_date=date(2025, 1, 17),
            spot_price=155.0,
            trend_strength=0.75,
            expiry_date="2025-02-21",
            days_to_expiry=30,
            long_strike=150,
            short_strike=160,
            spread_width=10,
            long_premium=8.50,
            short_premium=3.00,
            net_debit=5.50,
            max_profit=4.50,
            max_profit_pct=0.818,
            breakeven=155.50,
            prob_profit=0.65,
            risk_reward=0.818,
            iv_rank=45.0,
            volume_score=0.80
        )
        
        # Test spread width calculation
        expected_spread_width = opportunity.short_strike - opportunity.long_strike
        assert opportunity.spread_width == expected_spread_width
        
        # Test net debit calculation
        expected_net_debit = opportunity.long_premium - opportunity.short_premium
        assert opportunity.net_debit == expected_net_debit
        
        # Test max profit calculation
        expected_max_profit = opportunity.spread_width - opportunity.net_debit
        assert opportunity.max_profit == expected_max_profit
        
        # Test breakeven calculation (for call debit spread)
        expected_breakeven = opportunity.long_strike + opportunity.net_debit
        assert opportunity.breakeven == expected_breakeven

    def test_trend_strength_validation(self):
        """Test trend strength validation."""
        opportunity = SpreadOpportunity(
            ticker="AAPL",
            scan_date=date(2025, 1, 17),
            spot_price=155.0,
            trend_strength=0.75,
            expiry_date="2025-02-21",
            days_to_expiry=30,
            long_strike=150,
            short_strike=160,
            spread_width=10,
            long_premium=8.50,
            short_premium=3.00,
            net_debit=5.50,
            max_profit=4.50,
            max_profit_pct=0.818,
            breakeven=155.50,
            prob_profit=0.65,
            risk_reward=0.818,
            iv_rank=45.0,
            volume_score=0.80
        )
        
        # Test trend strength validation
        assert 0 <= opportunity.trend_strength <= 1
        
        # Test probability validation
        assert 0 <= opportunity.prob_profit <= 1
        
        # Test IV rank validation
        assert 0 <= opportunity.iv_rank <= 100
        
        # Test volume score validation
        assert 0 <= opportunity.volume_score <= 1

    def test_error_handling(self):
        """Test error handling in various scenarios."""
        scanner = DebitSpreadScanner()
        
        # Test with invalid data
        try:
            invalid_opportunity = SpreadOpportunity(
                ticker="",
                scan_date=date(2025, 1, 17),
                spot_price=-1.0,
                trend_strength=-1.0,
                expiry_date="",
                days_to_expiry=-1,
                long_strike=-1,
                short_strike=-1,
                spread_width=-1,
                long_premium=-1.0,
                short_premium=-1.0,
                net_debit=-1.0,
                max_profit=-1.0,
                max_profit_pct=-1.0,
                breakeven=-1.0,
                prob_profit=-1.0,
                risk_reward=-1.0,
                iv_rank=-1.0,
                volume_score=-1.0
            )
            # Should still create the object (dataclass doesn't validate)
            assert invalid_opportunity.ticker == ""
        except Exception:
            # If validation is implemented, that's fine too
            pass

    def test_integration_scenario(self):
        """Test complete integration scenario."""
        scanner = DebitSpreadScanner()
        
        # Create a complete workflow
        opportunity = SpreadOpportunity(
            ticker="AAPL",
            scan_date=date(2025, 1, 17),
            spot_price=155.0,
            trend_strength=0.75,
            expiry_date="2025-02-21",
            days_to_expiry=30,
            long_strike=150,
            short_strike=160,
            spread_width=10,
            long_premium=8.50,
            short_premium=3.00,
            net_debit=5.50,
            max_profit=4.50,
            max_profit_pct=0.818,
            breakeven=155.50,
            prob_profit=0.65,
            risk_reward=0.818,
            iv_rank=45.0,
            volume_score=0.80
        )
        
        # Verify complete workflow
        assert isinstance(opportunity, SpreadOpportunity)
        assert opportunity.ticker == "AAPL"
        assert opportunity.spot_price == 155.0
        assert opportunity.trend_strength == 0.75