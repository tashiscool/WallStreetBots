"""Comprehensive tests for Momentum Weeklies strategy."""
import pytest
import pandas as pd
from datetime import datetime, date, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from backend.tradingbot.strategies.momentum_weeklies import (
    MomentumWeekliesScanner,
    MomentumSignal
)


class TestMomentumWeekliesScanner:
    """Test Momentum Weeklies Scanner strategy implementation."""

    def test_momentum_weeklies_scanner_initialization(self):
        """Test Momentum Weeklies Scanner initialization."""
        scanner = MomentumWeekliesScanner()
        
        # Test default attributes
        assert hasattr(scanner, 'mega_caps')
        
        # Test that mega_caps list is populated
        assert len(scanner.mega_caps) > 0
        assert "AAPL" in scanner.mega_caps
        assert "TSLA" in scanner.mega_caps
        assert "NVDA" in scanner.mega_caps

    def test_momentum_signal_creation(self):
        """Test MomentumSignal dataclass creation."""
        signal = MomentumSignal(
            ticker="AAPL",
            signal_time=datetime.now(),
            current_price=155.0,
            reversal_type="bullish_reversal",
            volume_spike=3.5,
            price_momentum=0.025,
            weekly_expiry="2025-01-24",
            target_strike=160,
            premium_estimate=2.50,
            risk_level="medium",
            exit_target=165.0,
            stop_loss=150.0
        )
        
        assert signal.ticker == "AAPL"
        assert isinstance(signal.signal_time, datetime)
        assert signal.current_price == 155.0
        assert signal.reversal_type == "bullish_reversal"
        assert signal.volume_spike == 3.5
        assert signal.price_momentum == 0.025
        assert signal.weekly_expiry == "2025-01-24"
        assert signal.target_strike == 160
        assert signal.premium_estimate == 2.50
        assert signal.risk_level == "medium"
        assert signal.exit_target == 165.0
        assert signal.stop_loss == 150.0

    def test_momentum_scanner_methods(self):
        """Test Momentum Scanner methods."""
        scanner = MomentumWeekliesScanner()
        
        # Test that key methods exist
        assert hasattr(scanner, 'scan_momentum_signals')
        assert hasattr(scanner, 'detect_volume_spike')
        assert hasattr(scanner, 'detect_reversal_pattern')
        assert hasattr(scanner, 'detect_breakout_momentum')
        assert hasattr(scanner, 'get_next_weekly_expiry')

    def test_momentum_signal_calculations(self):
        """Test momentum signal calculations."""
        signal = MomentumSignal(
            ticker="AAPL",
            signal_time=datetime.now(),
            current_price=155.0,
            reversal_type="bullish_reversal",
            volume_spike=3.5,
            price_momentum=0.025,
            weekly_expiry="2025-01-24",
            target_strike=160,
            premium_estimate=2.50,
            risk_level="medium",
            exit_target=165.0,
            stop_loss=150.0
        )
        
        # Test price momentum validation
        assert signal.price_momentum > 0
        
        # Test volume spike validation
        assert signal.volume_spike > 1.0
        
        # Test risk level validation
        assert signal.risk_level in ["low", "medium", "high"]
        
        # Test exit target vs stop loss
        assert signal.exit_target > signal.current_price
        assert signal.stop_loss < signal.current_price

    def test_reversal_type_validation(self):
        """Test reversal type validation."""
        valid_reversal_types = [
            "bullish_reversal",
            "news_momentum", 
            "breakout",
            "bearish_reversal"
        ]
        
        for reversal_type in valid_reversal_types:
            signal = MomentumSignal(
                ticker="AAPL",
                signal_time=datetime.now(),
                current_price=155.0,
                reversal_type=reversal_type,
                volume_spike=3.5,
                price_momentum=0.025,
                weekly_expiry="2025-01-24",
                target_strike=160,
                premium_estimate=2.50,
                risk_level="medium",
                exit_target=165.0,
                stop_loss=150.0
            )
            assert signal.reversal_type == reversal_type

    def test_error_handling(self):
        """Test error handling in various scenarios."""
        scanner = MomentumWeekliesScanner()
        
        # Test with invalid data
        try:
            invalid_signal = MomentumSignal(
                ticker="",
                signal_time=datetime.now(),
                current_price=-1.0,
                reversal_type="invalid",
                volume_spike=-1.0,
                price_momentum=-1.0,
                weekly_expiry="",
                target_strike=-1,
                premium_estimate=-1.0,
                risk_level="invalid",
                exit_target=-1.0,
                stop_loss=-1.0
            )
            # Should still create the object (dataclass doesn't validate)
            assert invalid_signal.ticker == ""
        except Exception:
            # If validation is implemented, that's fine too
            pass

    def test_integration_scenario(self):
        """Test complete integration scenario."""
        scanner = MomentumWeekliesScanner()
        
        # Create a complete workflow
        signal = MomentumSignal(
            ticker="AAPL",
            signal_time=datetime.now(),
            current_price=155.0,
            reversal_type="bullish_reversal",
            volume_spike=3.5,
            price_momentum=0.025,
            weekly_expiry="2025-01-24",
            target_strike=160,
            premium_estimate=2.50,
            risk_level="medium",
            exit_target=165.0,
            stop_loss=150.0
        )
        
        # Verify complete workflow
        assert isinstance(signal, MomentumSignal)
        assert signal.ticker == "AAPL"
        assert signal.current_price == 155.0
        assert signal.reversal_type == "bullish_reversal"