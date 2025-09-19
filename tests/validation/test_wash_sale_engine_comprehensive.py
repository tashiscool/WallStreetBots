#!/usr/bin/env python3
"""
Comprehensive tests for WashSaleEngine to improve coverage.
Tests various wash sale scenarios and edge cases.
"""

import pytest
from datetime import date, timedelta
from backend.validation.broker_accounting.wash_sale import (
    WashSaleEngine, Lot, Trade, RealizedPnL
)


class TestWashSaleEngineComprehensive:
    """Comprehensive tests for WashSaleEngine."""

    def test_basic_wash_sale_scenario(self):
        """Test basic wash sale scenario."""
        engine = WashSaleEngine(window_days=30)
        
        # Create lots
        lots = [
            Lot(open_date=date(2024, 1, 1), qty=100, cost_basis=100.0),
            Lot(open_date=date(2024, 1, 15), qty=50, cost_basis=105.0)
        ]
        
        # Create trades
        trades = [
            Trade(trade_date=date(2024, 1, 20), symbol="AAPL", side="sell", qty=100, price=95.0),  # Loss
            Trade(trade_date=date(2024, 1, 25), symbol="AAPL", side="buy", qty=50, price=98.0)   # Replacement
        ]
        
        results = engine.process("AAPL", lots, trades)
        
        assert len(results) == 1
        result = results[0]
        assert result.symbol == "AAPL"
        assert result.trade_date == date(2024, 1, 20)
        assert result.proceeds == 9500.0  # 100 * 95
        assert result.basis == 10000.0    # 100 * 100
        assert result.pnl == -250.0       # Loss after wash sale adjustment
        assert result.wash_sale_disallowed > 0  # Some loss disallowed
        assert result.deferred_basis_added > 0  # Basis added to replacement

    def test_no_wash_sale_scenario(self):
        """Test scenario with no wash sale."""
        engine = WashSaleEngine(window_days=30)
        
        # Create lots
        lots = [
            Lot(open_date=date(2024, 1, 1), qty=100, cost_basis=100.0)
        ]
        
        # Create trades
        trades = [
            Trade(trade_date=date(2024, 1, 20), symbol="AAPL", side="sell", qty=100, price=110.0),  # Profit
            Trade(trade_date=date(2024, 1, 25), symbol="AAPL", side="buy", qty=50, price=98.0)     # No wash sale for profit
        ]
        
        results = engine.process("AAPL", lots, trades)
        
        assert len(results) == 1
        result = results[0]
        assert result.symbol == "AAPL"
        assert result.proceeds == 11000.0  # 100 * 110
        assert result.basis == 10000.0     # 100 * 100
        assert result.pnl == 1000.0        # Profit
        assert result.wash_sale_disallowed == 0.0  # No wash sale disallowed
        assert result.deferred_basis_added == 0.0  # No deferred basis

    def test_multiple_lots_fifo(self):
        """Test FIFO lot matching with multiple lots."""
        engine = WashSaleEngine(window_days=30)
        
        # Create lots (should be sorted by open_date)
        lots = [
            Lot(open_date=date(2024, 1, 1), qty=50, cost_basis=100.0),
            Lot(open_date=date(2024, 1, 10), qty=75, cost_basis=105.0),
            Lot(open_date=date(2024, 1, 20), qty=25, cost_basis=110.0)
        ]
        
        # Create trades
        trades = [
            Trade(trade_date=date(2024, 1, 25), symbol="AAPL", side="sell", qty=100, price=95.0)  # Loss
        ]
        
        results = engine.process("AAPL", lots, trades)
        
        assert len(results) == 1
        result = results[0]
        assert result.proceeds == 9500.0  # 100 * 95
        
        # Should use FIFO: 50 from first lot + 50 from second lot
        expected_basis = (50 * 100.0) + (50 * 105.0)
        assert result.basis == expected_basis

    def test_partial_lot_consumption(self):
        """Test partial consumption of lots."""
        engine = WashSaleEngine(window_days=30)
        
        # Create lots
        lots = [
            Lot(open_date=date(2024, 1, 1), qty=100, cost_basis=100.0)
        ]
        
        # Create trades
        trades = [
            Trade(trade_date=date(2024, 1, 20), symbol="AAPL", side="sell", qty=30, price=95.0)  # Partial sell
        ]
        
        results = engine.process("AAPL", lots, trades)
        
        assert len(results) == 1
        result = results[0]
        assert result.proceeds == 2850.0  # 30 * 95
        assert result.basis == 3000.0    # 30 * 100
        assert result.pnl == -150.0      # Loss

    def test_multiple_sells_same_day(self):
        """Test multiple sells on the same day."""
        engine = WashSaleEngine(window_days=30)
        
        # Create lots
        lots = [
            Lot(open_date=date(2024, 1, 1), qty=200, cost_basis=100.0)
        ]
        
        # Create trades
        trades = [
            Trade(trade_date=date(2024, 1, 20), symbol="AAPL", side="sell", qty=50, price=95.0),
            Trade(trade_date=date(2024, 1, 20), symbol="AAPL", side="sell", qty=75, price=98.0)
        ]
        
        results = engine.process("AAPL", lots, trades)
        
        assert len(results) == 2
        
        # First sell
        result1 = results[0]
        assert result1.proceeds == 4750.0  # 50 * 95
        assert result1.basis == 5000.0     # 50 * 100
        assert result1.pnl == -250.0      # Loss
        
        # Second sell
        result2 = results[1]
        assert result2.proceeds == 7350.0  # 75 * 98
        assert result2.basis == 7500.0     # 75 * 100
        assert result2.pnl == -150.0       # Loss

    def test_wash_sale_window_boundary(self):
        """Test wash sale window boundary conditions."""
        engine = WashSaleEngine(window_days=30)
        
        # Create lots
        lots = [
            Lot(open_date=date(2024, 1, 1), qty=100, cost_basis=100.0)
        ]
        
        # Create trades
        trades = [
            Trade(trade_date=date(2024, 1, 20), symbol="AAPL", side="sell", qty=100, price=95.0),  # Loss
            Trade(trade_date=date(2024, 2, 20), symbol="AAPL", side="buy", qty=50, price=98.0)     # Exactly 31 days later
        ]
        
        results = engine.process("AAPL", lots, trades)
        
        assert len(results) == 1
        result = results[0]
        # Should not be a wash sale since replacement is outside window
        assert result.wash_sale_disallowed == 0.0

    def test_wash_sale_within_window(self):
        """Test wash sale within window."""
        engine = WashSaleEngine(window_days=30)
        
        # Create lots
        lots = [
            Lot(open_date=date(2024, 1, 1), qty=100, cost_basis=100.0)
        ]
        
        # Create trades
        trades = [
            Trade(trade_date=date(2024, 1, 20), symbol="AAPL", side="sell", qty=100, price=95.0),  # Loss
            Trade(trade_date=date(2024, 1, 25), symbol="AAPL", side="buy", qty=50, price=98.0)     # Within window
        ]
        
        results = engine.process("AAPL", lots, trades)
        
        assert len(results) == 1
        result = results[0]
        assert result.wash_sale_disallowed > 0.0  # Should have wash sale disallowed

    def test_multiple_replacement_purchases(self):
        """Test multiple replacement purchases."""
        engine = WashSaleEngine(window_days=30)
        
        # Create lots
        lots = [
            Lot(open_date=date(2024, 1, 1), qty=100, cost_basis=100.0)
        ]
        
        # Create trades
        trades = [
            Trade(trade_date=date(2024, 1, 20), symbol="AAPL", side="sell", qty=100, price=95.0),  # Loss
            Trade(trade_date=date(2024, 1, 25), symbol="AAPL", side="buy", qty=30, price=98.0),    # Replacement 1
            Trade(trade_date=date(2024, 1, 30), symbol="AAPL", side="buy", qty=40, price=99.0)     # Replacement 2
        ]
        
        results = engine.process("AAPL", lots, trades)
        
        assert len(results) == 1
        result = results[0]
        assert result.wash_sale_disallowed > 0.0  # Should have wash sale disallowed

    def test_empty_lots_and_trades(self):
        """Test with empty lots and trades."""
        engine = WashSaleEngine(window_days=30)
        
        results = engine.process("AAPL", [], [])
        assert len(results) == 0

    def test_empty_lots_with_trades(self):
        """Test with empty lots but with trades."""
        engine = WashSaleEngine(window_days=30)
        
        trades = [
            Trade(trade_date=date(2024, 1, 20), symbol="AAPL", side="sell", qty=100, price=95.0)
        ]
        
        results = engine.process("AAPL", [], trades)
        
        assert len(results) == 1
        result = results[0]
        assert result.proceeds == 9500.0
        assert result.basis == 0.0  # No lots to match
        assert result.pnl == 9500.0  # All proceeds

    def test_empty_trades_with_lots(self):
        """Test with lots but no trades."""
        engine = WashSaleEngine(window_days=30)
        
        lots = [
            Lot(open_date=date(2024, 1, 1), qty=100, cost_basis=100.0)
        ]
        
        results = engine.process("AAPL", lots, [])
        assert len(results) == 0

    def test_custom_window_days(self):
        """Test with custom window days."""
        engine = WashSaleEngine(window_days=60)  # 60-day window
        
        # Create lots
        lots = [
            Lot(open_date=date(2024, 1, 1), qty=100, cost_basis=100.0)
        ]
        
        # Create trades
        trades = [
            Trade(trade_date=date(2024, 1, 20), symbol="AAPL", side="sell", qty=100, price=95.0),  # Loss
            Trade(trade_date=date(2024, 2, 20), symbol="AAPL", side="buy", qty=50, price=98.0)      # Within 60-day window
        ]
        
        results = engine.process("AAPL", lots, trades)
        
        assert len(results) == 1
        result = results[0]
        assert result.wash_sale_disallowed > 0.0  # Should have wash sale disallowed

    def test_lot_sorting_fifo(self):
        """Test that lots are properly sorted FIFO."""
        engine = WashSaleEngine(window_days=30)
        
        # Create lots in non-chronological order
        lots = [
            Lot(open_date=date(2024, 1, 20), qty=50, cost_basis=110.0),  # Latest
            Lot(open_date=date(2024, 1, 1), qty=100, cost_basis=100.0),  # Earliest
            Lot(open_date=date(2024, 1, 10), qty=75, cost_basis=105.0)   # Middle
        ]
        
        # Create trades
        trades = [
            Trade(trade_date=date(2024, 1, 25), symbol="AAPL", side="sell", qty=100, price=95.0)
        ]
        
        results = engine.process("AAPL", lots, trades)
        
        assert len(results) == 1
        result = results[0]
        
        # Should use FIFO: 100 from earliest lot
        expected_basis = 100 * 100.0
        assert result.basis == expected_basis

    def test_trade_sorting_by_date(self):
        """Test that trades are properly sorted by date."""
        engine = WashSaleEngine(window_days=30)
        
        # Create lots
        lots = [
            Lot(open_date=date(2024, 1, 1), qty=200, cost_basis=100.0)
        ]
        
        # Create trades in non-chronological order
        trades = [
            Trade(trade_date=date(2024, 1, 30), symbol="AAPL", side="sell", qty=50, price=95.0),
            Trade(trade_date=date(2024, 1, 20), symbol="AAPL", side="sell", qty=75, price=98.0),
            Trade(trade_date=date(2024, 1, 25), symbol="AAPL", side="sell", qty=25, price=97.0)
        ]
        
        results = engine.process("AAPL", lots, trades)
        
        assert len(results) == 3
        
        # Check that trades are processed in chronological order
        assert results[0].trade_date == date(2024, 1, 20)
        assert results[1].trade_date == date(2024, 1, 25)
        assert results[2].trade_date == date(2024, 1, 30)

    def test_complex_wash_sale_scenario(self):
        """Test complex wash sale scenario with multiple lots and trades."""
        engine = WashSaleEngine(window_days=30)
        
        # Create multiple lots
        lots = [
            Lot(open_date=date(2024, 1, 1), qty=100, cost_basis=100.0),
            Lot(open_date=date(2024, 1, 15), qty=50, cost_basis=105.0),
            Lot(open_date=date(2024, 1, 30), qty=75, cost_basis=110.0)
        ]
        
        # Create multiple trades
        trades = [
            Trade(trade_date=date(2024, 2, 1), symbol="AAPL", side="sell", qty=80, price=95.0),   # Loss
            Trade(trade_date=date(2024, 2, 5), symbol="AAPL", side="buy", qty=40, price=98.0),    # Replacement
            Trade(trade_date=date(2024, 2, 10), symbol="AAPL", side="sell", qty=60, price=102.0), # Profit
            Trade(trade_date=date(2024, 2, 15), symbol="AAPL", side="buy", qty=30, price=99.0)    # Another replacement
        ]
        
        results = engine.process("AAPL", lots, trades)
        
        assert len(results) == 2  # Two sell trades
        
        # First sell (loss with wash sale)
        result1 = results[0]
        assert result1.trade_date == date(2024, 2, 1)
        assert result1.pnl < 0  # Loss
        assert result1.wash_sale_disallowed > 0  # Wash sale disallowed
        
        # Second sell (loss with wash sale adjustment)
        result2 = results[1]
        assert result2.trade_date == date(2024, 2, 10)
        assert result2.pnl == 0.0  # Break-even due to wash sale basis adjustment
        assert result2.wash_sale_disallowed > 0.0  # Wash sale disallowed for this loss

    def test_edge_case_zero_quantities(self):
        """Test edge cases with zero quantities."""
        engine = WashSaleEngine(window_days=30)
        
        # Create lots
        lots = [
            Lot(open_date=date(2024, 1, 1), qty=100, cost_basis=100.0)
        ]
        
        # Create trades with zero quantity
        trades = [
            Trade(trade_date=date(2024, 1, 20), symbol="AAPL", side="sell", qty=0, price=95.0)
        ]
        
        results = engine.process("AAPL", lots, trades)
        
        assert len(results) == 1
        result = results[0]
        assert result.proceeds == 0.0
        assert result.basis == 0.0
        assert result.pnl == 0.0

    def test_edge_case_negative_prices(self):
        """Test edge cases with negative prices."""
        engine = WashSaleEngine(window_days=30)
        
        # Create lots
        lots = [
            Lot(open_date=date(2024, 1, 1), qty=100, cost_basis=100.0)
        ]
        
        # Create trades with negative price
        trades = [
            Trade(trade_date=date(2024, 1, 20), symbol="AAPL", side="sell", qty=100, price=-10.0)  # Negative price
        ]
        
        results = engine.process("AAPL", lots, trades)
        
        assert len(results) == 1
        result = results[0]
        assert result.proceeds == -1000.0  # 100 * -10
        assert result.basis == 10000.0    # 100 * 100
        assert result.pnl == -11000.0     # Large loss
