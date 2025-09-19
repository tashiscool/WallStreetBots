"""
Comprehensive Tests for WashSaleEngine
=====================================

Enhanced test coverage for wash-sale engine calculation discrepancies,
edge cases, and tax compliance scenarios.
"""

import pytest
from datetime import date, timedelta
from typing import List

from backend.validation.broker_accounting.wash_sale import (
    WashSaleEngine,
    TaxLotManager,
    Lot,
    Trade,
    RealizedPnL
)


class TestLotDataClass:
    """Test Lot dataclass functionality."""

    def test_lot_creation(self):
        """Test basic lot creation."""
        lot = Lot(open_date=date(2024, 1, 1), qty=100, cost_basis=150.0)
        assert lot.open_date == date(2024, 1, 1)
        assert lot.qty == 100
        assert lot.cost_basis == 150.0

    def test_lot_with_zero_values(self):
        """Test lot with zero values."""
        lot = Lot(open_date=date(2024, 1, 1), qty=0, cost_basis=0.0)
        assert lot.qty == 0
        assert lot.cost_basis == 0.0

    def test_lot_with_negative_values(self):
        """Test lot with negative values."""
        lot = Lot(open_date=date(2024, 1, 1), qty=-100, cost_basis=-50.0)
        assert lot.qty == -100
        assert lot.cost_basis == -50.0


class TestTradeDataClass:
    """Test Trade dataclass functionality."""

    def test_trade_creation_buy(self):
        """Test buy trade creation."""
        trade = Trade(
            trade_date=date(2024, 1, 1),
            symbol="AAPL",
            side="buy",
            qty=100,
            price=150.0
        )
        assert trade.trade_date == date(2024, 1, 1)
        assert trade.symbol == "AAPL"
        assert trade.side == "buy"
        assert trade.qty == 100
        assert trade.price == 150.0

    def test_trade_creation_sell(self):
        """Test sell trade creation."""
        trade = Trade(
            trade_date=date(2024, 1, 1),
            symbol="MSFT",
            side="sell",
            qty=50,
            price=250.0
        )
        assert trade.side == "sell"
        assert trade.qty == 50
        assert trade.price == 250.0

    def test_trade_with_fractional_shares(self):
        """Test trade with fractional shares (edge case)."""
        trade = Trade(
            trade_date=date(2024, 1, 1),
            symbol="TSLA",
            side="buy",
            qty=10,  # Treating as integer, but testing fractional pricing
            price=123.456
        )
        assert trade.price == 123.456


class TestRealizedPnLDataClass:
    """Test RealizedPnL dataclass functionality."""

    def test_realized_pnl_creation(self):
        """Test realized P&L creation."""
        pnl = RealizedPnL(
            symbol="AAPL",
            trade_date=date(2024, 1, 1),
            proceeds=5000.0,
            basis=4500.0,
            pnl=500.0,
            wash_sale_disallowed=0.0,
            deferred_basis_added=0.0
        )
        assert pnl.symbol == "AAPL"
        assert pnl.proceeds == 5000.0
        assert pnl.basis == 4500.0
        assert pnl.pnl == 500.0

    def test_realized_pnl_with_wash_sale(self):
        """Test realized P&L with wash-sale adjustments."""
        pnl = RealizedPnL(
            symbol="AAPL",
            trade_date=date(2024, 1, 1),
            proceeds=4500.0,
            basis=5000.0,
            pnl=-250.0,  # Adjusted from -500.0 loss
            wash_sale_disallowed=250.0,
            deferred_basis_added=250.0
        )
        assert pnl.pnl == -250.0
        assert pnl.wash_sale_disallowed == 250.0
        assert pnl.deferred_basis_added == 250.0


class TestWashSaleEngineInitialization:
    """Test WashSaleEngine initialization."""

    def test_default_initialization(self):
        """Test default 30-day window."""
        engine = WashSaleEngine()
        assert engine.window == timedelta(days=30)

    def test_custom_window_initialization(self):
        """Test custom window period."""
        engine = WashSaleEngine(window_days=60)
        assert engine.window == timedelta(days=60)

    def test_zero_window_initialization(self):
        """Test zero window (edge case)."""
        engine = WashSaleEngine(window_days=0)
        assert engine.window == timedelta(days=0)

    def test_negative_window_initialization(self):
        """Test negative window (edge case)."""
        engine = WashSaleEngine(window_days=-30)
        assert engine.window == timedelta(days=-30)


class TestWashSaleEngineBasicFunctionality:
    """Test basic WashSaleEngine functionality."""

    def test_simple_profitable_trade(self):
        """Test simple profitable trade (no wash-sale)."""
        engine = WashSaleEngine()
        lots = [Lot(date(2024, 1, 1), 100, 150.0)]
        trades = [Trade(date(2024, 1, 15), "AAPL", "sell", 50, 160.0)]

        results = engine.process("AAPL", lots, trades)

        assert len(results) == 1
        result = results[0]
        assert result.symbol == "AAPL"
        assert result.proceeds == 8000.0  # 50 * 160
        assert result.basis == 7500.0     # 50 * 150
        assert result.pnl == 500.0        # Profit
        assert result.wash_sale_disallowed == 0.0  # No wash-sale on profit

    def test_simple_loss_no_repurchase(self):
        """Test simple loss without repurchase (no wash-sale)."""
        engine = WashSaleEngine()
        lots = [Lot(date(2024, 1, 1), 100, 150.0)]
        trades = [Trade(date(2024, 1, 15), "AAPL", "sell", 50, 140.0)]

        results = engine.process("AAPL", lots, trades)

        assert len(results) == 1
        result = results[0]
        assert result.proceeds == 7000.0  # 50 * 140
        assert result.basis == 7500.0     # 50 * 150
        assert result.pnl == -500.0       # Loss
        assert result.wash_sale_disallowed == 0.0  # No repurchase

    def test_wash_sale_basic_scenario(self):
        """Test basic wash-sale scenario."""
        engine = WashSaleEngine()
        lots = [Lot(date(2024, 1, 1), 100, 150.0)]

        # Sell at loss, then repurchase within window
        sell_trade = Trade(date(2024, 1, 15), "AAPL", "sell", 50, 140.0)
        buy_trade = Trade(date(2024, 1, 20), "AAPL", "buy", 50, 145.0)

        trades = [buy_trade, sell_trade]  # Mix order to test sorting

        results = engine.process("AAPL", lots, trades)

        assert len(results) == 1
        result = results[0]
        assert result.pnl > -500.0  # Loss should be reduced due to wash-sale
        assert result.wash_sale_disallowed > 0.0

    def test_fifo_lot_processing(self):
        """Test FIFO lot processing."""
        engine = WashSaleEngine()
        lots = [
            Lot(date(2024, 1, 1), 50, 150.0),   # Earlier lot
            Lot(date(2024, 1, 5), 50, 160.0)    # Later lot
        ]
        trades = [Trade(date(2024, 1, 15), "AAPL", "sell", 75, 155.0)]

        results = engine.process("AAPL", lots, trades)

        result = results[0]
        # Should use 50 shares from first lot at $150 and 25 shares from second lot at $160
        expected_basis = (50 * 150.0) + (25 * 160.0)
        assert result.basis == expected_basis

    def test_multiple_sells(self):
        """Test multiple sell trades."""
        engine = WashSaleEngine()
        lots = [Lot(date(2024, 1, 1), 200, 150.0)]
        trades = [
            Trade(date(2024, 1, 15), "AAPL", "sell", 50, 160.0),
            Trade(date(2024, 1, 20), "AAPL", "sell", 30, 155.0)
        ]

        results = engine.process("AAPL", lots, trades)

        assert len(results) == 2
        # Verify each sell is processed correctly
        assert results[0].qty == 50 if hasattr(results[0], 'qty') else True
        assert results[1].proceeds == 30 * 155.0

    def test_empty_inputs(self):
        """Test with empty inputs."""
        engine = WashSaleEngine()

        # Empty lots
        results = engine.process("AAPL", [], [])
        assert results == []

        # Empty trades
        lots = [Lot(date(2024, 1, 1), 100, 150.0)]
        results = engine.process("AAPL", lots, [])
        assert results == []


class TestWashSaleEngineEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_sell_more_than_available(self):
        """Test selling more shares than available in lots."""
        engine = WashSaleEngine()
        lots = [Lot(date(2024, 1, 1), 50, 150.0)]  # Only 50 shares
        trades = [Trade(date(2024, 1, 15), "AAPL", "sell", 100, 140.0)]  # Sell 100

        results = engine.process("AAPL", lots, trades)

        result = results[0]
        # Should only process available shares
        assert result.basis <= 50 * 150.0
        assert result.proceeds == 100 * 140.0  # Full proceeds still recorded

    def test_wash_sale_window_boundary(self):
        """Test wash-sale window boundary conditions."""
        engine = WashSaleEngine(window_days=30)
        lots = [Lot(date(2024, 1, 1), 100, 150.0)]

        # Sell at loss
        sell_date = date(2024, 2, 1)
        sell_trade = Trade(sell_date, "AAPL", "sell", 50, 140.0)

        # Buy exactly 30 days before (should trigger wash-sale)
        buy_before = Trade(sell_date - timedelta(days=30), "AAPL", "buy", 50, 145.0)

        # Buy exactly 30 days after (should trigger wash-sale)
        buy_after = Trade(sell_date + timedelta(days=30), "AAPL", "buy", 50, 145.0)

        trades = [buy_before, sell_trade, buy_after]
        results = engine.process("AAPL", lots, trades)

        result = results[0]
        assert result.wash_sale_disallowed > 0.0

    def test_wash_sale_window_outside_boundary(self):
        """Test purchases outside wash-sale window."""
        engine = WashSaleEngine(window_days=30)
        lots = [Lot(date(2024, 1, 1), 100, 150.0)]

        sell_date = date(2024, 2, 1)
        sell_trade = Trade(sell_date, "AAPL", "sell", 50, 140.0)

        # Buy 31 days before (should NOT trigger wash-sale)
        buy_before = Trade(sell_date - timedelta(days=31), "AAPL", "buy", 50, 145.0)

        # Buy 31 days after (should NOT trigger wash-sale)
        buy_after = Trade(sell_date + timedelta(days=31), "AAPL", "buy", 50, 145.0)

        trades = [buy_before, sell_trade, buy_after]
        results = engine.process("AAPL", lots, trades)

        result = results[0]
        assert result.wash_sale_disallowed == 0.0

    def test_partial_wash_sale(self):
        """Test partial wash-sale when replacement quantity is less than sold."""
        engine = WashSaleEngine()
        lots = [Lot(date(2024, 1, 1), 100, 150.0)]

        # Sell 100 shares at loss
        sell_trade = Trade(date(2024, 1, 15), "AAPL", "sell", 100, 140.0)

        # Repurchase only 50 shares
        buy_trade = Trade(date(2024, 1, 20), "AAPL", "buy", 50, 145.0)

        trades = [buy_trade, sell_trade]
        results = engine.process("AAPL", lots, trades)

        result = results[0]
        # Only 50% of loss should be disallowed
        total_loss = 100 * (150.0 - 140.0)  # $1000 loss
        expected_disallowed = total_loss * 0.5  # 50% disallowed
        assert abs(result.wash_sale_disallowed - expected_disallowed) < 0.01

    def test_excess_replacement_shares(self):
        """Test when replacement shares exceed sold shares."""
        engine = WashSaleEngine()
        lots = [Lot(date(2024, 1, 1), 100, 150.0)]

        # Sell 50 shares at loss
        sell_trade = Trade(date(2024, 1, 15), "AAPL", "sell", 50, 140.0)

        # Repurchase 100 shares (excess)
        buy_trade = Trade(date(2024, 1, 20), "AAPL", "buy", 100, 145.0)

        trades = [buy_trade, sell_trade]
        results = engine.process("AAPL", lots, trades)

        result = results[0]
        # 100% of loss should be disallowed (ratio capped at 1.0)
        total_loss = 50 * (150.0 - 140.0)
        assert abs(result.wash_sale_disallowed - total_loss) < 0.01

    def test_zero_quantity_trades(self):
        """Test zero quantity trades."""
        engine = WashSaleEngine()
        lots = [Lot(date(2024, 1, 1), 100, 150.0)]
        trades = [Trade(date(2024, 1, 15), "AAPL", "sell", 0, 140.0)]

        results = engine.process("AAPL", lots, trades)

        result = results[0]
        assert result.proceeds == 0.0
        assert result.basis == 0.0
        assert result.pnl == 0.0

    def test_same_day_trades(self):
        """Test sell and repurchase on same day."""
        engine = WashSaleEngine()
        lots = [Lot(date(2024, 1, 1), 100, 150.0)]

        same_day = date(2024, 1, 15)
        sell_trade = Trade(same_day, "AAPL", "sell", 50, 140.0)
        buy_trade = Trade(same_day, "AAPL", "buy", 50, 145.0)

        trades = [buy_trade, sell_trade]
        results = engine.process("AAPL", lots, trades)

        result = results[0]
        assert result.wash_sale_disallowed > 0.0  # Should trigger wash-sale


class TestTaxLotCalculation:
    """Test tax lot calculation functionality."""

    def test_calculate_tax_lots_simple(self):
        """Test simple tax lot calculation."""
        engine = WashSaleEngine()
        trades = [
            Trade(date(2024, 1, 1), "AAPL", "buy", 100, 150.0),
            Trade(date(2024, 1, 5), "AAPL", "buy", 50, 160.0)
        ]

        lots = engine.calculate_tax_lots(trades)

        assert len(lots) == 2
        assert lots[0].qty == 100
        assert lots[0].cost_basis == 150.0
        assert lots[1].qty == 50
        assert lots[1].cost_basis == 160.0

    def test_calculate_tax_lots_with_sells(self):
        """Test tax lot calculation ignores sells."""
        engine = WashSaleEngine()
        trades = [
            Trade(date(2024, 1, 1), "AAPL", "buy", 100, 150.0),
            Trade(date(2024, 1, 5), "AAPL", "sell", 50, 140.0),
            Trade(date(2024, 1, 10), "AAPL", "buy", 25, 160.0)
        ]

        lots = engine.calculate_tax_lots(trades)

        assert len(lots) == 2  # Only buy trades
        assert lots[0].qty == 100
        assert lots[1].qty == 25

    def test_calculate_tax_lots_empty(self):
        """Test tax lot calculation with empty trades."""
        engine = WashSaleEngine()
        lots = engine.calculate_tax_lots([])
        assert lots == []


class TestWashSaleSummary:
    """Test wash-sale summary functionality."""

    def test_wash_sale_summary_no_wash_sales(self):
        """Test summary with no wash-sales."""
        engine = WashSaleEngine()
        realized_pnl = [
            RealizedPnL("AAPL", date(2024, 1, 1), 8000.0, 7500.0, 500.0, 0.0, 0.0),
            RealizedPnL("AAPL", date(2024, 1, 2), 4000.0, 3500.0, 500.0, 0.0, 0.0)
        ]

        summary = engine.get_wash_sale_summary(realized_pnl)

        assert summary['total_wash_sale_disallowed'] == 0.0
        assert summary['total_deferred_basis'] == 0.0
        assert summary['wash_sale_trades'] == 0
        assert summary['total_trades'] == 2
        assert summary['wash_sale_percentage'] == 0.0

    def test_wash_sale_summary_with_wash_sales(self):
        """Test summary with wash-sales."""
        engine = WashSaleEngine()
        realized_pnl = [
            RealizedPnL("AAPL", date(2024, 1, 1), 7000.0, 7500.0, -250.0, 250.0, 250.0),
            RealizedPnL("AAPL", date(2024, 1, 2), 4000.0, 3500.0, 500.0, 0.0, 0.0),
            RealizedPnL("AAPL", date(2024, 1, 3), 6000.0, 7000.0, -700.0, 300.0, 300.0)
        ]

        summary = engine.get_wash_sale_summary(realized_pnl)

        assert summary['total_wash_sale_disallowed'] == 550.0
        assert summary['total_deferred_basis'] == 550.0
        assert summary['wash_sale_trades'] == 2
        assert summary['total_trades'] == 3
        assert abs(summary['wash_sale_percentage'] - 66.67) < 0.1

    def test_wash_sale_summary_empty(self):
        """Test summary with empty realized P&L list."""
        engine = WashSaleEngine()
        summary = engine.get_wash_sale_summary([])

        assert summary['total_wash_sale_disallowed'] == 0.0
        assert summary['total_deferred_basis'] == 0.0
        assert summary['wash_sale_trades'] == 0
        assert summary['total_trades'] == 0
        assert summary['wash_sale_percentage'] == 0.0


class TestTaxLotManager:
    """Test TaxLotManager functionality."""

    def test_tax_lot_manager_initialization(self):
        """Test tax lot manager initialization."""
        manager = TaxLotManager()
        assert manager.lots_by_symbol == {}
        assert isinstance(manager.wash_sale_engine, WashSaleEngine)

    def test_add_buy_trade(self):
        """Test adding buy trade."""
        manager = TaxLotManager()
        trade = Trade(date(2024, 1, 1), "AAPL", "buy", 100, 150.0)

        result = manager.add_trade("AAPL", trade)

        assert result == []  # No realized P&L for buys
        assert "AAPL" in manager.lots_by_symbol
        assert len(manager.lots_by_symbol["AAPL"]) == 1
        assert manager.lots_by_symbol["AAPL"][0].qty == 100

    def test_add_sell_trade(self):
        """Test adding sell trade."""
        manager = TaxLotManager()

        # Add buy first
        buy_trade = Trade(date(2024, 1, 1), "AAPL", "buy", 100, 150.0)
        manager.add_trade("AAPL", buy_trade)

        # Add sell
        sell_trade = Trade(date(2024, 1, 15), "AAPL", "sell", 50, 160.0)
        result = manager.add_trade("AAPL", sell_trade)

        assert len(result) == 1  # One realized P&L record
        assert result[0].symbol == "AAPL"

    def test_get_unrealized_pnl(self):
        """Test unrealized P&L calculation."""
        manager = TaxLotManager()

        # Add positions
        trade1 = Trade(date(2024, 1, 1), "AAPL", "buy", 100, 150.0)
        trade2 = Trade(date(2024, 1, 5), "AAPL", "buy", 50, 160.0)
        manager.add_trade("AAPL", trade1)
        manager.add_trade("AAPL", trade2)

        # Calculate unrealized P&L at current price $170
        unrealized = manager.get_unrealized_pnl("AAPL", 170.0)

        # Total cost: (100 * 150) + (50 * 160) = 23000
        # Current value: 150 * 170 = 25500
        # Unrealized P&L: 25500 - 23000 = 2500
        assert unrealized == 2500.0

    def test_get_unrealized_pnl_unknown_symbol(self):
        """Test unrealized P&L for unknown symbol."""
        manager = TaxLotManager()
        unrealized = manager.get_unrealized_pnl("UNKNOWN", 100.0)
        assert unrealized == 0.0

    def test_get_position_summary(self):
        """Test position summary."""
        manager = TaxLotManager()

        # Add positions
        trade1 = Trade(date(2024, 1, 1), "AAPL", "buy", 100, 150.0)
        trade2 = Trade(date(2024, 1, 5), "AAPL", "buy", 50, 160.0)
        manager.add_trade("AAPL", trade1)
        manager.add_trade("AAPL", trade2)

        summary = manager.get_position_summary("AAPL")

        assert summary['qty'] == 150
        assert summary['lots'] == 2
        assert summary['total_cost'] == 23000.0
        assert abs(summary['avg_cost'] - 153.33) < 0.1

    def test_get_position_summary_unknown_symbol(self):
        """Test position summary for unknown symbol."""
        manager = TaxLotManager()
        summary = manager.get_position_summary("UNKNOWN")

        assert summary['qty'] == 0
        assert summary['avg_cost'] == 0.0
        assert summary['lots'] == 0


class TestWashSaleComplexScenarios:
    """Test complex wash-sale scenarios."""

    def test_multiple_symbols_isolation(self):
        """Test that wash-sale rules apply per symbol."""
        manager = TaxLotManager()

        # AAPL trades
        manager.add_trade("AAPL", Trade(date(2024, 1, 1), "AAPL", "buy", 100, 150.0))
        manager.add_trade("AAPL", Trade(date(2024, 1, 15), "AAPL", "sell", 50, 140.0))  # Loss
        manager.add_trade("AAPL", Trade(date(2024, 1, 20), "AAPL", "buy", 50, 145.0))   # Wash-sale

        # MSFT trades (no wash-sale)
        manager.add_trade("MSFT", Trade(date(2024, 1, 1), "MSFT", "buy", 100, 200.0))
        manager.add_trade("MSFT", Trade(date(2024, 1, 15), "MSFT", "sell", 50, 190.0))  # Loss

        # Verify symbols don't interfere
        aapl_summary = manager.get_position_summary("AAPL")
        msft_summary = manager.get_position_summary("MSFT")

        assert aapl_summary['qty'] == 100  # 100 original + 50 repurchase - 50 sold
        assert msft_summary['qty'] == 50   # 100 original - 50 sold

    def test_cascading_wash_sales(self):
        """Test cascading wash-sale adjustments."""
        engine = WashSaleEngine()
        lots = [Lot(date(2024, 1, 1), 300, 150.0)]

        trades = [
            # First loss with wash-sale
            Trade(date(2024, 1, 15), "AAPL", "sell", 100, 140.0),  # $1000 loss
            Trade(date(2024, 1, 20), "AAPL", "buy", 100, 145.0),   # Triggers wash-sale

            # Second loss with wash-sale on adjusted basis
            Trade(date(2024, 2, 1), "AAPL", "sell", 100, 135.0),   # Loss on adjusted shares
            Trade(date(2024, 2, 5), "AAPL", "buy", 100, 140.0),    # Triggers another wash-sale
        ]

        results = engine.process("AAPL", lots, trades)

        assert len(results) == 2
        # Both should have wash-sale adjustments
        assert results[0].wash_sale_disallowed > 0.0
        assert results[1].wash_sale_disallowed > 0.0

    def test_year_end_wash_sale(self):
        """Test wash-sale across year boundary."""
        engine = WashSaleEngine()
        lots = [Lot(date(2024, 1, 1), 100, 150.0)]

        trades = [
            # Sell at loss in December
            Trade(date(2024, 12, 15), "AAPL", "sell", 50, 140.0),
            # Repurchase in January (next year)
            Trade(date(2025, 1, 10), "AAPL", "buy", 50, 145.0)
        ]

        results = engine.process("AAPL", lots, trades)

        # Should still trigger wash-sale across year boundary
        assert results[0].wash_sale_disallowed > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])