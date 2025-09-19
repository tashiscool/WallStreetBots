"""Wash-Sale and Tax-Lot Model for US Tax Compliance."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from datetime import date, timedelta


@dataclass
class Lot:
    """Tax lot for FIFO accounting."""
    open_date: date
    qty: int
    cost_basis: float  # per share


@dataclass
class Trade:
    """Individual trade record."""
    trade_date: date
    symbol: str
    side: str       # 'buy' or 'sell'
    qty: int
    price: float


@dataclass
class RealizedPnL:
    """Realized P&L with wash-sale adjustments."""
    symbol: str
    trade_date: date
    proceeds: float
    basis: float
    pnl: float
    wash_sale_disallowed: float  # amount disallowed this sale
    deferred_basis_added: float  # added to replacement lot(s)


class WashSaleEngine:
    """Implements wash-sale rules and FIFO tax-lot accounting."""
    
    def __init__(self, window_days: int = 30):
        """
        Initialize wash-sale engine.
        
        Args:
            window_days: Wash-sale window in days (30 for US)
        """
        self.window = timedelta(days=window_days)

    def process(self, symbol: str, lots: List[Lot], trades: List[Trade]) -> List[RealizedPnL]:
        """
        Process trades with wash-sale adjustments.
        
        Args:
            symbol: Symbol being traded
            lots: Existing tax lots (FIFO order)
            trades: All trades for the symbol
            
        Returns:
            List of realized P&L records
        """
        # Sort lots by open date (FIFO)
        lots = sorted(lots, key=lambda x: x.open_date)
        
        # Separate buys and sells
        buys = [t for t in trades if t.side == 'buy']
        sells = [t for t in trades if t.side == 'sell']
        sells = sorted(sells, key=lambda x: x.trade_date)
        
        out: List[RealizedPnL] = []
        
        for sell_trade in sells:
            remaining = sell_trade.qty
            proceeds = sell_trade.qty * sell_trade.price
            basis = 0.0
            disallowed = 0.0
            
            # Match lots FIFO
            while remaining > 0 and lots:
                lot = lots[0]
                use = min(remaining, lot.qty)
                basis += use * lot.cost_basis
                lot.qty -= use
                remaining -= use
                
                if lot.qty == 0:
                    lots.pop(0)
            
            pnl = proceeds - basis
            
            # Check for wash-sale (only if loss)
            if pnl < 0:
                ws_start = sell_trade.trade_date - self.window
                ws_end = sell_trade.trade_date + self.window
                
                # Find replacement purchases in wash-sale window
                replacement_buys = [
                    t for t in buys 
                    if ws_start <= t.trade_date <= ws_end
                ]
                
                if replacement_buys:
                    # Calculate wash-sale disallowance
                    replacement_qty = sum(t.qty for t in replacement_buys)
                    ratio = min(1.0, replacement_qty / sell_trade.qty)
                    disallowed = -pnl * ratio
                    pnl += disallowed  # reduce loss
                    
                    # Add disallowed loss to replacement lot basis
                    need = int(sell_trade.qty * ratio)
                    for buy_trade in sorted(replacement_buys, key=lambda x: x.trade_date):
                        if need <= 0:
                            break
                        add_qty = min(need, buy_trade.qty)
                        adjusted_basis = buy_trade.price + (disallowed / max(need, 1))
                        
                        # Create/adjust lot with increased basis
                        lots.append(Lot(
                            open_date=buy_trade.trade_date,
                            qty=add_qty,
                            cost_basis=adjusted_basis
                        ))
                        need -= add_qty
            
            out.append(RealizedPnL(
                symbol=symbol,
                trade_date=sell_trade.trade_date,
                proceeds=proceeds,
                basis=basis,
                pnl=pnl,
                wash_sale_disallowed=disallowed,
                deferred_basis_added=disallowed
            ))
        
        return out

    def calculate_tax_lots(self, trades: List[Trade]) -> List[Lot]:
        """
        Calculate tax lots from trades (simplified).
        
        Args:
            trades: All trades for a symbol
            
        Returns:
            List of tax lots
        """
        lots = []
        buys = [t for t in trades if t.side == 'buy']
        
        for buy_trade in buys:
            lots.append(Lot(
                open_date=buy_trade.trade_date,
                qty=buy_trade.qty,
                cost_basis=buy_trade.price
            ))
        
        return lots

    def get_wash_sale_summary(self, realized_pnl: List[RealizedPnL]) -> dict:
        """Get summary of wash-sale adjustments."""
        total_disallowed = sum(r.wash_sale_disallowed for r in realized_pnl)
        total_deferred = sum(r.deferred_basis_added for r in realized_pnl)
        
        return {
            'total_wash_sale_disallowed': total_disallowed,
            'total_deferred_basis': total_deferred,
            'wash_sale_trades': len([r for r in realized_pnl if r.wash_sale_disallowed > 0]),
            'total_trades': len(realized_pnl),
            'wash_sale_percentage': (
                len([r for r in realized_pnl if r.wash_sale_disallowed > 0]) / 
                max(len(realized_pnl), 1) * 100
            )
        }


class TaxLotManager:
    """Manages tax lots across multiple symbols."""
    
    def __init__(self):
        self.lots_by_symbol: dict[str, List[Lot]] = {}
        self.wash_sale_engine = WashSaleEngine()

    def add_trade(self, symbol: str, trade: Trade) -> List[RealizedPnL]:
        """
        Add a trade and process wash-sale rules.
        
        Args:
            symbol: Symbol being traded
            trade: Trade to add
            
        Returns:
            List of realized P&L (empty for buys)
        """
        if symbol not in self.lots_by_symbol:
            self.lots_by_symbol[symbol] = []
        
        if trade.side == 'buy':
            # Add new lot
            self.lots_by_symbol[symbol].append(Lot(
                open_date=trade.trade_date,
                qty=trade.qty,
                cost_basis=trade.price
            ))
            return []
        else:
            # Process sell with wash-sale rules
            return self.wash_sale_engine.process(
                symbol, 
                self.lots_by_symbol[symbol], 
                [trade]
            )

    def get_unrealized_pnl(self, symbol: str, current_price: float) -> float:
        """Calculate unrealized P&L for a symbol."""
        if symbol not in self.lots_by_symbol:
            return 0.0
        
        total_cost = sum(lot.qty * lot.cost_basis for lot in self.lots_by_symbol[symbol])
        total_qty = sum(lot.qty for lot in self.lots_by_symbol[symbol])
        current_value = total_qty * current_price
        
        return current_value - total_cost

    def get_position_summary(self, symbol: str) -> dict:
        """Get position summary for a symbol."""
        if symbol not in self.lots_by_symbol:
            return {'qty': 0, 'avg_cost': 0.0, 'lots': 0}
        
        lots = self.lots_by_symbol[symbol]
        total_qty = sum(lot.qty for lot in lots)
        total_cost = sum(lot.qty * lot.cost_basis for lot in lots)
        avg_cost = total_cost / max(total_qty, 1)
        
        return {
            'qty': total_qty,
            'avg_cost': avg_cost,
            'lots': len(lots),
            'total_cost': total_cost
        }


# Example usage and testing
if __name__ == "__main__":
    def test_wash_sale():
        """Test wash-sale engine."""
        print("=== Wash-Sale Engine Test ===")
        
        # Create test trades
        trades = [
            Trade(date(2024, 1, 1), 'AAPL', 'buy', 100, 150.0),
            Trade(date(2024, 1, 15), 'AAPL', 'sell', 50, 140.0),  # Loss
            Trade(date(2024, 1, 20), 'AAPL', 'buy', 30, 145.0),   # Replacement buy
            Trade(date(2024, 2, 1), 'AAPL', 'sell', 30, 155.0),  # Gain
        ]
        
        # Process with wash-sale engine
        engine = WashSaleEngine()
        lots = engine.calculate_tax_lots(trades)
        
        # Process sells
        sells = [t for t in trades if t.side == 'sell']
        realized_pnl = engine.process('AAPL', lots, sells)
        
        print("Realized P&L:")
        for pnl in realized_pnl:
            print(f"  {pnl.trade_date}: P&L=${pnl.pnl:.2f}, "
                  f"Wash-sale disallowed=${pnl.wash_sale_disallowed:.2f}")
        
        # Summary
        summary = engine.get_wash_sale_summary(realized_pnl)
        print(f"\nWash-sale summary: {summary}")
    
    test_wash_sale()


