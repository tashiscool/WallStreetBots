from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class LocateQuote:
    symbol: str
    available: bool
    borrow_bps: float  # annualized
    reason: Optional[str] = None

class BorrowClient:
    """Abstract locate/borrow interface. Replace with broker API if available."""
    def locate(self, symbol: str, qty: float) -> LocateQuote:
        # Placeholder policy: deny microcaps and return GC 30 bps for large caps
        sym = symbol.upper()
        if len(sym) > 5:  # naive microcap guard
            return LocateQuote(symbol=sym, available=False, borrow_bps=0.0, reason="Microcap/no locate")
        return LocateQuote(symbol=sym, available=True, borrow_bps=30.0)

def guard_can_short(borrow: BorrowClient, symbol: str, qty: float) -> float:
    q = borrow.locate(symbol, qty)
    if not q.available:
        raise PermissionError(f"Locate failed for {symbol}: {q.reason}")
    return q.borrow_bps