"""Property-based tests for order sizing.

This module contains property-based tests to ensure order sizing
logic is robust against edge cases, rounding errors, and invalid inputs.
"""
from __future__ import annotations
import pytest
from decimal import Decimal, ROUND_DOWN, InvalidOperation
import numpy as np

# Try to import hypothesis, skip tests if not available
try:
    from hypothesis import given, settings, strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Create dummy decorators for when hypothesis is not available
    def given(*args, **kwargs):
        def decorator(func):
            return pytest.mark.skip(reason="hypothesis not available")(func)
        return decorator

    def settings(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    class st:
        @staticmethod
        def floats(**kwargs):
            return None

        @staticmethod
        def integers(**kwargs):
            return None


def size_order(equity: float, pct: float, px: float, lot: int = 1) -> int:
    """Size order based on equity percentage.

    Args:
        equity: Available equity
        pct: Percentage of equity to use (0.0-1.0)
        px: Price per share
        lot: Lot size (minimum order size)

    Returns:
        Number of shares to order
    """
    # Handle NaN and infinite values
    if np.isnan(equity) or np.isnan(pct) or np.isnan(px):
        return 0
    if np.isinf(equity) or np.isinf(pct) or np.isinf(px):
        return 0
    
    try:
        notional = Decimal(str(equity)) * Decimal(str(pct))
        raw_shares = notional / Decimal(str(px))
        # Round down to the nearest lot size
        lot_count = int(raw_shares) // lot
        return lot_count * lot
    except (ValueError, OverflowError, InvalidOperation):
        return 0


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not available")
@settings(max_examples=500)
@given(
    equity=st.floats(min_value=1_000, max_value=10_000_000),
    pct=st.floats(min_value=0.0, max_value=0.5),
    px=st.floats(min_value=0.5, max_value=5_000)
)
def test_never_oversize(equity: float, pct: float, px: float):
    """Test that order sizing never exceeds intended allocation."""
    q = size_order(equity, pct, px)
    assert q * px <= equity * pct + 1e-6


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not available")
@settings(max_examples=500)
@given(
    equity=st.floats(min_value=1_000, max_value=10_000_000),
    pct=st.floats(min_value=0.0, max_value=0.5),
    px=st.floats(min_value=0.5, max_value=5_000)
)
def test_never_negative_quantity(equity: float, pct: float, px: float):
    """Test that order quantity is never negative."""
    q = size_order(equity, pct, px)
    assert q >= 0


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not available")
@settings(max_examples=500)
@given(
    equity=st.floats(min_value=1_000, max_value=10_000_000),
    pct=st.floats(min_value=0.0, max_value=0.5),
    px=st.floats(min_value=0.5, max_value=5_000),
    lot=st.integers(min_value=1, max_value=100)
)
def test_lot_size_compliance(equity: float, pct: float, px: float, lot: int):
    """Test that order quantity respects lot size."""
    q = size_order(equity, pct, px, lot)
    if q > 0:
        assert q % lot == 0


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not available")
@settings(max_examples=500)
@given(
    equity=st.floats(min_value=1_000, max_value=10_000_000),
    px=st.floats(min_value=0.5, max_value=5_000)
)
def test_zero_allocation_zero_quantity(equity: float, px: float):
    """Test that zero allocation results in zero quantity."""
    q = size_order(equity, 0.0, px)
    assert q == 0


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not available")
@settings(max_examples=500)
@given(
    equity=st.floats(min_value=1_000, max_value=10_000_000),
    pct=st.floats(min_value=0.0, max_value=0.5),
    px=st.floats(min_value=0.5, max_value=5_000)
)
def test_handles_nan_inputs(equity: float, pct: float, px: float):
    """Test that NaN inputs are handled gracefully."""
    # Test with NaN equity
    if not np.isnan(equity):
        q = size_order(float('nan'), pct, px)
        assert q == 0
    
    # Test with NaN percentage
    if not np.isnan(pct):
        q = size_order(equity, float('nan'), px)
        assert q == 0
    
    # Test with NaN price
    if not np.isnan(px):
        q = size_order(equity, pct, float('nan'))
        assert q == 0


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not available")
@settings(max_examples=500)
@given(
    equity=st.floats(min_value=1_000, max_value=10_000_000),
    pct=st.floats(min_value=0.0, max_value=0.5),
    px=st.floats(min_value=0.5, max_value=5_000)
)
def test_handles_inf_inputs(equity: float, pct: float, px: float):
    """Test that infinite inputs are handled gracefully."""
    # Test with infinite equity
    if not np.isinf(equity):
        q = size_order(float('inf'), pct, px)
        assert q >= 0  # Should not crash
    
    # Test with infinite percentage
    if not np.isinf(pct):
        q = size_order(equity, float('inf'), px)
        assert q >= 0  # Should not crash
    
    # Test with infinite price
    if not np.isinf(px):
        q = size_order(equity, pct, float('inf'))
        assert q == 0  # Infinite price should result in zero quantity


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not available")
@settings(max_examples=500)
@given(
    equity=st.floats(min_value=1_000, max_value=10_000_000),
    pct=st.floats(min_value=0.0, max_value=0.5),
    px=st.floats(min_value=0.5, max_value=5_000)
)
def test_monotonicity(equity: float, pct: float, px: float):
    """Test that order size increases monotonically with allocation."""
    q1 = size_order(equity, pct, px)
    q2 = size_order(equity, pct * 1.1, px)
    assert q2 >= q1


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not available")
@settings(max_examples=500)
@given(
    equity=st.floats(min_value=1_000, max_value=10_000_000),
    pct=st.floats(min_value=0.0, max_value=0.5),
    px=st.floats(min_value=0.5, max_value=5_000)
)
def test_price_inverse_relationship(equity: float, pct: float, px: float):
    """Test that order size decreases as price increases."""
    if px > 0.5:  # Ensure we have room to increase price
        q1 = size_order(equity, pct, px)
        q2 = size_order(equity, pct, px * 1.1)
        assert q2 <= q1


def test_edge_cases():
    """Test specific edge cases."""
    # Zero equity
    assert size_order(0, 0.1, 100) == 0
    
    # Very small equity
    assert size_order(1, 0.1, 100) == 0
    
    # Very high price
    assert size_order(10000, 0.1, 100000) == 0
    
    # Very small price
    q = size_order(10000, 0.1, 0.01)
    assert q > 0
    
    # Maximum allocation
    q = size_order(10000, 1.0, 100)
    assert q == 100


def test_lot_size_edge_cases():
    """Test lot size edge cases."""
    # Lot size larger than allocation (1000 * 0.1 / 100 = 1.0, but lot=1000 > 1.0)
    q = size_order(1000, 0.1, 100, lot=1000)
    assert q == 0  # Cannot buy 1000-lot when only 1 share is affordable
    
    # Lot size exactly equal to allocation
    q = size_order(1000, 0.1, 100, lot=1)
    assert q == 1
    
    # Large lot size
    q = size_order(10000, 0.1, 100, lot=10)
    assert q % 10 == 0

