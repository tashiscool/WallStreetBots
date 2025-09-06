#!/usr/bin/env python3
"""
YFinance hardening utilities - drop-in fixes for common crash points.
Implements safe mid calculation and empty history guards.
"""

import yfinance as yf
from typing import Optional

def safe_mid(bid: float, ask: float, last: float) -> float:
    """
    Safe mid calculation with fallbacks for 0 bid/ask scenarios.
    
    Args:
        bid: Bid price
        ask: Ask price  
        last: Last trade price
        
    Returns:
        Safe mid price with fallbacks
    """
    if bid > 0 and ask > 0: 
        return (bid + ask) / 2.0
    if last > 0: 
        return last
    if bid > 0: 
        return bid
    if ask > 0: 
        return ask
    return 0.01

def safe_history(ticker: str, period: str = "2d", interval: str = "5m") -> Optional[yf.Ticker]:
    """
    Safe history fetch with empty data guards.
    
    Args:
        ticker: Stock ticker symbol
        period: Data period
        interval: Data interval
        
    Returns:
        Ticker object with valid data or None if market closed/API glitch
    """
    try:
        tkr = yf.Ticker(ticker)
        hist = tkr.history(period=period, interval=interval)
        
        if hist is None or hist.empty:
            # market closed / API glitch
            return None
            
        return tkr
        
    except Exception:
        return None

def safe_option_chain(ticker: str, expiry: str) -> Optional[dict]:
    """
    Safe options chain fetch with validation.
    
    Args:
        ticker: Stock ticker symbol
        expiry: Options expiry date
        
    Returns:
        Options chain dict or None if unavailable
    """
    try:
        tkr = yf.Ticker(ticker)
        chain = tkr.option_chain(expiry)
        
        # Validate chain has data
        if chain.calls.empty and chain.puts.empty:
            return None
            
        return {
            'calls': chain.calls,
            'puts': chain.puts,
            'ticker': ticker,
            'expiry': expiry
        }
        
    except Exception:
        return None

def safe_current_price(ticker: str) -> Optional[float]:
    """
    Safe current price fetch with multiple fallbacks.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Current price or None if unavailable
    """
    try:
        tkr = yf.Ticker(ticker)
        
        # Try info first
        info = tkr.info
        if 'currentPrice' in info and info['currentPrice']:
            return float(info['currentPrice'])
            
        # Fallback to history
        hist = tkr.history(period="1d", interval="1m")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
            
        return None
        
    except Exception:
        return None
