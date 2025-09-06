#!/usr/bin/env python3
"""
WSB Strategy: SPX/SPY 0DTE Credit Spreads
Most cited "actually profitable" 0DTE strategy on WSB
Sell ~30-delta defined-risk strangles/credit spreads at open, auto-close at ~25% profit
"""

import argparse
import math
import json
from datetime import date, datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import time

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Run: pip install -r wsb_requirements.txt")
    exit(1)


@dataclass
class CreditSpreadOpportunity:
    ticker: str
    strategy_type: str  # "put_credit_spread", "call_credit_spread", "iron_condor", "strangle"
    expiry_date: str
    dte: int  # Days to expiry (0 for 0DTE)
    
    # For spreads
    short_strike: float
    long_strike: float
    spread_width: float
    net_credit: float
    max_profit: float
    max_loss: float
    
    # For strangles/condors
    put_short_strike: Optional[float] = None
    put_long_strike: Optional[float] = None
    call_short_strike: Optional[float] = None 
    call_long_strike: Optional[float] = None
    
    # Risk metrics
    short_delta: float  # Target ~0.30 delta
    prob_profit: float  # Probability of profit
    profit_target: float  # 25% profit target
    break_even_lower: float
    break_even_upper: float
    
    # Market conditions
    iv_rank: float
    underlying_price: float
    expected_move: float  # Expected daily move
    volume_score: float  # Options liquidity score


class SPXCreditSpreadsScanner:
    def __init__(self):
        # Focus on SPX and liquid ETFs for credit spreads
        self.credit_tickers = [
            "SPX",   # S&P 500 Index (preferred for tax treatment)
            "SPY",   # SPDR S&P 500 ETF
            "QQQ",   # Invesco QQQ ETF
            "IWM",   # Russell 2000 ETF
        ]
        
        # Target delta for short strikes (WSB standard)
        self.target_short_delta = 0.30
        self.profit_target_pct = 0.25  # 25% profit target
        
    def norm_cdf(self, x: float) -> float:
        """Standard normal CDF"""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))
    
    def black_scholes_put(self, S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
        """Black-Scholes put price and delta"""
        if T <= 0 or sigma <= 0:
            return max(K - S, 0), -1.0 if S < K else 0.0
            
        d1 = (math.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        
        put_price = K * math.exp(-r*T) * self.norm_cdf(-d2) - S * self.norm_cdf(-d1)
        delta = -self.norm_cdf(-d1)
        
        return max(put_price, 0), delta
    
    def black_scholes_call(self, S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
        """Black-Scholes call price and delta"""
        if T <= 0 or sigma <= 0:
            return max(S - K, 0), 1.0 if S > K else 0.0
            
        d1 = (math.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        
        call_price = S * self.norm_cdf(d1) - K * math.exp(-r*T) * self.norm_cdf(d2)
        delta = self.norm_cdf(d1)
        
        return max(call_price, 0), delta
    
    def get_0dte_expiry(self) -> Optional[str]:
        """Get 0DTE expiry if available (Mon/Wed/Fri for SPX/SPY)"""
        today = date.today()
        weekday = today.weekday()  # 0=Monday, 4=Friday
        
        # SPX has 0DTE on Mon/Wed/Fri
        # SPY typically has 0DTE on Mon/Wed/Fri
        if weekday in [0, 2, 4]:  # Mon, Wed, Fri
            return today.strftime("%Y-%m-%d")
        
        return None
    
    def estimate_iv_from_expected_move(self, ticker: str, expected_move_pct: float) -> float:
        """Estimate IV from expected daily move"""
        try:
            # Convert daily expected move to annual IV
            # Expected move = S * IV * sqrt(T)
            # For 1 day: IV = expected_move / (S * sqrt(1/365))
            iv_estimate = expected_move_pct / math.sqrt(1/365)
            return max(0.10, min(1.0, iv_estimate))  # Reasonable bounds
        except:
            return 0.20  # Default IV
    
    def get_expected_move(self, ticker: str) -> float:
        """Estimate expected daily move from recent volatility"""
        try:
            if ticker == "SPX":
                # Use SPY as proxy for SPX
                proxy_ticker = "SPY"
            else:
                proxy_ticker = ticker
                
            stock = yf.Ticker(proxy_ticker)
            hist = stock.history(period="20d")
            
            if len(hist) < 10:
                return 0.015  # Default 1.5% daily move
            
            # Calculate recent daily volatility
            returns = hist['Close'].pct_change().dropna()
            daily_vol = returns.std()
            
            # Expected move is roughly 1 standard deviation
            return min(0.05, max(0.01, daily_vol))  # Cap between 1%-5%
            
        except:
            return 0.015  # Default fallback
    
    def find_target_delta_strike(self, ticker: str, expiry: str, option_type: str, 
                                target_delta: float, spot_price: float) -> Tuple[Optional[float], float, float]:
        """Find strike closest to target delta"""
        try:
            if ticker == "SPX":
                # Use SPY options as proxy for SPX (similar behavior)
                options_ticker = "SPY"
                multiplier = spot_price / yf.Ticker("SPY").history(period="1d")['Close'].iloc[-1]
            else:
                options_ticker = ticker
                multiplier = 1.0
            
            stock = yf.Ticker(options_ticker)
            chain = stock.option_chain(expiry)
            
            if option_type == "put":
                options_df = chain.puts
            else:
                options_df = chain.calls
            
            if options_df.empty:
                return None, 0.0, 0.0
            
            # Filter for reasonable strikes and volume
            options_df = options_df[
                (options_df['bid'] > 0.05) & 
                (options_df['ask'] > 0.05) &
                ((options_df['volume'] >= 10) | (options_df['openInterest'] >= 50))
            ].copy()
            
            if options_df.empty:
                return None, 0.0, 0.0
            
            # Find strike closest to target delta
            if 'delta' in options_df.columns:
                # Use actual delta if available
                if option_type == "put":
                    options_df['delta_abs'] = abs(options_df['delta'] + target_delta)
                else:
                    options_df['delta_abs'] = abs(options_df['delta'] - target_delta)
                
                best_option = options_df.loc[options_df['delta_abs'].idxmin()]
                strike = best_option['strike'] * multiplier
                actual_delta = abs(best_option['delta'])
                premium = (best_option['bid'] + best_option['ask']) / 2 * multiplier
                
            else:
                # Estimate delta using Black-Scholes
                best_strike = None
                best_delta_diff = float('inf')
                best_premium = 0.0
                actual_delta = 0.0
                
                iv_estimate = 0.20  # Default IV for estimation
                time_to_exp = 1/365 if expiry == date.today().strftime("%Y-%m-%d") else 7/365
                
                for _, option in options_df.iterrows():
                    strike_adj = option['strike'] * multiplier
                    
                    if option_type == "put":
                        _, delta = self.black_scholes_put(spot_price, strike_adj, time_to_exp, 0.04, iv_estimate)
                        delta_diff = abs(abs(delta) - target_delta)
                    else:
                        _, delta = self.black_scholes_call(spot_price, strike_adj, time_to_exp, 0.04, iv_estimate)
                        delta_diff = abs(delta - target_delta)
                    
                    if delta_diff < best_delta_diff:
                        best_delta_diff = delta_diff
                        best_strike = strike_adj
                        actual_delta = abs(delta)
                        best_premium = (option['bid'] + option['ask']) / 2 * multiplier
                
                strike = best_strike
                premium = best_premium
            
            return strike, actual_delta, premium
            
        except Exception as e:
            return None, 0.0, 0.0
    
    def calculate_spread_metrics(self, short_strike: float, long_strike: float, 
                               short_premium: float, long_premium: float) -> Tuple[float, float, float]:
        """Calculate spread metrics"""
        spread_width = abs(short_strike - long_strike)
        net_credit = short_premium - long_premium
        max_profit = net_credit
        max_loss = spread_width - net_credit
        
        return net_credit, max_profit, max_loss
    
    def scan_credit_spreads(self, dte_target: int = 0) -> List[CreditSpreadOpportunity]:
        """Scan for credit spread opportunities"""
        opportunities = []
        
        # Get target expiry
        if dte_target == 0:
            expiry = self.get_0dte_expiry()
            if not expiry:
                print("❌ No 0DTE expiry available today")
                return opportunities
        else:
            expiry = (date.today() + timedelta(days=dte_target)).strftime("%Y-%m-%d")
        
        print(f"🎯 Scanning credit spreads for {expiry} ({dte_target}DTE)...")
        
        for ticker in self.credit_tickers:
            try:
                if ticker == "SPX":
                    # SPX pricing (use SPY * ~10 as approximation)
                    spy_price = yf.Ticker("SPY").history(period="1d")['Close'].iloc[-1]
                    spot_price = spy_price * 10  # Rough SPX approximation
                else:
                    stock = yf.Ticker(ticker)
                    spot_price = stock.history(period="1d")['Close'].iloc[-1]
                
                expected_move_pct = self.get_expected_move(ticker)
                expected_move_points = spot_price * expected_move_pct
                
                print(f"  📊 {ticker}: ${spot_price:.2f}, Expected move: ±{expected_move_pct:.1%}")
                
                # 1. PUT CREDIT SPREADS (bullish/neutral)
                put_short_strike, put_short_delta, put_short_premium = self.find_target_delta_strike(
                    ticker, expiry, "put", self.target_short_delta, spot_price
                )
                
                if put_short_strike:
                    # Long strike is typically 5-10 points below short strike
                    spread_width = min(10, max(5, spot_price * 0.02))  # 2% of underlying
                    put_long_strike = put_short_strike - spread_width
                    
                    # Get long put premium
                    _, _, put_long_premium = self.find_target_delta_strike(
                        ticker, expiry, "put", 0.15, spot_price  # Lower delta for long strike
                    )
                    
                    if put_long_premium > 0:
                        net_credit, max_profit, max_loss = self.calculate_spread_metrics(
                            put_short_strike, put_long_strike, put_short_premium, put_long_premium
                        )
                        
                        if net_credit > 0.10:  # Minimum viable credit
                            prob_profit = 100 - (put_short_delta * 100)  # Rough approximation
                            profit_target = net_credit * self.profit_target_pct
                            
                            opportunity = CreditSpreadOpportunity(
                                ticker=ticker,
                                strategy_type="put_credit_spread",
                                expiry_date=expiry,
                                dte=dte_target,
                                short_strike=put_short_strike,
                                long_strike=put_long_strike,
                                spread_width=spread_width,
                                net_credit=net_credit,
                                max_profit=max_profit,
                                max_loss=max_loss,
                                short_delta=put_short_delta,
                                prob_profit=prob_profit,
                                profit_target=profit_target,
                                break_even_lower=put_short_strike - net_credit,
                                break_even_upper=float('inf'),
                                iv_rank=50.0,  # Simplified
                                underlying_price=spot_price,
                                expected_move=expected_move_pct,
                                volume_score=70.0  # Simplified
                            )
                            
                            opportunities.append(opportunity)
                
                # 2. CALL CREDIT SPREADS (bearish/neutral) 
                call_short_strike, call_short_delta, call_short_premium = self.find_target_delta_strike(
                    ticker, expiry, "call", self.target_short_delta, spot_price
                )
                
                if call_short_strike:
                    spread_width = min(10, max(5, spot_price * 0.02))
                    call_long_strike = call_short_strike + spread_width
                    
                    _, _, call_long_premium = self.find_target_delta_strike(
                        ticker, expiry, "call", 0.15, spot_price
                    )
                    
                    if call_long_premium > 0:
                        net_credit, max_profit, max_loss = self.calculate_spread_metrics(
                            call_short_strike, call_long_strike, call_short_premium, call_long_premium
                        )
                        
                        if net_credit > 0.10:
                            prob_profit = 100 - (call_short_delta * 100)
                            profit_target = net_credit * self.profit_target_pct
                            
                            opportunity = CreditSpreadOpportunity(
                                ticker=ticker,
                                strategy_type="call_credit_spread",
                                expiry_date=expiry,
                                dte=dte_target,
                                short_strike=call_short_strike,
                                long_strike=call_long_strike,
                                spread_width=spread_width,
                                net_credit=net_credit,
                                max_profit=max_profit,
                                max_loss=max_loss,
                                short_delta=call_short_delta,
                                prob_profit=prob_profit,
                                profit_target=profit_target,
                                break_even_lower=0,
                                break_even_upper=call_short_strike + net_credit,
                                iv_rank=50.0,
                                underlying_price=spot_price,
                                expected_move=expected_move_pct,
                                volume_score=70.0
                            )
                            
                            opportunities.append(opportunity)
                
                # 3. IRON CONDOR (neutral strategy)
                if put_short_strike and call_short_strike:
                    # Combine both spreads for iron condor
                    total_credit = net_credit * 2  # Rough estimate
                    total_max_loss = max_loss * 2  # Conservative estimate
                    
                    condor = CreditSpreadOpportunity(
                        ticker=ticker,
                        strategy_type="iron_condor",
                        expiry_date=expiry,
                        dte=dte_target,
                        short_strike=0,  # N/A for condor
                        long_strike=0,   # N/A for condor
                        spread_width=spread_width,
                        net_credit=total_credit,
                        max_profit=total_credit,
                        max_loss=total_max_loss,
                        put_short_strike=put_short_strike,
                        put_long_strike=put_long_strike,
                        call_short_strike=call_short_strike,
                        call_long_strike=call_long_strike,
                        short_delta=0.0,  # Net delta should be ~0
                        prob_profit=prob_profit * 0.8,  # Lower prob for condor
                        profit_target=total_credit * self.profit_target_pct,
                        break_even_lower=put_short_strike - total_credit,
                        break_even_upper=call_short_strike + total_credit,
                        iv_rank=50.0,
                        underlying_price=spot_price,
                        expected_move=expected_move_pct,
                        volume_score=70.0
                    )
                    
                    opportunities.append(condor)
                
            except Exception as e:
                print(f"  ❌ {ticker}: Error - {e}")
                continue
        
        # Sort by profit potential and probability
        opportunities.sort(
            key=lambda x: x.prob_profit * (x.max_profit / max(x.max_loss, 0.01)),
            reverse=True
        )
        
        return opportunities
    
    def format_opportunities(self, opportunities: List[CreditSpreadOpportunity]) -> str:
        """Format credit spread opportunities"""
        if not opportunities:
            return "🎯 No credit spread opportunities found."
        
        output = f"\n🎯 SPX/SPY CREDIT SPREAD OPPORTUNITIES ({len(opportunities)} found)\n"
        output += "=" * 80 + "\n"
        
        for i, opp in enumerate(opportunities, 1):
            if opp.strategy_type == "iron_condor":
                strategy_desc = f"IRON CONDOR: {opp.put_short_strike:.0f}P/{opp.call_short_strike:.0f}C short"
            else:
                direction = "PUT" if "put" in opp.strategy_type else "CALL"
                strategy_desc = f"{direction} SPREAD: {opp.short_strike:.0f}/{opp.long_strike:.0f}"
            
            risk_reward = opp.max_profit / opp.max_loss if opp.max_loss > 0 else 0
            
            output += f"\n{i}. {opp.ticker} {opp.dte}DTE - {strategy_desc}\n"
            output += f"   Underlying: ${opp.underlying_price:.2f} | Expected move: ±{opp.expected_move:.1%}\n"
            output += f"   Net Credit: ${opp.net_credit:.2f} | Max Profit: ${opp.max_profit:.2f}\n"
            output += f"   Max Loss: ${opp.max_loss:.2f} | Risk/Reward: {risk_reward:.2f}\n"
            output += f"   Prob Profit: {opp.prob_profit:.0f}% | 25% Target: ${opp.profit_target:.2f}\n"
            
            if opp.break_even_upper != float('inf'):
                output += f"   Break-evens: ${opp.break_even_lower:.2f} - ${opp.break_even_upper:.2f}\n"
            else:
                output += f"   Break-even: ${opp.break_even_lower:.2f}\n"
        
        output += "\n" + "=" * 80
        output += "\n🎯 WSB 0DTE CREDIT SPREAD RULES:\n"
        output += "• Target ~30 delta short strikes\n"
        output += "• Auto-close at 25% profit target\n"
        output += "• Prefer SPX for tax treatment (60/40 vs 100% short-term)\n"
        output += "• High win rate but occasional max loss weeks\n"
        output += "• Best on Mon/Wed/Fri (0DTE availability)\n"
        output += "• Position size: 1-2% of account max\n"
        
        output += "\n⚠️  CREDIT SPREAD WARNINGS:\n"
        output += "• Pin risk at expiration (stock at short strike)\n"
        output += "• Early assignment risk on ITM short options\n"
        output += "• Margin requirements can be substantial\n"
        output += "• One bad week can wipe out months of profits\n"
        output += "• Avoid earnings weeks and major events\n"
        
        return output


def main():
    parser = argparse.ArgumentParser(description="SPX/SPY 0DTE Credit Spreads Scanner")
    parser.add_argument('--dte', type=int, default=0,
                       help='Days to expiration (0 for same day)')
    parser.add_argument('--output', choices=['json', 'text'], default='text',
                       help='Output format')
    parser.add_argument('--min-credit', type=float, default=0.10,
                       help='Minimum net credit required')
    parser.add_argument('--target-delta', type=float, default=0.30,
                       help='Target delta for short strikes')
    
    args = parser.parse_args()
    
    scanner = SPXCreditSpreadsScanner()
    scanner.target_short_delta = args.target_delta
    
    opportunities = scanner.scan_credit_spreads(args.dte)
    
    # Filter by minimum credit
    opportunities = [opp for opp in opportunities if opp.net_credit >= args.min_credit]
    
    if args.output == 'json':
        print(json.dumps([asdict(opp) for opp in opportunities], indent=2, default=str))
    else:
        print(scanner.format_opportunities(opportunities))
    
    if opportunities:
        print(f"\n💡 Found {len(opportunities)} credit spread opportunities")
        print("⚠️  Remember: Most WSB users prefer SPX over SPY for tax advantages!")
    else:
        print("\n❌ No suitable credit spread opportunities found")
        if args.dte == 0:
            print("💡 Try running with --dte 1 or --dte 2 for more options")


if __name__ == "__main__":
    main()