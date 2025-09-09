#!/usr/bin/env python3
"""
WSB Strategy #3: Debit Call Spreads
More repeatable than naked calls with reduced theta/IV risk
"""

import argparse
import math
import json
from datetime import date, datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import csv

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Run: pip install -r wsb_requirements.txt")
    exit(1)


@dataclass
class SpreadOpportunity:
    ticker: str
    scan_date: date
    spot_price: float
    trend_strength: float  # 0-1 score
    expiry_date: str
    days_to_expiry: int
    long_strike: int
    short_strike: int
    spread_width: int
    long_premium: float
    short_premium: float
    net_debit: float
    max_profit: float
    max_profit_pct: float
    breakeven: float
    prob_profit: float  # Estimated based on delta
    risk_reward: float
    iv_rank: float  # 0-100 percentile
    volume_score: float  # Liquidity score


class DebitSpreadScanner:
    def __init__(self):
        self.watchlist=[
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX",
            "CRM", "ADBE", "ORCL", "AMD", "QCOM", "UBER", "SNOW", "COIN",
            "PLTR", "ROKU", "ZM", "SHOP", "SQ", "PYPL", "TWLO"
        ]
    
    def norm_cdf(self, x: float) -> float:
        """Standard normal cumulative distribution function"""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))
    
    def black_scholes_call(self, S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
        """Black-Scholes call price and delta"""
        if T <= 0 or sigma <= 0:
            return max(S - K, 0), 1.0 if S > K else 0.0
            
        d1=(math.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))
        d2=d1 - sigma*math.sqrt(T)
        
        call_price=S * self.norm_cdf(d1) - K * math.exp(-r*T) * self.norm_cdf(d2)
        delta=self.norm_cdf(d1)
        
        return max(call_price, 0), delta
    
    def calculate_iv_rank(self, ticker: str, current_iv: float) -> float:
        """Calculate IV rank (current IV vs 52-week range)"""
        try:
            stock=yf.Ticker(ticker)
            hist=stock.history(period="1y")
            if hist.empty:
                return 50.0  # Neutral
            
            # Estimate historical IV from price volatility
            returns=hist['Close'].pct_change().dropna()
            hist_vol=returns.rolling(20).std() * math.sqrt(252)
            
            if hist_vol.empty:
                return 50.0
                
            iv_min=hist_vol.min()
            iv_max=hist_vol.max()
            
            if iv_max== iv_min:
                return 50.0
                
            rank = (current_iv - iv_min) / (iv_max - iv_min) * 100
            return max(0, min(100, rank))
            
        except:
            return 50.0
    
    def assess_trend_strength(self, ticker: str) -> float:
        """Assess bullish trend strength (0-1 score)"""
        try:
            stock=yf.Ticker(ticker)
            hist=stock.history(period="60d")
            if len(hist) < 50:
                return 0.5
            
            prices=hist['Close'].values
            current = prices[-1]
            
            # Multiple trend indicators
            scores = []
            
            # 1. Price vs moving averages
            sma_20 = np.mean(prices[-20:])
            sma_50=np.mean(prices[-50:])
            
            if current > sma_20 > sma_50:
                scores.append(0.8)
            elif current > sma_20:
                scores.append(0.6)
            else:
                scores.append(0.2)
            
            # 2. Recent momentum (10-day)
            momentum=(current / prices[-10] - 1) * 5  # Scale to 0-1
            scores.append(max(0, min(1, momentum)))
            
            # 3. Trend consistency (fewer whipsaws=higher score)
            direction_changes=0
            for i in range(len(prices)-10, len(prices)-1):
                if i <= 0:
                    continue
                curr_trend=prices[i+1] > prices[i]
                prev_trend = prices[i] > prices[i-1] if i > 0 else curr_trend
                if curr_trend != prev_trend:
                    direction_changes += 1
            
            consistency = max(0, 1 - direction_changes / 10)
            scores.append(consistency)
            
            # 4. Volume trend (rising volume=conviction)
            volumes=hist['Volume'].values
            recent_vol = np.mean(volumes[-10:])
            past_vol=np.mean(volumes[-30:-10])
            
            if past_vol > 0:
                vol_trend=min(1, recent_vol / past_vol)
                scores.append(vol_trend * 0.5)  # Lower weight
            else:
                scores.append(0.5)
            
            return np.mean(scores)
            
        except:
            return 0.5
    
    def get_options_data(self, ticker: str, expiry: str) -> Optional[pd.DataFrame]:
        """Get options chain for expiry"""
        try:
            stock=yf.Ticker(ticker)
            chain=stock.option_chain(expiry)
            
            if chain.calls.empty:
                return None
                
            calls=chain.calls.copy()
            
            # Filter for reasonable strikes and volume
            calls=calls[(calls['volume'] >= 10) | (calls['openInterest'] >= 50)]
            calls=calls[calls['bid'] > 0.05]
            calls = calls[calls['ask'] > 0.05]
            
            # Add mid price
            calls['mid'] = (calls['bid'] + calls['ask']) / 2
            calls['spread'] = calls['ask'] - calls['bid']
            calls['spread_pct'] = calls['spread'] / calls['mid']
            
            return calls
            
        except Exception as e:
            return None
    
    def find_optimal_spreads(self, ticker: str, spot: float, expiry: str, 
                           calls: pd.DataFrame) -> List[SpreadOpportunity]:
        """Find optimal spread combinations"""
        opportunities=[]
        
        days_to_exp = (datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()).days
        if days_to_exp <= 0:
            return opportunities
        
        # Target strikes around current price
        min_long_strike=spot * 0.95  # Max 5% ITM
        max_long_strike = spot * 1.10  # Max 10% OTM
        
        suitable_calls = calls[
            (calls['strike'] >= min_long_strike) & 
            (calls['strike'] <= max_long_strike) &
            (calls['spread_pct'] < 0.3)  # Reasonable spreads
        ].copy()
        
        if len(suitable_calls) < 2:
            return opportunities
        
        # Try different spread widths
        for width in [5, 10, 15, 20]:
            for _, long_call in suitable_calls.iterrows():
                long_strike=long_call['strike']
                short_strike = long_strike + width
                
                # Find matching short call
                short_calls = suitable_calls[
                    (suitable_calls['strike'] == short_strike)
                ]
                
                if short_calls.empty:
                    continue
                    
                short_call=short_calls.iloc[0]
                
                # Calculate spread economics
                long_premium = long_call['mid']
                short_premium = short_call['mid']
                net_debit = long_premium - short_premium
                
                if net_debit <= 0.10:  # Minimum viable premium
                    continue
                    
                max_profit = width - net_debit
                max_profit_pct = (max_profit / net_debit) * 100
                breakeven=long_strike + net_debit
                
                # Risk-reward ratio
                risk_reward = max_profit / net_debit
                
                # Estimate probability of profit using delta
                long_delta = long_call.get('delta', 0.5)
                short_delta=short_call.get('delta', 0.3)
                
                # Rough probability estimate
                if breakeven <= spot:
                    prob_profit=0.7  # ITM breakeven
                else:
                    distance_ratio = (breakeven - spot) / spot
                    prob_profit=max(0.1, 0.6 - distance_ratio * 3)
                
                # Volume/liquidity score
                long_vol=long_call.get('volume', 0) + long_call.get('openInterest', 0)
                short_vol=short_call.get('volume', 0) + short_call.get('openInterest', 0)
                volume_score=min(long_vol, short_vol) / 1000  # Normalize
                volume_score=max(0, min(1, volume_score))
                
                # Only include spreads with decent risk-reward
                if risk_reward >= 1.2 and max_profit_pct >= 20:
                    
                    # Estimate IV from option prices
                    try:
                        estimated_iv=self.estimate_iv_from_price(
                            spot, long_strike, days_to_exp/365, long_premium
                        )
                    except:
                        estimated_iv=0.25
                    
                    iv_rank = self.calculate_iv_rank(ticker, estimated_iv)
                    trend_strength=self.assess_trend_strength(ticker)
                    
                    opportunity=SpreadOpportunity(
                        ticker=ticker,
                        scan_date=date.today(),
                        spot_price=spot,
                        trend_strength=trend_strength,
                        expiry_date=expiry,
                        days_to_expiry=days_to_exp,
                        long_strike=int(long_strike),
                        short_strike=int(short_strike),
                        spread_width=width,
                        long_premium=long_premium,
                        short_premium=short_premium,
                        net_debit=net_debit,
                        max_profit=max_profit,
                        max_profit_pct=max_profit_pct,
                        breakeven=breakeven,
                        prob_profit=prob_profit,
                        risk_reward=risk_reward,
                        iv_rank=iv_rank,
                        volume_score=volume_score
                    )
                    
                    opportunities.append(opportunity)
        
        # Sort by risk-adjusted return
        opportunities.sort(
            key=lambda x: x.risk_reward * x.prob_profit * x.volume_score * x.trend_strength, 
            reverse=True
        )
        
        return opportunities[:3]  # Top 3 per ticker
    
    def estimate_iv_from_price(self, S: float, K: float, T: float, market_price: float) -> float:
        """Estimate implied volatility using Newton-Raphson"""
        try:
            iv=0.25  # Initial guess
            
            for _ in range(20):  # Max iterations
                price, delta=self.black_scholes_call(S, K, T, 0.04, iv)
                vega=S * math.sqrt(T) * self.norm_cdf(
                    (math.log(S/K) + (0.04 + 0.5*iv*iv)*T) / (iv*math.sqrt(T))
                ) * math.exp(-0.04*T)
                
                if abs(vega) < 1e-6:
                    break
                    
                diff=price - market_price
                if abs(diff) < 0.01:
                    break
                    
                iv -= diff / vega
                iv=max(0.01, min(2.0, iv))  # Keep reasonable bounds
            
            return iv
            
        except:
            return 0.25
    
    def scan_all_spreads(self, min_days: int=20, max_days: int=60) -> List[SpreadOpportunity]:
        """Scan all tickers for spread opportunities"""
        all_opportunities=[]
        
        print(f"🔍 Scanning {len(self.watchlist)} tickers for debit spreads...")
        
        for ticker in self.watchlist:
            try:
                stock=yf.Ticker(ticker)
                
                # Get current price
                hist=stock.history(period="1d")
                if hist.empty:
                    continue
                    
                spot=hist['Close'].iloc[-1]
                
                # Get available expiries
                try:
                    expiries = stock.options
                    if not expiries:
                        continue
                except:
                    continue
                
                # Filter expiries by DTE range
                valid_expiries = []
                today = date.today()
                
                for exp_str in expiries:
                    try:
                        exp_date=datetime.strptime(exp_str, "%Y-%m-%d").date()
                        days=(exp_date - today).days
                        if min_days <= days <= max_days:
                            valid_expiries.append(exp_str)
                    except:
                        continue
                
                if not valid_expiries:
                    continue
                
                # Scan expiries (focus on first 2-3 to avoid rate limits)
                for expiry in valid_expiries[:3]:
                    calls=self.get_options_data(ticker, expiry)
                    if calls is None:
                        continue
                    
                    spreads=self.find_optimal_spreads(ticker, spot, expiry, calls)
                    all_opportunities.extend(spreads)
                    
                    if spreads:
                        print(f"📊 {ticker}: Found {len(spreads)} spread opportunities")
                        
            except Exception as e:
                print(f"❌ Error scanning {ticker}: {e}")
                continue
        
        # Global ranking
        all_opportunities.sort(
            key=lambda x: (
                x.risk_reward * 
                x.prob_profit * 
                x.volume_score * 
                x.trend_strength * 
                (1 + x.max_profit_pct/100)
            ),
            reverse=True
        )
        
        return all_opportunities
    
    def format_opportunities(self, opportunities: List[SpreadOpportunity], limit: int=10) -> str:
        """Format opportunities for display"""
        if not opportunities:
            return "🔍 No suitable debit spread opportunities found."
            
        output=f"\n📈 TOP DEBIT SPREAD OPPORTUNITIES ({min(limit, len(opportunities))} shown)\n"
        output += "=" * 80 + "\n"
        
        for i, opp in enumerate(opportunities[:limit], 1):
            output += f"\n{i}. {opp.ticker} ${opp.long_strike}/{opp.short_strike} Call Spread"
            output += f" ({opp.expiry_date}, {opp.days_to_expiry}d)\n"
            output += f"   Spot: ${opp.spot_price:.2f} | Breakeven: ${opp.breakeven:.2f}\n"
            output += f"   Net Debit: ${opp.net_debit:.2f} | Max Profit: ${opp.max_profit:.2f} ({opp.max_profit_pct:.0f}%)\n"
            output += f"   Risk/Reward: {opp.risk_reward:.1f} | Prob Profit: {opp.prob_profit:.0%}\n"
            output += f"   Trend Strength: {opp.trend_strength:.2f} | IV Rank: {opp.iv_rank:.0f}\n"
            output += f"   Volume Score: {opp.volume_score:.2f}\n"
        
        output += "\n💡 DEBIT SPREAD ADVANTAGES:\n"
        output += "• Lower cost than naked calls\n"
        output += "• Reduced theta decay risk\n"
        output += "• Less IV crush exposure\n"
        output += "• More consistent profit potential\n"
        output += "• Better risk management\n"
        
        output += "\n⚠️  RISK WARNINGS:\n"
        output += "• Limited upside vs naked calls\n"
        output += "• Both legs can expire worthless\n"
        output += "• Liquidity risk on both strikes\n"
        output += "• Early assignment risk on short leg\n"
        
        return output
    
    def save_to_csv(self, opportunities: List[SpreadOpportunity], filename: str):
        """Save opportunities to CSV"""
        if not opportunities:
            return
            
        with open(filename, 'w', newline='') as csvfile:
            fieldnames=opportunities[0].__dict__.keys()
            writer=csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for opp in opportunities:
                writer.writerow(asdict(opp))
        
        print(f"💾 Saved {len(opportunities)} opportunities to {filename}")


def main():
    parser=argparse.ArgumentParser(description="WSB Debit Spread Scanner")
    parser.add_argument('--min-days', type=int, default=20,
                       help='Minimum days to expiry')
    parser.add_argument('--max-days', type=int, default=60,
                       help='Maximum days to expiry') 
    parser.add_argument('--output', choices=['json', 'text', 'csv'], default='text',
                       help='Output format')
    parser.add_argument('--limit', type=int, default=10,
                       help='Maximum results to show')
    parser.add_argument('--min-risk-reward', type=float, default=1.2,
                       help='Minimum risk-reward ratio')
    parser.add_argument('--save-csv', type=str,
                       help='Save results to CSV file')
    
    args=parser.parse_args()
    
    scanner=DebitSpreadScanner()
    opportunities=scanner.scan_all_spreads(args.min_days, args.max_days)
    
    # Filter by risk-reward
    opportunities=[opp for opp in opportunities if opp.risk_reward >= args.min_risk_reward]
    
    if args.save_csv:
        scanner.save_to_csv(opportunities, args.save_csv)
    
    if args.output== 'json':print(json.dumps([asdict(opp) for opp in opportunities[:args.limit]], 
                        indent=2, default=str))
    elif args.output== 'csv':scanner.save_to_csv(opportunities[:args.limit], 'debit_spreads.csv')
    else:
        print(scanner.format_opportunities(opportunities, args.limit))


if __name__== "__main__":main()