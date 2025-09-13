#!/usr/bin/env python3
"""
WSB Strategy #5: 0DTE/Earnings Lotto Scanner
High-risk, high-reward plays with strict position sizing and discipline
"""

import argparse
import math
import json
from datetime import date, datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import calendar

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Run: pip install -r wsb_requirements.txt")
    exit(1)


@dataclass
class LottoPlay:
    ticker: str
    play_type: str  # "0dte", "earnings", "catalyst"
    expiry_date: str
    days_to_expiry: int
    strike: int
    option_type: str  # "call" or "put"
    current_premium: float
    breakeven: float
    current_spot: float
    catalyst_event: str
    expected_move: float  # Expected % move from earnings/catalyst
    max_position_size: float  # Max $ amount (strict risk control)
    max_contracts: int
    risk_level: str  # "extreme", "very_high", "high"
    win_probability: float  # Estimated probability of profit
    potential_return: float  # Potential return if successful
    stop_loss_price: float
    profit_target_price: float


@dataclass
class EarningsEvent:
    ticker: str
    company_name: str
    earnings_date: date
    time_of_day: str  # "BMO", "AMC", "Unknown"
    expected_move: float  # Expected % move based on IV
    avg_move_historical: float  # Average historical earnings move
    revenue_estimate: Optional[float]
    eps_estimate: Optional[float]
    sector: str


class LottoScanner:
    def __init__(self, max_risk_pct: float=1.0):
        """
        Initialize with strict risk limits
        max_risk_pct: Maximum % of account to risk per play (default 1%)
        """
        self.max_risk_pct=max_risk_pct / 100
        
        # High-volatility tickers suitable for lotto plays
        self.lotto_tickers = [
            # Mega caps with options liquidity
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX",
            
            # High-beta growth names
            "PLTR", "COIN", "HOOD", "SNOW", "CRWD", "ZM", "ROKU", "SHOP",
            "SQ", "PYPL", "UBER", "LYFT", "DOCU", "ZS", "NET", "DDOG",
            
            # Meme stocks (high volatility)
            "GME", "AMC", "BB", "NOK", "CLOV", "WISH", "SOFI", "RIVN",
            
            # High-beta sectors
            "XLF", "XLE", "XBI", "ARKK", "QQQ", "SPY", "IWM"
        ]
    
    def get_0dte_expiry(self) -> Optional[str]:
        """Get 0DTE expiry if available (usually Monday/Wednesday/Friday)"""
        today=date.today()
        
        # Check if today has 0DTE options (Mon/Wed/Fri for SPY, daily for others)
        weekday=today.weekday()  # 0=Monday, 4=Friday
        
        if weekday in [0, 2, 4]:  # Mon, Wed, Fri
            return today.strftime("%Y-%m-%d")
        
        # Find next available 0DTE date
        days_ahead=1
        while days_ahead <= 7:
            next_date = today + timedelta(days=days_ahead)
            if next_date.weekday() in [0, 2, 4]:
                return next_date.strftime("%Y-%m-%d")
            days_ahead += 1
        
        return None
    
    def get_weekly_expiry(self) -> str:
        """Get next weekly expiry (Friday)"""
        today=date.today()
        days_until_friday=(4 - today.weekday()) % 7
        if days_until_friday== 0:  # If today is Friday
            days_until_friday = 7
        
        friday = today + timedelta(days=days_until_friday)
        return friday.strftime("%Y-%m-%d")
    
    def estimate_expected_move(self, ticker: str, expiry: str) -> float:
        """Estimate expected move from implied volatility"""
        try:
            stock=yf.Ticker(ticker)
            spot=stock.history(period="1d")['Close'].iloc[-1]
            
            # Get options chain to estimate IV
            chain=stock.option_chain(expiry)
            if chain.calls.empty:
                return 0.05  # Default 5%
            
            # Find ATM options to estimate IV
            calls=chain.calls
            atm_call = calls.iloc[(calls['strike'] - spot).abs().argsort()[:1]]
            
            if not atm_call.empty:
                mid_price=(atm_call['bid'].iloc[0] + atm_call['ask'].iloc[0]) / 2
                
                # Very rough IV estimation from option price
                days_to_exp=(datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()).days
                if days_to_exp <= 0:
                    days_to_exp=1
                
                # Rough straddle approximation for expected move
                # Expected move ≈ (ATM call + ATM put) price
                puts=chain.puts
                atm_put = puts.iloc[(puts['strike'] - spot).abs().argsort()[:1]]
                
                if not atm_put.empty:
                    put_mid=(atm_put['bid'].iloc[0] + atm_put['ask'].iloc[0]) / 2
                    straddle_price=mid_price + put_mid
                    expected_move_pct = straddle_price / spot
                else:
                    expected_move_pct = (mid_price * 2) / spot  # Rough approximation
                
                return min(0.3, max(0.02, expected_move_pct))  # Cap between 2%-30%
            
        except:
            pass
        
        # Fallback based on historical volatility
        try:
            stock=yf.Ticker(ticker)
            hist=stock.history(period="30d")
            if len(hist) >= 20:
                returns=hist['Close'].pct_change().dropna()
                daily_vol=returns.std()
                
                days_to_exp=(datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()).days
                expected_move=daily_vol * math.sqrt(max(1, days_to_exp))
                return min(0.3, max(0.02, expected_move))
        except:
            pass
        
        return 0.05  # Default fallback
    
    def get_earnings_calendar(self, weeks_ahead: int=2) -> List[EarningsEvent]:
        """Get upcoming earnings events (simplified - would use real earnings API)"""
        # This is a simplified version - in practice, use Alpha Vantage, FMP, or similar API
        
        # Mock earnings data for demonstration
        mock_earnings=[
            {"ticker":"AAPL", "days_out":7, "time":"AMC", "expected_move":0.04},
            {"ticker":"GOOGL", "days_out":3, "time":"AMC", "expected_move":0.05},
            {"ticker":"MSFT", "days_out":10, "time":"AMC", "expected_move":0.03},
            {"ticker":"TSLA", "days_out":5, "time":"AMC", "expected_move":0.08},
            {"ticker":"NVDA", "days_out":14, "time":"AMC", "expected_move":0.06},
            {"ticker":"META", "days_out":8, "time":"AMC", "expected_move":0.07},
        ]
        
        events=[]
        today = date.today()
        
        for earning in mock_earnings:
            if earning["days_out"] <= weeks_ahead * 7:
                earnings_date=today + timedelta(days=earning["days_out"])
                
                try:
                    stock=yf.Ticker(earning["ticker"])
                    info=stock.info
                    company_name = info.get('shortName', earning["ticker"])
                    sector=info.get('sector', 'Unknown')
                except:
                    company_name=earning["ticker"]
                    sector = 'Unknown'
                
                event = EarningsEvent(
                    ticker=earning["ticker"],
                    company_name=company_name,
                    earnings_date=earnings_date,
                    time_of_day=earning["time"],
                    expected_move=earning["expected_move"],
                    avg_move_historical=earning["expected_move"] * 0.8,  # Estimate
                    revenue_estimate=None,
                    eps_estimate=None,
                    sector=sector
                )
                events.append(event)
        
        return sorted(events, key=lambda x: x.earnings_date)
    
    def scan_0dte_opportunities(self, account_size: float) -> List[LottoPlay]:
        """Scan for 0DTE opportunities"""
        opportunities=[]
        expiry = self.get_0dte_expiry()
        
        if not expiry:
            return opportunities
        
        print(f"🎰 Scanning for 0DTE opportunities ({expiry})...")
        
        # Focus on most liquid names for 0DTE
        liquid_tickers=["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN"]
        
        for ticker in liquid_tickers:
            try:
                stock=yf.Ticker(ticker)
                spot=stock.history(period="1d")['Close'].iloc[-1]
                
                # Check if ticker has options on this expiry
                try:
                    available_expiries=stock.options
                    if expiry not in available_expiries:
                        continue
                except:
                    continue
                
                chain = stock.option_chain(expiry)
                if chain.calls.empty and chain.puts.empty:
                    continue
                
                expected_move=self.estimate_expected_move(ticker, expiry)
                
                # Look for liquid strikes around expected move
                targets=[
                    ("call", spot * (1 + expected_move * 0.5)),  # Half expected move up
                    ("call", spot * (1 + expected_move)),       # Full expected move up  
                    ("put", spot * (1 - expected_move * 0.5)),  # Half expected move down
                    ("put", spot * (1 - expected_move)),        # Full expected move down
                ]
                
                for option_type, target_price in targets:
                    if option_type== "call" and not chain.calls.empty:
                        options_df = chain.calls
                    elif option_type == "put" and not chain.puts.empty:
                        options_df = chain.puts
                    else:
                        continue
                    
                    # Find closest strike
                    best_strike = options_df.iloc[(options_df['strike'] - target_price).abs().argsort()[:1]]
                    
                    if best_strike.empty:
                        continue
                    
                    strike=best_strike['strike'].iloc[0]
                    bid = best_strike['bid'].iloc[0]
                    ask = best_strike['ask'].iloc[0]
                    volume = best_strike.get('volume', [0]).iloc[0]
                    
                    # Liquidity filter
                    if bid <= 0.05 or ask <= 0.05 or volume < 100:
                        continue
                    
                    mid_price=(bid + ask) / 2
                    
                    # Calculate position sizing with strict limits
                    max_dollar_risk=account_size * self.max_risk_pct
                    max_contracts = int(max_dollar_risk / (mid_price * 100))
                    max_contracts=min(max_contracts, 10)  # Hard cap at 10 contracts
                    
                    if max_contracts <= 0:
                        continue
                    
                    # Calculate metrics
                    if option_type== "call":
                        breakeven = strike + mid_price
                        profit_target = breakeven * 1.5  # 50% beyond breakeven
                    else:  # put
                        breakeven = strike - mid_price
                        profit_target = breakeven * 0.67  # 33% below breakeven
                    
                    stop_loss = mid_price * 0.5  # 50% stop loss
                    
                    # Rough win probability (very rough estimate)
                    if option_type== "call":
                        distance_to_breakeven = (breakeven - spot) / spot
                        win_prob=max(0.05, 0.4 - distance_to_breakeven * 10)
                    else:
                        distance_to_breakeven=(spot - breakeven) / spot  
                        win_prob=max(0.05, 0.4 - distance_to_breakeven * 10)
                    
                    # Potential return if hit profit target
                    target_premium=mid_price * 3  # 3x return target
                    potential_return = (target_premium - mid_price) / mid_price
                    
                    lotto_play=LottoPlay(
                        ticker=ticker,
                        play_type="0dte",
                        expiry_date=expiry,
                        days_to_expiry=0,
                        strike=int(strike),
                        option_type=option_type,
                        current_premium=mid_price,
                        breakeven=breakeven,
                        current_spot=spot,
                        catalyst_event="Intraday momentum",
                        expected_move=expected_move,
                        max_position_size=max_dollar_risk,
                        max_contracts=max_contracts,
                        risk_level="extreme",
                        win_probability=win_prob,
                        potential_return=potential_return,
                        stop_loss_price=stop_loss,
                        profit_target_price=target_premium
                    )
                    
                    opportunities.append(lotto_play)
                    
            except Exception as e:
                print(f"Error scanning {ticker}: {e}")
                continue
        
        # Sort by risk-adjusted expected value
        opportunities.sort(
            key=lambda x: x.win_probability * x.potential_return,
            reverse=True
        )
        
        return opportunities[:10]  # Top 10 plays
    
    def scan_earnings_lottos(self, account_size: float) -> List[LottoPlay]:
        """Scan for earnings lotto plays"""
        opportunities=[]
        earnings_events = self.get_earnings_calendar()
        
        print("🎰 Scanning earnings lotto opportunities...")
        
        for event in earnings_events:
            try:
                stock=yf.Ticker(event.ticker)
                spot=stock.history(period="1d")['Close'].iloc[-1]
                
                # Find expiry closest to earnings (weekly or monthly)
                target_date=event.earnings_date
                best_expiry = None
                min_diff = float('inf')
                
                try:
                    available_expiries=stock.options
                    for exp_str in available_expiries:
                        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                        diff=abs((exp_date - target_date).days)
                        if diff < min_diff and exp_date >= target_date:
                            min_diff=diff
                            best_expiry = exp_str
                except:
                    continue
                
                if not best_expiry:
                    continue
                
                days_to_expiry = (datetime.strptime(best_expiry, "%Y-%m-%d").date() - date.today()).days
                chain=stock.option_chain(best_expiry)
                
                if chain.calls.empty and chain.puts.empty:
                    continue
                
                # Create straddle/strangle plays around expected move
                expected_up=spot * (1 + event.expected_move)
                expected_down=spot * (1 - event.expected_move)
                
                plays_to_check=[
                    ("call", expected_up, "Bullish earnings beat"),
                    ("put", expected_down, "Bearish earnings miss"),
                    ("call", spot * 1.02, "Minor upside surprise"),  # 2% OTM calls
                    ("put", spot * 0.98, "Minor downside surprise")   # 2% OTM puts
                ]
                
                for option_type, target_strike, catalyst in plays_to_check:
                    if option_type== "call" and not chain.calls.empty:
                        options_df = chain.calls
                    elif option_type == "put" and not chain.puts.empty:
                        options_df = chain.puts
                    else:
                        continue
                    
                    # Find best strike
                    best_option = options_df.iloc[(options_df['strike'] - target_strike).abs().argsort()[:1]]
                    
                    if best_option.empty:
                        continue
                    
                    strike=best_option['strike'].iloc[0]
                    bid = best_option['bid'].iloc[0] 
                    ask = best_option['ask'].iloc[0]
                    volume = best_option.get('volume', [0]).iloc[0]
                    
                    if bid <= 0.10 or ask <= 0.10:
                        continue
                    
                    mid_price=(bid + ask) / 2
                    
                    # Position sizing
                    max_dollar_risk=account_size * self.max_risk_pct
                    max_contracts = int(max_dollar_risk / (mid_price * 100))
                    max_contracts=min(max_contracts, 5)  # Even stricter for earnings
                    
                    if max_contracts <= 0:
                        continue
                    
                    # Calculate metrics
                    if option_type== "call":
                        breakeven = strike + mid_price
                        win_prob = 0.3 if strike < expected_up else 0.2
                    else:
                        breakeven = strike - mid_price
                        win_prob = 0.3 if strike > expected_down else 0.2
                    
                    # Earnings plays aim for 3-5x returns
                    potential_return = 4.0  # Target 5x return
                    
                    risk_level = "extreme" if days_to_expiry <= 2 else "very_high"
                    
                    lotto_play = LottoPlay(
                        ticker=event.ticker,
                        play_type="earnings",
                        expiry_date=best_expiry,
                        days_to_expiry=days_to_expiry,
                        strike=int(strike),
                        option_type=option_type,
                        current_premium=mid_price,
                        breakeven=breakeven,
                        current_spot=spot,
                        catalyst_event=f"Earnings {event.earnings_date} {event.time_of_day}",
                        expected_move=event.expected_move,
                        max_position_size=max_dollar_risk,
                        max_contracts=max_contracts,
                        risk_level=risk_level,
                        win_probability=win_prob,
                        potential_return=potential_return,
                        stop_loss_price=mid_price * 0.5,
                        profit_target_price=mid_price * 5
                    )
                    
                    opportunities.append(lotto_play)
                    
            except Exception as e:
                print(f"Error scanning earnings for {event.ticker}: {e}")
                continue
        
        # Sort by expected value
        opportunities.sort(
            key=lambda x: x.win_probability * x.potential_return,
            reverse=True
        )
        
        return opportunities[:15]
    
    def format_lotto_plays(self, plays: List[LottoPlay]) -> str:
        """Format lotto plays for display"""
        if not plays:
            return "🎰 No suitable lotto plays found."
        
        output=f"\n🎰 LOTTO PLAYS ({len(plays)} found)\n"
        output += "=" * 70 + "\n"
        output += "⚠️  EXTREME RISK - USE ONLY 0.5-1% OF ACCOUNT PER PLAY\n"
        output += "=" * 70 + "\n"
        
        for i, play in enumerate(plays, 1):
            risk_emoji={
                "extreme":"💀",
                "very_high":"🔥", 
                "high":"⚠️"
            }.get(play.risk_level, "⚠️")
            
            output += f"\n{i}. {play.ticker} {risk_emoji} {play.play_type.upper()}\n"
            output += f"   {play.strike} {play.option_type.upper()} exp {play.expiry_date}\n"
            output += f"   Catalyst: {play.catalyst_event}\n"
            output += f"   Spot: ${play.current_spot:.2f} | Strike: ${play.strike} | Premium: ${play.current_premium:.2f}\n"
            output += f"   Breakeven: ${play.breakeven:.2f} | Expected Move: {play.expected_move:.1%}\n"
            output += f"   Max Position: {play.max_contracts} contracts (${play.max_position_size:.0f})\n"
            output += f"   Win Prob: {play.win_probability:.1%} | Target Return: {play.potential_return:.1f}x\n"
            output += f"   Stop: ${play.stop_loss_price:.2f} | Target: ${play.profit_target_price:.2f}\n"
        
        output += "\n" + "=" * 70
        output += "\n💀 LOTTO PLAY RULES (MANDATORY):\n"
        output += "• MAX 1% of account per play\n"
        output += "• MAX 3 positions at once\n" 
        output += "• 50% stop loss on ALL positions\n"
        output += "• Take profits at 3-5x quickly\n"
        output += "• NO DOUBLING DOWN on losers\n"
        output += "• Track every trade for learning\n"
        output += "\n⚰️  EXPECTATION: Most trades will expire worthless\n"
        output += "    The few winners must pay for many losers\n"
        
        return output


def main():
    parser=argparse.ArgumentParser(description="0DTE/Earnings Lotto Scanner")
    parser.add_argument('command', choices=['0dte', 'earnings', 'both'],
                       help='Type of lotto scan')
    parser.add_argument('--account-size', type=float, required=True,
                       help='Account size for position sizing')
    parser.add_argument('--max-risk-pct', type=float, default=1.0,
                       help='Max risk %% per play (default 1%%)')
    parser.add_argument('--output', choices=['json', 'text'], default='text',
                       help='Output format')
    
    args=parser.parse_args()
    
    if args.account_size < 10000:
        print("⚠️  WARNING: Account size too small for safe lotto plays")
        print("   Recommended minimum: $10,000")
        print("   Consider paper trading first")
        return
    
    scanner=LottoScanner(args.max_risk_pct)
    
    all_plays=[]
    
    if args.command in ['0dte', 'both']:
        plays_0dte=scanner.scan_0dte_opportunities(args.account_size)
        all_plays.extend(plays_0dte)
    
    if args.command in ['earnings', 'both']:
        earnings_plays=scanner.scan_earnings_lottos(args.account_size)
        all_plays.extend(earnings_plays)
    
    if args.output== 'json':print(json.dumps([asdict(play) for play in all_plays], 
                        indent=2, default=str))
    else:
        print(scanner.format_lotto_plays(all_plays))


if __name__== "__main__":main()