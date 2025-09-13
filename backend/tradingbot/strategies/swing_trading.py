#!/usr / bin / env python3
"""
WSB Strategy: Enhanced Breakout Swing Trading
Fast profit - taking swing trades with same-day exit discipline
Based on WSB successful swing trading patterns with ≤30 day expiries
"""

import argparse
import json
from datetime import date, datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Tuple
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
class SwingSignal: 
    ticker: str
    signal_time: datetime
    signal_type: str  # "breakout", "momentum", "reversal"
    entry_price: float
    breakout_level: float
    volume_confirmation: float  # Volume vs average
    strength_score: float  # 0 - 100 breakout strength
    target_strike: int
    target_expiry: str
    option_premium: float
    max_hold_hours: int  # WSB rule: same-day exits preferred
    profit_target_1: float  # 25% profit
    profit_target_2: float  # 50% profit  
    profit_target_3: float  # 100% profit
    stop_loss: float
    risk_level: str


@dataclass
class ActiveSwingTrade: 
    signal: SwingSignal
    entry_time: datetime
    entry_premium: float
    current_premium: float
    unrealized_pnl: float
    unrealized_pct: float
    hours_held: float
    hit_profit_target: int  # 0=none, 1=25%, 2=50%, 3 = 100%
    should_exit: bool
    exit_reason: str


class SwingTradingScanner: 
    def __init__(self): 
        # Focus on liquid, high - beta names for swing trading
        self.swing_tickers=[
            # Mega caps with options liquidity
            "SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META",
            
            # High - beta swing favorites
            "AMD", "NFLX", "CRM", "ADBE", "PYPL", "SQ", "ROKU", "ZM", "PLTR", "COIN",
            
            # Volatile sectors good for swings
            "XLF", "XLE", "XLK", "XBI", "ARKK", "TQQQ", "SOXL", "SPXL"
        ]
        
        self.active_trades: List[ActiveSwingTrade] = []
    
    def detect_breakout(self, ticker: str)->Tuple[bool, float, float]: 
        """Detect breakout above resistance with volume confirmation"""
        try: 
            stock = yf.Ticker(ticker)
            
            # Get intraday data for breakout detection
            data = stock.history(period="5d", interval="15m")
            if len(data)  <  50: 
                return False, 0.0, 0.0
            
            prices = data['Close'].values
            volumes = data['Volume'].values
            data['High'].values
            
            current_price = prices[-1]
            current_volume = volumes[-5: ].mean()  # Recent 5 periods
            avg_volume = volumes[: -10].mean()  # Historical average
            
            # Calculate resistance levels (pivot highs)
            resistance_levels = []
            for i in range(20, len(prices) - 5): 
                if (prices[i]  ==  max(prices[i - 3: i + 4]) and 
                    prices[i]  >  np.percentile(prices[: i], 80)): 
                    resistance_levels.append(prices[i])
            
            if not resistance_levels: 
                return False, 0.0, 0.0
            
            # Find nearest resistance level
            key_resistance = max(resistance_levels[-3: ]) if len(resistance_levels)  >=  3 else max(resistance_levels)
            
            # Breakout criteria: 
            # 1. Price breaks above key resistance by  > 0.5%
            # 2. Volume is  > 2x average
            # 3. Strong momentum in last few bars
            
            volume_multiple = current_volume / avg_volume if avg_volume  >  0 else 0
            breakout_strength = (current_price-key_resistance) / key_resistance
            
            recent_momentum = (prices[-1] - prices[-5]) / prices[-5]  # Last 5 bars
            
            is_breakout = (
                current_price  >  key_resistance * 1.005 and  # 0.5% above resistance
                volume_multiple  >=  2.0 and                  # 2x volume
                breakout_strength  >  0.002 and              # Clear breakout
                recent_momentum  >  0.005                     # Positive momentum
            )
            
            strength_score = min(100, (breakout_strength * 100 + volume_multiple * 10 + recent_momentum * 50))
            
            return is_breakout, key_resistance, strength_score
            
        except Exception: 
            return False, 0.0, 0.0
    
    def detect_momentum_continuation(self, ticker: str)->Tuple[bool, float]: 
        """Detect strong momentum continuation patterns"""
        try: 
            stock = yf.Ticker(ticker)
            data = stock.history(period="2d", interval="5m")
            
            if len(data)  <  30: 
                return False, 0.0
            
            prices = data['Close'].values
            volumes = data['Volume'].values
            
            # Look for accelerating momentum
            short_ma = np.mean(prices[-5: ])   # 25min MA
            medium_ma = np.mean(prices[-10: ])  # 50min MA
            long_ma = np.mean(prices[-20: ])   # 100min MA
            
            # Momentum strength
            if short_ma  >  medium_ma  >  long_ma: 
                momentum_strength = (short_ma / long_ma - 1) * 100
                
                # Volume confirmation
                recent_vol = volumes[-10: ].mean()
                earlier_vol = volumes[-30: -10].mean()
                vol_increase = recent_vol / earlier_vol if earlier_vol  >  0 else 1
                
                # Strong momentum criteria
                if momentum_strength  >  1.0 and vol_increase  >  1.3: 
                    return True, momentum_strength
            
            return False, 0.0
            
        except Exception: 
            return False, 0.0
    
    def detect_reversal_setup(self, ticker: str)->Tuple[bool, str, float]: 
        """Detect oversold bounce setups"""
        try: 
            stock = yf.Ticker(ticker)
            data = stock.history(period="3d", interval="15m")
            
            if len(data)  <  40: 
                return False, "insufficient_data", 0.0
            
            prices = data['Close'].values
            lows = data['Low'].values
            volumes = data['Volume'].values
            
            current_price = prices[-1]
            
            # Look for bounce from oversold levels
            recent_low = min(lows[-20: ])  # 20 - period low
            bounce_strength = (current_price-recent_low) / recent_low
            
            # Volume spike on bounce
            current_vol = volumes[-3: ].mean()
            avg_vol = volumes[: -10].mean()
            vol_spike = current_vol / avg_vol if avg_vol  >  0 else 1
            
            # RSI - like oversold condition (simplified)
            up_moves = sum(1 for i in range(len(prices) - 10, len(prices) - 1) if prices[i + 1]  >  prices[i])
            down_moves = 10 - up_moves
            
            if (bounce_strength  >  0.015 and  # 1.5% bounce from low
                vol_spike  >  2.0 and         # Volume spike
                down_moves  >=  7):           # Was oversold
                
                return True, "oversold_bounce", bounce_strength * 100
            
            return False, "no_setup", 0.0
            
        except Exception: 
            return False, "error", 0.0
    
    def get_optimal_expiry(self, max_days: int=30)->str:
        """Get optimal expiry (WSB rule: ≤30 days for swing trades)"""
        today = date.today()
        
        # Prefer weekly expirations for faster theta decay management
        target_days = min(max_days, 21)  # Max 3 weeks
        
        # Find next Friday
        days_to_friday = (4 - today.weekday()) % 7
        if days_to_friday ==  0:  # If today is Friday
            days_to_friday = 7
        
        # If too far out, use closer weekly
        if days_to_friday  >  target_days: 
            days_to_friday -= 7
        
        if days_to_friday  <=  0: 
            days_to_friday = 7
        
        expiry_date = today + timedelta(days=days_to_friday)
        return expiry_date.strftime("%Y-%m-%d")
    
    def calculate_option_targets(self, current_price: float, strike: int, premium: float)->Tuple[float, float, float, float]: 
        """Calculate profit targets and stop loss for swing trade"""
        # WSB swing trading targets: fast profit - taking
        profit_25 = premium * 1.25   # 25% profit - take some off
        profit_50 = premium * 1.50   # 50% profit - take more off  
        profit_100 = premium * 2.00  # 100% profit - close position
        
        stop_loss = premium * 0.70   # 30% stop loss - cut losses fast
        
        return profit_25, profit_50, profit_100, stop_loss
    
    def estimate_swing_premium(self, ticker: str, strike: int, expiry: str)->float:
        """Estimate option premium for swing trade"""
        try: 
            stock = yf.Ticker(ticker)
            
            # Try actual options chain first
            try: 
                chain = stock.option_chain(expiry)
                if not chain.calls.empty: 
                    calls = chain.calls
                    closest = calls.iloc[(calls['strike'] - strike).abs().argsort()[: 1]]
                    if not closest.empty: 
                        bid = closest['bid'].iloc[0]
                        ask = closest['ask'].iloc[0]
                        if bid  >  0 and ask  >  0: 
                            return (bid + ask) / 2
            except: 
                pass
            
            # Fallback estimate
            current_price = stock.history(period="1d")['Close'].iloc[-1]
            days_to_exp = (datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()).days
            
            # Swing trade premium estimate (higher IV assumption)
            time_premium = max(0.5, current_price * 0.08 * (days_to_exp / 21))
            
            if strike  >  current_price:  # OTM
                otm_discount = max(0.2, 1 - (strike-current_price) / current_price * 5)
                return time_premium * otm_discount
            else:  # ITM
                intrinsic = current_price-strike
                return intrinsic + time_premium * 0.5
                
        except: 
            return 2.0  # Conservative fallback
    
    def scan_swing_opportunities(self)->List[SwingSignal]: 
        """Scan for swing trading opportunities"""
        signals = []
        expiry = self.get_optimal_expiry()
        
        print(f"🎯 Scanning swing opportunities targeting {expiry}...")
        
        for ticker in self.swing_tickers: 
            try: 
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1d", interval="1m")
                
                if hist.empty: 
                    continue
                
                current_price = hist['Close'].iloc[-1]
                
                # Check for different signal types
                signals_found = []
                
                # 1. Breakout detection
                is_breakout, resistance_level, breakout_strength=self.detect_breakout(ticker)
                if is_breakout: 
                    signals_found.append(("breakout", breakout_strength, resistance_level))
                
                # 2. Momentum continuation
                is_momentum, momentum_strength=self.detect_momentum_continuation(ticker)
                if is_momentum: 
                    signals_found.append(("momentum", momentum_strength, current_price))
                
                # 3. Reversal setup
                is_reversal, reversal_type, reversal_strength=self.detect_reversal_setup(ticker)
                if is_reversal: 
                    signals_found.append(("reversal", reversal_strength, current_price))
                
                # Process signals
                for signal_type, strength, ref_level in signals_found: 
                    # Target strike selection based on signal type
                    if signal_type ==  "breakout": 
                        strike_multiplier = 1.02  # 2% OTM for breakouts
                        max_hold_hours = 6        # Breakouts can be held longer
                    elif signal_type  ==  "momentum": 
                        strike_multiplier = 1.015 # 1.5% OTM for momentum
                        max_hold_hours = 4        # Momentum fades fast
                    else:  # reversal
                        strike_multiplier = 1.025 # 2.5% OTM for reversals
                        max_hold_hours = 8        # Reversals take time
                    
                    target_strike = round(current_price * strike_multiplier)
                    premium = self.estimate_swing_premium(ticker, target_strike, expiry)
                    
                    if premium  <  0.25:  # Minimum premium threshold
                        continue
                    
                    # Calculate targets
                    profit_25, profit_50, profit_100, stop_loss=self.calculate_option_targets(
                        current_price, target_strike, premium
                    )
                    
                    # Risk assessment
                    if strength  >  80: 
                        risk_level = "low"
                    elif strength  >  60: 
                        risk_level = "medium"
                    else: 
                        risk_level = "high"
                    
                    signal = SwingSignal(
                        ticker=ticker,
                        signal_time = datetime.now(),
                        signal_type=signal_type,
                        entry_price=current_price,
                        breakout_level=ref_level,
                        volume_confirmation = 2.0,  # Simplified
                        strength_score=strength,
                        target_strike=target_strike,
                        target_expiry=expiry,
                        option_premium=premium,
                        max_hold_hours=max_hold_hours,
                        profit_target_1=profit_25,
                        profit_target_2=profit_50,
                        profit_target_3=profit_100,
                        stop_loss=stop_loss,
                        risk_level = risk_level)
                    
                    signals.append(signal)
                    print(f"  🎯 {ticker} {signal_type.upper()} - Strength: {strength:.0f}")
                    
            except Exception as e: 
                print(f"  ❌ {ticker}: Error - {e}")
                continue
        
        # Sort by strength score
        signals.sort(key=lambda x: x.strength_score, reverse=True)
        return signals
    
    def monitor_active_trades(self)->List[str]: 
        """Monitor active swing trades and generate exit signals"""
        exit_recommendations = []
        
        if not self.active_trades: 
            return exit_recommendations
        
        print("📊 Monitoring active swing trades...")
        
        for trade in self.active_trades: 
            try: 
                # Update trade status
                hours_held = (datetime.now() - trade.entry_time).total_seconds() / 3600
                trade.hours_held = hours_held
                
                # Get current option price (simplified)
                stock = yf.Ticker(trade.signal.ticker)
                current_stock_price = stock.history(period="1d", interval="1m")['Close'].iloc[-1]
                
                # Estimate current premium (simplified)
                if current_stock_price  >=  trade.signal.target_strike: 
                    # ITM - estimate intrinsic + remaining time value
                    intrinsic = current_stock_price-trade.signal.target_strike
                    time_value = trade.entry_premium * max(0.1, 1 - hours_held / 24)
                    current_premium = intrinsic + time_value
                else: 
                    # OTM - time decay
                    decay_factor = max(0.1, 1 - hours_held / (trade.signal.max_hold_hours * 2))
                    current_premium = trade.entry_premium * decay_factor
                
                trade.current_premium = current_premium
                trade.unrealized_pnl = current_premium - trade.entry_premium
                trade.unrealized_pct=(trade.unrealized_pnl / trade.entry_premium) * 100
                
                # Check exit conditions
                exit_reason = None
                
                # 1. Profit targets hit
                if current_premium  >=  trade.signal.profit_target_3: 
                    trade.hit_profit_target = 3
                    exit_reason = "100% profit target hit - CLOSE POSITION"
                elif current_premium  >=  trade.signal.profit_target_2: 
                    if trade.hit_profit_target  <  2: 
                        trade.hit_profit_target = 2
                        exit_reason = "50% profit target - consider partial exit"
                elif current_premium  >=  trade.signal.profit_target_1: 
                    if trade.hit_profit_target  <  1: 
                        trade.hit_profit_target = 1
                        exit_reason = "25% profit target - take some profits"
                
                # 2. Stop loss hit
                elif current_premium  <=  trade.signal.stop_loss: 
                    exit_reason = "STOP LOSS HIT - exit immediately"
                
                # 3. Time-based exits (WSB rule: don't hold too long)
                elif hours_held  >=  trade.signal.max_hold_hours: 
                    exit_reason = f"Max hold time ({trade.signal.max_hold_hours}h) reached"
                
                # 4. End of day exit rule
                elif datetime.now().hour  >=  15 and trade.signal.signal_type ==  "momentum": exit_reason="End of day - close momentum trades"
                
                if exit_reason: 
                    trade.should_exit = True
                    trade.exit_reason = exit_reason
                    exit_recommendations.append(
                        f"{trade.signal.ticker} {trade.signal.target_strike}C: {exit_reason} "
                        f"(P & L: {trade.unrealized_pct:+.1f}%)"
                    )
                
            except Exception as e: 
                print(f"Error monitoring {trade.signal.ticker}: {e}")
        
        return exit_recommendations
    
    def format_signals(self, signals: List[SwingSignal])->str:
        """Format swing signals for display"""
        if not signals: 
            return "🎯 No swing trading signals found at this time."
        
        output = f"\n🎯 SWING TRADING SIGNALS ({len(signals)} found)\n"
        output += " = " * 70 + "\n"
        
        for i, signal in enumerate(signals[: 10], 1):  # Top 10
            hours_display = f"{signal.max_hold_hours}h max"
            
            output += f"\n{i}. {signal.ticker} - {signal.signal_type.upper()} 🎯\n"
            output += f"   Entry: ${signal.entry_price:.2f} | Strike: ${signal.target_strike}\n"
            output += f"   Premium: ${signal.option_premium:.2f} | Expiry: {signal.target_expiry}\n"
            output += f"   Strength: {signal.strength_score:.0f}/100 | Risk: {signal.risk_level.upper()}\n"
            output += f"   Max Hold: {hours_display} | Profit Targets: 25%/50%/100%\n"
            output += f"   Stop Loss: ${signal.stop_loss:.2f} (-30%)\n"
        
        output += "\n" + " = " * 70
        output += "\n🎯 WSB SWING TRADING RULES: \n"
        output += "• FAST profit - taking: 25% → 50% → 100%\n"
        output += "• MAX 30 days expiry (prefer weeklies)\n"
        output += "• Same-day exits preferred for momentum + n"
        output += "• 30% stop loss - cut losses FAST + n"
        output += "• Don't hold overnight unless strong setup + n"
        output += "• Volume confirmation is MANDATORY + n"
        
        output += "\n⚠️  SWING TRADING WARNINGS: \n"
        output += "• Options decay fast - time is your enemy + n"
        output += "• Don't chase breakouts that already moved  > 5%\n"
        output += "• Avoid earnings weeks (IV crush risk)\n"
        output += "• Use 1 - 2% position sizing max + n"
        
        return output


def main(): 
    parser = argparse.ArgumentParser(description="WSB Enhanced Swing Trading Scanner")
    parser.add_argument('command', 
                       choices = ['scan', 'monitor', 'continuous'],
                       help = 'Command to execute')
    parser.add_argument('--max - expiry - days', type=int, default=21,
                       help = 'Maximum days to expiry (WSB rule: ≤30)')
    parser.add_argument('--output', choices=['json', 'text'], default='text',
                       help = 'Output format')
    parser.add_argument('--min - strength', type=float, default=60.0,
                       help = 'Minimum signal strength score')
    
    args = parser.parse_args()
    
    scanner = SwingTradingScanner()
    
    if args.command ==  'scan': 
        signals = scanner.scan_swing_opportunities()
        
        # Filter by minimum strength
        signals = [s for s in signals if s.strength_score  >=  args.min_strength]
        
        if args.output  ==  'json': 
            print(json.dumps([asdict(s) for s in signals], indent=2, default=str))
        else: 
            print(scanner.format_signals(signals))
    
    elif args.command ==  'monitor': 
        exit_signals = scanner.monitor_active_trades()
        
        if exit_signals: 
            print("\n🚨 SWING TRADE EXIT SIGNALS: ")
            for signal in exit_signals: 
                print(f"  • {signal}")
        else: 
            print("📊 No active trades to monitor")
    
    elif args.command ==  'continuous': 
        print("🔄 Starting continuous swing scanning (Ctrl + C to stop)...")
        try: 
            while True: 
                print(f"\n⏰ {datetime.now().strftime('%H: %M:%S')} - Scanning...")
                
                signals = scanner.scan_swing_opportunities()
                signals = [s for s in signals if s.strength_score  >=  args.min_strength]
                
                if signals: 
                    print(scanner.format_signals(signals))
                
                # Also monitor any active trades
                exit_signals = scanner.monitor_active_trades()
                if exit_signals: 
                    print("\n🚨 EXIT SIGNALS: ")
                    for signal in exit_signals: 
                        print(f"  • {signal}")
                
                print("\nNext scan in 5 minutes...")
                time.sleep(300)  # 5 minutes
                
        except KeyboardInterrupt: 
            print("\n🛑 Scanning stopped by user")


if __name__ ==  "__main__": main()