#!/usr / bin / env python3
"""
WSB Strategy: Earnings IV Crush Protection
Avoid the #1 WSB earnings mistake-getting crushed by IV collapse
Focus on IV - resistant structures: Deep ITM options, calendar spreads, balanced hedges
"""

import argparse
import math
import json
from datetime import date, datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

try: 
    import yfinance as yf
    import pandas as pd
    import numpy as np
except ImportError as e: 
    print(f"Missing required package: {e}")
    print("Run: pip install -r wsb_requirements.txt")
    exit(1)


@dataclass
class EarningsEvent: 
    ticker: str
    company_name: str
    earnings_date: date
    earnings_time: str  # "BMO", "AMC", "Unknown"
    days_until_earnings: int
    current_price: float
    expected_move: float  # Expected % move from straddle pricing
    iv_current: float
    iv_historical_avg: float
    iv_crush_risk: str  # "low", "medium", "high", "extreme"


@dataclass
class EarningsProtectionStrategy: 
    ticker: str
    strategy_name: str
    strategy_type: str  # "deep_itm", "calendar_spread", "ratio_spread", "protective_hedge"
    earnings_date: date
    
    # Strategy specifics
    strikes: List[float]
    expiry_dates: List[str]
    option_types: List[str]  # ["call", "put", etc.]
    quantities: List[int]    # [+1, -1, +2, etc.] (+buy, -sell)
    
    # Cost and risk
    net_debit: float
    max_profit: float
    max_loss: float
    breakeven_points: List[float]
    
    # IV protection metrics
    iv_sensitivity: float  # How much strategy loses from IV crush (0 - 1)
    theta_decay: float     # Daily theta decay
    gamma_risk: float      # Gamma exposure
    
    # Profit scenarios
    profit_if_up_5pct: float
    profit_if_down_5pct: float
    profit_if_flat: float
    
    risk_level: str


class EarningsProtectionScanner: 
    def __init__(self): 
        # Focus on liquid names with reliable earnings dates
        self.earnings_candidates=[
            # Mega caps with predictable earnings
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX",
            
            # High IV earnings movers
            "AMD", "CRM", "ADBE", "ORCL", "INTC", "QCOM", "PYPL", "SQ",
            
            # Volatile but tradeable
            "PLTR", "COIN", "ROKU", "ZM", "DOCU", "SNOW", "CRWD", "ZS"
        ]
    
    def get_upcoming_earnings(self, days_ahead: int=14)->List[EarningsEvent]:
        """Get upcoming earnings events (simplified - would use earnings API in production)"""
        # Mock earnings data - in production, use Alpha Vantage, FMP, or similar
        mock_earnings = [
            {"ticker": "AAPL", "days_out": 3, "time": "AMC"},
            {"ticker": "GOOGL", "days_out": 7, "time": "AMC"}, 
            {"ticker": "MSFT", "days_out": 5, "time": "AMC"},
            {"ticker": "TSLA", "days_out": 10, "time": "AMC"},
            {"ticker": "META", "days_out": 8, "time": "AMC"},
            {"ticker": "AMD", "days_out": 12, "time": "AMC"},
            {"ticker": "CRM", "days_out": 6, "time": "AMC"},
            {"ticker": "NVDA", "days_out": 4, "time": "AMC"},
        ]
        
        events = []
        today = date.today()
        
        for earning in mock_earnings: 
            if earning["days_out"]  <=  days_ahead: 
                ticker = earning["ticker"]
                earnings_date = today + timedelta(days=earning["days_out"])
                
                try: 
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="1d")
                    if hist.empty: 
                        continue
                        
                    current_price = hist['Close'].iloc[-1]
                    
                    # Get company info
                    try: 
                        info = stock.info
                        company_name = info.get('shortName', ticker)
                    except: 
                        company_name = ticker
                    
                    # Estimate expected move and IV
                    expected_move, iv_current=self.estimate_earnings_move(ticker, earning["days_out"])
                    iv_historical = self.estimate_historical_iv(ticker)
                    
                    # Assess IV crush risk
                    iv_premium = iv_current / iv_historical if iv_historical  >  0 else 1.0
                    if iv_premium  >  2.0: 
                        crush_risk = "extreme"
                    elif iv_premium  >  1.5: 
                        crush_risk = "high"
                    elif iv_premium  >  1.2: 
                        crush_risk = "medium"
                    else: 
                        crush_risk = "low"
                    
                    event = EarningsEvent(
                        ticker=ticker,
                        company_name=company_name,
                        earnings_date=earnings_date,
                        earnings_time = earning["time"],
                        days_until_earnings = earning["days_out"],
                        current_price=current_price,
                        expected_move=expected_move,
                        iv_current=iv_current,
                        iv_historical_avg=iv_historical,
                        iv_crush_risk = crush_risk)
                    
                    events.append(event)
                    
                except Exception as e: 
                    print(f"Error processing {ticker}: {e}")
                    continue
        
        return sorted(events, key=lambda x: x.days_until_earnings)
    
    def estimate_earnings_move(self, ticker: str, days_to_earnings: int)->Tuple[float, float]: 
        """Estimate expected move and current IV from options"""
        try: 
            stock = yf.Ticker(ticker)
            
            # Find nearest expiry after earnings
            expiries = stock.options
            target_date = date.today() + timedelta(days=days_to_earnings)
            
            best_expiry = None
            for exp_str in expiries: 
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                if exp_date  >=  target_date: 
                    best_expiry = exp_str
                    break
            
            if not best_expiry: 
                return 0.06, 0.30  # Default 6% move, 30% IV
            
            # Get ATM straddle to estimate expected move
            spot = stock.history(period="1d")['Close'].iloc[-1]
            chain = stock.option_chain(best_expiry)
            
            # Find ATM call and put
            calls = chain.calls
            puts = chain.puts
            
            if calls.empty or puts.empty: 
                return 0.06, 0.30
            
            atm_call = calls.iloc[(calls['strike'] - spot).abs().argsort()[: 1]]
            atm_put = puts.iloc[(puts['strike'] - spot).abs().argsort()[: 1]]
            
            if atm_call.empty or atm_put.empty: 
                return 0.06, 0.30
            
            call_mid = (atm_call['bid'].iloc[0] + atm_call['ask'].iloc[0]) / 2
            put_mid = (atm_put['bid'].iloc[0] + atm_put['ask'].iloc[0]) / 2
            
            # Expected move ≈ straddle price
            straddle_price = call_mid + put_mid
            expected_move_pct = straddle_price / spot
            
            # Estimate IV from call price (simplified)
            call_price = call_mid
            atm_call['strike'].iloc[0]
            
            # Rough IV estimate using simplified formula
            days_to_exp = (datetime.strptime(best_expiry, "%Y-%m-%d").date() - date.today()).days
            time_factor = math.sqrt(days_to_exp / 365) if days_to_exp  >  0 else 0.1
            
            if time_factor  >  0: 
                iv_estimate = (call_price / spot) / time_factor
                iv_estimate = max(0.15, min(1.5, iv_estimate))  # Reasonable bounds
            else: 
                iv_estimate = 0.30
            
            return expected_move_pct, iv_estimate
            
        except Exception: 
            return 0.06, 0.30  # Conservative defaults
    
    def estimate_historical_iv(self, ticker: str)->float:
        """Estimate historical IV from price volatility"""
        try: 
            stock = yf.Ticker(ticker)
            hist = stock.history(period="60d")
            
            if len(hist)  <  30: 
                return 0.25
            
            returns = hist['Close'].pct_change().dropna()
            historical_vol = returns.std() * math.sqrt(252)
            
            return max(0.10, min(1.0, historical_vol))
            
        except: 
            return 0.25
    
    def create_deep_itm_strategy(self, event: EarningsEvent)->Optional[EarningsProtectionStrategy]:
        """Create deep ITM call strategy (less IV sensitive)"""
        try: 
            ticker = event.ticker
            stock = yf.Ticker(ticker)
            spot = event.current_price
            
            # Find expiry after earnings
            expiries = stock.options
            target_expiry = None
            
            for exp_str in expiries: 
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                if exp_date  >  event.earnings_date: 
                    target_expiry = exp_str
                    break
            
            if not target_expiry: 
                return None
            
            chain = stock.option_chain(target_expiry)
            calls = chain.calls
            
            if calls.empty: 
                return None
            
            # Target deep ITM call (15 - 20% ITM)
            target_strike = spot * 0.85  # 15% ITM
            deep_call = calls.iloc[(calls['strike'] - target_strike).abs().argsort()[: 1]]
            
            if deep_call.empty: 
                return None
            
            strike = deep_call['strike'].iloc[0]
            premium = (deep_call['bid'].iloc[0] + deep_call['ask'].iloc[0]) / 2
            
            if premium  <=  0: 
                return None
            
            # Calculate metrics
            intrinsic_value = max(0, spot - strike)
            time_value = premium - intrinsic_value
            breakeven = strike+premium
            
            # Profit scenarios
            profit_up_5 = (spot * 1.05) - strike-premium
            profit_down_5 = max(0, spot * 0.95 - strike) - premium
            profit_flat = spot - strike-premium
            
            # IV sensitivity (deep ITM has less time value)
            iv_sensitivity = time_value / premium if premium  >  0 else 0.0
            
            strategy = EarningsProtectionStrategy(
                ticker=ticker,
                strategy_name = f"Deep ITM Call ${strike: .0f}",
                strategy_type = "deep_itm",
                earnings_date = event.earnings_date,
                strikes = [strike],
                expiry_dates = [target_expiry],
                option_types = ["call"],
                quantities = [1],
                net_debit=premium,
                max_profit = float('inf'),  # Unlimited upside
                max_loss=premium,
                breakeven_points = [breakeven],
                iv_sensitivity=iv_sensitivity,
                theta_decay = time_value * 0.1,  # Rough estimate
                gamma_risk = 0.3,  # Lower gamma for deep ITM
                profit_if_up_5pct=profit_up_5,
                profit_if_down_5pct=profit_down_5,
                profit_if_flat=profit_flat,
                risk_level = "medium"
            )
            
            return strategy
            
        except Exception: 
            return None
    
    def create_calendar_spread_strategy(self, event: EarningsEvent)->Optional[EarningsProtectionStrategy]:
        """Create calendar spread (sell front month, buy back month)"""
        try: 
            ticker = event.ticker
            stock = yf.Ticker(ticker)
            spot = event.current_price
            
            expiries = stock.options
            
            # Find expiry before and after earnings
            front_expiry = None
            back_expiry = None
            
            for exp_str in expiries: 
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                if exp_date  <=  event.earnings_date and not front_expiry: 
                    front_expiry = exp_str
                elif exp_date  >  event.earnings_date and not back_expiry: 
                    back_expiry = exp_str
                    if front_expiry: 
                        break
            
            if not front_expiry or not back_expiry: 
                return None
            
            # Get chains for both expiries
            front_chain = stock.option_chain(front_expiry)
            back_chain = stock.option_chain(back_expiry)
            
            if front_chain.calls.empty or back_chain.calls.empty: 
                return None
            
            # ATM strikes for calendar spread
            front_calls = front_chain.calls
            back_calls = back_chain.calls
            
            # Find ATM strike in both chains
            front_atm = front_calls.iloc[(front_calls['strike'] - spot).abs().argsort()[: 1]]
            back_atm = back_calls.iloc[(back_calls['strike'] - spot).abs().argsort()[: 1]]
            
            if front_atm.empty or back_atm.empty: 
                return None
            
            strike = front_atm['strike'].iloc[0]
            front_premium = (front_atm['bid'].iloc[0] + front_atm['ask'].iloc[0]) / 2
            back_premium = (back_atm['bid'].iloc[0] + back_atm['ask'].iloc[0]) / 2
            
            net_debit = back_premium - front_premium
            
            if net_debit  <=  0: 
                return None
            
            # Calendar spread profits from time decay and IV crush
            max_profit = strike * 0.05  # Rough estimate
            
            # Profit scenarios (calendar spreads benefit from no movement)
            profit_up_5 = -net_debit * 0.3    # Lose some if moves too much
            profit_down_5 = -net_debit * 0.3  # Lose some if moves too much  
            profit_flat = max_profit * 0.7    # Best case scenario
            
            strategy = EarningsProtectionStrategy(
                ticker=ticker,
                strategy_name = f"Calendar Spread ${strike: .0f}",
                strategy_type = "calendar_spread",
                earnings_date = event.earnings_date,
                strikes = [strike, strike],
                expiry_dates = [front_expiry, back_expiry],
                option_types = ["call", "call"],
                quantities = [-1, 1],  # Sell front, buy back
                net_debit=net_debit,
                max_profit=max_profit,
                max_loss=net_debit,
                breakeven_points = [strike-net_debit, strike+net_debit],
                iv_sensitivity = 0.2,  # Lower IV sensitivity
                theta_decay = -net_debit * 0.05,  # Benefits from theta
                gamma_risk = 0.1,  # Low gamma risk
                profit_if_up_5pct=profit_up_5,
                profit_if_down_5pct=profit_down_5,
                profit_if_flat=profit_flat,
                risk_level = "low"
            )
            
            return strategy
            
        except Exception: 
            return None
    
    def create_protective_hedge_strategy(self, event: EarningsEvent)->Optional[EarningsProtectionStrategy]:
        """Create protective hedge (long call + long put with different strikes)"""
        try: 
            ticker = event.ticker
            stock = yf.Ticker(ticker)
            spot = event.current_price
            
            # Find expiry after earnings
            expiries = stock.options
            target_expiry = None
            
            for exp_str in expiries: 
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                if exp_date  >  event.earnings_date: 
                    target_expiry = exp_str
                    break
            
            if not target_expiry: 
                return None
            
            chain = stock.option_chain(target_expiry)
            calls = chain.calls
            puts = chain.puts
            
            if calls.empty or puts.empty: 
                return None
            
            # OTM call and OTM put (cheaper than ATM straddle)
            call_strike = spot * 1.05  # 5% OTM call
            put_strike = spot * 0.95   # 5% OTM put
            
            # Find closest strikes
            call_option = calls.iloc[(calls['strike'] - call_strike).abs().argsort()[: 1]]
            put_option = puts.iloc[(puts['strike'] - put_strike).abs().argsort()[: 1]]
            
            if call_option.empty or put_option.empty: 
                return None
            
            call_strike_actual = call_option['strike'].iloc[0]
            put_strike_actual = put_option['strike'].iloc[0]
            
            call_premium = (call_option['bid'].iloc[0] + call_option['ask'].iloc[0]) / 2
            put_premium = (put_option['bid'].iloc[0] + put_option['ask'].iloc[0]) / 2
            
            total_premium = call_premium + put_premium
            
            # Profit scenarios
            profit_up_5 = max(0, spot * 1.05 - call_strike_actual) - total_premium
            profit_down_5 = max(0, put_strike_actual - spot * 0.95) - total_premium
            profit_flat = -total_premium
            
            strategy = EarningsProtectionStrategy(
                ticker=ticker,
                strategy_name = f"Protective Hedge {call_strike_actual: .0f}C / {put_strike_actual: .0f}P",
                strategy_type = "protective_hedge", 
                earnings_date = event.earnings_date,
                strikes = [call_strike_actual, put_strike_actual],
                expiry_dates = [target_expiry, target_expiry],
                option_types = ["call", "put"],
                quantities = [1, 1],
                net_debit=total_premium,
                max_profit = float('inf'),  # Unlimited if big move
                max_loss=total_premium,
                breakeven_points = [call_strike_actual + total_premium, put_strike_actual - total_premium],
                iv_sensitivity = 0.6,  # Moderate IV sensitivity
                theta_decay = total_premium * 0.1,
                gamma_risk = 0.5,
                profit_if_up_5pct=profit_up_5,
                profit_if_down_5pct=profit_down_5,
                profit_if_flat=profit_flat,
                risk_level = "medium"
            )
            
            return strategy
            
        except Exception: 
            return None
    
    def scan_earnings_protection(self)->List[EarningsProtectionStrategy]: 
        """Scan for earnings protection strategies"""
        strategies = []
        events = self.get_upcoming_earnings()
        
        print(f"📊 Found {len(events)} upcoming earnings events")
        
        for event in events: 
            print(f"\n📈 {event.ticker} earnings in {event.days_until_earnings} days")
            print(f"   Current: ${event.current_price:.2f}, Expected move: ±{event.expected_move:.1%}")
            print(f"   IV Crush Risk: {event.iv_crush_risk.upper()}")
            
            # Skip if IV crush risk is low (regular options might be fine)
            if event.iv_crush_risk ==  "low": 
                print("   ✅ Low IV crush risk - regular strategies may work")
                continue
            
            # Create protection strategies
            event_strategies = []
            
            # 1. Deep ITM strategy
            deep_itm = self.create_deep_itm_strategy(event)
            if deep_itm: 
                event_strategies.append(deep_itm)
                print(f"   🛡️  Deep ITM: IV sensitivity {deep_itm.iv_sensitivity:.1%}")
            
            # 2. Calendar spread
            calendar = self.create_calendar_spread_strategy(event)
            if calendar: 
                event_strategies.append(calendar)
                print(f"   📅 Calendar: IV sensitivity {calendar.iv_sensitivity:.1%}")
            
            # 3. Protective hedge
            hedge = self.create_protective_hedge_strategy(event)
            if hedge: 
                event_strategies.append(hedge)
                print(f"   🛡️  Hedge: IV sensitivity {hedge.iv_sensitivity:.1%}")
            
            strategies.extend(event_strategies)
        
        # Sort by lowest IV sensitivity (best protection)
        strategies.sort(key=lambda x: x.iv_sensitivity)
        
        return strategies
    
    def format_strategies(self, strategies: List[EarningsProtectionStrategy])->str:
        """Format earnings protection strategies"""
        if not strategies: 
            return "📊 No earnings protection strategies found."
        
        output = f"\n🛡️  EARNINGS IV CRUSH PROTECTION STRATEGIES ({len(strategies)} found)\n"
        output += " = " * 80 + "\n"
        
        for i, strategy in enumerate(strategies, 1): 
            days_to_earnings = (strategy.earnings_date-date.today()).days
            
            output += f"\n{i}. {strategy.ticker} - {strategy.strategy_name}\n"
            output += f"   Earnings: {strategy.earnings_date} ({days_to_earnings} days) | Strategy: {strategy.strategy_type}\n"
            output += f"   Cost: ${strategy.net_debit:.2f} | Max Loss: ${strategy.max_loss:.2f}\n"
            output += f"   IV Sensitivity: {strategy.iv_sensitivity:.1%} | Risk: {strategy.risk_level.upper()}\n"
            
            output += f"   Profit Scenarios: Up 5%: ${strategy.profit_if_up_5pct:.2f} | "
            output += f"Flat: ${strategy.profit_if_flat:.2f} | Down 5%: ${strategy.profit_if_down_5pct:.2f}\n"
            
            if strategy.breakeven_points: 
                breakevens = ", ".join([f"${be: .2f}" for be in strategy.breakeven_points])
                output += f"   Break - evens: {breakevens}\n"
        
        output += "\n" + " = " * 80
        output += "\n🛡️  EARNINGS PROTECTION PRINCIPLES: \n"
        output += "• Avoid long straddles / strangles (high IV crush risk)\n"
        output += "• Deep ITM options have less time value → less IV risk + n"
        output += "• Calendar spreads benefit from IV crush on front month + n"
        output += "• Protective hedges limit downside with defined risk + n"
        output += "• Lower IV sensitivity = better protection + n"
        
        output += "\n⚠️  WSB EARNINGS WARNINGS: \n"
        output += "• Most earnings plays lose money due to IV crush + n"
        output += "• Expected moves are often overestimated + n"
        output += "• Even 'right' direction can lose if IV crushes + n"
        output += "• Consider avoiding earnings altogether + n"
        output += "• If playing, use only 1 - 2% position sizing + n"
        
        output += "\n💡 ALTERNATIVE: Wait until AFTER earnings + n"
        output += "• IV crush creates opportunities to buy cheap options + n"
        output += "• Lower IV = better risk / reward for future moves + n"
        output += "• Let others get crushed, then pick up the pieces + n"
        
        return output


def main(): 
    parser = argparse.ArgumentParser(description="Earnings IV Crush Protection Scanner")
    parser.add_argument('--days - ahead', type=int, default=14,
                       help = 'Days ahead to scan for earnings')
    parser.add_argument('--output', choices=['json', 'text'], default='text',
                       help = 'Output format')
    parser.add_argument('--max - iv - sensitivity', type=float, default=0.5,
                       help = 'Maximum IV sensitivity (0.0 - 1.0)')
    
    args = parser.parse_args()
    
    scanner = EarningsProtectionScanner()
    strategies = scanner.scan_earnings_protection()
    
    # Filter by IV sensitivity
    strategies = [s for s in strategies if s.iv_sensitivity  <=  args.max_iv_sensitivity]
    
    if args.output  ==  'json': print(json.dumps([asdict(s) for s in strategies], indent=2, default=str))
    else: 
        print(scanner.format_strategies(strategies))
    
    if strategies: 
        avg_iv_sensitivity = np.mean([s.iv_sensitivity for s in strategies])
        print(f"\n📊 Average IV sensitivity: {avg_iv_sensitivity:.1%}")
        print("💡 Lower is better for earnings protection!")
    else: 
        print("\n❌ No suitable protection strategies found")
        print("💡 Consider avoiding earnings plays entirely")


if __name__ ==  "__main__": main()