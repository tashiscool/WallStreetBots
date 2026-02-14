#!/usr / bin / env python3
"""WSB Strategy #6: Covered Calls / Wheel Strategy
Consistent income generation on volatile names with positive expectancy.
"""

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta

try:
    import numpy as np
    import pandas as pd
    import yfinance as yf
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Run: pip install -r wsb_requirements.txt")
    sys.exit(1)


@dataclass
class WheelPosition:
    ticker: str
    position_type: str  # "cash_secured_put", "covered_call", "assigned_shares"
    shares: int
    avg_cost: float  # For shares
    strike: int | None  # For options
    expiry: str | None  # For options
    premium_collected: float
    days_to_expiry: int | None
    current_price: float
    unrealized_pnl: float
    total_premium_collected: float  # Lifetime for this wheel cycle
    assignment_risk: float  # Probability of assignment
    annualized_return: float
    status: str  # "active", "assigned", "called_away", "expired"


@dataclass
class WheelCandidate:
    ticker: str
    company_name: str
    current_price: float
    iv_rank: float  # Implied volatility rank (0 - 100)
    put_strike: int  # Recommended put strike to sell
    put_expiry: str
    put_premium: float
    put_delta: float
    call_strike: int  # If assigned, call strike to sell
    call_premium: float
    call_delta: float
    wheel_annual_return: float  # Estimated annual return from wheeling
    dividend_yield: float
    liquidity_score: float
    quality_score: float  # Fundamental quality (0 - 100)
    volatility_score: float  # How good for premium collection
    risk_factors: list[str]


class WheelStrategy:
    def __init__(self, portfolio_file: str = "wheel_portfolio.json"):
        self.portfolio_file = portfolio_file
        self.positions: list[WheelPosition] = []
        self.load_portfolio()

        # Good wheel candidates - stable companies with decent volatility
        self.wheel_candidates = [
            # Blue chips with decent volatility
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "TSLA",
            "NVDA",
            "META",
            "NFLX",
            # High IV but stable business
            "AMD",
            "INTC",
            "CRM",
            "ORCL",
            "ADBE",
            "QCOM",
            "TXN",
            # Financial sector (good dividends + premium)
            "JPM",
            "BAC",
            "WFC",
            "GS",
            "MS",
            "C",
            "V",
            "MA",
            "COF",
            # REITs (high dividends)
            "O",
            "SPG",
            "PLD",
            "AMT",
            "CCI",
            "SBAC",
            # Energy (volatile but established)
            "XOM",
            "CVX",
            "COP",
            "EOG",
            "SLB",
            # Dividend aristocrats with options
            "KO",
            "PG",
            "JNJ",
            "PFE",
            "T",
            "VZ",
            "MCD",
            "WMT",
        ]

    def load_portfolio(self):
        """Load existing wheel positions."""
        if os.path.exists(self.portfolio_file):
            try:
                with open(self.portfolio_file) as f:
                    data = json.load(f)
                    self.positions = [
                        WheelPosition(**pos) for pos in data.get("positions", [])
                    ]
            except Exception as e:
                print(f"Error loading portfolio: {e}")
                self.positions = []
        else:
            self.positions = []

    def save_portfolio(self):
        """Save wheel portfolio."""
        try:
            data = {
                "last_updated": datetime.now().isoformat(),
                "positions": [asdict(pos) for pos in self.positions],
            }
            with open(self.portfolio_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving portfolio: {e}")

    def norm_cdf(self, x: float) -> float:
        """Standard normal CDF."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

    def black_scholes_put(
        self, S: float, K: float, T: float, r: float, sigma: float
    ) -> tuple[float, float]:
        """Black - Scholes put price and delta."""
        if T <= 0 or sigma <= 0:
            return max(K - S, 0), -1.0 if S < K else 0.0

        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        put_price = K * math.exp(-r * T) * self.norm_cdf(-d2) - S * self.norm_cdf(-d1)
        delta = -self.norm_cdf(-d1)

        return max(put_price, 0), delta

    def calculate_iv_rank(self, ticker: str) -> float:
        """Calculate IV rank based on historical volatility."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")

            if len(hist) < 252:
                return 50.0

            # Calculate rolling 20 - day realized volatility
            returns = hist["Close"].pct_change().dropna()
            rolling_vol = returns.rolling(20).std() * math.sqrt(252)

            if rolling_vol.empty:
                return 50.0

            current_vol = rolling_vol.iloc[-1]
            min_vol = rolling_vol.min()
            max_vol = rolling_vol.max()

            if max_vol == min_vol:
                return 50.0

            rank = (current_vol - min_vol) / (max_vol - min_vol) * 100
            return max(0, min(100, rank))

        except Exception:
            return 50.0

    def get_quality_score(self, ticker: str) -> float:
        """Assess fundamental quality (0 - 100)."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            factors = []

            # Market cap (prefer large caps for wheel)
            market_cap = info.get("marketCap", 0)
            if market_cap > 100e9:  #  >  $100B
                factors.append(90)
            elif market_cap > 10e9:  #  >  $10B
                factors.append(70)
            elif market_cap > 1e9:  #  >  $1B
                factors.append(50)
            else:
                factors.append(20)

            # Profitability
            profit_margin = info.get("profitMargins", 0)
            if profit_margin:
                factors.append(min(100, max(0, 50 + profit_margin * 200)))

            # Debt levels
            debt_to_equity = info.get("debtToEquity", 50)
            if debt_to_equity:
                debt_score = max(0, 100 - debt_to_equity * 2)
                factors.append(debt_score)

            # Revenue growth
            rev_growth = info.get("revenueGrowth", 0)
            if rev_growth:
                factors.append(min(100, max(0, 50 + rev_growth * 100)))

            # Beta (prefer moderate beta for wheel)
            beta = info.get("beta", 1.0)
            if 0.7 <= beta <= 1.3:
                factors.append(80)  # Good beta for wheel
            elif 0.5 <= beta <= 1.5:
                factors.append(60)
            else:
                factors.append(30)  # Too low or too high

            return np.mean(factors) if factors else 50.0

        except Exception:
            return 50.0

    def get_dividend_yield(self, ticker: str) -> float:
        """Get dividend yield."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info.get("dividendYield", 0.0) * 100  # Convert to percentage
        except Exception:
            return 0.0

    def calculate_liquidity_score(self, ticker: str) -> float:
        """Calculate options liquidity score (0 - 100)."""
        try:
            stock = yf.Ticker(ticker)

            # Check recent volume
            hist = stock.history(period="5d")
            if hist.empty:
                return 20.0

            avg_volume = hist["Volume"].mean()

            # Volume tiers
            if avg_volume > 10e6:  #  >  10M
                volume_score = 95
            elif avg_volume > 5e6:  #  >  5M
                volume_score = 85
            elif avg_volume > 1e6:  #  >  1M
                volume_score = 70
            elif avg_volume > 500e3:  #  >  500K
                volume_score = 50
            else:
                volume_score = 20

            # Check if options exist
            try:
                expiries = stock.options
                if len(expiries) >= 8:  # Good options coverage
                    options_score = 100
                elif len(expiries) >= 4:
                    options_score = 70
                else:
                    options_score = 40
            except Exception:
                options_score = 20

            return volume_score * 0.6 + options_score * 0.4

        except Exception:
            return 30.0

    def find_optimal_strikes(
        self, ticker: str, current_price: float, expiry: str
    ) -> tuple[int | None, int | None, float, float, float, float]:
        """Find optimal put and call strikes for wheel strategy."""
        try:
            stock = yf.Ticker(ticker)
            chain = stock.option_chain(expiry)

            if chain.puts.empty or chain.calls.empty:
                return None, None, 0.0, 0.0, 0.0, 0.0

            # Target put strike: 5 - 10% OTM (below current price)
            target_put_strike = current_price * 0.92  # 8% OTM

            # Find closest put strike
            puts = chain.puts[chain.puts["bid"] > 0.05]  # Minimum bid
            if puts.empty:
                return None, None, 0.0, 0.0, 0.0, 0.0

            best_put = puts.iloc[
                (puts["strike"] - target_put_strike).abs().argsort()[:1]
            ]
            if best_put.empty:
                return None, None, 0.0, 0.0, 0.0, 0.0

            put_strike = int(best_put["strike"].iloc[0])
            put_premium = (best_put["bid"].iloc[0] + best_put["ask"].iloc[0]) / 2
            put_delta = (
                abs(best_put.get("delta", [0.3]).iloc[0])
                if "delta" in best_put.columns
                else 0.3
            )

            # Target call strike: 5 - 10% OTM (above current price)
            target_call_strike = current_price * 1.08  # 8% OTM

            calls = chain.calls[chain.calls["bid"] > 0.05]
            if calls.empty:
                return put_strike, None, put_premium, put_delta, 0.0, 0.0

            best_call = calls.iloc[
                (calls["strike"] - target_call_strike).abs().argsort()[:1]
            ]
            if best_call.empty:
                return put_strike, None, put_premium, put_delta, 0.0, 0.0

            call_strike = int(best_call["strike"].iloc[0])
            call_premium = (best_call["bid"].iloc[0] + best_call["ask"].iloc[0]) / 2
            call_delta = (
                best_call.get("delta", [0.3]).iloc[0]
                if "delta" in best_call.columns
                else 0.3
            )

            return (
                put_strike,
                call_strike,
                put_premium,
                put_delta,
                call_premium,
                call_delta,
            )

        except Exception:
            return None, None, 0.0, 0.0, 0.0, 0.0

    def get_monthly_expiry(self) -> str:
        """Get next monthly expiry (3rd Friday)."""
        today = date.today()

        # Find 3rd Friday of current month
        year = today.year
        month = today.month

        # Find first day of month and its weekday
        first_day = date(year, month, 1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(days=14)  # Add 2 weeks

        if third_friday <= today:
            # Move to next month
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1

            first_day = date(year, month, 1)
            first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
            third_friday = first_friday + timedelta(days=14)

        return third_friday.strftime("%Y-%m-%d")

    def scan_wheel_candidates(self) -> list[WheelCandidate]:
        """Scan for good wheel candidates."""
        candidates = []
        expiry = self.get_monthly_expiry()

        print(f"🎡 Scanning wheel candidates for {expiry}...")

        for ticker in self.wheel_candidates:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1d")

                if hist.empty:
                    continue

                current_price = hist["Close"].iloc[-1]

                try:
                    company_name = stock.info.get("shortName", ticker)
                except Exception:
                    company_name = ticker

                # Calculate scores
                iv_rank = self.calculate_iv_rank(ticker)
                quality_score = self.get_quality_score(ticker)
                liquidity_score = self.calculate_liquidity_score(ticker)
                dividend_yield = self.get_dividend_yield(ticker)

                # Need decent IV for premium collection
                if iv_rank < 30:
                    continue

                # Need decent liquidity
                if liquidity_score < 50:
                    continue

                # Find optimal strikes
                (
                    put_strike,
                    call_strike,
                    put_premium,
                    put_delta,
                    call_premium,
                    call_delta,
                ) = self.find_optimal_strikes(ticker, current_price, expiry)

                if not put_strike or put_premium < 0.20:  # Minimum premium threshold
                    continue

                if not call_strike:
                    call_strike = int(current_price * 1.08)
                    call_premium = 1.0  # Estimate
                    call_delta = 0.3

                # Calculate estimated wheel returns
                days_to_expiry = (
                    datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()
                ).days

                # Annualized put premium return
                safe_dte = max(days_to_expiry, 1)
                put_annual_return = (
                    (put_premium / put_strike) * (365 / safe_dte) * 100
                )

                # If assigned, annualized call premium return
                call_annual_return = (
                    (call_premium / current_price) * (365 / safe_dte) * 100
                )

                # Conservative wheel return estimate (weighted by assignment probability)
                assignment_prob = abs(put_delta)  # Rough approximation
                wheel_annual_return = (
                    put_annual_return * (1 - assignment_prob)
                    + (put_annual_return + call_annual_return + dividend_yield)
                    * assignment_prob
                )

                volatility_score = min(100, iv_rank + 20)  # IV rank + buffer

                # Risk factors
                risk_factors = []
                if quality_score < 40:
                    risk_factors.append("Low quality score")
                if iv_rank > 80:
                    risk_factors.append("Very high IV - crash risk")
                if dividend_yield == 0 and assignment_prob > 0.5:
                    risk_factors.append("No dividend buffer if assigned")
                if liquidity_score < 70:
                    risk_factors.append("Lower liquidity")
                if put_delta > 0.4:
                    risk_factors.append("High assignment risk")

                # Only include candidates with reasonable returns
                if wheel_annual_return >= 8.0:  # Minimum 8% annual return
                    candidate = WheelCandidate(
                        ticker=ticker,
                        company_name=company_name,
                        current_price=current_price,
                        iv_rank=iv_rank,
                        put_strike=put_strike,
                        put_expiry=expiry,
                        put_premium=put_premium,
                        put_delta=put_delta,
                        call_strike=call_strike,
                        call_premium=call_premium,
                        call_delta=call_delta,
                        wheel_annual_return=wheel_annual_return,
                        dividend_yield=dividend_yield,
                        liquidity_score=liquidity_score,
                        quality_score=quality_score,
                        volatility_score=volatility_score,
                        risk_factors=risk_factors,
                    )

                    candidates.append(candidate)
                    print(f"  ✅ {ticker}: {wheel_annual_return:.1f}% annual return")

            except Exception as e:
                print(f"  ❌ {ticker}: Error - {e}")
                continue

        # Sort by risk - adjusted return
        candidates.sort(
            key=lambda x: x.wheel_annual_return
            * (x.quality_score / 100)
            * (x.liquidity_score / 100),
            reverse=True,
        )

        return candidates

    def update_positions(self):
        """Update all wheel positions."""
        print("📊 Updating wheel positions...")

        for pos in self.positions:
            try:
                stock = yf.Ticker(pos.ticker)
                current_price = stock.history(period="1d")["Close"].iloc[-1]
                pos.current_price = current_price

                if (
                    pos.position_type == "assigned_shares"
                ):  # Calculate unrealized P & L on shares
                    pos.unrealized_pnl = (current_price - pos.avg_cost) * pos.shares

                elif pos.strike and pos.expiry:
                    # Update days to expiry
                    pos.days_to_expiry = (
                        datetime.strptime(pos.expiry, "%Y-%m-%d").date() - date.today()
                    ).days

                    # Estimate assignment risk
                    if pos.position_type == "cash_secured_put":
                        pos.assignment_risk = min(
                            1.0,
                            max(0, (pos.strike - current_price) / current_price),
                        )
                    else:  # covered call
                        pos.assignment_risk = min(
                            1.0,
                            max(0, (current_price - pos.strike) / pos.strike),
                        )

                # Calculate annualized return
                if pos.days_to_expiry and pos.days_to_expiry > 0:
                    if pos.position_type == "cash_secured_put":
                        elapsed_days = max(30 - pos.days_to_expiry, 1)
                        pos.annualized_return = (
                            (pos.premium_collected / pos.strike)
                            * (365 / elapsed_days)
                            * 100
                        )
                    else:
                        pos.annualized_return = (
                            pos.total_premium_collected / (pos.avg_cost * pos.shares)
                        ) * 100

                # TODO: State transition handling between wheel phases.
                # When a cash_secured_put expires ITM (assignment_risk ~1.0),
                # transition to assigned_shares with avg_cost = strike - premium.
                # When assigned_shares exist without an active covered_call,
                # prompt to sell a covered_call above cost basis.
                # When a covered_call expires ITM (call-away risk ~1.0),
                # transition back to cash (close the wheel cycle).

            except Exception as e:
                print(f"Error updating {pos.ticker}: {e}")

        self.save_portfolio()

    def format_candidates(
        self, candidates: list[WheelCandidate], limit: int = 15
    ) -> str:
        """Format wheel candidates for display."""
        if not candidates:
            return "🔍 No suitable wheel candidates found."

        output = f"\n🎡 TOP WHEEL CANDIDATES ({min(limit, len(candidates))} shown)\n"
        output += " = " * 80 + "\n"

        for i, cand in enumerate(candidates[:limit], 1):
            assignment_risk = f"{abs(cand.put_delta): .0%}"

            output += f"\n{i}. {cand.ticker} - {cand.company_name}\n"
            output += (
                f"   Current: ${cand.current_price:.2f} | IV Rank: {cand.iv_rank:.0f}\n"
            )
            output += f"   SELL ${cand.put_strike} PUT: ${cand.put_premium:.2f} premium (Δ{cand.put_delta: .2f})\n"
            output += f"   IF ASSIGNED, SELL ${cand.call_strike} CALL: ${cand.call_premium:.2f} premium + n"
            output += f"   Estimated Annual Return: {cand.wheel_annual_return:.1f}%\n"
            output += f"   Dividend Yield: {cand.dividend_yield:.1f}% | Assignment Risk: {assignment_risk}\n"
            output += f"   Quality: {cand.quality_score:.0f} | Liquidity: {cand.liquidity_score:.0f}\n"

            if cand.risk_factors:
                output += f"   ⚠️  Risks: {', '.join(cand.risk_factors)}\n"

        output += "\n💡 WHEEL STRATEGY PROCESS: \n"
        output += "1. Sell cash - secured puts on quality names + n"
        output += "2. If assigned → own shares at discount + n"
        output += "3. Sell covered calls above your cost basis + n"
        output += "4. If called away → profit + start over + n"
        output += "5. Collect dividends while holding shares + n"

        output += "\n✅ WHEEL ADVANTAGES: \n"
        output += "• Positive expected value over time+n"
        output += "• Lower risk than naked options + n"
        output += "• Generates income in sideways markets + n"
        output += "• Forces buying low, selling high + n"

        return output

    def format_portfolio(self) -> str:
        """Format current wheel portfolio."""
        if not self.positions:
            return "🎡 No wheel positions in portfolio."

        self.update_positions()

        # Separate by position type
        puts = [p for p in self.positions if p.position_type == "cash_secured_put"]
        calls = [p for p in self.positions if p.position_type == "covered_call"]
        shares = [p for p in self.positions if p.position_type == "assigned_shares"]

        output = "\n🎡 WHEEL PORTFOLIO SUMMARY + n"
        output += " = " * 60 + "\n"

        total_premium = sum(p.total_premium_collected for p in self.positions)
        total_unrealized = sum(p.unrealized_pnl for p in self.positions)

        output += f"Total Premium Collected: ${total_premium:,.0f}\n"
        output += f"Unrealized P & L: ${total_unrealized:,.0f}\n"
        output += f"Active Positions: {len(self.positions)}\n + n"

        if puts:
            output += "CASH - SECURED PUTS: \n"
            output += "-" * 40 + "\n"
            for put in puts:
                days_left = put.days_to_expiry or 0
                risk_indicator = (
                    "🟥"
                    if put.assignment_risk > 0.3
                    else "🟨"
                    if put.assignment_risk > 0.1
                    else "🟩"
                )

                output += f"{put.ticker} ${put.strike} PUT exp {put.expiry} {risk_indicator}\n"
                output += (
                    f"  Premium: ${put.premium_collected:.2f} | {days_left}d left + n"
                )
                output += f"  Assignment risk: {put.assignment_risk:.1%}\n + n"

        if shares:
            output += "ASSIGNED SHARES: \n"
            output += "-" * 40 + "\n"
            for share_pos in shares:
                pnl_indicator = "🟩" if share_pos.unrealized_pnl > 0 else "🟥"

                output += f"{share_pos.ticker}: {share_pos.shares} shares @ ${share_pos.avg_cost: .2f} {pnl_indicator}\n"
                output += f"  Current: ${share_pos.current_price:.2f} | P & L: ${share_pos.unrealized_pnl:.0f}\n"
                output += (
                    f"  Total premium: ${share_pos.total_premium_collected:.2f}\n + n"
                )

        if calls:
            output += "COVERED CALLS: \n"
            output += "-" * 40 + "\n"
            for call in calls:
                days_left = call.days_to_expiry or 0
                risk_indicator = (
                    "🟥"
                    if call.assignment_risk > 0.3
                    else "🟨"
                    if call.assignment_risk > 0.1
                    else "🟩"
                )

                output += f"{call.ticker} ${call.strike} CALL exp {call.expiry} {risk_indicator}\n"
                output += (
                    f"  Premium: ${call.premium_collected:.2f} | {days_left}d left + n"
                )
                output += f"  Call - away risk: {call.assignment_risk:.1%}\n + n"

        return output


def main():
    parser = argparse.ArgumentParser(description="Wheel Strategy Scanner")
    parser.add_argument(
        "command", choices=["scan", "portfolio", "update"], help="Command to execute"
    )
    parser.add_argument(
        "--output", choices=["json", "text"], default="text", help="Output format"
    )
    parser.add_argument("--limit", type=int, default=15, help="Maximum results to show")
    parser.add_argument(
        "--min - return", type=float, default=8.0, help="Minimum annual return %%"
    )
    parser.add_argument("--save-csv", type=str, help="Save results to CSV file")

    args = parser.parse_args()

    wheel = WheelStrategy()

    if args.command == "scan":
        candidates = wheel.scan_wheel_candidates()

        # Filter by minimum return
        candidates = [c for c in candidates if c.wheel_annual_return >= args.min_return]

        if args.save_csv:
            with open(args.save_csv, "w", newline="") as csvfile:
                if candidates:
                    fieldnames = candidates[0].__dict__.keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for cand in candidates:
                        writer.writerow(asdict(cand))
            print(f"💾 Saved {len(candidates)} candidates to {args.save_csv}")

        if args.output == "json":
            print(
                json.dumps(
                    [asdict(c) for c in candidates[: args.limit]], indent=2, default=str
                )
            )
        else:
            print(wheel.format_candidates(candidates, args.limit))

    elif args.command == "portfolio":
        print(wheel.format_portfolio())

    elif args.command == "update":
        wheel.update_positions()
        print("✅ Wheel portfolio updated successfully")


if __name__ == "__main__":
    main()
