#!/usr / bin / env python3
"""WSB Strategy #4: LEAPS Secular Winners Tracking System
Long - term positions on secular growth trends with systematic profit - taking.
"""

import argparse
import csv
import json
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
class SecularTrend:
    theme: str
    description: str
    tickers: list[str]
    growth_drivers: list[str]
    time_horizon: str  # "3 - 5 years", "5 - 10 years", etc.


@dataclass
class LEAPSPosition:
    ticker: str
    theme: str
    entry_date: date
    expiry_date: str
    strike: int
    entry_premium: float
    current_premium: float
    spot_at_entry: float
    current_spot: float
    contracts: int
    cost_basis: float
    current_value: float
    unrealized_pnl: float
    unrealized_pct: float
    days_held: int
    days_to_expiry: int
    delta: float
    profit_target_hit: bool
    stop_loss_hit: bool
    scale_out_level: int  # 0=none, 1=25%, 2=50%, 3 = 75%


@dataclass
class MovingAverageCross:
    cross_type: str  # "golden_cross", "death_cross", "neutral"
    cross_date: date | None
    days_since_cross: int | None
    sma_50: float
    sma_200: float
    price_above_50sma: bool
    price_above_200sma: bool
    cross_strength: float  # 0 - 100, strength of the cross signal
    trend_direction: str  # "bullish", "bearish", "sideways"


@dataclass
class LEAPSCandidate:
    ticker: str
    company_name: str
    theme: str
    current_price: float
    trend_score: float  # 0 - 100 multi - factor score
    financial_score: float
    momentum_score: float
    valuation_score: float
    composite_score: float
    expiry_date: str
    recommended_strike: int
    premium_estimate: float
    break_even: float
    target_return_1y: float
    target_return_3y: float
    risk_factors: list[str]
    # New golden / death cross fields
    ma_cross_signal: MovingAverageCross
    entry_timing_score: float  # 0 - 100, best time to enter based on MA cross
    exit_timing_score: float  # 0 - 100, time to consider exit


class LEAPSTracker:
    def __init__(self, portfolio_file: str = "leaps_portfolio.json"):
        self.portfolio_file = portfolio_file
        self.positions: list[LEAPSPosition] = []
        self.load_portfolio()

        # Secular growth themes
        self.secular_themes = {
            "ai_revolution": SecularTrend(
                theme="AI Revolution",
                description="Artificial intelligence transforming industries",
                tickers=["NVDA", "AMD", "GOOGL", "MSFT", "META", "ORCL", "CRM", "SNOW"],
                growth_drivers=["GPU compute", "Cloud AI", "Enterprise adoption", "Consumer AI"],
                time_horizon="5 - 10 years",
            ),
            "cloud_transformation": SecularTrend(
                theme="Cloud Transformation",
                description="Enterprise digital transformation",
                tickers=["MSFT", "AMZN", "GOOGL", "CRM", "SNOW", "DDOG", "NET", "OKTA"],
                growth_drivers=[
                    "Remote work",
                    "Digital transformation",
                    "Data analytics",
                    "Security",
                ],
                time_horizon="3 - 7 years",
            ),
            "electric_mobility": SecularTrend(
                theme="Electric Mobility",
                description="Transportation electrification",
                tickers=["TSLA", "RIVN", "LCID", "NIO", "XPEV", "BYD", "GM", "F"],
                growth_drivers=[
                    "Battery tech",
                    "Charging infrastructure",
                    "Regulation",
                    "Cost parity",
                ],
                time_horizon="5 - 15 years",
            ),
            "fintech_disruption": SecularTrend(
                theme="Fintech Disruption",
                description="Financial services digitization",
                tickers=["SQ", "PYPL", "COIN", "HOOD", "SOFI", "AFRM", "V", "MA"],
                growth_drivers=[
                    "Digital payments",
                    "Crypto adoption",
                    "Banking disruption",
                    "Global expansion",
                ],
                time_horizon="3 - 10 years",
            ),
            "cybersecurity": SecularTrend(
                theme="Cybersecurity",
                description="Digital security imperative",
                tickers=["CRWD", "ZS", "PANW", "OKTA", "NET", "S", "FTNT", "RPD"],
                growth_drivers=[
                    "Remote work security",
                    "Cloud security",
                    "Compliance",
                    "Threat landscape",
                ],
                time_horizon="5 - 10 years",
            ),
            "genomics_biotech": SecularTrend(
                theme="Genomics & Biotech",
                description="Precision medicine revolution",
                tickers=["ILMN", "NVTA", "PACB", "ARKG", "CRSP", "EDIT", "NTLA", "BEAM"],
                growth_drivers=[
                    "Gene therapy",
                    "Personalized medicine",
                    "Aging population",
                    "Technology costs",
                ],
                time_horizon="10 - 20 years",
            ),
        }

    def load_portfolio(self):
        """Load existing LEAPS portfolio."""
        if os.path.exists(self.portfolio_file):
            try:
                with open(self.portfolio_file) as f:
                    data = json.load(f)
                    self.positions = [LEAPSPosition(**pos) for pos in data.get("positions", [])]
            except Exception as e:
                print(f"Error loading portfolio: {e}")
                self.positions = []
        else:
            self.positions = []

    def save_portfolio(self):
        """Save LEAPS portfolio."""
        try:
            data = {
                "last_updated": datetime.now().isoformat(),
                "positions": [asdict(pos) for pos in self.positions],
            }
            with open(self.portfolio_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving portfolio: {e}")

    def analyze_moving_average_cross(self, ticker: str) -> MovingAverageCross:
        """Analyze golden cross / death cross signals."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")

            if len(hist) < 250:
                return MovingAverageCross(
                    cross_type="neutral",
                    cross_date=None,
                    days_since_cross=None,
                    sma_50=0.0,
                    sma_200=0.0,
                    price_above_50sma=False,
                    price_above_200sma=False,
                    cross_strength=0.0,
                    trend_direction="sideways",
                )

            prices = hist["Close"].values
            current_price = prices[-1]

            # Calculate moving averages
            sma_50 = np.mean(prices[-50:])
            sma_200 = np.mean(prices[-200:])

            # Check current position relative to MAs
            price_above_50 = current_price > sma_50
            price_above_200 = current_price > sma_200

            # Calculate historical 50 and 200 SMAs to find crosses
            sma_50_series = np.array(
                [np.mean(prices[max(0, i - 49) : i + 1]) for i in range(49, len(prices))]
            )
            sma_200_series = np.array(
                [np.mean(prices[max(0, i - 199) : i + 1]) for i in range(199, len(prices))]
            )

            if len(sma_50_series) < 50 or len(sma_200_series) < 50:
                return MovingAverageCross(
                    cross_type="neutral",
                    cross_date=None,
                    days_since_cross=None,
                    sma_50=sma_50,
                    sma_200=sma_200,
                    price_above_50sma=price_above_50,
                    price_above_200sma=price_above_200,
                    cross_strength=0.0,
                    trend_direction="sideways",
                )

            # Find most recent cross
            cross_type = "neutral"
            cross_date = None
            days_since_cross = None
            cross_strength = 0.0

            # Look for crosses in the last 120 days
            lookback_days = min(120, len(sma_50_series) - 1)
            recent_50 = sma_50_series[-lookback_days:]
            recent_200 = sma_200_series[-lookback_days:]

            # Find crossovers
            for i in range(1, len(recent_50)):
                prev_50 = recent_50[i - 1]
                prev_200 = recent_200[i - 1]
                curr_50 = recent_50[i]
                curr_200 = recent_200[i]

                # Golden cross: 50 SMA crosses above 200 SMA
                if prev_50 <= prev_200 and curr_50 > curr_200:
                    cross_type = "golden_cross"
                    days_ago = len(recent_50) - i - 1
                    cross_date = date.today() - timedelta(days=days_ago)
                    days_since_cross = days_ago

                    # Calculate strength based on separation and volume
                    separation = abs(curr_50 - curr_200) / curr_200
                    cross_strength = min(100, separation * 1000)  # Scale to 0 - 100

                # Death cross: 50 SMA crosses below 200 SMA
                elif prev_50 >= prev_200 and curr_50 < curr_200:
                    cross_type = "death_cross"
                    days_ago = len(recent_50) - i - 1
                    cross_date = date.today() - timedelta(days=days_ago)
                    days_since_cross = days_ago

                    # Calculate strength
                    separation = abs(curr_50 - curr_200) / curr_200
                    cross_strength = min(100, separation * 1000)

            # Determine trend direction
            if sma_50 > sma_200 and price_above_50 and price_above_200:
                trend_direction = "bullish"
            elif sma_50 < sma_200 and not price_above_50 and not price_above_200:
                trend_direction = "bearish"
            else:
                trend_direction = "sideways"

            return MovingAverageCross(
                cross_type=cross_type,
                cross_date=cross_date,
                days_since_cross=days_since_cross,
                sma_50=sma_50,
                sma_200=sma_200,
                price_above_50sma=price_above_50,
                price_above_200sma=price_above_200,
                cross_strength=cross_strength,
                trend_direction=trend_direction,
            )

        except Exception:
            return MovingAverageCross(
                cross_type="neutral",
                cross_date=None,
                days_since_cross=None,
                sma_50=0.0,
                sma_200=0.0,
                price_above_50sma=False,
                price_above_200sma=False,
                cross_strength=0.0,
                trend_direction="sideways",
            )

    def calculate_entry_exit_timing_scores(
        self, ma_cross: MovingAverageCross, current_price: float
    ) -> tuple[float, float]:
        """Calculate entry and exit timing scores based on MA cross analysis."""
        entry_score = 50.0  # Default neutral
        exit_score = 50.0  # Default neutral

        if ma_cross.cross_type == "golden_cross":  # Golden cross scenarios
            if ma_cross.days_since_cross is not None:
                if ma_cross.days_since_cross <= 30:
                    # Recent golden cross - good entry timing
                    entry_score = 85.0 + ma_cross.cross_strength * 0.15
                    exit_score = 20.0  # Don't exit on golden cross
                elif ma_cross.days_since_cross <= 60:
                    # Still good entry window
                    entry_score = 75.0 + ma_cross.cross_strength * 0.10
                    exit_score = 25.0
                else:
                    # Golden cross getting old
                    entry_score = 60.0
                    exit_score = 40.0

        elif ma_cross.cross_type == "death_cross":  # Death cross scenarios
            if ma_cross.days_since_cross is not None:
                if ma_cross.days_since_cross <= 15:
                    # Recent death cross - poor entry timing
                    entry_score = 15.0
                    exit_score = 90.0  # Strong exit signal
                elif ma_cross.days_since_cross <= 45:
                    # Avoid entries during death cross period
                    entry_score = 25.0
                    exit_score = 75.0
                else:
                    # Death cross effects may be waning
                    entry_score = 40.0
                    exit_score = 60.0

        # Neutral / sideways - use MA position for guidance
        elif ma_cross.price_above_50sma and ma_cross.price_above_200sma:
            if ma_cross.sma_50 > ma_cross.sma_200:
                # Bullish setup without recent cross
                entry_score = 65.0
                exit_score = 35.0
            else:
                # Mixed signals
                entry_score = 55.0
                exit_score = 45.0
        else:
            # Price below key MAs
            entry_score = 35.0
            exit_score = 65.0

        # Adjust for trend strength
        if ma_cross.trend_direction == "bullish":
            entry_score = min(95.0, entry_score + 5.0)
            exit_score = max(5.0, exit_score - 5.0)
        elif ma_cross.trend_direction == "bearish":
            entry_score = max(5.0, entry_score - 10.0)
            exit_score = min(95.0, exit_score + 10.0)

        return entry_score, exit_score

    def calculate_trend_score(self, ticker: str) -> tuple[float, float, float, float]:
        """Calculate multi - factor trend score (0 - 100)."""
        try:
            stock = yf.Ticker(ticker)

            # Get historical data
            hist = stock.history(period="2y")
            if len(hist) < 250:
                return 50.0, 50.0, 50.0, 50.0

            prices = hist["Close"].values
            hist["Volume"].values
            current = prices[-1]

            # 1. Momentum Score (0 - 100)
            returns = {
                "1m": (current / prices[-21] - 1) * 100 if len(prices) > 21 else 0,
                "3m": (current / prices[-63] - 1) * 100 if len(prices) > 63 else 0,
                "6m": (current / prices[-126] - 1) * 100 if len(prices) > 126 else 0,
                "1y": (current / prices[-252] - 1) * 100 if len(prices) > 252 else 0,
                "2y": (current / prices[0] - 1) * 100,
            }

            # Weight recent performance more heavily
            momentum_raw = (
                returns["1m"] * 0.1
                + returns["3m"] * 0.2
                + returns["6m"] * 0.3
                + returns["1y"] * 0.4
            )
            momentum_score = max(0, min(100, 50 + momentum_raw))

            # 2. Trend Consistency Score
            # Calculate moving averages
            sma_20 = np.mean(prices[-20:])
            sma_50 = np.mean(prices[-50:])
            sma_200 = np.mean(prices[-200:])

            trend_alignment = 0
            if current > sma_20 > sma_50 > sma_200:
                trend_alignment = 100
            elif current > sma_20 > sma_50:
                trend_alignment = 75
            elif current > sma_20:
                trend_alignment = 50
            else:
                trend_alignment = 25

            # 3. Financial Strength Score
            financial_score = 50.0  # Default
            try:
                info = stock.info

                financial_factors = []

                # Revenue growth
                rev_growth = info.get("revenueGrowth", 0)
                if rev_growth:
                    financial_factors.append(min(100, max(0, 50 + rev_growth * 100)))

                # Profit margins
                profit_margin = info.get("profitMargins", 0)
                if profit_margin:
                    financial_factors.append(min(100, max(0, 50 + profit_margin * 200)))

                # Return on equity
                roe = info.get("returnOnEquity", 0)
                if roe:
                    financial_factors.append(min(100, max(0, 50 + roe * 300)))

                # Debt to equity
                debt_to_equity = info.get("debtToEquity", 50)
                if debt_to_equity:
                    debt_score = max(0, 100 - debt_to_equity)
                    financial_factors.append(debt_score)

                if financial_factors:
                    financial_score = np.mean(financial_factors)

            except:
                pass

            # 4. Valuation Score (inverted - lower valuations get higher scores)
            valuation_score = 50.0
            try:
                info = stock.info
                pe_ratio = info.get("forwardPE", info.get("trailingPE", 25))

                if pe_ratio and pe_ratio > 0:
                    if pe_ratio < 15:
                        valuation_score = 90
                    elif pe_ratio < 25:
                        valuation_score = 70
                    elif pe_ratio < 40:
                        valuation_score = 50
                    elif pe_ratio < 60:
                        valuation_score = 30
                    else:
                        valuation_score = 10

            except:
                pass

            return momentum_score, trend_alignment, financial_score, valuation_score

        except Exception:
            return 50.0, 50.0, 50.0, 50.0

    def get_leaps_expiries(self, ticker: str) -> list[str]:
        """Get LEAPS expiries (12+ months out)."""
        try:
            stock = yf.Ticker(ticker)
            expiries = stock.options

            leaps_expiries = []
            today = date.today()

            for exp_str in expiries:
                try:
                    exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                    days_out = (exp_date - today).days

                    # LEAPS: 12+ months, prefer January expiries
                    if days_out >= 365:
                        leaps_expiries.append(exp_str)

                except:
                    continue

            return sorted(leaps_expiries)[:3]  # Top 3 LEAPS dates

        except:
            return []

    def estimate_leaps_premium(self, ticker: str, strike: int, expiry: str) -> float:
        """Estimate LEAPS premium."""
        try:
            stock = yf.Ticker(ticker)

            # Try to get actual options data
            try:
                chain = stock.option_chain(expiry)
                if not chain.calls.empty:
                    calls = chain.calls
                    closest_strike = calls.iloc[(calls["strike"] - strike).abs().argsort()[:1]]
                    if not closest_strike.empty:
                        bid = closest_strike["bid"].iloc[0]
                        ask = closest_strike["ask"].iloc[0]
                        if bid > 0 and ask > 0:
                            return (bid + ask) / 2
            except:
                pass

            # Fallback: rough estimate
            current_price = stock.history(period="1d")["Close"].iloc[-1]
            days_to_exp = (datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()).days

            # Rough LEAPS pricing model
            time_value = max(5.0, current_price * 0.15 * (days_to_exp / 365))

            if strike > current_price:  # OTM
                otm_amount = strike - current_price
                otm_discount = max(0.3, 1.0 - (otm_amount / current_price))
                return time_value * otm_discount
            else:  # ITM
                intrinsic = current_price - strike
                return intrinsic + time_value * 0.4

        except:
            return 10.0  # Default estimate

    def scan_secular_winners(self) -> list[LEAPSCandidate]:
        """Scan all themes for LEAPS candidates."""
        candidates = []

        print("🔍 Scanning secular growth themes for LEAPS opportunities...")

        for _theme_key, theme in self.secular_themes.items():
            print(f"\n📈 Analyzing {theme.theme}...")

            for ticker in theme.tickers:
                try:
                    stock = yf.Ticker(ticker)

                    # Get current price and company info
                    hist = stock.history(period="1d")
                    if hist.empty:
                        continue

                    current_price = hist["Close"].iloc[-1]

                    try:
                        company_name = stock.info.get("shortName", ticker)
                    except:
                        company_name = ticker

                    # Calculate scores
                    momentum_score, trend_score, financial_score, valuation_score = (
                        self.calculate_trend_score(ticker)
                    )

                    # Analyze golden / death cross signals
                    ma_cross_signal = self.analyze_moving_average_cross(ticker)
                    entry_timing_score, exit_timing_score = self.calculate_entry_exit_timing_scores(
                        ma_cross_signal, current_price
                    )

                    # Enhanced composite score including timing
                    composite_score = (
                        trend_score * 0.25
                        + momentum_score * 0.20
                        + financial_score * 0.20
                        + valuation_score * 0.15
                        + entry_timing_score * 0.20  # Weight entry timing significantly
                    )

                    # Get LEAPS expiries
                    leaps_expiries = self.get_leaps_expiries(ticker)
                    if not leaps_expiries:
                        continue

                    # Target strike: 10 - 20% OTM for growth names
                    target_strike = round(current_price * 1.15)

                    # Use nearest LEAPS expiry
                    expiry = leaps_expiries[0]
                    premium = self.estimate_leaps_premium(ticker, target_strike, expiry)
                    breakeven = target_strike + premium

                    # Return targets
                    target_1y = (
                        (target_strike * 1.3) / current_price - 1
                    ) * 100  # 30% above strike
                    target_3y = (
                        (target_strike * 2.0) / current_price - 1
                    ) * 100  # 100% above strike

                    # Risk factors including timing concerns
                    risk_factors = []
                    if valuation_score < 30:
                        risk_factors.append("High valuation")
                    if momentum_score < 40:
                        risk_factors.append("Weak momentum")
                    if financial_score < 40:
                        risk_factors.append("Financial concerns")
                    if premium > current_price * 0.25:
                        risk_factors.append("High premium cost")
                    if (
                        ma_cross_signal.cross_type == "death_cross"
                        and ma_cross_signal.days_since_cross
                        and ma_cross_signal.days_since_cross < 30
                    ):
                        risk_factors.append("Recent death cross")
                    if entry_timing_score < 40:
                        risk_factors.append("Poor entry timing")
                    if exit_timing_score > 70:
                        risk_factors.append("Exit signal active")

                    # Only include strong candidates
                    if composite_score >= 60:
                        candidate = LEAPSCandidate(
                            ticker=ticker,
                            company_name=company_name,
                            theme=theme.theme,
                            current_price=current_price,
                            trend_score=trend_score,
                            financial_score=financial_score,
                            momentum_score=momentum_score,
                            valuation_score=valuation_score,
                            composite_score=composite_score,
                            expiry_date=expiry,
                            recommended_strike=target_strike,
                            premium_estimate=premium,
                            break_even=breakeven,
                            target_return_1y=target_1y,
                            target_return_3y=target_3y,
                            risk_factors=risk_factors,
                            ma_cross_signal=ma_cross_signal,
                            entry_timing_score=entry_timing_score,
                            exit_timing_score=exit_timing_score,
                        )

                        candidates.append(candidate)
                        print(f"  ✅ {ticker}: Score {composite_score:.0f}")

                except Exception as e:
                    print(f"  ❌ {ticker}: Error - {e}")
                    continue

        # Sort by composite score
        candidates.sort(key=lambda x: x.composite_score, reverse=True)
        return candidates

    def update_positions(self):
        """Update all LEAPS positions with current data."""
        print("📊 Updating LEAPS positions...")

        for pos in self.positions:
            try:
                stock = yf.Ticker(pos.ticker)
                current_price = stock.history(period="1d")["Close"].iloc[-1]

                # Try to get current option price
                try:
                    chain = stock.option_chain(pos.expiry_date)
                    calls = chain.calls
                    matching_strike = calls[calls["strike"] == pos.strike]

                    if not matching_strike.empty:
                        bid = matching_strike["bid"].iloc[0]
                        ask = matching_strike["ask"].iloc[0]
                        current_premium = (
                            (bid + ask) / 2 if bid > 0 and ask > 0 else pos.current_premium
                        )
                    else:
                        # Estimate if no exact match
                        current_premium = max(0, current_price - pos.strike) + 5.0  # Rough estimate

                except:
                    # Fallback estimate
                    intrinsic = max(0, current_price - pos.strike)
                    time_value = max(1.0, pos.current_premium * 0.8)  # Decay estimate
                    current_premium = intrinsic + time_value

                # Update position values
                pos.current_spot = current_price
                pos.current_premium = current_premium
                pos.current_value = pos.contracts * current_premium * 100
                pos.unrealized_pnl = pos.current_value - pos.cost_basis
                pos.unrealized_pct = (pos.unrealized_pnl / pos.cost_basis) * 100

                pos.days_held = (date.today() - pos.entry_date).days
                pos.days_to_expiry = (
                    datetime.strptime(pos.expiry_date, "%Y-%m-%d").date() - date.today()
                ).days

                # Rough delta estimate
                if current_price > pos.strike:
                    pos.delta = min(0.95, 0.5 + (current_price - pos.strike) / current_price)
                else:
                    pos.delta = max(0.05, 0.5 - (pos.strike - current_price) / current_price)

                # Check profit targets
                if pos.unrealized_pct >= 100:  # 2x return
                    pos.profit_target_hit = True

                if pos.unrealized_pct <= -50:  # 50% loss
                    pos.stop_loss_hit = True

            except Exception as e:
                print(f"Error updating {pos.ticker}: {e}")

        self.save_portfolio()

    def format_candidates(self, candidates: list[LEAPSCandidate], limit: int = 15) -> str:
        """Format LEAPS candidates for display."""
        if not candidates:
            return "🔍 No strong LEAPS candidates found."

        output = f"\n🚀 TOP LEAPS CANDIDATES ({min(limit, len(candidates))} shown)\n"
        output += " = " * 80 + "\n"

        for i, cand in enumerate(candidates[:limit], 1):
            # Timing indicators
            timing_icon = (
                "🟢"
                if cand.entry_timing_score > 70
                else "🟡"
                if cand.entry_timing_score > 50
                else "🔴"
            )

            # Cross type indicators
            if cand.ma_cross_signal.cross_type == "golden_cross":
                cross_icon = "✨"
                cross_info = (
                    f"Golden Cross ({cand.ma_cross_signal.days_since_cross}d ago)"
                    if cand.ma_cross_signal.days_since_cross
                    else "Golden Cross"
                )
            elif cand.ma_cross_signal.cross_type == "death_cross":
                cross_icon = "💀"
                cross_info = (
                    f"Death Cross ({cand.ma_cross_signal.days_since_cross}d ago)"
                    if cand.ma_cross_signal.days_since_cross
                    else "Death Cross"
                )
            else:
                cross_icon = "📊"
                cross_info = f"{cand.ma_cross_signal.trend_direction.title()} Trend"

            output += f"\n{i}. {cand.ticker} - {cand.company_name} {timing_icon}\n"
            output += f"   Theme: {cand.theme}\n"
            output += f"   Current: ${cand.current_price:.2f} | Target Strike: ${cand.recommended_strike}\n"
            output += f"   Expiry: {cand.expiry_date} | Premium: ${cand.premium_estimate:.2f}\n"
            output += (
                f"   Breakeven: ${cand.break_even:.2f} | 1Y Target: {cand.target_return_1y:.0f}%\n"
            )
            output += f"   Scores - Composite: {cand.composite_score:.0f} | Trend: {cand.trend_score:.0f} | Financial: {cand.financial_score:.0f}\n"
            output += f"   {cross_icon} MA Signal: {cross_info} | Entry Score: {cand.entry_timing_score:.0f} | Exit Score: {cand.exit_timing_score:.0f}\n"
            output += f"   Price vs SMA: 50d {cand.current_price / cand.ma_cross_signal.sma_50:.2f}x | 200d {cand.current_price / cand.ma_cross_signal.sma_200: .2f}x + n"

            if cand.risk_factors:
                output += f"   ⚠️  Risks: {', '.join(cand.risk_factors)}\n"

        output += "\n💡 ENHANCED LEAPS STRATEGY with GOLDEN / DEATH CROSS TIMING: \n"
        output += "• 🟢 BEST ENTRIES: Recent golden cross (30d) + high entry score (70+)\n"
        output += "• 🟡 GOOD ENTRIES: Bullish trend + price above 50 / 200 SMA + entry score 50+\n"
        output += "• 🔴 AVOID ENTRIES: Death cross (45d) + low entry score ( < 40)\n"
        output += "• ✨ Golden cross timing can add 10 - 20% to returns + n"
        output += "• 💀 Death cross signals - consider exits even on profitable positions + n"
        output += "• 📊 Use 50 / 200 SMA position for trend confirmation + n"
        output += "• Scale out at 2x, 3x, 4x returns (25% each) OR on death cross + n"
        output += "• Stop loss at -50% to preserve capital + n"
        output += "• Target 12 - 24 month expiries for time buffer + n"
        output += "• Diversify across 3 - 5 themes with good timing scores + n"

        return output

    def format_portfolio(self) -> str:
        """Format current LEAPS portfolio."""
        if not self.positions:
            return "📊 No LEAPS positions in portfolio."

        self.update_positions()

        total_cost = sum(pos.cost_basis for pos in self.positions)
        total_value = sum(pos.current_value for pos in self.positions)
        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0

        output = "\n📊 LEAPS PORTFOLIO SUMMARY + n"
        output += " = " * 60 + "\n"
        output += f"Total Positions: {len(self.positions)}\n"
        output += f"Total Cost Basis: ${total_cost:,.0f}\n"
        output += f"Current Value: ${total_value:,.0f}\n"
        output += f"Unrealized P & L: ${total_pnl:,.0f} ({total_pnl_pct:+.1f}%)\n + n"

        # Sort by P & L percentage
        sorted_positions = sorted(self.positions, key=lambda x: x.unrealized_pct, reverse=True)

        output += "INDIVIDUAL POSITIONS: \n"
        output += "-" * 60 + "\n"

        for pos in sorted_positions:
            status_indicators = []
            if pos.profit_target_hit:
                status_indicators.append("🎯")
            if pos.stop_loss_hit:
                status_indicators.append("🛑")
            if pos.days_to_expiry < 180:
                status_indicators.append("⏰")

            status = " ".join(status_indicators)

            output += f"{pos.ticker} ${pos.strike} Call {pos.expiry_date} {status}\n"
            output += f"  Entry: ${pos.entry_premium:.2f} @ ${pos.spot_at_entry: .2f} ({pos.days_held}d ago)\n"
            output += f"  Current: ${pos.current_premium:.2f} @ ${pos.current_spot: .2f} (Δ{pos.delta: .2f})\n"
            output += f"  P & L: ${pos.unrealized_pnl:,.0f} ({pos.unrealized_pct:+.1f}%) | {pos.days_to_expiry}d left + n\n"

        # Scale-out recommendations
        scale_recommendations = [
            pos for pos in self.positions if pos.unrealized_pct >= 100 and pos.scale_out_level < 3
        ]

        if scale_recommendations:
            output += "🎯 SCALE - OUT RECOMMENDATIONS: \n"
            for pos in scale_recommendations:
                next_level = pos.scale_out_level + 1
                output += f"• {pos.ticker}: Consider {25 * next_level}% scale-out at {pos.unrealized_pct: .0f}% gain + n"

        return output


def main():
    parser = argparse.ArgumentParser(description="LEAPS Secular Winners Tracker")
    parser.add_argument(
        "command", choices=["scan", "portfolio", "update"], help="Command to execute"
    )
    parser.add_argument("--output", choices=["json", "text"], default="text", help="Output format")
    parser.add_argument("--limit", type=int, default=15, help="Maximum results to show")
    parser.add_argument("--min - score", type=int, default=60, help="Minimum composite score")
    parser.add_argument("--save-csv", type=str, help="Save results to CSV file")
    parser.add_argument(
        "--sort - by - timing",
        action="store_true",
        help="Sort by entry timing score instead of composite score",
    )

    args = parser.parse_args()

    tracker = LEAPSTracker()

    if args.command == "scan":
        candidates = tracker.scan_secular_winners()

        # Filter by minimum score
        candidates = [c for c in candidates if c.composite_score >= args.min_score]

        # Sort by timing score if requested
        if args.sort_by_timing:
            candidates.sort(key=lambda x: x.entry_timing_score, reverse=True)

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
            print(json.dumps([asdict(c) for c in candidates[: args.limit]], indent=2, default=str))
        else:
            print(tracker.format_candidates(candidates, args.limit))

    elif args.command == "portfolio":
        print(tracker.format_portfolio())

    elif args.command == "update":
        tracker.update_positions()
        print("✅ Portfolio updated successfully")


if __name__ == "__main__":
    main()
