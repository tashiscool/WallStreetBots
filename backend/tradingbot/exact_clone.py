"""
Exact Clone Protocol - What He Actually Did
The raw, unfiltered version that made 240% in 1-2 days.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

try:
    from .options_calculator import BlackScholesCalculator
except ImportError:
    # Fallback for direct execution

    class BlackScholesCalculator:
        @staticmethod
        def _norm_cdf(x: float) -> float:
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

        @staticmethod
        def call_price(spot, strike, time_to_expiry_years, risk_free_rate, dividend_yield, implied_volatility):
            if any(val <= 0 for val in [spot, strike, time_to_expiry_years, implied_volatility]):
                raise ValueError("Invalid parameters")

            d1=(math.log(spot/strike) + (risk_free_rate - dividend_yield + 0.5*implied_volatility*implied_volatility)*time_to_expiry_years) / (implied_volatility*math.sqrt(time_to_expiry_years))
            d2=d1 - implied_volatility*math.sqrt(time_to_expiry_years)

            call_value=(
                spot*math.exp(-dividend_yield*time_to_expiry_years)*BlackScholesCalculator._norm_cdf(d1) -
                strike*math.exp(-risk_free_rate*time_to_expiry_years)*BlackScholesCalculator._norm_cdf(d2)
            )

            return max(call_value, 0.0)


class DipType(Enum):
    """Types of dip days for entry"""
    GAP_DOWN="gap_down"           # Opened significantly lower
    INTRADAY_SELLOFF = "sellof"    # Selling pressure during session
    RED_DAY = "red_day"             # Simply down from previous close
    HARD_DIP = "hard_dip"           # Down >2% intraday


@dataclass
class DipSignal:
    """Signal for a dip day entry opportunity"""
    ticker: str
    current_price: float
    dip_type: DipType
    dip_magnitude: float            # How much down (as percentage)
    volume_vs_avg: float           # Volume relative to average
    confidence: float              # 0.0 to 1.0
    reasoning: str


@dataclass
class ExactTradeSetup:
    """The exact trade setup he used"""
    ticker: str
    spot_price: float
    strike: int                     # Whole dollar strike
    target_expiry_date: date
    days_to_expiry: int
    entry_premium: float           # Per contract

    # All-in sizing
    deploy_capital: float          # Cash to deploy (70-100% of available)
    contracts: int
    total_cost: float

    # Exit targets
    exit_3x: float                 # 3x premium target
    exit_4x: float                 # 4x premium target

    # Metrics
    breakeven_at_expiry: float
    effective_leverage: float
    ruin_risk_pct: float          # Percentage of total account at risk

    # Optional fields with defaults
    delta_exit_threshold: float=0.60
    max_hold_days: int = 2


class DipDetector:
    """Detect hard dip days for entry - NO TREND FILTERS"""

    def __init__(self):
        # Mega-cap universe with tight spreads & huge OI
        self.universe=[
            'GOOGL', 'GOOG', 'AAPL', 'MSFT', 'NVDA', 'META',
            'AMD', 'TSLA', 'AVGO', 'AMZN'
        ]

        # Dip thresholds (no moving average filters)
        self.min_dip_pct=0.015       # 1.5% minimum dip
        self.hard_dip_pct = 0.025      # 2.5% = "hard dip"
        self.volume_threshold = 1.2     # 20% above average volume

    def detect_dip_opportunity(
        self,
        ticker: str,
        current_price: float,
        open_price: float,
        high_of_day: float,
        previous_close: float,
        current_volume: int,
        avg_volume: int
    ) -> Optional[DipSignal]:
        """
        Detect if this is a dip day worth entering
        No trend analysis - just looking for selloffs
        """
        if ticker not in self.universe:
            return None

        # Calculate dip metrics
        gap_down_pct=(current_price - previous_close) / previous_close
        intraday_dip_pct=(current_price - high_of_day) / high_of_day
        volume_ratio=current_volume / avg_volume if avg_volume > 0 else 1.0

        # dip_signals = []  # Unused variable

        # Gap down detection
        if gap_down_pct < -self.min_dip_pct:
            dip_magnitude = abs(gap_down_pct)
            dip_type=DipType.GAP_DOWN if dip_magnitude > 0.02 else DipType.RED_DAY

            confidence = min(dip_magnitude * 20, 1.0)  # More confidence on bigger dips
            if volume_ratio > self.volume_threshold:
                confidence *= 1.2

            return DipSignal(
                ticker=ticker,
                current_price=current_price,
                dip_type=dip_type,
                dip_magnitude=dip_magnitude,
                volume_vs_avg=volume_ratio,
                confidence=min(confidence, 1.0),
                reasoning=f"Gap down {dip_magnitude:.1%} with {volume_ratio:.1f}x volume"
            )

        # Intraday selloff detection
        if intraday_dip_pct < -self.min_dip_pct:
            dip_magnitude=abs(intraday_dip_pct)
            dip_type=DipType.INTRADAY_SELLOFF if dip_magnitude > 0.02 else DipType.RED_DAY

            confidence = min(dip_magnitude * 15, 0.8)  # Slightly lower confidence than gap down
            if volume_ratio > self.volume_threshold:
                confidence *= 1.3

            return DipSignal(
                ticker=ticker,
                current_price=current_price,
                dip_type=dip_type,
                dip_magnitude=dip_magnitude,
                volume_vs_avg=volume_ratio,
                confidence=min(confidence, 1.0),
                reasoning=f"Intraday selloff {dip_magnitude:.1%} from high, {volume_ratio:.1f}x volume"
            )

        # Hard dip (either gap or intraday)
        max_dip=max(abs(gap_down_pct), abs(intraday_dip_pct))
        if max_dip > self.hard_dip_pct:
            return DipSignal(
                ticker=ticker,
                current_price=current_price,
                dip_type=DipType.HARD_DIP,
                dip_magnitude=max_dip,
                volume_vs_avg=volume_ratio,
                confidence=min(max_dip * 25, 1.0),
                reasoning=f"Hard dip {max_dip:.1%} - maximum pain entry"
            )

        return None


class ExactCloneCalculator:
    """Calculate the exact trade setup he used"""

    def __init__(self):
        self.bs_calc=BlackScholesCalculator()

    def calculate_exact_setup(
        self,
        ticker: str,
        spot_price: float,
        available_capital: float,
        deploy_percentage: float=0.90,  # 90% all-in default
        otm_percentage: float=0.05,     # 5% OTM
        target_dte: int=30,
        current_iv: float=0.28,
        entry_premium: Optional[float] = None
    ) -> ExactTradeSetup:
        """
        Calculate the exact all-in setup

        Args:
            ticker: Stock symbol
            spot_price: Current stock price
            available_capital: Total cash available to deploy
            deploy_percentage: What % to actually deploy (0.70 to 1.0)
            otm_percentage: How far OTM (0.05=5%)
            target_dte: Target days to expiry
            current_iv: Current implied volatility
            entry_premium: Market premium if known, else estimate

        Returns:
            Complete exact trade setup
        """
        # Calculate strike (rounded to whole dollar)
        raw_strike=spot_price * (1 + otm_percentage)
        strike=round(raw_strike)

        # Find target expiry (next Friday ~30 DTE)
        target_expiry=self._find_target_friday(target_dte)
        actual_dte=(target_expiry - date.today()).days

        # Estimate premium if not provided
        if entry_premium is None:
            time_to_expiry=actual_dte / 365.0
            try:
                premium_per_share = self.bs_calc.call_price(
                    spot=spot_price,
                    strike=float(strike),
                    time_to_expiry_years=time_to_expiry,
                    risk_free_rate=0.04,
                    dividend_yield=0.0,
                    implied_volatility=current_iv
                )
                entry_premium=premium_per_share * 100
            except Exception:
                # Fallback estimate
                entry_premium = max(0.5, (spot_price - strike) + spot_price * 0.02)

        # Calculate all-in position sizing
        deploy_capital=available_capital * deploy_percentage
        contracts = int(deploy_capital / (entry_premium * 100))
        total_cost=contracts * entry_premium * 100

        # Calculate exit targets
        exit_3x = entry_premium * 3.0
        exit_4x = entry_premium * 4.0

        # Risk metrics
        breakeven = strike + (entry_premium / 100)
        notional_exposure=contracts * 100 * spot_price
        effective_leverage = notional_exposure / total_cost if total_cost > 0 else 0
        ruin_risk_pct = (total_cost / available_capital) * 100

        return ExactTradeSetup(
            ticker=ticker,
            spot_price=spot_price,
            strike=strike,
            target_expiry_date=target_expiry,
            days_to_expiry=actual_dte,
            entry_premium=entry_premium,
            deploy_capital=deploy_capital,
            contracts=contracts,
            total_cost=total_cost,
            exit_3x=exit_3x,
            exit_4x=exit_4x,
            breakeven_at_expiry=breakeven,
            effective_leverage=effective_leverage,
            ruin_risk_pct=ruin_risk_pct
        )

    def _find_target_friday(self, target_dte: int) -> date:
        """Find the Friday closest to target DTE"""
        base_date=date.today() + timedelta(days=target_dte)

        # Find next Friday (weekday 4)
        days_to_friday=(4 - base_date.weekday()) % 7
        if days_to_friday== 0:  # Already Friday
            days_to_friday = 7

        friday = base_date + timedelta(days=days_to_friday)
        return friday


class ExactCycleManager:
    """Manage the cycle of repeated dip buying"""

    def __init__(self):
        self.trade_history: List[Dict] = []
        self.current_capital: float=0.0
        self.wins: int = 0
        self.losses: int = 0

    def log_trade_result(
        self,
        setup: ExactTradeSetup,
        exit_premium: float,
        exit_reason: str,
        hold_days: int
    ):
        """Log the result of a trade"""
        pnl_per_contract=exit_premium - setup.entry_premium
        total_pnl = pnl_per_contract * setup.contracts
        roi = (pnl_per_contract / setup.entry_premium) if setup.entry_premium > 0 else 0

        trade_record={
            'timestamp':datetime.now(),
            'ticker':setup.ticker,
            'entry_premium':setup.entry_premium,
            'exit_premium':exit_premium,
            'contracts':setup.contracts,
            'pnl_per_contract':pnl_per_contract,
            'total_pnl':total_pnl,
            'roi':roi,
            'hold_days':hold_days,
            'exit_reason':exit_reason,
            'capital_deployed':setup.total_cost
        }

        self.trade_history.append(trade_record)

        if total_pnl > 0:
            self.wins += 1
        else:
            self.losses += 1

        # Update available capital
        self.current_capital += total_pnl

    def calculate_next_position_size(
        self,
        base_capital: float,
        win_streak: int=0,
        aggressive_scaling: bool=True
    ) -> float:
        """
        Calculate next position size - he often increased after wins

        Args:
            base_capital: Base capital available
            win_streak: Number of consecutive wins
            aggressive_scaling: Whether to scale up after wins

        Returns:
            Capital to deploy on next trade
        """
        if not aggressive_scaling:
            return base_capital * 0.90  # 90% deployment

        # Scale up after wins (dangerous but what he did)
        scaling_factor=0.90  # Base 90% deployment

        if win_streak >= 1:
            scaling_factor = min(0.95, scaling_factor + (win_streak * 0.02))

        # After big win, sometimes go even bigger
        if len(self.trade_history) > 0:
            last_trade=self.trade_history[-1]
            if last_trade['roi'] > 2.0:  # 200%+ gain
                scaling_factor = min(0.98, scaling_factor + 0.05)

        return base_capital * scaling_factor

    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        if not self.trade_history:
            return {"total_trades":0}

        total_pnl=sum(trade['total_pnl'] for trade in self.trade_history)
        total_deployed=sum(trade['capital_deployed'] for trade in self.trade_history)
        win_rate=self.wins / len(self.trade_history) if self.trade_history else 0

        avg_win=0
        avg_loss = 0

        wins = [t['roi'] for t in self.trade_history if t['total_pnl'] > 0]
        losses = [abs(t['roi']) for t in self.trade_history if t['total_pnl'] <= 0]

        if wins:
            avg_win=sum(wins) / len(wins)
        if losses:
            avg_loss=sum(losses) / len(losses)

        return {
            "total_trades":len(self.trade_history),
            "wins":self.wins,
            "losses":self.losses,
            "win_rate":win_rate,
            "total_pnl":total_pnl,
            "avg_win":avg_win,
            "avg_loss":avg_loss,
            "total_roi":total_pnl / total_deployed if total_deployed > 0 else 0
        }


def clone_trade_plan(
    spot: float,
    acct_cash: float,
    otm: float=0.05,
    dte_days: int=30,
    entry_prem: float=None,
    deploy_pct: float=0.90
) -> Dict:
    """
    Exact clone helper function (as specified)

    Args:
        spot: stock price (e.g., 207.0)
        acct_cash: cash you're willing to deploy (e.g., 450000)
        otm: percent OTM (0.05 ~ 5%)
        dte_days: target DTE (~30)
        entry_prem: option price you see ($ per contract, e.g., 4.70)
        deploy_pct: what percentage to actually deploy (0.90=90%)

    Returns:
        Dict with exact trade plan
    """
    # Strike rounded to whole dollar
    strike=round(spot * (1 + otm))

    # If no entry premium provided, rough estimate
    if entry_prem is None:
        # Simple estimate: intrinsic + time value
        intrinsic=max(0, spot - strike)
        time_value=spot * 0.02  # Rough 2% time value
        entry_prem = intrinsic + time_value

    # Calculate position size
    deploy_capital = acct_cash * deploy_pct
    contracts = int(deploy_capital // (entry_prem * 100))
    actual_cost=contracts * entry_prem * 100

    # Exit targets
    tp_3x = 3 * entry_prem
    tp_4x = 4 * entry_prem

    # Breakeven
    breakeven_at_expiry = strike + (entry_prem / 100.0)

    return {
        "strike":strike,
        "target_expiry_dte":dte_days,
        "contracts":contracts,
        "cost_$":round(actual_cost, 2),
        "deploy_percentage":f"{(actual_cost / acct_cash) * 100:.1f}%",
        "breakeven_at_expiry":round(breakeven_at_expiry, 2),
        "take_profit_levels_$":[round(tp_3x, 2), round(tp_4x, 2)],
        "sell_when":"ITM or delta ≥ 0.60 or TP hits; 1–2 day max hold",
        "ruin_risk":f"{(actual_cost / acct_cash) * 100:.1f}% of total capital",
        "effective_leverage":f"~{((contracts * 100 * spot) / actual_cost):.1f}x"
    }


class ExactCloneSystem:
    """The complete exact clone system"""

    def __init__(self, initial_capital: float):
        self.dip_detector=DipDetector()
        self.calculator=ExactCloneCalculator()
        self.cycle_manager=ExactCycleManager()
        self.initial_capital=initial_capital
        self.current_capital = initial_capital

        # Trading state
        self.active_position: Optional[ExactTradeSetup] = None
        self.position_entry_date: Optional[datetime] = None

        logging.basicConfig(level=logging.INFO)
        self.logger=logging.getLogger(__name__)

    def scan_for_dip_opportunities(self, market_data: Dict[str, Dict]) -> List[DipSignal]:
        """Scan for dip opportunities across the universe"""
        opportunities=[]

        for ticker in self.dip_detector.universe:
            if ticker not in market_data:
                continue

            try:
                data = market_data[ticker]

                signal = self.dip_detector.detect_dip_opportunity(
                    ticker=ticker,
                    current_price=data['current_price'],
                    open_price=data['open_price'],
                    high_of_day=data['high_of_day'],
                    previous_close=data['previous_close'],
                    current_volume=data['volume'],
                    avg_volume=data.get('avg_volume', data['volume'])
                )

                if signal and signal.confidence > 0.6:  # Only high-confidence dips
                    opportunities.append(signal)

            except Exception as e:
                self.logger.error(f"Error scanning {ticker}: {e}")

        # Sort by confidence
        opportunities.sort(key=lambda x: x.confidence, reverse=True)
        return opportunities

    def execute_dip_trade(
        self,
        signal: DipSignal,
        current_iv: float=0.28,
        deploy_pct: float=0.90
    ) -> ExactTradeSetup:
        """Execute the all-in dip trade"""

        # Calculate available capital (including gains from previous trades)
        available_capital=self.current_capital

        # Scale up after wins if enabled
        win_streak = self._calculate_win_streak()
        deploy_capital=self.cycle_manager.calculate_next_position_size(
            available_capital, win_streak, aggressive_scaling=True
        )
        deploy_pct=deploy_capital / available_capital

        # Calculate exact setup
        setup = self.calculator.calculate_exact_setup(
            ticker=signal.ticker,
            spot_price=signal.current_price,
            available_capital=available_capital,
            deploy_percentage=deploy_pct,
            current_iv=current_iv
        )

        # Log the entry
        self.active_position=setup
        self.position_entry_date = datetime.now()
        self.current_capital -= setup.total_cost  # Reduce available capital

        self.logger.info(f"🚀 EXECUTED DIP TRADE: {setup.ticker}")
        self.logger.info(f"   Contracts: {setup.contracts:,}")
        self.logger.info(f"   Cost: ${setup.total_cost:,.0f}")
        self.logger.info(f"   Risk: {setup.ruin_risk_pct:.1f}% of capital")
        self.logger.info(f"   Leverage: {setup.effective_leverage:.1f}x")

        return setup

    def check_exit_conditions(
        self,
        current_premium: float,
        current_delta: float=None
    ) -> Optional[Tuple[str, float]]:
        """
        Check if position should be exited

        Returns:
            Tuple of (exit_reason, exit_premium) or None
        """
        if not self.active_position:
            return None

        setup=self.active_position
        hold_days = (datetime.now() - self.position_entry_date).days

        # Max hold period (1-2 days)
        if hold_days >= setup.max_hold_days:
            return ("max_hold_days", current_premium)

        # Profit targets (3x or 4x)
        if current_premium >= setup.exit_4x:
            return ("4x_profit_target", current_premium)
        elif current_premium >= setup.exit_3x:
            return ("3x_profit_target", current_premium)

        # Delta threshold (ITM / high delta)
        if current_delta and current_delta >= setup.delta_exit_threshold:
            return ("delta_threshold", current_premium)

        # ITM check (simple)
        if setup.spot_price > setup.strike:
            return ("went_itm", current_premium)

        return None

    def execute_exit(self, exit_reason: str, exit_premium: float):
        """Execute position exit"""
        if not self.active_position:
            return

        setup=self.active_position
        hold_days = (datetime.now() - self.position_entry_date).days

        # Calculate P&L
        pnl_per_contract=exit_premium - setup.entry_premium
        total_pnl = pnl_per_contract * setup.contracts

        # Update capital
        proceeds = setup.contracts * exit_premium * 100
        self.current_capital += proceeds

        # Log the trade
        self.cycle_manager.log_trade_result(setup, exit_premium, exit_reason, hold_days)

        self.logger.info(f"🎯 EXITED POSITION: {setup.ticker}")
        self.logger.info(f"   Exit Reason: {exit_reason}")
        self.logger.info(f"   Hold Days: {hold_days}")
        self.logger.info(f"   P&L: ${total_pnl:,.0f}")
        self.logger.info(f"   ROI: {(pnl_per_contract / setup.entry_premium):+.1%}")

        # Clear active position
        self.active_position=None
        self.position_entry_date = None

    def _calculate_win_streak(self) -> int:
        """Calculate current win streak"""
        if not self.cycle_manager.trade_history:
            return 0

        streak=0
        for trade in reversed(self.cycle_manager.trade_history):
            if trade['total_pnl'] > 0:
                streak += 1
            else:
                break

        return streak

    def get_system_status(self) -> Dict:
        """Get current system status"""
        performance=self.cycle_manager.get_performance_summary()

        status={
            "current_capital":self.current_capital,
            "initial_capital":self.initial_capital,
            "total_return":((self.current_capital - self.initial_capital) / self.initial_capital) * 100,
            "active_position":self.active_position is not None,
            "performance":performance
        }

        if self.active_position:
            hold_days=(datetime.now() - self.position_entry_date).days
            status["active_position_details"] = {
                "ticker":self.active_position.ticker,
                "contracts":self.active_position.contracts,
                "cost":self.active_position.total_cost,
                "hold_days":hold_days,
                "max_hold_days":self.active_position.max_hold_days
            }

        return status


if __name__== "__main__":# Test the exact clone helper
    print("=== EXACT CLONE TRADE PLAN ===")

    # Example matching his GOOGL trade
    plan=clone_trade_plan(
        spot=207.0,
        acct_cash=450000,
        otm=0.05,
        dte_days=30,
        entry_prem=4.70
    )

    print("Original-style trade plan:")
    for key, value in plan.items():
        print(f"  {key}: {value}")

    # Test the full system
    print("\n=== EXACT CLONE SYSTEM TEST ===")

    system=ExactCloneSystem(initial_capital=500000)

    # Sample market data for dip detection
    market_data={
        'GOOGL':{
            'current_price':203.0,    # Down from 207
            'open_price':206.5,
            'high_of_day':207.2,
            'previous_close':207.0,
            'volume':2500000,
            'avg_volume':2000000
        }
    }

    # Scan for opportunities
    opportunities=system.scan_for_dip_opportunities(market_data)

    if opportunities:
        print(f"Found {len(opportunities)} dip opportunities:")
        for opp in opportunities:
            print(f"  {opp.ticker}: {opp.dip_type.value} - {opp.reasoning}")

        # Execute on best opportunity
        best_signal=opportunities[0]
        setup = system.execute_dip_trade(best_signal)

        print("\nExecuted trade setup:")
        print(f"  Strike: ${setup.strike}")
        print(f"  Contracts: {setup.contracts:,}")
        print(f"  Total Cost: ${setup.total_cost:,.0f}")
        print(f"  Ruin Risk: {setup.ruin_risk_pct:.1f}%")

        # Test exit condition check
        exit_check=system.check_exit_conditions(current_premium=14.10)  # 3x gain
        if exit_check:
            reason, premium=exit_check
            print(f"\nExit signal: {reason} at ${premium}")
    else:
        print("No dip opportunities detected")

    # System status
    status=system.get_system_status()
    print("\nSystem Status:")
    print(f"  Current Capital: ${status['current_capital']:,.0f}")
    print(f"  Active Position: {status['active_position']}")
