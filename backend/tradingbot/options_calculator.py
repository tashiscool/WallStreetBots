"""
Options Trading Calculator - Black - Scholes Implementation
Replicates the successful 240% options trade playbook with risk management.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
from decimal import Decimal, getcontext

# Set high precision for financial calculations
getcontext().prec = 10


class BlackScholesCalculator: 
    """Black - Scholes - Merton options pricing calculator"""

    @staticmethod
    def _norm_cdf(x: float)->float:
        """Standard normal cumulative distribution function using error function"""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

    @staticmethod
    def call_price(
        spot: float,
        strike: float,
        time_to_expiry_years: float,
        risk_free_rate: float,
        dividend_yield: float,
        implied_volatility: float
    )->float: 
        """
        Calculate Black - Scholes call option price per share

        Args: 
            spot: Current stock price
            strike: Option strike price
            time_to_expiry_years: Time to expiration in years
            risk_free_rate: Risk - free interest rate (annual)
            dividend_yield: Continuous dividend yield (annual)
            implied_volatility: Implied volatility (annual)

        Returns: 
            Call option price per share (multiply by 100 for contract value)
        """
        if any(val  <=  0 for val in [spot, strike, time_to_expiry_years, implied_volatility]): 
            raise ValueError("Spot, strike, time to expiry, and IV must be positive")

        # Calculate d1 and d2
        d1 = (
            math.log(spot / strike) +
            (risk_free_rate-dividend_yield + 0.5 * implied_volatility ** 2) * time_to_expiry_years
        ) / (implied_volatility * math.sqrt(time_to_expiry_years))

        d2 = d1 - implied_volatility * math.sqrt(time_to_expiry_years)

        # Calculate call price
        call_value = (
            spot * math.exp(-dividend_yield * time_to_expiry_years) * BlackScholesCalculator._norm_cdf(d1) -
            strike * math.exp(-risk_free_rate * time_to_expiry_years) * BlackScholesCalculator._norm_cdf(d2)
        )

        return max(call_value, 0.0)

    @staticmethod
    def delta(
        spot: float,
        strike: float,
        time_to_expiry_years: float,
        risk_free_rate: float,
        dividend_yield: float,
        implied_volatility: float
    )->float: 
        """Calculate option delta (price sensitivity to underlying movement)"""
        if any(val  <=  0 for val in [spot, strike, time_to_expiry_years, implied_volatility]): 
            return 0.0

        d1 = (
            math.log(spot / strike) +
            (risk_free_rate-dividend_yield + 0.5 * implied_volatility ** 2) * time_to_expiry_years
        ) / (implied_volatility * math.sqrt(time_to_expiry_years))

        return math.exp(-dividend_yield * time_to_expiry_years) * BlackScholesCalculator._norm_cdf(d1)

    @staticmethod
    def put_price(
        spot: float,
        strike: float,
        time_to_expiry_years: float,
        risk_free_rate: float,
        dividend_yield: float,
        implied_volatility: float
    )->float: 
        """Calculate Black - Scholes put option price per share"""
        if any(val  <=  0 for val in [spot, strike, time_to_expiry_years, implied_volatility]): 
            raise ValueError("Spot, strike, time to expiry, and IV must be positive")

        d1 = (
            math.log(spot / strike) +
            (risk_free_rate-dividend_yield + 0.5 * implied_volatility ** 2) * time_to_expiry_years
        ) / (implied_volatility * math.sqrt(time_to_expiry_years))

        d2 = d1 - implied_volatility * math.sqrt(time_to_expiry_years)

        put_value = (
            strike * math.exp(-risk_free_rate * time_to_expiry_years) * BlackScholesCalculator._norm_cdf(-d2) -
            spot * math.exp(-dividend_yield * time_to_expiry_years) * BlackScholesCalculator._norm_cdf(-d1)
        )

        return max(put_value, 0.0)

    @staticmethod
    def gamma(
        spot: float,
        strike: float,
        time_to_expiry_years: float,
        risk_free_rate: float,
        dividend_yield: float,
        implied_volatility: float
    )->float: 
        """Calculate option gamma (delta sensitivity to underlying movement)"""
        if any(val  <=  0 for val in [spot, strike, time_to_expiry_years, implied_volatility]): 
            return 0.0

        d1 = (
            math.log(spot / strike) +
            (risk_free_rate-dividend_yield + 0.5 * implied_volatility ** 2) * time_to_expiry_years
        ) / (implied_volatility * math.sqrt(time_to_expiry_years))

        # Standard normal PDF
        pdf_d1 = math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
        
        return (
            math.exp(-dividend_yield * time_to_expiry_years) * pdf_d1 / 
            (spot * implied_volatility * math.sqrt(time_to_expiry_years))
        )

    @staticmethod
    def theta(
        spot: float,
        strike: float,
        time_to_expiry_years: float,
        risk_free_rate: float,
        dividend_yield: float,
        implied_volatility: float
    )->float: 
        """Calculate option theta (time decay)"""
        if any(val  <=  0 for val in [spot, strike, time_to_expiry_years, implied_volatility]): 
            return 0.0

        d1 = (
            math.log(spot / strike) +
            (risk_free_rate-dividend_yield + 0.5 * implied_volatility ** 2) * time_to_expiry_years
        ) / (implied_volatility * math.sqrt(time_to_expiry_years))

        d2 = d1 - implied_volatility * math.sqrt(time_to_expiry_years)
        
        # Standard normal PDF
        pdf_d1 = math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
        
        # Call theta
        theta_value = (
            -spot * math.exp(-dividend_yield * time_to_expiry_years) * pdf_d1 * implied_volatility / 
            (2 * math.sqrt(time_to_expiry_years)) -
            risk_free_rate * strike * math.exp(-risk_free_rate * time_to_expiry_years) * 
            BlackScholesCalculator._norm_cdf(d2) +
            dividend_yield * spot * math.exp(-dividend_yield * time_to_expiry_years) * 
            BlackScholesCalculator._norm_cdf(d1)
        )
        
        return theta_value / 365  # Convert to daily theta

    @staticmethod
    def vega(
        spot: float,
        strike: float,
        time_to_expiry_years: float,
        risk_free_rate: float,
        dividend_yield: float,
        implied_volatility: float
    )->float: 
        """Calculate option vega (volatility sensitivity)"""
        if any(val  <=  0 for val in [spot, strike, time_to_expiry_years, implied_volatility]): 
            return 0.0

        d1 = (
            math.log(spot / strike) +
            (risk_free_rate-dividend_yield + 0.5 * implied_volatility ** 2) * time_to_expiry_years
        ) / (implied_volatility * math.sqrt(time_to_expiry_years))
        
        # Standard normal PDF
        pdf_d1 = math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
        
        return (
            spot * math.exp(-dividend_yield * time_to_expiry_years) * pdf_d1 * 
            math.sqrt(time_to_expiry_years) / 100  # Convert to 1% volatility change
        )


@dataclass
class OptionsStrategySetup: 
    """Configuration for options trade setup based on successful playbook"""

    # Universe filter
    mega_cap_tickers: List[str] = field(default_factory=lambda: [
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'AVGO', 'AMD', 'TSLA'
    ])

    # Option selection criteria (from successful 240% trade)
    target_dte_min: int = 21
    target_dte_max: int = 45
    target_dte_optimal: int = 30
    otm_percentage: float=0.05  # 5% out of money
    target_delta_range: Tuple[float, float] = (0.30, 0.40)

    # Risk management (learned from existential bet)
    max_single_trade_risk_pct: float=0.15  # Max 15% of account per trade
    recommended_risk_pct: float=0.10       # Recommended 10% per trade
    max_kelly_fraction: float=0.50         # Never exceed 50% Kelly

    # Market regime filters
    min_iv_percentile: float=0.0   # Prefer low IV
    max_iv_percentile: float=40.0  # Avoid high IV entries

    # Exit discipline
    profit_take_levels: List[float] = field(default_factory=lambda: [1.0, 2.0, 2.5])  # 100%, 200%, 250%
    stop_loss_pct: float=0.45      # Stop at 45% loss
    max_hold_days: int = 5           # Time stop if no progress
    delta_exit_threshold: float=0.60  # Exit when delta  >=  0.60


@dataclass
class OptionsSetup: 
    """Individual options position setup"""
    ticker: str
    entry_date: date
    expiry_date: date
    strike: float
    spot_at_entry: float
    premium_paid: float
    contracts: int
    
    @property
    def total_cost(self)->float: 
        """Total cost of the position"""
        return self.contracts * self.premium_paid * 100
    
    @property
    def breakeven(self)->float: 
        """Breakeven price at expiration"""
        return self.strike+self.premium_paid
    
    @property
    def intrinsic_value(self)->float: 
        """Current intrinsic value"""
        return max(0.0, self.spot_at_entry - self.strike)
    
    def is_itm(self)->bool: 
        """Check if option is in the money"""
        return self.spot_at_entry  >  self.strike
    
    def is_otm(self)->bool: 
        """Check if option is out of the money"""
        return self.spot_at_entry  <  self.strike
    
    def calculate_pnl(self, current_spot: float, current_premium: float)->float:
        """Calculate current P & L"""
        return self.contracts * (current_premium - self.premium_paid) * 100


@dataclass
class TradeCalculation: 
    """Results of options trade calculation"""
    ticker: str
    spot_price: float
    strike: float
    expiry_date: date
    days_to_expiry: int
    estimated_premium: float        # Per contract
    recommended_contracts: int
    total_cost: float
    breakeven_price: float
    estimated_delta: float
    leverage_ratio: float
    risk_amount: float
    account_risk_pct: float

    def __str__(self)->str: 
        return """
=== OPTIONS TRADE CALCULATION===
Ticker: {self.ticker}
Current Price: ${self.spot_price:.2f}
Strike: ${self.strike:.2f} ({((self.strike / self.spot_price-1) * 100):+.1f}% OTM)
Expiry: {self.expiry_date} ({self.days_to_expiry} DTE)

POSITION SIZING: 
Contracts: {self.recommended_contracts:,}
Premium per Contract: ${self.estimated_premium:.2f}
Total Cost: ${self.total_cost:,.0f}
Account Risk: {self.account_risk_pct:.1f}%

RISK METRICS: 
Breakeven: ${self.breakeven_price:.2f}
Estimated Delta: {self.estimated_delta:.3f}
Effective Leverage: {self.leverage_ratio:.1f}x
Max Loss: ${self.risk_amount:,.0f}
        """


class OptionsTradeCalculator: 
    """Main calculator implementing the successful options playbook"""

    def __init__(self, setup: OptionsStrategySetup=None):
        self.setup=setup or OptionsStrategySetup()
        self.bs_calc=BlackScholesCalculator()

    def find_optimal_expiry(self, target_dte: int=None)->date:
        """Find the Friday closest to target DTE within acceptable range"""
        target_dte = target_dte or self.setup.target_dte_optimal

        # Start from target date
        base_date = date.today() + timedelta(days=target_dte)

        # Find nearest Friday
        days_to_friday = (4 - base_date.weekday()) % 7  # Friday is weekday 4
        candidate_date = base_date+timedelta(days=days_to_friday)

        # Ensure within acceptable DTE range
        actual_dte = (candidate_date-date.today()).days

        if actual_dte  <  self.setup.target_dte_min: 
            # Move to next Friday
            candidate_date += timedelta(days=7)
        elif actual_dte  >  self.setup.target_dte_max: 
            # Move to previous Friday
            candidate_date -= timedelta(days=7)

        return candidate_date

    def calculate_otm_strike(self, spot_price: float, increment: float=1.0)->float:
        """Calculate 5% OTM strike rounded to proper increment"""
        raw_strike = spot_price * (1 + self.setup.otm_percentage)
        return round(raw_strike / increment) * increment

    def calculate_trade(
        self,
        ticker: str,
        spot_price: float,
        account_size: float,
        implied_volatility: float,
        risk_pct: float=None,
        risk_free_rate: float=0.04,
        dividend_yield: float=0.0,
        custom_dte: int=None)->TradeCalculation: 
        """
        Calculate complete options trade based on successful playbook

        Args: 
            ticker: Stock ticker symbol
            spot_price: Current stock price
            account_size: Total account value
            implied_volatility: Expected IV (annualized)
            risk_pct: Percentage of account to risk (defaults to setup recommendation)
            risk_free_rate: Risk - free rate (annual)
            dividend_yield: Dividend yield (annual)
            custom_dte: Custom days to expiry (overrides setup)

        Returns: 
            Complete trade calculation with all metrics
        """
        # Validate inputs
        if spot_price  <=  0 or account_size  <=  0 or implied_volatility  <=  0: 
            raise ValueError("Spot price, account size, and IV must be positive")

        risk_pct = risk_pct or self.setup.recommended_risk_pct
        if not (0  <  risk_pct  <=  self.setup.max_single_trade_risk_pct): 
            raise ValueError(f"Risk percentage must be between 0 and {self.setup.max_single_trade_risk_pct}")

        # Calculate expiry and strike
        expiry_date = self.find_optimal_expiry(custom_dte)
        days_to_expiry = (expiry_date-date.today()).days
        time_to_expiry_years = days_to_expiry / 365.0

        strike = self.calculate_otm_strike(spot_price)

        # Calculate option premium using Black - Scholes
        premium_per_share = self.bs_calc.call_price(
            spot=spot_price,
            strike=strike,
            time_to_expiry_years=time_to_expiry_years,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            implied_volatility = implied_volatility)

        premium_per_contract = max(premium_per_share * 100, 0.01)

        # Calculate delta
        estimated_delta = self.bs_calc.delta(
            spot=spot_price,
            strike=strike,
            time_to_expiry_years=time_to_expiry_years,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            implied_volatility = implied_volatility)

        # Position sizing
        risk_amount = account_size * risk_pct
        recommended_contracts = max(int(risk_amount / premium_per_contract), 0)
        total_cost = recommended_contracts * premium_per_contract
        actual_risk_pct = total_cost / account_size

        # Calculate metrics
        breakeven_price = strike+(premium_per_contract / 100)
        notional_exposure = recommended_contracts * 100 * spot_price
        leverage_ratio = notional_exposure / total_cost if total_cost  >  0 else 0

        return TradeCalculation(
            ticker=ticker,
            spot_price=spot_price,
            strike=strike,
            expiry_date=expiry_date,
            days_to_expiry=days_to_expiry,
            estimated_premium=premium_per_contract,
            recommended_contracts=recommended_contracts,
            total_cost=total_cost,
            breakeven_price=breakeven_price,
            estimated_delta=estimated_delta,
            leverage_ratio=leverage_ratio,
            risk_amount=total_cost,
            account_risk_pct = actual_risk_pct * 100
        )

    def scenario_analysis(
        self,
        trade_calc: TradeCalculation,
        spot_moves: List[float],
        implied_volatility: float,
        days_passed: int=0,
        risk_free_rate: float=0.04,
        dividend_yield: float=0.0
    )->List[Dict]: 
        """
        Generate scenario analysis for different spot price moves

        Args: 
            trade_calc: Original trade calculation
            spot_moves: List of percentage moves to analyze (e.g., [-0.05, 0, 0.05])
            implied_volatility: IV to use for scenario pricing
            days_passed: Days that have passed since entry

        Returns: 
            List of scenario dictionaries with P & L analysis
        """
        remaining_days = max(trade_calc.days_to_expiry - days_passed, 1)
        time_to_expiry_years = remaining_days / 365.0

        scenarios = []

        for move in spot_moves: 
            new_spot = trade_calc.spot_price * (1 + move)

            try: 
                new_premium_per_share = self.bs_calc.call_price(
                    spot=new_spot,
                    strike = trade_calc.strike,
                    time_to_expiry_years=time_to_expiry_years,
                    risk_free_rate=risk_free_rate,
                    dividend_yield=dividend_yield,
                    implied_volatility = implied_volatility)

                new_premium_per_contract = new_premium_per_share * 100
                pnl_per_contract = new_premium_per_contract - trade_calc.estimated_premium
                total_pnl = pnl_per_contract * trade_calc.recommended_contracts
                roi = (pnl_per_contract / trade_calc.estimated_premium) if trade_calc.estimated_premium  >  0 else 0

                scenarios.append({
                    'spot_move': f"{move:+.1%}",
                    'new_spot_price': round(new_spot, 2),
                    'new_premium': round(new_premium_per_contract, 2),
                    'pnl_per_contract': round(pnl_per_contract, 2),
                    'total_pnl': round(total_pnl, 2),
                    'roi': f"{roi:+.1%}",
                    'days_remaining': remaining_days
                })

            except (ValueError, ZeroDivisionError): 
                scenarios.append({
                    'spot_move': f"{move:+.1%}",
                    'new_spot_price': round(new_spot, 2),
                    'new_premium': 0.0,
                    'pnl_per_contract': -trade_calc.estimated_premium,
                    'total_pnl': -trade_calc.total_cost,
                    'roi': "-100.0%",
                    'days_remaining': remaining_days
                })

        return scenarios


def validate_successful_trade(): 
    """Validate calculator against the documented successful trade"""
    # Original trade parameters
    contracts = 950
    entry_premium = 4.70
    exit_premium = 16.00
    strike = 220
    spot_at_entry = 207

    # Calculate actual returns
    cost = contracts * 100 * entry_premium
    proceeds = contracts * 100 * exit_premium
    profit = proceeds - cost
    roi = profit / cost

    print("=== VALIDATION OF SUCCESSFUL TRADE ===")
    print(f"Contracts: {contracts:,}")
    print(f"Strike: ${strike}")
    print(f"Entry Premium: ${entry_premium}")
    print(f"Exit Premium: ${exit_premium}")
    print(f"Total Cost: ${cost:,.0f}")
    print(f"Total Proceeds: ${proceeds:,.0f}")
    print(f"Profit: ${profit:,.0f}")
    print(f"ROI: {roi:+.1%}")
    print(f"Breakeven: ${strike+entry_premium}")
    print(f"Effective Leverage: {(contracts * 100 * spot_at_entry) / cost:.1f}x")


if __name__ ==  "__main__": # Test the calculator
    validate_successful_trade()

    # Example usage
    calculator = OptionsTradeCalculator()

    trade = calculator.calculate_trade(
        ticker = "GOOGL",
        spot_price = 207.00,
        account_size = 500_000,
        implied_volatility = 0.28,
        risk_pct = 0.10
    )

    print(trade)

    # Scenario analysis
    scenarios = calculator.scenario_analysis(
        trade,
        spot_moves = [-0.05, -0.03, 0.00, 0.03, 0.05, 0.08],
        implied_volatility = 0.28
    )

    print("\n=== SCENARIO ANALYSIS ===")
    for scenario in scenarios: 
        print(scenario)
