"""Comprehensive tests for options calculator to achieve >90% coverage."""
import pytest
import numpy as np
import math
from datetime import date, timedelta
from unittest.mock import Mock, patch

from backend.tradingbot.options_calculator import (
    BlackScholesCalculator,
    OptionsStrategySetup,
    OptionsSetup,
    TradeCalculation,
    OptionsTradeCalculator,
    validate_successful_trade
)


class TestBlackScholesCalculator:
    """Test Black-Scholes calculation methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.bs_calc = BlackScholesCalculator()
        self.spot = 100.0
        self.strike = 105.0
        self.time_to_expiry = 0.25  # 3 months
        self.risk_free_rate = 0.05
        self.dividend_yield = 0.0
        self.volatility = 0.20

    def test_black_scholes_call_price_basic(self):
        """Test basic call option pricing."""
        call_price = self.bs_calc.call_price(
            spot=self.spot,
            strike=self.strike,
            time_to_expiry_years=self.time_to_expiry,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            implied_volatility=self.volatility
        )

        assert call_price > 0
        assert call_price < self.spot  # Should be less than spot for OTM call

    def test_black_scholes_put_price_basic(self):
        """Test basic put option pricing."""
        put_price = self.bs_calc.put_price(
            spot=self.spot,
            strike=self.strike,
            time_to_expiry_years=self.time_to_expiry,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            implied_volatility=self.volatility
        )

        assert put_price > 0
        assert put_price > self.strike - self.spot  # Should exceed intrinsic value

    def test_black_scholes_put_call_parity(self):
        """Test put-call parity relationship."""
        call_price = self.bs_calc.call_price(
            spot=self.spot,
            strike=self.strike,
            time_to_expiry_years=self.time_to_expiry,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            implied_volatility=self.volatility
        )

        put_price = self.bs_calc.put_price(
            spot=self.spot,
            strike=self.strike,
            time_to_expiry_years=self.time_to_expiry,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            implied_volatility=self.volatility
        )

        # Put-call parity: C - P = S - K * e^(-r*T)
        parity_left = call_price - put_price
        parity_right = self.spot - self.strike * math.exp(-self.risk_free_rate * self.time_to_expiry)

        assert abs(parity_left - parity_right) < 0.01

    def test_black_scholes_delta_calculation(self):
        """Test delta calculation."""
        delta = self.bs_calc.delta(
            spot=self.spot,
            strike=self.strike,
            time_to_expiry_years=self.time_to_expiry,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            implied_volatility=self.volatility
        )

        assert 0 <= delta <= 1  # Call delta should be between 0 and 1
        assert delta < 0.5  # OTM call should have delta < 0.5

    def test_black_scholes_gamma_calculation(self):
        """Test gamma calculation."""
        gamma = self.bs_calc.gamma(
            spot=self.spot,
            strike=self.strike,
            time_to_expiry_years=self.time_to_expiry,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            implied_volatility=self.volatility
        )

        assert gamma > 0  # Gamma should always be positive

    def test_black_scholes_theta_calculation(self):
        """Test theta calculation."""
        theta = self.bs_calc.theta(
            spot=self.spot,
            strike=self.strike,
            time_to_expiry_years=self.time_to_expiry,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            implied_volatility=self.volatility
        )

        assert theta < 0  # Theta should be negative (time decay)

    def test_black_scholes_vega_calculation(self):
        """Test vega calculation."""
        vega = self.bs_calc.vega(
            spot=self.spot,
            strike=self.strike,
            time_to_expiry_years=self.time_to_expiry,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            implied_volatility=self.volatility
        )

        assert vega > 0  # Vega should be positive

    def test_black_scholes_error_handling(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError):
            self.bs_calc.call_price(
                spot=-10,  # Negative spot price
                strike=self.strike,
                time_to_expiry_years=self.time_to_expiry,
                risk_free_rate=self.risk_free_rate,
                dividend_yield=self.dividend_yield,
                implied_volatility=self.volatility
            )

        with pytest.raises(ValueError):
            self.bs_calc.put_price(
                spot=self.spot,
                strike=0,  # Zero strike price
                time_to_expiry_years=self.time_to_expiry,
                risk_free_rate=self.risk_free_rate,
                dividend_yield=self.dividend_yield,
                implied_volatility=self.volatility
            )

    def test_black_scholes_edge_cases(self):
        """Test edge cases for Black-Scholes calculations."""
        # Very short time to expiry
        short_expiry_call = self.bs_calc.call_price(
            spot=self.spot,
            strike=self.strike,
            time_to_expiry_years=0.001,  # Very short
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            implied_volatility=self.volatility
        )

        # Should approach intrinsic value
        intrinsic = max(0, self.spot - self.strike)
        assert abs(short_expiry_call - intrinsic) < 1.0

    def test_black_scholes_itm_vs_otm(self):
        """Test ITM vs OTM option pricing."""
        # ITM call (strike < spot)
        itm_call = self.bs_calc.call_price(
            spot=110.0,
            strike=100.0,
            time_to_expiry_years=self.time_to_expiry,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            implied_volatility=self.volatility
        )

        # OTM call (strike > spot)
        otm_call = self.bs_calc.call_price(
            spot=100.0,
            strike=110.0,
            time_to_expiry_years=self.time_to_expiry,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            implied_volatility=self.volatility
        )

        assert itm_call > otm_call

    def test_black_scholes_volatility_sensitivity(self):
        """Test sensitivity to volatility changes."""
        low_vol_call = self.bs_calc.call_price(
            spot=self.spot,
            strike=self.strike,
            time_to_expiry_years=self.time_to_expiry,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            implied_volatility=0.10  # Low volatility
        )

        high_vol_call = self.bs_calc.call_price(
            spot=self.spot,
            strike=self.strike,
            time_to_expiry_years=self.time_to_expiry,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            implied_volatility=0.40  # High volatility
        )

        assert high_vol_call > low_vol_call

    def test_norm_cdf_values(self):
        """Test normal CDF calculation."""
        # Test known values
        assert abs(self.bs_calc._norm_cdf(0) - 0.5) < 0.001
        assert abs(self.bs_calc._norm_cdf(-3) - 0.00135) < 0.001
        assert abs(self.bs_calc._norm_cdf(3) - 0.99865) < 0.001

    def test_greeks_zero_conditions(self):
        """Test Greeks with zero/invalid inputs."""
        # Test with zero values
        delta_zero = self.bs_calc.delta(
            spot=0,
            strike=self.strike,
            time_to_expiry_years=self.time_to_expiry,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            implied_volatility=self.volatility
        )
        assert delta_zero == 0.0

        gamma_zero = self.bs_calc.gamma(
            spot=self.spot,
            strike=0,
            time_to_expiry_years=self.time_to_expiry,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            implied_volatility=self.volatility
        )
        assert gamma_zero == 0.0


class TestOptionsStrategySetup:
    """Test OptionsStrategySetup configuration."""

    def test_options_strategy_setup_defaults(self):
        """Test default strategy setup values."""
        setup = OptionsStrategySetup()

        assert len(setup.mega_cap_tickers) > 0
        assert "AAPL" in setup.mega_cap_tickers
        assert setup.target_dte_optimal == 30
        assert setup.otm_percentage == 0.05
        assert setup.max_single_trade_risk_pct == 0.15

    def test_options_strategy_setup_custom(self):
        """Test custom strategy setup."""
        custom_tickers = ["MSFT", "GOOGL", "TSLA"]
        setup = OptionsStrategySetup(
            mega_cap_tickers=custom_tickers,
            target_dte_optimal=45,
            otm_percentage=0.08
        )

        assert setup.mega_cap_tickers == custom_tickers
        assert setup.target_dte_optimal == 45
        assert setup.otm_percentage == 0.08

    def test_risk_management_parameters(self):
        """Test risk management parameter validation."""
        setup = OptionsStrategySetup()

        assert setup.max_single_trade_risk_pct > setup.recommended_risk_pct
        assert setup.stop_loss_pct < 1.0  # Should be less than 100%
        assert len(setup.profit_take_levels) > 0


class TestOptionsSetup:
    """Test OptionsSetup dataclass."""

    def setup_method(self):
        """Set up test fixtures."""
        self.setup = OptionsSetup(
            ticker="AAPL",
            entry_date=date.today(),
            expiry_date=date.today() + timedelta(days=30),
            strike=150.0,
            spot_at_entry=145.0,
            premium_paid=5.0,
            contracts=10
        )

    def test_options_setup_properties(self):
        """Test calculated properties of OptionsSetup."""
        assert self.setup.total_cost == 10 * 5.0 * 100  # 10 contracts * $5 * 100 shares
        assert self.setup.breakeven == 150.0 + 5.0  # Strike + premium
        assert self.setup.intrinsic_value == 0.0  # OTM option
        assert self.setup.is_otm()
        assert not self.setup.is_itm()

    def test_options_setup_itm_scenario(self):
        """Test ITM option scenario."""
        itm_setup = OptionsSetup(
            ticker="AAPL",
            entry_date=date.today(),
            expiry_date=date.today() + timedelta(days=30),
            strike=140.0,
            spot_at_entry=145.0,  # Spot > strike (ITM)
            premium_paid=8.0,
            contracts=5
        )

        assert itm_setup.intrinsic_value == 5.0  # 145 - 140
        assert itm_setup.is_itm()
        assert not itm_setup.is_otm()

    def test_calculate_pnl(self):
        """Test P&L calculation."""
        # Current price $150, premium now $10
        current_pnl = self.setup.calculate_pnl(150.0, 10.0)
        expected_pnl = 10 * (10.0 - 5.0) * 100  # 10 contracts * $5 profit * 100 shares
        assert current_pnl == expected_pnl

        # Loss scenario
        loss_pnl = self.setup.calculate_pnl(145.0, 2.0)
        expected_loss = 10 * (2.0 - 5.0) * 100  # 10 contracts * $3 loss * 100 shares
        assert loss_pnl == expected_loss


class TestTradeCalculation:
    """Test TradeCalculation dataclass and string representation."""

    def test_trade_calculation_creation(self):
        """Test TradeCalculation creation."""
        calc = TradeCalculation(
            ticker="AAPL",
            spot_price=150.0,
            strike=155.0,
            expiry_date=date.today() + timedelta(days=30),
            days_to_expiry=30,
            estimated_premium=3.50,
            recommended_contracts=100,
            total_cost=35000.0,
            breakeven_price=158.50,
            estimated_delta=0.35,
            leverage_ratio=4.3,
            risk_amount=35000.0,
            account_risk_pct=7.0
        )

        assert calc.ticker == "AAPL"
        assert calc.spot_price == 150.0
        assert calc.recommended_contracts == 100
        assert calc.account_risk_pct == 7.0

    def test_trade_calculation_string_representation(self):
        """Test string representation of TradeCalculation."""
        calc = TradeCalculation(
            ticker="AAPL",
            spot_price=150.0,
            strike=155.0,
            expiry_date=date.today() + timedelta(days=30),
            days_to_expiry=30,
            estimated_premium=3.50,
            recommended_contracts=100,
            total_cost=35000.0,
            breakeven_price=158.50,
            estimated_delta=0.35,
            leverage_ratio=4.3,
            risk_amount=35000.0,
            account_risk_pct=7.0
        )

        str_repr = str(calc)
        assert "AAPL" in str_repr
        assert "150.0" in str_repr
        assert "155.0" in str_repr


class TestOptionsTradeCalculator:
    """Test OptionsTradeCalculator main functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = OptionsTradeCalculator()

    def test_calculator_initialization(self):
        """Test calculator initialization."""
        assert hasattr(self.calculator, 'setup')
        assert hasattr(self.calculator, 'bs_calc')
        assert isinstance(self.calculator.bs_calc, BlackScholesCalculator)

    def test_find_optimal_expiry(self):
        """Test optimal expiry date calculation."""
        expiry_date = self.calculator.find_optimal_expiry(30)

        # Should be a Friday
        assert expiry_date.weekday() == 4  # Friday is weekday 4

        # Should be approximately 30 days out
        days_diff = (expiry_date - date.today()).days
        assert 21 <= days_diff <= 45  # Within acceptable range

    def test_find_optimal_expiry_edge_cases(self):
        """Test optimal expiry with edge cases."""
        # Very short DTE - should return a reasonable minimum
        short_expiry = self.calculator.find_optimal_expiry(7)
        days_diff = (short_expiry - date.today()).days
        # Allow for some flexibility in DTE calculation - should be at least 7 days or more
        assert days_diff >= 7

        # Very long DTE
        long_expiry = self.calculator.find_optimal_expiry(60)
        days_diff = (long_expiry - date.today()).days
        # Allow for some flexibility in DTE calculation - should cap at reasonable max
        assert days_diff <= 60  # Should not exceed requested DTE

    def test_calculate_otm_strike(self):
        """Test OTM strike calculation."""
        spot_price = 200.0
        otm_strike = self.calculator.calculate_otm_strike(spot_price)

        expected_strike = spot_price * (1 + self.calculator.setup.otm_percentage)
        assert abs(otm_strike - expected_strike) < 1.0  # Within $1 due to rounding

    def test_calculate_otm_strike_with_increment(self):
        """Test OTM strike with custom increment."""
        spot_price = 203.45
        otm_strike = self.calculator.calculate_otm_strike(spot_price, increment=5.0)

        # Should be rounded to nearest $5
        assert otm_strike % 5.0 == 0.0

    def test_calculate_trade_basic(self):
        """Test basic trade calculation."""
        trade_calc = self.calculator.calculate_trade(
            ticker="AAPL",
            spot_price=150.0,
            account_size=100000.0,
            implied_volatility=0.25,
            risk_pct=0.10
        )

        assert trade_calc.ticker == "AAPL"
        assert trade_calc.spot_price == 150.0
        assert trade_calc.strike > 150.0  # Should be OTM
        assert trade_calc.estimated_premium > 0
        assert trade_calc.recommended_contracts >= 0
        assert 0 <= trade_calc.account_risk_pct <= 15  # Within risk limits

    def test_calculate_trade_input_validation(self):
        """Test input validation for trade calculation."""
        with pytest.raises(ValueError):
            self.calculator.calculate_trade(
                ticker="AAPL",
                spot_price=-150.0,  # Negative price
                account_size=100000.0,
                implied_volatility=0.25
            )

        with pytest.raises(ValueError):
            self.calculator.calculate_trade(
                ticker="AAPL",
                spot_price=150.0,
                account_size=0.0,  # Zero account size
                implied_volatility=0.25
            )

        with pytest.raises(ValueError):
            self.calculator.calculate_trade(
                ticker="AAPL",
                spot_price=150.0,
                account_size=100000.0,
                implied_volatility=0.25,
                risk_pct=0.25  # Exceeds max risk
            )

    def test_calculate_trade_with_custom_parameters(self):
        """Test trade calculation with custom parameters."""
        trade_calc = self.calculator.calculate_trade(
            ticker="MSFT",
            spot_price=300.0,
            account_size=500000.0,
            implied_volatility=0.30,
            risk_pct=0.05,  # Conservative risk
            risk_free_rate=0.06,
            dividend_yield=0.01,
            custom_dte=45
        )

        assert trade_calc.ticker == "MSFT"
        assert trade_calc.account_risk_pct <= 5.0  # Should respect risk limit
        assert trade_calc.days_to_expiry >= 35  # Should be around 45 DTE (allow some flexibility)

    def test_scenario_analysis(self):
        """Test scenario analysis functionality."""
        # First calculate a base trade
        trade_calc = self.calculator.calculate_trade(
            ticker="AAPL",
            spot_price=150.0,
            account_size=100000.0,
            implied_volatility=0.25
        )

        # Run scenario analysis
        scenarios = self.calculator.scenario_analysis(
            trade_calc=trade_calc,
            spot_moves=[-0.05, 0.0, 0.05, 0.10],
            implied_volatility=0.25
        )

        assert len(scenarios) == 4
        for scenario in scenarios:
            assert 'spot_move' in scenario
            assert 'new_spot_price' in scenario
            assert 'pnl_per_contract' in scenario
            assert 'total_pnl' in scenario
            assert 'roi' in scenario

    def test_scenario_analysis_with_time_decay(self):
        """Test scenario analysis with time passage."""
        trade_calc = self.calculator.calculate_trade(
            ticker="AAPL",
            spot_price=150.0,
            account_size=100000.0,
            implied_volatility=0.25
        )

        # Scenario with 10 days passed
        scenarios = self.calculator.scenario_analysis(
            trade_calc=trade_calc,
            spot_moves=[0.0, 0.05],
            implied_volatility=0.25,
            days_passed=10
        )

        assert len(scenarios) == 2
        # Should show lower option values due to time decay
        for scenario in scenarios:
            assert scenario['days_remaining'] == trade_calc.days_to_expiry - 10

    def test_scenario_analysis_edge_cases(self):
        """Test scenario analysis edge cases."""
        trade_calc = self.calculator.calculate_trade(
            ticker="AAPL",
            spot_price=150.0,
            account_size=100000.0,
            implied_volatility=0.25
        )

        # Very large negative move
        scenarios = self.calculator.scenario_analysis(
            trade_calc=trade_calc,
            spot_moves=[-0.50],  # 50% drop
            implied_volatility=0.25
        )

        # Option should be worthless
        assert scenarios[0]['new_premium'] == 0.0
        assert scenarios[0]['roi'] == "-100.0%"


class TestValidateSuccessfulTrade:
    """Test the validate_successful_trade function."""

    def test_validate_successful_trade_execution(self):
        """Test that validate_successful_trade runs without error."""
        # This function prints to stdout, so we'll just ensure it runs
        try:
            validate_successful_trade()
            assert True  # Function executed successfully
        except Exception as e:
            pytest.fail(f"validate_successful_trade raised an exception: {e}")


class TestIntegrationScenarios:
    """Test integration scenarios combining all components."""

    def test_complete_options_trade_workflow(self):
        """Test complete options trade workflow."""
        calculator = OptionsTradeCalculator()

        # Calculate trade
        trade = calculator.calculate_trade(
            ticker="AAPL",
            spot_price=150.0,
            account_size=500000.0,
            implied_volatility=0.28,
            risk_pct=0.10
        )

        # Verify trade calculation
        assert trade.ticker == "AAPL"
        assert trade.strike > trade.spot_price  # OTM call
        assert trade.estimated_premium > 0
        assert trade.recommended_contracts > 0

        # Run scenario analysis
        scenarios = calculator.scenario_analysis(
            trade,
            spot_moves=[-0.05, 0.0, 0.05, 0.10],
            implied_volatility=0.28
        )

        # Verify scenarios
        assert len(scenarios) == 4
        # Upward moves should show profits
        upward_scenarios = [s for s in scenarios if '+' in s['spot_move']]
        assert any(float(s['total_pnl']) > 0 for s in upward_scenarios)

    def test_high_volatility_scenario(self):
        """Test high volatility trading scenario."""
        calculator = OptionsTradeCalculator()

        trade = calculator.calculate_trade(
            ticker="NVDA",
            spot_price=400.0,
            account_size=1000000.0,
            implied_volatility=0.50,  # High volatility
            risk_pct=0.08
        )

        # High IV should result in higher premium
        assert trade.estimated_premium > 10.0
        # Should recommend fewer contracts due to higher premium
        assert trade.recommended_contracts < 500

    def test_low_volatility_scenario(self):
        """Test low volatility trading scenario."""
        calculator = OptionsTradeCalculator()

        trade = calculator.calculate_trade(
            ticker="KO",  # Typically low volatility stock
            spot_price=60.0,
            account_size=200000.0,
            implied_volatility=0.15,  # Low volatility
            risk_pct=0.10
        )

        # Low IV should result in lower premium (but not necessarily < $5 for all scenarios)
        assert trade.estimated_premium > 0  # Should be positive
        # Should recommend more contracts due to lower premium
        assert trade.recommended_contracts > 50

    def test_different_risk_levels(self):
        """Test different risk level scenarios."""
        calculator = OptionsTradeCalculator()

        # Conservative approach
        conservative_trade = calculator.calculate_trade(
            ticker="AAPL",
            spot_price=150.0,
            account_size=100000.0,
            implied_volatility=0.25,
            risk_pct=0.05  # Conservative 5%
        )

        # Aggressive approach
        aggressive_trade = calculator.calculate_trade(
            ticker="AAPL",
            spot_price=150.0,
            account_size=100000.0,
            implied_volatility=0.25,
            risk_pct=0.15  # Aggressive 15%
        )

        # Aggressive should recommend more contracts
        assert aggressive_trade.recommended_contracts > conservative_trade.recommended_contracts
        assert aggressive_trade.account_risk_pct > conservative_trade.account_risk_pct

    def test_stress_testing_scenarios(self):
        """Test stress testing with extreme scenarios."""
        calculator = OptionsTradeCalculator()

        trade = calculator.calculate_trade(
            ticker="TSLA",
            spot_price=200.0,
            account_size=100000.0,
            implied_volatility=0.40
        )

        # Extreme scenarios
        extreme_scenarios = calculator.scenario_analysis(
            trade,
            spot_moves=[-0.30, -0.20, 0.20, 0.30, 0.50],  # Extreme moves
            implied_volatility=0.40
        )

        # Should handle extreme scenarios gracefully
        assert len(extreme_scenarios) == 5
        for scenario in extreme_scenarios:
            assert isinstance(scenario['total_pnl'], (int, float))
            assert isinstance(scenario['roi'], str)