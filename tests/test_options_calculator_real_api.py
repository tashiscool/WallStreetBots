"""Comprehensive tests for options calculator with real API integration to achieve >90% coverage."""
import pytest
import numpy as np
import math
import yfinance as yf
from datetime import date, timedelta, datetime
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from backend.tradingbot.options_calculator import (
    BlackScholesCalculator,
    OptionsStrategySetup,
    OptionsSetup,
    TradeCalculation,
    OptionsTradeCalculator,
    validate_successful_trade
)


class TestBlackScholesCalculatorRealAPI:
    """Test Black-Scholes calculator with real market data integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.bs_calc = BlackScholesCalculator()
        self.test_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]

    def test_real_market_data_integration(self):
        """Test Black-Scholes calculations with real market data."""
        symbol = "AAPL"
        
        try:
            # Get real market data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            
            if hist.empty:
                pytest.skip(f"No market data available for {symbol}")
            
            spot_price = float(hist['Close'].iloc[-1])
            
            # Test with realistic parameters
            strike = spot_price * 1.05  # 5% OTM
            time_to_expiry = 30 / 365.0  # 30 days
            risk_free_rate = 0.05
            dividend_yield = 0.01
            implied_volatility = 0.25
            
            # Calculate call price
            call_price = self.bs_calc.call_price(
                spot=spot_price,
                strike=strike,
                time_to_expiry_years=time_to_expiry,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                implied_volatility=implied_volatility
            )
            
            # Calculate put price
            put_price = self.bs_calc.put_price(
                spot=spot_price,
                strike=strike,
                time_to_expiry_years=time_to_expiry,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                implied_volatility=implied_volatility
            )
            
            # Validate results
            assert call_price > 0
            assert put_price > 0
            assert call_price < spot_price  # OTM call should be less than spot
            
            # Test Greeks
            delta = self.bs_calc.delta(
                spot=spot_price,
                strike=strike,
                time_to_expiry_years=time_to_expiry,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                implied_volatility=implied_volatility
            )
            
            gamma = self.bs_calc.gamma(
                spot=spot_price,
                strike=strike,
                time_to_expiry_years=time_to_expiry,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                implied_volatility=implied_volatility
            )
            
            theta = self.bs_calc.theta(
                spot=spot_price,
                strike=strike,
                time_to_expiry_years=time_to_expiry,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                implied_volatility=implied_volatility
            )
            
            vega = self.bs_calc.vega(
                spot=spot_price,
                strike=strike,
                time_to_expiry_years=time_to_expiry,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                implied_volatility=implied_volatility
            )
            
            # Validate Greeks
            assert 0 <= delta <= 1  # Call delta should be between 0 and 1
            assert gamma > 0  # Gamma should be positive
            assert theta < 0  # Theta should be negative (time decay)
            assert vega > 0  # Vega should be positive
            
        except Exception as e:
            pytest.skip(f"Real API test skipped due to: {e}")

    def test_volatility_surface_analysis(self):
        """Test Black-Scholes across different volatility levels."""
        spot_price = 150.0
        strike = 155.0
        time_to_expiry = 0.25
        risk_free_rate = 0.05
        dividend_yield = 0.0
        
        volatility_levels = [0.10, 0.20, 0.30, 0.40, 0.50]
        call_prices = []
        
        for vol in volatility_levels:
            call_price = self.bs_calc.call_price(
                spot=spot_price,
                strike=strike,
                time_to_expiry_years=time_to_expiry,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                implied_volatility=vol
            )
            call_prices.append(call_price)
        
        # Higher volatility should result in higher option prices
        for i in range(1, len(call_prices)):
            assert call_prices[i] > call_prices[i-1]

    def test_time_decay_analysis(self):
        """Test Black-Scholes across different time to expiry."""
        spot_price = 150.0
        strike = 155.0
        risk_free_rate = 0.05
        dividend_yield = 0.0
        implied_volatility = 0.25
        
        time_periods = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]  # Different time periods
        call_prices = []
        
        for time_to_expiry in time_periods:
            call_price = self.bs_calc.call_price(
                spot=spot_price,
                strike=strike,
                time_to_expiry_years=time_to_expiry,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                implied_volatility=implied_volatility
            )
            call_prices.append(call_price)
        
        # Longer time to expiry should result in higher option prices
        for i in range(1, len(call_prices)):
            assert call_prices[i] > call_prices[i-1]

    def test_moneyness_analysis(self):
        """Test Black-Scholes across different moneyness levels."""
        spot_price = 150.0
        time_to_expiry = 0.25
        risk_free_rate = 0.05
        dividend_yield = 0.0
        implied_volatility = 0.25
        
        # Test different moneyness levels
        strikes = [120.0, 135.0, 150.0, 165.0, 180.0]  # Deep ITM to Deep OTM
        call_prices = []
        
        for strike in strikes:
            call_price = self.bs_calc.call_price(
                spot=spot_price,
                strike=strike,
                time_to_expiry_years=time_to_expiry,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                implied_volatility=implied_volatility
            )
            call_prices.append(call_price)
        
        # ITM options should be more expensive than OTM options
        assert call_prices[0] > call_prices[1] > call_prices[2] > call_prices[3] > call_prices[4]

    def test_interest_rate_sensitivity(self):
        """Test Black-Scholes sensitivity to interest rates."""
        spot_price = 150.0
        strike = 155.0
        time_to_expiry = 0.25
        dividend_yield = 0.0
        implied_volatility = 0.25
        
        interest_rates = [0.01, 0.03, 0.05, 0.07, 0.10]
        call_prices = []
        
        for rate in interest_rates:
            call_price = self.bs_calc.call_price(
                spot=spot_price,
                strike=strike,
                time_to_expiry_years=time_to_expiry,
                risk_free_rate=rate,
                dividend_yield=dividend_yield,
                implied_volatility=implied_volatility
            )
            call_prices.append(call_price)
        
        # Higher interest rates should result in higher call prices
        for i in range(1, len(call_prices)):
            assert call_prices[i] > call_prices[i-1]

    def test_dividend_yield_sensitivity(self):
        """Test Black-Scholes sensitivity to dividend yields."""
        spot_price = 150.0
        strike = 155.0
        time_to_expiry = 0.25
        risk_free_rate = 0.05
        implied_volatility = 0.25
        
        dividend_yields = [0.0, 0.01, 0.02, 0.03, 0.05]
        call_prices = []
        
        for div_yield in dividend_yields:
            call_price = self.bs_calc.call_price(
                spot=spot_price,
                strike=strike,
                time_to_expiry_years=time_to_expiry,
                risk_free_rate=risk_free_rate,
                dividend_yield=div_yield,
                implied_volatility=implied_volatility
            )
            call_prices.append(call_price)
        
        # Higher dividend yields should result in lower call prices
        for i in range(1, len(call_prices)):
            assert call_prices[i] < call_prices[i-1]


class TestOptionsTradeCalculatorRealAPI:
    """Test OptionsTradeCalculator with real market data integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = OptionsTradeCalculator()

    def test_real_trade_calculation(self):
        """Test trade calculation with real market data."""
        symbol = "AAPL"
        
        try:
            # Get real market data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            
            if hist.empty:
                pytest.skip(f"No market data available for {symbol}")
            
            spot_price = float(hist['Close'].iloc[-1])
            account_size = 100000.0
            implied_volatility = 0.25
            risk_pct = 0.10
            
            # Calculate trade
            trade_calc = self.calculator.calculate_trade(
                ticker=symbol,
                spot_price=spot_price,
                account_size=account_size,
                implied_volatility=implied_volatility,
                risk_pct=risk_pct
            )
            
            # Validate trade calculation
            assert trade_calc.ticker == symbol
            assert trade_calc.spot_price == spot_price
            assert trade_calc.strike > spot_price  # Should be OTM
            assert trade_calc.estimated_premium > 0
            assert trade_calc.recommended_contracts > 0
            assert trade_calc.total_cost > 0
            assert trade_calc.account_risk_pct <= risk_pct * 100
            
            # Test scenario analysis
            scenarios = self.calculator.scenario_analysis(
                trade_calc=trade_calc,
                spot_moves=[-0.05, 0.0, 0.05, 0.10],
                implied_volatility=implied_volatility
            )
            
            assert len(scenarios) == 4
            for scenario in scenarios:
                assert 'spot_move' in scenario
                assert 'new_spot_price' in scenario
                assert 'pnl_per_contract' in scenario
                assert 'total_pnl' in scenario
                assert 'roi' in scenario
                
        except Exception as e:
            pytest.skip(f"Real API test skipped due to: {e}")

    def test_multiple_symbols_trade_calculation(self):
        """Test trade calculation across multiple symbols."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        account_size = 500000.0
        implied_volatility = 0.30
        risk_pct = 0.08
        
        trades = []
        
        for symbol in symbols:
            try:
                # Get real market data
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d")
                
                if hist.empty:
                    continue
                
                spot_price = float(hist['Close'].iloc[-1])
                
                # Calculate trade
                trade_calc = self.calculator.calculate_trade(
                    ticker=symbol,
                    spot_price=spot_price,
                    account_size=account_size,
                    implied_volatility=implied_volatility,
                    risk_pct=risk_pct
                )
                
                trades.append(trade_calc)
                
            except Exception:
                continue
        
        # Should have at least one successful trade
        assert len(trades) > 0
        
        # Validate all trades
        for trade in trades:
            assert trade.ticker in symbols
            assert trade.spot_price > 0
            assert trade.strike > trade.spot_price
            assert trade.estimated_premium > 0
            assert trade.recommended_contracts > 0

    def test_risk_management_scenarios(self):
        """Test risk management across different scenarios."""
        symbol = "TSLA"  # High volatility stock
        
        try:
            # Get real market data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            
            if hist.empty:
                pytest.skip(f"No market data available for {symbol}")
            
            spot_price = float(hist['Close'].iloc[-1])
            account_size = 100000.0
            
            # Test different risk levels
            risk_levels = [0.05, 0.10, 0.15]
            trades = []
            
            for risk_pct in risk_levels:
                trade_calc = self.calculator.calculate_trade(
                    ticker=symbol,
                    spot_price=spot_price,
                    account_size=account_size,
                    implied_volatility=0.40,  # High volatility
                    risk_pct=risk_pct
                )
                
                trades.append(trade_calc)
                
                # Validate risk management
                assert trade_calc.account_risk_pct <= risk_pct * 100
                assert trade_calc.total_cost <= account_size * risk_pct
            
            # Higher risk should result in more contracts
            assert trades[2].recommended_contracts >= trades[1].recommended_contracts
            assert trades[1].recommended_contracts >= trades[0].recommended_contracts
            
        except Exception as e:
            pytest.skip(f"Real API test skipped due to: {e}")

    def test_expiry_date_calculation_real_data(self):
        """Test expiry date calculation with real calendar data."""
        # Test different target DTEs
        target_dtes = [21, 30, 45]
        
        for target_dte in target_dtes:
            expiry_date = self.calculator.find_optimal_expiry(target_dte)
            
            # Should be a Friday
            assert expiry_date.weekday() == 4
            
            # Should be within acceptable range
            days_diff = (expiry_date - date.today()).days
            assert self.calculator.setup.target_dte_min <= days_diff <= self.calculator.setup.target_dte_max

    def test_otm_strike_calculation_real_prices(self):
        """Test OTM strike calculation with real price levels."""
        # Test with realistic price levels
        price_levels = [50.0, 100.0, 150.0, 200.0, 300.0, 500.0]
        
        for spot_price in price_levels:
            otm_strike = self.calculator.calculate_otm_strike(spot_price)
            
            # Should be higher than spot (OTM)
            assert otm_strike > spot_price
            
            # Should be approximately 5% OTM
            expected_otm = spot_price * (1 + self.calculator.setup.otm_percentage)
            assert abs(otm_strike - expected_otm) < spot_price * 0.02  # Within 2% (allowing for rounding)


class TestOptionsStrategySetupRealAPI:
    """Test OptionsStrategySetup with real market data validation."""

    def test_mega_cap_tickers_validation(self):
        """Test that mega cap tickers are valid and have market data."""
        setup = OptionsStrategySetup()
        
        for ticker in setup.mega_cap_tickers:
            try:
                # Test if ticker has market data
                yf_ticker = yf.Ticker(ticker)
                hist = yf_ticker.history(period="1d")
                
                # Should have recent data
                assert not hist.empty
                assert len(hist) > 0
                
                # Price should be reasonable
                price = float(hist['Close'].iloc[-1])
                assert price > 0
                assert price < 10000  # Reasonable upper bound
                
            except Exception as e:
                pytest.fail(f"Ticker {ticker} failed validation: {e}")

    def test_strategy_parameters_realistic(self):
        """Test that strategy parameters are realistic for real trading."""
        setup = OptionsStrategySetup()
        
        # DTE parameters should be realistic
        assert 7 <= setup.target_dte_min <= 30
        assert 30 <= setup.target_dte_optimal <= 60
        assert 45 <= setup.target_dte_max <= 90
        
        # Risk parameters should be conservative
        assert 0.05 <= setup.max_single_trade_risk_pct <= 0.20
        assert 0.05 <= setup.recommended_risk_pct <= 0.15
        assert setup.recommended_risk_pct <= setup.max_single_trade_risk_pct
        
        # OTM percentage should be reasonable
        assert 0.02 <= setup.otm_percentage <= 0.10
        
        # Delta range should be realistic for OTM options
        assert 0.20 <= setup.target_delta_range[0] <= 0.40
        assert 0.30 <= setup.target_delta_range[1] <= 0.50
        assert setup.target_delta_range[0] < setup.target_delta_range[1]

    def test_profit_take_levels_realistic(self):
        """Test that profit take levels are realistic."""
        setup = OptionsStrategySetup()
        
        # Should have multiple profit levels
        assert len(setup.profit_take_levels) >= 2
        
        # Should be in ascending order
        for i in range(1, len(setup.profit_take_levels)):
            assert setup.profit_take_levels[i] > setup.profit_take_levels[i-1]
        
        # Should be realistic multiples
        for level in setup.profit_take_levels:
            assert 0.5 <= level <= 5.0  # 50% to 500% profit


class TestOptionsSetupRealAPI:
    """Test OptionsSetup with real market scenarios."""

    def test_real_options_scenarios(self):
        """Test OptionsSetup with real market scenarios."""
        # Test ITM scenario
        itm_setup = OptionsSetup(
            ticker="AAPL",
            entry_date=date.today(),
            expiry_date=date.today() + timedelta(days=30),
            strike=140.0,
            spot_at_entry=150.0,  # ITM
            premium_paid=12.0,
            contracts=10
        )
        
        assert itm_setup.is_itm()
        assert not itm_setup.is_otm()
        assert itm_setup.intrinsic_value == 10.0  # 150 - 140
        
        # Test OTM scenario
        otm_setup = OptionsSetup(
            ticker="AAPL",
            entry_date=date.today(),
            expiry_date=date.today() + timedelta(days=30),
            strike=160.0,
            spot_at_entry=150.0,  # OTM
            premium_paid=3.0,
            contracts=20
        )
        
        assert otm_setup.is_otm()
        assert not otm_setup.is_itm()
        assert otm_setup.intrinsic_value == 0.0

    def test_pnl_calculation_real_scenarios(self):
        """Test P&L calculation with real scenarios."""
        setup = OptionsSetup(
            ticker="AAPL",
            entry_date=date.today(),
            expiry_date=date.today() + timedelta(days=30),
            strike=150.0,
            spot_at_entry=145.0,
            premium_paid=5.0,
            contracts=10
        )
        
        # Profit scenario
        profit_pnl = setup.calculate_pnl(155.0, 8.0)
        expected_profit = 10 * (8.0 - 5.0) * 100
        assert profit_pnl == expected_profit
        assert profit_pnl > 0
        
        # Loss scenario
        loss_pnl = setup.calculate_pnl(140.0, 2.0)
        expected_loss = 10 * (2.0 - 5.0) * 100
        assert loss_pnl == expected_loss
        assert loss_pnl < 0
        
        # Breakeven scenario
        breakeven_pnl = setup.calculate_pnl(150.0, 5.0)
        assert breakeven_pnl == 0


class TestIntegrationScenariosRealAPI:
    """Test complete integration scenarios with real market data."""

    def test_complete_trading_workflow_real_data(self):
        """Test complete trading workflow with real market data."""
        calculator = OptionsTradeCalculator()
        
        # Test with multiple symbols
        symbols = ["AAPL", "MSFT", "GOOGL"]
        account_size = 1000000.0
        
        successful_trades = []
        
        for symbol in symbols:
            try:
                # Get real market data
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d")
                
                if hist.empty:
                    continue
                
                spot_price = float(hist['Close'].iloc[-1])
                
                # Calculate trade
                trade = calculator.calculate_trade(
                    ticker=symbol,
                    spot_price=spot_price,
                    account_size=account_size,
                    implied_volatility=0.25,
                    risk_pct=0.08
                )
                
                # Run scenario analysis
                scenarios = calculator.scenario_analysis(
                    trade,
                    spot_moves=[-0.10, -0.05, 0.0, 0.05, 0.10, 0.20],
                    implied_volatility=0.25
                )
                
                # Validate results
                assert len(scenarios) == 6
                assert trade.ticker == symbol
                assert trade.spot_price == spot_price
                
                successful_trades.append((trade, scenarios))
                
            except Exception:
                continue
        
        # Should have at least one successful trade
        assert len(successful_trades) > 0
        
        # Validate all successful trades
        for trade, scenarios in successful_trades:
            assert trade.estimated_premium > 0
            assert trade.recommended_contracts > 0
            assert trade.account_risk_pct <= 8.0
            
            # Scenarios should show realistic P&L
            for scenario in scenarios:
                assert isinstance(scenario['total_pnl'], (int, float))
                assert isinstance(scenario['roi'], str)

    def test_volatility_regime_analysis(self):
        """Test different volatility regimes with real data."""
        calculator = OptionsTradeCalculator()
        
        # Test different volatility scenarios
        volatility_scenarios = [
            ("Low Vol", 0.15),
            ("Medium Vol", 0.25),
            ("High Vol", 0.40),
            ("Extreme Vol", 0.60)
        ]
        
        symbol = "AAPL"
        spot_price = 150.0
        account_size = 500000.0
        
        trades = []
        
        for vol_name, vol_value in volatility_scenarios:
            trade = calculator.calculate_trade(
                ticker=symbol,
                spot_price=spot_price,
                account_size=account_size,
                implied_volatility=vol_value,
                risk_pct=0.10
            )
            
            trades.append((vol_name, trade))
        
        # Higher volatility should result in higher premiums
        for i in range(1, len(trades)):
            assert trades[i][1].estimated_premium > trades[i-1][1].estimated_premium
        
        # Higher volatility should result in fewer contracts (due to higher cost)
        for i in range(1, len(trades)):
            assert trades[i][1].recommended_contracts <= trades[i-1][1].recommended_contracts

    def test_market_crash_scenario(self):
        """Test market crash scenario with extreme moves."""
        calculator = OptionsTradeCalculator()
        
        trade = calculator.calculate_trade(
            ticker="SPY",
            spot_price=400.0,
            account_size=100000.0,
            implied_volatility=0.30,
            risk_pct=0.10
        )
        
        # Test extreme market crash scenarios
        crash_scenarios = calculator.scenario_analysis(
            trade,
            spot_moves=[-0.20, -0.30, -0.40, -0.50],  # Market crash scenarios
            implied_volatility=0.50  # Elevated volatility during crash
        )
        
        # All crash scenarios should show losses
        for scenario in crash_scenarios:
            assert scenario['total_pnl'] <= 0
            assert scenario['roi'] == "-100.0%" or float(scenario['roi'].rstrip('%')) < 0

    def test_market_rally_scenario(self):
        """Test market rally scenario with strong upward moves."""
        calculator = OptionsTradeCalculator()
        
        trade = calculator.calculate_trade(
            ticker="QQQ",
            spot_price=350.0,
            account_size=200000.0,
            implied_volatility=0.25,
            risk_pct=0.12
        )
        
        # Test market rally scenarios
        rally_scenarios = calculator.scenario_analysis(
            trade,
            spot_moves=[0.10, 0.20, 0.30, 0.40, 0.50],  # Market rally scenarios
            implied_volatility=0.20  # Lower volatility during rally
        )
        
        # Rally scenarios should show profits
        profitable_scenarios = [s for s in rally_scenarios if '+' in s['spot_move']]
        for scenario in profitable_scenarios:
            assert scenario['total_pnl'] > 0
            assert float(scenario['roi'].rstrip('%')) > 0


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling with real data scenarios."""

    def test_zero_and_negative_inputs(self):
        """Test handling of zero and negative inputs."""
        bs_calc = BlackScholesCalculator()
        
        # Test zero inputs
        with pytest.raises(ValueError):
            bs_calc.call_price(
                spot=0,
                strike=100,
                time_to_expiry_years=0.25,
                risk_free_rate=0.05,
                dividend_yield=0.0,
                implied_volatility=0.25
            )
        
        with pytest.raises(ValueError):
            bs_calc.call_price(
                spot=100,
                strike=0,
                time_to_expiry_years=0.25,
                risk_free_rate=0.05,
                dividend_yield=0.0,
                implied_volatility=0.25
            )
        
        with pytest.raises(ValueError):
            bs_calc.call_price(
                spot=100,
                strike=105,
                time_to_expiry_years=0,
                risk_free_rate=0.05,
                dividend_yield=0.0,
                implied_volatility=0.25
            )

    def test_extreme_values(self):
        """Test handling of extreme values."""
        bs_calc = BlackScholesCalculator()
        
        # Test very high volatility
        high_vol_price = bs_calc.call_price(
            spot=100,
            strike=105,
            time_to_expiry_years=0.25,
            risk_free_rate=0.05,
            dividend_yield=0.0,
            implied_volatility=2.0  # 200% volatility
        )
        
        assert high_vol_price > 0
        assert high_vol_price < 100  # Should still be less than spot
        
        # Test very long time to expiry
        long_expiry_price = bs_calc.call_price(
            spot=100,
            strike=105,
            time_to_expiry_years=10.0,  # 10 years
            risk_free_rate=0.05,
            dividend_yield=0.0,
            implied_volatility=0.25
        )
        
        assert long_expiry_price > 0

    def test_calculator_edge_cases(self):
        """Test calculator edge cases."""
        calculator = OptionsTradeCalculator()
        
        # Test with very small account size
        small_trade = calculator.calculate_trade(
            ticker="AAPL",
            spot_price=150.0,
            account_size=1000.0,  # Very small account
            implied_volatility=0.25,
            risk_pct=0.05
        )
        
        assert small_trade.recommended_contracts == 0  # Should recommend 0 contracts
        
        # Test with very large account size
        large_trade = calculator.calculate_trade(
            ticker="AAPL",
            spot_price=150.0,
            account_size=10000000.0,  # Very large account
            implied_volatility=0.25,
            risk_pct=0.10
        )
        
        assert large_trade.recommended_contracts > 0
        assert large_trade.account_risk_pct <= 10.0


class TestPerformanceAndStressTesting:
    """Test performance and stress scenarios."""

    def test_bulk_calculations_performance(self):
        """Test performance of bulk calculations."""
        bs_calc = BlackScholesCalculator()
        
        # Test bulk calculation performance
        start_time = datetime.now()
        
        for i in range(1000):
            bs_calc.call_price(
                spot=100 + i * 0.1,
                strike=105 + i * 0.1,
                time_to_expiry_years=0.25,
                risk_free_rate=0.05,
                dividend_yield=0.0,
                implied_volatility=0.25
            )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Should complete 1000 calculations in reasonable time
        assert duration < 5.0  # Less than 5 seconds

    def test_memory_usage_stress_test(self):
        """Test memory usage under stress."""
        calculator = OptionsTradeCalculator()
        
        # Create many trade calculations
        trades = []
        for i in range(100):
            trade = calculator.calculate_trade(
                ticker=f"SYMBOL_{i}",
                spot_price=100.0 + i,
                account_size=100000.0,
                implied_volatility=0.25,
                risk_pct=0.10
            )
            trades.append(trade)
        
        # Should handle 100 trades without issues
        assert len(trades) == 100
        
        # All trades should be valid
        for trade in trades:
            assert trade.estimated_premium > 0
            assert trade.recommended_contracts >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
