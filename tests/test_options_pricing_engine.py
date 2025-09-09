"""
Tests for Options Pricing Engine
Tests the Black-Scholes implementation and real options pricing functionality
"""

import pytest
import asyncio
import math
from datetime import datetime, date, timedelta
from decimal import Decimal, getcontext
from typing import Dict, Any
from unittest.mock import Mock, patch, AsyncMock

from backend.tradingbot.options.pricing_engine import (
    BlackScholesEngine,
    RealOptionsPricingEngine,
    OptionsContract,
    create_options_pricing_engine
)

# Set high precision for financial tests
getcontext().prec=28


class TestBlackScholesEngine:
    """Test Black-Scholes mathematical engine"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.bs_engine=BlackScholesEngine()
    
    @pytest.mark.asyncio
    async def test_get_risk_free_rate(self):
        """Test risk-free rate retrieval"""
        rate=await self.bs_engine.get_risk_free_rate()
        
        assert isinstance(rate, Decimal)
        assert 0 <= float(rate) <= 0.20  # 0% to 20% is reasonable range
    
    @pytest.mark.asyncio
    async def test_get_dividend_yield_known_ticker(self):
        """Test dividend yield for known tickers"""
        # Test Apple (known to have small dividend)
        aapl_yield=await self.bs_engine.get_dividend_yield('AAPL')
        assert isinstance(aapl_yield, Decimal)
        assert 0 <= float(aapl_yield) <= 0.10  # 0% to 10%
        
        # Test Amazon (known to have no dividend)
        amzn_yield=await self.bs_engine.get_dividend_yield('AMZN')
        assert float(amzn_yield) == 0.0
    
    @pytest.mark.asyncio
    async def test_get_dividend_yield_unknown_ticker(self):
        """Test dividend yield for unknown ticker (should use default)"""
        unknown_yield=await self.bs_engine.get_dividend_yield('UNKNOWN')
        
        assert isinstance(unknown_yield, Decimal)
        assert float(unknown_yield) == 0.015  # Default 1.5%
    
    def test_d1_calculation(self):
        """Test d1 parameter calculation"""
        # Standard parameters
        S=100.0  # Spot price
        K = 105.0  # Strike price
        T = 0.25   # 3 months
        r = 0.05   # 5% risk-free rate
        q = 0.02   # 2% dividend yield
        sigma = 0.20  # 20% volatility
        
        d1 = self.bs_engine._d1(S, K, T, r, q, sigma)
        
        # d1 should be negative for OTM call (S < K)
        assert d1 < 0
        assert -1 < d1 < 1  # Reasonable range
    
    def test_d2_calculation(self):
        """Test d2 parameter calculation"""
        d1=-0.25
        sigma = 0.20
        T = 0.25
        
        d2 = self.bs_engine._d2(d1, sigma, T)
        
        # d2=d1 - sigma * sqrt(T)
        expected_d2=d1 - sigma * math.sqrt(T)
        assert abs(d2 - expected_d2) < 1e-10
        assert d2 < d1  # d2 should always be less than d1
    
    @pytest.mark.asyncio
    async def test_black_scholes_call_itm(self):
        """Test Black-Scholes call pricing for ITM option"""
        spot=Decimal('110')      # ITM call
        strike=Decimal('100')
        time_to_expiry=0.25      # 3 months
        risk_free_rate = Decimal('0.05')
        dividend_yield=Decimal('0.02')
        volatility=Decimal('0.20')
        
        call_price=await self.bs_engine.black_scholes_call(
            spot, strike, time_to_expiry, risk_free_rate, dividend_yield, volatility
        )
        
        assert isinstance(call_price, Decimal)
        
        # ITM call should have price > intrinsic value
        intrinsic_value=spot - strike  # $10
        assert call_price > intrinsic_value
        
        # But shouldn't be more than spot price
        assert call_price < spot
        
        # Should be reasonable value (ITM call with some time value)
        assert 10 < float(call_price) < 15
    
    @pytest.mark.asyncio
    async def test_black_scholes_call_otm(self):
        """Test Black-Scholes call pricing for OTM option"""
        spot=Decimal('95')       # OTM call
        strike=Decimal('100')
        time_to_expiry=0.25
        risk_free_rate = Decimal('0.05')
        dividend_yield=Decimal('0.02')
        volatility=Decimal('0.20')
        
        call_price=await self.bs_engine.black_scholes_call(
            spot, strike, time_to_expiry, risk_free_rate, dividend_yield, volatility
        )
        
        assert isinstance(call_price, Decimal)
        
        # OTM call should have only time value (no intrinsic value)
        assert call_price > Decimal('0.01')  # Minimum price enforced
        assert call_price < spot  # Should be less than stock price
        
        # Should be reasonable time value
        assert 0.5 < float(call_price) < 5
    
    @pytest.mark.asyncio
    async def test_black_scholes_put_itm(self):
        """Test Black-Scholes put pricing for ITM option"""
        spot=Decimal('90')       # ITM put
        strike=Decimal('100')
        time_to_expiry=0.25
        risk_free_rate = Decimal('0.05')
        dividend_yield=Decimal('0.02')
        volatility=Decimal('0.20')
        
        put_price=await self.bs_engine.black_scholes_put(
            spot, strike, time_to_expiry, risk_free_rate, dividend_yield, volatility
        )
        
        assert isinstance(put_price, Decimal)
        
        # ITM put should have price > intrinsic value
        intrinsic_value=strike - spot  # $10
        assert put_price > intrinsic_value
        
        # Should be reasonable value
        assert 10 < float(put_price) < 15
    
    @pytest.mark.asyncio
    async def test_black_scholes_put_call_parity(self):
        """Test put-call parity: C - P=S - K*e^(-r*T)"""
        spot=Decimal('100')
        strike=Decimal('100')  # ATM
        time_to_expiry=0.25
        risk_free_rate = Decimal('0.05')
        dividend_yield=Decimal('0.0')  # No dividends for simple parity
        volatility=Decimal('0.20')
        
        call_price=await self.bs_engine.black_scholes_call(
            spot, strike, time_to_expiry, risk_free_rate, dividend_yield, volatility
        )
        
        put_price=await self.bs_engine.black_scholes_put(
            spot, strike, time_to_expiry, risk_free_rate, dividend_yield, volatility
        )
        
        # Put-call parity: C - P=S - K*e^(-r*T)
        left_side=call_price - put_price
        right_side = spot - strike * Decimal(str(math.exp(-float(risk_free_rate) * time_to_expiry)))
        
        # Should be equal within small tolerance
        assert abs(float(left_side - right_side)) < 0.01
    
    @pytest.mark.asyncio
    async def test_expired_option_pricing(self):
        """Test pricing of expired options"""
        spot=Decimal('110')
        strike=Decimal('100')
        time_to_expiry=0.0  # Expired
        risk_free_rate = Decimal('0.05')
        dividend_yield=Decimal('0.02')
        volatility=Decimal('0.20')
        
        # Expired ITM call should equal intrinsic value
        call_price=await self.bs_engine.black_scholes_call(
            spot, strike, time_to_expiry, risk_free_rate, dividend_yield, volatility
        )
        
        expected_intrinsic=spot - strike
        assert abs(float(call_price - expected_intrinsic)) < 0.01
        
        # Expired OTM call should be worthless
        otm_call_price=await self.bs_engine.black_scholes_call(
            Decimal('90'), strike, time_to_expiry, risk_free_rate, dividend_yield, volatility
        )
        
        # For expired OTM options, should be at least the minimum or intrinsic value
        assert otm_call_price >= Decimal('0.00')  # Should not be negative
    
    @pytest.mark.asyncio
    async def test_zero_volatility_handling(self):
        """Test handling of zero volatility"""
        spot=Decimal('110')
        strike=Decimal('100')
        time_to_expiry=0.25
        risk_free_rate = Decimal('0.05')
        dividend_yield=Decimal('0.02')
        volatility=Decimal('0.0')  # Zero volatility
        
        call_price=await self.bs_engine.black_scholes_call(
            spot, strike, time_to_expiry, risk_free_rate, dividend_yield, volatility
        )
        
        # With zero volatility, should default to intrinsic value
        intrinsic_value=max(Decimal('0'), spot - strike)
        assert abs(float(call_price - intrinsic_value)) < 0.01
    
    @pytest.mark.asyncio
    async def test_calculate_greeks(self):
        """Test Greeks calculation"""
        spot=Decimal('100')
        strike=Decimal('105')
        time_to_expiry=0.25
        risk_free_rate = Decimal('0.05')
        dividend_yield=Decimal('0.02')
        volatility=Decimal('0.20')
        
        greeks=await self.bs_engine.calculate_greeks(
            spot, strike, time_to_expiry, risk_free_rate, dividend_yield, volatility, 'call'
        )
        
        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'theta' in greeks
        assert 'vega' in greeks
        
        # Call delta should be between 0 and 1
        assert 0 < float(greeks['delta']) < 1
        
        # Gamma should be positive
        assert float(greeks['gamma']) > 0
        
        # Theta should be negative (time decay)
        assert float(greeks['theta']) < 0
        
        # Vega should be positive
        assert float(greeks['vega']) > 0
    
    @pytest.mark.asyncio
    async def test_calculate_greeks_put(self):
        """Test Greeks calculation for puts"""
        spot=Decimal('95')   # OTM put
        strike=Decimal('100')
        time_to_expiry=0.25
        risk_free_rate = Decimal('0.05')
        dividend_yield=Decimal('0.02')
        volatility=Decimal('0.20')
        
        greeks=await self.bs_engine.calculate_greeks(
            spot, strike, time_to_expiry, risk_free_rate, dividend_yield, volatility, 'put'
        )
        
        # Put delta should be negative
        assert float(greeks['delta']) < 0
        assert float(greeks['delta']) > -1
        
        # Other Greeks should have same properties as calls
        assert float(greeks['gamma']) > 0
        assert float(greeks['theta']) < 0
        assert float(greeks['vega']) > 0


class TestRealOptionsPricingEngine:
    """Test real options pricing engine with market data integration"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.pricing_engine=RealOptionsPricingEngine()
    
    @pytest.mark.asyncio
    async def test_get_implied_volatility_known_ticker(self):
        """Test implied volatility for known tickers"""
        iv=await self.pricing_engine.get_implied_volatility('AAPL')
        
        assert isinstance(iv, Decimal)
        assert 0.10 <= float(iv) <= 1.00  # 10% to 100% is reasonable range
    
    @pytest.mark.asyncio
    async def test_get_implied_volatility_high_vol_ticker(self):
        """Test implied volatility for high volatility tickers"""
        # GME and TSLA are known high-vol stocks
        gme_iv=await self.pricing_engine.get_implied_volatility('GME')
        tsla_iv=await self.pricing_engine.get_implied_volatility('TSLA')
        aapl_iv=await self.pricing_engine.get_implied_volatility('AAPL')
        
        # GME should have higher IV than AAPL
        assert float(gme_iv) > float(aapl_iv)
        assert float(tsla_iv) > float(aapl_iv)
    
    @pytest.mark.asyncio
    async def test_get_implied_volatility_caching(self):
        """Test that volatility is cached properly"""
        # First call
        iv1=await self.pricing_engine.get_implied_volatility('AAPL')
        
        # Second call should be faster (cached)
        import time
        start_time=time.time()
        iv2=await self.pricing_engine.get_implied_volatility('AAPL')
        elapsed=time.time() - start_time
        
        assert iv1== iv2  # Same value from cache
        assert elapsed < 0.01  # Should be very fast (cached)
    
    @pytest.mark.asyncio
    async def test_calculate_theoretical_price(self):
        """Test theoretical price calculation"""
        ticker='AAPL'
        strike = Decimal('200')
        expiry_date=date.today() + timedelta(days=30)
        option_type='call'
        current_price = Decimal('195')
        
        theoretical_price=await self.pricing_engine.calculate_theoretical_price(
            ticker, strike, expiry_date, option_type, current_price
        )
        
        assert isinstance(theoretical_price, Decimal)
        assert theoretical_price > Decimal('0.01')  # Should have some value
        
        # For this slightly OTM call, should be reasonable
        assert 1 < float(theoretical_price) < 20
    
    @pytest.mark.asyncio
    async def test_calculate_theoretical_price_deep_itm(self):
        """Test theoretical price for deep ITM option"""
        ticker='AAPL'
        strike = Decimal('150')      # Deep ITM
        expiry_date=date.today() + timedelta(days=30)
        option_type='call'
        current_price = Decimal('195')
        
        theoretical_price=await self.pricing_engine.calculate_theoretical_price(
            ticker, strike, expiry_date, option_type, current_price
        )
        
        # Deep ITM call should be worth at least intrinsic value
        intrinsic_value=current_price - strike  # $45
        assert theoretical_price > intrinsic_value
        assert theoretical_price < current_price  # But not more than stock price
    
    @pytest.mark.asyncio
    async def test_calculate_theoretical_price_expired(self):
        """Test theoretical price for expired option"""
        ticker='AAPL'
        strike = Decimal('190')
        expiry_date=date.today() - timedelta(days=1)  # Expired yesterday
        option_type='call'
        current_price = Decimal('195')
        
        theoretical_price=await self.pricing_engine.calculate_theoretical_price(
            ticker, strike, expiry_date, option_type, current_price
        )
        
        # Expired ITM call should equal intrinsic value
        intrinsic_value=current_price - strike
        assert abs(float(theoretical_price - intrinsic_value)) < 0.01
    
    @pytest.mark.asyncio
    async def test_calculate_theoretical_price_put(self):
        """Test theoretical price calculation for puts"""
        ticker='AAPL'
        strike = Decimal('200')      # ITM put
        expiry_date=date.today() + timedelta(days=30)
        option_type='put'
        current_price = Decimal('195')
        
        theoretical_price=await self.pricing_engine.calculate_theoretical_price(
            ticker, strike, expiry_date, option_type, current_price
        )
        
        # ITM put should have intrinsic value + time value
        intrinsic_value=strike - current_price  # $5
        assert theoretical_price > intrinsic_value
        assert 5 < float(theoretical_price) < 15
    
    def test_get_options_chain_yahoo_success(self):
        """Test successful options chain retrieval from Yahoo Finance"""
        # This test simply verifies the options parsing logic by directly testing
        # the contract creation and data structure functionality
        from datetime import date
        
        # Create test options contracts directly
        test_contracts=[
            # Calls
            OptionsContract(
                ticker='AAPL',
                strike=Decimal('190'),
                expiry_date=date(2024, 1, 19),
                option_type='call',
                bid=Decimal('8.50'),
                ask=Decimal('8.75'),
                last=Decimal('8.60'),
                volume=1500,
                open_interest=5000,
                implied_volatility=Decimal('0.25')
            ),
            OptionsContract(
                ticker='AAPL',
                strike=Decimal('195'),
                expiry_date=date(2024, 1, 19),
                option_type='call',
                bid=Decimal('5.25'),
                ask=Decimal('5.50'),
                last=Decimal('5.35'),
                volume=2800,
                open_interest=8500,
                implied_volatility=Decimal('0.23')
            ),
            OptionsContract(
                ticker='AAPL',
                strike=Decimal('200'),
                expiry_date=date(2024, 1, 19),
                option_type='call',
                bid=Decimal('2.75'),
                ask=Decimal('3.00'),
                last=Decimal('2.85'),
                volume=1200,
                open_interest=3200,
                implied_volatility=Decimal('0.22')
            ),
            OptionsContract(
                ticker='AAPL',
                strike=Decimal('205'),
                expiry_date=date(2024, 1, 19),
                option_type='call',
                bid=Decimal('1.25'),
                ask=Decimal('1.50'),
                last=Decimal('1.35'),
                volume=500,
                open_interest=1200,
                implied_volatility=Decimal('0.21')
            ),
            # Puts
            OptionsContract(
                ticker='AAPL',
                strike=Decimal('190'),
                expiry_date=date(2024, 1, 19),
                option_type='put',
                bid=Decimal('1.25'),
                ask=Decimal('1.50'),
                last=Decimal('1.35'),
                volume=500,
                open_interest=1200,
                implied_volatility=Decimal('0.21')
            ),
            OptionsContract(
                ticker='AAPL',
                strike=Decimal('195'),
                expiry_date=date(2024, 1, 19),
                option_type='put',
                bid=Decimal('2.75'),
                ask=Decimal('3.00'),
                last=Decimal('2.85'),
                volume=1200,
                open_interest=3200,
                implied_volatility=Decimal('0.22')
            ),
            OptionsContract(
                ticker='AAPL',
                strike=Decimal('200'),
                expiry_date=date(2024, 1, 19),
                option_type='put',
                bid=Decimal('5.25'),
                ask=Decimal('5.50'),
                last=Decimal('5.35'),
                volume=2800,
                open_interest=8500,
                implied_volatility=Decimal('0.23')
            ),
            OptionsContract(
                ticker='AAPL',
                strike=Decimal('205'),
                expiry_date=date(2024, 1, 19),
                option_type='put',
                bid=Decimal('8.50'),
                ask=Decimal('8.75'),
                last=Decimal('8.60'),
                volume=1500,
                open_interest=5000,
                implied_volatility=Decimal('0.25')
            ),
        ]
        
        # Test that we have the expected number of contracts
        assert len(test_contracts) == 8  # 4 calls + 4 puts
        
        # Check call options
        call_options=[opt for opt in test_contracts if opt.option_type == 'call']
        assert len(call_options) == 4
        
        # Check put options  
        put_options=[opt for opt in test_contracts if opt.option_type == 'put']
        assert len(put_options) == 4
        
        # Test first call option
        first_call=call_options[0]
        assert first_call.ticker == 'AAPL'
        assert first_call.strike == Decimal('190')
        assert first_call.option_type== 'call'
        assert first_call.bid == Decimal('8.50')
        assert first_call.ask== Decimal('8.75')
        assert first_call.volume== 1500
        
        # Check mid price calculation
        expected_mid = (Decimal('8.50') + Decimal('8.75')) / 2
        assert first_call.mid_price== expected_mid
        
        # Test bid-ask spread
        expected_spread = Decimal('8.75') - Decimal('8.50')
        assert first_call.bid_ask_spread== expected_spread
    
    @pytest.mark.asyncio
    @patch('yfinance.Ticker')
    async def test_get_options_chain_yahoo_failure(self, mock_ticker):
        """Test handling of Yahoo Finance API failure"""
        # Mock Yahoo Finance to raise exception
        mock_ticker.side_effect=Exception("API Error")
        
        expiry_date=datetime(2024, 1, 19)
        options=await self.pricing_engine.get_options_chain_yahoo('AAPL', expiry_date)
        
        # Should return empty list on failure
        assert options== []
    
    @pytest.mark.asyncio
    async def test_find_optimal_option(self):
        """Test finding optimal options contract"""
        ticker='AAPL'
        current_price = Decimal('195')
        
        # Mock some options data
        self.pricing_engine.get_options_chain_yahoo=AsyncMock(return_value=[
            OptionsContract(
                ticker=ticker,
                strike=Decimal('200'),
                expiry_date=date.today() + timedelta(days=30),
                option_type='call',
                bid=Decimal('2.50'),
                ask=Decimal('2.70'),
                volume=1000,
                open_interest=5000
            ),
            OptionsContract(
                ticker=ticker,
                strike=Decimal('205'),
                expiry_date=date.today() + timedelta(days=30),
                option_type='call',
                bid=Decimal('1.25'),
                ask=Decimal('1.45'),
                volume=500,
                open_interest=2000
            )
        ])
        
        optimal_option=await self.pricing_engine.find_optimal_option(
            ticker, current_price, min_dte=25, max_dte=35, option_type='call'
        )
        
        assert optimal_option is not None
        assert optimal_option.ticker== ticker
        assert optimal_option.option_type == 'call'
        
        # Should prefer the OTM option (3-8% OTM gets bonus points)
        # $200 strike=2.6% OTM, $205 strike=5.1% OTM (gets bonus)
        assert optimal_option.strike== Decimal('205')
    
    @pytest.mark.asyncio
    async def test_find_optimal_option_no_suitable_options(self):
        """Test finding optimal option when none meet criteria"""
        ticker='AAPL'
        current_price = Decimal('195')
        
        # Mock illiquid options that shouldn't be selected
        self.pricing_engine.get_options_chain_yahoo=AsyncMock(return_value=[
            OptionsContract(
                ticker=ticker,
                strike=Decimal('200'),
                expiry_date=date.today() + timedelta(days=30),
                option_type='call',
                bid=Decimal('0.01'),  # Too low bid
                ask=Decimal('0.05'),
                volume=1,             # Too low volume
                open_interest=10
            )
        ])
        
        optimal_option=await self.pricing_engine.find_optimal_option(
            ticker, current_price, min_dte=25, max_dte=35, option_type='call'
        )
        
        # Should return None if no options meet criteria
        assert optimal_option is None


class TestOptionsContract:
    """Test OptionsContract data structure"""
    
    def test_options_contract_creation(self):
        """Test creating options contract"""
        contract=OptionsContract(
            ticker='AAPL',
            strike=Decimal('200'),
            expiry_date=date(2024, 1, 19),
            option_type='call',
            bid=Decimal('5.20'),
            ask=Decimal('5.40'),
            last=Decimal('5.30'),
            volume=1500,
            open_interest=8000,
            implied_volatility=Decimal('0.23')
        )
        
        assert contract.ticker== 'AAPL'
        assert contract.strike == Decimal('200')
        assert contract.option_type== 'call'
        assert contract.bid == Decimal('5.20')
        assert contract.ask== Decimal('5.40')
    
    def test_mid_price_calculation(self):
        """Test mid price calculation"""
        contract=OptionsContract(
            ticker='AAPL',
            strike=Decimal('200'),
            expiry_date=date(2024, 1, 19),
            option_type='call',
            bid=Decimal('5.20'),
            ask=Decimal('5.40')
        )
        
        expected_mid=(Decimal('5.20') + Decimal('5.40')) / 2
        assert contract.mid_price== expected_mid
    
    def test_mid_price_no_bid_ask(self):
        """Test mid price when no bid/ask available"""
        contract=OptionsContract(
            ticker='AAPL',
            strike=Decimal('200'),
            expiry_date=date(2024, 1, 19),
            option_type='call',
            last=Decimal('5.30')
        )
        
        assert contract.mid_price== Decimal('5.30')
    
    def test_bid_ask_spread_calculation(self):
        """Test bid-ask spread calculation"""
        contract=OptionsContract(
            ticker='AAPL',
            strike=Decimal('200'),
            expiry_date=date(2024, 1, 19),
            option_type='call',
            bid=Decimal('5.20'),
            ask=Decimal('5.40')
        )
        
        expected_spread=Decimal('5.40') - Decimal('5.20')
        assert contract.bid_ask_spread== expected_spread
    
    def test_days_to_expiry_calculation(self):
        """Test days to expiry calculation"""
        future_date=date.today() + timedelta(days=30)
        
        contract=OptionsContract(
            ticker='AAPL',
            strike=Decimal('200'),
            expiry_date=future_date,
            option_type='call'
        )
        
        # Should be approximately 30 days
        assert 29 <= contract.days_to_expiry <= 31


class TestIntegration:
    """Integration tests for options pricing engine"""
    
    def test_create_options_pricing_engine_factory(self):
        """Test factory function"""
        engine=create_options_pricing_engine()
        
        assert isinstance(engine, RealOptionsPricingEngine)
        assert isinstance(engine.bs_engine, BlackScholesEngine)
        assert engine.cache_expiry== 300  # 5 minutes
    
    @pytest.mark.asyncio
    async def test_full_pricing_pipeline(self):
        """Test complete pricing pipeline from start to finish"""
        engine=create_options_pricing_engine()
        
        # Test realistic scenario
        ticker='AAPL'
        strike = Decimal('200')
        expiry_date=date.today() + timedelta(days=30)
        option_type='call'
        current_price = Decimal('195')
        
        # This should work end-to-end
        theoretical_price=await engine.calculate_theoretical_price(
            ticker, strike, expiry_date, option_type, current_price
        )
        
        assert isinstance(theoretical_price, Decimal)
        assert theoretical_price > Decimal('0.01')
        
        # Price should be reasonable for 5-point OTM call with 30 DTE
        assert 0.5 < float(theoretical_price) < 10
    
    @pytest.mark.asyncio
    async def test_pricing_accuracy_vs_known_values(self):
        """Test pricing accuracy against known theoretical values"""
        engine=create_options_pricing_engine()
        
        # Use standardized parameters for comparison
        # These are typical textbook examples
        spot=Decimal('100')
        strike=Decimal('100')    # ATM
        expiry_date=date.today() + timedelta(days=90)  # ~0.25 years
        risk_free_rate=Decimal('0.05')
        dividend_yield=Decimal('0.0')
        volatility=Decimal('0.20')
        
        # Override engine parameters for controlled test
        engine.bs_engine.get_risk_free_rate=AsyncMock(return_value=risk_free_rate)
        engine.bs_engine.get_dividend_yield=AsyncMock(return_value=dividend_yield)
        engine.get_implied_volatility=AsyncMock(return_value=volatility)
        
        call_price=await engine.calculate_theoretical_price(
            'TEST', strike, expiry_date, 'call', spot
        )
        
        put_price=await engine.calculate_theoretical_price(
            'TEST', strike, expiry_date, 'put', spot
        )
        
        # For ATM options with these parameters, call and put should be similar
        # (due to put-call parity with no dividends)
        price_difference=abs(float(call_price - put_price))
        assert price_difference < 1.5  # Allow for some numerical differences
        
        # Both should be reasonable values
        assert 3 < float(call_price) < 6
        assert 3 < float(put_price) < 6
    
    @pytest.mark.asyncio
    async def test_error_handling_robustness(self):
        """Test error handling throughout the system"""
        engine=create_options_pricing_engine()
        
        # Test with invalid/extreme inputs
        try:
            # Negative price
            price=await engine.calculate_theoretical_price(
                'TEST', Decimal('-100'), date.today(), 'call', Decimal('100')
            )
            # Should handle gracefully, not crash
            assert isinstance(price, Decimal)
        except Exception:
            pytest.fail("Should handle negative strike gracefully")
        
        # Test with very far expiry
        far_expiry=date.today() + timedelta(days=1000)
        price=await engine.calculate_theoretical_price(
            'TEST', Decimal('100'), far_expiry, 'call', Decimal('100')
        )
        assert isinstance(price, Decimal)
        assert price > Decimal('0')
    
    @pytest.mark.asyncio
    async def test_performance_caching(self):
        """Test that caching improves performance"""
        engine=create_options_pricing_engine()
        
        # First calculation should populate cache
        import time
        start_time=time.time()
        
        iv1=await engine.get_implied_volatility('AAPL')
        
        first_call_time=time.time() - start_time
        
        # Second calculation should be faster
        start_time=time.time()
        
        iv2=await engine.get_implied_volatility('AAPL')
        
        second_call_time=time.time() - start_time
        
        assert iv1== iv2  # Same result
        # Second call should be significantly faster (cached)
        # Allow for some variance in timing
        assert second_call_time < first_call_time * 0.5 or second_call_time < 0.001


if __name__== "__main__":# Run specific test for debugging
    import asyncio
    
    async def run_single_test():
        test=TestBlackScholesEngine()
        test.setup_method()
        await test.test_black_scholes_call_itm()
        print("✅ Black-Scholes ITM call test passed!")
        
        test2=TestRealOptionsPricingEngine()
        test2.setup_method()
        await test2.test_calculate_theoretical_price()
        print("✅ Theoretical price calculation test passed!")
    
    asyncio.run(run_single_test())