"""Test utilities for consistent mocking and test isolation."""

import pandas as pd
from datetime import datetime, date, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any, Optional


class MockDataFactory:
    """Factory for creating consistent mock data across tests."""
    
    @staticmethod
    def create_stock_data(ticker: str = "AAPL", price: float = 150.0, volume: int = 1000000) -> Dict[str, Any]:
        """Create consistent stock data."""
        return {
            'ticker': ticker,
            'price': price,
            'volume': volume,
            'change': 2.5,
            'change_percent': 1.67,
            'high': price + 5,
            'low': price - 5,
            'open': price - 1,
            'close': price,
            'timestamp': datetime.now().isoformat(),
        }
    
    @staticmethod
    def create_historical_data(days: int = 30, start_price: float = 150.0) -> pd.DataFrame:
        """Create historical price data."""
        import numpy as np
        
        dates = pd.date_range(start=date.today() - timedelta(days=days), periods=days, freq='D')
        prices = []
        current_price = start_price
        
        for _ in range(days):
            # Random walk with slight upward bias
            change = np.random.normal(0.001, 0.02) * current_price
            current_price = max(1.0, current_price + change)
            prices.append(current_price)
        
        return pd.DataFrame({
            'Date': dates,
            'Open': [p * 0.99 for p in prices],
            'High': [p * 1.02 for p in prices],
            'Low': [p * 0.98 for p in prices],
            'Close': prices,
            'Volume': np.random.randint(500000, 2000000, days),
        })
    
    @staticmethod
    def create_options_chain(strikes: List[float] | None = None, current_price: float = 150.0,
                           option_type: str = 'call') -> pd.DataFrame:
        """Create options chain data."""
        if strikes is None:
            strikes = [140, 145, 150, 155, 160]
        
        data = []
        for strike in strikes:
            intrinsic = max(0, current_price - strike) if option_type == 'call' else max(0, strike - current_price)
            time_value = max(0.5, 5.0 - abs(current_price - strike) * 0.1)
            premium = intrinsic + time_value
            
            data.append({
                'strike': strike,
                'option_type': option_type,
                'bid': premium - 0.1,
                'ask': premium + 0.1,
                'last': premium,
                'volume': 1000,
                'open_interest': 5000,
                'delta': 0.5 if option_type == 'call' else -0.5,
                'gamma': 0.01,
                'theta': -0.05,
                'vega': 0.1,
                'iv': 0.25,
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def create_earnings_data(ticker: str = "AAPL", days_ahead: int = 5) -> Dict[str, Any]:
        """Create earnings data."""
        earnings_date = date.today() + timedelta(days=days_ahead)
        return {
            'symbol': ticker,
            'date': earnings_date.strftime('%Y-%m-%d'),
            'time': 'AMC',
            'epsEstimated': 2.10,
            'epsActual': None,
            'revenueEstimated': 123000000000,
            'revenueActual': None,
        }


class MockAPIFactory:
    """Factory for creating consistent API mocks."""
    
    @staticmethod
    def create_yfinance_mock(ticker: str = "AAPL", price: float = 150.0) -> Mock:
        """Create yfinance Ticker mock."""
        mock_ticker = Mock()
        
        # Mock history data
        mock_history = pd.DataFrame({
            'Close': [145, 147, 149, 150, 152],
            'Volume': [1000000, 1100000, 1200000, 1000000, 1300000],
        })
        mock_ticker.history.return_value = mock_history
        
        # Mock options data
        mock_ticker.options = ['2024-02-16', '2024-03-15', '2024-04-19']
        
        mock_chain = Mock()
        mock_chain.calls = MockDataFactory.create_options_chain(option_type='call')
        mock_chain.puts = MockDataFactory.create_options_chain(option_type='put')
        mock_ticker.option_chain.return_value = mock_chain
        
        # Mock info data
        mock_ticker.info = {
            'shortName': 'Apple Inc.',
            'longName': 'Apple Inc.',
            'currentPrice': price,
            'marketCap': 2500000000000,
            'volume': 1000000,
            'averageVolume': 1200000,
            'dividendYield': 0.0044,
            'beta': 1.2,
        }
        
        return mock_ticker
    
    @staticmethod
    def create_alpaca_mock() -> Mock:
        """Create Alpaca API mock."""
        mock_alpaca = Mock()
        
        # Mock account info
        mock_account = Mock()
        mock_account.equity = 100000.0
        mock_account.buying_power = 50000.0
        mock_account.cash = 25000.0
        mock_alpaca.get_account.return_value = mock_account
        
        # Mock positions
        mock_alpaca.get_positions.return_value = []
        
        # Mock orders
        mock_order = Mock()
        mock_order.id = 'test_order_123'
        mock_order.status = 'filled'
        mock_alpaca.submit_order.return_value = mock_order
        
        # Mock clock
        mock_clock = Mock()
        mock_clock.is_open = True
        mock_alpaca.get_clock.return_value = mock_clock
        
        return mock_alpaca


class TestIsolationHelper:
    """Helper for test isolation and cleanup."""
    
    @staticmethod
    def patch_yfinance():
        """Context manager for patching yfinance consistently."""
        return patch('yfinance.Ticker', side_effect=MockAPIFactory.create_yfinance_mock)
    
    @staticmethod
    def patch_alpaca():
        """Context manager for patching Alpaca API consistently."""
        return patch('alpaca.trading.client.TradingClient', side_effect=MockAPIFactory.create_alpaca_mock)
    
    @staticmethod
    def patch_pandas():
        """Context manager for patching pandas functions."""
        return patch('pandas.DataFrame', wraps=pd.DataFrame)
    
    @staticmethod
    def patch_numpy():
        """Context manager for patching numpy functions."""
        import numpy as np
        return patch('numpy.array', wraps=np.array)


def create_mock_trading_interface() -> Mock:
    """Create a consistent mock trading interface."""
    mock_interface = Mock()
    mock_interface.get_account_info = AsyncMock(return_value={
        'equity': 100000.0,
        'buying_power': 50000.0,
        'cash': 25000.0,
    })
    mock_interface.get_positions = AsyncMock(return_value=[])
    mock_interface.place_order = AsyncMock(return_value={'order_id': 'test_123'})
    mock_interface.cancel_order = AsyncMock(return_value=True)
    mock_interface.get_order_status = AsyncMock(return_value={'status': 'filled'})
    return mock_interface


def create_mock_data_provider() -> Mock:
    """Create a consistent mock data provider."""
    mock_provider = Mock()
    mock_provider.get_market_data = AsyncMock(return_value=MockDataFactory.create_stock_data())
    mock_provider.get_historical_data = AsyncMock(return_value=MockDataFactory.create_historical_data())
    mock_provider.get_options_chain = AsyncMock(return_value={
        'calls': MockDataFactory.create_options_chain(option_type='call'),
        'puts': MockDataFactory.create_options_chain(option_type='put'),
    })
    mock_provider.get_earnings_data = AsyncMock(return_value=[MockDataFactory.create_earnings_data()])
    return mock_provider


def create_mock_config() -> Mock:
    """Create a consistent mock configuration."""
    mock_config = Mock()
    mock_config.risk = Mock()
    mock_config.risk.max_position_risk = 0.10
    mock_config.risk.account_size = 100000.0
    mock_config.risk.max_portfolio_risk = 0.20
    
    mock_config.trading = Mock()
    mock_config.trading.universe = ['AAPL', 'MSFT', 'GOOGL']
    mock_config.trading.max_positions = 10
    
    mock_config.data = Mock()
    mock_config.data.cache_ttl = 300
    mock_config.data.retry_attempts = 3
    
    return mock_config
