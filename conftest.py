"""Pytest configuration for Django tests."""

# This must run first, before any other imports
import os
import sys

# Add project root to path for ml module imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Also ensure ml is importable by adding it explicitly
ml_path = os.path.join(project_root, "ml")
if ml_path not in sys.path:
    sys.path.insert(0, ml_path)

from unittest.mock import MagicMock
import django
import numpy as np
import pandas as pd
import pytest


def pytest_configure():
    """Configure Django for pytest."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.test_settings")
    django.setup()


def pytest_unconfigure():
    """Clean up after tests."""
    pass


@pytest.fixture(autouse=True)
def restore_critical_modules():
    """Only restore modules that are commonly mocked."""
    # Store original functions
    original_numpy_functions = {
        'array': np.array,
        'mean': np.mean,
        'std': np.std,
        'var': np.var,
        'cov': np.cov,
        'sqrt': np.sqrt,
        'sum': np.sum,
        'max': np.max,
        'min': np.min,
        'abs': np.abs,
        'log': np.log,
        'exp': np.exp,
    }
    
    original_pandas_functions = {
        'DataFrame': pd.DataFrame,
        'Series': pd.Series,
        'read_csv': pd.read_csv,
        'to_datetime': pd.to_datetime,
    }
    
    # Store original modules
    original_modules = {
        'numpy': sys.modules.get('numpy'),
        'pandas': sys.modules.get('pandas'),
        'yfinance': sys.modules.get('yfinance'),
    }
    
    yield
    
    # Selective restoration only for MagicMock instances
    for name, original in original_numpy_functions.items():
        current = getattr(np, name, None)
        if isinstance(current, MagicMock):
            setattr(np, name, original)
    
    for name, original in original_pandas_functions.items():
        current = getattr(pd, name, None)
        if isinstance(current, MagicMock):
            setattr(pd, name, original)
    
    # Restore modules if they were mocked
    for module_name, original_module in original_modules.items():
        if original_module and module_name in sys.modules:
            current_module = sys.modules[module_name]
            if isinstance(current_module, MagicMock):
                sys.modules[module_name] = original_module


@pytest.fixture(autouse=True)
def isolate_test_environment():
    """Isolate test environment from global state."""
    # Clear any global caches or state
    import backend.tradingbot.core.data_providers as data_providers
    if hasattr(data_providers, '_cache'):
        data_providers._cache.clear()
    
    yield
    
    # Clean up after test
    if hasattr(data_providers, '_cache'):
        data_providers._cache.clear()


@pytest.fixture
def mock_yfinance_data():
    """Factory for creating consistent yfinance mock data."""
    def _create_mock_data(price=150.0, volume=1000000, change=2.5):
        return {
            'price': price,
            'volume': volume,
            'change': change,
            'change_percent': (change / price) * 100,
            'high': price + 5,
            'low': price - 5,
            'open': price - 1,
            'close': price,
        }
    return _create_mock_data


@pytest.fixture
def mock_options_data():
    """Factory for creating consistent options mock data."""
    def _create_options_data(strikes=None, option_type='call', current_price=150.0):
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
    return _create_options_data


@pytest.fixture
def mock_market_data():
    """Factory for creating consistent market data."""
    def _create_market_data(ticker='AAPL', price=150.0, volume=1000000):
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
            'timestamp': '2024-01-15T16:00:00Z',
        }
    return _create_market_data


@pytest.fixture
def mock_trading_interface():
    """Factory for creating consistent trading interface mocks."""
    from unittest.mock import Mock, AsyncMock
    
    def _create_trading_interface():
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
    
    return _create_trading_interface


@pytest.fixture
def mock_data_provider():
    """Factory for creating consistent data provider mocks."""
    from unittest.mock import Mock, AsyncMock
    
    def _create_data_provider():
        mock_provider = Mock()
        mock_provider.get_market_data = AsyncMock(return_value={
            'ticker': 'AAPL',
            'price': 150.0,
            'volume': 1000000,
            'change': 2.5,
        })
        mock_provider.get_historical_data = AsyncMock(return_value=pd.DataFrame({
            'Close': [145, 147, 149, 150, 152],
            'Volume': [1000000, 1100000, 1200000, 1000000, 1300000],
        }))
        mock_provider.get_options_chain = AsyncMock(return_value={
            'calls': pd.DataFrame(),
            'puts': pd.DataFrame(),
        })
        return mock_provider
    
    return _create_data_provider
