"""Test isolation utilities for problematic tests."""

import os
import sys
import tempfile
from contextlib import contextmanager
from unittest.mock import patch, MagicMock
from typing import Dict, Any, Generator


class TestIsolationManager:
    """Manages test isolation for problematic tests."""
    
    def __init__(self):
        self.original_modules = {}
        self.original_env = {}
        self.temp_dirs = []
    
    def isolate_modules(self, modules_to_isolate: list) -> Generator[None, None, None]:
        """Isolate specific modules from global state."""
        # Store original modules
        for module_name in modules_to_isolate:
            if module_name in sys.modules:
                self.original_modules[module_name] = sys.modules[module_name]
        
        # Create isolated modules
        isolated_modules = {}
        for module_name in modules_to_isolate:
            isolated_modules[module_name] = MagicMock()
            sys.modules[module_name] = isolated_modules[module_name]
        
        try:
            yield isolated_modules
        finally:
            # Restore original modules
            for module_name, original_module in self.original_modules.items():
                sys.modules[module_name] = original_module
    
    def isolate_environment(self, env_vars: Dict[str, str]) -> Generator[None, None, None]:
        """Isolate environment variables."""
        # Store original environment
        for key, value in env_vars.items():
            self.original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            yield
        finally:
            # Restore original environment
            for key, original_value in self.original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value
    
    def create_temp_directory(self) -> Generator[str, None, None]:
        """Create a temporary directory for test isolation."""
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        
        try:
            yield temp_dir
        finally:
            # Clean up temp directory
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            if temp_dir in self.temp_dirs:
                self.temp_dirs.remove(temp_dir)


@contextmanager
def isolated_test_environment():
    """Context manager for complete test isolation."""
    isolation_manager = TestIsolationManager()
    
    # Isolate common problematic modules
    modules_to_isolate = [
        'yfinance',
        'alpaca',
        'pandas',
        'numpy',
        'requests',
        'aiohttp',
    ]
    
    with isolation_manager.isolate_modules(modules_to_isolate):
        with isolation_manager.isolate_environment({
            'TEST_MODE': 'true',
            'PYTHONPATH': os.pathsep.join(sys.path),
        }):
            yield isolation_manager


@contextmanager
def mock_external_apis():
    """Context manager for mocking all external APIs."""
    with patch('yfinance.Ticker') as mock_yfinance, \
         patch('alpaca.trading.client.TradingClient') as mock_alpaca, \
         patch('requests.get') as mock_requests, \
         patch('aiohttp.ClientSession.get') as mock_aiohttp:
        
        # Configure mocks
        mock_yfinance.return_value = MagicMock()
        mock_alpaca.return_value = MagicMock()
        mock_requests.return_value = MagicMock()
        mock_aiohttp.return_value = MagicMock()
        
        yield {
            'yfinance': mock_yfinance,
            'alpaca': mock_alpaca,
            'requests': mock_requests,
            'aiohttp': mock_aiohttp,
        }


@contextmanager
def isolated_data_providers():
    """Context manager for isolating data providers."""
    with patch('backend.tradingbot.core.data_providers.UnifiedDataProvider') as mock_provider:
        mock_instance = MagicMock()
        mock_instance.get_market_data = MagicMock(return_value={
            'ticker': 'AAPL',
            'price': 150.0,
            'volume': 1000000,
        })
        mock_instance.get_historical_data = MagicMock(return_value=MagicMock())
        mock_instance.get_options_chain = MagicMock(return_value={
            'calls': MagicMock(),
            'puts': MagicMock(),
        })
        mock_provider.return_value = mock_instance
        
        yield mock_instance


@contextmanager
def isolated_trading_interface():
    """Context manager for isolating trading interface."""
    with patch('backend.tradingbot.core.trading_interface.TradingInterface') as mock_interface:
        mock_instance = MagicMock()
        mock_instance.get_account_info = MagicMock(return_value={
            'equity': 100000.0,
            'buying_power': 50000.0,
        })
        mock_instance.get_positions = MagicMock(return_value=[])
        mock_instance.place_order = MagicMock(return_value={'order_id': 'test_123'})
        mock_interface.return_value = mock_instance
        
        yield mock_instance


def mark_test_as_isolated(test_func):
    """Decorator to mark a test as requiring isolation."""
    test_func._requires_isolation = True
    return test_func


def requires_isolation(test_func):
    """Decorator for tests that require complete isolation."""
    def wrapper(*args, **kwargs):
        with isolated_test_environment():
            with mock_external_apis():
                with isolated_data_providers():
                    with isolated_trading_interface():
                        return test_func(*args, **kwargs)
    return wrapper


class IsolatedTestCase:
    """Base class for tests that require isolation."""
    
    def setUp(self):
        """Set up isolated test environment."""
        self.isolation_manager = TestIsolationManager()
    
    def tearDown(self):
        """Clean up isolated test environment."""
        # Clean up any remaining temp directories
        import shutil
        for temp_dir in self.isolation_manager.temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def isolated_test(self, test_func):
        """Run a test in isolation."""
        with isolated_test_environment():
            with mock_external_apis():
                return test_func()
