"""Common import utilities to reduce duplication across modules."""

# Standard library imports
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal
import os
import logging

# Data analysis imports with fallbacks
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    # Create a mock pandas for graceful degradation
    class MockPandas:
        def DataFrame(self, *args, **kwargs):
            raise ImportError("pandas not available")
        def read_csv(self, *args, **kwargs):
            raise ImportError("pandas not available")
    pd = MockPandas()
    PANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    # Create a mock numpy for graceful degradation
    class MockNumpy:
        def array(self, *args, **kwargs):
            raise ImportError("numpy not available")
        def mean(self, *args, **kwargs):
            raise ImportError("numpy not available")
    np = MockNumpy()
    NUMPY_AVAILABLE = False

# Market data imports with fallbacks
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    # Create a mock yfinance for graceful degradation
    class MockYfinance:
        def download(self, *args, **kwargs):
            raise ImportError("yfinance not available")
        def Ticker(self, *args, **kwargs):
            raise ImportError("yfinance not available")
    yf = MockYfinance()
    YFINANCE_AVAILABLE = False

# Logging setup
logger = logging.getLogger(__name__)

def require_package(package_name: str, available: bool):
    """Check if a required package is available."""
    if not available:
        logger.error(f"{package_name} is required but not available")
        raise ImportError(f"{package_name} is required for this functionality")

def get_data_packages():
    """Get status of data analysis packages."""
    return {
        "pandas": PANDAS_AVAILABLE,
        "numpy": NUMPY_AVAILABLE,
        "yfinance": YFINANCE_AVAILABLE
    }