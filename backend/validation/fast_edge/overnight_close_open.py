"""Overnight Close→Open Strategy for Fast Edge Validation."""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OvernightConfig:
    """Configuration for overnight strategy."""
    lookback_days: int = 20
    min_overnight_return: float = 0.001  # 0.1% minimum
    max_overnight_return: float = 0.01   # 1% maximum
    vix_threshold: float = 25.0
    volume_threshold: float = 1000000   # Minimum daily volume
    max_position_size: float = 0.1      # 10% of portfolio


class OvernightCloseOpenStrategy:
    """
    Simple overnight Close→Open strategy for liquid ETFs.
    Buys at close, sells at open with VIX and volume filters.
    """
    
    def __init__(self, config: OvernightConfig | None = None):
        self.config = config or OvernightConfig()
        self.positions = {}
        self.trades = []
        
    def generate_signals(self, data: pd.DataFrame, vix_data: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Generate trading signals based on overnight returns.
        
        Args:
            data: OHLCV data with close and open prices
            vix_data: VIX data for volatility filter
            
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index)
        signals['close'] = data['Close']
        signals['open'] = data['Open']
        signals['volume'] = data['Volume']
        
        # Calculate overnight returns
        signals['overnight_return'] = (signals['open'] / signals['close'].shift(1)) - 1
        
        # Calculate rolling statistics
        signals['avg_overnight'] = signals['overnight_return'].rolling(self.config.lookback_days).mean()
        signals['std_overnight'] = signals['overnight_return'].rolling(self.config.lookback_days).std()
        
        # VIX filter
        if vix_data is not None:
            signals['vix'] = vix_data.reindex(data.index, method='ffill')
            vix_filter = signals['vix'] < self.config.vix_threshold
        else:
            vix_filter = pd.Series(True, index=data.index)
        
        # Volume filter
        volume_filter = signals['volume'] > self.config.volume_threshold
        
        # Signal conditions
        overnight_filter = (
            (signals['overnight_return'] > self.config.min_overnight_return) &
            (signals['overnight_return'] < self.config.max_overnight_return)
        )
        
        # Combine filters
        buy_signal = overnight_filter & vix_filter & volume_filter
        sell_signal = pd.Series(False, index=data.index)
        
        signals['buy_signal'] = buy_signal
        signals['sell_signal'] = sell_signal
        signals['position'] = buy_signal.astype(int)
        
        return signals
    
    def backtest(self, data: pd.DataFrame, vix_data: Optional[pd.Series] = None, 
                start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Run backtest of overnight strategy.
        
        Args:
            data: OHLCV data
            vix_data: VIX data
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Backtest results
        """
        # Filter data by date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        # Generate signals
        signals = self.generate_signals(data, vix_data)
        
        # Calculate returns
        returns = []
        position = 0
        
        for i, (date, row) in enumerate(signals.iterrows()):
            if row['buy_signal'] and position == 0:
                # Buy at close
                position = 1
                entry_price = row['close']
                
            elif position == 1:
                # Sell at next open
                if i + 1 < len(signals):
                    exit_price = signals.iloc[i + 1]['open']
                    trade_return = (exit_price / entry_price) - 1
                    returns.append(trade_return)
                    position = 0
        
        if not returns:
            return {
                'returns': pd.Series(dtype=float),
                'sharpe': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'trades': 0,
                'win_rate': 0.0
            }
        
        returns_series = pd.Series(returns)
        
        # Calculate metrics
        sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0
        total_return = (1 + returns_series).prod() - 1
        cumulative = (1 + returns_series).cumprod()
        max_drawdown = (cumulative / cumulative.cummax() - 1).min()
        win_rate = (returns_series > 0).mean()
        
        return {
            'returns': returns_series,
            'sharpe': sharpe,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'trades': len(returns),
            'win_rate': win_rate,
            'signals': signals
        }


class OvernightStrategyValidator:
    """Validator for overnight strategy performance."""
    
    def __init__(self):
        self.min_sharpe = 0.5
        self.max_drawdown = 0.15
        self.min_trades = 10
        self.min_win_rate = 0.4
    
    def validate_strategy(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate strategy performance."""
        results = {
            'passed': True,
            'issues': [],
            'metrics': backtest_results
        }
        
        # Check Sharpe ratio
        if backtest_results['sharpe'] < self.min_sharpe:
            results['passed'] = False
            results['issues'].append(f"Sharpe ratio {backtest_results['sharpe']:.2f} < {self.min_sharpe}")
        
        # Check max drawdown
        if backtest_results['max_drawdown'] < -self.max_drawdown:
            results['passed'] = False
            results['issues'].append(f"Max drawdown {backtest_results['max_drawdown']:.2f} > {self.max_drawdown}")
        
        # Check minimum trades
        if backtest_results['trades'] < self.min_trades:
            results['passed'] = False
            results['issues'].append(f"Too few trades: {backtest_results['trades']} < {self.min_trades}")
        
        # Check win rate
        if backtest_results['win_rate'] < self.min_win_rate:
            results['passed'] = False
            results['issues'].append(f"Win rate {backtest_results['win_rate']:.2f} < {self.min_win_rate}")
        
        return results


# Example usage and testing
if __name__ == "__main__":
    def test_overnight_strategy():
        """Test the overnight strategy."""
        print("=== Overnight Strategy Test ===")
        
        # Generate test data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Generate realistic price data
        base_price = 100
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = base_price * (1 + pd.Series(returns, index=dates)).cumprod()
        
        # Create OHLCV data
        data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        # Generate VIX data
        vix_data = pd.Series(np.random.uniform(15, 35, len(dates)), index=dates)
        
        # Test strategy
        strategy = OvernightCloseOpenStrategy()
        results = strategy.backtest(data, vix_data)
        
        print(f"Sharpe ratio: {results['sharpe']:.2f}")
        print(f"Total return: {results['total_return']:.2%}")
        print(f"Max drawdown: {results['max_drawdown']:.2%}")
        print(f"Number of trades: {results['trades']}")
        print(f"Win rate: {results['win_rate']:.2%}")
        
        # Validate
        validator = OvernightStrategyValidator()
        validation = validator.validate_strategy(results)
        print(f"Validation passed: {validation['passed']}")
        if validation['issues']:
            print(f"Issues: {validation['issues']}")
    
    test_overnight_strategy()



