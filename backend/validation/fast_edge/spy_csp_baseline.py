"""SPY/QQQ Cash-Secured Put (CSP) Baseline Strategy."""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class CSPConfig:
    """Configuration for CSP strategy."""
    delta_target: float = 0.25  # Target delta for puts
    dte_min: int = 21          # Minimum days to expiry
    dte_max: int = 45          # Maximum days to expiry
    iv_rank_min: float = 0.3   # Minimum IV rank
    earnings_buffer_days: int = 7  # Days before/after earnings to avoid
    max_position_size: float = 0.1  # 10% of portfolio per position
    stop_loss: float = 0.5     # 50% loss stop


class SPYCSPBaselineStrategy:
    """
    Cash-Secured Put strategy on SPY/QQQ.
    Systematic CSP with IV rank and earnings filters.
    """
    
    def __init__(self, config: CSPConfig | None = None):
        self.config = config or CSPConfig()
        self.positions = {}
        self.trades = []
        
    def calculate_iv_rank(self, current_iv: float, iv_history: pd.Series) -> float:
        """Calculate IV rank (percentile of current IV in historical range)."""
        if len(iv_history) < 20:
            return 0.5  # Default to middle if insufficient data
        
        return (iv_history < current_iv).mean()
    
    def is_earnings_period(self, date: pd.Timestamp, earnings_dates: pd.Series) -> bool:
        """Check if date is within earnings buffer period."""
        buffer = timedelta(days=self.config.earnings_buffer_days)
        
        for earnings_date in earnings_dates:
            if abs(date - earnings_date) <= buffer:
                return True
        return False
    
    def generate_signals(self, price_data: pd.DataFrame, iv_data: pd.Series, 
                        earnings_dates: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Generate CSP signals based on market conditions.
        
        Args:
            price_data: OHLCV data
            iv_data: Implied volatility data
            earnings_dates: Earnings announcement dates
            
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=price_data.index)
        signals['price'] = price_data['Close']
        signals['iv'] = iv_data.reindex(price_data.index, method='ffill')
        
        # Calculate IV rank
        signals['iv_rank'] = signals['iv'].rolling(252).apply(
            lambda x: self.calculate_iv_rank(x.iloc[-1], x.iloc[:-1]) if len(x) > 1 else 0.5
        )
        
        # Calculate rolling metrics
        signals['price_ma_20'] = signals['price'].rolling(20).mean()
        signals['price_ma_50'] = signals['price'].rolling(50).mean()
        signals['volatility_20'] = signals['price'].pct_change().rolling(20).std()
        
        # Signal conditions
        iv_filter = signals['iv_rank'] >= self.config.iv_rank_min
        trend_filter = signals['price'] > signals['price_ma_20']  # Above 20-day MA
        
        # Earnings filter
        if earnings_dates is not None:
            earnings_filter = ~signals.index.to_series().apply(
                lambda x: self.is_earnings_period(x, earnings_dates)
            )
        else:
            earnings_filter = pd.Series(True, index=signals.index)
        
        # Combine filters
        csp_signal = iv_filter & trend_filter & earnings_filter
        
        signals['csp_signal'] = csp_signal
        signals['position'] = csp_signal.astype(int)
        
        return signals
    
    def simulate_csp_trades(self, signals: pd.DataFrame, option_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Simulate CSP trades with realistic option pricing.
        
        Args:
            signals: Trading signals
            option_data: Option pricing data (if available)
            
        Returns:
            DataFrame with trade results
        """
        trades = []
        position = None
        
        for i, (date, row) in enumerate(signals.iterrows()):
            if row['csp_signal'] and position is None:
                # Enter CSP position
                strike_price = row['price'] * 0.95  # 5% OTM
                
                # Estimate put premium (simplified Black-Scholes approximation)
                time_to_expiry = 30 / 365  # 30 days
                volatility = row['iv'] if 'iv' in row else 0.2
                risk_free_rate = 0.05
                
                # Simplified put premium calculation
                put_premium = self._estimate_put_premium(
                    row['price'], strike_price, time_to_expiry, volatility, risk_free_rate
                )
                
                position = {
                    'entry_date': date,
                    'strike': strike_price,
                    'premium': put_premium,
                    'quantity': 100,  # Standard option contract
                    'dte': 30
                }
                
            elif position is not None:
                # Check for exit conditions
                current_price = row['price']
                days_held = (date - position['entry_date']).days
                
                # Exit conditions
                should_exit = (
                    days_held >= position['dte'] or  # Time expiry
                    current_price <= position['strike'] * 0.9 or  # 10% below strike
                    days_held >= 7  # Minimum hold period
                )
                
                if should_exit:
                    # Calculate P&L
                    if current_price < position['strike']:
                        # Assigned - buy stock at strike
                        pnl = position['premium'] - (position['strike'] - current_price)
                        outcome = 'assigned'
                    else:
                        # Expired worthless
                        pnl = position['premium']
                        outcome = 'expired'
                    
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': date,
                        'strike': position['strike'],
                        'premium': position['premium'],
                        'exit_price': current_price,
                        'pnl': pnl,
                        'outcome': outcome,
                        'days_held': days_held
                    })
                    
                    position = None
        
        return pd.DataFrame(trades)
    
    def _estimate_put_premium(self, spot: float, strike: float, time_to_expiry: float, 
                             volatility: float, risk_free_rate: float) -> float:
        """Estimate put option premium using simplified Black-Scholes."""
        # Simplified approximation for ATM/OTM puts
        intrinsic_value = max(strike - spot, 0)
        time_value = spot * volatility * np.sqrt(time_to_expiry) * 0.4  # Rough approximation
        
        return intrinsic_value + time_value
    
    def backtest(self, price_data: pd.DataFrame, iv_data: pd.Series, 
                earnings_dates: Optional[pd.Series] = None,
                start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Run backtest of CSP strategy.
        
        Args:
            price_data: OHLCV data
            iv_data: Implied volatility data
            earnings_dates: Earnings announcement dates
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Backtest results
        """
        # Filter data by date range
        if start_date:
            price_data = price_data[price_data.index >= start_date]
        if end_date:
            price_data = price_data[price_data.index <= end_date]
        
        # Generate signals
        signals = self.generate_signals(price_data, iv_data, earnings_dates)
        
        # Simulate trades
        trades_df = self.simulate_csp_trades(signals)
        
        if trades_df.empty:
            return {
                'returns': pd.Series(dtype=float),
                'sharpe': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'trades': 0,
                'win_rate': 0.0,
                'avg_trade': 0.0
            }
        
        # Calculate returns
        returns = trades_df['pnl'] / (trades_df['strike'] * trades_df['quantity'])  # Return on capital
        returns_series = pd.Series(returns.values, index=trades_df['exit_date'])
        
        # Calculate metrics
        sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0
        total_return = returns_series.sum()
        cumulative = (1 + returns_series).cumsum()
        max_drawdown = (cumulative - cumulative.cummax()).min()
        win_rate = (returns_series > 0).mean()
        avg_trade = returns_series.mean()
        
        return {
            'returns': returns_series,
            'sharpe': sharpe,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'trades': len(trades_df),
            'win_rate': win_rate,
            'avg_trade': avg_trade,
            'trades_df': trades_df,
            'signals': signals
        }


class CSPStrategyValidator:
    """Validator for CSP strategy performance."""
    
    def __init__(self):
        self.min_sharpe = 0.3
        self.max_drawdown = 0.20
        self.min_trades = 5
        self.min_win_rate = 0.6
        self.min_avg_trade = 0.01  # 1% average trade return
    
    def validate_strategy(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate CSP strategy performance."""
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
        
        # Check average trade
        if backtest_results['avg_trade'] < self.min_avg_trade:
            results['passed'] = False
            results['issues'].append(f"Average trade {backtest_results['avg_trade']:.2%} < {self.min_avg_trade:.2%}")
        
        return results


# Example usage and testing
if __name__ == "__main__":
    def test_csp_strategy():
        """Test the CSP strategy."""
        print("=== CSP Strategy Test ===")
        
        # Generate test data
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        np.random.seed(42)
        
        # Generate realistic price data
        base_price = 400  # SPY-like price
        returns = np.random.normal(0.0008, 0.015, len(dates))
        prices = base_price * (1 + pd.Series(returns, index=dates)).cumprod()
        
        # Create price data
        price_data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.003, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.003, len(dates)))),
            'Close': prices,
            'Volume': np.random.randint(50000000, 100000000, len(dates))
        }, index=dates)
        
        # Generate IV data
        iv_data = pd.Series(np.random.uniform(0.15, 0.35, len(dates)), index=dates)
        
        # Test strategy
        strategy = SPYCSPBaselineStrategy()
        results = strategy.backtest(price_data, iv_data)
        
        print(f"Sharpe ratio: {results['sharpe']:.2f}")
        print(f"Total return: {results['total_return']:.2%}")
        print(f"Max drawdown: {results['max_drawdown']:.2%}")
        print(f"Number of trades: {results['trades']}")
        print(f"Win rate: {results['win_rate']:.2%}")
        print(f"Average trade: {results['avg_trade']:.2%}")
        
        # Validate
        validator = CSPStrategyValidator()
        validation = validator.validate_strategy(results)
        print(f"Validation passed: {validation['passed']}")
        if validation['issues']:
            print(f"Issues: {validation['issues']}")
    
    test_csp_strategy()


