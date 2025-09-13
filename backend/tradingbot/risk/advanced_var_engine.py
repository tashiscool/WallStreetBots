"""
Advanced VaR Engine-2025 Implementation
Multi - method VaR calculation with machine learning enhancements
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

@dataclass
class VaRResult: 
    """VaR calculation result"""
    var_value: float
    confidence_level: float
    method: str
    horizon_days: int
    calculation_date: str
    additional_metrics: Dict = None

@dataclass
class VaRSuite: 
    """Comprehensive VaR results across methods and confidence levels"""
    results: Dict[str, VaRResult]
    portfolio_value: float
    calculation_timestamp: str
    
    def get_summary(self)->Dict: 
        """Get summary of all VaR calculations"""
        summary = {}
        for key, result in self.results.items(): 
            summary[key] = {
                'var_value': result.var_value,
                'var_percentage': (result.var_value / self.portfolio_value) * 100,
                'method': result.method,
                'confidence': result.confidence_level
            }
        return summary

class AdvancedVaREngine: 
    """2025 - standard VaR calculation with multiple methodologies"""
    
    def __init__(self, portfolio_value: float=100000.0):
        self.portfolio_value = portfolio_value
        self.min_data_points = 30  # Minimum data points required
        
    def calculate_var_suite(self, 
                           returns: np.ndarray,
                           confidence_levels: List[float] = [0.95, 0.99, 0.999],
                           methods: List[str] = ['parametric', 'historical', 'monte_carlo'])->VaRSuite: 
        """
        Calculate comprehensive VaR using 2025 best practices
        
        Args: 
            returns: Array of portfolio returns
            confidence_levels: List of confidence levels (e.g., [0.95, 0.99])
            methods: List of VaR methods to use
            
        Returns: 
            VaRSuite with results from all methods and confidence levels
        """
        if len(returns)  <  self.min_data_points: 
            raise ValueError(f"Insufficient data: need at least {self.min_data_points} observations")
        
        results = {}
        
        for method in methods: 
            for conf_level in confidence_levels: 
                try: 
                    var_result = self._calculate_var_method(returns, conf_level, method)
                    key = f"{method}_{int(conf_level * 100)}"
                    results[key] = var_result
                except Exception as e: 
                    print(f"Warning: Failed to calculate {method} VaR at {conf_level}: {e}")
                    continue
        
        return VaRSuite(
            results=results,
            portfolio_value = self.portfolio_value,
            calculation_timestamp = pd.Timestamp.now().isoformat()
        )
    
    def _calculate_var_method(self, 
                             returns: np.ndarray, 
                             confidence_level: float, 
                             method: str)->VaRResult:
        """Calculate VaR using specific method"""
        
        if method ==  'parametric': return self._parametric_var(returns, confidence_level)
        elif method ==  'historical': return self._historical_var(returns, confidence_level)
        elif method ==  'monte_carlo': return self._monte_carlo_var(returns, confidence_level)
        elif method ==  'evt': return self._evt_var(returns, confidence_level)
        else: 
            raise ValueError(f"Unknown VaR method: {method}")
    
    def _parametric_var(self, returns: np.ndarray, confidence_level: float)->VaRResult:
        """Parametric VaR using normal distribution assumption"""
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # Z - score for confidence level
        z_score = stats.norm.ppf(1 - confidence_level)
        
        # VaR calculation
        var_value = abs(mean_return + z_score * std_return) * self.portfolio_value
        
        return VaRResult(
            var_value=var_value,
            confidence_level=confidence_level,
            method = 'parametric_normal',
            horizon_days = 1,
            calculation_date = pd.Timestamp.now().strftime('%Y-%m-%d'),
            additional_metrics = {
                'mean_return': mean_return,
                'std_return': std_return,
                'z_score': z_score
            }
        )
    
    def _historical_var(self, returns: np.ndarray, confidence_level: float)->VaRResult:
        """Historical simulation VaR"""
        # Sort returns in ascending order
        sorted_returns = np.sort(returns)
        
        # Calculate index for confidence level
        index = int((1 - confidence_level) * len(sorted_returns))
        
        # VaR is the return at the confidence level
        var_return = sorted_returns[index]
        var_value = abs(var_return) * self.portfolio_value
        
        return VaRResult(
            var_value=var_value,
            confidence_level=confidence_level,
            method = 'historical_simulation',
            horizon_days = 1,
            calculation_date = pd.Timestamp.now().strftime('%Y-%m-%d'),
            additional_metrics = {
                'percentile_index': index,
                'var_return': var_return,
                'data_points': len(returns)
            }
        )
    
    def _monte_carlo_var(self, returns: np.ndarray, confidence_level: float, n_simulations: int=10000)->VaRResult:
        """Monte Carlo simulation VaR"""
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # Generate random returns
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
        
        # Sort simulated returns
        sorted_simulated = np.sort(simulated_returns)
        
        # Calculate VaR
        index = int((1 - confidence_level) * len(sorted_simulated))
        var_return = sorted_simulated[index]
        var_value = abs(var_return) * self.portfolio_value
        
        return VaRResult(
            var_value=var_value,
            confidence_level=confidence_level,
            method = 'monte_carlo',
            horizon_days = 1,
            calculation_date = pd.Timestamp.now().strftime('%Y-%m-%d'),
            additional_metrics = {
                'simulations': n_simulations,
                'var_return': var_return,
                'mean_simulated': np.mean(simulated_returns),
                'std_simulated': np.std(simulated_returns)
            }
        )
    
    def _evt_var(self, returns: np.ndarray, confidence_level: float)->VaRResult:
        """Extreme Value Theory VaR using Generalized Pareto Distribution"""
        # Use only negative returns (losses) for EVT
        losses = returns[returns  <  0]
        
        if len(losses)  <  10: 
            # Fallback to historical if insufficient loss data
            return self._historical_var(returns, confidence_level)
        
        # Set threshold (e.g., 90th percentile of losses)
        threshold = np.percentile(losses, 90)
        excesses = losses[losses  <  threshold] - threshold
        
        if len(excesses)  <  5: 
            return self._historical_var(returns, confidence_level)
        
        # Fit Generalized Pareto Distribution
        try: 
            shape, loc, scale=stats.genpareto.fit(excesses, floc=0)
            
            # Calculate VaR using GPD
            var_return = threshold + scale * ((1 - confidence_level) ** (-shape) - 1) / shape
            var_value = abs(var_return) * self.portfolio_value
            
            return VaRResult(
                var_value=var_value,
                confidence_level=confidence_level,
                method = 'evt_gpd',
                horizon_days = 1,
                calculation_date = pd.Timestamp.now().strftime('%Y-%m-%d'),
                additional_metrics = {
                    'threshold': threshold,
                    'shape_parameter': shape,
                    'scale_parameter': scale,
                    'excesses_count': len(excesses)
                }
            )
        except: 
            # Fallback to historical if EVT fails
            return self._historical_var(returns, confidence_level)
    
    def calculate_cvar(self, returns: np.ndarray, confidence_level: float=0.95)->float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        # Get VaR first
        var_result = self._historical_var(returns, confidence_level)
        var_threshold = var_result.additional_metrics['var_return']
        
        # Calculate CVaR as mean of returns below VaR threshold
        tail_returns = returns[returns  <=  var_threshold]
        cvar = np.mean(tail_returns) * self.portfolio_value
        
        return abs(cvar)
    
    def detect_regime_and_adjust(self, returns: np.ndarray, lookback_days: int=60)->Dict:
        """Simple regime detection based on volatility"""
        recent_returns = returns[-lookback_days: ] if len(returns)  >  lookback_days else returns
        
        # Calculate rolling volatility
        rolling_vol = pd.Series(returns).rolling(window=20).std().dropna()
        
        if len(rolling_vol)  ==  0: 
            return {'regime': 'normal', 'adjustment_factor': 1.0}
        
        current_vol = rolling_vol.iloc[-1]
        historical_vol = rolling_vol.mean()
        
        # Simple regime classification
        if current_vol  >  historical_vol * 1.5: 
            regime = 'high_volatility'
            adjustment_factor = 0.7  # Reduce position sizes
        elif current_vol  <  historical_vol * 0.7: 
            regime = 'low_volatility'
            adjustment_factor = 1.2  # Increase position sizes
        else: 
            regime = 'normal'
            adjustment_factor = 1.0
        
        return {
            'regime': regime,
            'adjustment_factor': adjustment_factor,
            'current_volatility': current_vol,
            'historical_volatility': historical_vol,
            'volatility_ratio': current_vol / historical_vol
        }

# Example usage and testing
if __name__ ==  "__main__": # Generate sample returns data
    np.random.seed(42)
    sample_returns = np.random.normal(0.001, 0.02, 252)  # Daily returns
    
    # Initialize VaR engine
    var_engine = AdvancedVaREngine(portfolio_value=100000.0)
    
    # Calculate VaR suite
    print("Calculating VaR Suite...")
    var_suite = var_engine.calculate_var_suite(
        returns=sample_returns,
        confidence_levels = [0.95, 0.99],
        methods = ['parametric', 'historical', 'monte_carlo']
    )
    
    # Print results
    print("\nVaR Results Summary: ")
    print(" = " * 50)
    summary = var_suite.get_summary()
    for key, result in summary.items(): 
        print(f"{key: 20}: ${result['var_value']:8.2f} ({result['var_percentage']: 5.2f}%)")
    
    # Calculate CVaR
    cvar_95 = var_engine.calculate_cvar(sample_returns, 0.95)
    print(f"\nCVaR (95%): ${cvar_95:.2f}")
    
    # Regime detection
    regime_info = var_engine.detect_regime_and_adjust(sample_returns)
    print(f"\nRegime Detection: ")
    print(f"Current Regime: {regime_info['regime']}")
    print(f"Adjustment Factor: {regime_info['adjustment_factor']:.2f}")
    print(f"Volatility Ratio: {regime_info['volatility_ratio']:.2f}")


