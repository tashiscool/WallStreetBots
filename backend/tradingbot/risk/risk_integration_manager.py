"""
Risk Integration Manager
Connects sophisticated risk models to WallStreetBots trading strategies

This module provides real-time risk management integration:
- Live VaR/CVaR calculations during trading
- Automated risk controls and position sizing
- Cross-strategy risk coordination
- Real-time risk alerts and monitoring
- Portfolio-level risk management

Month 3-4: Integration with WallStreetBots
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

# Import from complete risk engine if available
try:
    from .risk_engine_complete import (
        RiskEngine, var_historical, cvar_historical, var_parametric, 
        cvar_parametric, var_cvar_mc, liquidity_adjusted_var
    )
    COMPLETE_ENGINE_AVAILABLE=True
except ImportError:
    COMPLETE_ENGINE_AVAILABLE = False
from .advanced_var_engine import AdvancedVaREngine
from .stress_testing_engine import StressTesting2025
from .ml_risk_predictor import MLRiskPredictor
from .risk_dashboard import RiskDashboard2025
from .database_schema import RiskDatabaseManager


@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_total_var: float = 0.05  # 5% max total VaR
    max_total_cvar: float = 0.07  # 7% max total CVaR
    max_position_var: float = 0.02  # 2% max per position VaR
    max_drawdown: float = 0.15  # 15% max drawdown
    max_concentration: float = 0.30  # 30% max concentration
    max_greeks_risk: float = 0.10  # 10% max Greeks risk


@dataclass
class RiskMetrics:
    """Current risk metrics"""
    portfolio_var: float = 0.0
    portfolio_cvar: float = 0.0
    portfolio_lvar: float = 0.0
    total_exposure: float = 0.0
    concentration_risk: float = 0.0
    greeks_risk: float = 0.0
    stress_test_score: float = 0.0
    ml_risk_score: float = 0.0
    within_limits: bool = True
    alerts: List[str] = field(default_factory=list)


class RiskIntegrationManager:
    """
    Integrates sophisticated risk models with WallStreetBots trading strategies
    
    Provides:
    - Real-time risk assessment during trading
    - Automated risk controls and position sizing
    - Cross-strategy risk coordination
    - Risk alerts and monitoring
    - Portfolio-level risk management
    """
    
    def __init__(self, 
                 risk_limits: RiskLimits=None,
                 enable_ml: bool=True,
                 enable_stress_testing: bool=True,
                 enable_dashboard: bool=True):
        """
        Initialize risk integration manager
        
        Args:
            risk_limits: Risk limits configuration
            enable_ml: Enable machine learning risk prediction
            enable_stress_testing: Enable stress testing
            enable_dashboard: Enable risk dashboard
        """
        self.risk_limits=risk_limits or RiskLimits()
        self.logger=logging.getLogger(__name__)
        
        # Initialize risk engines
        self.var_engine=AdvancedVaREngine()
        self.stress_engine=StressTesting2025() if enable_stress_testing else None
        self.ml_predictor=MLRiskPredictor() if enable_ml else None
        self.dashboard=RiskDashboard2025() if enable_dashboard else None
        self.db_manager=RiskDatabaseManager()
        
        # Risk state
        self.current_metrics=RiskMetrics()
        self.portfolio_positions={}
        self.strategy_risks = {}
        self.risk_history = []
        
        # Performance tracking
        self.last_calculation = None
        self.calculation_count = 0
        
        self.logger.info("Risk Integration Manager initialized")
    
    async def calculate_portfolio_risk(self, 
                                     positions: Dict[str, Dict],
                                     market_data: Dict[str, pd.DataFrame],
                                     portfolio_value: float) -> RiskMetrics:
        """
        Calculate comprehensive portfolio risk metrics
        
        Args:
            positions: Current portfolio positions {symbol: {qty, value, delta, gamma, vega}}
            market_data: Market data {symbol: DataFrame with OHLCV}
            portfolio_value: Total portfolio value
            
        Returns:
            RiskMetrics: Current risk metrics
        """
        try:
            self.portfolio_positions=positions
            self.calculation_count += 1
            self.last_calculation = datetime.now()
            
            # Calculate portfolio returns
            portfolio_returns=self._calculate_portfolio_returns(positions, market_data)
            
            if portfolio_returns is None or len(portfolio_returns) < 30:
                self.logger.warning("Insufficient data for risk calculation")
                return self.current_metrics
            
            # Calculate VaR/CVaR using multiple methods
            var_metrics=await self._calculate_var_metrics(portfolio_returns, portfolio_value)
            
            # Calculate concentration risk
            concentration_risk=self._calculate_concentration_risk(positions, portfolio_value)
            
            # Calculate Greeks risk
            greeks_risk=self._calculate_greeks_risk(positions, portfolio_value)
            
            # Calculate stress test score
            stress_score=await self._calculate_stress_score(positions, market_data, portfolio_value)
            
            # Calculate ML risk score
            ml_score=await self._calculate_ml_risk_score(market_data, portfolio_value)
            
            # Check risk limits
            within_limits, alerts=self._check_risk_limits(
                var_metrics, concentration_risk, greeks_risk, stress_score, ml_score
            )
            
            # Update current metrics
            self.current_metrics=RiskMetrics(
                portfolio_var=var_metrics['var_99'],
                portfolio_cvar=var_metrics['cvar_99'],
                portfolio_lvar=var_metrics['lvar_99'],
                total_exposure=sum(pos.get('value', 0) for pos in positions.values()),
                concentration_risk=concentration_risk,
                greeks_risk=greeks_risk,
                stress_test_score=stress_score,
                ml_risk_score=ml_score,
                within_limits=within_limits,
                alerts=alerts
            )
            
            # Store in database
            await self._store_risk_metrics(self.current_metrics, portfolio_value)
            
            # Update dashboard
            if self.dashboard:
                await self._update_dashboard(positions, self.current_metrics)
            
            self.logger.info(f"Portfolio risk calculated: VaR={var_metrics['var_99']:.2%}, "
                           f"CVaR={var_metrics['cvar_99']:.2%}, Within limits: {within_limits}")
            
            return self.current_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk: {e}")
            return self.current_metrics
    
    def _calculate_portfolio_returns(self, 
                                   positions: Dict[str, Dict], 
                                   market_data: Dict[str, pd.DataFrame]) -> Optional[pd.Series]:
        """Calculate portfolio returns from positions and market data"""
        try:
            if not positions or not market_data:
                return None
            
            # Get common date range
            all_dates=set()
            for symbol, data in market_data.items():
                if symbol in positions and not data.empty:
                    all_dates.update(data.index)
            
            if not all_dates:
                return None
            
            common_dates=sorted(list(all_dates))[-252:]  # Last 252 trading days
            
            # Calculate weighted returns
            portfolio_returns=[]
            for date in common_dates:
                daily_return = 0.0
                total_weight = 0.0
                
                for symbol, pos in positions.items():
                    if symbol in market_data and not market_data[symbol].empty:
                        data=market_data[symbol]
                        if date in data.index:
                            # Calculate daily return
                            price = data.loc[date, 'Close']
                            prev_date=data.index[data.index < date]
                            if len(prev_date) > 0:
                                prev_price=data.loc[prev_date[-1], 'Close']
                                daily_return += (price / prev_price - 1) * pos.get('value', 0)
                                total_weight += pos.get('value', 0)
                
                if total_weight > 0:
                    portfolio_returns.append(daily_return / total_weight)
            
            return pd.Series(portfolio_returns, index=common_dates[-len(portfolio_returns):])
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio returns: {e}")
            return None
    
    async def _calculate_var_metrics(self, 
                                   portfolio_returns: pd.Series, 
                                   portfolio_value: float) -> Dict[str, float]:
        """Calculate VaR metrics using multiple methods"""
        try:
            # Historical VaR
            hist_var_95=var_historical(portfolio_returns, 0.95)
            hist_var_99=var_historical(portfolio_returns, 0.99)
            hist_cvar_95=cvar_historical(portfolio_returns, 0.95)
            hist_cvar_99=cvar_historical(portfolio_returns, 0.99)
            
            # Parametric VaR with Cornish-Fisher
            param_var_95=var_parametric(portfolio_returns, 0.95, use_student_t=True, cornish_fisher=True)
            param_var_99=var_parametric(portfolio_returns, 0.99, use_student_t=True, cornish_fisher=True)
            param_cvar_95=cvar_parametric(portfolio_returns, 0.95, use_student_t=True)
            param_cvar_99=cvar_parametric(portfolio_returns, 0.99, use_student_t=True)
            
            # Monte Carlo VaR
            if len(portfolio_returns) >= 30:
                mu=portfolio_returns.mean()
                cov=np.array([[portfolio_returns.var()]])
                weights=np.array([1.0])
                
                mc_var_95, mc_cvar_95=var_cvar_mc(mu, cov, weights, 0.95, n_paths=10000, student_t=True)
                mc_var_99, mc_cvar_99=var_cvar_mc(mu, cov, weights, 0.99, n_paths=10000, student_t=True)
            else:
                mc_var_95=mc_var_99 = mc_cvar_95 = mc_cvar_99 = 0.0
            
            # Use conservative estimate (max of methods)
            var_95=max(hist_var_95, param_var_95, mc_var_95)
            var_99=max(hist_var_99, param_var_99, mc_var_99)
            cvar_95=max(hist_cvar_95, param_cvar_95, mc_cvar_95)
            cvar_99=max(hist_cvar_99, param_cvar_99, mc_cvar_99)
            
            # Liquidity-adjusted VaR
            lvar_95=liquidity_adjusted_var(var_95, bid_ask_bps=10.0, slippage_bps=5.0)
            lvar_99=liquidity_adjusted_var(var_99, bid_ask_bps=10.0, slippage_bps=5.0)
            
            return {
                'var_95':var_95,
                'var_99':var_99,
                'cvar_95':cvar_95,
                'cvar_99':cvar_99,
                'lvar_95':lvar_95,
                'lvar_99':lvar_99
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating VaR metrics: {e}")
            return {'var_95':0.0, 'var_99':0.0, 'cvar_95':0.0, 'cvar_99':0.0, 'lvar_95':0.0, 'lvar_99':0.0}
    
    def _calculate_concentration_risk(self, 
                                    positions: Dict[str, Dict], 
                                    portfolio_value: float) -> float:
        """Calculate concentration risk (max single position weight)"""
        if not positions or portfolio_value <= 0:
            return 0.0
        
        max_weight=0.0
        for pos in positions.values():
            weight=pos.get('value', 0) / portfolio_value
            max_weight=max(max_weight, weight)
        
        return max_weight
    
    def _calculate_greeks_risk(self, 
                             positions: Dict[str, Dict], 
                             portfolio_value: float) -> float:
        """Calculate Greeks risk (delta, gamma, vega exposure)"""
        if not positions or portfolio_value <= 0:
            return 0.0
        
        total_greeks_risk=0.0
        for pos in positions.values():
            # Calculate Greeks P&L for 1% move
            delta_pnl=abs(pos.get('delta', 0) * 0.01)
            gamma_pnl=abs(pos.get('gamma', 0) * 0.01 * 0.01)
            vega_pnl=abs(pos.get('vega', 0) * 0.01)
            
            total_greeks_risk += delta_pnl + gamma_pnl + vega_pnl
        
        return total_greeks_risk / portfolio_value
    
    async def _calculate_stress_score(self, 
                                    positions: Dict[str, Dict], 
                                    market_data: Dict[str, pd.DataFrame],
                                    portfolio_value: float) -> float:
        """Calculate stress test score"""
        if not self.stress_engine or not positions:
            return 0.0
        
        try:
            # Run stress tests
            stress_results=await self.stress_engine.run_comprehensive_stress_tests(
                positions, market_data, portfolio_value
            )
            
            # Calculate score based on worst case scenario
            if stress_results and 'scenarios' in stress_results:
                worst_case=min(scenario.get('p_and_l', 0) for scenario in stress_results['scenarios'].values())
                return abs(worst_case) / portfolio_value if portfolio_value > 0 else 0.0
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating stress score: {e}")
            return 0.0
    
    async def _calculate_ml_risk_score(self, 
                                     market_data: Dict[str, pd.DataFrame],
                                     portfolio_value: float) -> float:
        """Calculate ML risk score"""
        if not self.ml_predictor or not market_data:
            return 0.0
        
        try:
            # Use first available symbol for ML prediction
            symbol=list(market_data.keys())[0]
            data=market_data[symbol]
            
            if len(data) < 30:
                return 0.0
            
            # Get ML risk prediction
            risk_forecast=await self.ml_predictor.predict_risk(
                data, horizon_days=5
            )
            
            return risk_forecast.risk_score / 100.0  # Convert to 0-1 scale
            
        except Exception as e:
            self.logger.error(f"Error calculating ML risk score: {e}")
            return 0.0
    
    def _check_risk_limits(self, 
                          var_metrics: Dict[str, float],
                          concentration_risk: float,
                          greeks_risk: float,
                          stress_score: float,
                          ml_score: float) -> Tuple[bool, List[str]]:
        """Check if current risk is within limits"""
        alerts=[]
        within_limits = True
        
        # Check VaR limits
        if var_metrics['var_99'] > self.risk_limits.max_total_var:
            alerts.append(f"VaR {var_metrics['var_99']:.2%} exceeds limit {self.risk_limits.max_total_var:.2%}")
            within_limits=False
        
        # Check CVaR limits
        if var_metrics['cvar_99'] > self.risk_limits.max_total_cvar:
            alerts.append(f"CVaR {var_metrics['cvar_99']:.2%} exceeds limit {self.risk_limits.max_total_cvar:.2%}")
            within_limits=False
        
        # Check concentration risk
        if concentration_risk > self.risk_limits.max_concentration:
            alerts.append(f"Concentration {concentration_risk:.2%} exceeds limit {self.risk_limits.max_concentration:.2%}")
            within_limits=False
        
        # Check Greeks risk
        if greeks_risk > self.risk_limits.max_greeks_risk:
            alerts.append(f"Greeks risk {greeks_risk:.2%} exceeds limit {self.risk_limits.max_greeks_risk:.2%}")
            within_limits=False
        
        # Check stress test score
        if stress_score > self.risk_limits.max_total_var:
            alerts.append(f"Stress test score {stress_score:.2%} exceeds VaR limit")
            within_limits=False
        
        # Check ML risk score
        if ml_score > 0.8:  # High risk threshold
            alerts.append(f"ML risk score {ml_score:.2%} indicates high risk environment")
            within_limits=False
        
        return within_limits, alerts
    
    async def _store_risk_metrics(self, metrics: RiskMetrics, portfolio_value: float):
        """Store risk metrics in database"""
        try:
            await self.db_manager.store_risk_result(
                timestamp=datetime.now(),
                portfolio_value=portfolio_value,
                var_99=metrics.portfolio_var,
                cvar_99=metrics.portfolio_cvar,
                lvar_99=metrics.portfolio_lvar,
                concentration_risk=metrics.concentration_risk,
                greeks_risk=metrics.greeks_risk,
                stress_score=metrics.stress_test_score,
                ml_risk_score=metrics.ml_risk_score,
                within_limits=metrics.within_limits,
                alerts=metrics.alerts
            )
        except Exception as e:
            self.logger.error(f"Error storing risk metrics: {e}")
    
    async def _update_dashboard(self, positions: Dict[str, Dict], metrics: RiskMetrics):
        """Update risk dashboard"""
        try:
            if self.dashboard:
                await self.dashboard.update_risk_metrics(positions, metrics)
        except Exception as e:
            self.logger.error(f"Error updating dashboard: {e}")
    
    async def get_risk_adjusted_position_size(self, 
                                            strategy_name: str,
                                            symbol: str,
                                            base_position_size: float,
                                            portfolio_value: float) -> float:
        """
        Calculate risk-adjusted position size based on current risk metrics
        
        Args:
            strategy_name: Name of the strategy
            symbol: Symbol to trade
            base_position_size: Base position size from strategy
            portfolio_value: Total portfolio value
            
        Returns:
            float: Risk-adjusted position size
        """
        try:
            # Start with base position size
            adjusted_size=base_position_size
            
            # Reduce size if portfolio risk is high
            if self.current_metrics.portfolio_var > self.risk_limits.max_total_var * 0.8:
                risk_factor = 0.5  # Reduce by 50%
                adjusted_size *= risk_factor
                self.logger.info(f"Reducing position size due to high portfolio VaR: {risk_factor:.1%}")
            
            # Reduce size if concentration risk is high
            if self.current_metrics.concentration_risk > self.risk_limits.max_concentration * 0.8:
                concentration_factor=0.7  # Reduce by 30%
                adjusted_size *= concentration_factor
                self.logger.info(f"Reducing position size due to high concentration: {concentration_factor:.1%}")
            
            # Reduce size if ML risk score is high
            if self.current_metrics.ml_risk_score > 0.7:
                ml_factor=0.6  # Reduce by 40%
                adjusted_size *= ml_factor
                self.logger.info(f"Reducing position size due to high ML risk score: {ml_factor:.1%}")
            
            # Ensure minimum position size
            min_size=portfolio_value * 0.001  # 0.1% minimum
            adjusted_size = max(adjusted_size, min_size)
            
            # Ensure maximum position size
            max_size=portfolio_value * self.risk_limits.max_position_var
            adjusted_size = min(adjusted_size, max_size)
            
            self.logger.info(f"Risk-adjusted position size for {symbol}: {base_position_size:.2%} -> {adjusted_size:.2%}")
            
            return adjusted_size
            
        except Exception as e:
            self.logger.error(f"Error calculating risk-adjusted position size: {e}")
            return base_position_size
    
    async def should_allow_trade(self, 
                               strategy_name: str,
                               symbol: str,
                               trade_value: float,
                               portfolio_value: float) -> Tuple[bool, str]:
        """
        Determine if a trade should be allowed based on risk limits
        
        Args:
            strategy_name: Name of the strategy
            symbol: Symbol to trade
            trade_value: Value of the proposed trade
            portfolio_value: Total portfolio value
            
        Returns:
            Tuple[bool, str]: (allowed, reason)
        """
        try:
            # Check if within overall risk limits
            if not self.current_metrics.within_limits:
                return False, f"Portfolio risk exceeds limits: {', '.join(self.current_metrics.alerts)}"
            
            # Check position size limits
            position_weight=trade_value / portfolio_value
            if position_weight > self.risk_limits.max_position_var:
                return False, f"Position size {position_weight:.2%} exceeds limit {self.risk_limits.max_position_var:.2%}"
            
            # Check if adding this trade would exceed concentration limits
            new_concentration=self._calculate_new_concentration(symbol, trade_value, portfolio_value)
            if new_concentration > self.risk_limits.max_concentration:
                return False, f"Concentration would exceed limit: {new_concentration:.2%} > {self.risk_limits.max_concentration:.2%}"
            
            return True, "Trade allowed"
            
        except Exception as e:
            self.logger.error(f"Error checking trade allowance: {e}")
            return False, f"Error checking trade: {e}"
    
    def _calculate_new_concentration(self, 
                                   symbol: str, 
                                   trade_value: float, 
                                   portfolio_value: float) -> float:
        """Calculate concentration risk if this trade is added"""
        current_value=self.portfolio_positions.get(symbol, {}).get('value', 0)
        new_value=current_value + trade_value
        return new_value / portfolio_value
    
    async def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        return {
            'timestamp':self.last_calculation,
            'calculation_count':self.calculation_count,
            'metrics':{
                'portfolio_var':self.current_metrics.portfolio_var,
                'portfolio_cvar':self.current_metrics.portfolio_cvar,
                'portfolio_lvar':self.current_metrics.portfolio_lvar,
                'concentration_risk':self.current_metrics.concentration_risk,
                'greeks_risk':self.current_metrics.greeks_risk,
                'stress_test_score':self.current_metrics.stress_test_score,
                'ml_risk_score':self.current_metrics.ml_risk_score,
                'within_limits':self.current_metrics.within_limits,
                'alerts':self.current_metrics.alerts
            },
            'limits':{
                'max_total_var':self.risk_limits.max_total_var,
                'max_total_cvar':self.risk_limits.max_total_cvar,
                'max_position_var':self.risk_limits.max_position_var,
                'max_concentration':self.risk_limits.max_concentration,
                'max_greeks_risk':self.risk_limits.max_greeks_risk
            },
            'utilization':{
                'var_utilization':self.current_metrics.portfolio_var / self.risk_limits.max_total_var,
                'cvar_utilization':self.current_metrics.portfolio_cvar / self.risk_limits.max_total_cvar,
                'concentration_utilization':self.current_metrics.concentration_risk / self.risk_limits.max_concentration,
                'greeks_utilization':self.current_metrics.greeks_risk / self.risk_limits.max_greeks_risk
            }
        }


