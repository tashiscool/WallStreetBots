"""
Multi-Asset Risk Manager
Extended risk models for crypto, forex, commodities with cross-asset correlation

This module provides:
- Cross-asset correlation modeling
- Multi-asset VaR calculations
- Asset-specific risk factors
- Cross-asset hedging strategies
- Multi-asset portfolio optimization
- Real-time cross-asset monitoring

Month 5-6: Advanced Features and Automation
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json


class AssetClass(str, Enum):
    """Asset class types"""
    EQUITY="equity"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"
    BOND = "bond"
    OPTION = "option"
    FUTURE = "future"


class RiskFactor(str, Enum):
    """Risk factors for multi-asset modeling"""
    MARKET_RISK="market_risk"
    CREDIT_RISK = "credit_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    OPERATIONAL_RISK = "operational_risk"
    REGULATORY_RISK = "regulatory_risk"
    CURRENCY_RISK = "currency_risk"
    INTEREST_RATE_RISK = "interest_rate_risk"
    COMMODITY_RISK = "commodity_risk"
    CRYPTO_RISK = "crypto_risk"


@dataclass
class AssetPosition:
    """Multi-asset position representation"""
    symbol: str
    asset_class: AssetClass
    quantity: float
    value: float
    currency: str
    risk_factors: Dict[RiskFactor, float] = field(default_factory=dict)
    correlation_factors: Dict[str, float] = field(default_factory=dict)
    liquidity_score: float=1.0
    volatility: float = 0.0
    beta: float = 1.0


@dataclass
class CrossAssetCorrelation:
    """Cross-asset correlation data"""
    asset1: str
    asset2: str
    correlation: float
    correlation_type: str  # "pearson", "spearman", "kendall"
    time_horizon: int  # days
    last_updated: datetime
    confidence: float


@dataclass
class MultiAssetRiskMetrics:
    """Multi-asset risk metrics"""
    total_var: float
    total_cvar: float
    asset_class_vars: Dict[AssetClass, float]
    risk_factor_exposures: Dict[RiskFactor, float]
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float
    currency_risk: float
    cross_asset_hedge_ratio: float
    diversification_ratio: float


class MultiAssetRiskManager:
    """
    Multi-asset risk management system
    
    Provides:
    - Cross-asset correlation modeling
    - Multi-asset VaR calculations
    - Asset-specific risk factors
    - Cross-asset hedging strategies
    - Real-time multi-asset monitoring
    """
    
    def __init__(self, 
                 base_currency: str="USD",
                 correlation_window: int=252,
                 enable_crypto: bool=True,
                 enable_forex: bool=True,
                 enable_commodities: bool=True):
        """
        Initialize multi-asset risk manager
        
        Args:
            base_currency: Base currency for calculations
            correlation_window: Window for correlation calculations
            enable_crypto: Enable crypto asset support
            enable_forex: Enable forex asset support
            enable_commodities: Enable commodity asset support
        """
        self.base_currency=base_currency
        self.correlation_window = correlation_window
        self.enable_crypto = enable_crypto
        self.enable_forex = enable_forex
        self.enable_commodities = enable_commodities
        
        self.logger = logging.getLogger(__name__)
        
        # Asset data
        self.positions: Dict[str, AssetPosition] = {}
        self.correlations: Dict[Tuple[str, str], CrossAssetCorrelation] = {}
        self.asset_class_weights: Dict[AssetClass, float] = {}
        
        # Risk models
        self.correlation_matrix=None
        self.risk_factor_loadings = {}
        self.asset_class_correlations = {}
        
        # Performance tracking
        self.calculation_count = 0
        self.last_calculation = None
        
        self.logger.info("Multi-Asset Risk Manager initialized")
    
    async def add_position(self, 
                         symbol: str,
                         asset_class: AssetClass,
                         quantity: float,
                         value: float,
                         currency: str="USD",
                         risk_factors: Dict[RiskFactor, float] = None,
                         liquidity_score: float=1.0,
                         volatility: float=0.0,
                         beta: float=1.0):
        """
        Add multi-asset position
        
        Args:
            symbol: Asset symbol
            asset_class: Asset class
            quantity: Position quantity
            value: Position value
            currency: Position currency
            risk_factors: Asset-specific risk factors
            liquidity_score: Liquidity score (0-1)
            volatility: Asset volatility
            beta: Asset beta
        """
        try:
            position=AssetPosition(
                symbol=symbol,
                asset_class=asset_class,
                quantity=quantity,
                value=value,
                currency=currency,
                risk_factors=risk_factors or {},
                liquidity_score=liquidity_score,
                volatility=volatility,
                beta=beta
            )
            
            self.positions[symbol] = position
            
            # Update asset class weights
            await self._update_asset_class_weights()
            
            self.logger.info(f"Added position: {symbol} ({asset_class}) - ${value:,.0f}")
            
        except Exception as e:
            self.logger.error(f"Error adding position {symbol}: {e}")
    
    async def calculate_cross_asset_correlations(self, 
                                                market_data: Dict[str, pd.DataFrame]) -> Dict[Tuple[str, str], CrossAssetCorrelation]:
        """
        Calculate cross-asset correlations
        
        Args:
            market_data: Market data for all assets
            
        Returns:
            Dict of cross-asset correlations
        """
        try:
            correlations={}
            
            # Get all asset symbols
            symbols = list(market_data.keys())
            
            # Calculate pairwise correlations
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i+1:], i+1):
                    if symbol1 in market_data and symbol2 in market_data:
                        correlation=await self._calculate_asset_correlation(
                            symbol1, symbol2, market_data[symbol1], market_data[symbol2]
                        )
                        
                        if correlation:
                            correlations[(symbol1, symbol2)] = correlation
                            correlations[(symbol2, symbol1)] = correlation  # Symmetric
            
            self.correlations.update(correlations)
            
            # Update correlation matrix
            await self._update_correlation_matrix()
            
            self.logger.info(f"Calculated {len(correlations)} cross-asset correlations")
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error calculating cross-asset correlations: {e}")
            return {}
    
    async def _calculate_asset_correlation(self, 
                                         symbol1: str,
                                         symbol2: str,
                                         data1: pd.DataFrame,
                                         data2: pd.DataFrame) -> Optional[CrossAssetCorrelation]:
        """Calculate correlation between two assets"""
        try:
            # Ensure data alignment
            common_dates=data1.index.intersection(data2.index)
            if len(common_dates) < 30:  # Need minimum data
                return None
            
            # Get aligned data
            aligned_data1=data1.loc[common_dates]
            aligned_data2 = data2.loc[common_dates]
            
            # Calculate returns
            returns1 = aligned_data1['Close'].pct_change().dropna()
            returns2=aligned_data2['Close'].pct_change().dropna()
            
            # Ensure same length
            min_length=min(len(returns1), len(returns2))
            returns1=returns1.iloc[-min_length:]
            returns2 = returns2.iloc[-min_length:]
            
            if len(returns1) < 30:
                return None
            
            # Calculate Pearson correlation
            pearson_corr=returns1.corr(returns2)
            
            # Calculate Spearman correlation
            spearman_corr=returns1.corr(returns2, method='spearman')
            
            # Use Pearson as primary correlation
            correlation=CrossAssetCorrelation(
                asset1=symbol1,
                asset2=symbol2,
                correlation=pearson_corr,
                correlation_type="pearson",
                time_horizon=self.correlation_window,
                last_updated=datetime.now(),
                confidence=min(1.0, len(returns1) / self.correlation_window)
            )
            
            return correlation
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation {symbol1}-{symbol2}: {e}")
            return None
    
    async def _update_correlation_matrix(self):
        """Update correlation matrix"""
        try:
            if not self.correlations:
                return
            
            # Get unique assets
            assets=set()
            for (asset1, asset2) in self.correlations.keys():
                assets.add(asset1)
                assets.add(asset2)
            
            assets=sorted(list(assets))
            n_assets=len(assets)
            
            # Initialize correlation matrix
            corr_matrix=np.eye(n_assets)
            
            # Fill correlation matrix
            for i, asset1 in enumerate(assets):
                for j, asset2 in enumerate(assets):
                    if i != j:
                        corr_data=self.correlations.get((asset1, asset2))
                        if corr_data:
                            corr_matrix[i, j] = corr_data.correlation
                        else:
                            corr_matrix[i, j] = 0.0  # Default correlation
            
            self.correlation_matrix=pd.DataFrame(corr_matrix, index=assets, columns=assets)
            
            self.logger.info(f"Updated correlation matrix: {n_assets}x{n_assets}")
            
        except Exception as e:
            self.logger.error(f"Error updating correlation matrix: {e}")
    
    async def calculate_multi_asset_var(self, 
                                      confidence_level: float=0.99,
                                      time_horizon: int=1) -> MultiAssetRiskMetrics:
        """
        Calculate multi-asset VaR
        
        Args:
            confidence_level: VaR confidence level
            time_horizon: Time horizon in days
            
        Returns:
            MultiAssetRiskMetrics: Multi-asset risk metrics
        """
        try:
            if not self.positions:
                return MultiAssetRiskMetrics(0, 0, {}, {}, 0, 0, 0, 0, 0, 0)
            
            # Calculate asset class VaRs
            asset_class_vars={}
            asset_class_values = {}
            
            for asset_class in AssetClass:
                class_positions = [pos for pos in self.positions.values() 
                                 if pos.asset_class== asset_class]
                
                if class_positions:
                    class_value = sum(pos.value for pos in class_positions)
                    class_var=await self._calculate_asset_class_var(
                        class_positions, confidence_level, time_horizon
                    )
                    
                    asset_class_vars[asset_class] = class_var
                    asset_class_values[asset_class] = class_value
            
            # Calculate risk factor exposures
            risk_factor_exposures=await self._calculate_risk_factor_exposures()
            
            # Calculate correlation risk
            correlation_risk=await self._calculate_correlation_risk()
            
            # Calculate concentration risk
            concentration_risk=await self._calculate_concentration_risk()
            
            # Calculate liquidity risk
            liquidity_risk=await self._calculate_liquidity_risk()
            
            # Calculate currency risk
            currency_risk=await self._calculate_currency_risk()
            
            # Calculate cross-asset hedge ratio
            hedge_ratio=await self._calculate_hedge_ratio()
            
            # Calculate diversification ratio
            diversification_ratio=await self._calculate_diversification_ratio()
            
            # Calculate total VaR (simplified)
            total_var=sum(asset_class_vars.values()) * 0.8  # Diversification benefit
            total_cvar=total_var * 1.2  # CVaR is typically higher than VaR
            
            metrics = MultiAssetRiskMetrics(
                total_var=total_var,
                total_cvar=total_cvar,
                asset_class_vars=asset_class_vars,
                risk_factor_exposures=risk_factor_exposures,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                liquidity_risk=liquidity_risk,
                currency_risk=currency_risk,
                cross_asset_hedge_ratio=hedge_ratio,
                diversification_ratio=diversification_ratio
            )
            
            # Update tracking
            self.calculation_count += 1
            self.last_calculation=datetime.now()
            
            self.logger.info(f"Multi-asset VaR calculated: {total_var:.2%}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating multi-asset VaR: {e}")
            return MultiAssetRiskMetrics(0, 0, {}, {}, 0, 0, 0, 0, 0, 0)
    
    async def _calculate_asset_class_var(self, 
                                       positions: List[AssetPosition],
                                       confidence_level: float,
                                       time_horizon: int) -> float:
        """Calculate VaR for an asset class"""
        try:
            if not positions:
                return 0.0
            
            # Calculate weighted volatility
            total_value=sum(pos.value for pos in positions)
            if total_value== 0:
                return 0.0
            
            weighted_vol = sum(pos.volatility * pos.value for pos in positions) / total_value
            
            # Calculate VaR (simplified)
            z_score=2.33 if confidence_level == 0.99 else 1.96  # Simplified
            var = z_score * weighted_vol * np.sqrt(time_horizon) * total_value
            
            return var / total_value  # Return as percentage
            
        except Exception as e:
            self.logger.error(f"Error calculating asset class VaR: {e}")
            return 0.0
    
    async def _calculate_risk_factor_exposures(self) -> Dict[RiskFactor, float]:
        """Calculate risk factor exposures"""
        try:
            exposures={}
            
            for risk_factor in RiskFactor:
                exposure = 0.0
                for position in self.positions.values():
                    factor_exposure=position.risk_factors.get(risk_factor, 0.0)
                    exposure += factor_exposure * position.value
                
                exposures[risk_factor] = exposure
            
            return exposures
            
        except Exception as e:
            self.logger.error(f"Error calculating risk factor exposures: {e}")
            return {}
    
    async def _calculate_correlation_risk(self) -> float:
        """Calculate correlation risk"""
        try:
            if not self.correlation_matrix is not None:
                return 0.0
            
            # Calculate average correlation
            corr_values=self.correlation_matrix.values
            mask = ~np.eye(corr_values.shape[0], dtype=bool)
            avg_correlation=np.mean(corr_values[mask])
            
            # Higher correlation=higher risk
            correlation_risk = avg_correlation
            
            return correlation_risk
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation risk: {e}")
            return 0.0
    
    async def _calculate_concentration_risk(self) -> float:
        """Calculate concentration risk"""
        try:
            if not self.positions:
                return 0.0
            
            total_value=sum(pos.value for pos in self.positions.values())
            if total_value== 0:
                return 0.0
            
            # Calculate Herfindahl index
            weights = [pos.value / total_value for pos in self.positions.values()]
            herfindahl_index=sum(w ** 2 for w in weights)
            
            return herfindahl_index
            
        except Exception as e:
            self.logger.error(f"Error calculating concentration risk: {e}")
            return 0.0
    
    async def _calculate_liquidity_risk(self) -> float:
        """Calculate liquidity risk"""
        try:
            if not self.positions:
                return 0.0
            
            total_value=sum(pos.value for pos in self.positions.values())
            if total_value== 0:
                return 0.0
            
            # Calculate weighted liquidity score
            weighted_liquidity = sum(pos.liquidity_score * pos.value for pos in self.positions.values())
            avg_liquidity=weighted_liquidity / total_value
            
            # Convert to risk (lower liquidity = higher risk)
            liquidity_risk=1.0 - avg_liquidity
            
            return liquidity_risk
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity risk: {e}")
            return 0.0
    
    async def _calculate_currency_risk(self) -> float:
        """Calculate currency risk"""
        try:
            if not self.positions:
                return 0.0
            
            # Count non-base currency positions
            non_base_positions=[pos for pos in self.positions.values() 
                                if pos.currency != self.base_currency]
            
            if not non_base_positions:
                return 0.0
            
            total_value=sum(pos.value for pos in self.positions.values())
            non_base_value=sum(pos.value for pos in non_base_positions)
            
            currency_risk=non_base_value / total_value
            
            return currency_risk
            
        except Exception as e:
            self.logger.error(f"Error calculating currency risk: {e}")
            return 0.0
    
    async def _calculate_hedge_ratio(self) -> float:
        """Calculate cross-asset hedge ratio"""
        try:
            if not self.positions:
                return 0.0
            
            # Calculate hedge ratio based on asset class diversification
            asset_classes=set(pos.asset_class for pos in self.positions.values())
            total_asset_classes=len(AssetClass)
            
            hedge_ratio=len(asset_classes) / total_asset_classes
            
            return hedge_ratio
            
        except Exception as e:
            self.logger.error(f"Error calculating hedge ratio: {e}")
            return 0.0
    
    async def _calculate_diversification_ratio(self) -> float:
        """Calculate diversification ratio"""
        try:
            if not self.positions:
                return 0.0
            
            # Calculate diversification ratio
            # Higher ratio=better diversification
            asset_count = len(self.positions)
            asset_classes=len(set(pos.asset_class for pos in self.positions.values()))
            
            diversification_ratio=asset_classes / asset_count if asset_count > 0 else 0.0
            
            return diversification_ratio
            
        except Exception as e:
            self.logger.error(f"Error calculating diversification ratio: {e}")
            return 0.0
    
    async def _update_asset_class_weights(self):
        """Update asset class weights"""
        try:
            if not self.positions:
                return
            
            total_value=sum(pos.value for pos in self.positions.values())
            if total_value== 0:
                return
            
            # Calculate weights by asset class
            for asset_class in AssetClass:
                class_value = sum(pos.value for pos in self.positions.values() 
                                if pos.asset_class== asset_class)
                self.asset_class_weights[asset_class] = class_value / total_value
            
        except Exception as e:
            self.logger.error(f"Error updating asset class weights: {e}")
    
    async def get_cross_asset_hedge_suggestions(self) -> List[Dict[str, Any]]:
        """Get cross-asset hedge suggestions"""
        try:
            suggestions=[]
            
            # Analyze correlations for hedge opportunities
            for (asset1, asset2), correlation in self.correlations.items():
                if correlation.correlation < -0.7:  # Strong negative correlation
                    suggestion={
                        'type':'hedge_pair',
                        'asset1':asset1,
                        'asset2':asset2,
                        'correlation':correlation.correlation,
                        'confidence':correlation.confidence,
                        'reasoning':f"Strong negative correlation ({correlation.correlation:.3f}) suggests hedging opportunity"
                    }
                    suggestions.append(suggestion)
            
            # Analyze asset class diversification
            if len(set(pos.asset_class for pos in self.positions.values())) < 3:
                suggestion={
                    'type':'diversification',
                    'reasoning':"Portfolio lacks asset class diversification",
                    'recommendation':"Consider adding positions in different asset classes"
                }
                suggestions.append(suggestion)
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error getting hedge suggestions: {e}")
            return []
    
    async def get_multi_asset_summary(self) -> Dict[str, Any]:
        """Get comprehensive multi-asset summary"""
        try:
            # Calculate current metrics
            metrics=await self.calculate_multi_asset_var()
            
            # Get hedge suggestions
            hedge_suggestions=await self.get_cross_asset_hedge_suggestions()
            
            summary={
                'timestamp':self.last_calculation,
                'calculation_count':self.calculation_count,
                'total_positions':len(self.positions),
                'asset_classes':list(set(pos.asset_class for pos in self.positions.values())),
                'asset_class_weights':self.asset_class_weights,
                'risk_metrics':{
                    'total_var':metrics.total_var,
                    'total_cvar':metrics.total_cvar,
                    'correlation_risk':metrics.correlation_risk,
                    'concentration_risk':metrics.concentration_risk,
                    'liquidity_risk':metrics.liquidity_risk,
                    'currency_risk':metrics.currency_risk,
                    'diversification_ratio':metrics.diversification_ratio
                },
                'correlations_count':len(self.correlations),
                'hedge_suggestions':hedge_suggestions
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting multi-asset summary: {e}")
            return {'error':str(e)}


# Example usage and testing
if __name__== "__main__":
    async def test_multi_asset_risk():
        # Initialize multi-asset risk manager
        risk_manager=MultiAssetRiskManager(
            base_currency="USD",
            enable_crypto=True,
            enable_forex=True,
            enable_commodities=True
        )
        
        # Add sample positions
        await risk_manager.add_position(
            "AAPL", AssetClass.EQUITY, 100, 15000, "USD", 
            volatility=0.25, liquidity_score=0.9
        )
        
        await risk_manager.add_position(
            "BTC", AssetClass.CRYPTO, 0.5, 20000, "USD",
            volatility=0.60, liquidity_score=0.8
        )
        
        await risk_manager.add_position(
            "EURUSD", AssetClass.FOREX, 100000, 110000, "USD",
            volatility=0.15, liquidity_score=0.95
        )
        
        await risk_manager.add_position(
            "GOLD", AssetClass.COMMODITY, 10, 18000, "USD",
            volatility=0.20, liquidity_score=0.85
        )
        
        # Simulate market data
        dates=pd.date_range(end=datetime.now(), periods=252, freq='D')
        
        market_data={
            "AAPL":pd.DataFrame({
                'Close':150 + np.random.normal(0, 5, 252)
            }, index=dates),
            "BTC":pd.DataFrame({
                'Close':40000 + np.random.normal(0, 2000, 252)
            }, index=dates),
            "EURUSD":pd.DataFrame({
                'Close':1.1 + np.random.normal(0, 0.02, 252)
            }, index=dates),
            "GOLD":pd.DataFrame({
                'Close':1800 + np.random.normal(0, 50, 252)
            }, index=dates)
        }
        
        # Calculate correlations
        correlations=await risk_manager.calculate_cross_asset_correlations(market_data)
        print(f"Calculated {len(correlations)} correlations")
        
        # Calculate multi-asset VaR
        metrics=await risk_manager.calculate_multi_asset_var()
        print(f"Multi-asset VaR: {metrics.total_var:.2%}")
        print(f"Asset class VaRs: {metrics.asset_class_vars}")
        
        # Get summary
        summary=await risk_manager.get_multi_asset_summary()
        print(f"Summary: {summary}")
    
    asyncio.run(test_multi_asset_risk())


