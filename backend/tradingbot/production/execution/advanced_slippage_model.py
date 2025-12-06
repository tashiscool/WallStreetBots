"""Advanced Slippage Models

Enhanced slippage prediction using ML and market microstructure.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass
class MarketMicrostructureFeatures:
    """Market microstructure features for slippage prediction."""
    bid_ask_spread: float
    order_book_imbalance: float
    volume_profile: float
    volatility: float
    time_of_day: float  # Normalized 0-1
    day_of_week: int
    recent_volume: float
    price_momentum: float
    liquidity_score: float


@dataclass
class SlippagePrediction:
    """Slippage prediction result."""
    expected_slippage_bps: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    model_confidence: float
    model_type: str
    features_used: Dict[str, float]


class AdvancedSlippageModel:
    """Advanced slippage model using ML and market microstructure."""
    
    def __init__(self, model_type: str = 'random_forest'):
        """Initialize advanced slippage model.
        
        Args:
            model_type: 'random_forest', 'gradient_boosting', or 'ensemble'
        """
        self.model_type = model_type
        self.logger = logging.getLogger(__name__)
        
        # ML models
        self.rf_model: Optional[RandomForestRegressor] = None
        self.gb_model: Optional[GradientBoostingRegressor] = None
        self.scaler = StandardScaler()
        
        # Training data
        self.training_data: List[Dict[str, Any]] = []
        self.is_trained = False
        
        # Model performance metrics
        self.model_metrics: Dict[str, float] = {}
        
    def predict_slippage(
        self,
        symbol: str,
        side: str,
        quantity: int,
        market_conditions: Dict[str, Any],
        microstructure_features: Optional[MarketMicrostructureFeatures] = None
    ) -> SlippagePrediction:
        """Predict slippage using advanced model.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Order size
            market_conditions: Market conditions dict
            microstructure_features: Optional microstructure features
            
        Returns:
            SlippagePrediction
        """
        try:
            # Extract features
            features = self._extract_features(
                symbol, side, quantity, market_conditions, microstructure_features
            )
            
            # Use ML model if trained
            if self.is_trained:
                return self._predict_with_ml(features, side)
            else:
                # Fallback to rule-based model
                return self._predict_rule_based(features, side)
                
        except Exception as e:
            self.logger.error(f"Error predicting slippage: {e}")
            # Return conservative estimate
            return SlippagePrediction(
                expected_slippage_bps=10.0,  # 10 bps default
                confidence_interval_lower=5.0,
                confidence_interval_upper=20.0,
                model_confidence=0.5,
                model_type='fallback',
                features_used={}
            )
    
    def _extract_features(
        self,
        symbol: str,
        side: str,
        quantity: int,
        market_conditions: Dict[str, Any],
        microstructure_features: Optional[MarketMicrostructureFeatures]
    ) -> np.ndarray:
        """Extract features for prediction."""
        features = []
        
        # Basic features
        price = market_conditions.get('price', 100.0)
        volume = market_conditions.get('volume', 1000000)
        volatility = market_conditions.get('volatility', 0.20)
        
        # Order size relative to average volume
        order_size_pct = (quantity * price) / max(volume, 1)
        features.append(order_size_pct)
        
        # Volatility
        features.append(volatility)
        
        # Microstructure features if available
        if microstructure_features:
            features.append(microstructure_features.bid_ask_spread)
            features.append(microstructure_features.order_book_imbalance)
            features.append(microstructure_features.volume_profile)
            features.append(microstructure_features.liquidity_score)
            features.append(microstructure_features.time_of_day)
            features.append(microstructure_features.price_momentum)
        else:
            # Use defaults
            features.extend([0.001, 0.0, 0.5, 0.5, 0.5, 0.0])
        
        # Side indicator (1 for buy, -1 for sell)
        features.append(1.0 if side == 'buy' else -1.0)
        
        return np.array(features).reshape(1, -1)
    
    def _predict_with_ml(
        self,
        features: np.ndarray,
        side: str
    ) -> SlippagePrediction:
        """Predict using ML model."""
        try:
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict with appropriate model
            if self.model_type == 'random_forest' and self.rf_model:
                prediction = self.rf_model.predict(features_scaled)[0]
                # Get prediction intervals from tree variance
                trees_preds = [tree.predict(features_scaled)[0] for tree in self.rf_model.estimators_]
                std_dev = np.std(trees_preds)
            elif self.model_type == 'gradient_boosting' and self.gb_model:
                prediction = self.gb_model.predict(features_scaled)[0]
                std_dev = 0.05 * abs(prediction)  # Approximate
            elif self.model_type == 'ensemble' and self.rf_model and self.gb_model:
                rf_pred = self.rf_model.predict(features_scaled)[0]
                gb_pred = self.gb_model.predict(features_scaled)[0]
                prediction = (rf_pred + gb_pred) / 2
                std_dev = abs(rf_pred - gb_pred) / 2
            else:
                return self._predict_rule_based(features, side)
            
            # Ensure positive slippage
            prediction = max(0.0, prediction)
            
            # Calculate confidence intervals (95%)
            ci_lower = max(0.0, prediction - 2 * std_dev)
            ci_upper = prediction + 2 * std_dev
            
            # Model confidence based on training metrics
            confidence = self.model_metrics.get('r_squared', 0.7)
            
            return SlippagePrediction(
                expected_slippage_bps=float(prediction),
                confidence_interval_lower=float(ci_lower),
                confidence_interval_upper=float(ci_upper),
                model_confidence=float(confidence),
                model_type=self.model_type,
                features_used={f'feature_{i}': float(f) for i, f in enumerate(features[0])}
            )
            
        except Exception as e:
            self.logger.error(f"Error in ML prediction: {e}")
            return self._predict_rule_based(features, side)
    
    def _predict_rule_based(
        self,
        features: np.ndarray,
        side: str
    ) -> SlippagePrediction:
        """Rule-based slippage prediction (fallback)."""
        order_size_pct = features[0, 0]
        volatility = features[0, 1]
        
        # Base slippage
        base_slippage = 5.0  # 5 bps
        
        # Size impact
        size_impact = min(50.0, order_size_pct * 100)  # Cap at 50 bps
        
        # Volatility impact
        vol_impact = volatility * 20  # Scale volatility
        
        # Total slippage
        total_slippage = base_slippage + size_impact + vol_impact
        
        return SlippagePrediction(
            expected_slippage_bps=float(total_slippage),
            confidence_interval_lower=float(total_slippage * 0.5),
            confidence_interval_upper=float(total_slippage * 2.0),
            model_confidence=0.6,
            model_type='rule_based',
            features_used={'order_size_pct': float(order_size_pct), 'volatility': float(volatility)}
        )
    
    def train_model(
        self,
        execution_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Train ML model on execution data.
        
        Args:
            execution_data: List of execution records with:
                - actual_slippage_bps
                - symbol, side, quantity
                - market_conditions
                - microstructure_features (optional)
        
        Returns:
            Training metrics
        """
        try:
            if len(execution_data) < 50:
                self.logger.warning("Insufficient data for training (need at least 50 samples)")
                return {'error': 'insufficient_data'}
            
            # Prepare training data
            X = []
            y = []
            
            for record in execution_data:
                features = self._extract_features(
                    record['symbol'],
                    record['side'],
                    record['quantity'],
                    record['market_conditions'],
                    record.get('microstructure_features')
                )
                X.append(features[0])
                y.append(record['actual_slippage_bps'])
            
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train models
            metrics = {}
            
            if self.model_type in ['random_forest', 'ensemble']:
                self.rf_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                self.rf_model.fit(X_train_scaled, y_train)
                rf_pred = self.rf_model.predict(X_test_scaled)
                rf_r2 = 1 - np.sum((y_test - rf_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
                rf_mae = np.mean(np.abs(y_test - rf_pred))
                metrics['random_forest_r2'] = float(rf_r2)
                metrics['random_forest_mae'] = float(rf_mae)
            
            if self.model_type in ['gradient_boosting', 'ensemble']:
                self.gb_model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42
                )
                self.gb_model.fit(X_train_scaled, y_train)
                gb_pred = self.gb_model.predict(X_test_scaled)
                gb_r2 = 1 - np.sum((y_test - gb_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
                gb_mae = np.mean(np.abs(y_test - gb_pred))
                metrics['gradient_boosting_r2'] = float(gb_r2)
                metrics['gradient_boosting_mae'] = float(gb_mae)
            
            # Overall metrics
            if self.model_type == 'ensemble':
                ensemble_pred = (rf_pred + gb_pred) / 2
                ensemble_r2 = 1 - np.sum((y_test - ensemble_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
                ensemble_mae = np.mean(np.abs(y_test - ensemble_pred))
                metrics['ensemble_r2'] = float(ensemble_r2)
                metrics['ensemble_mae'] = float(ensemble_mae)
                metrics['r_squared'] = float(ensemble_r2)
            elif self.model_type == 'random_forest':
                metrics['r_squared'] = float(rf_r2)
            elif self.model_type == 'gradient_boosting':
                metrics['r_squared'] = float(gb_r2)
            
            self.is_trained = True
            self.model_metrics = metrics
            
            self.logger.info(f"Model trained. R²: {metrics.get('r_squared', 0):.3f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            return {'error': str(e)}
    
    def record_execution(
        self,
        symbol: str,
        side: str,
        quantity: int,
        market_conditions: Dict[str, Any],
        actual_slippage_bps: float,
        microstructure_features: Optional[MarketMicrostructureFeatures] = None
    ):
        """Record an execution for model training."""
        record = {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'market_conditions': market_conditions,
            'microstructure_features': microstructure_features,
            'actual_slippage_bps': actual_slippage_bps,
            'timestamp': datetime.now()
        }
        
        self.training_data.append(record)
        
        # Retrain if we have enough new data
        if len(self.training_data) % 50 == 0 and len(self.training_data) >= 50:
            self.logger.info("Retraining model with new execution data")
            self.train_model(self.training_data)

