"""Real-time slippage calibration and execution quality monitoring.

Continuously calibrates slippage models against live market data and
adjusts execution expectations based on actual fills.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import json

logger = logging.getLogger(__name__)


@dataclass
class ExecutionRecord:
    """Record of a single execution with all relevant data."""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit', 'ioc', etc.
    intended_quantity: int
    filled_quantity: int
    intended_price: Optional[float]  # For limit orders
    fill_price: float
    market_price_at_order: float  # Best bid/ask when order placed
    spread_at_order: float
    volume_at_order: int  # Volume in order book
    volatility_at_order: float  # Recent realized volatility
    time_of_day: str  # Market session info
    days_to_expiry: Optional[int]  # For options
    order_id: str
    latency_ms: float  # Order to ack latency
    partial_fill: bool


@dataclass
class SlippageModel:
    """Slippage prediction model with coefficients."""
    model_type: str  # 'linear', 'random_forest'
    coefficients: Dict[str, float]
    feature_names: List[str]
    r_squared: float
    mae: float  # Mean absolute error
    training_samples: int
    last_updated: datetime
    asset_class: str  # 'equity', 'option', 'etf'


class SlippagePredictor:
    """Predicts expected slippage based on market conditions."""

    def __init__(self):
        self.models = {}  # Asset class -> SlippageModel
        self.execution_history = []
        self.calibration_window = 30  # Days of data for model training

    def predict_slippage(self, symbol: str, side: str, quantity: int,
                        market_conditions: Dict[str, float]) -> Dict[str, float]:
        """Predict expected slippage for an order.

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Order size
            market_conditions: Current market state

        Returns:
            Dictionary with slippage predictions and confidence intervals
        """
        asset_class = self._classify_asset(symbol)

        if asset_class not in self.models:
            # Use default model if no calibrated model exists
            return self._default_slippage_estimate(symbol, side, quantity, market_conditions)

        model = self.models[asset_class]
        features = self._extract_features(symbol, side, quantity, market_conditions)

        try:
            if model.model_type == 'linear':
                predicted_slippage = self._predict_linear(model, features)
            elif model.model_type == 'random_forest':
                predicted_slippage = self._predict_rf(model, features)
            else:
                return self._default_slippage_estimate(symbol, side, quantity, market_conditions)

            # Add confidence intervals based on model error
            confidence_interval = model.mae * 2  # Approximate 95% CI

            return {
                'expected_slippage_bps': predicted_slippage,
                'confidence_interval_bps': confidence_interval,
                'model_r_squared': model.r_squared,
                'model_last_updated': model.last_updated.isoformat(),
                'prediction_confidence': min(0.9, model.r_squared)
            }

        except Exception as e:
            logger.error(f"Slippage prediction failed: {e}")
            return self._default_slippage_estimate(symbol, side, quantity, market_conditions)

    def record_execution(self, execution: ExecutionRecord):
        """Record an execution for model calibration."""
        self.execution_history.append(execution)

        # Calculate actual slippage
        if execution.side == 'buy':
            slippage_bps = (execution.fill_price - execution.market_price_at_order) / execution.market_price_at_order * 10000
        else:  # sell
            slippage_bps = (execution.market_price_at_order - execution.fill_price) / execution.market_price_at_order * 10000

        # Store for model training
        execution.actual_slippage_bps = slippage_bps

        logger.info(f"Recorded execution: {execution.symbol} {execution.side} "
                   f"{execution.filled_quantity} @ {execution.fill_price:.4f}, "
                   f"slippage: {slippage_bps:.2f} bps")

    def calibrate_models(self) -> Dict[str, Any]:
        """Recalibrate slippage models with recent execution data."""
        if len(self.execution_history) < 50:
            return {'status': 'insufficient_data', 'executions': len(self.execution_history)}

        calibration_results = {}

        # Group executions by asset class
        executions_by_class = {}
        for execution in self.execution_history:
            asset_class = self._classify_asset(execution.symbol)
            if asset_class not in executions_by_class:
                executions_by_class[asset_class] = []
            executions_by_class[asset_class].append(execution)

        # Train models for each asset class
        for asset_class, executions in executions_by_class.items():
            if len(executions) < 30:
                logger.warning(f"Insufficient data for {asset_class}: {len(executions)} executions")
                continue

            try:
                # Filter to recent data
                cutoff_date = datetime.now() - timedelta(days=self.calibration_window)
                recent_executions = [e for e in executions if e.timestamp >= cutoff_date]

                if len(recent_executions) < 20:
                    continue

                # Train models
                linear_model = self._train_linear_model(recent_executions, asset_class)
                rf_model = self._train_rf_model(recent_executions, asset_class)

                # Choose best model
                if rf_model.r_squared > linear_model.r_squared + 0.05:  # RF significantly better
                    self.models[asset_class] = rf_model
                    calibration_results[asset_class] = 'random_forest'
                else:
                    self.models[asset_class] = linear_model
                    calibration_results[asset_class] = 'linear'

                logger.info(f"Calibrated {calibration_results[asset_class]} model for {asset_class}: "
                           f"R² = {self.models[asset_class].r_squared:.3f}")

            except Exception as e:
                logger.error(f"Model calibration failed for {asset_class}: {e}")
                calibration_results[asset_class] = f'failed: {e!s}'

        return calibration_results

    def _classify_asset(self, symbol: str) -> str:
        """Classify asset type for model selection."""
        if any(x in symbol.upper() for x in ['SPY', 'QQQ', 'IWM', 'VIX', 'ETF']):
            return 'etf'
        elif len(symbol) > 5 or any(x in symbol for x in ['C', 'P', '2']):  # Options-like
            return 'option'
        else:
            return 'equity'

    def _extract_features(self, symbol: str, side: str, quantity: int,
                         market_conditions: Dict[str, float]) -> np.ndarray:
        """Extract features for slippage prediction."""
        features = [
            np.log(quantity + 1),  # Log order size
            market_conditions.get('spread_bps', 10),  # Bid-ask spread in bps
            market_conditions.get('volume', 1000),  # Current volume
            market_conditions.get('volatility', 0.02) * 10000,  # Volatility in bps
            1 if side == 'buy' else 0,  # Buy indicator
            market_conditions.get('time_score', 0.5),  # Time of day score (0-1)
            market_conditions.get('market_impact_score', 0.1),  # Expected market impact
        ]

        return np.array(features).reshape(1, -1)

    def _train_linear_model(self, executions: List[ExecutionRecord], asset_class: str) -> SlippageModel:
        """Train linear regression model for slippage prediction."""
        # Prepare training data
        X, y = self._prepare_training_data(executions)

        if len(X) < 10:
            raise ValueError(f"Insufficient training data: {len(X)} samples")

        # Train model
        model = LinearRegression()
        model.fit(X, y)

        # Calculate metrics
        y_pred = model.predict(X)
        r_squared = model.score(X, y)
        mae = np.mean(np.abs(y - y_pred))

        # Extract coefficients
        feature_names = ['log_quantity', 'spread_bps', 'volume', 'volatility', 'is_buy', 'time_score', 'market_impact']
        coefficients = dict(zip(feature_names, model.coef_))
        coefficients['intercept'] = model.intercept_

        return SlippageModel(
            model_type='linear',
            coefficients=coefficients,
            feature_names=feature_names,
            r_squared=r_squared,
            mae=mae,
            training_samples=len(X),
            last_updated=datetime.now(),
            asset_class=asset_class
        )

    def _train_rf_model(self, executions: List[ExecutionRecord], asset_class: str) -> SlippageModel:
        """Train random forest model for slippage prediction."""
        X, y = self._prepare_training_data(executions)

        if len(X) < 20:
            raise ValueError(f"Insufficient training data for RF: {len(X)} samples")

        # Train model
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        model.fit(X, y)

        # Calculate metrics
        y_pred = model.predict(X)
        r_squared = model.score(X, y)
        mae = np.mean(np.abs(y - y_pred))

        # Feature importance as "coefficients"
        feature_names = ['log_quantity', 'spread_bps', 'volume', 'volatility', 'is_buy', 'time_score', 'market_impact']
        coefficients = dict(zip(feature_names, model.feature_importances_))

        return SlippageModel(
            model_type='random_forest',
            coefficients=coefficients,
            feature_names=feature_names,
            r_squared=r_squared,
            mae=mae,
            training_samples=len(X),
            last_updated=datetime.now(),
            asset_class=asset_class
        )

    def _prepare_training_data(self, executions: List[ExecutionRecord]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from execution records."""
        X, y = [], []

        for execution in executions:
            if not hasattr(execution, 'actual_slippage_bps'):
                continue

            # Features
            features = [
                np.log(execution.filled_quantity + 1),
                execution.spread_at_order / execution.market_price_at_order * 10000,  # Spread in bps
                execution.volume_at_order,
                execution.volatility_at_order * 10000,  # Vol in bps
                1 if execution.side == 'buy' else 0,
                self._time_to_score(execution.time_of_day),
                self._calculate_market_impact_score(execution)
            ]

            X.append(features)
            y.append(execution.actual_slippage_bps)

        return np.array(X), np.array(y)

    def _time_to_score(self, time_of_day: str) -> float:
        """Convert time of day to a score (0-1)."""
        try:
            if 'open' in time_of_day.lower():
                return 0.1  # Market open - higher slippage
            elif 'close' in time_of_day.lower():
                return 0.2  # Market close - higher slippage
            else:
                return 0.5  # Mid-day - normal slippage
        except Exception:
            return 0.5

    def _calculate_market_impact_score(self, execution: ExecutionRecord) -> float:
        """Calculate a market impact score for the execution."""
        # Simplified market impact based on order size relative to volume
        if execution.volume_at_order > 0:
            impact_ratio = execution.filled_quantity / execution.volume_at_order
            return min(impact_ratio * 100, 1.0)  # Cap at 1.0
        return 0.1

    def _predict_linear(self, model: SlippageModel, features: np.ndarray) -> float:
        """Make prediction using linear model."""
        prediction = model.coefficients['intercept']
        for i, feature_name in enumerate(model.feature_names):
            prediction += model.coefficients[feature_name] * features[0, i]
        return max(0, prediction)  # Non-negative slippage

    def _predict_rf(self, model: SlippageModel, features: np.ndarray) -> float:
        """Make prediction using random forest model (simplified)."""
        # This is a simplified version - in production you'd want to save/load the actual sklearn model
        # For now, use a weighted average based on feature importance
        weighted_score = sum(
            model.coefficients[feature_name] * features[0, i]
            for i, feature_name in enumerate(model.feature_names)
        )
        return max(0, weighted_score * 10)  # Scale and ensure non-negative

    def _default_slippage_estimate(self, symbol: str, side: str, quantity: int,
                                 market_conditions: Dict[str, float]) -> Dict[str, float]:
        """Provide default slippage estimate when no model is available."""
        # Simple heuristic based on spread and quantity
        spread_bps = market_conditions.get('spread_bps', 10)
        base_slippage = spread_bps * 0.3  # 30% of spread

        # Adjust for order size
        if quantity > 10000:
            base_slippage *= 1.5
        elif quantity > 1000:
            base_slippage *= 1.2

        # Adjust for volatility
        volatility = market_conditions.get('volatility', 0.02)
        if volatility > 0.03:  # High volatility
            base_slippage *= 1.3

        return {
            'expected_slippage_bps': base_slippage,
            'confidence_interval_bps': base_slippage * 2,
            'model_r_squared': 0.0,
            'model_last_updated': 'never',
            'prediction_confidence': 0.3
        }


class ExecutionQualityMonitor:
    """Monitors execution quality and detects degradation."""

    def __init__(self):
        self.slippage_predictor = SlippagePredictor()
        self.execution_metrics = {}
        self.quality_thresholds = {
            'max_slippage_bps': 20,
            'max_latency_ms': 250,
            'min_fill_rate': 0.95,
            'max_adverse_selection_bps': 5
        }

    def monitor_execution(self, execution: ExecutionRecord) -> Dict[str, Any]:
        """Monitor a single execution and update quality metrics."""
        # Record execution for slippage calibration
        self.slippage_predictor.record_execution(execution)

        # Calculate quality metrics
        quality_metrics = {
            'timestamp': execution.timestamp,
            'symbol': execution.symbol,
            'fill_rate': execution.filled_quantity / execution.intended_quantity,
            'latency_ms': execution.latency_ms,
            'slippage_bps': execution.actual_slippage_bps,
            'adverse_selection_bps': self._calculate_adverse_selection(execution)
        }

        # Check against thresholds
        violations = []
        if quality_metrics['slippage_bps'] > self.quality_thresholds['max_slippage_bps']:
            violations.append(f"High slippage: {quality_metrics['slippage_bps']:.1f} bps")

        if quality_metrics['latency_ms'] > self.quality_thresholds['max_latency_ms']:
            violations.append(f"High latency: {quality_metrics['latency_ms']:.0f} ms")

        if quality_metrics['fill_rate'] < self.quality_thresholds['min_fill_rate']:
            violations.append(f"Low fill rate: {quality_metrics['fill_rate']:.1%}")

        quality_metrics['violations'] = violations
        quality_metrics['quality_score'] = self._calculate_quality_score(quality_metrics)

        # Store in metrics history
        symbol = execution.symbol
        if symbol not in self.execution_metrics:
            self.execution_metrics[symbol] = []
        self.execution_metrics[symbol].append(quality_metrics)

        # Keep only recent history (last 1000 executions per symbol)
        if len(self.execution_metrics[symbol]) > 1000:
            self.execution_metrics[symbol] = self.execution_metrics[symbol][-1000:]

        return quality_metrics

    def _calculate_adverse_selection(self, execution: ExecutionRecord) -> float:
        """Calculate adverse selection cost."""
        # Simplified calculation - in production you'd track price movement after execution
        # For now, use latency as a proxy
        if execution.latency_ms > 100:
            return execution.latency_ms * 0.02  # Rough approximation
        return 0

    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score (0-1, higher is better)."""
        score = 1.0

        # Penalize high slippage
        if metrics['slippage_bps'] > 0:
            score -= min(metrics['slippage_bps'] / 50, 0.5)  # Max 50% penalty

        # Penalize high latency
        if metrics['latency_ms'] > 100:
            score -= min((metrics['latency_ms'] - 100) / 500, 0.3)  # Max 30% penalty

        # Penalize low fill rate
        if metrics['fill_rate'] < 1.0:
            score -= (1.0 - metrics['fill_rate']) * 0.5  # 50% penalty for unfilled

        return max(0, score)

    def get_execution_summary(self, symbol: Optional[str] = None,
                            hours: int = 24) -> Dict[str, Any]:
        """Get execution quality summary."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        if symbol:
            symbols_to_check = [symbol] if symbol in self.execution_metrics else []
        else:
            symbols_to_check = list(self.execution_metrics.keys())

        summary = {
            'period_hours': hours,
            'symbols_analyzed': len(symbols_to_check),
            'total_executions': 0,
            'avg_slippage_bps': 0,
            'avg_latency_ms': 0,
            'avg_fill_rate': 0,
            'avg_quality_score': 0,
            'violation_count': 0,
            'symbol_breakdown': {}
        }

        all_metrics = []
        for sym in symbols_to_check:
            recent_metrics = [
                m for m in self.execution_metrics[sym]
                if m['timestamp'] >= cutoff_time
            ]

            if recent_metrics:
                symbol_summary = {
                    'executions': len(recent_metrics),
                    'avg_slippage_bps': np.mean([m['slippage_bps'] for m in recent_metrics]),
                    'avg_latency_ms': np.mean([m['latency_ms'] for m in recent_metrics]),
                    'avg_fill_rate': np.mean([m['fill_rate'] for m in recent_metrics]),
                    'avg_quality_score': np.mean([m['quality_score'] for m in recent_metrics]),
                    'violations': sum(len(m['violations']) for m in recent_metrics)
                }
                summary['symbol_breakdown'][sym] = symbol_summary
                all_metrics.extend(recent_metrics)

        if all_metrics:
            summary['total_executions'] = len(all_metrics)
            summary['avg_slippage_bps'] = np.mean([m['slippage_bps'] for m in all_metrics])
            summary['avg_latency_ms'] = np.mean([m['latency_ms'] for m in all_metrics])
            summary['avg_fill_rate'] = np.mean([m['fill_rate'] for m in all_metrics])
            summary['avg_quality_score'] = np.mean([m['quality_score'] for m in all_metrics])
            summary['violation_count'] = sum(len(m['violations']) for m in all_metrics)

        return summary

    def calibrate_slippage_models(self) -> Dict[str, Any]:
        """Trigger slippage model recalibration."""
        return self.slippage_predictor.calibrate_models()

    def save_models(self, filepath: str) -> bool:
        """Save calibrated models to file."""
        try:
            models_data = {}
            for asset_class, model in self.slippage_predictor.models.items():
                models_data[asset_class] = asdict(model)

            with open(filepath, 'w') as f:
                json.dump(models_data, f, indent=2, default=str)

            logger.info(f"Saved {len(models_data)} slippage models to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save models: {e}")
            return False