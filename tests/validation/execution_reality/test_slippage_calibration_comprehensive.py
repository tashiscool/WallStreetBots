#!/usr/bin/env python3
"""
Comprehensive tests for execution_reality/slippage_calibration module.
Tests all public methods, edge cases, and error handling.
Target: 80%+ coverage
"""

import pytest
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, mock_open
from dataclasses import asdict

from backend.validation.execution_reality.slippage_calibration import (
    ExecutionRecord,
    SlippageModel,
    SlippagePredictor,
    ExecutionQualityMonitor
)


class TestExecutionRecord:
    """Test ExecutionRecord dataclass."""

    def test_execution_record_creation(self):
        """Test creating execution record."""
        record = ExecutionRecord(
            timestamp=datetime.now(),
            symbol='SPY',
            side='buy',
            order_type='market',
            intended_quantity=100,
            filled_quantity=100,
            intended_price=None,
            fill_price=450.50,
            market_price_at_order=450.45,
            spread_at_order=0.02,
            volume_at_order=10000,
            volatility_at_order=0.015,
            time_of_day='mid_day',
            days_to_expiry=None,
            order_id='ORDER123',
            latency_ms=50.0,
            partial_fill=False
        )

        assert record.symbol == 'SPY'
        assert record.filled_quantity == 100
        assert record.partial_fill is False


class TestSlippageModel:
    """Test SlippageModel dataclass."""

    def test_slippage_model_creation(self):
        """Test creating slippage model."""
        model = SlippageModel(
            model_type='linear',
            coefficients={'mkt_impact': 0.5, 'intercept': 1.0},
            feature_names=['mkt_impact', 'spread'],
            r_squared=0.75,
            mae=2.5,
            training_samples=100,
            last_updated=datetime.now(),
            asset_class='equity'
        )

        assert model.model_type == 'linear'
        assert model.r_squared == 0.75
        assert model.asset_class == 'equity'


class TestSlippagePredictor:
    """Test SlippagePredictor class."""

    @pytest.fixture
    def predictor(self):
        """Create predictor instance."""
        return SlippagePredictor()

    @pytest.fixture
    def sample_execution(self):
        """Create sample execution record."""
        return ExecutionRecord(
            timestamp=datetime.now(),
            symbol='AAPL',
            side='buy',
            order_type='market',
            intended_quantity=100,
            filled_quantity=100,
            intended_price=None,
            fill_price=150.10,
            market_price_at_order=150.05,
            spread_at_order=0.05,
            volume_at_order=5000,
            volatility_at_order=0.02,
            time_of_day='open',
            days_to_expiry=None,
            order_id='TEST001',
            latency_ms=75.0,
            partial_fill=False
        )

    def test_initialization(self, predictor):
        """Test predictor initialization."""
        assert len(predictor.models) == 0
        assert len(predictor.execution_history) == 0
        assert predictor.calibration_window == 30

    def test_predict_slippage_no_model(self, predictor):
        """Test prediction without calibrated model."""
        market_conditions = {
            'spread_bps': 5.0,
            'volume': 10000,
            'volatility': 0.02
        }

        prediction = predictor.predict_slippage('AAPL', 'buy', 100, market_conditions)

        assert 'expected_slippage_bps' in prediction
        assert 'confidence_interval_bps' in prediction
        assert prediction['prediction_confidence'] < 1.0

    def test_predict_slippage_with_model(self, predictor):
        """Test prediction with calibrated model."""
        # Create mock model
        model = SlippageModel(
            model_type='linear',
            coefficients={'intercept': 1.0, 'log_quantity': 0.5},
            feature_names=['log_quantity'],
            r_squared=0.7,
            mae=2.0,
            training_samples=100,
            last_updated=datetime.now(),
            asset_class='equity'
        )

        predictor.models['equity'] = model

        market_conditions = {
            'spread_bps': 5.0,
            'volume': 10000,
            'volatility': 0.02,
            'time_score': 0.5,
            'market_impact_score': 0.1
        }

        prediction = predictor.predict_slippage('AAPL', 'buy', 100, market_conditions)

        assert prediction['model_r_squared'] == 0.7
        assert prediction['prediction_confidence'] > 0

    def test_record_execution(self, predictor, sample_execution):
        """Test recording execution."""
        predictor.record_execution(sample_execution)

        assert len(predictor.execution_history) == 1
        assert hasattr(sample_execution, 'actual_slippage_bps')

    def test_record_execution_buy_slippage(self, predictor):
        """Test slippage calculation for buy orders."""
        execution = ExecutionRecord(
            timestamp=datetime.now(),
            symbol='AAPL',
            side='buy',
            order_type='market',
            intended_quantity=100,
            filled_quantity=100,
            intended_price=None,
            fill_price=150.10,  # Paid more
            market_price_at_order=150.00,
            spread_at_order=0.05,
            volume_at_order=5000,
            volatility_at_order=0.02,
            time_of_day='mid_day',
            days_to_expiry=None,
            order_id='TEST001',
            latency_ms=50.0,
            partial_fill=False
        )

        predictor.record_execution(execution)

        # Buy slippage should be positive (paid more)
        assert execution.actual_slippage_bps > 0

    def test_record_execution_sell_slippage(self, predictor):
        """Test slippage calculation for sell orders."""
        execution = ExecutionRecord(
            timestamp=datetime.now(),
            symbol='AAPL',
            side='sell',
            order_type='market',
            intended_quantity=100,
            filled_quantity=100,
            intended_price=None,
            fill_price=149.90,  # Received less
            market_price_at_order=150.00,
            spread_at_order=0.05,
            volume_at_order=5000,
            volatility_at_order=0.02,
            time_of_day='mid_day',
            days_to_expiry=None,
            order_id='TEST001',
            latency_ms=50.0,
            partial_fill=False
        )

        predictor.record_execution(execution)

        # Sell slippage should be positive (received less)
        assert execution.actual_slippage_bps > 0

    def test_calibrate_models_insufficient_data(self, predictor):
        """Test calibration with insufficient data."""
        result = predictor.calibrate_models()

        assert result['status'] == 'insufficient_data'
        assert result['executions'] < 50

    def test_calibrate_models_success(self, predictor):
        """Test successful model calibration."""
        # Create 100 executions
        np.random.seed(42)
        for i in range(100):
            execution = ExecutionRecord(
                timestamp=datetime.now() - timedelta(days=i % 30),
                symbol='AAPL',
                side='buy' if i % 2 == 0 else 'sell',
                order_type='market',
                intended_quantity=100 + i,
                filled_quantity=100 + i,
                intended_price=None,
                fill_price=150.0 + np.random.normal(0, 1),
                market_price_at_order=150.0,
                spread_at_order=0.05,
                volume_at_order=5000,
                volatility_at_order=0.02,
                time_of_day='mid_day',
                days_to_expiry=None,
                order_id=f'TEST{i:03d}',
                latency_ms=50.0 + np.random.normal(0, 10),
                partial_fill=False
            )

            predictor.record_execution(execution)

        result = predictor.calibrate_models()

        # Should have calibrated models
        assert len(predictor.models) > 0

    def test_classify_asset(self, predictor):
        """Test asset classification."""
        assert predictor._classify_asset('SPY') == 'etf'
        assert predictor._classify_asset('QQQ') == 'etf'
        assert predictor._classify_asset('AAPL') == 'equity'
        assert predictor._classify_asset('AAPL240315C00150000') == 'option'

    def test_extract_features(self, predictor):
        """Test feature extraction."""
        market_conditions = {
            'spread_bps': 5.0,
            'volume': 10000,
            'volatility': 0.02,
            'time_score': 0.5,
            'market_impact_score': 0.1
        }

        features = predictor._extract_features('AAPL', 'buy', 100, market_conditions)

        assert features.shape == (1, 7)
        assert all(np.isfinite(features[0]))

    def test_train_linear_model(self, predictor):
        """Test linear model training."""
        # Create sample executions
        executions = []
        np.random.seed(42)

        for i in range(50):
            execution = ExecutionRecord(
                timestamp=datetime.now() - timedelta(days=i % 20),
                symbol='AAPL',
                side='buy',
                order_type='market',
                intended_quantity=100,
                filled_quantity=100,
                intended_price=None,
                fill_price=150.05,
                market_price_at_order=150.00,
                spread_at_order=0.05,
                volume_at_order=5000,
                volatility_at_order=0.02,
                time_of_day='mid_day',
                days_to_expiry=None,
                order_id=f'TEST{i}',
                latency_ms=50.0,
                partial_fill=False
            )
            execution.actual_slippage_bps = 3.0 + np.random.normal(0, 0.5)
            executions.append(execution)

        model = predictor._train_linear_model(executions, 'equity')

        assert model.model_type == 'linear'
        assert model.r_squared >= 0
        assert model.training_samples == 50

    def test_train_rf_model(self, predictor):
        """Test random forest model training."""
        # Create sample executions
        executions = []
        np.random.seed(42)

        for i in range(50):
            execution = ExecutionRecord(
                timestamp=datetime.now() - timedelta(days=i % 20),
                symbol='AAPL',
                side='buy',
                order_type='market',
                intended_quantity=100,
                filled_quantity=100,
                intended_price=None,
                fill_price=150.05,
                market_price_at_order=150.00,
                spread_at_order=0.05,
                volume_at_order=5000,
                volatility_at_order=0.02,
                time_of_day='mid_day',
                days_to_expiry=None,
                order_id=f'TEST{i}',
                latency_ms=50.0,
                partial_fill=False
            )
            execution.actual_slippage_bps = 3.0 + np.random.normal(0, 0.5)
            executions.append(execution)

        model = predictor._train_rf_model(executions, 'equity')

        assert model.model_type == 'random_forest'
        assert model.r_squared >= 0

    def test_prepare_training_data(self, predictor):
        """Test training data preparation."""
        executions = []

        for i in range(10):
            execution = ExecutionRecord(
                timestamp=datetime.now(),
                symbol='AAPL',
                side='buy',
                order_type='market',
                intended_quantity=100,
                filled_quantity=100,
                intended_price=None,
                fill_price=150.05,
                market_price_at_order=150.00,
                spread_at_order=0.05,
                volume_at_order=5000,
                volatility_at_order=0.02,
                time_of_day='mid_day',
                days_to_expiry=None,
                order_id=f'TEST{i}',
                latency_ms=50.0,
                partial_fill=False
            )
            execution.actual_slippage_bps = 3.0
            executions.append(execution)

        X, y = predictor._prepare_training_data(executions)

        assert X.shape[0] == 10
        assert y.shape[0] == 10

    def test_time_to_score(self, predictor):
        """Test time of day scoring."""
        assert predictor._time_to_score('open') == 0.1
        assert predictor._time_to_score('close') == 0.2
        assert predictor._time_to_score('mid_day') == 0.5

    def test_calculate_market_impact_score(self, predictor):
        """Test market impact score calculation."""
        execution = ExecutionRecord(
            timestamp=datetime.now(),
            symbol='AAPL',
            side='buy',
            order_type='market',
            intended_quantity=1000,
            filled_quantity=1000,
            intended_price=None,
            fill_price=150.0,
            market_price_at_order=150.0,
            spread_at_order=0.05,
            volume_at_order=10000,
            volatility_at_order=0.02,
            time_of_day='mid_day',
            days_to_expiry=None,
            order_id='TEST',
            latency_ms=50.0,
            partial_fill=False
        )

        impact = predictor._calculate_market_impact_score(execution)

        assert 0 <= impact <= 1.0

    def test_predict_linear(self, predictor):
        """Test linear model prediction."""
        model = SlippageModel(
            model_type='linear',
            coefficients={
                'intercept': 1.0,
                'log_quantity': 0.5,
                'spread_bps': 0.3,
                'volume': 0.0001,
                'volatility': 0.2,
                'is_buy': 0.1,
                'time_score': 0.05,
                'market_impact': 0.15
            },
            feature_names=['log_quantity', 'spread_bps', 'volume', 'volatility', 'is_buy', 'time_score', 'market_impact'],
            r_squared=0.7,
            mae=2.0,
            training_samples=100,
            last_updated=datetime.now(),
            asset_class='equity'
        )

        features = np.array([[4.6, 5.0, 1000, 0.2, 1, 0.5, 0.1]])

        prediction = predictor._predict_linear(model, features)

        assert prediction >= 0

    def test_default_slippage_estimate(self, predictor):
        """Test default slippage estimation."""
        market_conditions = {
            'spread_bps': 10.0,
            'volatility': 0.03
        }

        estimate = predictor._default_slippage_estimate('AAPL', 'buy', 100, market_conditions)

        assert 'expected_slippage_bps' in estimate
        assert estimate['prediction_confidence'] == 0.3


class TestExecutionQualityMonitor:
    """Test ExecutionQualityMonitor class."""

    @pytest.fixture
    def monitor(self):
        """Create monitor instance."""
        return ExecutionQualityMonitor()

    @pytest.fixture
    def sample_execution(self):
        """Create sample execution."""
        execution = ExecutionRecord(
            timestamp=datetime.now(),
            symbol='AAPL',
            side='buy',
            order_type='market',
            intended_quantity=100,
            filled_quantity=100,
            intended_price=None,
            fill_price=150.05,
            market_price_at_order=150.00,
            spread_at_order=0.05,
            volume_at_order=5000,
            volatility_at_order=0.02,
            time_of_day='mid_day',
            days_to_expiry=None,
            order_id='TEST001',
            latency_ms=50.0,
            partial_fill=False
        )
        execution.actual_slippage_bps = 3.33
        return execution

    def test_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor.slippage_predictor is not None
        assert len(monitor.execution_metrics) == 0
        assert 'max_slippage_bps' in monitor.quality_thresholds

    def test_monitor_execution(self, monitor, sample_execution):
        """Test monitoring single execution."""
        metrics = monitor.monitor_execution(sample_execution)

        assert 'timestamp' in metrics
        assert 'symbol' in metrics
        assert 'fill_rate' in metrics
        assert 'latency_ms' in metrics
        assert 'slippage_bps' in metrics
        assert 'quality_score' in metrics
        assert 'violations' in metrics

    def test_monitor_execution_high_slippage(self, monitor):
        """Test monitoring with high slippage."""
        execution = ExecutionRecord(
            timestamp=datetime.now(),
            symbol='AAPL',
            side='buy',
            order_type='market',
            intended_quantity=100,
            filled_quantity=100,
            intended_price=None,
            fill_price=151.00,  # High slippage
            market_price_at_order=150.00,
            spread_at_order=0.05,
            volume_at_order=5000,
            volatility_at_order=0.02,
            time_of_day='mid_day',
            days_to_expiry=None,
            order_id='TEST001',
            latency_ms=50.0,
            partial_fill=False
        )
        execution.actual_slippage_bps = 66.67  # High slippage

        metrics = monitor.monitor_execution(execution)

        assert len(metrics['violations']) > 0
        assert 'High slippage' in metrics['violations'][0]

    def test_monitor_execution_high_latency(self, monitor, sample_execution):
        """Test monitoring with high latency."""
        sample_execution.latency_ms = 300.0

        metrics = monitor.monitor_execution(sample_execution)

        assert any('latency' in v.lower() for v in metrics['violations'])

    def test_monitor_execution_partial_fill(self, monitor):
        """Test monitoring with partial fill."""
        execution = ExecutionRecord(
            timestamp=datetime.now(),
            symbol='AAPL',
            side='buy',
            order_type='market',
            intended_quantity=100,
            filled_quantity=90,  # Partial fill
            intended_price=None,
            fill_price=150.05,
            market_price_at_order=150.00,
            spread_at_order=0.05,
            volume_at_order=5000,
            volatility_at_order=0.02,
            time_of_day='mid_day',
            days_to_expiry=None,
            order_id='TEST001',
            latency_ms=50.0,
            partial_fill=True
        )
        execution.actual_slippage_bps = 3.33

        metrics = monitor.monitor_execution(execution)

        assert metrics['fill_rate'] == 0.9

    def test_calculate_adverse_selection(self, monitor):
        """Test adverse selection calculation."""
        execution = ExecutionRecord(
            timestamp=datetime.now(),
            symbol='AAPL',
            side='buy',
            order_type='market',
            intended_quantity=100,
            filled_quantity=100,
            intended_price=None,
            fill_price=150.05,
            market_price_at_order=150.00,
            spread_at_order=0.05,
            volume_at_order=5000,
            volatility_at_order=0.02,
            time_of_day='mid_day',
            days_to_expiry=None,
            order_id='TEST001',
            latency_ms=150.0,
            partial_fill=False
        )

        adverse_selection = monitor._calculate_adverse_selection(execution)

        assert adverse_selection >= 0

    def test_calculate_quality_score(self, monitor):
        """Test quality score calculation."""
        metrics = {
            'slippage_bps': 5.0,
            'latency_ms': 100.0,
            'fill_rate': 1.0,
            'adverse_selection_bps': 1.0
        }

        score = monitor._calculate_quality_score(metrics)

        assert 0 <= score <= 1

    def test_calculate_quality_score_poor(self, monitor):
        """Test quality score with poor execution."""
        metrics = {
            'slippage_bps': 50.0,  # High slippage
            'latency_ms': 500.0,    # High latency
            'fill_rate': 0.8,       # Partial fill
            'adverse_selection_bps': 10.0
        }

        score = monitor._calculate_quality_score(metrics)

        assert score < 0.5

    def test_get_execution_summary(self, monitor, sample_execution):
        """Test getting execution summary."""
        # Add multiple executions
        for i in range(10):
            execution = ExecutionRecord(
                timestamp=datetime.now(),
                symbol='AAPL',
                side='buy',
                order_type='market',
                intended_quantity=100,
                filled_quantity=100,
                intended_price=None,
                fill_price=150.05,
                market_price_at_order=150.00,
                spread_at_order=0.05,
                volume_at_order=5000,
                volatility_at_order=0.02,
                time_of_day='mid_day',
                days_to_expiry=None,
                order_id=f'TEST{i:03d}',
                latency_ms=50.0 + i,
                partial_fill=False
            )
            execution.actual_slippage_bps = 3.0 + i * 0.1

            monitor.monitor_execution(execution)

        summary = monitor.get_execution_summary(symbol='AAPL', hours=24)

        assert summary['symbols_analyzed'] == 1
        assert summary['total_executions'] == 10
        assert 'avg_slippage_bps' in summary
        assert 'avg_latency_ms' in summary

    def test_get_execution_summary_no_data(self, monitor):
        """Test summary with no data."""
        summary = monitor.get_execution_summary()

        assert summary['total_executions'] == 0

    def test_calibrate_slippage_models(self, monitor):
        """Test triggering model calibration."""
        # Add enough executions
        for i in range(60):
            execution = ExecutionRecord(
                timestamp=datetime.now() - timedelta(days=i % 25),
                symbol='AAPL',
                side='buy',
                order_type='market',
                intended_quantity=100,
                filled_quantity=100,
                intended_price=None,
                fill_price=150.05,
                market_price_at_order=150.00,
                spread_at_order=0.05,
                volume_at_order=5000,
                volatility_at_order=0.02,
                time_of_day='mid_day',
                days_to_expiry=None,
                order_id=f'TEST{i:03d}',
                latency_ms=50.0,
                partial_fill=False
            )
            execution.actual_slippage_bps = 3.0

            monitor.monitor_execution(execution)

        result = monitor.calibrate_slippage_models()

        # Should have attempted calibration
        assert isinstance(result, dict)

    def test_save_models(self, monitor, tmp_path):
        """Test saving models to file."""
        # Create a model
        model = SlippageModel(
            model_type='linear',
            coefficients={'intercept': 1.0},
            feature_names=['test'],
            r_squared=0.7,
            mae=2.0,
            training_samples=100,
            last_updated=datetime.now(),
            asset_class='equity'
        )

        monitor.slippage_predictor.models['equity'] = model

        filepath = tmp_path / 'models.json'
        success = monitor.save_models(str(filepath))

        assert success is True
        assert filepath.exists()

        # Verify content
        with open(filepath, 'r') as f:
            data = json.load(f)

        assert 'equity' in data


class TestEdgeCasesAndErrors:
    """Test edge cases and error handling."""

    def test_zero_volume_execution(self):
        """Test with zero volume."""
        predictor = SlippagePredictor()

        execution = ExecutionRecord(
            timestamp=datetime.now(),
            symbol='AAPL',
            side='buy',
            order_type='market',
            intended_quantity=100,
            filled_quantity=100,
            intended_price=None,
            fill_price=150.05,
            market_price_at_order=150.00,
            spread_at_order=0.05,
            volume_at_order=0,  # Zero volume
            volatility_at_order=0.02,
            time_of_day='mid_day',
            days_to_expiry=None,
            order_id='TEST001',
            latency_ms=50.0,
            partial_fill=False
        )

        # Should handle gracefully
        impact = predictor._calculate_market_impact_score(execution)
        assert impact == 0.1

    def test_extreme_slippage(self):
        """Test with extreme slippage values."""
        monitor = ExecutionQualityMonitor()

        execution = ExecutionRecord(
            timestamp=datetime.now(),
            symbol='AAPL',
            side='buy',
            order_type='market',
            intended_quantity=100,
            filled_quantity=100,
            intended_price=None,
            fill_price=200.00,  # Extreme slippage
            market_price_at_order=150.00,
            spread_at_order=0.05,
            volume_at_order=5000,
            volatility_at_order=0.02,
            time_of_day='mid_day',
            days_to_expiry=None,
            order_id='TEST001',
            latency_ms=50.0,
            partial_fill=False
        )
        execution.actual_slippage_bps = 33333.33

        metrics = monitor.monitor_execution(execution)

        assert len(metrics['violations']) > 0

    def test_model_save_failure(self, tmp_path):
        """Test handling of model save failure."""
        monitor = ExecutionQualityMonitor()

        # Try to save to invalid path
        success = monitor.save_models('/invalid/path/models.json')

        assert success is False
