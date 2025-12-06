"""Standalone Test Suite for All New Components

Tests components without Django dependencies.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from decimal import Decimal
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.insert(0, project_root)

# Import components with absolute paths
from backend.tradingbot.production.testing.portfolio_allocator_tester import PortfolioAllocatorTester
from backend.tradingbot.production.monitoring.validation_accuracy_monitor import ValidationAccuracyMonitor
from backend.tradingbot.production.optimization.allocation_weight_tuner import AllocationWeightTuner, AllocationWeights
from backend.tradingbot.production.execution.advanced_slippage_model import AdvancedSlippageModel, MarketMicrostructureFeatures
from backend.tradingbot.production.backtesting.walk_forward_backtester import WalkForwardBacktester


class MockIntegrationManager:
    """Mock integration manager for testing."""
    
    def __init__(self):
        self.trades = []
        self.active_positions = {}
    
    async def get_portfolio_value(self) -> Decimal:
        return Decimal("100000.00")
    
    async def get_all_positions(self) -> dict:
        return {}


class MockStrategy:
    """Mock strategy for testing."""
    
    def __init__(self, name: str, returns: pd.Series = None):
        self.strategy_name = name
        self.returns = returns or self._generate_mock_returns()
        self.is_running = True
    
    def _generate_mock_returns(self) -> pd.Series:
        """Generate mock returns."""
        dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
        returns = np.random.normal(0.001, 0.02, 60)  # ~0.1% daily return, 2% volatility
        return pd.Series(returns, index=dates)
    
    def get_strategy_status(self) -> dict:
        return {
            'sharpe_ratio': 0.8,
            'expected_return': 0.12,
            'volatility': 0.20,
            'max_drawdown': 0.15
        }
    
    async def get_returns_history(self) -> pd.Series:
        return self.returns
    
    async def get_recent_trades(self, limit: int = 50) -> list:
        return []


def test_portfolio_allocator_tester():
    """Test portfolio allocator tester."""
    print("\n" + "="*60)
    print("TESTING PORTFOLIO ALLOCATOR TESTER")
    print("="*60)
    
    try:
        mock_integration = MockIntegrationManager()
        tester = PortfolioAllocatorTester(mock_integration)
        
        # Create mock strategies
        strategies = {
            'strategy1': MockStrategy('strategy1'),
            'strategy2': MockStrategy('strategy2'),
            'strategy3': MockStrategy('strategy3')
        }
        
        # Run test
        result = asyncio.run(tester.test_with_real_strategies(
            strategies=strategies,
            portfolio_value=Decimal("100000.00")
        ))
        
        assert 'allocation_result' in result, "Missing allocation_result"
        assert 'metrics' in result, "Missing metrics"
        assert 'validation' in result, "Missing validation"
        
        print("✅ Portfolio Allocator Tester: PASSED")
        print(f"   - Tested {result['metrics']['total_strategies']} strategies")
        print(f"   - Portfolio Sharpe: {result['metrics']['portfolio_sharpe']:.2f}")
        print(f"   - Total Allocation: {result['metrics']['total_allocation']:.1%}")
        
        return True
    except Exception as e:
        print(f"❌ Portfolio Allocator Tester: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation_accuracy_monitor():
    """Test validation accuracy monitor."""
    print("\n" + "="*60)
    print("TESTING VALIDATION ACCURACY MONITOR")
    print("="*60)
    
    try:
        monitor = ValidationAccuracyMonitor(lookback_days=30)
        
        # Record some validation results
        for i in range(20):
            validation_result = {
                'recommended_action': 'execute' if i % 3 != 0 else 'reject',
                'strength_score': 60 + np.random.randint(-20, 20)
            }
            actual_outcome = {
                'win': i % 2 == 0,
                'return': np.random.normal(0.01, 0.05)
            }
            monitor.record_validation('strategy1', validation_result, actual_outcome)
        
        # Calculate metrics
        metrics = monitor.calculate_metrics('strategy1')
        assert metrics is not None, "Metrics should not be None"
        assert metrics.accuracy_rate >= 0, "Accuracy rate should be non-negative"
        assert metrics.precision >= 0, "Precision should be non-negative"
        assert metrics.recall >= 0, "Recall should be non-negative"
        
        # Generate report
        report = monitor.generate_report()
        assert report is not None, "Report should not be None"
        assert 'strategy1' in report.strategies, "Strategy should be in report"
        
        # Get dashboard data
        dashboard = monitor.get_dashboard_data()
        assert 'overall_metrics' in dashboard, "Dashboard should have overall_metrics"
        
        print("✅ Validation Accuracy Monitor: PASSED")
        print(f"   - Accuracy Rate: {metrics.accuracy_rate:.1%}")
        print(f"   - Precision: {metrics.precision:.1%}")
        print(f"   - F1 Score: {metrics.f1_score:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Validation Accuracy Monitor: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_allocation_weight_tuner():
    """Test allocation weight tuner."""
    print("\n" + "="*60)
    print("TESTING ALLOCATION WEIGHT TUNER")
    print("="*60)
    
    try:
        tuner = AllocationWeightTuner()
        
        # Create mock historical returns
        dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
        historical_returns = {
            'strategy1': pd.Series(np.random.normal(0.001, 0.02, 90), index=dates),
            'strategy2': pd.Series(np.random.normal(0.0008, 0.018, 90), index=dates),
            'strategy3': pd.Series(np.random.normal(0.0012, 0.022, 90), index=dates)
        }
        
        # Run tuning
        result = tuner.tune_weights(
            historical_returns=historical_returns,
            optimization_objective='sharpe'
        )
        
        assert result is not None, "Result should not be None"
        assert result.optimal_weights is not None, "Optimal weights should not be None"
        assert len(result.backtest_results) > 0, "Should have backtest results"
        
        # Check weights sum to approximately 1.0
        weights = result.optimal_weights
        total_weight = weights.risk_parity_weight + weights.sharpe_weight_weight + weights.priority_weight_weight
        assert abs(total_weight - 1.0) < 0.01, f"Weights should sum to 1.0, got {total_weight}"
        
        print("✅ Allocation Weight Tuner: PASSED")
        print(f"   - Optimal Weights: {result.optimal_weights.to_dict()}")
        print(f"   - Backtest Results: {len(result.backtest_results)}")
        if result.improvement:
            print(f"   - Sharpe Improvement: {result.improvement.get('sharpe_improvement', 0):.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Allocation Weight Tuner: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_advanced_slippage_model():
    """Test advanced slippage model."""
    print("\n" + "="*60)
    print("TESTING ADVANCED SLIPPAGE MODEL")
    print("="*60)
    
    try:
        model = AdvancedSlippageModel(model_type='random_forest')
        
        # Test prediction without training (should use rule-based)
        market_conditions = {
            'price': 100.0,
            'volume': 1000000,
            'volatility': 0.20
        }
        
        prediction = model.predict_slippage(
            symbol='AAPL',
            side='buy',
            quantity=100,
            market_conditions=market_conditions
        )
        
        assert prediction is not None, "Prediction should not be None"
        assert prediction.expected_slippage_bps >= 0, "Slippage should be non-negative"
        assert prediction.model_type in ['rule_based', 'random_forest'], "Model type should be valid"
        
        print("✅ Advanced Slippage Model (Rule-based): PASSED")
        print(f"   - Expected Slippage: {prediction.expected_slippage_bps:.2f} bps")
        print(f"   - Model Type: {prediction.model_type}")
        
        # Test with training data
        execution_data = []
        for i in range(60):
            microstructure = MarketMicrostructureFeatures(
                bid_ask_spread=0.001 + np.random.random() * 0.002,
                order_book_imbalance=np.random.normal(0, 0.1),
                volume_profile=0.3 + np.random.random() * 0.4,
                volatility=0.15 + np.random.random() * 0.1,
                time_of_day=np.random.random(),
                day_of_week=np.random.randint(0, 5),
                recent_volume=500000 + np.random.random() * 1000000,
                price_momentum=np.random.normal(0, 0.01),
                liquidity_score=0.4 + np.random.random() * 0.3
            )
            execution_data.append({
                'symbol': 'AAPL',
                'side': 'buy' if i % 2 == 0 else 'sell',
                'quantity': 100 + np.random.randint(0, 900),
                'market_conditions': {
                    'price': 100.0 + np.random.normal(0, 5),
                    'volume': 1000000,
                    'volatility': 0.20
                },
                'microstructure_features': microstructure,
                'actual_slippage_bps': 5.0 + np.random.normal(0, 3)
            })
        
        # Train model
        training_metrics = model.train_model(execution_data)
        assert 'r_squared' in training_metrics or 'error' in training_metrics, "Should have training results"
        
        if 'error' not in training_metrics:
            print("✅ Advanced Slippage Model (ML Training): PASSED")
            print(f"   - R² Score: {training_metrics.get('r_squared', 0):.3f}")
            
            # Test prediction with trained model
            prediction = model.predict_slippage(
                symbol='AAPL',
                side='buy',
                quantity=100,
                market_conditions=market_conditions,
                microstructure_features=execution_data[0]['microstructure_features']
            )
            print(f"   - ML Prediction: {prediction.expected_slippage_bps:.2f} bps")
            print(f"   - Model Confidence: {prediction.model_confidence:.2f}")
        else:
            print(f"⚠️  Training skipped: {training_metrics.get('error')}")
        
        return True
    except Exception as e:
        print(f"❌ Advanced Slippage Model: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_walk_forward_backtester():
    """Test walk-forward backtester."""
    print("\n" + "="*60)
    print("TESTING WALK-FORWARD BACKTESTER")
    print("="*60)
    
    try:
        backtester = WalkForwardBacktester(
            train_window_days=30,
            test_window_days=10,
            step_days=10
        )
        
        # Create mock strategy function
        def mock_strategy(data: pd.DataFrame, params: dict) -> list:
            """Mock strategy that generates random trades."""
            trades = []
            for i in range(0, len(data), 5):
                trades.append({
                    'entry_date': data.index[i] if isinstance(data.index, pd.DatetimeIndex) else datetime.now(),
                    'exit_date': data.index[min(i+3, len(data)-1)] if isinstance(data.index, pd.DatetimeIndex) else datetime.now(),
                    'return': np.random.normal(0.01, 0.05)
                })
            return trades
        
        # Create mock historical data
        dates = pd.date_range(start='2024-01-01', periods=120, freq='D')
        historical_data = pd.DataFrame({
            'price': 100 + np.cumsum(np.random.normal(0, 1, 120)),
            'volume': np.random.randint(1000000, 5000000, 120)
        }, index=dates)
        
        # Run walk-forward backtest
        report = backtester.run_walk_forward(
            strategy_function=mock_strategy,
            historical_data=historical_data,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 4, 30),
            strategy_params={}
        )
        
        assert report is not None, "Report should not be None"
        assert report.total_windows > 0, "Should have at least one window"
        assert len(report.results) > 0, "Should have results"
        
        # Check metrics
        overall = report.overall_metrics
        assert 'average_test_sharpe' in overall, "Should have average test Sharpe"
        assert 'average_stability' in overall, "Should have average stability"
        
        print("✅ Walk-Forward Backtester: PASSED")
        print(f"   - Total Windows: {report.total_windows}")
        print(f"   - Average Test Sharpe: {overall.get('average_test_sharpe', 0):.2f}")
        print(f"   - Average Stability: {overall.get('average_stability', 0):.2f}")
        print(f"   - Recommendations: {len(report.recommendations)}")
        print(f"   - Warnings: {len(report.warnings)}")
        
        return True
    except Exception as e:
        print(f"❌ Walk-Forward Backtester: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("COMPREHENSIVE COMPONENT TEST SUITE")
    print("="*60)
    
    results = []
    
    # Run all tests
    results.append(("Portfolio Allocator Tester", test_portfolio_allocator_tester()))
    results.append(("Validation Accuracy Monitor", test_validation_accuracy_monitor()))
    results.append(("Allocation Weight Tuner", test_allocation_weight_tuner()))
    results.append(("Advanced Slippage Model", test_advanced_slippage_model()))
    results.append(("Walk-Forward Backtester", test_walk_forward_backtester()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! All components are working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

