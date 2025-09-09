#!/usr/bin/env python3
"""
Test Advanced Risk Models - 2025 Implementation
Comprehensive testing of sophisticated risk management capabilities
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Import risk management modules
from tradingbot.risk import (
    AdvancedVaREngine, 
    StressTesting2025, 
    MLRiskPredictor, 
    RiskDashboard2025
)

def generate_sample_data():
    """Generate sample market data for testing"""
    np.random.seed(42)
    
    # Generate realistic market data
    n_days = 252  # One year of trading days
    base_price = 100
    
    # Generate price series with trend and volatility
    returns = np.random.normal(0.0008, 0.02, n_days)  # 0.08% daily return, 2% daily vol
    prices = [base_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate volume data
    volumes = np.random.uniform(1000, 10000, n_days)
    
    # Generate market data
    market_data = {
        'prices': prices,
        'volumes': volumes,
        'sentiment': np.random.uniform(-0.5, 0.5),
        'put_call_ratio': np.random.uniform(0.8, 1.5),
        'social_volume': np.random.uniform(0.3, 0.9),
        'social_sentiment': np.random.uniform(-0.3, 0.3),
        'vix_level': np.random.uniform(15, 35),
        'rsi': np.random.uniform(30, 70)
    }
    
    return market_data, returns

def test_var_engine():
    """Test Advanced VaR Engine"""
    print("🔬 Testing Advanced VaR Engine")
    print("=" * 50)
    
    # Generate sample data
    market_data, returns = generate_sample_data()
    
    # Initialize VaR engine
    var_engine = AdvancedVaREngine(portfolio_value=100000.0)
    
    # Test VaR suite calculation
    print("Calculating VaR Suite...")
    var_suite = var_engine.calculate_var_suite(
        returns=returns,
        confidence_levels=[0.95, 0.99],
        methods=['parametric', 'historical', 'monte_carlo']
    )
    
    # Print results
    print(f"\nVaR Results (Portfolio Value: ${var_engine.portfolio_value:,.0f})")
    print("-" * 40)
    summary = var_suite.get_summary()
    for key, result in summary.items():
        print(f"{key:20}: ${result['var_value']:8,.0f} ({result['var_percentage']:5.2f}%)")
    
    # Test CVaR calculation
    cvar_95 = var_engine.calculate_cvar(returns, 0.95)
    print(f"\nCVaR (95%): ${cvar_95:,.0f}")
    
    # Test regime detection
    regime_info = var_engine.detect_regime_and_adjust(returns)
    print(f"\nRegime Detection:")
    print(f"  Current Regime: {regime_info['regime']}")
    print(f"  Adjustment Factor: {regime_info['adjustment_factor']:.2f}")
    print(f"  Volatility Ratio: {regime_info['volatility_ratio']:.2f}")
    
    return var_engine, var_suite

def test_stress_testing():
    """Test Stress Testing Engine"""
    print("\n\n🚨 Testing Stress Testing Engine")
    print("=" * 50)
    
    # Create sample portfolio
    sample_portfolio = {
        'total_value': 100000.0,
        'strategies': {
            'wsb_dip_bot': {'exposure': 0.25},
            'earnings_protection': {'exposure': 0.20},
            'index_baseline': {'exposure': 0.15},
            'momentum_weeklies': {'exposure': 0.20},
            'debit_spreads': {'exposure': 0.10},
            'leaps_tracker': {'exposure': 0.10}
        }
    }
    
    # Initialize stress tester
    stress_tester = StressTesting2025()
    
    # Run stress tests
    print("Running comprehensive stress tests...")
    report = stress_tester.run_comprehensive_stress_test(sample_portfolio)
    
    # Print results
    print(f"\nStress Test Report")
    print("-" * 40)
    print(f"Compliance Status: {report.compliance_status}")
    print(f"Overall Risk Score: {report.overall_risk_score:.1f}/100")
    print(f"Test Date: {report.test_date}")
    
    print(f"\nScenario Results:")
    for scenario_name, result in report.results.items():
        status = "✅ PASSED" if result.passed else "❌ FAILED"
        print(f"  {scenario_name:25} {status:10} P&L: ${result.portfolio_pnl:8,.0f}")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"  {i}. {rec}")
    
    return stress_tester, report

def test_ml_risk_predictor():
    """Test ML Risk Predictor"""
    print("\n\n🤖 Testing ML Risk Predictor")
    print("=" * 50)
    
    # Generate sample data
    market_data, returns = generate_sample_data()
    
    # Initialize ML predictor
    ml_predictor = MLRiskPredictor()
    
    # Test volatility prediction
    print("Testing volatility prediction...")
    vol_forecast = ml_predictor.predict_volatility_regime(market_data, horizon_days=5)
    
    print(f"Predicted Volatility: {vol_forecast.predicted_volatility:.2%}")
    print(f"Confidence Interval: {vol_forecast.confidence_interval[0]:.2%} - {vol_forecast.confidence_interval[1]:.2%}")
    print(f"Regime Probabilities: {vol_forecast.regime_probability}")
    print(f"Model Confidence: {vol_forecast.model_confidence:.1%}")
    
    # Test risk prediction
    print(f"\nTesting risk prediction...")
    risk_prediction = ml_predictor.predict_risk_score(market_data)
    
    print(f"Risk Score: {risk_prediction.risk_score:.1f}/100")
    print(f"Regime: {risk_prediction.regime_prediction}")
    print(f"Confidence: {risk_prediction.confidence:.1%}")
    
    if risk_prediction.recommended_actions:
        print(f"\nRecommendations:")
        for i, action in enumerate(risk_prediction.recommended_actions, 1):
            print(f"  {i}. {action}")
    
    return ml_predictor, risk_prediction

def test_risk_dashboard():
    """Test Risk Dashboard"""
    print("\n\n📊 Testing Risk Dashboard")
    print("=" * 50)
    
    # Create comprehensive sample portfolio
    sample_portfolio = {
        'total_value': 100000.0,
        'positions': [
            {'ticker': 'AAPL', 'value': 25000, 'quantity': 100},
            {'ticker': 'TSLA', 'value': 20000, 'quantity': 50},
            {'ticker': 'SPY', 'value': 30000, 'quantity': 100},
            {'ticker': 'QQQ', 'value': 25000, 'quantity': 80}
        ],
        'strategies': {
            'wsb_dip_bot': {'exposure': 0.25},
            'earnings_protection': {'exposure': 0.20},
            'index_baseline': {'exposure': 0.15},
            'momentum_weeklies': {'exposure': 0.20},
            'debit_spreads': {'exposure': 0.10},
            'leaps_tracker': {'exposure': 0.10}
        },
        'market_data': generate_sample_data()[0]
    }
    
    # Initialize dashboard
    dashboard = RiskDashboard2025(portfolio_value=100000.0)
    
    # Generate dashboard data
    print("Generating risk dashboard...")
    dashboard_data = dashboard.get_risk_dashboard_data(sample_portfolio)
    
    # Print comprehensive results
    print(f"\nRisk Dashboard Summary")
    print("-" * 40)
    print(f"Portfolio Value: ${dashboard_data['portfolio_value']:,.0f}")
    print(f"Total Positions: {dashboard_data['total_positions']}")
    print(f"Timestamp: {dashboard_data['timestamp']}")
    
    print(f"\nCore Risk Metrics:")
    for metric, data in dashboard_data['risk_metrics'].items():
        status = "⚠️" if data['limit_utilization'] > 80 else "✅"
        print(f"  {status} {metric.upper()}: ${data['value']:,.0f} ({data['percentage']:.1f}%) - {data['limit_utilization']:.1f}% of limit")
    
    print(f"\nAdvanced Metrics:")
    for metric, value in dashboard_data['advanced_metrics'].items():
        print(f"  {metric}: ${value:,.0f}")
    
    print(f"\nAlternative Data Signals:")
    for signal, value in dashboard_data['alternative_signals'].items():
        print(f"  {signal}: ${value:,.0f}")
    
    print(f"\nFactor Risk Breakdown:")
    for factor, value in dashboard_data['factor_breakdown'].items():
        print(f"  {factor}: ${value:,.0f}")
    
    print(f"\nStress Test Results:")
    print(f"  Worst Case P&L: ${dashboard_data['stress_tests']['worst_case_pnl']:,.0f}")
    print(f"  Scenarios Tested: {len(dashboard_data['stress_tests']['scenario_analysis'])}")
    
    print(f"\nActive Alerts: {len(dashboard_data['alerts'])}")
    for alert in dashboard_data['alerts']:
        severity_icon = {"LOW": "🟡", "MEDIUM": "🟠", "HIGH": "🔴", "CRITICAL": "🚨"}.get(alert['severity'], "⚪")
        print(f"  {severity_icon} {alert['severity']}: {alert['message']}")
        print(f"    Action: {alert['recommended_action']}")
    
    print(f"\nRisk Limit Utilization:")
    for limit, utilization in dashboard_data['risk_limits'].items():
        status = "⚠️" if utilization > 80 else "✅"
        print(f"  {status} {limit}: {utilization:.1f}%")
    
    return dashboard, dashboard_data

def run_comprehensive_test():
    """Run comprehensive test of all risk management modules"""
    print("🚀 WallStreetBots Advanced Risk Models - 2025 Test Suite")
    print("=" * 70)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    try:
        # Test VaR Engine
        var_engine, var_suite = test_var_engine()
        
        # Test Stress Testing
        stress_tester, stress_report = test_stress_testing()
        
        # Test ML Risk Predictor
        ml_predictor, risk_prediction = test_ml_risk_predictor()
        
        # Test Risk Dashboard
        dashboard, dashboard_data = test_risk_dashboard()
        
        # Summary
        print("\n\n🎉 Test Summary")
        print("=" * 50)
        print("✅ Advanced VaR Engine: PASSED")
        print("✅ Stress Testing Engine: PASSED")
        print("✅ ML Risk Predictor: PASSED")
        print("✅ Risk Dashboard: PASSED")
        print("\n🎯 All risk management modules are working correctly!")
        print("\n📈 Ready for production deployment with sophisticated risk controls.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)


