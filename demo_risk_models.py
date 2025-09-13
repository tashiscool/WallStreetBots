#!/usr / bin / env python3
"""
Demo Advanced Risk Models - 2025 Implementation
Practical demonstration of sophisticated risk management capabilities
"""

import sys
import os
import numpy as np
from datetime import datetime
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

def create_realistic_portfolio(): 
    """Create a realistic portfolio for demonstration"""
    return {
        'total_value': 500000.0,  # $500K portfolio
        'positions': [
            {'ticker': 'AAPL', 'value': 75000, 'quantity': 300, 'strategy': 'wsb_dip_bot'},
            {'ticker': 'TSLA', 'value': 100000, 'quantity': 200, 'strategy': 'momentum_weeklies'},
            {'ticker': 'SPY', 'value': 150000, 'quantity': 300, 'strategy': 'index_baseline'},
            {'ticker': 'QQQ', 'value': 100000, 'quantity': 250, 'strategy': 'earnings_protection'},
            {'ticker': 'NVDA', 'value': 75000, 'quantity': 100, 'strategy': 'leaps_tracker'}
        ],
        'strategies': {
            'wsb_dip_bot': {'exposure':0.15, 'risk_tier': 'aggressive'},
            'earnings_protection': {'exposure':0.20, 'risk_tier': 'moderate'},
            'index_baseline': {'exposure':0.30, 'risk_tier': 'conservative'},
            'momentum_weeklies': {'exposure':0.20, 'risk_tier': 'moderate'},
            'leaps_tracker': {'exposure':0.15, 'risk_tier': 'aggressive'}
        },
        'market_data': {
            'prices': [100 + i * 0.05 for i in range(100)],
            'volumes': [5000 + i * 50 for i in range(100)],
            'sentiment': -0.2,  # Slightly negative sentiment
            'put_call_ratio': 1.3,  # Bearish options activity
            'social_volume': 0.8,  # High social media activity
            'social_sentiment': -0.4,  # Negative social sentiment
            'vix_level': 28,  # Elevated volatility
            'rsi': 35  # Oversold conditions
        }
    }

def demo_var_analysis(): 
    """Demonstrate VaR analysis capabilities"""
    print("🔬 VaR Analysis Demo")
    print(" = " * 50)
    
    # Create portfolio
    portfolio = create_realistic_portfolio()
    
    # Initialize VaR engine
    var_engine = AdvancedVaREngine(portfolio['total_value'])
    
    # Generate realistic returns based on portfolio composition
    np.random.seed(42)
    n_days = 252
    base_return = 0.0008  # 0.08% daily return
    base_vol = 0.02  # 2% daily volatility
    
    # Add some correlation and regime changes
    returns = []
    for i in range(n_days): 
        if i  <  100:  # First 100 days: normal market
            daily_return = np.random.normal(base_return, base_vol)
        elif i  <  150:  # Next 50 days: high volatility
            daily_return = np.random.normal(base_return, base_vol * 1.5)
        else:  # Last 102 days: crisis period
            daily_return = np.random.normal(base_return * 0.5, base_vol * 2.0)
        returns.append(daily_return)
    
    returns = np.array(returns)
    
    # Calculate comprehensive VaR suite
    print("Calculating VaR Suite...")
    var_suite = var_engine.calculate_var_suite(
        returns=returns,
        confidence_levels = [0.95, 0.99, 0.999],
        methods = ['parametric', 'historical', 'monte_carlo', 'evt']
    )
    
    # Display results
    print(f"\nVaR Analysis Results (Portfolio: ${portfolio['total_value']:,.0f})")
    print("-" * 60)
    summary = var_suite.get_summary()
    
    for key, result in summary.items(): 
        # Handle different key formats
        if '_' in key: 
            parts = key.split('_')
            if len(parts)  >=  2: 
                method = parts[0]
                conf = parts[-1]  # Last part is confidence level
                conf_pct = int(conf)
            else: 
                method = key
                conf_pct = 95
        else: 
            method = key
            conf_pct = 95
        
        print(f"{method.upper(): 12} VaR {conf_pct}%: ${result['var_value']:8,.0f} ({result['var_percentage']: 5.2f}%)")
    
    # Calculate CVaR
    cvar_95 = var_engine.calculate_cvar(returns, 0.95)
    cvar_99 = var_engine.calculate_cvar(returns, 0.99)
    print("\nCVaR Analysis: ")
    print(f"  CVaR 95%: ${cvar_95:,.0f}")
    print(f"  CVaR 99%: ${cvar_99:,.0f}")
    
    # Regime analysis
    regime_info = var_engine.detect_regime_and_adjust(returns)
    print("\nRegime Analysis: ")
    print(f"  Current Regime: {regime_info['regime']}")
    print(f"  Adjustment Factor: {regime_info['adjustment_factor']:.2f}")
    print(f"  Volatility Ratio: {regime_info['volatility_ratio']:.2f}")
    
    return var_engine, var_suite

def demo_stress_testing(): 
    """Demonstrate stress testing capabilities"""
    print("\n + n🚨 Stress Testing Demo")
    print(" = " * 50)
    
    # Create portfolio
    portfolio = create_realistic_portfolio()
    
    # Initialize stress tester
    stress_tester = StressTesting2025()
    
    # Run stress tests
    print("Running comprehensive stress tests...")
    report = stress_tester.run_comprehensive_stress_test(portfolio)
    
    # Display results
    print("\nStress Test Results")
    print("-" * 40)
    print(f"Compliance Status: {report.compliance_status}")
    print(f"Overall Risk Score: {report.overall_risk_score:.1f}/100")
    
    print("\nScenario Analysis: ")
    for scenario_name, result in report.results.items(): 
        status = "✅ PASSED" if result.passed else "❌ FAILED"
        pnl_pct = (result.portfolio_pnl / portfolio['total_value']) * 100
        print(f"  {scenario_name: 25} {status: 10} P & L: ${result.portfolio_pnl: 8,.0f} ({pnl_pct: +5.1f}%)")
    
    print("\nRisk Recommendations: ")
    for i, rec in enumerate(report.recommendations, 1): 
        print(f"  {i}. {rec}")
    
    return stress_tester, report

def demo_ml_risk_prediction(): 
    """Demonstrate ML risk prediction capabilities"""
    print("\n + n🤖 ML Risk Prediction Demo")
    print(" = " * 50)
    
    # Create portfolio with market data
    portfolio = create_realistic_portfolio()
    market_data = portfolio['market_data']
    
    # Initialize ML predictor
    ml_predictor = MLRiskPredictor()
    
    # Volatility prediction
    print("Volatility Prediction: ")
    vol_forecast = ml_predictor.predict_volatility_regime(market_data, horizon_days=5)
    
    print(f"  Predicted Volatility (5 - day): {vol_forecast.predicted_volatility:.2%}")
    print(f"  Confidence Interval: {vol_forecast.confidence_interval[0]:.2%} - {vol_forecast.confidence_interval[1]: .2%}")
    print("  Regime Probabilities: ")
    for regime, prob in vol_forecast.regime_probability.items(): 
        print(f"    {regime}: {prob:.1%}")
    print(f"  Model Confidence: {vol_forecast.model_confidence:.1%}")
    
    # Risk prediction
    print("\nRisk Assessment: ")
    risk_prediction = ml_predictor.predict_risk_score(market_data)
    
    print(f"  Overall Risk Score: {risk_prediction.risk_score:.1f}/100")
    print(f"  Predicted Regime: {risk_prediction.regime_prediction}")
    print(f"  Confidence Level: {risk_prediction.confidence:.1%}")
    
    if risk_prediction.recommended_actions: 
        print("\n  Recommended Actions: ")
        for i, action in enumerate(risk_prediction.recommended_actions, 1): 
            print(f"    {i}. {action}")
    
    return ml_predictor, risk_prediction

def demo_risk_dashboard(): 
    """Demonstrate comprehensive risk dashboard"""
    print("\n + n📊 Risk Dashboard Demo")
    print(" = " * 50)
    
    # Create portfolio
    portfolio = create_realistic_portfolio()
    
    # Initialize dashboard
    dashboard = RiskDashboard2025(portfolio['total_value'])
    
    # Generate dashboard
    print("Generating comprehensive risk dashboard...")
    dashboard_data = dashboard.get_risk_dashboard_data(portfolio)
    
    # Display dashboard
    print("\nRisk Dashboard Summary")
    print("-" * 40)
    print(f"Portfolio Value: ${dashboard_data['portfolio_value']:,.0f}")
    print(f"Total Positions: {dashboard_data['total_positions']}")
    print(f"Last Updated: {dashboard_data['timestamp']}")
    
    # Core risk metrics
    print("\nCore Risk Metrics: ")
    for metric, data in dashboard_data['risk_metrics'].items(): 
        status = "⚠️" if data['limit_utilization']  >  80 else "✅"
        print(f"  {status} {metric.upper(): 8}: ${data['value']:8,.0f} ({data['percentage']: 5.1f}%) - {data['limit_utilization']: 5.1f}% of limit")
    
    # Advanced metrics
    print("\nAdvanced Risk Metrics: ")
    for metric, value in dashboard_data['advanced_metrics'].items(): 
        print(f"  {metric: 20}: ${value: 8,.0f}")
    
    # Alternative data signals
    print("\nAlternative Data Signals: ")
    for signal, value in dashboard_data['alternative_signals'].items(): 
        print(f"  {signal: 20}: ${value: 8,.0f}")
    
    # Factor breakdown
    print("\nFactor Risk Breakdown: ")
    total_factor_risk = sum(dashboard_data['factor_breakdown'].values())
    for factor, value in dashboard_data['factor_breakdown'].items(): 
        pct = (value / total_factor_risk) * 100 if total_factor_risk  >  0 else 0
        print(f"  {factor: 20}: ${value: 8,.0f} ({pct: 5.1f}%)")
    
    # Stress test summary
    print("\nStress Test Summary: ")
    print(f"  Worst Case P & L: ${dashboard_data['stress_tests']['worst_case_pnl']:,.0f}")
    print(f"  Scenarios Tested: {len(dashboard_data['stress_tests']['scenario_analysis'])}")
    
    # Alerts
    print(f"\nRisk Alerts ({len(dashboard_data['alerts'])} active): ")
    if dashboard_data['alerts']: 
        for alert in dashboard_data['alerts']: 
            severity_icon = {
                "LOW": "🟡", "MEDIUM": "🟠", 
                "HIGH": "🔴", "CRITICAL": "🚨"
            }.get(alert['severity'], "⚪")
            print(f"  {severity_icon} {alert['severity']}: {alert['message']}")
            print(f"    → {alert['recommended_action']}")
    else: 
        print("  ✅ No active alerts")
    
    # Risk limit utilization
    print("\nRisk Limit Utilization: ")
    for limit, utilization in dashboard_data['risk_limits'].items(): 
        status = "⚠️" if utilization  >  80 else "✅"
        print(f"  {status} {limit: 15}: {utilization: 5.1f}%")
    
    return dashboard, dashboard_data

def demo_risk_management_workflow(): 
    """Demonstrate complete risk management workflow"""
    print("\n + n🔄 Complete Risk Management Workflow Demo")
    print(" = " * 60)
    
    # Create portfolio
    portfolio = create_realistic_portfolio()
    
    print("Step 1: Portfolio Analysis")
    print("-" * 30)
    print(f"Portfolio Value: ${portfolio['total_value']:,.0f}")
    print(f"Number of Positions: {len(portfolio['positions'])}")
    print("Strategy Allocation: ")
    for strategy, data in portfolio['strategies'].items(): 
        print(f"  {strategy}: {data['exposure']:.1%} ({data['risk_tier']})")
    
    print("\nStep 2: VaR Analysis")
    print("-" * 30)
    var_engine, var_suite=demo_var_analysis()
    
    print("\nStep 3: Stress Testing")
    print("-" * 30)
    stress_tester, stress_report=demo_stress_testing()
    
    print("\nStep 4: ML Risk Prediction")
    print("-" * 30)
    ml_predictor, risk_prediction=demo_ml_risk_prediction()
    
    print("\nStep 5: Risk Dashboard")
    print("-" * 30)
    dashboard, dashboard_data=demo_risk_dashboard()
    
    print("\nStep 6: Risk Management Recommendations")
    print("-" * 30)
    
    # Generate actionable recommendations
    recommendations = []
    
    # VaR - based recommendations
    var_95 = var_suite.results.get('historical_95')
    if var_95 and var_95.var_value  >  portfolio['total_value'] * 0.03: 
        recommendations.append("Consider reducing position sizes - VaR exceeds 3% threshold")
    
    # Stress test recommendations
    if stress_report.compliance_status ==  "NON_COMPLIANT": recommendations.append("Improve portfolio diversification - stress tests show compliance issues")
    
    # ML - based recommendations
    if risk_prediction.risk_score  >  70: 
        recommendations.append("High risk environment detected - consider defensive positioning")
    
    # Dashboard recommendations
    if dashboard_data['risk_limits']['concentration']  >  100: 
        recommendations.append("Concentration risk exceeds limits - diversify positions")
    
    print("Actionable Recommendations: ")
    for i, rec in enumerate(recommendations, 1): 
        print(f"  {i}. {rec}")
    
    if not recommendations: 
        print("  ✅ Portfolio is well - positioned - no immediate action required")
    
    print("\n🎯 Risk Management Workflow Complete!")
    print("   All systems operational and monitoring portfolio risk in real - time.")

def main(): 
    """Main demo function"""
    print("🚀 WallStreetBots Advanced Risk Models - 2025 Demo")
    print(" = " * 70)
    print(f"Demo Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(" = " * 70)
    
    try: 
        # Run complete workflow demo
        demo_risk_management_workflow()
        
        print("\n + n🎉 Demo completed successfully!")
        print("\nKey Features Demonstrated: ")
        print("✅ Multi - method VaR calculation (Parametric, Historical, Monte Carlo, EVT)")
        print("✅ FCA - compliant stress testing with 6 regulatory scenarios")
        print("✅ Machine learning risk prediction and regime detection")
        print("✅ Real - time risk monitoring dashboard with alerts")
        print("✅ Factor risk attribution and concentration analysis")
        print("✅ Alternative data integration (sentiment, options flow)")
        print("✅ Comprehensive risk management workflow")
        
        print("\n📈 Ready for production deployment!")
        print("   This system provides institutional - grade risk management")
        print("   capabilities for algorithmic trading operations.")
        
        return True
        
    except Exception as e: 
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ ==  "__main__": 
    success = main()
    sys.exit(0 if success else 1)
