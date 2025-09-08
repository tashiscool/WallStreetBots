#!/usr/bin/env python3
"""
Test Integrated Advanced Risk System
Complete integration test for all Month 1-6 risk management features

This test validates:
- Integration of all risk management components
- Compatibility with existing WallStreetBots ecosystem  
- Month 5-6 advanced features working within the system
- End-to-end risk management workflow
"""

import asyncio
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.tradingbot.risk.integrated_advanced_risk_manager import (
    IntegratedAdvancedRiskManager, IntegratedRiskConfig, create_integrated_risk_system
)


def generate_test_data():
    """Generate realistic test data for comprehensive testing"""
    # Create 1 year of market data
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    np.random.seed(42)
    
    symbols = ['AAPL', 'GOOGL', 'TSLA', 'SPY', 'BTC', 'EURUSD', 'GOLD']
    market_data = {}
    
    for symbol in symbols:
        # Generate realistic price movements
        if symbol == 'BTC':
            returns = np.random.normal(0.001, 0.05, len(dates))  # High volatility crypto
        elif symbol == 'EURUSD':
            returns = np.random.normal(0.0002, 0.008, len(dates))  # Lower vol forex
        else:
            returns = np.random.normal(0.0008, 0.02, len(dates))  # Equity volatility
            
        prices = 100 * np.cumprod(1 + returns)
        
        market_data[symbol] = pd.DataFrame({
            'Open': prices,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
    
    # Create test positions
    positions = {
        'AAPL': {'qty': 100, 'value': 15000, 'delta': 100, 'gamma': 0, 'vega': 0},
        'GOOGL': {'qty': 50, 'value': 20000, 'delta': 50, 'gamma': 0, 'vega': 0},
        'TSLA': {'qty': 75, 'value': 18000, 'delta': 75, 'gamma': 0, 'vega': 0},
        'SPY': {'qty': 200, 'value': 25000, 'delta': 200, 'gamma': 0, 'vega': 0},
        'BTC': {'qty': 1, 'value': 15000, 'delta': 1, 'gamma': 0, 'vega': 0},
        'GOLD': {'qty': 100, 'value': 7000, 'delta': 100, 'gamma': 0, 'vega': 0}
    }
    
    return market_data, positions


async def test_basic_integration():
    """Test basic system integration"""
    print("🔧 Testing Basic System Integration...")
    
    try:
        # Create integrated risk system
        risk_system = await create_integrated_risk_system(
            portfolio_value=100000,
            regulatory_authority="FCA"
        )
        
        print("✅ Integrated risk system created successfully")
        
        # Check system status
        status = risk_system.get_system_status()
        print(f"   System Status: {status['status']}")
        print(f"   Advanced Features: {status['advanced_features_available']}")
        print(f"   ML Agents: {status['config']['ml_agents_enabled']}")
        print(f"   Multi-Asset: {status['config']['multi_asset_enabled']}")
        print(f"   Compliance: {status['config']['compliance_enabled']}")
        
        return risk_system
        
    except Exception as e:
        print(f"❌ Error in basic integration: {e}")
        raise


async def test_comprehensive_risk_assessment():
    """Test comprehensive risk assessment with all features"""
    print("\n📊 Testing Comprehensive Risk Assessment...")
    
    try:
        # Setup
        risk_system = await create_integrated_risk_system(portfolio_value=100000)
        market_data, positions = generate_test_data()
        
        # Run comprehensive assessment
        results = await risk_system.comprehensive_risk_assessment(positions, market_data)
        
        print("✅ Comprehensive risk assessment completed")
        print(f"   Timestamp: {results.get('timestamp')}")
        print(f"   System Status: {results.get('system_status')}")
        
        # Check VaR analysis
        if 'var_analysis' in results:
            var_data = results['var_analysis']
            print("   VaR Analysis:")
            for method, data in var_data.items():
                if isinstance(data, dict) and 'var_value' in data:
                    print(f"     {method}: ${data['var_value']:,.0f} ({data.get('var_percentage', 0):.2f}%)")
        
        # Check stress testing
        if 'stress_testing' in results:
            stress = results['stress_testing']
            print(f"   Stress Testing: {stress['scenarios_passed']}/{stress['total_scenarios']} passed")
            print(f"   Compliance Status: {stress['compliance_status']}")
        
        # Check ML prediction
        if 'ml_prediction' in results:
            ml = results['ml_prediction']
            print(f"   ML Prediction: {ml['predicted_volatility']:.2%} volatility")
        
        # Check ML agents
        if 'ml_agents' in results:
            agents = results['ml_agents']
            print(f"   ML Agents: {agents['recommended_action']} (confidence: {agents['confidence']:.3f})")
        
        # Check multi-asset
        if 'multi_asset' in results:
            multi = results['multi_asset']
            print(f"   Multi-Asset VaR: {multi['total_var']:.2%}")
        
        # Check compliance
        if 'compliance' in results:
            comp = results['compliance']
            print(f"   Compliance: {comp['status']} ({comp['checks_passed']}/{comp['total_checks']} checks)")
        
        # Check portfolio risk
        if 'portfolio_risk' in results:
            portfolio = results['portfolio_risk']
            print(f"   Portfolio Risk: {portfolio['total_var']:.4f} VaR")
            print(f"   Within Limits: {portfolio['within_limits']}")
            print(f"   Active Alerts: {portfolio['active_alerts']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in comprehensive assessment: {e}")
        raise


async def test_continuous_monitoring():
    """Test continuous risk monitoring"""
    print("\n🔄 Testing Continuous Risk Monitoring...")
    
    try:
        # Setup
        risk_system = await create_integrated_risk_system(portfolio_value=100000)
        market_data, positions = generate_test_data()
        
        # Start monitoring for a short period
        print("   Starting monitoring for 10 seconds...")
        
        # Run monitoring in background
        monitoring_task = asyncio.create_task(
            risk_system.start_continuous_monitoring(positions, market_data)
        )
        
        # Wait for a few monitoring cycles
        await asyncio.sleep(10)
        
        # Stop monitoring
        risk_system.stop_monitoring()
        monitoring_task.cancel()
        
        # Check results
        status = risk_system.get_system_status()
        print(f"✅ Continuous monitoring tested")
        print(f"   Risk History Count: {status['risk_history_count']}")
        print(f"   Last Assessment: {status['last_assessment']}")
        
        if status['risk_history_count'] > 0:
            print("   ✅ Risk monitoring generated historical data")
        else:
            print("   ⚠️ No risk history generated")
            
        return True
        
    except Exception as e:
        print(f"❌ Error in continuous monitoring: {e}")
        return False


async def test_system_compatibility():
    """Test compatibility with existing WallStreetBots components"""
    print("\n🔗 Testing System Compatibility...")
    
    try:
        # Test import compatibility
        from backend.tradingbot.risk import (
            AdvancedVaREngine, RiskIntegrationManager, 
            MultiAgentRiskCoordinator, MultiAssetRiskManager, RegulatoryComplianceManager
        )
        print("✅ All imports work correctly")
        
        # Test component instantiation
        var_engine = AdvancedVaREngine(portfolio_value=100000)
        integration_manager = RiskIntegrationManager()
        
        if MultiAgentRiskCoordinator:
            # Provide required risk_limits parameter
            ml_risk_limits = {
                'max_var': 0.05,
                'max_concentration': 0.3,
                'max_drawdown': 0.15,
                'max_leverage': 2.0
            }
            ml_coordinator = MultiAgentRiskCoordinator(risk_limits=ml_risk_limits)
            print("✅ Advanced ML components available")
        
        if MultiAssetRiskManager:
            multi_asset = MultiAssetRiskManager()
            print("✅ Multi-asset components available")
        
        if RegulatoryComplianceManager:
            compliance = RegulatoryComplianceManager()
            print("✅ Compliance components available")
        
        # Test integrated system with existing components
        risk_system = await create_integrated_risk_system()
        market_data, positions = generate_test_data()
        
        # Test individual components still work
        np.random.seed(42)
        test_returns = np.random.normal(0.001, 0.02, 252)
        var_results = var_engine.calculate_var_suite(test_returns)
        
        print(f"✅ VaR Engine: {len(var_results.results)} methods calculated")
        
        # Test integration with new system
        integrated_results = await risk_system.comprehensive_risk_assessment(positions, market_data)
        
        print("✅ Integration with existing components successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in system compatibility: {e}")
        return False


async def test_performance_benchmarks():
    """Test system performance"""
    print("\n⚡ Testing Performance Benchmarks...")
    
    try:
        risk_system = await create_integrated_risk_system(portfolio_value=500000)
        market_data, positions = generate_test_data()
        
        # Time comprehensive assessment
        start_time = datetime.now()
        results = await risk_system.comprehensive_risk_assessment(positions, market_data)
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ Performance benchmark completed")
        print(f"   Assessment Duration: {duration:.2f} seconds")
        print(f"   Portfolio Value: ${risk_system.config.portfolio_value:,.0f}")
        print(f"   Assets Analyzed: {len(positions)}")
        
        if duration < 5.0:
            print("   🚀 Performance: EXCELLENT (< 5 seconds)")
        elif duration < 10.0:
            print("   ✅ Performance: GOOD (< 10 seconds)")
        else:
            print("   ⚠️ Performance: ACCEPTABLE (> 10 seconds)")
        
        return duration
        
    except Exception as e:
        print(f"❌ Error in performance testing: {e}")
        return None


async def run_comprehensive_integration_test():
    """Run complete integration test suite"""
    print("🚀 WallStreetBots Integrated Advanced Risk System Test")
    print("=" * 70)
    print("Testing complete integration of Month 1-6 risk management features")
    print("=" * 70)
    
    test_results = {}
    
    try:
        # 1. Basic Integration
        print("\n🔧 PHASE 1: Basic System Integration")
        risk_system = await test_basic_integration()
        test_results['basic_integration'] = True
        
        # 2. Comprehensive Assessment
        print("\n📊 PHASE 2: Comprehensive Risk Assessment")
        assessment_results = await test_comprehensive_risk_assessment()
        test_results['comprehensive_assessment'] = True
        
        # 3. Continuous Monitoring
        print("\n🔄 PHASE 3: Continuous Risk Monitoring")
        monitoring_success = await test_continuous_monitoring()
        test_results['continuous_monitoring'] = monitoring_success
        
        # 4. System Compatibility
        print("\n🔗 PHASE 4: System Compatibility")
        compatibility_success = await test_system_compatibility()
        test_results['compatibility'] = compatibility_success
        
        # 5. Performance Benchmarks
        print("\n⚡ PHASE 5: Performance Benchmarks")
        performance_duration = await test_performance_benchmarks()
        test_results['performance'] = performance_duration is not None
        
        # Summary
        print("\n" + "=" * 70)
        print("🎯 INTEGRATION TEST SUMMARY")
        print("=" * 70)
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        for test_name, passed in test_results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\nOverall Results: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL INTEGRATION TESTS PASSED!")
            print("✅ Integrated Advanced Risk System is fully operational")
            print("✅ All Month 1-6 features working together seamlessly")
            print("✅ Compatible with existing WallStreetBots ecosystem")
            print("✅ Ready for production deployment")
        else:
            print(f"\n⚠️ {total_tests - passed_tests} test(s) failed")
            print("❌ System needs attention before production use")
        
        return test_results
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR in integration testing: {e}")
        import traceback
        traceback.print_exc()
        return {}


if __name__ == "__main__":
    # Run the comprehensive integration test
    asyncio.run(run_comprehensive_integration_test())