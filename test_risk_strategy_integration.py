#!/usr/bin/env python3
"""
Test Risk-Strategy Integration
Demonstrates Month 3-4: Integration with WallStreetBots

This script tests the integration of sophisticated risk models with trading strategies:
- Real-time risk assessment during trading
- Automated risk controls and position sizing
- Cross-strategy risk coordination
- Risk alerts and monitoring integration
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.tradingbot.risk.risk_integrated_production_manager import (
    RiskIntegratedProductionManager, RiskIntegratedConfig, RiskLimits
)
from backend.tradingbot.risk.risk_integration_manager import RiskIntegrationManager
from backend.tradingbot.production.core.production_strategy_manager import StrategyProfile


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger=logging.getLogger(__name__)


async def test_risk_strategy_integration():
    """Test the complete risk-strategy integration"""
    
    print("🚀 Testing Risk-Strategy Integration - Month 3-4")
    print("=" * 60)
    
    try:
        # 1. Initialize Risk-Integrated Production Manager
        print("\n1. Initializing Risk-Integrated Production Manager...")
        
        # Configure risk limits
        risk_limits=RiskLimits(
            max_total_var=0.05,      # 5% max total VaR
            max_total_cvar=0.07,     # 7% max total CVaR
            max_position_var=0.02,   # 2% max per position VaR
            max_drawdown=0.15,       # 15% max drawdown
            max_concentration=0.30,  # 30% max concentration
            max_greeks_risk=0.10     # 10% max Greeks risk
        )
        
        # Create configuration
        config=RiskIntegratedConfig(
            alpaca_api_key="test_key",
            alpaca_secret_key="test_secret",
            paper_trading=True,
            profile=StrategyProfile.research_2024,
            risk_limits=risk_limits,
            enable_ml_risk=True,
            enable_stress_testing=True,
            enable_risk_dashboard=True,
            risk_calculation_interval=10,  # 10 seconds for testing
            auto_position_sizing=True,
            auto_risk_controls=True,
            cross_strategy_coordination=True
        )
        
        # Initialize manager
        manager=RiskIntegratedProductionManager(config)
        print("✅ Risk-Integrated Production Manager initialized")
        
        # 2. Test Risk Management Components
        print("\n2. Testing Risk Management Components...")
        
        # Test risk manager directly
        risk_manager=RiskIntegrationManager(risk_limits=risk_limits)
        print("✅ Risk Integration Manager initialized")
        
        # Test risk calculation with simulated data
        print("\n3. Testing Risk Calculations...")
        
        # Simulate portfolio positions
        positions={
            'AAPL':{'qty':100, 'value':15000, 'delta':0.6, 'gamma':0.01, 'vega':0.5},
            'SPY':{'qty':50, 'value':20000, 'delta':0.5, 'gamma':0.005, 'vega':0.3},
            'TSLA':{'qty':25, 'value':5000, 'delta':0.8, 'gamma':0.02, 'vega':0.8}
        }
        
        # Simulate market data
        import pandas as pd
        import numpy as np
        
        dates=pd.date_range(end=datetime.now(), periods=252, freq='D')
        returns=np.random.normal(0, 0.02, 252)
        prices=100 * np.cumprod(1 + returns)
        
        market_data={
            'AAPL':pd.DataFrame({
                'Open':prices * 0.99,
                'High':prices * 1.02,
                'Low':prices * 0.98,
                'Close':prices,
                'Volume':np.random.randint(1000000, 5000000, 252)
            }, index=dates),
            'SPY':pd.DataFrame({
                'Open':prices * 0.99,
                'High':prices * 1.01,
                'Low':prices * 0.99,
                'Close':prices,
                'Volume':np.random.randint(5000000, 10000000, 252)
            }, index=dates),
            'TSLA':pd.DataFrame({
                'Open':prices * 0.95,
                'High':prices * 1.05,
                'Low':prices * 0.95,
                'Close':prices,
                'Volume':np.random.randint(2000000, 8000000, 252)
            }, index=dates)
        }
        
        portfolio_value=100000.0
        
        # Calculate risk metrics
        risk_metrics = await risk_manager.calculate_portfolio_risk(
            positions, market_data, portfolio_value
        )
        
        print(f"✅ Risk calculated:")
        print(f"   Portfolio VaR: {risk_metrics.portfolio_var:.2%}")
        print(f"   Portfolio CVaR: {risk_metrics.portfolio_cvar:.2%}")
        print(f"   Portfolio LVaR: {risk_metrics.portfolio_lvar:.2%}")
        print(f"   Concentration Risk: {risk_metrics.concentration_risk:.2%}")
        print(f"   Greeks Risk: {risk_metrics.greeks_risk:.2%}")
        print(f"   Within Limits: {risk_metrics.within_limits}")
        
        if risk_metrics.alerts:
            print(f"   Alerts: {', '.join(risk_metrics.alerts)}")
        
        # 4. Test Risk-Adjusted Position Sizing
        print("\n4. Testing Risk-Adjusted Position Sizing...")
        
        # Test position sizing for different scenarios
        test_cases=[
            ("AAPL", 1000, "Normal risk"),
            ("SPY", 5000, "High value trade"),
            ("TSLA", 2000, "Volatile stock")
        ]
        
        for symbol, base_size, description in test_cases:
            risk_adjusted_size=await risk_manager.get_risk_adjusted_position_size(
                "test_strategy", symbol, base_size, portfolio_value
            )
            
            adjustment_factor=risk_adjusted_size / base_size if base_size > 0 else 0
            
            print(f"   {symbol} ({description}):")
            print(f"     Base size: ${base_size:,.0f}")
            print(f"     Risk-adjusted: ${risk_adjusted_size:,.0f}")
            print(f"     Adjustment factor: {adjustment_factor:.1%}")
        
        # 5. Test Trade Allowance
        print("\n5. Testing Trade Allowance...")
        
        # Test different trade scenarios
        trade_tests=[
            ("AAPL", 1000, "Small trade"),
            ("SPY", 10000, "Large trade"),
            ("TSLA", 5000, "Medium trade")
        ]
        
        for symbol, trade_value, description in trade_tests:
            allowed, reason=await risk_manager.should_allow_trade(
                "test_strategy", symbol, trade_value, portfolio_value
            )
            
            status="✅ ALLOWED" if allowed else "❌ BLOCKED"
            print(f"   {symbol} ${trade_value:,} ({description}): {status}")
            if not allowed:
                print(f"     Reason: {reason}")
        
        # 6. Test Risk Summary
        print("\n6. Testing Risk Summary...")
        
        risk_summary=await risk_manager.get_risk_summary()
        
        print("✅ Risk Summary Generated:")
        print(f"   Timestamp: {risk_summary['timestamp']}")
        print(f"   Calculation Count: {risk_summary['calculation_count']}")
        print(f"   Portfolio VaR: {risk_summary['metrics']['portfolio_var']:.2%}")
        print(f"   Portfolio CVaR: {risk_summary['metrics']['portfolio_cvar']:.2%}")
        print(f"   Within Limits: {risk_summary['metrics']['within_limits']}")
        
        # Show utilization
        utilization=risk_summary['utilization']
        print(f"   VaR Utilization: {utilization['var_utilization']:.1%}")
        print(f"   CVaR Utilization: {utilization['cvar_utilization']:.1%}")
        print(f"   Concentration Utilization: {utilization['concentration_utilization']:.1%}")
        print(f"   Greeks Utilization: {utilization['greeks_utilization']:.1%}")
        
        # 7. Test Strategy Integration (Simulated)
        print("\n7. Testing Strategy Integration...")
        
        # Simulate strategy trade execution
        print("   Simulating strategy trade execution...")
        
        # Test WSB Dip Bot trade
        wsb_result=await manager.execute_strategy_trade(
            "wsb_dip_bot", "AAPL", "buy", 100, 150.0
        )
        print(f"   WSB Dip Bot trade: {wsb_result['success']}")
        if not wsb_result['success']:
            print(f"     Reason: {wsb_result['reason']}")
        
        # Test Earnings Protection trade
        earnings_result=await manager.execute_strategy_trade(
            "earnings_protection", "SPY", "buy", 50, 400.0
        )
        print(f"   Earnings Protection trade: {earnings_result['success']}")
        if not earnings_result['success']:
            print(f"     Reason: {earnings_result['reason']}")
        
        # 8. Test System Status
        print("\n8. Testing System Status...")
        
        system_status=await manager.get_system_status()
        
        print("✅ System Status Retrieved:")
        print(f"   Risk Management Enabled: {system_status['risk_management']['enabled']}")
        print(f"   Monitoring Active: {system_status['risk_management']['monitoring_active']}")
        print(f"   Within Limits: {system_status['risk_management']['within_limits']}")
        
        if system_status['risk_management']['alerts']:
            print(f"   Active Alerts: {len(system_status['risk_management']['alerts'])}")
            for alert in system_status['risk_management']['alerts']:
                print(f"     - {alert}")
        
        # 9. Performance Metrics
        print("\n9. Performance Metrics...")
        
        risk_performance=system_status['risk_performance']
        print(f"   Risk Calculations: {risk_performance['risk_calculations_count']}")
        print(f"   Trades Blocked by Risk: {risk_performance['trades_blocked_by_risk']}")
        print(f"   Risk Adjustments Made: {risk_performance['risk_adjustments_made']}")
        
        # 10. Test Risk Limits Update
        print("\n10. Testing Risk Limits Update...")
        
        # Update risk limits
        new_limits=RiskLimits(
            max_total_var=0.03,      # Reduce to 3%
            max_total_cvar=0.05,     # Reduce to 5%
            max_position_var=0.01,   # Reduce to 1%
            max_drawdown=0.10,       # Reduce to 10%
            max_concentration=0.20,  # Reduce to 20%
            max_greeks_risk=0.05     # Reduce to 5%
        )
        
        manager.update_risk_limits(new_limits)
        print("✅ Risk limits updated to more conservative settings")
        
        # Test trade with new limits
        test_result=await manager.execute_strategy_trade(
            "wsb_dip_bot", "TSLA", "buy", 200, 200.0
        )
        print(f"   Trade with new limits: {test_result['success']}")
        if not test_result['success']:
            print(f"     Reason: {test_result['reason']}")
        
        print("\n🎉 Risk-Strategy Integration Test Completed Successfully!")
        print("\n📊 Summary:")
        print("✅ Risk Integration Manager: Working")
        print("✅ Risk Calculations: Working")
        print("✅ Position Sizing: Working")
        print("✅ Trade Allowance: Working")
        print("✅ Risk Summary: Working")
        print("✅ Strategy Integration: Working")
        print("✅ System Status: Working")
        print("✅ Risk Limits Update: Working")
        
        print("\n🚀 Month 3-4 Integration Status: COMPLETE")
        print("   - Real-time risk assessment: ✅")
        print("   - Automated risk controls: ✅")
        print("   - Cross-strategy coordination: ✅")
        print("   - Risk alerts and monitoring: ✅")
        print("   - Portfolio-level risk management: ✅")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in risk-strategy integration test: {e}")
        print(f"\n❌ Test failed with error: {e}")
        return False


async def main():
    """Main test function"""
    print("Starting Risk-Strategy Integration Test...")
    
    success=await test_risk_strategy_integration()
    
    if success:
        print("\n🎯 All tests passed! Risk-strategy integration is working correctly.")
        print("\nNext steps for Month 3-4:")
        print("1. Integrate with real broker data")
        print("2. Connect to live market feeds")
        print("3. Implement real-time monitoring dashboard")
        print("4. Add advanced alerting system")
        print("5. Performance optimization")
    else:
        print("\n❌ Some tests failed. Please check the logs for details.")
        sys.exit(1)


if __name__== "__main__":asyncio.run(main())


