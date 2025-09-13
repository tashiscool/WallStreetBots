#!/usr / bin/env python3
"""
Quick test runner to verify all WSB strategy modules work correctly
Tests basic functionality without complex unit tests
"""

import sys
import os
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_module_imports(): 
    """Test that all strategy modules can be imported and instantiated"""
    print("🤖 WSB TRADING STRATEGIES - QUICK FUNCTIONALITY TEST")
    print("=" * 80)
    print(f"Testing at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    results={}
    
    # Test Swing Trading
    print("\n1. Testing Swing Trading Module...")
    try: 
        from swing_trading import SwingTradingScanner, SwingSignal
        scanner=SwingTradingScanner()
        print(f"   ✅ SwingTradingScanner instantiated")
        print(f"   ✅ Found {len(scanner.swing_tickers)} swing tickers")
        print(f"   ✅ Active trades: {len(scanner.active_trades)} currently")
        results['swing_trading']='SUCCESS'
    except Exception as e: 
        print(f"   ❌ Failed: {e}")
        results['swing_trading']=f'FAILED: {e}'
    
    # Test SPX Credit Spreads
    print("\n2. Testing SPX Credit Spreads Module...")
    try: 
        from spx_credit_spreads import SPXCreditSpreadsScanner, CreditSpreadOpportunity
        scanner=SPXCreditSpreadsScanner()
        print(f"   ✅ SPXCreditSpreadsScanner instantiated")
        print(f"   ✅ Target tickers: {scanner.credit_tickers}")
        print(f"   ✅ Target delta: {scanner.target_short_delta}")
        results['spx_credit_spreads']='SUCCESS'
    except Exception as e: 
        print(f"   ❌ Failed: {e}")
        results['spx_credit_spreads']=f'FAILED: {e}'
    
    # Test Earnings Protection
    print("\n3. Testing Earnings Protection Module...")
    try: 
        from earnings_protection import EarningsProtectionScanner, EarningsEvent
        scanner=EarningsProtectionScanner()
        print(f"   ✅ EarningsProtectionScanner instantiated")
        print(f"   ✅ Found {len(scanner.earnings_candidates)} earnings candidates")
        print(f"   ✅ Sample candidates: {scanner.earnings_candidates[:5]}")
        results['earnings_protection']='SUCCESS'
    except Exception as e: 
        print(f"   ❌ Failed: {e}")
        results['earnings_protection']=f'FAILED: {e}'
    
    # Test Index Baseline
    print("\n4. Testing Index Baseline Module...")
    try: 
        from index_baseline import IndexBaselineScanner, PerformanceComparison
        scanner=IndexBaselineScanner()
        print(f"   ✅ IndexBaselineScanner instantiated")
        print(f"   ✅ Benchmarks: {scanner.benchmarks}")
        print(f"   ✅ WSB strategies: {list(scanner.wsb_strategies.keys())}")
        results['index_baseline']='SUCCESS'
    except Exception as e: 
        print(f"   ❌ Failed: {e}")
        results['index_baseline']=f'FAILED: {e}'
    
    # Test Enhanced LEAPS Tracker
    print("\n5. Testing Enhanced LEAPS Tracker Module...")
    try: 
        from leaps_tracker import LEAPSTracker, MovingAverageCross
        tracker=LEAPSTracker()
        print(f"   ✅ LEAPSTracker instantiated")
        print(f"   ✅ Secular themes: {list(tracker.secular_themes.keys())}")
        
        # Test golden / death cross analysis
        ma_cross=MovingAverageCross(
            cross_type="golden_cross",
            cross_date=None,
            days_since_cross=None,
            sma_50=100.0,
            sma_200=95.0,
            price_above_50sma=True,
            price_above_200sma=True,
            cross_strength=50.0,
            trend_direction="bullish"
        )
        print(f"   ✅ MovingAverageCross dataclass works")
        print(f"   ✅ Cross type: {ma_cross.cross_type}")
        results['leaps_tracker']='SUCCESS (Enhanced with MA crosses)'
    except Exception as e: 
        print(f"   ❌ Failed: {e}")
        results['leaps_tracker']=f'FAILED: {e}'
    
    return results

def test_basic_functionality(): 
    """Test basic functionality of each module"""
    print("\n" + "=" * 80)
    print("TESTING BASIC FUNCTIONALITY")
    print("=" * 80)
    
    # Test some basic method calls to ensure modules work
    functionality_results={}
    
    # Test Swing Trading functionality
    try: 
        from swing_trading import SwingTradingScanner
        scanner=SwingTradingScanner()
        
        # Test that key methods and attributes exist
        assert hasattr(scanner, 'swing_tickers')
        assert hasattr(scanner, 'active_trades')
        assert hasattr(scanner, 'detect_breakout')
        
        print("✅ Swing Trading: All key methods available")
        functionality_results['swing_trading']='FUNCTIONAL'
    except Exception as e: 
        print(f"❌ Swing Trading: {e}")
        functionality_results['swing_trading']=f'ERROR: {e}'
    
    # Test SPX Credit Spreads functionality
    try: 
        from spx_credit_spreads import SPXCreditSpreadsScanner
        scanner=SPXCreditSpreadsScanner()
        
        # Test Black - Scholes methods
        assert hasattr(scanner, 'black_scholes_put')
        assert hasattr(scanner, 'black_scholes_call')
        
        # Test basic BS calculation
        price, delta=scanner.black_scholes_put(100, 95, 0.25, 0.05, 0.2)
        assert price > 0
        assert -1 < delta < 0  # Put delta should be negative
        
        print("✅ SPX Credit Spreads: Basic Black - Scholes calculations work")
        functionality_results['spx_credit_spreads']='FUNCTIONAL'
    except Exception as e: 
        print(f"❌ SPX Credit Spreads: {e}")
        functionality_results['spx_credit_spreads']=f'ERROR: {e}'
    
    # Test Earnings Protection functionality
    try: 
        from earnings_protection import EarningsProtectionScanner
        scanner=EarningsProtectionScanner()
        
        assert hasattr(scanner, 'get_upcoming_earnings')
        assert hasattr(scanner, 'estimate_earnings_move')
        assert hasattr(scanner, 'create_deep_itm_strategy')
        
        print("✅ Earnings Protection: All key methods available")
        functionality_results['earnings_protection']='FUNCTIONAL'
    except Exception as e: 
        print(f"❌ Earnings Protection: {e}")
        functionality_results['earnings_protection']=f'ERROR: {e}'
    
    # Test Index Baseline functionality
    try: 
        from index_baseline import IndexBaselineScanner
        scanner=IndexBaselineScanner()
        
        assert hasattr(scanner, 'get_baseline_performance')
        assert hasattr(scanner, 'compare_strategy_performance')
        assert hasattr(scanner, 'calculate_trading_costs')
        
        # Test cost calculation
        costs=scanner.calculate_trading_costs(50, 10000)
        assert costs >= 0
        assert costs <= 0.05  # Max 5% as per cap
        
        print("✅ Index Baseline: Basic cost calculations work")
        functionality_results['index_baseline']='FUNCTIONAL'
    except Exception as e: 
        print(f"❌ Index Baseline: {e}")
        functionality_results['index_baseline']=f'ERROR: {e}'
    
    # Test Enhanced LEAPS functionality
    try: 
        from leaps_tracker import LEAPSTracker
        tracker=LEAPSTracker()
        
        assert hasattr(tracker, 'analyze_moving_average_cross')
        assert hasattr(tracker, 'calculate_entry_exit_timing_scores')
        assert hasattr(tracker, 'scan_secular_winners')
        
        # Test timing calculation
        from leaps_tracker import MovingAverageCross
        ma_cross=MovingAverageCross(
            cross_type="golden_cross", cross_date=None, days_since_cross=20,
            sma_50=110.0, sma_200=105.0, price_above_50sma=True, 
            price_above_200sma=True, cross_strength=60.0, trend_direction="bullish"
        )
        
        entry_score, exit_score=tracker.calculate_entry_exit_timing_scores(ma_cross, 112.0)
        assert 0 <= entry_score <= 100
        assert 0 <= exit_score <= 100
        
        print("✅ Enhanced LEAPS: Golden / death cross timing calculations work")
        functionality_results['leaps_tracker']='FUNCTIONAL (Enhanced)'
    except Exception as e: 
        print(f"❌ Enhanced LEAPS: {e}")
        functionality_results['leaps_tracker']=f'ERROR: {e}'
    
    return functionality_results

def generate_summary(import_results, functionality_results): 
    """Generate final summary"""
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    total_modules=len(import_results)
    successful_imports=len([r for r in import_results.values() if 'SUCCESS' in r])
    functional_modules=len([r for r in functionality_results.values() if 'FUNCTIONAL' in r])
    
    print(f"📊 RESULTS: ")
    print(f"   Total WSB Strategy Modules: {total_modules}")
    print(f"   Successful Imports:         {successful_imports}/{total_modules} ({(successful_imports / total_modules)*100: .0f}%)")
    print(f"   Functional Modules:         {functional_modules}/{total_modules} ({(functional_modules / total_modules)*100: .0f}%)")
    
    print(f"\n📋 MODULE STATUS: ")
    all_modules=set(list(import_results.keys()) + list(functionality_results.keys()))
    for module in sorted(all_modules): 
        import_status="✅" if 'SUCCESS' in import_results.get(module, '') else "❌"
        func_status="✅" if 'FUNCTIONAL' in functionality_results.get(module, '') else "❌" 
        enhanced="(Enhanced)" if "Enhanced" in import_results.get(module, '') else ""
        print(f"   {module: 20} {import_status} Import  {func_status} Function  {enhanced}")
    
    # Overall assessment
    if successful_imports== total_modules and functional_modules == total_modules: 
        status="🎉 ALL SYSTEMS GO!"
        description="All WSB strategy modules are working correctly"
        readiness="READY"
    elif successful_imports >= total_modules * 0.8 and functional_modules >= total_modules * 0.8: 
        status="✅ MOSTLY WORKING"
        description="Most modules working, minor issues"
        readiness="MOSTLY_READY"
    elif successful_imports >= total_modules * 0.6: 
        status="⚠️ NEEDS ATTENTION"
        description="Some modules have issues"
        readiness="NEEDS_WORK"
    else: 
        status="❌ MAJOR ISSUES"
        description="Multiple modules failing"
        readiness="CRITICAL"
    
    print(f"\n🎯 OVERALL STATUS: {status}")
    print(f"   {description}")
    
    print(f"\n💡 WSB STRATEGIES IMPLEMENTED: ")
    print(f"   • Enhanced Swing Trading with Fast Exits")
    print(f"   • SPX / SPY 0DTE Credit Spreads (WSB favorite)")
    print(f"   • Earnings IV Crush Protection")
    print(f"   • Index Fund Baseline Comparison ('WSB Reality Check')")
    print(f"   • LEAPS with Golden / Death Cross Timing (Enhanced)")
    
    if readiness== "READY": 
        print(f"\n🚀 READY FOR TESTING!")
        print(f"   All modules can be used for paper trading and development")
    elif readiness== "MOSTLY_READY": 
        print(f"\n🔧 MOSTLY READY")
        print(f"   Most functionality works, some edge cases may need attention")
    
    return readiness

def main(): 
    """Main test function"""
    try: 
        # Test imports
        import_results=test_module_imports()
        
        # Test functionality  
        functionality_results=test_basic_functionality()
        
        # Generate summary
        readiness=generate_summary(import_results, functionality_results)
        
        print(f"\n{'='*80}")
        print(f"🏁 TESTING COMPLETE - Status: {readiness}")
        print(f"{'='*80}")
        
        return readiness== "READY" or readiness == "MOSTLY_READY"
        
    except Exception as e: 
        print(f"\n🚨 CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__== "__main__": 
    success=main()
    exit(0 if success else 1)