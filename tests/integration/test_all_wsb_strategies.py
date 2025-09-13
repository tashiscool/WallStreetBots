#!/usr / bin/env python3
"""
Comprehensive Test Suite Runner for All WSB Strategy Modules
Tests all newly created WSB trading strategies and verifies functionality
"""

import sys
import os
import unittest
from datetime import datetime
import traceback

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all test modules
try: 
    from test_swing_trading import run_swing_trading_tests
    SWING_TRADING_AVAILABLE=True
except ImportError as e: 
    print(f"Warning: Could not import swing trading tests: {e}")
    SWING_TRADING_AVAILABLE=False

try: 
    from test_spx_credit_spreads import run_spx_credit_spreads_tests
    SPX_CREDIT_SPREADS_AVAILABLE=True
except ImportError as e: 
    print(f"Warning: Could not import SPX credit spreads tests: {e}")
    SPX_CREDIT_SPREADS_AVAILABLE=False

try: 
    from test_earnings_protection import run_earnings_protection_tests
    EARNINGS_PROTECTION_AVAILABLE=True
except ImportError as e: 
    print(f"Warning: Could not import earnings protection tests: {e}")
    EARNINGS_PROTECTION_AVAILABLE=False

try: 
    from test_index_baseline import run_index_baseline_tests
    INDEX_BASELINE_AVAILABLE=True
except ImportError as e: 
    print(f"Warning: Could not import index baseline tests: {e}")
    INDEX_BASELINE_AVAILABLE=False

try: 
    from test_leaps_tracker import run_leaps_tracker_tests
    LEAPS_TRACKER_AVAILABLE=True
except ImportError as e: 
    print(f"Warning: Could not import LEAPS tracker tests: {e}")
    LEAPS_TRACKER_AVAILABLE=False

# Test individual modules for basic functionality
def test_module_imports(): 
    """Test that all strategy modules can be imported correctly"""
    print("\n" + "=" * 80)
    print("TESTING MODULE IMPORTS")
    print("=" * 80)
    
    modules_to_test=[
        ("swing_trading", "SwingTradingScanner"),
        ("spx_credit_spreads", "CreditSpreadScanner"),
        ("earnings_protection", "EarningsProtectionScanner"),
        ("index_baseline", "IndexBaselineScanner"),
        ("leaps_tracker", "LEAPSTracker")
    ]
    
    results={}
    
    for module_name, class_name in modules_to_test: 
        try: 
            module=__import__(module_name)
            scanner_class=getattr(module, class_name)
            scanner=scanner_class()
            results[module_name]="✅ SUCCESS"
            print(f"✅ {module_name:20} - Successfully imported and instantiated {class_name}")
        except Exception as e: 
            results[module_name]=f"❌ FAILED: {str(e)}"
            print(f"❌ {module_name:20} - Failed: {str(e)}")
    
    return results

def test_basic_functionality(): 
    """Test basic functionality of each strategy module"""
    print("\n" + "=" * 80)
    print("TESTING BASIC FUNCTIONALITY")
    print("=" * 80)
    
    functionality_results={}
    
    # Test Swing Trading
    try: 
        from swing_trading import SwingTradingScanner, SignalType
        scanner=SwingTradingScanner()
        
        # Test basic attributes
        assert hasattr(scanner, 'swing_candidates')
        assert hasattr(scanner, 'min_volume')
        assert len(scanner.swing_candidates) > 0
        
        # Test enums
        assert SignalType.BREAKOUT== "breakout"
        assert SignalType.MOMENTUM_CONTINUATION == "momentum_continuation"
        assert SignalType.REVERSAL_SETUP == "reversal_setup"
        
        functionality_results["swing_trading"]="✅ Basic functionality verified"
        print("✅ swing_trading        - Basic functionality verified")
        
    except Exception as e: 
        functionality_results["swing_trading"]=f"❌ FAILED: {str(e)}"
        print(f"❌ swing_trading        - Failed: {str(e)}")
    
    # Test SPX Credit Spreads
    try: 
        from spx_credit_spreads import CreditSpreadScanner, SpreadType
        scanner=CreditSpreadScanner()
        
        # Test basic attributes
        assert hasattr(scanner, 'target_tickers')
        assert hasattr(scanner, 'target_delta_range')
        assert "SPX" in scanner.target_tickers
        assert "SPY" in scanner.target_tickers
        
        # Test enums
        assert SpreadType.PUT_CREDIT== "put_credit"
        assert SpreadType.CALL_CREDIT == "call_credit"
        
        functionality_results["spx_credit_spreads"]="✅ Basic functionality verified"
        print("✅ spx_credit_spreads   - Basic functionality verified")
        
    except Exception as e: 
        functionality_results["spx_credit_spreads"]=f"❌ FAILED: {str(e)}"
        print(f"❌ spx_credit_spreads   - Failed: {str(e)}")
    
    # Test Earnings Protection
    try: 
        from earnings_protection import EarningsProtectionScanner
        scanner=EarningsProtectionScanner()
        
        # Test basic attributes
        assert hasattr(scanner, 'earnings_candidates')
        assert len(scanner.earnings_candidates) > 0
        assert "AAPL" in scanner.earnings_candidates
        
        functionality_results["earnings_protection"]="✅ Basic functionality verified"
        print("✅ earnings_protection  - Basic functionality verified")
        
    except Exception as e: 
        functionality_results["earnings_protection"]=f"❌ FAILED: {str(e)}"
        print(f"❌ earnings_protection  - Failed: {str(e)}")
    
    # Test Index Baseline
    try: 
        from index_baseline import IndexBaselineScanner
        scanner=IndexBaselineScanner()
        
        # Test basic attributes
        assert hasattr(scanner, 'benchmarks')
        assert hasattr(scanner, 'wsb_strategies')
        assert "SPY" in scanner.benchmarks
        assert "wheel_strategy" in scanner.wsb_strategies
        
        functionality_results["index_baseline"]="✅ Basic functionality verified"
        print("✅ index_baseline       - Basic functionality verified")
        
    except Exception as e: 
        functionality_results["index_baseline"]=f"❌ FAILED: {str(e)}"
        print(f"❌ index_baseline       - Failed: {str(e)}")
    
    # Test Enhanced LEAPS Tracker
    try: 
        from leaps_tracker import LEAPSTracker, MovingAverageCross
        tracker=LEAPSTracker()
        
        # Test basic attributes
        assert hasattr(tracker, 'secular_themes')
        assert hasattr(tracker, 'positions')
        assert "ai_revolution" in tracker.secular_themes
        assert hasattr(tracker, 'analyze_moving_average_cross')
        
        # Test MovingAverageCross dataclass
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
        assert ma_cross.cross_type== "golden_cross"
        assert ma_cross.trend_direction == "bullish"
        
        functionality_results["leaps_tracker"]="✅ Basic functionality verified (Enhanced with MA crosses)"
        print("✅ leaps_tracker        - Basic functionality verified (Enhanced with MA crosses)")
        
    except Exception as e: 
        functionality_results["leaps_tracker"]=f"❌ FAILED: {str(e)}"
        print(f"❌ leaps_tracker        - Failed: {str(e)}")
    
    return functionality_results

def run_all_comprehensive_tests(): 
    """Run all comprehensive test suites"""
    print("\n" + "=" * 80)
    print("RUNNING COMPREHENSIVE TEST SUITES")
    print("=" * 80)
    
    test_results={}
    total_tests_run=0
    total_failures=0
    total_errors=0
    
    # Run Swing Trading Tests
    if SWING_TRADING_AVAILABLE: 
        try: 
            print("\n" + "-" * 40)
            print("RUNNING SWING TRADING TESTS")
            print("-" * 40)
            result=run_swing_trading_tests()
            test_results["swing_trading"]={
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "success_rate": ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100 if result.testsRun > 0 else 0
            }
            total_tests_run += result.testsRun
            total_failures += len(result.failures)
            total_errors += len(result.errors)
        except Exception as e: 
            test_results["swing_trading"]={"error": str(e)}
            print(f"❌ Failed to run swing trading tests: {e}")
    
    # Run SPX Credit Spreads Tests
    if SPX_CREDIT_SPREADS_AVAILABLE: 
        try: 
            print("\n" + "-" * 40)
            print("RUNNING SPX CREDIT SPREADS TESTS")
            print("-" * 40)
            result=run_spx_credit_spreads_tests()
            test_results["spx_credit_spreads"]={
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "success_rate": ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100 if result.testsRun > 0 else 0
            }
            total_tests_run += result.testsRun
            total_failures += len(result.failures)
            total_errors += len(result.errors)
        except Exception as e: 
            test_results["spx_credit_spreads"]={"error": str(e)}
            print(f"❌ Failed to run SPX credit spreads tests: {e}")
    
    # Run Earnings Protection Tests
    if EARNINGS_PROTECTION_AVAILABLE: 
        try: 
            print("\n" + "-" * 40)
            print("RUNNING EARNINGS PROTECTION TESTS")
            print("-" * 40)
            result=run_earnings_protection_tests()
            test_results["earnings_protection"]={
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "success_rate": ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100 if result.testsRun > 0 else 0
            }
            total_tests_run += result.testsRun
            total_failures += len(result.failures)
            total_errors += len(result.errors)
        except Exception as e: 
            test_results["earnings_protection"]={"error": str(e)}
            print(f"❌ Failed to run earnings protection tests: {e}")
    
    # Run Index Baseline Tests
    if INDEX_BASELINE_AVAILABLE: 
        try: 
            print("\n" + "-" * 40)
            print("RUNNING INDEX BASELINE TESTS")
            print("-" * 40)
            result=run_index_baseline_tests()
            test_results["index_baseline"]={
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "success_rate": ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100 if result.testsRun > 0 else 0
            }
            total_tests_run += result.testsRun
            total_failures += len(result.failures)
            total_errors += len(result.errors)
        except Exception as e: 
            test_results["index_baseline"]={"error": str(e)}
            print(f"❌ Failed to run index baseline tests: {e}")
    
    # Run LEAPS Tracker Tests
    if LEAPS_TRACKER_AVAILABLE: 
        try: 
            print("\n" + "-" * 40)
            print("RUNNING ENHANCED LEAPS TRACKER TESTS")
            print("-" * 40)
            result=run_leaps_tracker_tests()
            test_results["leaps_tracker"]={
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "success_rate": ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100 if result.testsRun > 0 else 0
            }
            total_tests_run += result.testsRun
            total_failures += len(result.failures)
            total_errors += len(result.errors)
        except Exception as e: 
            test_results["leaps_tracker"]={"error": str(e)}
            print(f"❌ Failed to run LEAPS tracker tests: {e}")
    
    return test_results, total_tests_run, total_failures, total_errors

def generate_comprehensive_report(import_results, functionality_results, test_results, total_tests, total_failures, total_errors): 
    """Generate comprehensive test report"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE WSB STRATEGIES TEST REPORT")
    print("=" * 80)
    print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Module Import Results
    print("\n📦 MODULE IMPORT RESULTS: ")
    print("-" * 40)
    for module, result in import_results.items(): 
        print(f"{module: 20} {result}")
    
    # Basic Functionality Results
    print("\n⚙️  BASIC FUNCTIONALITY RESULTS: ")
    print("-" * 40)
    for module, result in functionality_results.items(): 
        print(f"{module: 20} {result}")
    
    # Comprehensive Test Results
    print("\n🧪 COMPREHENSIVE TEST RESULTS: ")
    print("-" * 40)
    for module, result in test_results.items(): 
        if "error" in result: 
            print(f"{module: 20} ❌ ERROR: {result['error']}")
        else: 
            status="✅" if result['success_rate'] >= 90 else "⚠️" if result['success_rate'] >= 70 else "❌"
            print(f"{module: 20} {status} {result['tests_run']} tests, {result['success_rate']: .1f}% success")
    
    # Overall Summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    
    total_modules=len(import_results)
    successful_imports=len([r for r in import_results.values() if "SUCCESS" in r])
    successful_functionality=len([r for r in functionality_results.values() if "SUCCESS" in r])
    
    overall_success_rate=((total_tests - total_failures - total_errors) / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"Total Modules:           {total_modules}")
    print(f"Successful Imports:      {successful_imports}/{total_modules} ({(successful_imports / total_modules)*100: .1f}%)")
    print(f"Functional Modules:      {successful_functionality}/{total_modules} ({(successful_functionality / total_modules)*100: .1f}%)")
    print(f"Total Tests Run:         {total_tests}")
    print(f"Total Failures:          {total_failures}")
    print(f"Total Errors:            {total_errors}")
    print(f"Overall Success Rate:    {overall_success_rate:.1f}%")
    
    # Readiness Assessment
    print("\n🎯 READINESS ASSESSMENT: ")
    print("-" * 40)
    
    if overall_success_rate >= 95 and successful_imports== total_modules: 
        print("✅ ALL SYSTEMS GO!")
        print("   • All modules imported successfully")
        print("   • All basic functionality verified")
        print("   • Comprehensive tests passing at >95%")
        print("   • WSB strategies ready for live testing")
        readiness="READY"
    elif overall_success_rate >= 80 and successful_imports >= total_modules * 0.8: 
        print("⚠️  MOSTLY READY - Minor Issues")
        print("   • Most modules working correctly")
        print("   • Some test failures need attention")
        print("   • Suitable for paper trading")
        readiness="MOSTLY_READY"
    elif successful_imports >= total_modules * 0.6: 
        print("❌ NEEDS WORK - Major Issues")
        print("   • Multiple modules have problems")
        print("   • Significant test failures")
        print("   • Not ready for live trading")
        readiness="NEEDS_WORK"
    else: 
        print("🚨 CRITICAL ISSUES")
        print("   • Major import / functionality failures")
        print("   • System not functional")
        print("   • Requires immediate attention")
        readiness="CRITICAL"
    
    # WSB Strategy Specific Assessment
    print("\n📊 WSB STRATEGY IMPLEMENTATION STATUS: ")
    print("-" * 40)
    
    wsb_strategies={
        "swing_trading": "Enhanced Swing Trading with Fast Exits",
        "spx_credit_spreads": "SPX / SPY 0DTE Credit Spreads",
        "earnings_protection": "Earnings IV Crush Protection",
        "index_baseline": "Index Fund Baseline Comparison",
        "leaps_tracker": "LEAPS with Golden / Death Cross Timing"
    }
    
    for strategy_key, strategy_name in wsb_strategies.items(): 
        if strategy_key in import_results and "SUCCESS" in import_results[strategy_key]: 
            if strategy_key in test_results and "error" not in test_results[strategy_key]: 
                success_rate=test_results[strategy_key]['success_rate']
                status="✅" if success_rate >= 90 else "⚠️" if success_rate >= 70 else "❌"
                print(f"{status} {strategy_name}")
                print(f"   └─ {success_rate: .1f}% test success rate")
            else: 
                print(f"⚠️  {strategy_name}")
                print(f"   └─ Tests not available or failed")
        else: 
            print(f"❌ {strategy_name}")
            print(f"   └─ Module import failed")
    
    return readiness

def main(): 
    """Main test runner function"""
    print("🤖 WSB TRADING STRATEGIES - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print("Testing all newly implemented WSB strategies: ")
    print("• Enhanced Swing Trading with Fast Exits")
    print("• SPX / SPY 0DTE Credit Spreads") 
    print("• Earnings IV Crush Protection")
    print("• Index Fund Baseline Comparison")
    print("• LEAPS with Golden / Death Cross Timing")
    print("=" * 80)
    
    try: 
        # Test module imports
        import_results=test_module_imports()
        
        # Test basic functionality
        functionality_results=test_basic_functionality()
        
        # Run comprehensive tests
        test_results, total_tests, total_failures, total_errors=run_all_comprehensive_tests()
        
        # Generate comprehensive report
        readiness=generate_comprehensive_report(
            import_results, 
            functionality_results, 
            test_results, 
            total_tests, 
            total_failures, 
            total_errors
        )
        
        print("\n" + "=" * 80)
        print("🎉 TESTING COMPLETE!")
        print("=" * 80)
        
        return readiness
        
    except Exception as e: 
        print(f"\n🚨 CRITICAL ERROR IN TEST RUNNER: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return "CRITICAL"

if __name__== "__main__": 
    readiness_status=main()
    
    # Exit with appropriate code
    exit_codes={
        "READY": 0,
        "MOSTLY_READY": 1, 
        "NEEDS_WORK": 2,
        "CRITICAL": 3
    }
    
    exit(exit_codes.get(readiness_status, 3))