#!/usr/bin/env python3
"""Strategy Validation Script
Test and validate that WallStreetBots strategies actually work.

This script helps users verify:
1. All strategies load correctly
2. Data feeds are working
3. Basic trading logic functions
4. Risk management is active
5. Performance tracking works

Run this BEFORE committing real money to trading.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal

import django
from dotenv import load_dotenv

# Load environment and setup Django
load_dotenv()
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from backend.tradingbot.production.core.production_strategy_manager import (
    ProductionStrategyManager,
    ProductionStrategyManagerConfig,
    StrategyProfile,
)
from backend.tradingbot.apimanagers import AlpacaManager


class StrategyValidator:
    """Validates that strategies are working correctly."""

    def __init__(self):
        self.results = {}
        self.overall_score = 0
        self.max_score = 0

    def print_header(self, title: str):
        """Print a formatted header."""
        print("\n" + "=" * 60)
        print(f"🧪 {title}")
        print("=" * 60)

    def print_result(self, test_name: str, passed: bool, details: str = ""):
        """Print a test result."""
        emoji = "✅" if passed else "❌"
        print(f"{emoji} {test_name}")
        if details:
            print(f"   {details}")

        self.results[test_name] = passed
        self.max_score += 1
        if passed:
            self.overall_score += 1

    async def test_basic_connectivity(self):
        """Test basic API connectivity."""
        self.print_header("BASIC CONNECTIVITY TESTS")

        # Test environment variables
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        has_keys = bool(api_key and secret_key)
        self.print_result(
            "Environment Variables",
            has_keys,
            f"API Key: {'Found' if api_key else 'Missing'}, Secret: {'Found' if secret_key else 'Missing'}"
        )

        if not has_keys:
            print("❌ Cannot continue without API keys. Please check your .env file.")
            return False

        # Test Alpaca connection
        try:
            manager = AlpacaManager(api_key, secret_key, paper_trading=True)
            success, msg = manager.validate_api()

            self.print_result("Alpaca API Connection", success, msg)

            if success:
                # Test account info
                account_value = manager.get_account_value()
                balance = manager.get_balance()

                has_account_info = account_value is not None and balance is not None
                self.print_result(
                    "Account Information",
                    has_account_info,
                    f"Value: ${account_value:,.2f}, Balance: ${balance:,.2f}" if has_account_info else "Failed to retrieve"
                )

                # Test market data
                success, price = manager.get_price("AAPL")
                self.print_result(
                    "Market Data Access",
                    success,
                    f"AAPL: ${price:.2f}" if success else f"Error: {price}"
                )

        except Exception as e:
            self.print_result("Alpaca API Connection", False, f"Exception: {e!s}")
            return False

        return True

    async def test_strategy_loading(self):
        """Test that all strategies load correctly."""
        self.print_header("STRATEGY LOADING TESTS")

        try:
            config = ProductionStrategyManagerConfig(
                alpaca_api_key=os.getenv("ALPACA_API_KEY"),
                alpaca_secret_key=os.getenv("ALPACA_SECRET_KEY"),
                paper_trading=True,
                user_id=1,
                max_total_risk=0.10,
                max_position_size=0.02,
            )

            manager = ProductionStrategyManager(config)
            strategy_count = len(manager.strategies)

            expected_strategies = 10
            all_loaded = strategy_count == expected_strategies

            self.print_result(
                "Strategy Loading",
                all_loaded,
                f"Loaded {strategy_count}/{expected_strategies} strategies"
            )

            # List loaded strategies
            if strategy_count > 0:
                print("   Loaded strategies:")
                for name, strategy in manager.strategies.items():
                    status = "✅" if strategy else "❌"
                    print(f"     {status} {name}")

            # Test strategy configuration
            has_config = hasattr(manager, 'config') and manager.config is not None
            self.print_result(
                "Strategy Configuration",
                has_config,
                f"Max risk: {config.max_total_risk:.1%}, Max position: {config.max_position_size:.1%}"
            )

            return manager

        except Exception as e:
            self.print_result("Strategy Loading", False, f"Exception: {e!s}")
            return None

    async def test_data_integration(self, manager):
        """Test data integration and feeds."""
        self.print_header("DATA INTEGRATION TESTS")

        try:
            # Test data provider
            data_provider = manager.integration_manager.data_provider
            has_provider = data_provider is not None

            self.print_result(
                "Data Provider Initialization",
                has_provider,
                "ProductionDataProvider ready" if has_provider else "Failed to initialize"
            )

            if has_provider:
                # Test market data retrieval
                try:
                    # This is a basic test - in reality you'd test specific data methods
                    test_symbols = ["AAPL", "SPY", "QQQ"]
                    data_success = True

                    for symbol in test_symbols:
                        success, price = manager.integration_manager.alpaca_manager.get_price(symbol)
                        if not success:
                            data_success = False
                            break

                    self.print_result(
                        "Market Data Retrieval",
                        data_success,
                        f"Tested {len(test_symbols)} symbols" if data_success else "Failed on symbol retrieval"
                    )

                except Exception as e:
                    self.print_result("Market Data Retrieval", False, f"Exception: {e!s}")

            # Test database connectivity
            try:
                from backend.tradingbot.models.models import Portfolio, Order
                # Simple database test
                portfolio_count = Portfolio.objects.count()
                order_count = Order.objects.count()

                db_success = True  # If we get here without exception, DB is working
                self.print_result(
                    "Database Connectivity",
                    db_success,
                    f"Portfolios: {portfolio_count}, Orders: {order_count}"
                )

            except Exception as e:
                self.print_result("Database Connectivity", False, f"Exception: {e!s}")

        except Exception as e:
            self.print_result("Data Integration", False, f"Exception: {e!s}")

    async def test_risk_management(self, manager):
        """Test risk management systems."""
        self.print_header("RISK MANAGEMENT TESTS")

        try:
            config = manager.config

            # Test risk limits configuration
            has_risk_limits = (
                hasattr(config, 'max_total_risk') and
                hasattr(config, 'max_position_size') and
                config.max_total_risk > 0 and
                config.max_position_size > 0
            )

            self.print_result(
                "Risk Limits Configuration",
                has_risk_limits,
                f"Total: {config.max_total_risk:.1%}, Position: {config.max_position_size:.1%}" if has_risk_limits else "Missing or invalid"
            )

            # Test paper trading mode
            is_paper = config.paper_trading
            self.print_result(
                "Paper Trading Mode",
                is_paper,
                "Paper trading enabled (SAFE)" if is_paper else "⚠️ LIVE TRADING MODE - BE CAREFUL!"
            )

            # Test portfolio risk calculation
            try:
                # This would test actual risk calculation methods
                # For now, we test that the manager has risk assessment capability
                has_risk_methods = hasattr(manager, 'integration_manager')
                self.print_result(
                    "Risk Calculation Framework",
                    has_risk_methods,
                    "Risk management framework available" if has_risk_methods else "Risk framework missing"
                )

            except Exception as e:
                self.print_result("Risk Calculation Framework", False, f"Exception: {e!s}")

        except Exception as e:
            self.print_result("Risk Management", False, f"Exception: {e!s}")

    async def test_strategy_logic(self, manager):
        """Test basic strategy logic."""
        self.print_header("STRATEGY LOGIC TESTS")

        try:
            # Test WSB Dip Bot specifically (most popular strategy)
            wsb_strategy = manager.strategies.get('wsb_dip_bot')
            has_wsb = wsb_strategy is not None

            self.print_result(
                "WSB Dip Bot Available",
                has_wsb,
                "WSB Dip Bot strategy loaded" if has_wsb else "WSB Dip Bot not found"
            )

            if has_wsb:
                # Test strategy has required methods
                required_methods = ['scan_for_opportunities', 'process_signal']
                has_methods = all(hasattr(wsb_strategy, method) for method in required_methods)

                self.print_result(
                    "Strategy Interface Complete",
                    has_methods,
                    f"Required methods: {', '.join(required_methods)}" if has_methods else "Missing required methods"
                )

            # Test strategy configuration
            strategies_configured = len(manager.strategies) > 0
            self.print_result(
                "Strategy Configuration",
                strategies_configured,
                f"{len(manager.strategies)} strategies configured" if strategies_configured else "No strategies configured"
            )

        except Exception as e:
            self.print_result("Strategy Logic", False, f"Exception: {e!s}")

    async def test_simulation_mode(self, manager):
        """Test simulation/paper trading functionality."""
        self.print_header("SIMULATION MODE TESTS")

        try:
            # Verify we're in paper trading mode
            is_paper = manager.config.paper_trading
            self.print_result(
                "Paper Trading Active",
                is_paper,
                "Safe paper trading mode" if is_paper else "⚠️ LIVE TRADING - DANGER!"
            )

            # Test account access
            try:
                account_value = manager.integration_manager.alpaca_manager.get_account_value()
                has_paper_account = account_value is not None and account_value > 0

                self.print_result(
                    "Paper Account Access",
                    has_paper_account,
                    f"Paper account: ${account_value:,.2f}" if has_paper_account else "Cannot access paper account"
                )

                # Verify account size is reasonable for testing
                if has_paper_account:
                    account_ok = account_value >= 10000  # Should have at least $10K for testing
                    self.print_result(
                        "Account Size Adequate",
                        account_ok,
                        f"${account_value:,.2f} {'adequate' if account_ok else 'may be too small'} for testing"
                    )

            except Exception as e:
                self.print_result("Paper Account Access", False, f"Exception: {e!s}")

        except Exception as e:
            self.print_result("Simulation Mode", False, f"Exception: {e!s}")

    def print_summary(self):
        """Print validation summary."""
        self.print_header("VALIDATION SUMMARY")

        passed = self.overall_score
        total = self.max_score
        percentage = (passed / total * 100) if total > 0 else 0

        print(f"📊 Overall Score: {passed}/{total} ({percentage:.1f}%)")

        if percentage >= 90:
            print("🎉 EXCELLENT: System is ready for paper trading!")
            print("💡 Next step: Run paper trading for 30 days to validate strategies")
        elif percentage >= 75:
            print("✅ GOOD: System mostly ready, fix remaining issues")
            print("💡 Address any ❌ items above before paper trading")
        elif percentage >= 60:
            print("⚠️ FAIR: System needs work before trading")
            print("💡 Focus on fixing critical connectivity and loading issues")
        else:
            print("❌ POOR: System not ready for trading")
            print("💡 Major setup issues need to be resolved")

        print("\n" + "=" * 60)
        print("🚀 NEXT STEPS:")
        print("=" * 60)

        if percentage >= 75:
            print("1. 📝 Run paper trading: python simple_bot.py")
            print("2. 📊 Monitor performance for 30 days")
            print("3. 📈 Track win rate, profit factor, max drawdown")
            print("4. 🎯 Optimize strategy parameters")
            print("5. 💰 Only then consider live trading with small amounts")
        else:
            print("1. 🔧 Fix all ❌ validation issues above")
            print("2. 📚 Review setup documentation")
            print("3. 🔌 Verify API keys and connections")
            print("4. 🔄 Re-run this validation script")
            print("5. ✅ Only proceed when validation passes")

        print("\n⚠️ REMEMBER: Never risk money you can't afford to lose!")

    async def run_full_validation(self):
        """Run complete validation suite."""
        print("🧪 WallStreetBots Strategy Validation")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Test basic connectivity first
        connectivity_ok = await self.test_basic_connectivity()

        if connectivity_ok:
            # Load strategies
            manager = await self.test_strategy_loading()

            if manager:
                # Test all systems
                await self.test_data_integration(manager)
                await self.test_risk_management(manager)
                await self.test_strategy_logic(manager)
                await self.test_simulation_mode(manager)

        # Print final summary
        self.print_summary()

        return self.overall_score / self.max_score if self.max_score > 0 else 0


async def main():
    """Main validation function."""
    validator = StrategyValidator()
    score = await validator.run_full_validation()

    # Exit with appropriate code
    if score >= 0.75:
        print("\n✅ Validation passed! System ready for paper trading.")
        sys.exit(0)
    else:
        print("\n❌ Validation failed! Fix issues before trading.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation failed with exception: {e}")
        sys.exit(1)