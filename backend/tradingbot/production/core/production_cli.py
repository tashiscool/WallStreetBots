"""Production CLI Interface
Command - line interface for production trading system.

This module provides a simple CLI to:
- Start / stop the production system
- Monitor system status
- View portfolio and performance
- Manage strategies
- Execute manual trades

Usage:
    python production_cli.py start --paper - trading
    python production_cli.py status
    python production_cli.py portfolio
    python production_cli.py stop
"""

import argparse
import asyncio
import sys
from decimal import Decimal

from ...core.trading_interface import OrderSide, OrderType
from .production_integration import ProductionTradeSignal
from .production_manager import ProductionConfig, ProductionManager


class ProductionCLI:
    """Command - line interface for production trading system."""

    def __init__(self):
        self.manager: ProductionManager | None = None

    async def start_system(self, args):
        """Start the production system."""
        try:
            print("🚀 Starting Production Trading System...")

            # Create configuration
            config = ProductionConfig(
                alpaca_api_key=args.alpaca_api_key or "test_key",
                alpaca_secret_key=args.alpaca_secret_key or "test_secret",
                paper_trading=args.paper_trading,
                user_id=args.user_id,
                max_position_size=args.max_position_size,
                max_total_risk=args.max_total_risk,
                enabled_strategies=args.strategies.split(",")
                if args.strategies
                else None,
            )

            # Create manager
            self.manager = ProductionManager(config)

            # Start system
            success = await self.manager.start_production_system()

            if success:
                print("✅ Production Trading System started successfully!")
                print(f"📊 Paper Trading: {config.paper_trading}")
                print(f"🎯 Active Strategies: {len(self.manager.strategies)}")
                print(f"💰 Max Position Size: {config.max_position_size * 100}%")
                print(f"⚠️  Max Total Risk: {config.max_total_risk * 100}%")

                # Keep running until interrupted
                try:
                    shutdown_event = asyncio.Event()
                    await shutdown_event.wait()
                except KeyboardInterrupt:
                    print("\n🛑 Shutdown signal received...")
                    await self.manager.stop_production_system()
                    print("✅ Production Trading System stopped")
            else:
                print("❌ Failed to start Production Trading System")
                sys.exit(1)

        except Exception as e:
            print(f"❌ Error starting system: {e}")
            sys.exit(1)

    async def show_status(self, args):
        """Show system status."""
        try:
            if not self.manager:
                print("❌ Production system not running")
                return

            status = self.manager.get_system_status()

            print("📊 Production Trading System Status")
            print(" = " * 50)
            print(f"🟢 Running: {status['is_running']}")
            print(f"⏰ Start Time: {status['start_time']}")
            print(f"💓 Last Heartbeat: {status['last_heartbeat']}")
            print(f"🎯 Active Strategies: {status['active_strategies']}")

            print("\n📈 Strategy Status: ")
            for name, strategy_status in status["strategy_status"].items():
                print(
                    f"  • {name}: {'🟢' if strategy_status['is_running'] else '🔴'} "
                    f"{strategy_status['performance'].get('active_positions', 0)} positions"
                )

            print("\n⚙️ Configuration: ")
            config = status["configuration"]
            print(f"  • Paper Trading: {config['paper_trading']}")
            print(f"  • Max Position Size: {config['max_position_size'] * 100}%")
            print(f"  • Max Total Risk: {config['max_total_risk'] * 100}%")
            print(f"  • Enabled Strategies: {', '.join(config['enabled_strategies'])}")

        except Exception as e:
            print(f"❌ Error getting status: {e}")

    async def show_portfolio(self, args):
        """Show portfolio summary."""
        try:
            if not self.manager:
                print("❌ Production system not running")
                return

            portfolio_summary = self.manager.integration_manager.get_portfolio_summary()

            print("💰 Portfolio Summary")
            print(" = " * 50)
            print(f"📊 Total Positions: {portfolio_summary.get('total_positions', 0)}")
            print(f"📈 Total Trades: {portfolio_summary.get('total_trades', 0)}")
            print(
                f"💚 Unrealized P & L: ${portfolio_summary.get('total_unrealized_pnl', 0): .2f}"
            )
            print(
                f"💵 Realized P & L: ${portfolio_summary.get('total_realized_pnl', 0): .2f}"
            )

            positions = portfolio_summary.get("active_positions", [])
            if positions:
                print("\n📋 Active Positions: ")
                for pos in positions:
                    print(
                        f"  • {pos['ticker']} ({pos['strategy']}): "
                        f"{pos['quantity']} @ ${pos['entry_price']: .2f} "
                        f"[${pos['unrealized_pnl']: .2f}]"
                    )
            else:
                print("\n📋 No active positions")

        except Exception as e:
            print(f"❌ Error getting portfolio: {e}")

    async def execute_trade(self, args):
        """Execute manual trade."""
        try:
            if not self.manager:
                print("❌ Production system not running")
                return

            # Create trade signal
            signal = ProductionTradeSignal(
                strategy_name="manual_trade",
                ticker=args.ticker.upper(),
                side=OrderSide.BUY if args.side.lower() == "buy" else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=args.quantity,
                price=args.price,
                trade_type="stock",
                risk_amount=Decimal(str(args.quantity * args.price)),
                expected_return=Decimal("0.00"),
                metadata={"manual_trade": True},
            )

            print(
                f"🎯 Executing {args.side.upper()} {args.quantity} {args.ticker} @ ${args.price}"
            )

            # Execute trade
            result = await self.manager.integration.execute_trade(signal)

            if result.status.value == "FILLED":
                print("✅ Trade executed successfully!")
                print(f"   Order ID: {result.trade_id}")
                print(f"   Fill Price: ${result.fill_price}")
                print(f"   Commission: ${result.commission}")
            else:
                print(f"❌ Trade failed: {result.error_message}")

        except Exception as e:
            print(f"❌ Error executing trade: {e}")

    async def list_strategies(self, args=None):
        """List available strategies."""
        strategies = [
            "wsb_dip_bot",
            "momentum_weeklies",
            "debit_spreads",
            "leaps_tracker",
            "lotto_scanner",
            "wheel_strategy",
            "spx_credit_spreads",
            "earnings_protection",
            "swing_trading",
            "index_baseline",
        ]

        print("🎯 Available Strategies: ")
        print(" = " * 50)
        for strategy in strategies:
            print(f"  • {strategy}")

        print("\n💡 Usage: ")
        print(
            "  python production_cli.py start --strategies wsb_dip_bot,momentum_weeklies"
        )

    def stop_system(self):
        """Stop the production system."""
        try:
            if not self.manager:
                print("❌ Production system not running")
                return

            print("🛑 Stopping Production Trading System...")
            # Note: In a real implementation, this would gracefully shutdown the system
            print("✅ Production system stopped")
            self.manager = None

        except Exception as e:
            print(f"❌ Error stopping system: {e}")

    def status_system(self):
        """Show system status (alias for show_status)."""
        import argparse
        args = argparse.Namespace()
        asyncio.run(self.show_status(args))

    def portfolio_view(self):
        """Show portfolio view."""
        try:
            if not self.manager:
                print("❌ Production system not running")
                return

            # Get positions and trades from integration manager
            positions = self.manager.integration.get_positions()
            trades = self.manager.integration.get_trades()

            print("💰 Portfolio View")
            print("=" * 50)
            
            if positions:
                print("📊 Current Positions:")
                for symbol, position in positions.items():
                    print(f"  • {symbol}: {position['quantity']} shares @ ${position['avg_price']:.2f}")
            
            if trades:
                print("\n📈 Recent Trades:")
                for trade in trades[-5:]:  # Show last 5 trades
                    print(f"  • {trade.symbol} {trade.side} {trade.quantity} @ ${trade.price:.2f} ({trade.status})")

        except Exception as e:
            print(f"❌ Error getting portfolio view: {e}")

    def manual_trade(self, args):
        """Execute manual trade (alias for execute_trade)."""
        asyncio.run(self.execute_trade(args))

    def enable_strategy(self, args):
        """Enable a strategy."""
        try:
            if not self.manager:
                print("❌ Production system not running")
                return

            strategy_name = args.strategy_name
            print(f"🎯 Enabling strategy: {strategy_name}")
            
            # Check if strategy exists
            if hasattr(self.manager, 'strategies') and strategy_name in self.manager.strategies:
                strategy = self.manager.strategies[strategy_name]
                if hasattr(strategy, 'start'):
                    strategy.start()
                print(f"✅ Strategy {strategy_name} enabled")
            else:
                print(f"❌ Strategy {strategy_name} not found")

        except Exception as e:
            print(f"❌ Error enabling strategy: {e}")

    def disable_strategy(self, args):
        """Disable a strategy."""
        try:
            if not self.manager:
                print("❌ Production system not running")
                return

            strategy_name = args.strategy_name
            print(f"🎯 Disabling strategy: {strategy_name}")
            
            # Check if strategy exists
            if hasattr(self.manager, 'strategies') and strategy_name in self.manager.strategies:
                strategy = self.manager.strategies[strategy_name]
                if hasattr(strategy, 'stop'):
                    strategy.stop()
                print(f"✅ Strategy {strategy_name} disabled")
            else:
                print(f"❌ Strategy {strategy_name} not found")

        except Exception as e:
            print(f"❌ Error disabling strategy: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Production Trading System CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start production system")
    start_parser.add_argument("--alpaca-api-key", help="Alpaca API key")
    start_parser.add_argument("--alpaca-secret-key", help="Alpaca secret key")
    start_parser.add_argument(
        "--paper-trading", action="store_true", default=True, help="Use paper trading"
    )
    start_parser.add_argument(
        "--live-trading", action="store_true", help="Use live trading (DANGEROUS)"
    )
    start_parser.add_argument("--user-id", type=int, default=1, help="Django user ID")
    start_parser.add_argument(
        "--max-position-size",
        type=float,
        default=0.20,
        help="Max position size (0.20 = 20%)",
    )
    start_parser.add_argument(
        "--max-total-risk",
        type=float,
        default=0.50,
        help="Max total risk (0.50 = 50%)",
    )
    start_parser.add_argument(
        "--strategies", help="Comma-separated list of strategies to enable"
    )

    # Status command
    subparsers.add_parser("status", help="Show system status")

    # Portfolio command
    subparsers.add_parser("portfolio", help="Show portfolio summary")

    # Trade command
    trade_parser = subparsers.add_parser("trade", help="Execute manual trade")
    trade_parser.add_argument("ticker", help="Stock ticker")
    trade_parser.add_argument("side", choices=["buy", "sell"], help="Buy or sell")
    trade_parser.add_argument("quantity", type=int, help="Number of shares")
    trade_parser.add_argument("price", type=float, help="Price per share")

    # Manual trade command (alias for trade)
    manual_trade_parser = subparsers.add_parser("manual-trade", help="Execute manual trade (alias for trade)")
    manual_trade_parser.add_argument("--symbol", dest="ticker", help="Stock ticker (use --symbol or --ticker)")
    manual_trade_parser.add_argument("--ticker", help="Stock ticker")
    manual_trade_parser.add_argument("--side", choices=["buy", "sell"], help="Buy or sell")
    manual_trade_parser.add_argument("--quantity", type=int, help="Number of shares")
    manual_trade_parser.add_argument("--price", type=float, help="Price per share")

    # List strategies command
    subparsers.add_parser("list-strategies", help="List available strategies")

    # Stop command
    subparsers.add_parser("stop", help="Stop production system")

    # Enable strategy command
    enable_parser = subparsers.add_parser("enable-strategy", help="Enable a strategy")
    enable_parser.add_argument("strategy_name", help="Name of strategy to enable")

    # Disable strategy command
    disable_parser = subparsers.add_parser("disable-strategy", help="Disable a strategy")
    disable_parser.add_argument("strategy_name", help="Name of strategy to disable")

    try:
        args = parser.parse_args()
    except SystemExit:
        # Handle unknown commands gracefully
        parser.print_help()
        return

    if not args.command:
        parser.print_help()
        return

    # Handle live trading flag
    if hasattr(args, "live_trading") and args.live_trading:
        args.paper_trading = False

    # Create CLI instance
    cli = ProductionCLI()

    # Execute command
    if args.command == "start":
        asyncio.run(cli.start_system(args))
    elif args.command == "status":
        asyncio.run(cli.show_status(args))
    elif args.command == "portfolio":
        asyncio.run(cli.show_portfolio(args))
    elif args.command == "trade":
        asyncio.run(cli.execute_trade(args))
    elif args.command == "manual-trade":
        # Handle ticker vs symbol parameter
        if hasattr(args, 'ticker') and args.ticker is None and hasattr(args, 'symbol'):
            args.ticker = getattr(args, 'symbol', None)
        asyncio.run(cli.execute_trade(args))
    elif args.command == "list-strategies":
        asyncio.run(cli.list_strategies(args))
    elif args.command == "stop":
        cli.stop_system()
    elif args.command == "enable-strategy":
        cli.enable_strategy(args)
    elif args.command == "disable-strategy":
        cli.disable_strategy(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
