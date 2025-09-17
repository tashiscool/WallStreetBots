# run.py (root)
"""
WallStreetBots Production CLI
Robust launcher for trading strategies with safety controls.
"""

from __future__ import annotations
import sys
import os
from typing import Optional

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install: pip install typer rich")
    sys.exit(1)

try:
    from backend.tradingbot.config.settings import load_settings, StrategyProfile
    from pydantic import ValidationError
except ImportError:
    # Fallback to simple settings
    from backend.tradingbot.config.simple_settings import load_settings, StrategyProfile

    ValidationError = ValueError
from backend.tradingbot.data.providers.client import MarketDataClient, BarSpec
from backend.tradingbot.risk.engines.engine import RiskEngine, RiskLimits
from backend.tradingbot.infra.obs import jlog
from backend.tradingbot.infra.obs import metrics as obs_metrics

app = typer.Typer(
    add_completion=False,
    help="WallStreetBots - Production trading system with safety controls",
)
console = Console()


@app.command()
def status():
    """Show current system configuration and status"""
    try:
        settings = load_settings()

        # Create status table
        table = Table(title="WallStreetBots Status")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Profile", settings.profile)
        table.add_row(
            "Paper Trading", "✅ ENABLED" if settings.alpaca_paper else "❌ LIVE MODE"
        )
        table.add_row("Dry Run", "✅ ENABLED" if settings.dry_run else "❌ DISABLED")
        table.add_row(
            "Advanced Analytics", "✅" if settings.enable_advanced_analytics else "❌"
        )
        table.add_row(
            "Market Regime Adapt",
            "✅" if settings.enable_market_regime_adaptation else "❌",
        )
        table.add_row("Max Total Risk", f"{settings.max_total_risk:.1%}")
        table.add_row("Max Position Size", f"{settings.max_position_size:.1%}")

        console.print(table)

        # Safety warnings
        if not settings.alpaca_paper:
            console.print(
                Panel(
                    "⚠️  LIVE TRADING MODE ENABLED ⚠️\nReal money at risk!",
                    style="bold red",
                )
            )

        if not settings.dry_run:
            console.print(
                Panel(
                    "🔥 DRY RUN DISABLED 🔥\nOrders will be executed!",
                    style="bold yellow",
                )
            )

    except ValidationError as e:
        console.print(f"❌ Configuration error: {e}", style="bold red")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"❌ Status check failed: {e}", style="bold red")
        raise typer.Exit(1) from e


@app.command()
def validate():
    """Validate system configuration and connections"""
    try:
        console.print("🔍 Validating configuration...")
        settings = load_settings()

        console.print("✅ Configuration loaded successfully")

        # Test market data connection
        console.print("🔍 Testing market data connection...")
        data_client = MarketDataClient()

        # Simple test - get SPY data
        test_spec = BarSpec("SPY", "1d", "5d")
        data = data_client.get_bars(test_spec)

        if not data.empty:
            console.print(f"✅ Market data connection OK ({len(data)} bars retrieved)")
        else:
            console.print("❌ Market data connection failed - no data returned")
            raise typer.Exit(1)

        # Test risk engine
        console.print("🔍 Testing risk engine...")
        risk_limits = RiskLimits(
            max_total_risk=settings.max_total_risk,
            max_position_size=settings.max_position_size,
        )
        risk_engine = RiskEngine(risk_limits)

        # Test basic functionality
        test_passed = risk_engine.pretrade_check(0.05, 0.03)  # 5% current + 3% new
        if test_passed:
            console.print("✅ Risk engine functioning correctly")
        else:
            console.print("❌ Risk engine test failed")
            raise typer.Exit(1)

        console.print("🎉 All validations passed!")

    except ValidationError as e:
        console.print(f"❌ Configuration validation failed: {e}", style="bold red")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"❌ System validation failed: {e}", style="bold red")
        raise typer.Exit(1) from e


@app.command()
def bars(
    symbol: str = typer.Argument(..., help="Stock symbol (e.g., AAPL, SPY)"),
    interval: str = typer.Option("1d", help="Bar interval (1m, 5m, 1h, 1d)"),
    lookback: str = typer.Option("30d", help="Lookback period (5d, 30d, 1y)"),
):
    """Fetch and display market data bars"""
    try:
        console.print(
            f"📊 Fetching {symbol} data ({interval} bars, {lookback} lookback)..."
        )

        data_client = MarketDataClient()
        spec = BarSpec(symbol.upper(), interval, lookback)
        data = data_client.get_bars(spec)

        if data.empty:
            console.print(f"❌ No data found for {symbol}")
            raise typer.Exit(1)

        # Display last 10 bars
        console.print(f"\n📈 Last 10 bars for {symbol}:")
        console.print(data.tail(10).to_string())

        # Basic stats
        if "close" in data.columns:
            current_price = data["close"].iloc[-1]
            price_change = (
                data["close"].iloc[-1] - data["close"].iloc[-2] if len(data) > 1 else 0
            )
            pct_change = (
                (price_change / data["close"].iloc[-2]) * 100
                if len(data) > 1 and data["close"].iloc[-2] != 0
                else 0
            )

            console.print(f"\n💰 Current: ${current_price:.2f} ({pct_change:+.2f}%)")

    except Exception as e:
        console.print(f"❌ Failed to fetch data: {e}", style="bold red")
        raise typer.Exit(1) from e


@app.command()
def metrics():
    """Display current system metrics"""
    try:
        current_metrics = obs_metrics.get_metrics()

        table = Table(title="System Metrics")
        table.add_column("Type", style="cyan")
        table.add_column("Name", style="yellow")
        table.add_column("Value", style="green")

        # Counters
        for name, value in current_metrics["counters"].items():
            table.add_row("Counter", name, str(value))

        # Gauges
        for name, value in current_metrics["gauges"].items():
            table.add_row("Gauge", name, f"{value:.2f}")

        # Histograms
        for name, stats in current_metrics["histograms"].items():
            table.add_row("Histogram", f"{name} (count)", str(stats["count"]))
            table.add_row("", f"{name} (sum)", f"{stats['sum']:.2f}")

        console.print(table)

    except Exception as e:
        console.print(f"❌ Failed to get metrics: {e}", style="bold red")
        raise typer.Exit(1) from e


@app.command()
def market():
    """Check if market is currently open"""
    try:
        data_client = MarketDataClient()
        is_open = data_client.is_market_open()

        status = "🟢 OPEN" if is_open else "🔴 CLOSED"
        console.print(f"Market Status: {status}")

    except Exception as e:
        console.print(f"❌ Failed to check market status: {e}", style="bold red")
        raise typer.Exit(1) from e


@app.command()
def version():
    """Show version information"""
    console.print("WallStreetBots v0.1.0")
    console.print("Production-ready algorithmic trading system")


if __name__ == "__main__":
    app()
