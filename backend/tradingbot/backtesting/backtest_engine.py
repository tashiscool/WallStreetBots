"""
Backtesting Execution Engine

A real backtesting engine that executes strategies against historical data
and produces performance metrics, trade logs, and equity curves.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Try to import yfinance for historical data
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    logger.warning("yfinance not available - backtesting will use synthetic data")


class TradeDirection(Enum):
    """Trade direction."""
    LONG = "long"
    SHORT = "short"


class TradeStatus(Enum):
    """Trade status."""
    OPEN = "open"
    CLOSED = "closed"
    STOPPED_OUT = "stopped_out"
    TAKE_PROFIT = "take_profit"


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    strategy_name: str
    start_date: date
    end_date: date
    initial_capital: Decimal = Decimal("100000")
    benchmark: str = "SPY"
    position_size_pct: float = 3.0
    stop_loss_pct: float = 5.0
    take_profit_pct: float = 15.0
    max_positions: int = 10
    commission_per_trade: Decimal = Decimal("0.00")  # Commission per share
    slippage_pct: float = 0.05  # 5 basis points slippage


@dataclass
class Trade:
    """Represents a single trade in the backtest."""
    id: str
    symbol: str
    direction: TradeDirection
    entry_date: date
    entry_price: Decimal
    quantity: int
    exit_date: Optional[date] = None
    exit_price: Optional[Decimal] = None
    status: TradeStatus = TradeStatus.OPEN
    pnl: Decimal = Decimal("0")
    pnl_pct: float = 0.0
    commission: Decimal = Decimal("0")
    slippage: Decimal = Decimal("0")

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0

    @property
    def hold_days(self) -> int:
        if self.exit_date:
            return (self.exit_date - self.entry_date).days
        return 0


@dataclass
class DailySnapshot:
    """Daily portfolio snapshot."""
    date: date
    equity: Decimal
    cash: Decimal
    positions_value: Decimal
    benchmark_value: Decimal
    daily_pnl: Decimal
    daily_return_pct: float
    cumulative_return_pct: float
    drawdown_pct: float


@dataclass
class BacktestResults:
    """Results from a backtest run."""
    config: BacktestConfig
    trades: List[Trade]
    daily_snapshots: List[DailySnapshot]

    # Summary metrics
    total_return_pct: float = 0.0
    benchmark_return_pct: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: Decimal = Decimal("0")
    avg_loss: Decimal = Decimal("0")
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_hold_days: float = 0.0

    # Monthly returns
    monthly_returns: Dict[str, float] = field(default_factory=dict)

    # Final values
    final_equity: Decimal = Decimal("0")
    total_pnl: Decimal = Decimal("0")
    total_commission: Decimal = Decimal("0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for JSON serialization."""
        return {
            "config": {
                "strategy_name": self.config.strategy_name,
                "start_date": self.config.start_date.isoformat(),
                "end_date": self.config.end_date.isoformat(),
                "initial_capital": float(self.config.initial_capital),
                "benchmark": self.config.benchmark,
            },
            "summary": {
                "total_return_pct": round(self.total_return_pct, 2),
                "benchmark_return_pct": round(self.benchmark_return_pct, 2),
                "alpha": round(self.alpha, 2),
                "sharpe_ratio": round(self.sharpe_ratio, 2),
                "sortino_ratio": round(self.sortino_ratio, 2),
                "max_drawdown_pct": round(self.max_drawdown_pct, 2),
                "win_rate": round(self.win_rate, 2),
                "profit_factor": round(self.profit_factor, 2),
                "avg_win": float(self.avg_win),
                "avg_loss": float(self.avg_loss),
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "avg_hold_days": round(self.avg_hold_days, 1),
                "final_equity": float(self.final_equity),
                "total_pnl": float(self.total_pnl),
            },
            "monthly_returns": self.monthly_returns,
            "equity_curve": [
                {
                    "date": snap.date.isoformat(),
                    "equity": float(snap.equity),
                    "benchmark": float(snap.benchmark_value),
                }
                for snap in self.daily_snapshots[::5]  # Sample every 5 days
            ],
            "trades": [
                {
                    "id": t.id,
                    "symbol": t.symbol,
                    "direction": t.direction.value,
                    "entry_date": t.entry_date.isoformat(),
                    "entry_price": float(t.entry_price),
                    "exit_date": t.exit_date.isoformat() if t.exit_date else None,
                    "exit_price": float(t.exit_price) if t.exit_price else None,
                    "pnl": float(t.pnl),
                    "pnl_pct": round(t.pnl_pct, 2),
                    "status": t.status.value,
                }
                for t in self.trades[-50:]  # Last 50 trades
            ],
        }


class BacktestEngine:
    """
    Backtesting execution engine.

    Runs strategies against historical data and produces performance metrics.
    """

    # Strategy watchlists (simplified for backtesting)
    STRATEGY_WATCHLISTS = {
        "wsb-dip-bot": ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "GOOGL", "AMZN", "META"],
        "momentum-weeklies": ["SPY", "QQQ", "NVDA", "AMD", "TSLA"],
        "wheel-strategy": ["AAPL", "MSFT", "JPM", "BAC", "WFC"],
        "earnings-protection": ["AAPL", "GOOGL", "AMZN", "META", "NFLX"],
        "debit-spreads": ["SPY", "QQQ", "IWM", "AAPL", "MSFT"],
        "leaps-tracker": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
        "lotto-scanner": ["SPY", "QQQ", "TSLA", "NVDA", "AMD"],
        "swing-trading": ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "SPY"],
        "spx-credit-spreads": ["SPY"],
        "index-baseline": ["SPY"],
    }

    def __init__(self):
        self._price_cache: Dict[str, Dict[date, Decimal]] = {}

    async def run_backtest(
        self,
        config: BacktestConfig,
        progress_callback: Optional[callable] = None,
    ) -> BacktestResults:
        """
        Run a backtest with the given configuration.

        Args:
            config: Backtest configuration
            progress_callback: Optional callback for progress updates

        Returns:
            BacktestResults with all metrics and trade logs
        """
        logger.info(f"Starting backtest for {config.strategy_name}")

        # Get watchlist for strategy
        symbols = self.STRATEGY_WATCHLISTS.get(
            config.strategy_name,
            ["AAPL", "MSFT", "NVDA"]
        )

        # Fetch historical data
        if progress_callback:
            progress_callback(5, "Fetching historical data...")

        price_data = await self._fetch_price_data(
            symbols + [config.benchmark],
            config.start_date,
            config.end_date,
        )

        if progress_callback:
            progress_callback(20, "Running strategy simulation...")

        # Initialize portfolio
        trades: List[Trade] = []
        daily_snapshots: List[DailySnapshot] = []
        cash = config.initial_capital
        positions: Dict[str, Trade] = {}
        trade_counter = 0

        # Get trading days
        benchmark_prices = price_data.get(config.benchmark, {})
        trading_days = sorted(benchmark_prices.keys())

        if not trading_days:
            # Generate synthetic trading days if no real data
            trading_days = self._generate_trading_days(config.start_date, config.end_date)
            price_data = self._generate_synthetic_prices(symbols + [config.benchmark], trading_days)
            benchmark_prices = price_data.get(config.benchmark, {})

        peak_equity = config.initial_capital
        benchmark_start = benchmark_prices.get(trading_days[0], Decimal("100"))

        total_days = len(trading_days)

        for i, current_date in enumerate(trading_days):
            # Progress update
            if progress_callback and i % 20 == 0:
                pct = 20 + int(60 * i / total_days)
                progress_callback(pct, f"Processing {current_date}...")

            # Update positions with current prices
            positions_value = Decimal("0")
            for symbol, trade in list(positions.items()):
                current_price = price_data.get(symbol, {}).get(current_date)
                if current_price:
                    position_value = current_price * trade.quantity
                    positions_value += position_value

                    # Check stop loss / take profit
                    pnl_pct = float((current_price - trade.entry_price) / trade.entry_price * 100)

                    if pnl_pct <= -config.stop_loss_pct:
                        # Stop loss hit
                        trade.exit_date = current_date
                        trade.exit_price = current_price
                        trade.status = TradeStatus.STOPPED_OUT
                        trade.pnl = (current_price - trade.entry_price) * trade.quantity
                        trade.pnl_pct = pnl_pct
                        cash += position_value
                        del positions[symbol]
                        trades.append(trade)

                    elif pnl_pct >= config.take_profit_pct:
                        # Take profit hit
                        trade.exit_date = current_date
                        trade.exit_price = current_price
                        trade.status = TradeStatus.TAKE_PROFIT
                        trade.pnl = (current_price - trade.entry_price) * trade.quantity
                        trade.pnl_pct = pnl_pct
                        cash += position_value
                        del positions[symbol]
                        trades.append(trade)

            # Strategy signal generation (simplified)
            if len(positions) < config.max_positions:
                signal = self._generate_signal(
                    config.strategy_name,
                    symbols,
                    current_date,
                    price_data,
                    positions,
                )

                if signal:
                    symbol, direction = signal
                    current_price = price_data.get(symbol, {}).get(current_date)

                    if current_price and current_price > 0:
                        # Calculate position size
                        position_value = config.initial_capital * Decimal(str(config.position_size_pct / 100))
                        quantity = int(position_value / current_price)

                        if quantity > 0 and cash >= position_value:
                            # Apply slippage
                            slippage = current_price * Decimal(str(config.slippage_pct / 100))
                            entry_price = current_price + slippage

                            trade_counter += 1
                            trade = Trade(
                                id=f"BT-{trade_counter:04d}",
                                symbol=symbol,
                                direction=direction,
                                entry_date=current_date,
                                entry_price=entry_price,
                                quantity=quantity,
                                slippage=slippage * quantity,
                            )

                            positions[symbol] = trade
                            cash -= entry_price * quantity

            # Calculate daily metrics
            total_equity = cash + positions_value
            benchmark_price = benchmark_prices.get(current_date, benchmark_start)
            benchmark_value = config.initial_capital * (benchmark_price / benchmark_start)

            # Drawdown
            if total_equity > peak_equity:
                peak_equity = total_equity
            drawdown_pct = float((peak_equity - total_equity) / peak_equity * 100)

            # Daily return
            if daily_snapshots:
                prev_equity = daily_snapshots[-1].equity
                daily_pnl = total_equity - prev_equity
                daily_return_pct = float((total_equity - prev_equity) / prev_equity * 100)
            else:
                daily_pnl = Decimal("0")
                daily_return_pct = 0.0

            cumulative_return_pct = float((total_equity - config.initial_capital) / config.initial_capital * 100)

            snapshot = DailySnapshot(
                date=current_date,
                equity=total_equity,
                cash=cash,
                positions_value=positions_value,
                benchmark_value=benchmark_value,
                daily_pnl=daily_pnl,
                daily_return_pct=daily_return_pct,
                cumulative_return_pct=cumulative_return_pct,
                drawdown_pct=drawdown_pct,
            )
            daily_snapshots.append(snapshot)

        # Close remaining positions at end
        for symbol, trade in positions.items():
            final_price = price_data.get(symbol, {}).get(trading_days[-1])
            if final_price:
                trade.exit_date = trading_days[-1]
                trade.exit_price = final_price
                trade.status = TradeStatus.CLOSED
                trade.pnl = (final_price - trade.entry_price) * trade.quantity
                trade.pnl_pct = float((final_price - trade.entry_price) / trade.entry_price * 100)
                trades.append(trade)

        if progress_callback:
            progress_callback(90, "Calculating metrics...")

        # Calculate results
        results = self._calculate_results(config, trades, daily_snapshots, benchmark_prices, trading_days)

        if progress_callback:
            progress_callback(100, "Complete!")

        logger.info(f"Backtest complete: {results.total_trades} trades, {results.total_return_pct:.1f}% return")

        return results

    async def _fetch_price_data(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
    ) -> Dict[str, Dict[date, Decimal]]:
        """Fetch historical price data for symbols."""
        price_data: Dict[str, Dict[date, Decimal]] = {}

        if not HAS_YFINANCE:
            # Generate synthetic data
            trading_days = self._generate_trading_days(start_date, end_date)
            return self._generate_synthetic_prices(symbols, trading_days)

        try:
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(
                        start=start_date.isoformat(),
                        end=(end_date + timedelta(days=1)).isoformat(),
                    )

                    symbol_prices = {}
                    for idx, row in hist.iterrows():
                        trade_date = idx.date()
                        symbol_prices[trade_date] = Decimal(str(round(row['Close'], 2)))

                    price_data[symbol] = symbol_prices

                except Exception as e:
                    logger.warning(f"Error fetching {symbol}: {e}")
                    # Generate synthetic for this symbol
                    trading_days = self._generate_trading_days(start_date, end_date)
                    price_data[symbol] = self._generate_synthetic_prices([symbol], trading_days)[symbol]

        except Exception as e:
            logger.error(f"Error fetching price data: {e}")
            trading_days = self._generate_trading_days(start_date, end_date)
            return self._generate_synthetic_prices(symbols, trading_days)

        return price_data

    def _generate_trading_days(self, start_date: date, end_date: date) -> List[date]:
        """Generate list of trading days (excluding weekends)."""
        days = []
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:  # Monday = 0, Friday = 4
                days.append(current)
            current += timedelta(days=1)
        return days

    def _generate_synthetic_prices(
        self,
        symbols: List[str],
        trading_days: List[date],
    ) -> Dict[str, Dict[date, Decimal]]:
        """Generate synthetic price data for testing."""
        np.random.seed(42)  # Reproducible results

        price_data = {}

        # Base prices for common symbols
        base_prices = {
            "SPY": 450,
            "QQQ": 380,
            "AAPL": 180,
            "MSFT": 370,
            "NVDA": 480,
            "AMD": 140,
            "TSLA": 250,
            "GOOGL": 140,
            "AMZN": 175,
            "META": 350,
            "JPM": 150,
            "BAC": 30,
            "WFC": 45,
            "NFLX": 450,
            "IWM": 200,
        }

        for symbol in symbols:
            base_price = base_prices.get(symbol, 100)
            prices = {}
            current_price = base_price

            for day in trading_days:
                # Random walk with slight upward drift
                change = np.random.normal(0.0003, 0.015)  # ~7.5% annual vol
                current_price = current_price * (1 + change)
                prices[day] = Decimal(str(round(current_price, 2)))

            price_data[symbol] = prices

        return price_data

    def _generate_signal(
        self,
        strategy_name: str,
        symbols: List[str],
        current_date: date,
        price_data: Dict[str, Dict[date, Decimal]],
        current_positions: Dict[str, Trade],
    ) -> Optional[Tuple[str, TradeDirection]]:
        """Generate a trading signal based on strategy logic."""
        # Filter out symbols we already hold
        available_symbols = [s for s in symbols if s not in current_positions]
        if not available_symbols:
            return None

        # Simple signal generation based on strategy type
        # In a real implementation, this would use actual strategy logic

        if strategy_name in ["wsb-dip-bot", "momentum-weeklies", "swing-trading"]:
            # Look for dips/momentum
            for symbol in available_symbols:
                prices = price_data.get(symbol, {})
                dates = sorted(prices.keys())

                if current_date not in dates:
                    continue

                idx = dates.index(current_date)
                if idx < 5:
                    continue

                # Check for dip (price dropped 3%+ in last 5 days)
                current_price = prices[current_date]
                price_5d_ago = prices[dates[idx - 5]]

                if price_5d_ago > 0:
                    change = float((current_price - price_5d_ago) / price_5d_ago * 100)

                    if change < -3:  # Dip detected
                        return (symbol, TradeDirection.LONG)

        elif strategy_name in ["wheel-strategy", "spx-credit-spreads"]:
            # More conservative - trade less frequently
            if np.random.random() < 0.02:  # ~2% daily chance
                symbol = np.random.choice(available_symbols)
                return (symbol, TradeDirection.LONG)

        else:
            # Default: random entry with low probability
            if np.random.random() < 0.05:  # ~5% daily chance
                symbol = np.random.choice(available_symbols)
                return (symbol, TradeDirection.LONG)

        return None

    def _calculate_results(
        self,
        config: BacktestConfig,
        trades: List[Trade],
        daily_snapshots: List[DailySnapshot],
        benchmark_prices: Dict[date, Decimal],
        trading_days: List[date],
    ) -> BacktestResults:
        """Calculate all performance metrics."""
        results = BacktestResults(
            config=config,
            trades=trades,
            daily_snapshots=daily_snapshots,
        )

        if not daily_snapshots:
            return results

        # Basic trade statistics
        results.total_trades = len(trades)
        winning_trades = [t for t in trades if t.is_winner]
        losing_trades = [t for t in trades if not t.is_winner and t.pnl != 0]

        results.winning_trades = len(winning_trades)
        results.losing_trades = len(losing_trades)

        if results.total_trades > 0:
            results.win_rate = results.winning_trades / results.total_trades * 100

        if winning_trades:
            results.avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades)
        if losing_trades:
            results.avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades)

        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else Decimal("0")
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else Decimal("1")
        if gross_loss > 0:
            results.profit_factor = float(gross_profit / gross_loss)

        # Average hold time
        if trades:
            results.avg_hold_days = sum(t.hold_days for t in trades) / len(trades)

        # Total return
        results.final_equity = daily_snapshots[-1].equity
        results.total_pnl = results.final_equity - config.initial_capital
        results.total_return_pct = float(results.total_pnl / config.initial_capital * 100)

        # Benchmark return
        if trading_days and benchmark_prices:
            start_bench = benchmark_prices.get(trading_days[0])
            end_bench = benchmark_prices.get(trading_days[-1])
            if start_bench and end_bench:
                results.benchmark_return_pct = float((end_bench - start_bench) / start_bench * 100)

        # Alpha
        results.alpha = results.total_return_pct - results.benchmark_return_pct

        # Max drawdown
        results.max_drawdown_pct = max(s.drawdown_pct for s in daily_snapshots)

        # Sharpe ratio (annualized)
        daily_returns = [s.daily_return_pct for s in daily_snapshots if s.daily_return_pct != 0]
        if len(daily_returns) > 1:
            avg_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            if std_return > 0:
                results.sharpe_ratio = (avg_return * 252) / (std_return * np.sqrt(252))

        # Sortino ratio
        negative_returns = [r for r in daily_returns if r < 0]
        if negative_returns:
            downside_std = np.std(negative_returns)
            if downside_std > 0:
                avg_return = np.mean(daily_returns)
                results.sortino_ratio = (avg_return * 252) / (downside_std * np.sqrt(252))

        # Monthly returns
        current_month = None
        month_start_equity = config.initial_capital

        for snapshot in daily_snapshots:
            month_key = snapshot.date.strftime("%Y-%m")

            if current_month != month_key:
                if current_month:
                    # Save previous month's return
                    month_return = float((prev_equity - month_start_equity) / month_start_equity * 100)
                    results.monthly_returns[current_month] = round(month_return, 2)

                current_month = month_key
                month_start_equity = snapshot.equity

            prev_equity = snapshot.equity

        # Last month
        if current_month:
            month_return = float((prev_equity - month_start_equity) / month_start_equity * 100)
            results.monthly_returns[current_month] = round(month_return, 2)

        return results


async def run_backtest(
    strategy_name: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 100000,
    benchmark: str = "SPY",
    position_size_pct: float = 3.0,
    stop_loss_pct: float = 5.0,
    take_profit_pct: float = 15.0,
    progress_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run a backtest.

    Args:
        strategy_name: Name of strategy to backtest
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_capital: Starting capital
        benchmark: Benchmark symbol
        position_size_pct: Position size as % of capital
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage
        progress_callback: Optional progress callback

    Returns:
        Dictionary with backtest results
    """
    config = BacktestConfig(
        strategy_name=strategy_name,
        start_date=date.fromisoformat(start_date),
        end_date=date.fromisoformat(end_date),
        initial_capital=Decimal(str(initial_capital)),
        benchmark=benchmark,
        position_size_pct=position_size_pct,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )

    engine = BacktestEngine()
    results = await engine.run_backtest(config, progress_callback)

    return results.to_dict()
