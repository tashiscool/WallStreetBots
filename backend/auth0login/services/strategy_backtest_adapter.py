"""Strategy Backtest Adapter - Bridges CustomStrategy with BacktestEngine.

Converts CustomStrategy definitions into executable backtests using the
real BacktestEngine with historical price data.
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
import uuid

import pandas as pd

from backend.tradingbot.backtesting.backtest_engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResults,
    Trade,
    TradeDirection,
    TradeStatus,
    DailySnapshot,
)
from backend.tradingbot.models.models import CustomStrategy, BacktestRun, BacktestTrade
from backend.auth0login.services.custom_strategy_runner import CustomStrategyRunner

logger = logging.getLogger(__name__)

# Try to import yfinance for historical data
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    logger.warning("yfinance not available - using synthetic data")


@dataclass
class CustomStrategyBacktestConfig:
    """Configuration for custom strategy backtest."""
    strategy: CustomStrategy
    start_date: date
    end_date: date
    initial_capital: Decimal = Decimal("100000")
    benchmark: str = "SPY"


class CustomStrategyBacktestAdapter:
    """Adapts CustomStrategy for use with BacktestEngine.

    This adapter:
    1. Fetches historical price data for the strategy's universe
    2. Uses CustomStrategyRunner to evaluate entry/exit conditions
    3. Simulates trades based on signals
    4. Calculates performance metrics
    """

    # Universe symbol mappings
    UNIVERSE_SYMBOLS = {
        'sp500': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'JPM', 'V'],
        'nasdaq100': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'ADBE', 'COST'],
        'dow30': ['AAPL', 'MSFT', 'JPM', 'V', 'UNH', 'HD', 'PG', 'JNJ', 'WMT', 'CVX'],
        'russell2000': ['AMC', 'GME', 'RIOT', 'MARA', 'SOFI', 'PLTR', 'BB', 'NOK', 'CLOV', 'WISH'],
        'all': ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'AMD', 'TSLA', 'GOOGL', 'AMZN', 'META'],
    }

    def __init__(self, strategy: CustomStrategy):
        """Initialize adapter with a CustomStrategy.

        Args:
            strategy: The CustomStrategy to backtest
        """
        self.strategy = strategy
        self.runner = CustomStrategyRunner(strategy.definition)
        self._price_cache: Dict[str, pd.DataFrame] = {}

    def get_symbols(self) -> List[str]:
        """Get list of symbols to trade based on strategy universe."""
        universe = self.strategy.universe
        if universe == 'custom':
            return self.strategy.custom_symbols or ['SPY']
        return self.UNIVERSE_SYMBOLS.get(universe, self.UNIVERSE_SYMBOLS['sp500'])

    def get_position_size_pct(self) -> float:
        """Get position size percentage from strategy definition."""
        sizing = self.strategy.definition.get('position_sizing', {})
        return float(sizing.get('value', 5.0))

    def get_max_positions(self) -> int:
        """Get maximum concurrent positions."""
        sizing = self.strategy.definition.get('position_sizing', {})
        return int(sizing.get('max_positions', 5))

    def get_stop_loss_pct(self) -> Optional[float]:
        """Get stop loss percentage if defined."""
        exit_conditions = self.strategy.definition.get('exit_conditions', [])
        for cond in exit_conditions:
            if cond.get('type') == 'stop_loss':
                return float(cond.get('value', 5.0))
        return None

    def get_take_profit_pct(self) -> Optional[float]:
        """Get take profit percentage if defined."""
        exit_conditions = self.strategy.definition.get('exit_conditions', [])
        for cond in exit_conditions:
            if cond.get('type') == 'take_profit':
                return float(cond.get('value', 15.0))
        return None

    async def fetch_price_data(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> Dict[str, pd.DataFrame]:
        """Fetch historical price data for symbols.

        Args:
            symbols: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data

        Returns:
            Dict mapping symbol to DataFrame with OHLCV data
        """
        data = {}

        if HAS_YFINANCE:
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(
                        start=start_date.isoformat(),
                        end=end_date.isoformat(),
                        interval="1d"
                    )
                    if not df.empty:
                        # Normalize column names
                        df.columns = [c.lower() for c in df.columns]
                        data[symbol] = df
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {symbol}: {e}")
        else:
            # Generate synthetic data for testing
            data = self._generate_synthetic_data(symbols, start_date, end_date)

        return data

    def _generate_synthetic_data(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> Dict[str, pd.DataFrame]:
        """Generate synthetic price data for testing."""
        import numpy as np

        data = {}
        dates = pd.date_range(start=start_date, end=end_date, freq='B')

        for symbol in symbols:
            # Random walk with drift
            np.random.seed(hash(symbol) % 2**32)
            returns = np.random.normal(0.0005, 0.02, len(dates))
            prices = 100 * np.cumprod(1 + returns)

            df = pd.DataFrame({
                'open': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, len(dates)),
            }, index=dates)
            data[symbol] = df

        return data

    async def run_backtest(
        self,
        config: CustomStrategyBacktestConfig,
        progress_callback: Optional[callable] = None,
        save_to_db: bool = True,
        user=None,
    ) -> BacktestResults:
        """Run backtest for the custom strategy.

        Args:
            config: Backtest configuration
            progress_callback: Optional callback for progress updates
            save_to_db: Whether to save results to database
            user: User who initiated the backtest

        Returns:
            BacktestResults with all metrics and trade logs
        """
        logger.info(f"Starting custom strategy backtest for {self.strategy.name}")

        symbols = self.get_symbols()

        # Update progress
        if progress_callback:
            progress_callback(5, "Fetching historical data...")

        # Fetch price data
        price_data = await self.fetch_price_data(
            symbols + [config.benchmark],
            config.start_date,
            config.end_date
        )

        if progress_callback:
            progress_callback(20, "Evaluating entry conditions...")

        # Run simulation
        trades = []
        daily_snapshots = []
        cash = config.initial_capital
        positions: Dict[str, Dict] = {}  # symbol -> {quantity, entry_price, entry_date}
        trade_counter = 0
        peak_equity = config.initial_capital
        max_drawdown = 0.0

        # Get benchmark data
        benchmark_df = price_data.get(config.benchmark)
        if benchmark_df is not None and not benchmark_df.empty:
            benchmark_start = Decimal(str(benchmark_df['close'].iloc[0]))
        else:
            benchmark_start = Decimal("100")

        # Get trading days from first symbol with data
        trading_days = []
        for symbol, df in price_data.items():
            if not df.empty and symbol != config.benchmark:
                trading_days = list(df.index)
                break

        if not trading_days:
            # Fallback to benchmark days
            if benchmark_df is not None and not benchmark_df.empty:
                trading_days = list(benchmark_df.index)

        total_days = len(trading_days)

        # Simulate trading
        for day_idx, current_date in enumerate(trading_days):
            if progress_callback and day_idx % 10 == 0:
                pct = 20 + int(60 * day_idx / max(total_days, 1))
                progress_callback(pct, f"Processing day {day_idx + 1}/{total_days}...")

            # Calculate current positions value
            positions_value = Decimal("0")
            for sym, pos in positions.items():
                if sym in price_data and current_date in price_data[sym].index:
                    current_price = Decimal(str(price_data[sym].loc[current_date, 'close']))
                    positions_value += current_price * pos['quantity']

            equity = cash + positions_value

            # Track drawdown
            if equity > peak_equity:
                peak_equity = equity
            drawdown = float((peak_equity - equity) / peak_equity * 100) if peak_equity > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

            # Get benchmark value
            if benchmark_df is not None and current_date in benchmark_df.index:
                benchmark_value = Decimal(str(benchmark_df.loc[current_date, 'close']))
            else:
                benchmark_value = benchmark_start

            # Check exit conditions for open positions
            positions_to_close = []
            for sym, pos in positions.items():
                if sym not in price_data or current_date not in price_data[sym].index:
                    continue

                df = price_data[sym]
                idx = df.index.get_loc(current_date)
                current_price = Decimal(str(df.loc[current_date, 'close']))
                pnl_pct = float((current_price - pos['entry_price']) / pos['entry_price'] * 100)

                should_exit = False
                exit_reason = None

                # Check stop loss
                stop_loss = self.get_stop_loss_pct()
                if stop_loss and pnl_pct <= -stop_loss:
                    should_exit = True
                    exit_reason = 'stopped_out'

                # Check take profit
                take_profit = self.get_take_profit_pct()
                if take_profit and pnl_pct >= take_profit:
                    should_exit = True
                    exit_reason = 'take_profit'

                # Check indicator-based exits
                if not should_exit and idx > 0:
                    try:
                        exit_met, _, _ = self.runner.check_exit_conditions(
                            df.iloc[:idx + 1],
                            float(pos['entry_price']),
                            pos['entry_date'],
                            idx=-1
                        )
                        if exit_met:
                            should_exit = True
                            exit_reason = 'indicator_exit'
                    except Exception as e:
                        logger.debug(f"Exit check error for {sym}: {e}")

                if should_exit:
                    positions_to_close.append((sym, current_price, exit_reason))

            # Close positions
            for sym, exit_price, exit_reason in positions_to_close:
                pos = positions.pop(sym)
                trade_counter += 1

                pnl = (exit_price - pos['entry_price']) * pos['quantity']
                pnl_pct = float((exit_price - pos['entry_price']) / pos['entry_price'] * 100)

                trade = Trade(
                    id=f"T{trade_counter:04d}",
                    symbol=sym,
                    direction=TradeDirection.LONG,
                    entry_date=pos['entry_date'].date() if hasattr(pos['entry_date'], 'date') else pos['entry_date'],
                    entry_price=pos['entry_price'],
                    quantity=pos['quantity'],
                    exit_date=current_date.date() if hasattr(current_date, 'date') else current_date,
                    exit_price=exit_price,
                    status=TradeStatus.STOPPED_OUT if exit_reason == 'stopped_out' else TradeStatus.TAKE_PROFIT if exit_reason == 'take_profit' else TradeStatus.CLOSED,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                )
                trades.append(trade)
                cash += exit_price * pos['quantity']

            # Check entry conditions for new positions
            max_pos = self.get_max_positions()
            position_size_pct = self.get_position_size_pct()

            if len(positions) < max_pos:
                for sym in symbols:
                    if sym in positions:
                        continue
                    if sym not in price_data or current_date not in price_data[sym].index:
                        continue

                    df = price_data[sym]
                    idx = df.index.get_loc(current_date)

                    # Need enough history for indicators
                    if idx < 50:
                        continue

                    try:
                        entry_met, _ = self.runner.check_entry_conditions(
                            df.iloc[:idx + 1],
                            idx=-1
                        )
                    except Exception as e:
                        logger.debug(f"Entry check error for {sym}: {e}")
                        continue

                    if entry_met:
                        # Calculate position size
                        current_price = Decimal(str(df.loc[current_date, 'close']))
                        position_value = equity * Decimal(str(position_size_pct / 100))
                        quantity = int(position_value / current_price)

                        if quantity > 0 and cash >= current_price * quantity:
                            positions[sym] = {
                                'quantity': quantity,
                                'entry_price': current_price,
                                'entry_date': current_date,
                            }
                            cash -= current_price * quantity

                            # Check max positions
                            if len(positions) >= max_pos:
                                break

            # Record daily snapshot
            positions_value = Decimal("0")
            for sym, pos in positions.items():
                if sym in price_data and current_date in price_data[sym].index:
                    current_price = Decimal(str(price_data[sym].loc[current_date, 'close']))
                    positions_value += current_price * pos['quantity']

            equity = cash + positions_value
            daily_return = float((equity - config.initial_capital) / config.initial_capital * 100)

            snapshot = DailySnapshot(
                date=current_date.date() if hasattr(current_date, 'date') else current_date,
                equity=equity,
                cash=cash,
                positions_value=positions_value,
                benchmark_value=benchmark_value,
                daily_pnl=equity - (daily_snapshots[-1].equity if daily_snapshots else config.initial_capital),
                daily_return_pct=daily_return,
                cumulative_return_pct=daily_return,
                drawdown_pct=drawdown,
            )
            daily_snapshots.append(snapshot)

        if progress_callback:
            progress_callback(85, "Calculating metrics...")

        # Close any remaining positions at end
        for sym, pos in list(positions.items()):
            if sym in price_data and len(price_data[sym]) > 0:
                exit_price = Decimal(str(price_data[sym]['close'].iloc[-1]))
                trade_counter += 1

                pnl = (exit_price - pos['entry_price']) * pos['quantity']
                pnl_pct = float((exit_price - pos['entry_price']) / pos['entry_price'] * 100)

                trade = Trade(
                    id=f"T{trade_counter:04d}",
                    symbol=sym,
                    direction=TradeDirection.LONG,
                    entry_date=pos['entry_date'].date() if hasattr(pos['entry_date'], 'date') else pos['entry_date'],
                    entry_price=pos['entry_price'],
                    quantity=pos['quantity'],
                    exit_date=trading_days[-1].date() if trading_days and hasattr(trading_days[-1], 'date') else date.today(),
                    exit_price=exit_price,
                    status=TradeStatus.CLOSED,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                )
                trades.append(trade)
                cash += exit_price * pos['quantity']

        # Calculate final metrics
        final_equity = cash
        total_return_pct = float((final_equity - config.initial_capital) / config.initial_capital * 100)

        # Benchmark return
        if benchmark_df is not None and not benchmark_df.empty:
            benchmark_start_val = float(benchmark_df['close'].iloc[0])
            benchmark_end_val = float(benchmark_df['close'].iloc[-1])
            benchmark_return_pct = (benchmark_end_val - benchmark_start_val) / benchmark_start_val * 100
        else:
            benchmark_return_pct = 0.0

        # Trade statistics
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0

        avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else Decimal("0")
        avg_loss = abs(sum(t.pnl for t in losing_trades) / len(losing_trades)) if losing_trades else Decimal("0")

        total_gains = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = float(total_gains / total_losses) if total_losses > 0 else 2.0

        avg_hold_days = sum(t.hold_days for t in trades) / len(trades) if trades else 0

        # Calculate Sharpe and Sortino
        if daily_snapshots:
            returns = [
                (daily_snapshots[i].equity - daily_snapshots[i-1].equity) / daily_snapshots[i-1].equity
                for i in range(1, len(daily_snapshots))
            ]
            if returns:
                import statistics
                avg_return = statistics.mean(returns)
                std_return = statistics.stdev(returns) if len(returns) > 1 else 0.001
                risk_free_daily = 0.05 / 252  # 5% annual risk-free rate
                sharpe_ratio = (avg_return - risk_free_daily) / std_return * (252 ** 0.5) if std_return > 0 else 0

                downside_returns = [r for r in returns if r < 0]
                downside_std = statistics.stdev(downside_returns) if len(downside_returns) > 1 else std_return * 0.7
                sortino_ratio = (avg_return - risk_free_daily) / downside_std * (252 ** 0.5) if downside_std > 0 else 0
            else:
                sharpe_ratio = 0.0
                sortino_ratio = 0.0
        else:
            sharpe_ratio = 0.0
            sortino_ratio = 0.0

        # Monthly returns
        monthly_returns = {}
        if daily_snapshots:
            for snap in daily_snapshots:
                month_key = snap.date.strftime('%Y-%m') if hasattr(snap.date, 'strftime') else str(snap.date)[:7]
                if month_key not in monthly_returns:
                    monthly_returns[month_key] = 0.0
                monthly_returns[month_key] = snap.cumulative_return_pct

        # Build results
        bt_config = BacktestConfig(
            strategy_name=f"custom-{self.strategy.id}",
            start_date=config.start_date,
            end_date=config.end_date,
            initial_capital=config.initial_capital,
            benchmark=config.benchmark,
            position_size_pct=self.get_position_size_pct(),
            stop_loss_pct=self.get_stop_loss_pct() or 5.0,
            take_profit_pct=self.get_take_profit_pct() or 15.0,
            max_positions=self.get_max_positions(),
        )

        results = BacktestResults(
            config=bt_config,
            trades=trades,
            daily_snapshots=daily_snapshots,
            total_return_pct=total_return_pct,
            benchmark_return_pct=benchmark_return_pct,
            alpha=total_return_pct - benchmark_return_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown_pct=max_drawdown,
            win_rate=win_rate,
            profit_factor=min(profit_factor, 10.0),  # Cap at 10
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_hold_days=avg_hold_days,
            monthly_returns=monthly_returns,
            final_equity=final_equity,
            total_pnl=final_equity - config.initial_capital,
        )

        if progress_callback:
            progress_callback(95, "Saving results...")

        # Optionally save to database
        if save_to_db and user:
            await self._save_to_database(results, user)

        if progress_callback:
            progress_callback(100, "Complete")

        return results

    async def _save_to_database(self, results: BacktestResults, user) -> BacktestRun:
        """Save backtest results to database.

        Args:
            results: BacktestResults to save
            user: User who ran the backtest

        Returns:
            Created BacktestRun instance
        """
        from django.utils import timezone

        # Create BacktestRun
        run = BacktestRun.objects.create(
            run_id=str(uuid.uuid4()),
            user=user,
            name=f"{self.strategy.name} Backtest",
            strategy_name=f"custom-{self.strategy.id}",
            custom_strategy=self.strategy,
            start_date=results.config.start_date,
            end_date=results.config.end_date,
            initial_capital=results.config.initial_capital,
            benchmark=results.config.benchmark,
            parameters={
                'position_size_pct': results.config.position_size_pct,
                'stop_loss_pct': results.config.stop_loss_pct,
                'take_profit_pct': results.config.take_profit_pct,
                'max_positions': results.config.max_positions,
            },
            status='completed',
            progress=100,
            total_return_pct=results.total_return_pct,
            benchmark_return_pct=results.benchmark_return_pct,
            alpha=results.alpha,
            sharpe_ratio=results.sharpe_ratio,
            sortino_ratio=results.sortino_ratio,
            max_drawdown_pct=results.max_drawdown_pct,
            win_rate=results.win_rate,
            profit_factor=results.profit_factor,
            total_trades=results.total_trades,
            winning_trades=results.winning_trades,
            losing_trades=results.losing_trades,
            avg_win=results.avg_win,
            avg_loss=results.avg_loss,
            avg_hold_days=results.avg_hold_days,
            final_equity=results.final_equity,
            total_pnl=results.total_pnl,
            monthly_returns=results.monthly_returns,
            equity_curve=[
                {
                    'date': s.date.isoformat() if hasattr(s.date, 'isoformat') else str(s.date),
                    'equity': float(s.equity),
                    'benchmark': float(s.benchmark_value),
                }
                for s in results.daily_snapshots[::5]
            ],
            drawdown_curve=[
                {
                    'date': s.date.isoformat() if hasattr(s.date, 'isoformat') else str(s.date),
                    'drawdown_pct': s.drawdown_pct,
                }
                for s in results.daily_snapshots[::5]
            ],
            completed_at=timezone.now(),
        )

        # Create BacktestTrade records
        for trade in results.trades:
            BacktestTrade.objects.create(
                backtest_run=run,
                trade_id=trade.id,
                symbol=trade.symbol,
                direction=trade.direction.value,
                entry_date=trade.entry_date,
                entry_price=trade.entry_price,
                quantity=trade.quantity,
                exit_date=trade.exit_date,
                exit_price=trade.exit_price,
                status=trade.status.value,
                pnl=trade.pnl,
                pnl_pct=trade.pnl_pct,
            )

        # Update strategy with latest backtest results
        self.strategy.backtest_results = results.to_dict()
        self.strategy.last_backtest_at = timezone.now()
        self.strategy.save(update_fields=['backtest_results', 'last_backtest_at'])

        return run
