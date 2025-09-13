"""Production Index Baseline Implementation
Performance tracking and benchmarking against index funds.
"""

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from .data_providers import UnifiedDataProvider
from .production_config import ProductionConfig
from .production_logging import ErrorHandler, MetricsCollector, ProductionLogger
from .production_models import Trade
from .trading_interface import TradingInterface


class BenchmarkType(Enum):
    """Benchmark types."""

    SPY = "spy"  # S & P 500
    VTI = "vti"  # Total Stock Market
    QQQ = "qqq"  # NASDAQ 100
    IWM = "iwm"  # Russell 2000


@dataclass
class BenchmarkData:
    """Benchmark performance data."""

    ticker: str
    benchmark_type: BenchmarkType
    current_price: float
    daily_return: float
    weekly_return: float
    monthly_return: float
    ytd_return: float
    annual_return: float

    # Risk metrics
    volatility: float
    sharpe_ratio: float
    max_drawdown: float

    # Metadata
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class StrategyPerformance:
    """Strategy performance tracking."""

    strategy_name: str
    total_return: float
    daily_return: float
    weekly_return: float
    monthly_return: float
    ytd_return: float
    annual_return: float

    # Risk metrics
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float

    # Trade metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    profit_factor: float

    # Metadata
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceComparison:
    """Performance comparison between strategy and benchmarks."""

    strategy_name: str
    benchmark_ticker: str

    # Returns comparison
    strategy_return: float
    benchmark_return: float
    alpha: float  # Excess return
    beta: float  # Market sensitivity

    # Risk comparison
    strategy_volatility: float
    benchmark_volatility: float
    information_ratio: float

    # Risk - adjusted returns
    strategy_sharpe: float
    benchmark_sharpe: float

    # Metadata
    comparison_date: datetime = field(default_factory=datetime.now)


class PerformanceCalculator:
    """Performance calculation utilities."""

    def __init__(self, logger: ProductionLogger):
        self.logger = logger

    def calculate_returns(self, prices: list[float]) -> dict[str, float]:
        """Calculate various return metrics."""
        if len(prices) < 2:
            return {
                "daily_return": 0.0,
                "weekly_return": 0.0,
                "monthly_return": 0.0,
                "ytd_return": 0.0,
                "annual_return": 0.0,
            }

        # Daily return
        daily_return = (
            (prices[-1] - prices[-2]) / prices[-2] if len(prices) >= 2 else 0.0
        )

        # Weekly return (5 trading days)
        weekly_return = (
            (prices[-1] - prices[-6]) / prices[-6] if len(prices) >= 6 else 0.0
        )

        # Monthly return (20 trading days)
        monthly_return = (
            (prices[-1] - prices[-21]) / prices[-21] if len(prices) >= 21 else 0.0
        )

        # YTD return (simplified)
        ytd_return = (prices[-1] - prices[0]) / prices[0] if len(prices) > 0 else 0.0

        # Annual return (simplified)
        annual_return = ytd_return * (252 / len(prices)) if len(prices) > 0 else 0.0

        return {
            "daily_return": daily_return,
            "weekly_return": weekly_return,
            "monthly_return": monthly_return,
            "ytd_return": ytd_return,
            "annual_return": annual_return,
        }

    def calculate_volatility(self, returns: list[float]) -> float:
        """Calculate volatility (standard deviation of returns)."""
        if len(returns) < 2:
            return 0.0

        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        return math.sqrt(variance)

    def calculate_sharpe_ratio(
        self, returns: list[float], risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0

        mean_return = sum(returns) / len(returns)
        volatility = self.calculate_volatility(returns)

        if volatility == 0:
            return 0.0

        return (mean_return - risk_free_rate) / volatility

    def calculate_max_drawdown(self, prices: list[float]) -> float:
        """Calculate maximum drawdown."""
        if len(prices) < 2:
            return 0.0

        peak = prices[0]
        max_dd = 0.0

        for price in prices:
            if price > peak:
                peak = price
            else:
                drawdown = (peak - price) / peak
                max_dd = max(max_dd, drawdown)

        return max_dd

    def calculate_alpha_beta(
        self, strategy_returns: list[float], benchmark_returns: list[float]
    ) -> tuple[float, float]:
        """Calculate alpha and beta."""
        if len(strategy_returns) != len(benchmark_returns) or len(strategy_returns) < 2:
            return 0.0, 1.0

        # Calculate covariance and variance
        strategy_mean = sum(strategy_returns) / len(strategy_returns)
        benchmark_mean = sum(benchmark_returns) / len(benchmark_returns)

        covariance = sum(
            (s - strategy_mean) * (b - benchmark_mean)
            for s, b in zip(strategy_returns, benchmark_returns, strict=False)
        ) / len(strategy_returns)

        benchmark_variance = sum(
            (b - benchmark_mean) ** 2 for b in benchmark_returns
        ) / len(benchmark_returns)

        if benchmark_variance == 0:
            return 0.0, 1.0

        beta = covariance / benchmark_variance
        alpha = strategy_mean - beta * benchmark_mean

        return alpha, beta


class ProductionIndexBaseline:
    """Production Index Baseline Implementation."""

    def __init__(
        self,
        trading_interface: TradingInterface,
        data_provider: UnifiedDataProvider,
        config: ProductionConfig,
        logger: ProductionLogger,
    ):
        self.trading = trading_interface
        self.data = data_provider
        self.config = config
        self.logger = logger
        self.error_handler = ErrorHandler(logger)
        self.metrics = MetricsCollector(logger)
        self.calculator = PerformanceCalculator(logger)

        # Benchmark tracking
        self.benchmarks: dict[str, BenchmarkData] = {}
        self.strategy_performance: dict[str, StrategyPerformance] = {}
        self.comparisons: list[PerformanceComparison] = []

        # Performance tracking
        self.price_history: dict[str, list[float]] = {}
        self.return_history: dict[str, list[float]] = {}

        # Strategy state
        self.last_update_time: datetime | None = None
        self.update_interval = timedelta(hours=1)

        self.logger.info("Index Baseline Strategy initialized")

    async def initialize_benchmarks(self):
        """Initialize benchmark tracking."""
        self.logger.info("Initializing benchmark tracking")

        benchmark_tickers = ["SPY", "VTI", "QQQ", "IWM"]

        for ticker in benchmark_tickers:
            try:
                await self._initialize_benchmark(ticker)
            except Exception as e:
                self.error_handler.handle_error(
                    e, {"ticker": ticker, "operation": "initialize_benchmark"}
                )

    async def _initialize_benchmark(self, ticker: str):
        """Initialize individual benchmark."""
        try:
            # Get current market data
            market_data = await self.data.get_market_data(ticker)

            if market_data.price <= 0:
                self.logger.warning(f"No price data for {ticker}")
                return

            # Initialize benchmark data
            benchmark_type = (
                BenchmarkType.SPY
                if ticker == "SPY"
                else BenchmarkType.VTI
                if ticker == "VTI"
                else BenchmarkType.QQQ
                if ticker == "QQQ"
                else BenchmarkType.IWM
            )

            benchmark = BenchmarkData(
                ticker=ticker,
                benchmark_type=benchmark_type,
                current_price=market_data.price,
                daily_return=market_data.change_percent,
                weekly_return=0.0,  # Would calculate from historical data
                monthly_return=0.0,
                ytd_return=0.0,
                annual_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
            )

            self.benchmarks[ticker] = benchmark

            # Initialize price history
            self.price_history[ticker] = [market_data.price]
            self.return_history[ticker] = [market_data.change_percent]

            self.logger.info(
                f"Initialized benchmark: {ticker} @ ${market_data.price: .2f}"
            )

        except Exception as e:
            self.error_handler.handle_error(
                e, {"ticker": ticker, "operation": "initialize_benchmark"}
            )

    async def update_benchmarks(self):
        """Update benchmark performance data."""
        self.logger.info("Updating benchmark performance")

        for ticker, benchmark in self.benchmarks.items():
            try:
                await self._update_benchmark(ticker, benchmark)
            except Exception as e:
                self.error_handler.handle_error(
                    e, {"ticker": ticker, "operation": "update_benchmark"}
                )

    async def _update_benchmark(self, ticker: str, benchmark: BenchmarkData):
        """Update individual benchmark."""
        try:
            # Get current market data
            market_data = await self.data.get_market_data(ticker)

            if market_data.price <= 0:
                return

            # Update price history
            self.price_history[ticker].append(market_data.price)

            # Keep only last 252 days (1 year)
            if len(self.price_history[ticker]) > 252:
                self.price_history[ticker] = self.price_history[ticker][-252:]

            # Calculate returns
            returns = self.calculator.calculate_returns(self.price_history[ticker])

            # Update benchmark data
            benchmark.current_price = market_data.price
            benchmark.daily_return = market_data.change_percent
            benchmark.weekly_return = returns["weekly_return"]
            benchmark.monthly_return = returns["monthly_return"]
            benchmark.ytd_return = returns["ytd_return"]
            benchmark.annual_return = returns["annual_return"]

            # Calculate risk metrics
            if len(self.price_history[ticker]) > 1:
                price_returns = [
                    (self.price_history[ticker][i] - self.price_history[ticker][i - 1])
                    / self.price_history[ticker][i - 1]
                    for i in range(1, len(self.price_history[ticker]))
                ]

                benchmark.volatility = self.calculator.calculate_volatility(
                    price_returns
                )
                benchmark.sharpe_ratio = self.calculator.calculate_sharpe_ratio(
                    price_returns
                )
                benchmark.max_drawdown = self.calculator.calculate_max_drawdown(
                    self.price_history[ticker]
                )

            benchmark.last_update = datetime.now()

            self.logger.info(
                f"Updated benchmark: {ticker}",
                price=benchmark.current_price,
                daily_return=benchmark.daily_return,
                volatility=benchmark.volatility,
            )

        except Exception as e:
            self.error_handler.handle_error(
                e, {"ticker": ticker, "operation": "update_benchmark"}
            )

    async def track_strategy_performance(self, strategy_name: str, trades: list[Trade]):
        """Track strategy performance."""
        try:
            self.logger.info(f"Tracking performance for {strategy_name}")

            # Calculate strategy metrics
            total_return = self._calculate_strategy_return(trades)
            win_rate = self._calculate_win_rate(trades)
            avg_win, avg_loss = self._calculate_avg_win_loss(trades)
            profit_factor = self._calculate_profit_factor(trades)

            # Calculate risk metrics (simplified)
            volatility = 0.15  # Default volatility
            sharpe_ratio = total_return / volatility if volatility > 0 else 0.0
            max_drawdown = 0.10  # Default max drawdown

            performance = StrategyPerformance(
                strategy_name=strategy_name,
                total_return=total_return,
                daily_return=total_return / 252,  # Simplified daily return
                weekly_return=total_return / 52,  # Simplified weekly return
                monthly_return=total_return / 12,  # Simplified monthly return
                ytd_return=total_return,
                annual_return=total_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                total_trades=len(trades),
                winning_trades=sum(
                    1 for trade in trades if self._is_winning_trade(trade)
                ),
                losing_trades=sum(
                    1 for trade in trades if not self._is_winning_trade(trade)
                ),
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
            )

            self.strategy_performance[strategy_name] = performance

            self.logger.info(
                f"Strategy performance tracked: {strategy_name}",
                total_return=total_return,
                win_rate=win_rate,
                total_trades=len(trades),
            )

        except Exception as e:
            self.error_handler.handle_error(
                e, {"strategy": strategy_name, "operation": "track_performance"}
            )

    def _calculate_strategy_return(self, trades: list[Trade]) -> float:
        """Calculate total strategy return."""
        if not trades:
            return 0.0

        total_pnl = sum(self._calculate_trade_pnl(trade) for trade in trades)
        total_invested = sum(
            trade.filled_price * trade.filled_quantity * 100 for trade in trades
        )

        return total_pnl / total_invested if total_invested > 0 else 0.0

    def _calculate_trade_pnl(self, trade: Trade) -> float:
        """Calculate individual trade P & L."""
        # Simplified P & L calculation
        # In production, would track entry / exit prices
        return trade.commission * -1  # Simplified

    def _calculate_win_rate(self, trades: list[Trade]) -> float:
        """Calculate win rate."""
        if not trades:
            return 0.0

        winning_trades = sum(1 for trade in trades if self._is_winning_trade(trade))
        return winning_trades / len(trades)

    def _is_winning_trade(self, trade: Trade) -> bool:
        """Determine if trade is winning."""
        # Simplified - in production would check actual P & L
        return trade.commission > 0

    def _calculate_avg_win_loss(self, trades: list[Trade]) -> tuple[float, float]:
        """Calculate average win and loss."""
        if not trades:
            return 0.0, 0.0

        wins = [
            self._calculate_trade_pnl(trade)
            for trade in trades
            if self._is_winning_trade(trade)
        ]
        losses = [
            self._calculate_trade_pnl(trade)
            for trade in trades
            if not self._is_winning_trade(trade)
        ]

        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0

        return avg_win, avg_loss

    def _calculate_profit_factor(self, trades: list[Trade]) -> float:
        """Calculate profit factor."""
        if not trades:
            return 0.0

        wins = [
            self._calculate_trade_pnl(trade)
            for trade in trades
            if self._is_winning_trade(trade)
        ]
        losses = [
            abs(self._calculate_trade_pnl(trade))
            for trade in trades
            if not self._is_winning_trade(trade)
        ]

        total_wins = sum(wins)
        total_losses = sum(losses)

        return total_wins / total_losses if total_losses > 0 else 0.0

    async def generate_performance_comparison(
        self, strategy_name: str
    ) -> list[PerformanceComparison]:
        """Generate performance comparison against benchmarks."""
        comparisons = []

        if strategy_name not in self.strategy_performance:
            return comparisons

        strategy_perf = self.strategy_performance[strategy_name]

        for ticker, benchmark in self.benchmarks.items():
            try:
                # Calculate alpha and beta (simplified)
                alpha = strategy_perf.total_return - benchmark.annual_return
                beta = 1.0  # Simplified beta calculation

                # Calculate information ratio
                information_ratio = (
                    alpha / strategy_perf.volatility
                    if strategy_perf.volatility > 0
                    else 0.0
                )

                comparison = PerformanceComparison(
                    strategy_name=strategy_name,
                    benchmark_ticker=ticker,
                    strategy_return=strategy_perf.total_return,
                    benchmark_return=benchmark.annual_return,
                    alpha=alpha,
                    beta=beta,
                    strategy_volatility=strategy_perf.volatility,
                    benchmark_volatility=benchmark.volatility,
                    information_ratio=information_ratio,
                    strategy_sharpe=strategy_perf.sharpe_ratio,
                    benchmark_sharpe=benchmark.sharpe_ratio,
                )

                comparisons.append(comparison)

                self.logger.info(
                    f"Performance comparison: {strategy_name} vs {ticker}",
                    alpha=alpha,
                    strategy_return=strategy_perf.total_return,
                    benchmark_return=benchmark.annual_return,
                )

            except Exception as e:
                self.error_handler.handle_error(
                    e, {"strategy": strategy_name, "benchmark": ticker}
                )

        return comparisons

    async def get_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "report_date": datetime.now().isoformat(),
            "benchmarks": {},
            "strategies": {},
            "comparisons": [],
        }

        # Add benchmark data
        for ticker, benchmark in self.benchmarks.items():
            report["benchmarks"][ticker] = {
                "current_price": benchmark.current_price,
                "daily_return": benchmark.daily_return,
                "weekly_return": benchmark.weekly_return,
                "monthly_return": benchmark.monthly_return,
                "ytd_return": benchmark.ytd_return,
                "annual_return": benchmark.annual_return,
                "volatility": benchmark.volatility,
                "sharpe_ratio": benchmark.sharpe_ratio,
                "max_drawdown": benchmark.max_drawdown,
            }

        # Add strategy data
        for strategy_name, performance in self.strategy_performance.items():
            report["strategies"][strategy_name] = {
                "total_return": performance.total_return,
                "daily_return": performance.daily_return,
                "weekly_return": performance.weekly_return,
                "monthly_return": performance.monthly_return,
                "ytd_return": performance.ytd_return,
                "annual_return": performance.annual_return,
                "volatility": performance.volatility,
                "sharpe_ratio": performance.sharpe_ratio,
                "max_drawdown": performance.max_drawdown,
                "win_rate": performance.win_rate,
                "total_trades": performance.total_trades,
                "winning_trades": performance.winning_trades,
                "losing_trades": performance.losing_trades,
                "avg_win": performance.avg_win,
                "avg_loss": performance.avg_loss,
                "profit_factor": performance.profit_factor,
            }

        # Add comparisons
        for strategy_name in self.strategy_performance:
            comparisons = await self.generate_performance_comparison(strategy_name)
            report["comparisons"].extend(
                [
                    {
                        "strategy_name": comp.strategy_name,
                        "benchmark_ticker": comp.benchmark_ticker,
                        "strategy_return": comp.strategy_return,
                        "benchmark_return": comp.benchmark_return,
                        "alpha": comp.alpha,
                        "beta": comp.beta,
                        "information_ratio": comp.information_ratio,
                        "strategy_sharpe": comp.strategy_sharpe,
                        "benchmark_sharpe": comp.benchmark_sharpe,
                    }
                    for comp in comparisons
                ]
            )

        return report

    async def run_baseline_tracking(self):
        """Run baseline tracking main loop."""
        self.logger.info("Starting Index Baseline Tracking")

        # Initialize benchmarks
        await self.initialize_benchmarks()

        while True:
            try:
                # Update benchmarks
                await self.update_benchmarks()

                # Generate performance report
                await self.get_performance_report()

                # Log summary
                self.logger.info(
                    "Baseline tracking summary",
                    benchmarks_tracked=len(self.benchmarks),
                    strategies_tracked=len(self.strategy_performance),
                )

                # Record metrics
                self.metrics.record_metric("benchmarks_tracked", len(self.benchmarks))
                self.metrics.record_metric(
                    "strategies_tracked", len(self.strategy_performance)
                )

                # Wait for next update
                await asyncio.sleep(self.update_interval.total_seconds())

            except Exception as e:
                self.error_handler.handle_error(
                    e, {"operation": "run_baseline_tracking"}
                )
                await asyncio.sleep(300)  # Wait 5 minutes on error


# Factory function for easy initialization
def create_index_baseline_strategy(
    trading_interface: TradingInterface,
    data_provider: UnifiedDataProvider,
    config: ProductionConfig,
    logger: ProductionLogger,
) -> ProductionIndexBaseline:
    """Create index baseline strategy instance."""
    return ProductionIndexBaseline(trading_interface, data_provider, config, logger)
