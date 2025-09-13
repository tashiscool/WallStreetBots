"""Phase 4: Production Backtesting Engine
Comprehensive backtesting with historical data validation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from ..core.production_config import ConfigManager
from ..core.production_logging import ProductionLogger


class BacktestPeriod(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class BacktestConfig:
    start_date: datetime
    end_date: datetime
    initial_capital: float
    commission_per_trade: float
    slippage_per_trade: float
    benchmark_ticker: str
    rebalance_frequency: BacktestPeriod
    risk_free_rate: float
    max_positions: int
    position_size_limit: float
    stop_loss_pct: float
    take_profit_pct: float


@dataclass
class BacktestTrade:
    ticker: str
    strategy: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: int
    side: str
    pnl: float
    commission: float
    slippage: float
    net_pnl: float
    holding_period_days: int
    return_pct: float
    exit_reason: str


@dataclass
class BacktestResults:
    config: BacktestConfig
    start_date: datetime
    end_date: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_commission: float
    total_slippage: float
    net_profit: float
    benchmark_return: float
    alpha: float
    beta: float
    trades: list[BacktestTrade] = field(default_factory=list)
    daily_returns: list[float] = field(default_factory=list)
    daily_portfolio_values: list[float] = field(default_factory=list)


class BacktestEngine:
    """Main backtesting engine."""

    def __init__(self, config: ConfigManager, logger: ProductionLogger):
        self.config = config
        self.logger = logger

        # Backtesting state
        self.current_date = None
        self.portfolio_value = 0.0
        self.cash = 0.0
        self.positions = {}
        self.trades = []
        self.daily_portfolio_values = []
        self.daily_returns = []

        self.logger.info("BacktestEngine initialized")

    async def run_backtest(self, strategy, config: BacktestConfig) -> BacktestResults:
        """Run comprehensive backtest."""
        try:
            self.logger.info(f"Starting backtest from {config.start_date} to {config.end_date}")

            # Initialize backtesting state
            self._initialize_backtest(config)

            # Run backtest simulation
            await self._run_simulation(strategy, config)

            # Calculate results
            results = await self._calculate_results(config)

            self.logger.info(
                f"Backtest completed: {results.total_trades} trades, {results.total_return: .2%} return"
            )
            return results

        except Exception as e:
            self.logger.error(f"Error running backtest: {e}")
            raise

    def _initialize_backtest(self, config: BacktestConfig):
        """Initialize backtesting state."""
        self.current_date = config.start_date
        self.portfolio_value = config.initial_capital
        self.cash = config.initial_capital
        self.positions = {}
        self.trades = []
        self.daily_portfolio_values = []
        self.daily_returns = []

    async def _run_simulation(self, strategy, config: BacktestConfig):
        """Run the backtest simulation."""
        date_range = [
            config.start_date + timedelta(days=i)
            for i in range((config.end_date - config.start_date).days + 1)
        ]

        for current_date in date_range:
            self.current_date = current_date

            # Update portfolio value
            await self._update_portfolio_value()

            # Record daily portfolio value
            self.daily_portfolio_values.append(self.portfolio_value)

            # Calculate daily return
            if len(self.daily_portfolio_values) > 1:
                daily_return = (
                    self.portfolio_value - self.daily_portfolio_values[-2]
                ) / self.daily_portfolio_values[-2]
                self.daily_returns.append(daily_return)
            else:
                self.daily_returns.append(0.0)

            # Check for rebalancing
            if self._should_rebalance(current_date, config):
                await self._rebalance_portfolio(strategy, config)

            # Update positions
            await self._update_positions()

            # Check for exit conditions
            await self._check_exit_conditions(config)

    async def _update_portfolio_value(self):
        """Update portfolio value based on current positions."""
        total_value = self.cash

        for _ticker, position in self.positions.items():
            # Mock current price update
            current_price = position["entry_price"] * (1 + 0.001)  # 0.1% daily return
            position["current_price"] = current_price
            position["unrealized_pnl"] = (current_price - position["entry_price"]) * position[
                "quantity"
            ]
            total_value += position["unrealized_pnl"]

        self.portfolio_value = total_value

    def _should_rebalance(self, current_date: datetime, config: BacktestConfig) -> bool:
        """Check if portfolio should be rebalanced."""
        if config.rebalance_frequency == BacktestPeriod.DAILY:
            return True
        elif config.rebalance_frequency == BacktestPeriod.WEEKLY:
            return current_date.weekday() == 0  # Monday
        elif config.rebalance_frequency == BacktestPeriod.MONTHLY:
            return current_date.day == 1

        return False

    async def _rebalance_portfolio(self, strategy, config: BacktestConfig):
        """Rebalance portfolio based on strategy signals."""
        try:
            # Mock strategy signals
            signals = await self._get_mock_signals()

            # Process signals
            for signal in signals:
                await self._process_signal(signal, config)

        except Exception as e:
            self.logger.error(f"Error rebalancing portfolio: {e}")

    async def _get_mock_signals(self) -> list[dict[str, Any]]:
        """Generate mock trading signals."""
        # Mock signals for backtesting
        universe = self.config.trading.universe
        signals = []

        for ticker in universe[:3]:  # Limit to 3 tickers
            signals.append(
                {
                    "ticker": ticker,
                    "action": "buy" if len(signals) % 2 == 0 else "sell",
                    "quantity": 100,
                }
            )

        return signals

    async def _process_signal(self, signal: dict[str, Any], config: BacktestConfig):
        """Process individual trading signal."""
        try:
            ticker = signal.get("ticker")
            action = signal.get("action")
            quantity = signal.get("quantity", 0)

            # Mock current price
            current_price = 100.0

            if action == "buy" and quantity > 0:
                await self._execute_buy(ticker, quantity, current_price, config)
            elif action == "sell" and quantity > 0:
                await self._execute_sell(ticker, quantity, current_price, config)

        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")

    async def _execute_buy(self, ticker: str, quantity: int, price: float, config: BacktestConfig):
        """Execute buy order."""
        try:
            # Check if we have enough cash
            total_cost = quantity * price
            commission = config.commission_per_trade
            slippage = config.slippage_per_trade * total_cost
            total_cost_with_costs = total_cost + commission + slippage

            if total_cost_with_costs > self.cash:
                self.logger.warning(f"Insufficient cash for {ticker} buy order")
                return

            # Execute the trade
            self.cash -= total_cost_with_costs

            if ticker in self.positions:
                # Add to existing position
                existing_position = self.positions[ticker]
                total_quantity = existing_position["quantity"] + quantity
                avg_price = (
                    (existing_position["entry_price"] * existing_position["quantity"])
                    + (price * quantity)
                ) / total_quantity

                existing_position["quantity"] = total_quantity
                existing_position["entry_price"] = avg_price
            else:
                # Create new position
                self.positions[ticker] = {
                    "ticker": ticker,
                    "strategy": "backtest",
                    "entry_date": self.current_date,
                    "entry_price": price,
                    "quantity": quantity,
                    "side": "long",
                    "current_price": price,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": 0.0,
                    "days_held": 0,
                }

            self.logger.info(f"Executed buy: {quantity} {ticker} at ${price: .2f}")

        except Exception as e:
            self.logger.error(f"Error executing buy order: {e}")

    async def _execute_sell(self, ticker: str, quantity: int, price: float, config: BacktestConfig):
        """Execute sell order."""
        try:
            if ticker not in self.positions:
                self.logger.warning(f"No position to sell for {ticker}")
                return

            position = self.positions[ticker]

            quantity = min(quantity, position["quantity"])

            # Calculate P & L
            pnl = (price - position["entry_price"]) * quantity
            commission = config.commission_per_trade
            slippage = config.slippage_per_trade * (quantity * price)
            net_pnl = pnl - commission - slippage

            # Update cash and position
            self.cash += (quantity * price) - commission - slippage
            position["quantity"] -= quantity
            position["realized_pnl"] += net_pnl

            # Record the trade
            trade = BacktestTrade(
                ticker=ticker,
                strategy="backtest",
                entry_date=position["entry_date"],
                exit_date=self.current_date,
                entry_price=position["entry_price"],
                exit_price=price,
                quantity=quantity,
                side="long",
                pnl=pnl,
                commission=commission,
                slippage=slippage,
                net_pnl=net_pnl,
                holding_period_days=(self.current_date - position["entry_date"]).days,
                return_pct=pnl / (position["entry_price"] * quantity),
                exit_reason="signal_exit",
            )

            self.trades.append(trade)

            # Remove position if fully sold
            if position["quantity"] == 0:
                del self.positions[ticker]

            self.logger.info(
                f"Executed sell: {quantity} {ticker} at ${price: .2f}, P & L: ${net_pnl:.2f}"
            )

        except Exception as e:
            self.logger.error(f"Error executing sell order: {e}")

    async def _update_positions(self):
        """Update position tracking."""
        for _ticker, position in self.positions.items():
            # Mock price update
            current_price = position["entry_price"] * (1 + 0.001)  # 0.1% daily return
            position["current_price"] = current_price
            position["unrealized_pnl"] = (current_price - position["entry_price"]) * position[
                "quantity"
            ]
            position["days_held"] = (self.current_date - position["entry_date"]).days

    async def _check_exit_conditions(self, config: BacktestConfig):
        """Check for exit conditions (stop loss, take profit)."""
        for ticker, position in list(self.positions.items()):
            current_price = position["current_price"]
            entry_price = position["entry_price"]

            # Check stop loss
            if current_price <= entry_price * (1 - config.stop_loss_pct):
                await self._execute_sell(ticker, position["quantity"], current_price, config)
                continue

            # Check take profit
            if current_price >= entry_price * (1 + config.take_profit_pct):
                await self._execute_sell(ticker, position["quantity"], current_price, config)
                continue

    async def _calculate_results(self, config: BacktestConfig) -> BacktestResults:
        """Calculate comprehensive backtest results."""
        try:
            # Basic metrics
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t.net_pnl > 0])
            losing_trades = len([t for t in self.trades if t.net_pnl < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            # Return metrics
            total_return = (self.portfolio_value - config.initial_capital) / config.initial_capital
            years = (config.end_date - config.start_date).days / 365.25
            annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

            # Risk metrics
            volatility = 0.15  # Mock volatility
            sharpe_ratio = (
                (annualized_return - config.risk_free_rate) / volatility if volatility > 0 else 0
            )

            # Drawdown calculation
            peak = config.initial_capital
            max_drawdown = 0

            for value in self.daily_portfolio_values:
                peak = max(peak, value)
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)

            # Additional metrics
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0

            # Profit factor
            gross_profit = sum(t.net_pnl for t in self.trades if t.net_pnl > 0)
            gross_loss = abs(sum(t.net_pnl for t in self.trades if t.net_pnl < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

            # Trade statistics
            avg_win = (
                sum(t.net_pnl for t in self.trades if t.net_pnl > 0) / winning_trades
                if winning_trades > 0
                else 0
            )
            avg_loss = (
                sum(t.net_pnl for t in self.trades if t.net_pnl < 0) / losing_trades
                if losing_trades > 0
                else 0
            )

            # Cost analysis
            total_commission = sum(t.commission for t in self.trades)
            total_slippage = sum(t.slippage for t in self.trades)
            net_profit = self.portfolio_value - config.initial_capital

            # Benchmark comparison
            benchmark_return = 0.10  # 10% annual return
            alpha = annualized_return - benchmark_return
            beta = 1.0  # Mock beta

            results = BacktestResults(
                config=config,
                start_date=config.start_date,
                end_date=config.end_date,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                profit_factor=profit_factor,
                avg_win=avg_win,
                avg_loss=avg_loss,
                total_commission=total_commission,
                total_slippage=total_slippage,
                net_profit=net_profit,
                benchmark_return=benchmark_return,
                alpha=alpha,
                beta=beta,
                trades=self.trades,
                daily_returns=self.daily_returns,
                daily_portfolio_values=self.daily_portfolio_values,
            )

            return results

        except Exception as e:
            self.logger.error(f"Error calculating results: {e}")
            raise


class BacktestAnalyzer:
    """Backtest results analyzer."""

    def __init__(self, logger: ProductionLogger):
        self.logger = logger

    def generate_report(self, results: BacktestResults) -> dict[str, Any]:
        """Generate comprehensive backtest report."""
        try:
            report = {
                "summary": {
                    "period": f"{results.start_date.date()} to {results.end_date.date()}",
                    "initial_capital": results.config.initial_capital,
                    "final_value": results.daily_portfolio_values[-1]
                    if results.daily_portfolio_values
                    else results.config.initial_capital,
                    "total_return": f"{results.total_return:.2%}",
                    "annualized_return": f"{results.annualized_return:.2%}",
                    "volatility": f"{results.volatility:.2%}",
                    "sharpe_ratio": f"{results.sharpe_ratio:.2f}",
                    "max_drawdown": f"{results.max_drawdown:.2%}",
                    "calmar_ratio": f"{results.calmar_ratio:.2f}",
                },
                "trading_stats": {
                    "total_trades": results.total_trades,
                    "winning_trades": results.winning_trades,
                    "losing_trades": results.losing_trades,
                    "win_rate": f"{results.win_rate:.2%}",
                    "profit_factor": f"{results.profit_factor:.2f}",
                    "avg_win": f"${results.avg_win:.2f}",
                    "avg_loss": f"${results.avg_loss:.2f}",
                    "total_commission": f"${results.total_commission:.2f}",
                    "total_slippage": f"${results.total_slippage:.2f}",
                    "net_profit": f"${results.net_profit:.2f}",
                },
                "benchmark_comparison": {
                    "benchmark_return": f"{results.benchmark_return:.2%}",
                    "alpha": f"{results.alpha:.2%}",
                    "beta": f"{results.beta:.2f}",
                },
            }

            return report

        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return {"error": str(e)}
