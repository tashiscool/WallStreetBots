"""
Framework Strategy - Unified strategy composer.

Bridges the Algorithm Framework pipeline (AlphaModel -> PortfolioConstruction ->
Execution) with the BaseStrategy interface, allowing framework-composed strategies
to plug directly into the production trading system.

Usage:
    from backend.tradingbot.framework import (
        EmaCrossAlphaModel,
        EqualWeightingPortfolioModel,
        ImmediateExecutionModel,
    )

    strategy = FrameworkStrategy(
        config=StrategyConfig(name="ema-cross"),
        alpha_model=EmaCrossAlphaModel(fast=10, slow=30),
        portfolio_model=EqualWeightingPortfolioModel(),
        execution_model=ImmediateExecutionModel(),
    )

    # Works as a regular BaseStrategy
    result = strategy.analyze(data, "AAPL")

    # Or run the full lifecycle
    strategy.run_lifecycle_cycle()
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.tradingbot.strategies.base.base_strategy import (
    BaseStrategy,
    SignalType,
    StrategyConfig,
    StrategyResult,
)

logger = logging.getLogger(__name__)


class FrameworkStrategy(BaseStrategy):
    """Strategy composed from Algorithm Framework components.

    Bridges AlphaModel, PortfolioConstructionModel, and ExecutionModel
    into the BaseStrategy interface so framework-based strategies can run
    inside the production trading system.

    Attributes:
        alpha_model: Generates insights (trading signals) from market data
        portfolio_model: Converts insights to portfolio targets (position sizing)
        execution_model: Converts targets to executable orders
        risk_model: Optional risk management filter
        universe_model: Optional universe selection filter
    """

    def __init__(
        self,
        config: StrategyConfig,
        alpha_model: Any,
        portfolio_model: Any,
        execution_model: Any,
        risk_model: Any = None,
        universe_model: Any = None,
    ):
        super().__init__(config)
        self.alpha_model = alpha_model
        self.portfolio_model = portfolio_model
        self.execution_model = execution_model
        self.risk_model = risk_model
        self.universe_model = universe_model

        # State
        self._current_holdings: Dict[str, float] = {}
        self._current_prices: Dict[str, float] = {}
        self._portfolio_value: float = float(config.max_total_risk)
        self._active_insights: List[Any] = []
        self._order_tickets: List[Any] = []

    def analyze(self, data: pd.DataFrame, symbol: str) -> Optional[StrategyResult]:
        """Run the full framework pipeline for a single symbol.

        Args:
            data: Market data DataFrame with OHLCV columns
            symbol: Symbol to analyze

        Returns:
            StrategyResult if the pipeline produces a trade signal
        """
        # Prepare data dict for alpha model
        market_data = {symbol: self._dataframe_to_dict(data, symbol)}

        # Step 1: Generate insights from alpha model
        insights = self.alpha_model.update(market_data)
        if not insights:
            return None

        self._active_insights = insights

        # Step 2: Convert insights to portfolio targets
        targets = self.portfolio_model.create_targets(
            insights=insights,
            portfolio_value=self._portfolio_value,
            current_holdings=self._current_holdings,
        )
        if not targets:
            return None

        # Step 3: Apply risk management if available
        if self.risk_model is not None:
            try:
                result = self.risk_model.manage_risk(
                    targets=targets,
                    portfolio_value=self._portfolio_value,
                    current_holdings=self._current_holdings,
                )
                if hasattr(result, 'adjusted_targets'):
                    targets = result.adjusted_targets
            except Exception as e:
                logger.warning(f"Risk model error: {e}")

        # Step 4: Generate orders via execution model
        orders = self.execution_model.execute(
            targets=targets,
            current_holdings=self._current_holdings,
            current_prices=self._current_prices,
            portfolio_value=self._portfolio_value,
        )
        self._order_tickets = orders

        if not orders:
            return None

        # Convert first order to StrategyResult
        order = orders[0]
        price = data['close'].iloc[-1] if 'close' in data.columns else 0.0

        signal = SignalType.BUY if getattr(order, 'quantity', 0) > 0 else SignalType.SELL

        # Derive confidence from insight
        confidence = 0.5
        for insight in insights:
            if hasattr(insight, 'symbol') and insight.symbol == symbol:
                confidence = abs(getattr(insight, 'magnitude', 0.5))
                break

        return StrategyResult(
            symbol=symbol,
            signal=signal,
            confidence=min(1.0, confidence),
            price=price,
            quantity=abs(int(getattr(order, 'quantity', 0))),
            timestamp=datetime.now(),
            reasoning=f"Framework: {self.alpha_model.name} -> {self.portfolio_model.name}",
            metadata={
                "alpha_model": self.alpha_model.name,
                "portfolio_model": self.portfolio_model.name,
                "execution_model": self.execution_model.name,
                "n_insights": len(insights),
                "n_targets": len(targets),
                "n_orders": len(orders),
            },
        )

    def analyze_multi(self, data: Dict[str, pd.DataFrame]) -> List[StrategyResult]:
        """Run the framework pipeline across multiple symbols.

        Args:
            data: Dict mapping symbol -> DataFrame

        Returns:
            List of StrategyResults for all symbols with signals
        """
        # Convert to alpha model format
        market_data = {}
        for symbol, df in data.items():
            market_data[symbol] = self._dataframe_to_dict(df, symbol)
            if 'close' in df.columns:
                self._current_prices[symbol] = float(df['close'].iloc[-1])

        # Step 1: Generate insights
        insights = self.alpha_model.update(market_data)
        if not insights:
            return []

        self._active_insights = insights

        # Step 2: Portfolio targets
        targets = self.portfolio_model.create_targets(
            insights=insights,
            portfolio_value=self._portfolio_value,
            current_holdings=self._current_holdings,
        )

        # Step 3: Risk management
        if self.risk_model and targets:
            try:
                result = self.risk_model.manage_risk(
                    targets=targets,
                    portfolio_value=self._portfolio_value,
                    current_holdings=self._current_holdings,
                )
                if hasattr(result, 'adjusted_targets'):
                    targets = result.adjusted_targets
            except Exception as e:
                logger.warning(f"Risk model error: {e}")

        # Step 4: Execution
        if not targets:
            return []

        orders = self.execution_model.execute(
            targets=targets,
            current_holdings=self._current_holdings,
            current_prices=self._current_prices,
            portfolio_value=self._portfolio_value,
        )
        self._order_tickets = orders

        # Convert orders to results
        results = []
        for order in orders:
            symbol = getattr(order, 'symbol', '')
            price = self._current_prices.get(symbol, 0.0)
            qty = getattr(order, 'quantity', 0)
            signal = SignalType.BUY if qty > 0 else SignalType.SELL

            results.append(StrategyResult(
                symbol=symbol,
                signal=signal,
                confidence=0.5,
                price=price,
                quantity=abs(int(qty)),
                timestamp=datetime.now(),
                reasoning="Framework pipeline signal",
            ))

        return results

    def get_required_data(self) -> List[str]:
        """Required data columns for the framework strategy."""
        return ["open", "high", "low", "close", "volume"]

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data has required columns."""
        required = {"close", "volume"}
        return required.issubset(set(data.columns))

    def update_holdings(self, holdings: Dict[str, float]) -> None:
        """Update current holdings state."""
        self._current_holdings = holdings.copy()

    def update_portfolio_value(self, value: float) -> None:
        """Update portfolio value."""
        self._portfolio_value = value

    @property
    def active_insights(self) -> List[Any]:
        """Get the most recent insights."""
        return self._active_insights

    @property
    def order_tickets(self) -> List[Any]:
        """Get the most recent order tickets."""
        return self._order_tickets

    def trace_stats(self) -> Dict[str, Any]:
        """Track framework-specific stats."""
        return {
            "active_insights": len(self._active_insights),
            "pending_orders": len(self._order_tickets),
            "portfolio_value": self._portfolio_value,
            "n_holdings": len(self._current_holdings),
        }

    @staticmethod
    def _dataframe_to_dict(df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Convert DataFrame to dict format expected by alpha models."""
        result = {"symbol": symbol}
        for col in df.columns:
            result[col] = df[col].values
        return result
