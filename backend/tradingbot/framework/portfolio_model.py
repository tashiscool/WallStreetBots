"""
Portfolio Construction Model Base Class

PortfolioConstructionModels convert insights into portfolio targets.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set
import logging

from .insight import Insight, InsightDirection
from .portfolio_target import PortfolioTarget

logger = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    """Current portfolio state."""
    cash: Decimal = Decimal("100000")
    positions: Dict[str, Decimal] = field(default_factory=dict)  # symbol -> quantity
    position_values: Dict[str, Decimal] = field(default_factory=dict)  # symbol -> market value
    total_value: Decimal = Decimal("100000")
    last_update: Optional[datetime] = None


class PortfolioConstructionModel(ABC):
    """
    Base class for portfolio construction.

    PortfolioConstructionModels take Insights from AlphaModels and
    convert them into PortfolioTargets (position sizes).

    Override create_targets() to implement your sizing logic.

    Example:
        class EqualWeightPortfolioModel(PortfolioConstructionModel):
            def __init__(self, max_positions=10):
                super().__init__("EqualWeight")
                self.max_positions = max_positions

            def create_targets(self, insights, portfolio_state):
                weight = 1.0 / min(len(insights), self.max_positions)
                return [
                    PortfolioTarget(
                        symbol=insight.symbol,
                        target_weight=weight if insight.is_long else -weight,
                        quantity=self._calculate_quantity(insight.symbol, weight),
                    )
                    for insight in insights[:self.max_positions]
                ]
    """

    def __init__(self, name: str = "PortfolioModel"):
        """
        Initialize PortfolioConstructionModel.

        Args:
            name: Name of this model (for tracking)
        """
        self.name = name
        self.portfolio_state = PortfolioState()

    @abstractmethod
    def create_targets(
        self,
        insights: List[Insight],
        portfolio_state: Optional[PortfolioState] = None,
    ) -> List[PortfolioTarget]:
        """
        Convert insights to portfolio targets.

        This is the main method to override in subclasses.

        Args:
            insights: List of trading insights from AlphaModels
            portfolio_state: Current portfolio state

        Returns:
            List of PortfolioTarget objects
        """
        pass

    def on_securities_changed(
        self,
        added: List[str],
        removed: List[str],
        portfolio_state: PortfolioState,
    ) -> List[PortfolioTarget]:
        """
        Handle universe changes.

        Override to customize behavior when securities are added/removed.
        Default behavior: liquidate removed securities.

        Args:
            added: Symbols added to universe
            removed: Symbols removed from universe
            portfolio_state: Current portfolio state

        Returns:
            List of targets (typically liquidation targets for removed symbols)
        """
        targets = []

        # Liquidate removed symbols
        for symbol in removed:
            if symbol in portfolio_state.positions:
                targets.append(PortfolioTarget.liquidate(symbol))
                logger.info(f"{self.name}: Liquidating {symbol} (removed from universe)")

        return targets

    def update_portfolio_state(
        self,
        cash: Decimal,
        positions: Dict[str, Decimal],
        prices: Dict[str, Decimal],
    ) -> PortfolioState:
        """
        Update internal portfolio state.

        Args:
            cash: Current cash balance
            positions: Current positions {symbol: quantity}
            prices: Current prices {symbol: price}

        Returns:
            Updated PortfolioState
        """
        position_values = {
            symbol: qty * prices.get(symbol, Decimal("0"))
            for symbol, qty in positions.items()
        }

        total_value = cash + sum(position_values.values())

        self.portfolio_state = PortfolioState(
            cash=cash,
            positions=positions,
            position_values=position_values,
            total_value=total_value,
            last_update=datetime.now(),
        )

        return self.portfolio_state

    def get_current_weight(self, symbol: str) -> float:
        """Get current portfolio weight for a symbol."""
        if self.portfolio_state.total_value == 0:
            return 0.0
        position_value = self.portfolio_state.position_values.get(symbol, Decimal("0"))
        return float(position_value / self.portfolio_state.total_value)

    def calculate_quantity_from_weight(
        self,
        symbol: str,
        target_weight: float,
        current_price: Decimal,
    ) -> Decimal:
        """
        Calculate quantity needed to achieve target weight.

        Args:
            symbol: Ticker symbol
            target_weight: Target portfolio weight (e.g., 0.10 for 10%)
            current_price: Current price of the asset

        Returns:
            Number of shares/contracts to hold
        """
        if current_price <= 0:
            return Decimal("0")

        target_value = self.portfolio_state.total_value * Decimal(str(target_weight))
        return (target_value / current_price).quantize(Decimal("1"))

    def filter_insights(
        self,
        insights: List[Insight],
        min_confidence: float = 0.0,
        directions: Optional[List[InsightDirection]] = None,
    ) -> List[Insight]:
        """
        Filter insights based on criteria.

        Args:
            insights: List of insights to filter
            min_confidence: Minimum confidence threshold
            directions: Allowed directions (None = all)

        Returns:
            Filtered list of insights
        """
        filtered = [i for i in insights if i.confidence >= min_confidence]

        if directions:
            filtered = [i for i in filtered if i.direction in directions]

        return filtered

    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        return {
            'name': self.name,
            'portfolio_value': float(self.portfolio_state.total_value),
            'cash': float(self.portfolio_state.cash),
            'num_positions': len(self.portfolio_state.positions),
            'last_update': (
                self.portfolio_state.last_update.isoformat()
                if self.portfolio_state.last_update else None
            ),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
