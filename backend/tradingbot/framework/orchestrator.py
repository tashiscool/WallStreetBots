"""
Framework Orchestrator

Coordinates the full Alpha -> Portfolio -> Execution -> Risk pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Type
import logging

from .alpha_model import AlphaModel
from .portfolio_model import PortfolioConstructionModel, PortfolioState
from .execution_model import ExecutionModel, Order
from .risk_model import RiskManagementModel, PortfolioRiskMetrics, RiskAssessment
from .insight import Insight
from .portfolio_target import PortfolioTarget

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    # Insight handling
    combine_insights: bool = True  # Combine insights from multiple alpha models
    insight_expiry_check: bool = True

    # Execution
    execute_immediately: bool = True
    max_orders_per_update: int = 50

    # Risk
    check_risk_before_execution: bool = True
    halt_on_risk_breach: bool = True

    # Logging
    log_insights: bool = True
    log_targets: bool = True
    log_orders: bool = True


class FrameworkOrchestrator:
    """
    Orchestrates the algorithmic trading framework.

    Coordinates the flow:
    1. AlphaModel(s) generate Insights
    2. PortfolioConstructionModel converts Insights to Targets
    3. RiskManagementModel approves/modifies Targets
    4. ExecutionModel generates Orders

    Example:
        # Create models
        alpha = RSIAlphaModel(rsi_period=14)
        portfolio = EqualWeightPortfolioModel(max_positions=10)
        execution = ImmediateExecutionModel()
        risk = MaxDrawdownRiskModel(max_drawdown=0.10)

        # Create orchestrator
        orchestrator = FrameworkOrchestrator(
            alpha_models=[alpha],
            portfolio_model=portfolio,
            execution_model=execution,
            risk_model=risk,
        )

        # Run on market data
        orders = orchestrator.update(market_data, symbols)

        # Submit orders to broker
        for order in orders:
            broker.submit_order(order)
    """

    def __init__(
        self,
        alpha_models: List[AlphaModel],
        portfolio_model: PortfolioConstructionModel,
        execution_model: ExecutionModel,
        risk_model: Optional[RiskManagementModel] = None,
        config: Optional[OrchestratorConfig] = None,
    ):
        """
        Initialize FrameworkOrchestrator.

        Args:
            alpha_models: List of alpha models for signal generation
            portfolio_model: Portfolio construction model
            execution_model: Execution model
            risk_model: Risk management model (optional)
            config: Orchestrator configuration
        """
        self.alpha_models = alpha_models
        self.portfolio_model = portfolio_model
        self.execution_model = execution_model
        self.risk_model = risk_model
        self.config = config or OrchestratorConfig()

        # State tracking
        self._current_insights: List[Insight] = []
        self._current_targets: List[PortfolioTarget] = []
        self._current_orders: List[Order] = []
        self._risk_metrics = PortfolioRiskMetrics()
        self._last_update: Optional[datetime] = None

        # Universe tracking
        self._universe: List[str] = []

        logger.info(
            f"FrameworkOrchestrator initialized with {len(alpha_models)} alpha models"
        )

    def set_universe(self, symbols: List[str]) -> None:
        """
        Set the trading universe.

        Args:
            symbols: List of symbols to trade
        """
        added = set(symbols) - set(self._universe)
        removed = set(self._universe) - set(symbols)

        # Notify models of universe changes
        for alpha in self.alpha_models:
            alpha.on_securities_changed(list(added), list(removed))

        self._universe = symbols
        logger.info(f"Universe updated: {len(symbols)} symbols")

    def update_portfolio_state(
        self,
        cash: Decimal,
        positions: Dict[str, Decimal],
        prices: Dict[str, Decimal],
    ) -> PortfolioState:
        """
        Update portfolio state for position sizing.

        Args:
            cash: Current cash balance
            positions: Current positions {symbol: quantity}
            prices: Current prices {symbol: price}

        Returns:
            Updated PortfolioState
        """
        return self.portfolio_model.update_portfolio_state(cash, positions, prices)

    def update_risk_metrics(self, metrics: PortfolioRiskMetrics) -> None:
        """
        Update risk metrics.

        Args:
            metrics: Current portfolio risk metrics
        """
        self._risk_metrics = metrics

    def generate_insights(
        self,
        market_data: Dict[str, Any],
        symbols: Optional[List[str]] = None,
    ) -> List[Insight]:
        """
        Generate insights from all alpha models.

        Args:
            market_data: Market data dictionary
            symbols: Symbols to analyze (defaults to universe)

        Returns:
            List of all generated insights
        """
        symbols = symbols or self._universe
        all_insights = []

        for alpha in self.alpha_models:
            try:
                insights = alpha.update(market_data, symbols)
                all_insights.extend(insights)

                if self.config.log_insights:
                    logger.debug(
                        f"{alpha.name} generated {len(insights)} insights"
                    )
            except Exception as e:
                logger.error(f"Error in {alpha.name}: {e}")

        self._current_insights = all_insights
        return all_insights

    def create_targets(
        self,
        insights: Optional[List[Insight]] = None,
    ) -> List[PortfolioTarget]:
        """
        Create portfolio targets from insights.

        Args:
            insights: Insights to process (defaults to current insights)

        Returns:
            List of portfolio targets
        """
        insights = insights or self._current_insights

        # Filter expired insights
        if self.config.insight_expiry_check:
            insights = [i for i in insights if not i.is_expired]

        targets = self.portfolio_model.create_targets(
            insights,
            self.portfolio_model.portfolio_state,
        )

        if self.config.log_targets:
            logger.debug(f"Created {len(targets)} portfolio targets")

        self._current_targets = targets
        return targets

    def apply_risk_management(
        self,
        targets: Optional[List[PortfolioTarget]] = None,
    ) -> List[PortfolioTarget]:
        """
        Apply risk management to targets.

        Args:
            targets: Targets to assess (defaults to current targets)

        Returns:
            Approved/modified targets
        """
        targets = targets or self._current_targets

        if not self.risk_model:
            return targets

        # Check if trading is halted
        if self.risk_model.is_halted():
            logger.warning(
                f"Trading halted: {self.risk_model.get_halt_reason()}"
            )
            return []

        # Get risk assessments
        assessments = self.risk_model.manage_risk(targets, self._risk_metrics)

        # Apply assessments
        approved_targets = self.risk_model.apply_risk_assessments(assessments)

        logger.debug(
            f"Risk: {len(targets)} targets -> {len(approved_targets)} approved"
        )

        self._current_targets = approved_targets
        return approved_targets

    def execute_targets(
        self,
        targets: Optional[List[PortfolioTarget]] = None,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> List[Order]:
        """
        Generate orders from targets.

        Args:
            targets: Targets to execute (defaults to current targets)
            market_data: Market data for intelligent execution

        Returns:
            List of orders to submit
        """
        targets = targets or self._current_targets

        orders = self.execution_model.execute(targets, market_data)

        if self.config.log_orders:
            logger.debug(f"Generated {len(orders)} orders")

        self._current_orders = orders
        return orders

    def update(
        self,
        market_data: Dict[str, Any],
        symbols: Optional[List[str]] = None,
    ) -> List[Order]:
        """
        Run full update cycle: Insights -> Targets -> Risk -> Orders.

        Args:
            market_data: Market data dictionary
            symbols: Symbols to analyze

        Returns:
            List of orders to submit
        """
        self._last_update = datetime.now()

        # Step 1: Generate insights
        insights = self.generate_insights(market_data, symbols)
        if not insights:
            logger.debug("No insights generated")
            return []

        # Step 2: Create targets
        targets = self.create_targets(insights)
        if not targets:
            logger.debug("No targets created")
            return []

        # Step 3: Apply risk management
        if self.config.check_risk_before_execution:
            targets = self.apply_risk_management(targets)
            if not targets:
                logger.debug("All targets rejected by risk")
                return []

        # Step 4: Generate orders
        orders = self.execute_targets(targets, market_data)

        # Limit orders per update
        if len(orders) > self.config.max_orders_per_update:
            logger.warning(
                f"Limiting orders from {len(orders)} to {self.config.max_orders_per_update}"
            )
            orders = orders[:self.config.max_orders_per_update]

        return orders

    def on_order_filled(
        self,
        order: Order,
        fill_price: Decimal,
        fill_quantity: Decimal,
    ) -> None:
        """Handle order fill event."""
        self.execution_model.on_order_filled(order, fill_price, fill_quantity)

    def get_active_insights(self) -> List[Insight]:
        """Get all active (non-expired) insights."""
        all_active = []
        for alpha in self.alpha_models:
            all_active.extend(alpha.get_all_active_insights())
        return all_active

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            'alpha_models': [a.get_stats() for a in self.alpha_models],
            'portfolio_model': self.portfolio_model.get_stats(),
            'execution_model': self.execution_model.get_stats(),
            'risk_model': self.risk_model.get_stats() if self.risk_model else None,
            'universe_size': len(self._universe),
            'active_insights': len(self.get_active_insights()),
            'last_update': self._last_update.isoformat() if self._last_update else None,
        }

    def __repr__(self) -> str:
        return (
            f"FrameworkOrchestrator("
            f"alphas={len(self.alpha_models)}, "
            f"universe={len(self._universe)})"
        )
