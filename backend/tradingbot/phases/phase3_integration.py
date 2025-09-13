"""Phase 3 Integration Manager
Integrates all Phase 3 strategies with existing infrastructure.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .production_config import ConfigManager

# Import Phase 3 strategies
from .production_earnings_protection import EarningsProtectionStrategy
from .production_leaps_tracker import LEAPSTrackerStrategy
from .production_logging import ProductionLogger
from .production_lotto_scanner import LottoScannerStrategy
from .production_momentum_weeklies import MomentumWeekliesStrategy
from .production_swing_trading import SwingTradingStrategy
from .trading_interface import TradingInterface
from .unified_data_provider import UnifiedDataProvider


@dataclass
class Phase3StrategyStatus:
    """Phase 3 strategy status."""

    strategy_name: str
    is_active: bool
    active_positions: int
    total_pnl: float
    total_exposure: float
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class Phase3PortfolioSummary:
    """Phase 3 portfolio summary."""

    total_strategies: int
    active_strategies: int
    total_positions: int
    total_pnl: float
    total_exposure: float
    risk_alerts: list[str]
    last_update: datetime = field(default_factory=datetime.now)


class Phase3StrategyManager:
    """Phase 3 strategy manager."""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = ProductionLogger("phase3_manager")

        # Initialize data provider and trading interface
        self.data_provider = UnifiedDataProvider(self.logger)
        self.trading_interface = TradingInterface(
            alpaca_api_key=TEST_API_KEY, alpaca_secret_key=TEST_SECRET_KEY, paper_trading=True
        )

        # Initialize Phase 3 strategies
        self.earnings_protection = EarningsProtectionStrategy(
            self.trading_interface, self.data_provider, self.config, self.logger
        )

        self.swing_trading = SwingTradingStrategy(
            self.trading_interface, self.data_provider, self.config, self.logger
        )

        self.momentum_weeklies = MomentumWeekliesStrategy(
            self.trading_interface, self.data_provider, self.config, self.logger
        )

        self.lotto_scanner = LottoScannerStrategy(
            self.trading_interface, self.data_provider, self.config, self.logger
        )

        self.leaps_tracker = LEAPSTrackerStrategy(
            self.trading_interface, self.data_provider, self.config, self.logger
        )

        # Strategy registry
        self.strategies = {
            "earnings_protection": self.earnings_protection,
            "swing_trading": self.swing_trading,
            "momentum_weeklies": self.momentum_weeklies,
            "lotto_scanner": self.lotto_scanner,
            "leaps_tracker": self.leaps_tracker,
        }

        self.logger.info("Phase3StrategyManager initialized")

    async def initialize(self):
        """Initialize Phase 3 strategies."""
        try:
            self.logger.info("Initializing Phase 3 strategies")

            # Initialize data provider
            await self.data_provider.initialize()

            # Initialize trading interface
            await self.trading_interface.initialize()

            self.logger.info("Phase 3 strategies initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing Phase 3 strategies: {e}")
            raise

    async def scan_all_opportunities(self) -> dict[str, list[Any]]:
        """Scan for opportunities across all Phase 3 strategies."""
        try:
            self.logger.info("Scanning for opportunities across all Phase 3 strategies")

            opportunities = {}

            # Scan earnings protection opportunities
            try:
                earnings_opportunities = (
                    await self.earnings_protection.scan_for_earnings_opportunities()
                )
                opportunities["earnings_protection"] = earnings_opportunities
                self.logger.info(
                    f"Found {len(earnings_opportunities)} earnings protection opportunities"
                )
            except Exception as e:
                self.logger.error(f"Error scanning earnings protection opportunities: {e}")
                opportunities["earnings_protection"] = []

            # Scan swing trading opportunities
            try:
                swing_opportunities = await self.swing_trading.scan_for_swing_opportunities()
                opportunities["swing_trading"] = swing_opportunities
                self.logger.info(f"Found {len(swing_opportunities)} swing trading opportunities")
            except Exception as e:
                self.logger.error(f"Error scanning swing trading opportunities: {e}")
                opportunities["swing_trading"] = []

            # Scan momentum weeklies opportunities
            try:
                momentum_opportunities = (
                    await self.momentum_weeklies.scan_for_momentum_opportunities()
                )
                opportunities["momentum_weeklies"] = momentum_opportunities
                self.logger.info(
                    f"Found {len(momentum_opportunities)} momentum weeklies opportunities"
                )
            except Exception as e:
                self.logger.error(f"Error scanning momentum weeklies opportunities: {e}")
                opportunities["momentum_weeklies"] = []

            # Scan lotto scanner opportunities
            try:
                lotto_opportunities = await self.lotto_scanner.scan_for_lotto_opportunities()
                opportunities["lotto_scanner"] = lotto_opportunities
                self.logger.info(f"Found {len(lotto_opportunities)} lotto scanner opportunities")
            except Exception as e:
                self.logger.error(f"Error scanning lotto scanner opportunities: {e}")
                opportunities["lotto_scanner"] = []

            # Scan LEAPS tracker opportunities
            try:
                leaps_opportunities = await self.leaps_tracker.scan_for_leaps_opportunities()
                opportunities["leaps_tracker"] = leaps_opportunities
                self.logger.info(f"Found {len(leaps_opportunities)} LEAPS tracker opportunities")
            except Exception as e:
                self.logger.error(f"Error scanning LEAPS tracker opportunities: {e}")
                opportunities["leaps_tracker"] = []

            total_opportunities = sum(len(opps) for opps in opportunities.values())
            self.logger.info(f"Total Phase 3 opportunities found: {total_opportunities}")

            return opportunities

        except Exception as e:
            self.logger.error(f"Error scanning all opportunities: {e}")
            return {}

    async def execute_strategy_trades(
        self, opportunities: dict[str, list[Any]]
    ) -> dict[str, list[Any]]:
        """Execute trades for all strategies."""
        try:
            self.logger.info("Executing trades for all Phase 3 strategies")

            executed_trades = {}

            # Execute earnings protection trades
            if "earnings_protection" in opportunities:
                try:
                    earnings_trades = []
                    for candidate in opportunities["earnings_protection"][:3]:  # Limit to top 3
                        trade = await self.earnings_protection.execute_earnings_protection(
                            candidate
                        )
                        if trade:
                            earnings_trades.append(trade)
                    executed_trades["earnings_protection"] = earnings_trades
                    self.logger.info(f"Executed {len(earnings_trades)} earnings protection trades")
                except Exception as e:
                    self.logger.error(f"Error executing earnings protection trades: {e}")
                    executed_trades["earnings_protection"] = []

            # Execute swing trading trades
            if "swing_trading" in opportunities:
                try:
                    swing_trades = []
                    for candidate in opportunities["swing_trading"][:5]:  # Limit to top 5
                        trade = await self.swing_trading.execute_swing_trade(candidate)
                        if trade:
                            swing_trades.append(trade)
                    executed_trades["swing_trading"] = swing_trades
                    self.logger.info(f"Executed {len(swing_trades)} swing trading trades")
                except Exception as e:
                    self.logger.error(f"Error executing swing trading trades: {e}")
                    executed_trades["swing_trading"] = []

            # Execute momentum weeklies trades
            if "momentum_weeklies" in opportunities:
                try:
                    momentum_trades = []
                    for candidate in opportunities["momentum_weeklies"][:5]:  # Limit to top 5
                        trade = await self.momentum_weeklies.execute_momentum_trade(candidate)
                        if trade:
                            momentum_trades.append(trade)
                    executed_trades["momentum_weeklies"] = momentum_trades
                    self.logger.info(f"Executed {len(momentum_trades)} momentum weeklies trades")
                except Exception as e:
                    self.logger.error(f"Error executing momentum weeklies trades: {e}")
                    executed_trades["momentum_weeklies"] = []

            # Execute lotto scanner trades
            if "lotto_scanner" in opportunities:
                try:
                    lotto_trades = []
                    for candidate in opportunities["lotto_scanner"][:3]:  # Limit to top 3
                        trade = await self.lotto_scanner.execute_lotto_trade(candidate)
                        if trade:
                            lotto_trades.append(trade)
                    executed_trades["lotto_scanner"] = lotto_trades
                    self.logger.info(f"Executed {len(lotto_trades)} lotto scanner trades")
                except Exception as e:
                    self.logger.error(f"Error executing lotto scanner trades: {e}")
                    executed_trades["lotto_scanner"] = []

            # Execute LEAPS tracker trades
            if "leaps_tracker" in opportunities:
                try:
                    leaps_trades = []
                    for candidate in opportunities["leaps_tracker"][:3]:  # Limit to top 3
                        trade = await self.leaps_tracker.execute_leaps_trade(candidate)
                        if trade:
                            leaps_trades.append(trade)
                    executed_trades["leaps_tracker"] = leaps_trades
                    self.logger.info(f"Executed {len(leaps_trades)} LEAPS tracker trades")
                except Exception as e:
                    self.logger.error(f"Error executing LEAPS tracker trades: {e}")
                    executed_trades["leaps_tracker"] = []

            total_trades = sum(len(trades) for trades in executed_trades.values())
            self.logger.info(f"Total Phase 3 trades executed: {total_trades}")

            return executed_trades

        except Exception as e:
            self.logger.error(f"Error executing strategy trades: {e}")
            return {}

    async def monitor_all_positions(self) -> dict[str, Any]:
        """Monitor positions across all Phase 3 strategies."""
        try:
            self.logger.info("Monitoring positions across all Phase 3 strategies")

            monitoring_results = {}

            # Monitor earnings protection positions
            try:
                earnings_monitoring = await self.earnings_protection.monitor_earnings_positions()
                monitoring_results["earnings_protection"] = earnings_monitoring
                self.logger.info("Earnings protection positions monitored")
            except Exception as e:
                self.logger.error(f"Error monitoring earnings protection positions: {e}")
                monitoring_results["earnings_protection"] = {"error": str(e)}

            # Monitor swing trading positions
            try:
                swing_monitoring = await self.swing_trading.monitor_swing_positions()
                monitoring_results["swing_trading"] = swing_monitoring
                self.logger.info("Swing trading positions monitored")
            except Exception as e:
                self.logger.error(f"Error monitoring swing trading positions: {e}")
                monitoring_results["swing_trading"] = {"error": str(e)}

            # Monitor momentum weeklies positions
            try:
                momentum_monitoring = await self.momentum_weeklies.monitor_momentum_positions()
                monitoring_results["momentum_weeklies"] = momentum_monitoring
                self.logger.info("Momentum weeklies positions monitored")
            except Exception as e:
                self.logger.error(f"Error monitoring momentum weeklies positions: {e}")
                monitoring_results["momentum_weeklies"] = {"error": str(e)}

            # Monitor lotto scanner positions
            try:
                lotto_monitoring = await self.lotto_scanner.monitor_lotto_positions()
                monitoring_results["lotto_scanner"] = lotto_monitoring
                self.logger.info("Lotto scanner positions monitored")
            except Exception as e:
                self.logger.error(f"Error monitoring lotto scanner positions: {e}")
                monitoring_results["lotto_scanner"] = {"error": str(e)}

            # Monitor LEAPS tracker positions
            try:
                leaps_monitoring = await self.leaps_tracker.monitor_leaps_positions()
                monitoring_results["leaps_tracker"] = leaps_monitoring
                self.logger.info("LEAPS tracker positions monitored")
            except Exception as e:
                self.logger.error(f"Error monitoring LEAPS tracker positions: {e}")
                monitoring_results["leaps_tracker"] = {"error": str(e)}

            self.logger.info("All Phase 3 positions monitored")
            return monitoring_results

        except Exception as e:
            self.logger.error(f"Error monitoring all positions: {e}")
            return {"error": str(e)}

    async def get_strategy_status(self) -> dict[str, Phase3StrategyStatus]:
        """Get status of all Phase 3 strategies."""
        try:
            self.logger.info("Getting status of all Phase 3 strategies")

            status_dict = {}

            # Get earnings protection status
            try:
                earnings_status = await self.earnings_protection.get_strategy_status()
                status_dict["earnings_protection"] = Phase3StrategyStatus(
                    strategy_name="Earnings Protection",
                    is_active=True,
                    active_positions=earnings_status.get("active_positions", 0),
                    total_pnl=earnings_status.get("total_pnl", 0.0),
                    total_exposure=earnings_status.get("total_exposure", 0.0),
                )
            except Exception as e:
                self.logger.error(f"Error getting earnings protection status: {e}")
                status_dict["earnings_protection"] = Phase3StrategyStatus(
                    strategy_name="Earnings Protection",
                    is_active=False,
                    active_positions=0,
                    total_pnl=0.0,
                    total_exposure=0.0,
                )

            # Get swing trading status
            try:
                swing_status = await self.swing_trading.get_strategy_status()
                status_dict["swing_trading"] = Phase3StrategyStatus(
                    strategy_name="Swing Trading",
                    is_active=True,
                    active_positions=swing_status.get("active_positions", 0),
                    total_pnl=swing_status.get("total_pnl", 0.0),
                    total_exposure=swing_status.get("total_exposure", 0.0),
                )
            except Exception as e:
                self.logger.error(f"Error getting swing trading status: {e}")
                status_dict["swing_trading"] = Phase3StrategyStatus(
                    strategy_name="Swing Trading",
                    is_active=False,
                    active_positions=0,
                    total_pnl=0.0,
                    total_exposure=0.0,
                )

            # Get momentum weeklies status
            try:
                momentum_status = await self.momentum_weeklies.get_strategy_status()
                status_dict["momentum_weeklies"] = Phase3StrategyStatus(
                    strategy_name="Momentum Weeklies",
                    is_active=True,
                    active_positions=momentum_status.get("active_positions", 0),
                    total_pnl=momentum_status.get("total_pnl", 0.0),
                    total_exposure=momentum_status.get("total_exposure", 0.0),
                )
            except Exception as e:
                self.logger.error(f"Error getting momentum weeklies status: {e}")
                status_dict["momentum_weeklies"] = Phase3StrategyStatus(
                    strategy_name="Momentum Weeklies",
                    is_active=False,
                    active_positions=0,
                    total_pnl=0.0,
                    total_exposure=0.0,
                )

            # Get lotto scanner status
            try:
                lotto_status = await self.lotto_scanner.get_strategy_status()
                status_dict["lotto_scanner"] = Phase3StrategyStatus(
                    strategy_name="Lotto Scanner",
                    is_active=True,
                    active_positions=lotto_status.get("active_positions", 0),
                    total_pnl=lotto_status.get("total_pnl", 0.0),
                    total_exposure=lotto_status.get("total_exposure", 0.0),
                )
            except Exception as e:
                self.logger.error(f"Error getting lotto scanner status: {e}")
                status_dict["lotto_scanner"] = Phase3StrategyStatus(
                    strategy_name="Lotto Scanner",
                    is_active=False,
                    active_positions=0,
                    total_pnl=0.0,
                    total_exposure=0.0,
                )

            # Get LEAPS tracker status
            try:
                leaps_status = await self.leaps_tracker.get_strategy_status()
                status_dict["leaps_tracker"] = Phase3StrategyStatus(
                    strategy_name="LEAPS Tracker",
                    is_active=True,
                    active_positions=leaps_status.get("active_positions", 0),
                    total_pnl=leaps_status.get("total_pnl", 0.0),
                    total_exposure=leaps_status.get("total_exposure", 0.0),
                )
            except Exception as e:
                self.logger.error(f"Error getting LEAPS tracker status: {e}")
                status_dict["leaps_tracker"] = Phase3StrategyStatus(
                    strategy_name="LEAPS Tracker",
                    is_active=False,
                    active_positions=0,
                    total_pnl=0.0,
                    total_exposure=0.0,
                )

            self.logger.info("All Phase 3 strategy status retrieved")
            return status_dict

        except Exception as e:
            self.logger.error(f"Error getting strategy status: {e}")
            return {}

    async def get_portfolio_summary(self) -> Phase3PortfolioSummary:
        """Get comprehensive portfolio summary."""
        try:
            self.logger.info("Getting Phase 3 portfolio summary")

            # Get strategy status
            strategy_status = await self.get_strategy_status()

            # Calculate totals
            total_strategies = len(strategy_status)
            active_strategies = sum(1 for status in strategy_status.values() if status.is_active)
            total_positions = sum(status.active_positions for status in strategy_status.values())
            total_pnl = sum(status.total_pnl for status in strategy_status.values())
            total_exposure = sum(status.total_exposure for status in strategy_status.values())

            # Collect risk alerts
            risk_alerts = []
            for strategy_name, status in strategy_status.items():
                if status.total_pnl < -1000:  # Large loss threshold
                    risk_alerts.append(f"Large loss in {strategy_name}: ${status.total_pnl:.2f}")
                if status.total_exposure > 50000:  # High exposure threshold
                    risk_alerts.append(
                        f"High exposure in {strategy_name}: ${status.total_exposure:.2f}"
                    )

            summary = Phase3PortfolioSummary(
                total_strategies=total_strategies,
                active_strategies=active_strategies,
                total_positions=total_positions,
                total_pnl=total_pnl,
                total_exposure=total_exposure,
                risk_alerts=risk_alerts,
            )

            self.logger.info(
                f"Portfolio summary: {total_positions} positions, ${total_pnl: .2f} P & L"
            )
            return summary

        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}")
            return Phase3PortfolioSummary(
                total_strategies=0,
                active_strategies=0,
                total_positions=0,
                total_pnl=0.0,
                total_exposure=0.0,
                risk_alerts=[f"Error getting portfolio summary: {e}"],
            )

    async def run_complete_cycle(self) -> dict[str, Any]:
        """Run complete Phase 3 trading cycle."""
        try:
            self.logger.info("Running complete Phase 3 trading cycle")

            cycle_results = {
                "start_time": datetime.now(),
                "opportunities": {},
                "executed_trades": {},
                "monitoring_results": {},
                "portfolio_summary": {},
                "end_time": None,
                "success": False,
            }

            # Scan for opportunities
            opportunities = await self.scan_all_opportunities()
            cycle_results["opportunities"] = opportunities

            # Execute trades
            executed_trades = await self.execute_strategy_trades(opportunities)
            cycle_results["executed_trades"] = executed_trades

            # Monitor positions
            monitoring_results = await self.monitor_all_positions()
            cycle_results["monitoring_results"] = monitoring_results

            # Get portfolio summary
            portfolio_summary = await self.get_portfolio_summary()
            cycle_results["portfolio_summary"] = portfolio_summary

            cycle_results["end_time"] = datetime.now()
            cycle_results["success"] = True

            self.logger.info("Complete Phase 3 trading cycle completed successfully")
            return cycle_results

        except Exception as e:
            self.logger.error(f"Error running complete cycle: {e}")
            cycle_results["end_time"] = datetime.now()
            cycle_results["success"] = False
            cycle_results["error"] = str(e)
            return cycle_results


# Factory functions for Phase 3 components
async def create_phase3_strategy_manager(config_file: str) -> Phase3StrategyManager:
    """Create Phase 3 strategy manager."""
    config_manager = ConfigManager(config_file)
    config = config_manager.load_config()
    return Phase3StrategyManager(config)


async def create_phase3_data_provider(logger: ProductionLogger) -> UnifiedDataProvider:
    """Create Phase 3 data provider."""
    return UnifiedDataProvider(logger)


async def create_phase3_trading_interface(logger: ProductionLogger) -> TradingInterface:
    """Create Phase 3 trading interface."""
    return TradingInterface(
        alpaca_api_key=TEST_API_KEY, alpaca_secret_key=TEST_SECRET_KEY, paper_trading=True
    )
