"""Real - Time Risk Validation System
Implements dynamic risk management with live account data and position monitoring
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classifications"""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationResult(Enum):
    """Risk validation outcomes"""

    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"


@dataclass
class TradeSignal:
    """Trade signal data structure"""

    ticker: str
    action: str  # "BUY" or "SELL"
    quantity: int
    price: Decimal
    option_type: str | None = None
    strike: Decimal | None = None
    expiry_date: datetime | None = None
    strategy: str = "UNKNOWN"
    confidence: float = 0.0

    @property
    def total_value(self) -> Decimal:
        """Calculate total trade value"""
        return self.price * Decimal(str(self.quantity))


@dataclass
class AccountSnapshot:
    """Current account state from broker"""

    portfolio_value: Decimal
    buying_power: Decimal
    day_trade_count: int
    equity: Decimal
    cash: Decimal
    long_market_value: Decimal
    short_market_value: Decimal
    maintenance_margin: Decimal
    last_updated: datetime


@dataclass
class PositionSummary:
    """Summary of current positions"""

    ticker: str
    quantity: int
    market_value: Decimal
    unrealized_pnl: Decimal
    cost_basis: Decimal
    position_type: str  # "stock" or "option"
    risk_exposure: Decimal


@dataclass
class RiskValidationResult:
    """Result of risk validation check"""

    approved: bool
    validation_result: ValidationResult
    original_quantity: int
    approved_quantity: int
    risk_level: RiskLevel
    reason: str
    warnings: list[str]
    risk_metrics: dict[str, Any]
    timestamp: datetime

    @property
    def is_modified(self) -> bool:
        """Check if trade was modified"""
        return self.original_quantity != self.approved_quantity

    @property
    def rejection_reason(self) -> str | None:
        """Get rejection reason if trade was rejected"""
        return self.reason if not self.approved else None


class RealTimeRiskManager:
    """Advanced real - time risk management system"""

    def __init__(self, alpaca_manager=None):
        self.alpaca_manager = alpaca_manager

        # Risk parameters (configurable)
        self.max_portfolio_risk = Decimal("0.20")  # 20% max portfolio risk
        self.max_single_position_risk = Decimal("0.05")  # 5% max per position
        self.max_daily_loss = Decimal("0.10")  # 10% max daily loss
        self.max_options_allocation = Decimal("0.15")  # 15% max in options
        self.min_buying_power_buffer = Decimal("0.20")  # 20% buying power buffer
        self.max_correlation_risk = Decimal("0.30")  # 30% max in correlated positions

        # Day trading limits
        self.max_day_trades = 3  # For non - PDT accounts
        self.pdt_threshold = Decimal("25000")  # PDT threshold

        # Risk tracking
        self.risk_events = []
        self.daily_losses = Decimal("0")
        self.last_account_update = None
        self.cached_account = None
        self.cached_positions = []

    async def validate_trade_safety(self, trade_signal: TradeSignal) -> RiskValidationResult:
        """Comprehensive trade validation with live account data"""
        try:
            # Get fresh account data
            account = await self._get_current_account()
            positions = await self._get_current_positions()

            # Initialize validation result
            result = RiskValidationResult(
                approved=False,
                validation_result=ValidationResult.REJECTED,
                original_quantity=trade_signal.quantity,
                approved_quantity=0,
                risk_level=RiskLevel.HIGH,
                reason="Validation pending",
                warnings=[],
                risk_metrics={},
                timestamp=datetime.now(),
            )

            # Run all risk checks
            checks = [
                self._check_portfolio_risk(trade_signal, account, positions),
                self._check_position_sizing(trade_signal, account, positions),
                self._check_buying_power(trade_signal, account),
                self._check_daily_loss_limits(trade_signal, account),
                self._check_day_trading_limits(trade_signal, account),
                self._check_options_allocation(trade_signal, account, positions),
                self._check_correlation_risk(trade_signal, positions),
                self._check_volatility_risk(trade_signal),
            ]

            # Execute all checks concurrently
            check_results = await asyncio.gather(*checks, return_exceptions=True)

            # Process check results
            all_passed = True
            highest_risk = RiskLevel.LOW
            approved_qty = trade_signal.quantity

            for i, check_result in enumerate(check_results):
                if isinstance(check_result, Exception):
                    logger.error(f"Risk check {i} failed: {check_result}")
                    result.warnings.append(f"Risk check failed: {check_result}")
                    all_passed = False
                    continue

                check_passed, risk_level, suggested_qty, warning = check_result

                if not check_passed:
                    all_passed = False

                if risk_level.value > highest_risk.value:
                    highest_risk = risk_level

                approved_qty = min(approved_qty, suggested_qty)

                if warning:
                    result.warnings.append(warning)

            # Calculate final risk metrics
            result.risk_metrics = await self._calculate_risk_metrics(
                trade_signal, account, positions
            )

            # Determine final approval status
            result.risk_level = highest_risk
            result.approved_quantity = approved_qty

            if not all_passed:
                result.approved = False
                result.validation_result = ValidationResult.REJECTED
                result.reason = "Failed risk validation checks"
            elif approved_qty < trade_signal.quantity:
                result.approved = True
                result.validation_result = ValidationResult.MODIFIED
                result.reason = (
                    f"Position size reduced from {trade_signal.quantity} to {approved_qty}"
                )
            else:
                result.approved = True
                result.validation_result = ValidationResult.APPROVED
                result.reason = "All risk checks passed"

            # Log validation result
            await self._log_validation_result(trade_signal, result)

            return result

        except Exception as e:
            logger.error(f"Risk validation failed for {trade_signal.ticker}: {e}")
            return RiskValidationResult(
                approved=False,
                validation_result=ValidationResult.REJECTED,
                original_quantity=trade_signal.quantity,
                approved_quantity=0,
                risk_level=RiskLevel.CRITICAL,
                reason=f"Risk validation system error: {e}",
                warnings=[],
                risk_metrics={},
                timestamp=datetime.now(),
            )

    async def _get_current_account(self) -> AccountSnapshot:
        """Get current account data from broker with caching"""
        try:
            # Use cached data if recent ( <  30 seconds)
            if (
                self.cached_account
                and self.last_account_update
                and datetime.now() - self.last_account_update < timedelta(seconds=30)
            ):
                return self.cached_account

            # Get fresh account data from Alpaca
            if not self.alpaca_manager:
                raise Exception("Alpaca manager not configured")

            account_data = await self.alpaca_manager.get_account()

            # Convert to our data structure
            account_snapshot = AccountSnapshot(
                portfolio_value=Decimal(str(account_data.portfolio_value)),
                buying_power=Decimal(str(account_data.buying_power)),
                day_trade_count=account_data.daytrade_count,
                equity=Decimal(str(account_data.equity)),
                cash=Decimal(str(account_data.cash)),
                long_market_value=Decimal(str(account_data.long_market_value)),
                short_market_value=Decimal(str(account_data.short_market_value)),
                maintenance_margin=Decimal(str(account_data.maintenance_margin)),
                last_updated=datetime.now(),
            )

            # Cache the result
            self.cached_account = account_snapshot
            self.last_account_update = datetime.now()

            return account_snapshot

        except Exception as e:
            logger.error(f"Failed to get account data: {e}")
            raise

    async def _get_current_positions(self) -> list[PositionSummary]:
        """Get current positions from broker"""
        try:
            if not self.alpaca_manager:
                return []

            positions = await self.alpaca_manager.get_positions()
            position_summaries = []

            for pos in positions:
                # Determine position type
                position_type = "option" if "/" in pos.symbol else "stock"

                # Calculate risk exposure (market value * some risk factor)
                market_val = Decimal(str(pos.market_value))
                risk_exposure = abs(market_val)  # Simplified risk calculation

                position_summaries.append(
                    PositionSummary(
                        ticker=pos.symbol,
                        quantity=int(pos.qty),
                        market_value=market_val,
                        unrealized_pnl=Decimal(str(pos.unrealized_pnl)),
                        cost_basis=Decimal(str(pos.cost_basis)),
                        position_type=position_type,
                        risk_exposure=risk_exposure,
                    )
                )

            self.cached_positions = position_summaries
            return position_summaries

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    async def _check_portfolio_risk(
        self, trade_signal: TradeSignal, account: AccountSnapshot, positions: list[PositionSummary]
    ) -> tuple:
        """Check overall portfolio risk exposure"""
        try:
            # Calculate current risk exposure
            current_risk = sum(pos.risk_exposure for pos in positions)

            # Calculate proposed trade risk
            trade_risk = trade_signal.total_value

            # Total risk after trade
            total_risk = current_risk + trade_risk
            risk_percentage = total_risk / account.portfolio_value

            # Check against limit
            if risk_percentage > self.max_portfolio_risk:
                # Calculate maximum allowed quantity
                max_allowed_risk = account.portfolio_value * self.max_portfolio_risk
                available_risk = max_allowed_risk - current_risk
                max_quantity = (
                    int(available_risk / trade_signal.price) if trade_signal.price > 0 else 0
                )

                return (
                    False,
                    RiskLevel.HIGH,
                    max(0, max_quantity),
                    f"Portfolio risk would exceed {self.max_portfolio_risk: .1%} limit",
                )

            # Determine risk level
            if risk_percentage > self.max_portfolio_risk * Decimal("0.8"):
                risk_level = RiskLevel.HIGH
            elif risk_percentage > self.max_portfolio_risk * Decimal("0.6"):
                risk_level = RiskLevel.MODERATE
            else:
                risk_level = RiskLevel.LOW

            return (True, risk_level, trade_signal.quantity, None)

        except Exception as e:
            logger.error(f"Portfolio risk check failed: {e}")
            return (False, RiskLevel.CRITICAL, 0, f"Portfolio risk check error: {e}")

    async def _check_position_sizing(
        self, trade_signal: TradeSignal, account: AccountSnapshot, positions: list[PositionSummary]
    ) -> tuple:
        """Check individual position size limits"""
        try:
            trade_value = trade_signal.total_value
            position_percentage = trade_value / account.portfolio_value

            if position_percentage > self.max_single_position_risk:
                # Calculate max allowed quantity
                max_value = account.portfolio_value * self.max_single_position_risk
                max_quantity = int(max_value / trade_signal.price) if trade_signal.price > 0 else 0

                return (
                    False,
                    RiskLevel.HIGH,
                    max(0, max_quantity),
                    f"Single position would exceed {self.max_single_position_risk: .1%} limit",
                )

            return (True, RiskLevel.LOW, trade_signal.quantity, None)

        except Exception as e:
            logger.error(f"Position sizing check failed: {e}")
            return (False, RiskLevel.CRITICAL, 0, f"Position sizing error: {e}")

    async def _check_buying_power(
        self, trade_signal: TradeSignal, account: AccountSnapshot
    ) -> tuple:
        """Check buying power requirements"""
        try:
            required_buying_power = trade_signal.total_value

            # Add buffer for options margin requirements
            if trade_signal.option_type:
                required_buying_power *= Decimal("1.2")  # 20% buffer for options

            # Check available buying power with buffer
            available_bp = account.buying_power * (Decimal("1") - self.min_buying_power_buffer)

            if required_buying_power > available_bp:
                # Calculate max affordable quantity
                max_affordable = available_bp / (
                    trade_signal.price
                    * (Decimal("1.2") if trade_signal.option_type else Decimal("1"))
                )
                max_quantity = int(max_affordable)

                return (
                    False,
                    RiskLevel.HIGH,
                    max(0, max_quantity),
                    f"Insufficient buying power (need ${required_buying_power: ,.2f}, have ${available_bp: ,.2f})",
                )

            return (True, RiskLevel.LOW, trade_signal.quantity, None)

        except Exception as e:
            logger.error(f"Buying power check failed: {e}")
            return (False, RiskLevel.CRITICAL, 0, f"Buying power check error: {e}")

    async def _check_daily_loss_limits(
        self, trade_signal: TradeSignal, account: AccountSnapshot
    ) -> tuple:
        """Check daily loss limits"""
        try:
            # This would typically track daily P & L
            # For now, using a simple unrealized P & L check
            current_unrealized = account.equity - account.cash
            starting_equity = account.portfolio_value  # Simplified

            daily_loss_pct = (starting_equity - current_unrealized) / starting_equity

            if daily_loss_pct > self.max_daily_loss:
                return (
                    False,
                    RiskLevel.CRITICAL,
                    0,
                    f"Daily loss limit exceeded: {daily_loss_pct:.1%}  >  {self.max_daily_loss: .1%}",
                )

            return (True, RiskLevel.LOW, trade_signal.quantity, None)

        except Exception as e:
            logger.error(f"Daily loss check failed: {e}")
            return (
                True,
                RiskLevel.LOW,
                trade_signal.quantity,
                None,
            )  # Allow trade on check failure

    async def _check_day_trading_limits(
        self, trade_signal: TradeSignal, account: AccountSnapshot
    ) -> tuple:
        """Check day trading rules"""
        try:
            # Check if account is PDT eligible
            is_pdt = account.equity >= self.pdt_threshold

            if not is_pdt and account.day_trade_count >= self.max_day_trades:
                return (
                    False,
                    RiskLevel.HIGH,
                    0,
                    f"Day trading limit reached: {account.day_trade_count}/{self.max_day_trades}",
                )

            return (True, RiskLevel.LOW, trade_signal.quantity, None)

        except Exception as e:
            logger.error(f"Day trading check failed: {e}")
            return (True, RiskLevel.LOW, trade_signal.quantity, None)

    async def _check_options_allocation(
        self, trade_signal: TradeSignal, account: AccountSnapshot, positions: list[PositionSummary]
    ) -> tuple:
        """Check options allocation limits"""
        try:
            # Only check for options trades
            if not trade_signal.option_type:
                return (True, RiskLevel.LOW, trade_signal.quantity, None)

            # Calculate current options exposure
            current_options_value = sum(
                abs(pos.market_value) for pos in positions if pos.position_type == "option"
            )

            # Add proposed trade
            total_options_value = current_options_value + trade_signal.total_value
            options_percentage = total_options_value / account.portfolio_value

            if options_percentage > self.max_options_allocation:
                # Calculate max allowed options quantity
                max_options_value = account.portfolio_value * self.max_options_allocation
                available_options_capacity = max_options_value - current_options_value
                max_quantity = (
                    int(available_options_capacity / trade_signal.price)
                    if trade_signal.price > 0
                    else 0
                )

                return (
                    False,
                    RiskLevel.HIGH,
                    max(0, max_quantity),
                    f"Options allocation would exceed {self.max_options_allocation: .1%} limit",
                )

            return (True, RiskLevel.LOW, trade_signal.quantity, None)

        except Exception as e:
            logger.error(f"Options allocation check failed: {e}")
            return (True, RiskLevel.LOW, trade_signal.quantity, None)

    async def _check_correlation_risk(
        self, trade_signal: TradeSignal, positions: list[PositionSummary]
    ) -> tuple:
        """Check correlation risk (simplified sector / correlation analysis)"""
        try:
            # Simplified: check for excessive exposure to same ticker
            same_ticker_exposure = sum(
                abs(pos.market_value) for pos in positions if pos.ticker == trade_signal.ticker
            )

            # This is a simplified check - in practice would use sector / correlation data
            total_exposure = same_ticker_exposure + trade_signal.total_value

            # For now, just check if more than 10% in same ticker
            max_single_ticker = Decimal("0.10")  # 10% max in single ticker

            if total_exposure > max_single_ticker:
                return (
                    False,
                    RiskLevel.MODERATE,
                    trade_signal.quantity // 2,  # Reduce by half
                    f"Excessive concentration in {trade_signal.ticker}",
                )

            return (True, RiskLevel.LOW, trade_signal.quantity, None)

        except Exception as e:
            logger.error(f"Correlation risk check failed: {e}")
            return (True, RiskLevel.LOW, trade_signal.quantity, None)

    async def _check_volatility_risk(self, trade_signal: TradeSignal) -> tuple:
        """Check volatility - based risk"""
        try:
            # This would typically use historical volatility data
            # For now, apply conservative limits to options
            if trade_signal.option_type and trade_signal.quantity > 10:
                return (
                    True,
                    RiskLevel.MODERATE,
                    min(10, trade_signal.quantity),  # Max 10 contracts
                    "Options quantity limited for volatility risk",
                )

            return (True, RiskLevel.LOW, trade_signal.quantity, None)

        except Exception as e:
            logger.error(f"Volatility risk check failed: {e}")
            return (True, RiskLevel.LOW, trade_signal.quantity, None)

    async def _calculate_risk_metrics(
        self, trade_signal: TradeSignal, account: AccountSnapshot, positions: list[PositionSummary]
    ) -> dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        try:
            current_value = sum(abs(pos.market_value) for pos in positions)

            return {
                "current_portfolio_value": float(account.portfolio_value),
                "current_risk_exposure": float(current_value),
                "proposed_trade_value": float(trade_signal.total_value),
                "portfolio_risk_percentage": float(current_value / account.portfolio_value),
                "buying_power_utilization": float(
                    (account.portfolio_value - account.buying_power) / account.portfolio_value
                ),
                "options_allocation": float(
                    sum(abs(pos.market_value) for pos in positions if pos.position_type == "option")
                    / account.portfolio_value
                ),
                "cash_percentage": float(account.cash / account.portfolio_value),
                "day_trades_remaining": max(0, self.max_day_trades - account.day_trade_count)
                if account.equity < self.pdt_threshold
                else 999,
            }
        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {e}")
            return {}

    async def _log_validation_result(self, trade_signal: TradeSignal, result: RiskValidationResult):
        """Log risk validation results for monitoring"""
        try:
            log_entry = {
                "timestamp": result.timestamp.isoformat(),
                "ticker": trade_signal.ticker,
                "strategy": trade_signal.strategy,
                "original_quantity": result.original_quantity,
                "approved_quantity": result.approved_quantity,
                "validation_result": result.validation_result.value,
                "risk_level": result.risk_level.value,
                "reason": result.reason,
                "warnings_count": len(result.warnings),
                "risk_metrics": result.risk_metrics,
            }

            logger.info(f"Risk validation result: {log_entry}")

            # Store for analysis
            self.risk_events.append(log_entry)

            # Keep only last 1000 events
            if len(self.risk_events) > 1000:
                self.risk_events = self.risk_events[-1000:]

        except Exception as e:
            logger.error(f"Failed to log validation result: {e}")

    async def emergency_halt(self, reason: str):
        """Emergency halt all trading activities"""
        try:
            logger.critical(f"EMERGENCY HALT TRIGGERED: {reason}")

            # This would integrate with the main trading system to halt operations
            # For now, just log the critical event

            halt_event = {
                "timestamp": datetime.now().isoformat(),
                "reason": reason,
                "type": "EMERGENCY_HALT",
                "triggered_by": "RealTimeRiskManager",
            }

            self.risk_events.append(halt_event)

            # In a real system, this would:
            # 1. Cancel all pending orders
            # 2. Close risky positions
            # 3. Send alerts to administrators
            # 4. Set system status to halted

        except Exception as e:
            logger.error(f"Emergency halt procedure failed: {e}")

    async def get_risk_summary(self) -> dict[str, Any]:
        """Get current risk summary"""
        try:
            account = await self._get_current_account()
            positions = await self._get_current_positions()

            total_exposure = sum(abs(pos.market_value) for pos in positions)
            options_exposure = sum(
                abs(pos.market_value) for pos in positions if pos.position_type == "option"
            )

            return {
                "timestamp": datetime.now().isoformat(),
                "portfolio_value": float(account.portfolio_value),
                "total_risk_exposure": float(total_exposure),
                "risk_percentage": float(total_exposure / account.portfolio_value)
                if account.portfolio_value > 0
                else 0,
                "options_allocation": float(options_exposure / account.portfolio_value)
                if account.portfolio_value > 0
                else 0,
                "buying_power": float(account.buying_power),
                "day_trades_used": account.day_trade_count,
                "positions_count": len(positions),
                "recent_validations": len(
                    [
                        e
                        for e in self.risk_events
                        if datetime.fromisoformat(e["timestamp"])
                        > datetime.now() - timedelta(hours=1)
                    ]
                ),
            }
        except Exception as e:
            logger.error(f"Risk summary generation failed: {e}")
            return {"error": str(e)}


def create_risk_manager(alpaca_manager=None) -> RealTimeRiskManager:
    """Factory function to create risk manager"""
    return RealTimeRiskManager(alpaca_manager)
