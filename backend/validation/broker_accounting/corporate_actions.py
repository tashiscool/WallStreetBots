"""Corporate actions and earnings calendar management.

Provides guaranteed, versioned corporate actions feed and enforces
pre-trade blackout rules at the adapter boundary.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
import logging
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from enum import Enum
import json
import asyncio

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of corporate actions."""
    DIVIDEND = "dividend"
    STOCK_SPLIT = "stock_split"
    STOCK_DIVIDEND = "stock_dividend"
    SPIN_OFF = "spin_off"
    MERGER = "merger"
    RIGHTS_OFFERING = "rights_offering"
    SPECIAL_DIVIDEND = "special_dividend"
    REVERSE_SPLIT = "reverse_split"


class EarningsTime(Enum):
    """Earnings announcement timing."""
    BEFORE_MARKET = "before_market"
    AFTER_MARKET = "after_market"
    DURING_MARKET = "during_market"
    TIME_NOT_SUPPLIED = "time_not_supplied"


@dataclass
class CorporateAction:
    """Corporate action record."""
    symbol: str
    action_type: ActionType
    announcement_date: date
    ex_date: date
    record_date: date
    payable_date: Optional[date]
    value: float  # Dividend amount, split ratio, etc.
    description: str
    data_source: str
    version: int
    created_at: datetime
    updated_at: datetime
    confirmed: bool = False
    adjustment_factor: float = 1.0  # Price adjustment factor


@dataclass
class EarningsEvent:
    """Earnings announcement event."""
    symbol: str
    earnings_date: date
    announcement_time: EarningsTime
    fiscal_year: int
    fiscal_quarter: int
    estimated_eps: Optional[float]
    data_source: str
    version: int
    created_at: datetime
    updated_at: datetime
    confirmed: bool = False


class CorporateActionsManager:
    """Manages corporate actions data with versioning and validation."""

    def __init__(self):
        self.actions = {}  # symbol -> List[CorporateAction]
        self.actions_by_date = {}  # date -> List[CorporateAction]
        self.version_counter = 1
        self.data_sources = set()

    def add_corporate_action(self, action: CorporateAction) -> bool:
        """Add or update a corporate action."""
        try:
            # Validate action
            validation_result = self._validate_action(action)
            if not validation_result['valid']:
                logger.error(f"Invalid corporate action: {validation_result['errors']}")
                return False

            # Set version if not provided
            if action.version == 0:
                action.version = self.version_counter
                self.version_counter += 1

            # Add to symbol index
            if action.symbol not in self.actions:
                self.actions[action.symbol] = []

            # Check for duplicates/updates
            existing_action = self._find_existing_action(action)
            if existing_action:
                # Update existing action
                existing_action.value = action.value
                existing_action.description = action.description
                existing_action.updated_at = datetime.now()
                existing_action.version = self.version_counter
                self.version_counter += 1
                logger.info(f"Updated corporate action for {action.symbol}: {action.action_type.value}")
            else:
                # Add new action
                action.created_at = datetime.now()
                action.updated_at = datetime.now()
                self.actions[action.symbol].append(action)
                logger.info(f"Added corporate action for {action.symbol}: {action.action_type.value}")

            # Add to date index
            ex_date = action.ex_date
            if ex_date not in self.actions_by_date:
                self.actions_by_date[ex_date] = []

            if action not in self.actions_by_date[ex_date]:
                self.actions_by_date[ex_date].append(action)

            # Track data source
            self.data_sources.add(action.data_source)

            return True

        except Exception as e:
            logger.error(f"Failed to add corporate action: {e}")
            return False

    def get_actions_for_symbol(self, symbol: str,
                              start_date: Optional[date] = None,
                              end_date: Optional[date] = None) -> List[CorporateAction]:
        """Get corporate actions for a symbol within date range."""
        if symbol not in self.actions:
            return []

        actions = self.actions[symbol]

        if start_date or end_date:
            filtered_actions = []
            for action in actions:
                if start_date and action.ex_date < start_date:
                    continue
                if end_date and action.ex_date > end_date:
                    continue
                filtered_actions.append(action)
            return filtered_actions

        return actions.copy()

    def get_actions_for_date(self, target_date: date) -> List[CorporateAction]:
        """Get all corporate actions for a specific date."""
        return self.actions_by_date.get(target_date, []).copy()

    def get_upcoming_actions(self, days_ahead: int = 30) -> List[CorporateAction]:
        """Get upcoming corporate actions within specified days."""
        today = date.today()
        end_date = today + timedelta(days=days_ahead)

        upcoming = []
        current_date = today
        while current_date <= end_date:
            upcoming.extend(self.get_actions_for_date(current_date))
            current_date += timedelta(days=1)

        return upcoming

    def calculate_adjustment_factor(self, symbol: str, price_date: date,
                                  target_date: date) -> float:
        """Calculate cumulative price adjustment factor between two dates."""
        if price_date >= target_date:
            return 1.0

        actions = self.get_actions_for_symbol(symbol, price_date, target_date)
        adjustment_factor = 1.0

        for action in actions:
            if action.ex_date > price_date and action.ex_date <= target_date:
                if action.action_type == ActionType.DIVIDEND:
                    # For dividends, factor is typically small
                    # This is simplified - actual adjustment depends on stock price
                    continue  # Dividends don't adjust price significantly

                elif action.action_type == ActionType.STOCK_SPLIT:
                    # For splits, multiply by split ratio
                    adjustment_factor *= action.value

                elif action.action_type == ActionType.REVERSE_SPLIT:
                    # For reverse splits, divide by ratio
                    adjustment_factor /= action.value

                elif action.action_type == ActionType.STOCK_DIVIDEND:
                    # Stock dividends adjust by percentage
                    adjustment_factor *= (1 + action.value / 100)

        return adjustment_factor

    def _validate_action(self, action: CorporateAction) -> Dict[str, Any]:
        """Validate corporate action data."""
        errors = []

        # Date validation
        if action.ex_date < action.announcement_date:
            errors.append("Ex-date cannot be before announcement date")

        if action.record_date and action.record_date < action.ex_date:
            errors.append("Record date cannot be before ex-date")

        if action.payable_date and action.record_date and action.payable_date < action.record_date:
            errors.append("Payable date cannot be before record date")

        # Value validation
        if action.action_type in [ActionType.DIVIDEND, ActionType.SPECIAL_DIVIDEND]:
            if action.value <= 0:
                errors.append("Dividend value must be positive")

        elif action.action_type in [ActionType.STOCK_SPLIT, ActionType.REVERSE_SPLIT]:
            if action.value <= 0:
                errors.append("Split ratio must be positive")

        # Symbol validation
        if not action.symbol or len(action.symbol.strip()) == 0:
            errors.append("Symbol cannot be empty")

        return {
            'valid': len(errors) == 0,
            'errors': errors
        }

    def _find_existing_action(self, action: CorporateAction) -> Optional[CorporateAction]:
        """Find existing action that matches this one."""
        if action.symbol not in self.actions:
            return None

        for existing in self.actions[action.symbol]:
            if (existing.action_type == action.action_type and
                existing.ex_date == action.ex_date):
                return existing

        return None


class EarningsCalendarManager:
    """Manages earnings calendar with blackout enforcement."""

    def __init__(self):
        self.earnings_events = {}  # symbol -> List[EarningsEvent]
        self.earnings_by_date = {}  # date -> List[EarningsEvent]
        self.version_counter = 1
        self.blackout_rules = {
            'default_days_before': 1,
            'default_days_after': 1,
            'extended_blackout_symbols': set(),  # Symbols with longer blackouts
            'no_blackout_symbols': set()  # Symbols exempt from blackouts
        }

    def add_earnings_event(self, event: EarningsEvent) -> bool:
        """Add or update an earnings event."""
        try:
            # Validate event
            if not self._validate_earnings_event(event):
                return False

            # Set version if not provided
            if event.version == 0:
                event.version = self.version_counter
                self.version_counter += 1

            # Add to symbol index
            if event.symbol not in self.earnings_events:
                self.earnings_events[event.symbol] = []

            # Check for duplicates
            existing_event = self._find_existing_earnings(event)
            if existing_event:
                # Update existing
                existing_event.announcement_time = event.announcement_time
                existing_event.estimated_eps = event.estimated_eps
                existing_event.updated_at = datetime.now()
                existing_event.version = self.version_counter
                self.version_counter += 1
            else:
                # Add new
                event.created_at = datetime.now()
                event.updated_at = datetime.now()
                self.earnings_events[event.symbol].append(event)

            # Add to date index
            earnings_date = event.earnings_date
            if earnings_date not in self.earnings_by_date:
                self.earnings_by_date[earnings_date] = []

            if event not in self.earnings_by_date[earnings_date]:
                self.earnings_by_date[earnings_date].append(event)

            logger.info(f"Added earnings event for {event.symbol}: {event.earnings_date}")
            return True

        except Exception as e:
            logger.error(f"Failed to add earnings event: {e}")
            return False

    def is_in_blackout_period(self, symbol: str, check_date: date) -> Tuple[bool, str]:
        """Check if symbol is in earnings blackout period."""
        if symbol in self.blackout_rules['no_blackout_symbols']:
            return False, "Symbol exempt from blackout"

        earnings = self.get_earnings_for_symbol(symbol)

        for event in earnings:
            earnings_date = event.earnings_date

            # Determine blackout window
            if symbol in self.blackout_rules['extended_blackout_symbols']:
                days_before = 3
                days_after = 2
            else:
                days_before = self.blackout_rules['default_days_before']
                days_after = self.blackout_rules['default_days_after']

            blackout_start = earnings_date - timedelta(days=days_before)
            blackout_end = earnings_date + timedelta(days=days_after)

            if blackout_start <= check_date <= blackout_end:
                return True, f"Earnings blackout: {earnings_date} ({event.announcement_time.value})"

        return False, "Not in blackout period"

    def get_earnings_for_symbol(self, symbol: str,
                               start_date: Optional[date] = None,
                               end_date: Optional[date] = None) -> List[EarningsEvent]:
        """Get earnings events for a symbol within date range."""
        if symbol not in self.earnings_events:
            return []

        events = self.earnings_events[symbol]

        if start_date or end_date:
            filtered_events = []
            for event in events:
                if start_date and event.earnings_date < start_date:
                    continue
                if end_date and event.earnings_date > end_date:
                    continue
                filtered_events.append(event)
            return filtered_events

        return events.copy()

    def get_upcoming_earnings(self, days_ahead: int = 7) -> List[EarningsEvent]:
        """Get upcoming earnings within specified days."""
        today = date.today()
        end_date = today + timedelta(days=days_ahead)

        upcoming = []
        current_date = today
        while current_date <= end_date:
            upcoming.extend(self.earnings_by_date.get(current_date, []))
            current_date += timedelta(days=1)

        return upcoming

    def update_blackout_rules(self, rules: Dict[str, Any]):
        """Update blackout rules configuration."""
        if 'default_days_before' in rules:
            self.blackout_rules['default_days_before'] = rules['default_days_before']

        if 'default_days_after' in rules:
            self.blackout_rules['default_days_after'] = rules['default_days_after']

        if 'extended_blackout_symbols' in rules:
            self.blackout_rules['extended_blackout_symbols'].update(rules['extended_blackout_symbols'])

        if 'no_blackout_symbols' in rules:
            self.blackout_rules['no_blackout_symbols'].update(rules['no_blackout_symbols'])

        logger.info(f"Updated blackout rules: {self.blackout_rules}")

    def _validate_earnings_event(self, event: EarningsEvent) -> bool:
        """Validate earnings event data."""
        if not event.symbol or len(event.symbol.strip()) == 0:
            logger.error("Earnings event symbol cannot be empty")
            return False

        if event.earnings_date < date.today() - timedelta(days=365):
            logger.error("Earnings date too far in the past")
            return False

        if event.fiscal_quarter not in [1, 2, 3, 4]:
            logger.error("Invalid fiscal quarter")
            return False

        return True

    def _find_existing_earnings(self, event: EarningsEvent) -> Optional[EarningsEvent]:
        """Find existing earnings event that matches this one."""
        if event.symbol not in self.earnings_events:
            return None

        for existing in self.earnings_events[event.symbol]:
            if (existing.earnings_date == event.earnings_date and
                existing.fiscal_year == event.fiscal_year and
                existing.fiscal_quarter == event.fiscal_quarter):
                return existing

        return None


class TradingBlackoutEnforcer:
    """Enforces trading blackouts at the adapter boundary."""

    def __init__(self, corporate_actions_manager: CorporateActionsManager,
                 earnings_calendar: EarningsCalendarManager):
        self.ca_manager = corporate_actions_manager
        self.earnings_calendar = earnings_calendar
        self.blocked_orders = []
        self.whitelist_overrides = set()  # Orders that can bypass blackouts

    def check_order_allowed(self, symbol: str, order_id: str,
                           order_date: Optional[date] = None) -> Tuple[bool, str]:
        """Check if order is allowed given blackout rules."""
        if order_date is None:
            order_date = date.today()

        # Check whitelist override
        if order_id in self.whitelist_overrides:
            return True, "Whitelist override active"

        # Check earnings blackout
        in_earnings_blackout, earnings_reason = self.earnings_calendar.is_in_blackout_period(
            symbol, order_date
        )

        if in_earnings_blackout:
            self._log_blocked_order(symbol, order_id, "earnings_blackout", earnings_reason)
            return False, earnings_reason

        # Check corporate actions blackout
        upcoming_actions = self.ca_manager.get_actions_for_symbol(
            symbol, order_date, order_date + timedelta(days=5)
        )

        for action in upcoming_actions:
            days_until_ex = (action.ex_date - order_date).days

            # Block trading within 1 day of ex-date for significant actions
            if (days_until_ex <= 1 and
                action.action_type in [ActionType.STOCK_SPLIT, ActionType.REVERSE_SPLIT,
                                     ActionType.SPIN_OFF, ActionType.MERGER]):
                reason = f"Corporate action blackout: {action.action_type.value} ex-date {action.ex_date}"
                self._log_blocked_order(symbol, order_id, "corporate_action_blackout", reason)
                return False, reason

        return True, "Order allowed"

    def add_whitelist_override(self, order_id: str, reason: str):
        """Add order to whitelist (for emergency trading)."""
        self.whitelist_overrides.add(order_id)
        logger.warning(f"Added whitelist override for order {order_id}: {reason}")

    def remove_whitelist_override(self, order_id: str):
        """Remove order from whitelist."""
        self.whitelist_overrides.discard(order_id)

    def get_blackout_summary(self, symbols: List[str]) -> Dict[str, Any]:
        """Get blackout status summary for multiple symbols."""
        today = date.today()
        summary = {
            'check_date': today,
            'symbols_checked': len(symbols),
            'symbols_in_blackout': 0,
            'blackout_details': {},
            'upcoming_blackouts': []
        }

        for symbol in symbols:
            # Current blackout status
            order_allowed, reason = self.check_order_allowed(symbol, f"check_{symbol}")

            if not order_allowed:
                summary['symbols_in_blackout'] += 1
                summary['blackout_details'][symbol] = {
                    'in_blackout': True,
                    'reason': reason
                }
            else:
                summary['blackout_details'][symbol] = {
                    'in_blackout': False,
                    'reason': reason
                }

            # Upcoming blackouts (next 7 days)
            for days_ahead in range(1, 8):
                future_date = today + timedelta(days=days_ahead)
                future_allowed, future_reason = self.check_order_allowed(symbol, f"future_{symbol}", future_date)

                if not future_allowed:
                    summary['upcoming_blackouts'].append({
                        'symbol': symbol,
                        'date': future_date,
                        'reason': future_reason
                    })
                    break  # Only report first upcoming blackout per symbol

        return summary

    def _log_blocked_order(self, symbol: str, order_id: str, block_type: str, reason: str):
        """Log blocked order for audit trail."""
        blocked_record = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'order_id': order_id,
            'block_type': block_type,
            'reason': reason
        }

        self.blocked_orders.append(blocked_record)

        # Keep only last 1000 blocked orders
        if len(self.blocked_orders) > 1000:
            self.blocked_orders = self.blocked_orders[-1000:]

        logger.warning(f"BLOCKED ORDER: {symbol} {order_id} - {reason}")

    def get_blocked_orders_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get report of recently blocked orders."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_blocks = [
            block for block in self.blocked_orders
            if block['timestamp'] >= cutoff_time
        ]

        if not recent_blocks:
            return {
                'period_hours': hours,
                'total_blocked': 0,
                'blocks_by_type': {},
                'blocks_by_symbol': {},
                'recent_blocks': []
            }

        # Group by type
        blocks_by_type = {}
        for block in recent_blocks:
            block_type = block['block_type']
            if block_type not in blocks_by_type:
                blocks_by_type[block_type] = 0
            blocks_by_type[block_type] += 1

        # Group by symbol
        blocks_by_symbol = {}
        for block in recent_blocks:
            symbol = block['symbol']
            if symbol not in blocks_by_symbol:
                blocks_by_symbol[symbol] = 0
            blocks_by_symbol[symbol] += 1

        return {
            'period_hours': hours,
            'total_blocked': len(recent_blocks),
            'blocks_by_type': blocks_by_type,
            'blocks_by_symbol': blocks_by_symbol,
            'recent_blocks': recent_blocks[-20:]  # Last 20 blocks
        }


class CorporateActionsDataFeed:
    """Data feed interface for corporate actions and earnings."""

    def __init__(self, ca_manager: CorporateActionsManager,
                 earnings_manager: EarningsCalendarManager):
        self.ca_manager = ca_manager
        self.earnings_manager = earnings_manager
        self.data_sources = {}  # source_name -> config

    def register_data_source(self, source_name: str, config: Dict[str, Any]):
        """Register a data source for corporate actions."""
        self.data_sources[source_name] = config
        logger.info(f"Registered data source: {source_name}")

    async def sync_corporate_actions(self, source_name: str, symbols: List[str]) -> Dict[str, Any]:
        """Sync corporate actions from external data source."""
        if source_name not in self.data_sources:
            return {'error': f'Data source {source_name} not registered'}

        logger.info(f"Syncing corporate actions from {source_name} for {len(symbols)} symbols")

        sync_results = {
            'source': source_name,
            'symbols_processed': 0,
            'actions_added': 0,
            'actions_updated': 0,
            'errors': []
        }

        try:
            # In production, this would call actual data provider APIs
            # For simulation, generate sample data
            for symbol in symbols[:5]:  # Limit for demo
                sample_actions = self._generate_sample_corporate_actions(symbol, source_name)

                for action in sample_actions:
                    success = self.ca_manager.add_corporate_action(action)
                    if success:
                        sync_results['actions_added'] += 1
                    else:
                        sync_results['errors'].append(f"Failed to add action for {symbol}")

                sync_results['symbols_processed'] += 1

        except Exception as e:
            sync_results['errors'].append(f"Sync failed: {e!s}")

        logger.info(f"Corporate actions sync completed: {sync_results}")
        return sync_results

    async def sync_earnings_calendar(self, source_name: str, days_ahead: int = 30) -> Dict[str, Any]:
        """Sync earnings calendar from external data source."""
        if source_name not in self.data_sources:
            return {'error': f'Data source {source_name} not registered'}

        logger.info(f"Syncing earnings calendar from {source_name} for next {days_ahead} days")

        sync_results = {
            'source': source_name,
            'events_added': 0,
            'events_updated': 0,
            'errors': []
        }

        try:
            # Generate sample earnings events
            sample_events = self._generate_sample_earnings_events(source_name, days_ahead)

            for event in sample_events:
                success = self.earnings_manager.add_earnings_event(event)
                if success:
                    sync_results['events_added'] += 1
                else:
                    sync_results['errors'].append(f"Failed to add earnings for {event.symbol}")

        except Exception as e:
            sync_results['errors'].append(f"Earnings sync failed: {e!s}")

        logger.info(f"Earnings calendar sync completed: {sync_results}")
        return sync_results

    def _generate_sample_corporate_actions(self, symbol: str, source: str) -> List[CorporateAction]:
        """Generate sample corporate actions for testing."""
        actions = []

        # Sample dividend
        if np.random.random() > 0.7:  # 30% chance of dividend
            dividend_date = date.today() + timedelta(days=np.random.randint(10, 60))
            actions.append(CorporateAction(
                symbol=symbol,
                action_type=ActionType.DIVIDEND,
                announcement_date=dividend_date - timedelta(days=30),
                ex_date=dividend_date,
                record_date=dividend_date + timedelta(days=2),
                payable_date=dividend_date + timedelta(days=30),
                value=round(np.random.uniform(0.25, 2.0), 2),
                description="Quarterly dividend",
                data_source=source,
                version=0,
                created_at=datetime.now(),
                updated_at=datetime.now()
            ))

        # Sample stock split (rare)
        if np.random.random() > 0.95:  # 5% chance of split
            split_date = date.today() + timedelta(days=np.random.randint(20, 90))
            actions.append(CorporateAction(
                symbol=symbol,
                action_type=ActionType.STOCK_SPLIT,
                announcement_date=split_date - timedelta(days=30),
                ex_date=split_date,
                record_date=split_date,
                payable_date=None,
                value=2.0,  # 2:1 split
                description="2-for-1 stock split",
                data_source=source,
                version=0,
                created_at=datetime.now(),
                updated_at=datetime.now()
            ))

        return actions

    def _generate_sample_earnings_events(self, source: str, days_ahead: int) -> List[EarningsEvent]:
        """Generate sample earnings events for testing."""
        events = []

        # Common symbols with upcoming earnings
        sample_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']

        for symbol in sample_symbols:
            if np.random.random() > 0.6:  # 40% chance of earnings in period
                earnings_date = date.today() + timedelta(days=np.random.randint(1, days_ahead))

                events.append(EarningsEvent(
                    symbol=symbol,
                    earnings_date=earnings_date,
                    announcement_time=np.random.choice(list(EarningsTime)),
                    fiscal_year=2024,
                    fiscal_quarter=np.random.randint(1, 5),
                    estimated_eps=round(np.random.uniform(1.0, 5.0), 2),
                    data_source=source,
                    version=0,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                ))

        return events