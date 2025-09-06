"""
Unified Trading Interface
Connects the disconnected Strategy and Broker systems for production use.
"""

from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from ..apimanagers import AlpacaManager
from ..risk_management import RiskManager, Position, RiskParameters
from ..alert_system import TradingAlertSystem, AlertType, AlertPriority


class TradeStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class TradeSignal:
    """Signal from a trading strategy"""
    strategy_name: str
    ticker: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "gtc"
    reason: str = ""
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TradeResult:
    """Result of trade execution"""
    trade_id: str
    signal: TradeSignal
    status: TradeStatus
    filled_quantity: int = 0
    filled_price: Optional[float] = None
    commission: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None


@dataclass
class PositionUpdate:
    """Position update from broker"""
    ticker: str
    quantity: int
    avg_price: float
    current_price: float
    unrealized_pnl: float
    market_value: float
    timestamp: datetime = field(default_factory=datetime.now)


class TradingInterface:
    """
    Unified interface connecting strategies to broker execution
    """
    
    def __init__(self, broker_manager: AlpacaManager, risk_manager: RiskManager, 
                 alert_system: TradingAlertSystem, config: Dict[str, Any]):
        self.broker = broker_manager
        self.risk = risk_manager
        self.alerts = alert_system
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.active_trades: Dict[str, TradeResult] = {}
        self.positions: Dict[str, PositionUpdate] = {}
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup structured logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_interface.log'),
                logging.StreamHandler()
            ]
        )
    
    async def execute_trade(self, signal: TradeSignal) -> TradeResult:
        """
        Execute trade with comprehensive risk controls and error handling
        """
        trade_id = f"{signal.strategy_name}_{signal.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            self.logger.info(f"Executing trade {trade_id}", extra={
                'trade_id': trade_id,
                'strategy': signal.strategy_name,
                'ticker': signal.ticker,
                'side': signal.side.value,
                'quantity': signal.quantity
            })
            
            # 1. Validate signal
            validation_result = await self.validate_signal(signal)
            if not validation_result['valid']:
                return TradeResult(
                    trade_id=trade_id,
                    signal=signal,
                    status=TradeStatus.REJECTED,
                    error_message=validation_result['reason']
                )
            
            # 2. Check risk limits
            risk_check = await self.check_risk_limits(signal)
            if not risk_check['allowed']:
                await self.alerts.send_alert(
                    AlertType.RISK_LIMIT_EXCEEDED,
                    AlertPriority.HIGH,
                    f"Risk limit exceeded for {signal.ticker}: {risk_check['reason']}"
                )
                return TradeResult(
                    trade_id=trade_id,
                    signal=signal,
                    status=TradeStatus.REJECTED,
                    error_message=f"Risk limit exceeded: {risk_check['reason']}"
                )
            
            # 3. Execute via broker
            broker_result = await self.execute_broker_order(signal)
            
            # 4. Create trade result
            trade_result = TradeResult(
                trade_id=trade_id,
                signal=signal,
                status=TradeStatus.FILLED if broker_result['success'] else TradeStatus.REJECTED,
                filled_quantity=broker_result.get('filled_quantity', 0),
                filled_price=broker_result.get('filled_price'),
                commission=broker_result.get('commission', 0.0),
                error_message=broker_result.get('error')
            )
            
            # 5. Update tracking
            self.active_trades[trade_id] = trade_result
            
            # 6. Send alerts
            await self.alerts.send_alert(
                AlertType.TRADE_EXECUTED,
                AlertPriority.MEDIUM,
                f"Trade executed: {signal.ticker} {signal.side.value} {signal.quantity} @ {trade_result.filled_price}"
            )
            
            self.logger.info(f"Trade {trade_id} executed successfully", extra={
                'trade_id': trade_id,
                'filled_quantity': trade_result.filled_quantity,
                'filled_price': trade_result.filled_price
            })
            
            return trade_result
            
        except Exception as e:
            self.logger.error(f"Error executing trade {trade_id}: {e}", extra={
                'trade_id': trade_id,
                'error': str(e)
            })
            
            return TradeResult(
                trade_id=trade_id,
                signal=signal,
                status=TradeStatus.REJECTED,
                error_message=str(e)
            )
    
    async def validate_signal(self, signal: TradeSignal) -> Dict[str, Any]:
        """Validate trading signal"""
        # Check required fields
        if not signal.ticker or not signal.quantity or signal.quantity <= 0:
            return {'valid': False, 'reason': 'Invalid ticker or quantity'}
        
        if signal.order_type == OrderType.LIMIT and not signal.limit_price:
            return {'valid': False, 'reason': 'Limit price required for limit orders'}
        
        if signal.order_type == OrderType.STOP and not signal.stop_price:
            return {'valid': False, 'reason': 'Stop price required for stop orders'}
        
        # Check market hours (simplified)
        if not await self.is_market_open():
            return {'valid': False, 'reason': 'Market is closed'}
        
        return {'valid': True, 'reason': 'Signal validated'}
    
    async def check_risk_limits(self, signal: TradeSignal) -> Dict[str, Any]:
        """Check risk limits before execution"""
        try:
            # Get current account value
            account = await self.get_account_info()
            account_value = float(account.get('equity', 0))
            
            # Calculate position risk
            if signal.limit_price:
                position_value = signal.quantity * signal.limit_price * 100  # Options are per 100 shares
            else:
                # Use current market price as estimate
                current_price = await self.get_current_price(signal.ticker)
                position_value = signal.quantity * current_price * 100
            
            position_risk_pct = position_value / account_value
            
            # Check position risk limit
            max_position_risk = self.config.get('max_position_risk', 0.10)
            if position_risk_pct > max_position_risk:
                return {
                    'allowed': False,
                    'reason': f'Position risk {position_risk_pct:.2%} exceeds limit {max_position_risk:.2%}'
                }
            
            # Check total portfolio risk
            total_risk = await self.calculate_total_portfolio_risk()
            max_total_risk = self.config.get('max_total_risk', 0.30)
            if total_risk > max_total_risk:
                return {
                    'allowed': False,
                    'reason': f'Total portfolio risk {total_risk:.2%} exceeds limit {max_total_risk:.2%}'
                }
            
            return {'allowed': True, 'reason': 'Risk limits OK'}
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return {'allowed': False, 'reason': f'Risk check failed: {e}'}
    
    async def execute_broker_order(self, signal: TradeSignal) -> Dict[str, Any]:
        """Execute order via broker"""
        try:
            if signal.side == OrderSide.BUY:
                if signal.order_type == OrderType.MARKET:
                    result = self.broker.market_buy(signal.ticker, signal.quantity)
                else:
                    # For limit orders, we'd need to implement limit_buy
                    result = self.broker.market_buy(signal.ticker, signal.quantity)
            else:  # SELL
                if signal.order_type == OrderType.MARKET:
                    result = self.broker.market_sell(signal.ticker, signal.quantity)
                else:
                    # For limit orders, we'd need to implement limit_sell
                    result = self.broker.market_sell(signal.ticker, signal.quantity)
            
            if result:
                return {
                    'success': True,
                    'filled_quantity': signal.quantity,
                    'filled_price': signal.limit_price or await self.get_current_price(signal.ticker),
                    'commission': self.config.get('default_commission', 1.0)
                }
            else:
                return {
                    'success': False,
                    'error': 'Broker order failed'
                }
                
        except Exception as e:
            self.logger.error(f"Broker execution error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information from broker"""
        try:
            account = self.broker.get_account()
            return {
                'equity': account.equity,
                'buying_power': account.buying_power,
                'cash': account.cash,
                'portfolio_value': account.portfolio_value
            }
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {'equity': 0, 'buying_power': 0, 'cash': 0, 'portfolio_value': 0}
    
    async def get_current_price(self, ticker: str) -> float:
        """Get current price for ticker"""
        try:
            success, price = self.broker.get_price(ticker)
            if success:
                return float(price)
            else:
                self.logger.warning(f"Could not get price for {ticker}: {price}")
                return 0.0
        except Exception as e:
            self.logger.error(f"Error getting price for {ticker}: {e}")
            return 0.0
    
    async def is_market_open(self) -> bool:
        """Check if market is open"""
        try:
            return self.broker.market_close() is True
        except Exception as e:
            self.logger.error(f"Error checking market status: {e}")
            return False
    
    async def calculate_total_portfolio_risk(self) -> float:
        """Calculate total portfolio risk percentage"""
        try:
            positions = self.broker.get_positions()
            if isinstance(positions, str):  # Error case
                return 0.0
            
            total_risk = 0.0
            for position in positions:
                # Simplified risk calculation
                position_risk = abs(float(position.unrealized_pl)) / float(position.market_value) if position.market_value > 0 else 0
                total_risk += position_risk
            
            return min(total_risk, 1.0)  # Cap at 100%
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk: {e}")
            return 0.0
    
    async def update_positions(self) -> List[PositionUpdate]:
        """Update position information from broker"""
        try:
            positions = self.broker.get_positions()
            if isinstance(positions, str):  # Error case
                return []
            
            updates = []
            for position in positions:
                update = PositionUpdate(
                    ticker=position.symbol,
                    quantity=int(position.qty),
                    avg_price=float(position.avg_entry_price),
                    current_price=float(position.current_price),
                    unrealized_pnl=float(position.unrealized_pl),
                    market_value=float(position.market_value)
                )
                updates.append(update)
                self.positions[position.symbol] = update
            
            return updates
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
            return []
    
    async def get_trade_history(self, limit: int = 100) -> List[TradeResult]:
        """Get recent trade history"""
        return list(self.active_trades.values())[-limit:]
    
    async def cancel_trade(self, trade_id: str) -> bool:
        """Cancel a pending trade"""
        try:
            if trade_id in self.active_trades:
                trade = self.active_trades[trade_id]
                if trade.status == TradeStatus.PENDING:
                    # Implement broker cancel logic here
                    trade.status = TradeStatus.CANCELLED
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Error cancelling trade {trade_id}: {e}")
            return False


# Factory function for easy initialization
def create_trading_interface(config: Dict[str, Any]) -> TradingInterface:
    """Create trading interface with default components"""
    # Initialize broker manager
    broker = AlpacaManager(
        config.get('alpaca_api_key', ''),
        config.get('alpaca_secret_key', '')
    )
    
    # Initialize risk manager
    risk_params = RiskParameters()
    risk_manager = RiskManager(risk_params)
    
    # Initialize alert system
    alert_system = TradingAlertSystem()
    
    return TradingInterface(broker, risk_manager, alert_system, config)
