"""
Production Wheel Strategy Implementation
Automated premium selling with real broker integration
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import math

from .trading_interface import TradingInterface, TradeSignal, OrderType, OrderSide
from .data_providers import UnifiedDataProvider, MarketData, OptionsData
from .production_config import ProductionConfig
from .production_logging import ProductionLogger, ErrorHandler, MetricsCollector
from .production_models import Strategy, Position, Trade, RiskLimit


class WheelStage(Enum):
    """Wheel strategy stages"""
    CASH_SECURED_PUT="cash_secured_put"
    ASSIGNED_STOCK = "assigned_stock"
    COVERED_CALL = "covered_call"
    CLOSED_POSITION = "closed_position"


class WheelStatus(Enum):
    """Wheel position status"""
    ACTIVE="active"
    EXPIRED = "expired"
    ASSIGNED = "assigned"
    CLOSED = "closed"
    ROLLED = "rolled"


@dataclass
class WheelPosition:
    """Wheel strategy position tracking"""
    ticker: str
    stage: WheelStage
    status: WheelStatus
    
    # Position details
    quantity: int
    entry_price: float
    current_price: float
    unrealized_pnl: float
    
    # Option details
    option_type: str  # 'put' or 'call'
    strike_price: float
    expiry_date: datetime
    premium_received: float
    premium_paid: float = 0.0
    
    # Timing
    entry_date: datetime = field(default_factory=datetime.now)
    last_update: datetime=field(default_factory=datetime.now)
    
    # Metadata
    days_to_expiry: int=0
    delta: float = 0.0
    theta: float = 0.0
    iv_rank: float = 0.0
    
    def update_pricing(self, market_data: MarketData, options_data: List[OptionsData]):
        """Update position with current market data"""
        self.current_price=market_data.price
        self.unrealized_pnl = self.calculate_unrealized_pnl()
        self.last_update=datetime.now()
        
        # Update option Greeks if available
        for option in options_data:
            if (option.strike== self.strike_price and 
                option.option_type == self.option_type and
                option.expiry_date == self.expiry_date.strftime('%Y-%m-%d')):
                self.delta=option.delta
                self.theta = option.theta
                break
    
    def calculate_unrealized_pnl(self) -> float:
        """Calculate unrealized P&L"""
        if self.stage== WheelStage.CASH_SECURED_PUT:
            # For puts, profit if stock stays above strike
            if self.current_price >= self.strike_price:
                return self.premium_received
            else:
                # Loss if assigned (stock drops below strike)
                loss=(self.strike_price - self.current_price) * self.quantity
                return self.premium_received - loss
        
        elif self.stage== WheelStage.ASSIGNED_STOCK:
            # For assigned stock, profit/loss based on stock movement
            stock_pnl=(self.current_price - self.entry_price) * self.quantity
            return stock_pnl + self.premium_received
        
        elif self.stage== WheelStage.COVERED_CALL:
            # For covered calls, profit if stock stays below strike
            stock_pnl=(self.current_price - self.entry_price) * self.quantity
            call_pnl=self.premium_received - self.premium_paid
            
            if self.current_price >= self.strike_price:
                # Call will be assigned, stock sold at strike
                assignment_pnl=(self.strike_price - self.entry_price) * self.quantity
                return assignment_pnl + call_pnl
            else:
                # Call expires worthless
                return stock_pnl + call_pnl
        
        return 0.0
    
    def calculate_days_to_expiry(self) -> int:
        """Calculate days to expiry"""
        if self.expiry_date:
            delta=self.expiry_date - datetime.now()
            return max(0, delta.days)
        return 0


@dataclass
class WheelCandidate:
    """Wheel strategy candidate screening"""
    ticker: str
    current_price: float
    volatility_rank: float
    earnings_date: Optional[datetime] = None
    earnings_risk: float=0.0
    
    # Technical indicators
    rsi: float = 50.0
    support_level: float = 0.0
    resistance_level: float = 0.0
    
    # Option metrics
    put_premium: float = 0.0
    call_premium: float = 0.0
    iv_rank: float = 0.0
    
    # Scoring
    wheel_score: float = 0.0
    risk_score: float = 0.0
    
    def calculate_wheel_score(self) -> float:
        """Calculate wheel strategy score"""
        score=0.0
        
        # Volatility rank (higher is better for premium)
        score += self.volatility_rank * 0.3
        
        # IV rank (higher is better for premium)
        score += self.iv_rank * 0.2
        
        # Put premium (higher is better)
        score += min(self.put_premium / self.current_price, 0.05) * 100
        
        # Earnings risk (lower is better)
        score -= self.earnings_risk * 0.2
        
        # Technical score (RSI not overbought)
        if 30 <= self.rsi <= 70:
            score += 0.1
        
        self.wheel_score=max(0.0, min(1.0, score))
        return self.wheel_score


class ProductionWheelStrategy:
    """Production Wheel Strategy Implementation"""
    
    def __init__(self, 
                 trading_interface: TradingInterface,
                 data_provider: UnifiedDataProvider,
                 config: ProductionConfig,
                 logger: ProductionLogger):
        self.trading=trading_interface
        self.data = data_provider
        self.config = config
        self.logger = logger
        self.error_handler = ErrorHandler(logger)
        self.metrics=MetricsCollector(logger)
        
        # Strategy parameters
        self.max_positions=config.trading.max_concurrent_trades
        self.max_position_size = config.risk.max_position_risk
        self.target_delta = 0.30  # Target delta for puts
        self.profit_target = 0.50  # Close at 50% profit
        self.roll_threshold = 0.20  # Roll at 20% loss
        
        # Position tracking
        self.positions: Dict[str, WheelPosition] = {}
        self.candidates: List[WheelCandidate] = []
        
        # Strategy state
        self.last_scan_time: Optional[datetime] = None
        self.scan_interval=timedelta(minutes=30)
        
        self.logger.info("Wheel Strategy initialized", 
                        max_positions=self.max_positions,
                        target_delta=self.target_delta)
    
    async def scan_for_opportunities(self) -> List[WheelCandidate]:
        """Scan for wheel strategy opportunities"""
        self.logger.info("Scanning for wheel opportunities")
        
        try:
            candidates=[]
            
            # Get universe of stocks
            universe = self.config.trading.universe
            
            for ticker in universe:
                try:
                    candidate = await self._analyze_ticker(ticker)
                    if candidate and candidate.wheel_score > 0.6:
                        candidates.append(candidate)
                        self.logger.info(f"Wheel candidate: {ticker}", 
                                       score=candidate.wheel_score,
                                       iv_rank=candidate.iv_rank)
                
                except Exception as e:
                    self.error_handler.handle_error(e, {"ticker":ticker, "operation":"scan"})
                    continue
            
            # Sort by wheel score
            candidates.sort(key=lambda x: x.wheel_score, reverse=True)
            
            self.candidates=candidates[:10]  # Top 10 candidates
            
            self.logger.info(f"Found {len(self.candidates)} wheel candidates")
            self.metrics.record_metric("wheel_candidates_found", len(self.candidates))
            
            return self.candidates
            
        except Exception as e:
            self.error_handler.handle_error(e, {"operation":"scan_opportunities"})
            return []
    
    async def _analyze_ticker(self, ticker: str) -> Optional[WheelCandidate]:
        """Analyze ticker for wheel strategy suitability"""
        try:
            # Get market data
            market_data=await self.data.get_market_data(ticker)
            if market_data.price <= 0:
                return None
            
            # Get options data
            options_data=await self.data.get_options_data(ticker)
            if not options_data:
                return None
            
            # Calculate volatility rank (simplified)
            volatility_rank=min(market_data.change_percent * 10, 1.0)
            
            # Find suitable put options
            suitable_puts=self._find_suitable_puts(options_data, market_data.price)
            if not suitable_puts:
                return None
            
            # Calculate IV rank (simplified)
            iv_rank=sum(opt.implied_volatility for opt in suitable_puts) / len(suitable_puts)
            iv_rank=min(iv_rank, 1.0)
            
            # Calculate premiums
            put_premium=sum(opt.bid for opt in suitable_puts) / len(suitable_puts)
            
            # Check for earnings
            earnings_events=await self.data.get_earnings_data(ticker, days_ahead=30)
            earnings_risk=0.0
            if earnings_events:
                earnings_risk = 0.3  # High risk if earnings within 30 days
            
            candidate = WheelCandidate(
                ticker=ticker,
                current_price=market_data.price,
                volatility_rank=volatility_rank,
                earnings_risk=earnings_risk,
                put_premium=put_premium,
                iv_rank=iv_rank
            )
            
            candidate.calculate_wheel_score()
            
            return candidate
            
        except Exception as e:
            self.error_handler.handle_error(e, {"ticker":ticker, "operation":"analyze"})
            return None
    
    def _find_suitable_puts(self, options_data: List[OptionsData], current_price: float) -> List[OptionsData]:
        """Find suitable put options for wheel strategy"""
        suitable_puts=[]
        
        # Look for puts 30-45 days out, 5-10% OTM
        target_days=30
        target_strike_range = (0.90, 0.95)  # 5-10% OTM
        
        for option in options_data:
            if option.option_type != 'put':continue
            
            # Check expiry (simplified - would need proper date parsing)
            days_to_expiry=30  # Simplified
            
            # Check strike range
            strike_ratio = option.strike / current_price
            if target_strike_range[0] <= strike_ratio <= target_strike_range[1]:
                suitable_puts.append(option)
        
        return suitable_puts[:3]  # Top 3 suitable puts
    
    async def execute_wheel_trade(self, candidate: WheelCandidate) -> bool:
        """Execute wheel strategy trade"""
        try:
            self.logger.info(f"Executing wheel trade for {candidate.ticker}")
            
            # Check if we already have a position
            if candidate.ticker in self.positions:
                self.logger.warning(f"Already have position in {candidate.ticker}")
                return False
            
            # Check position limits
            if len(self.positions) >= self.max_positions:
                self.logger.warning("Maximum positions reached")
                return False
            
            # Get current market data
            market_data=await self.data.get_market_data(candidate.ticker)
            options_data=await self.data.get_options_data(candidate.ticker)
            
            if not options_data:
                self.logger.error(f"No options data for {candidate.ticker}")
                return False
            
            # Find best put option
            best_put=self._select_best_put(options_data, market_data.price)
            if not best_put:
                self.logger.error(f"No suitable put found for {candidate.ticker}")
                return False
            
            # Calculate position size
            position_size=self._calculate_position_size(candidate.ticker, best_put.strike)
            if position_size <= 0:
                self.logger.error(f"Invalid position size for {candidate.ticker}")
                return False
            
            # Create trade signal
            signal=TradeSignal(
                strategy_name="Wheel Strategy",
                ticker=candidate.ticker,
                side=OrderSide.SELL,  # Selling puts
                order_type=OrderType.LIMIT,
                quantity=position_size,
                limit_price=best_put.bid,
                reason=f"Wheel strategy: Sell {best_put.strike} put",
                confidence=candidate.wheel_score
            )
            
            # Execute trade
            trade_result=await self.trading.execute_trade(signal)
            
            if trade_result.status.value== "filled":# Create wheel position
                position = WheelPosition(
                    ticker=candidate.ticker,
                    stage=WheelStage.CASH_SECURED_PUT,
                    status=WheelStatus.ACTIVE,
                    quantity=position_size,
                    entry_price=market_data.price,
                    current_price=market_data.price,
                    option_type="put",
                    strike_price=best_put.strike,
                    expiry_date=datetime.now() + timedelta(days=30),  # Simplified
                    premium_received=trade_result.filled_price * position_size * 100
                )
                
                self.positions[candidate.ticker] = position
                
                self.logger.info(f"Wheel trade executed: {candidate.ticker}",
                               strike=best_put.strike,
                               premium=position.premium_received,
                               quantity=position_size)
                
                self.metrics.record_metric("wheel_trades_executed", 1, {"ticker":candidate.ticker})
                
                return True
            else:
                self.logger.error(f"Wheel trade failed: {trade_result.error_message}")
                return False
                
        except Exception as e:
            self.error_handler.handle_error(e, {"ticker":candidate.ticker, "operation":"execute_trade"})
            return False
    
    def _select_best_put(self, options_data: List[OptionsData], current_price: float) -> Optional[OptionsData]:
        """Select best put option for wheel strategy"""
        suitable_puts=self._find_suitable_puts(options_data, current_price)
        
        if not suitable_puts:
            return None
        
        # Select put with best premium/risk ratio
        best_put=max(suitable_puts, key=lambda x: x.bid)
        return best_put
    
    def _calculate_position_size(self, ticker: str, strike_price: float) -> int:
        """Calculate position size for wheel strategy"""
        try:
            # Get account value
            account_value=self.config.risk.account_size
            
            # Calculate max position value (5% of account)
            max_position_value=account_value * 0.05
            
            # Calculate shares per contract
            shares_per_contract = 100
            
            # Calculate max contracts
            max_contracts = int(max_position_value / (strike_price * shares_per_contract))
            
            # Limit to reasonable size
            max_contracts=min(max_contracts, 10)
            
            return max(1, max_contracts)
            
        except Exception as e:
            self.error_handler.handle_error(e, {"ticker":ticker, "operation":"position_sizing"})
            return 1
    
    async def manage_positions(self):
        """Manage existing wheel positions"""
        self.logger.info("Managing wheel positions")
        
        for ticker, position in list(self.positions.items()):
            try:
                await self._manage_position(position)
            except Exception as e:
                self.error_handler.handle_error(e, {"ticker":ticker, "operation":"manage_position"})
    
    async def _manage_position(self, position: WheelPosition):
        """Manage individual wheel position"""
        # Get current market data
        market_data=await self.data.get_market_data(position.ticker)
        options_data=await self.data.get_options_data(position.ticker)
        
        # Update position pricing
        position.update_pricing(market_data, options_data)
        
        # Check for management actions
        if position.stage== WheelStage.CASH_SECURED_PUT:
            await self._manage_cash_secured_put(position)
        elif position.stage== WheelStage.ASSIGNED_STOCK:
            await self._manage_assigned_stock(position)
        elif position.stage== WheelStage.COVERED_CALL:
            await self._manage_covered_call(position)
    
    async def _manage_cash_secured_put(self, position: WheelPosition):
        """Manage cash secured put position"""
        # Check for profit target
        profit_pct=position.unrealized_pnl / position.premium_received
        
        if profit_pct >= self.profit_target:
            # Close position at profit target
            await self._close_position(position, "Profit target reached")
            return
        
        # Check for roll threshold
        if profit_pct <= -self.roll_threshold:
            # Roll position
            await self._roll_position(position, "Roll threshold reached")
            return
        
        # Check for expiry
        if position.calculate_days_to_expiry() <= 1:
            # Close position before expiry
            await self._close_position(position, "Expiry approaching")
            return
    
    async def _manage_assigned_stock(self, position: WheelPosition):
        """Manage assigned stock position"""
        # Look for covered call opportunities
        options_data=await self.data.get_options_data(position.ticker)
        
        if options_data:
            # Find suitable call options
            suitable_calls=self._find_suitable_calls(options_data, position.current_price)
            
            if suitable_calls:
                # Sell covered call
                await self._sell_covered_call(position, suitable_calls[0])
    
    async def _manage_covered_call(self, position: WheelPosition):
        """Manage covered call position"""
        # Check for profit target
        profit_pct=position.unrealized_pnl / (position.premium_received + position.premium_paid)
        
        if profit_pct >= self.profit_target:
            # Close position at profit target
            await self._close_position(position, "Profit target reached")
            return
        
        # Check for roll threshold
        if profit_pct <= -self.roll_threshold:
            # Roll position
            await self._roll_position(position, "Roll threshold reached")
            return
    
    def _find_suitable_calls(self, options_data: List[OptionsData], current_price: float) -> List[OptionsData]:
        """Find suitable call options for covered calls"""
        suitable_calls=[]
        
        # Look for calls 30-45 days out, 5-10% OTM
        target_strike_range=(1.05, 1.10)  # 5-10% OTM
        
        for option in options_data:
            if option.option_type != 'call':continue
            
            # Check strike range
            strike_ratio=option.strike / current_price
            if target_strike_range[0] <= strike_ratio <= target_strike_range[1]:
                suitable_calls.append(option)
        
        return suitable_calls[:3]  # Top 3 suitable calls
    
    async def _sell_covered_call(self, position: WheelPosition, call_option: OptionsData):
        """Sell covered call"""
        try:
            # Create trade signal
            signal=TradeSignal(
                strategy_name="Wheel Strategy",
                ticker=position.ticker,
                side=OrderSide.SELL,  # Selling calls
                order_type=OrderType.LIMIT,
                quantity=position.quantity,
                limit_price=call_option.bid,
                reason=f"Wheel strategy: Sell {call_option.strike} call",
                confidence=0.8
            )
            
            # Execute trade
            trade_result=await self.trading.execute_trade(signal)
            
            if trade_result.status.value== "filled":# Update position
                position.stage = WheelStage.COVERED_CALL
                position.option_type = "call"
                position.strike_price = call_option.strike
                position.premium_paid = trade_result.filled_price * position.quantity * 100
                
                self.logger.info(f"Covered call sold: {position.ticker}",
                               strike=call_option.strike,
                               premium=position.premium_paid)
                
                self.metrics.record_metric("covered_calls_sold", 1, {"ticker":position.ticker})
            
        except Exception as e:
            self.error_handler.handle_error(e, {"ticker":position.ticker, "operation":"sell_covered_call"})
    
    async def _close_position(self, position: WheelPosition, reason: str):
        """Close wheel position"""
        try:
            # Create closing trade signal
            signal=TradeSignal(
                strategy_name="Wheel Strategy",
                ticker=position.ticker,
                side=OrderSide.BUY,  # Buy to close
                order_type=OrderType.MARKET,
                quantity=position.quantity,
                reason=f"Wheel strategy: Close position - {reason}",
                confidence=0.9
            )
            
            # Execute trade
            trade_result=await self.trading.execute_trade(signal)
            
            if trade_result.status.value== "filled":# Update position
                position.status = WheelStatus.CLOSED
                position.stage = WheelStage.CLOSED_POSITION
                
                self.logger.info(f"Wheel position closed: {position.ticker}",
                               reason=reason,
                               pnl=position.unrealized_pnl)
                
                self.metrics.record_metric("wheel_positions_closed", 1, {"ticker":position.ticker})
                
                # Remove from active positions
                if position.ticker in self.positions:
                    del self.positions[position.ticker]
            
        except Exception as e:
            self.error_handler.handle_error(e, {"ticker":position.ticker, "operation":"close_position"})
    
    async def _roll_position(self, position: WheelPosition, reason: str):
        """Roll wheel position"""
        try:
            # Close current position
            await self._close_position(position, f"Rolling - {reason}")
            
            # Wait a moment for settlement
            await asyncio.sleep(1)
            
            # Open new position with better terms
            candidate=WheelCandidate(
                ticker=position.ticker,
                current_price=position.current_price,
                volatility_rank=0.5,  # Default
                wheel_score=0.7  # Default
            )
            
            await self.execute_wheel_trade(candidate)
            
            self.logger.info(f"Wheel position rolled: {position.ticker}", reason=reason)
            
        except Exception as e:
            self.error_handler.handle_error(e, {"ticker":position.ticker, "operation":"roll_position"})
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get wheel strategy portfolio summary"""
        total_pnl=sum(pos.unrealized_pnl for pos in self.positions.values())
        total_premium=sum(pos.premium_received for pos in self.positions.values())
        
        summary={
            "total_positions":len(self.positions),
            "total_pnl":total_pnl,
            "total_premium_received":total_premium,
            "positions":[]
        }
        
        for ticker, position in self.positions.items():
            summary["positions"].append({
                "ticker":ticker,
                "stage":position.stage.value,
                "status":position.status.value,
                "quantity":position.quantity,
                "current_price":position.current_price,
                "unrealized_pnl":position.unrealized_pnl,
                "premium_received":position.premium_received,
                "days_to_expiry":position.calculate_days_to_expiry()
            })
        
        return summary
    
    async def run_strategy(self):
        """Run wheel strategy main loop"""
        self.logger.info("Starting Wheel Strategy")
        
        while True:
            try:
                # Scan for opportunities
                candidates=await self.scan_for_opportunities()
                
                # Execute new trades
                for candidate in candidates[:3]:  # Top 3 candidates
                    if len(self.positions) < self.max_positions:
                        await self.execute_wheel_trade(candidate)
                
                # Manage existing positions
                await self.manage_positions()
                
                # Log portfolio summary
                summary=await self.get_portfolio_summary()
                self.logger.info("Wheel portfolio summary", **summary)
                
                # Record metrics
                self.metrics.record_metric("wheel_total_pnl", summary["total_pnl"])
                self.metrics.record_metric("wheel_active_positions", summary["total_positions"])
                
                # Wait for next scan
                await asyncio.sleep(self.scan_interval.total_seconds())
                
            except Exception as e:
                self.error_handler.handle_error(e, {"operation":"run_strategy"})
                await asyncio.sleep(60)  # Wait 1 minute on error


# Factory function for easy initialization
def create_wheel_strategy(trading_interface: TradingInterface,
                         data_provider: UnifiedDataProvider,
                         config: ProductionConfig,
                         logger: ProductionLogger) -> ProductionWheelStrategy:
    """Create wheel strategy instance"""
    return ProductionWheelStrategy(trading_interface, data_provider, config, logger)
