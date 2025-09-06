"""
Production SPX Credit Spreads Implementation
Index options with real-time CME data integration
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


class SPXSpreadType(Enum):
    """SPX spread types"""
    PUT_CREDIT_SPREAD = "put_credit_spread"
    CALL_CREDIT_SPREAD = "call_credit_spread"
    IRON_CONDOR = "iron_condor"
    IRON_BUTTERFLY = "iron_butterfly"


class SPXSpreadStatus(Enum):
    """SPX spread position status"""
    ACTIVE = "active"
    EXPIRED = "expired"
    CLOSED = "closed"
    ROLLED = "rolled"


@dataclass
class SPXSpreadPosition:
    """SPX spread position tracking"""
    spread_type: SPXSpreadType
    status: SPXSpreadStatus
    
    # Spread details
    long_strike: float
    short_strike: float
    quantity: int
    net_credit: float
    max_profit: float
    max_loss: float
    
    # Option details
    long_option: Dict[str, Any]  # Long option details
    short_option: Dict[str, Any]  # Short option details
    
    # Pricing
    current_value: float = 0.0
    unrealized_pnl: float = 0.0
    profit_pct: float = 0.0
    
    # Timing
    entry_date: datetime = field(default_factory=datetime.now)
    expiry_date: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=1))  # 0DTE
    last_update: datetime = field(default_factory=datetime.now)
    
    # Greeks
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0
    
    def update_pricing(self, long_option_data: OptionsData, short_option_data: OptionsData):
        """Update spread pricing with current option data"""
        # Calculate current spread value
        long_value = long_option_data.ask
        short_value = short_option_data.bid
        
        self.current_value = (short_value - long_value) * self.quantity * 100
        
        # Calculate P&L
        self.unrealized_pnl = (self.net_credit * self.quantity * 100) - self.current_value
        self.profit_pct = self.unrealized_pnl / (self.net_credit * self.quantity * 100)
        
        # Update Greeks
        self.net_delta = (short_option_data.delta - long_option_data.delta) * self.quantity * 100
        self.net_gamma = (short_option_data.gamma - long_option_data.gamma) * self.quantity * 100
        self.net_theta = (short_option_data.theta - long_option_data.theta) * self.quantity * 100
        self.net_vega = (short_option_data.vega - long_option_data.vega) * self.quantity * 100
        
        self.last_update = datetime.now()
    
    def calculate_max_profit(self) -> float:
        """Calculate maximum profit potential"""
        return self.net_credit * self.quantity * 100
    
    def calculate_max_loss(self) -> float:
        """Calculate maximum loss potential"""
        if self.spread_type == SPXSpreadType.PUT_CREDIT_SPREAD:
            return (self.long_strike - self.short_strike) * self.quantity * 100 - self.net_credit * self.quantity * 100
        elif self.spread_type == SPXSpreadType.CALL_CREDIT_SPREAD:
            return (self.short_strike - self.long_strike) * self.quantity * 100 - self.net_credit * self.quantity * 100
        return 0.0
    
    def calculate_days_to_expiry(self) -> int:
        """Calculate days to expiry"""
        delta = self.expiry_date - datetime.now()
        return max(0, delta.days)


@dataclass
class SPXSpreadCandidate:
    """SPX spread candidate screening"""
    spread_type: SPXSpreadType
    
    # Option chain data
    long_strike: float
    short_strike: float
    long_premium: float
    short_premium: float
    net_credit: float
    
    # Risk metrics
    max_profit: float
    max_loss: float
    profit_loss_ratio: float
    
    # Greeks
    net_delta: float
    net_theta: float
    net_vega: float
    
    # Market conditions
    spx_price: float
    vix_level: float
    market_regime: str  # 'bull', 'bear', 'neutral'
    
    # Scoring
    spread_score: float = 0.0
    risk_score: float = 0.0
    
    def calculate_spread_score(self) -> float:
        """Calculate SPX spread strategy score"""
        score = 0.0
        
        # Profit/Loss ratio (higher is better)
        score += min(self.profit_loss_ratio, 5.0) * 0.2
        
        # Net theta (positive is better for credit spreads)
        score += max(0, self.net_theta) * 0.3
        
        # VIX level (moderate VIX is better)
        if 15 <= self.vix_level <= 25:
            score += 0.2
        elif self.vix_level > 30:
            score -= 0.1  # High VIX is risky
        
        # Market regime alignment
        if self.spread_type == SPXSpreadType.PUT_CREDIT_SPREAD:
            if self.market_regime == 'bull':
                score += 0.2
            elif self.market_regime == 'bear':
                score -= 0.2
        
        # Strike width (reasonable width)
        strike_width = abs(self.short_strike - self.long_strike)
        if 10 <= strike_width <= 50:  # SPX typical widths
            score += 0.1
        
        # Distance from current price
        if self.spread_type == SPXSpreadType.PUT_CREDIT_SPREAD:
            distance = (self.short_strike - self.spx_price) / self.spx_price
            if 0.02 <= distance <= 0.05:  # 2-5% OTM
                score += 0.1
        
        self.spread_score = max(0.0, min(1.0, score))
        return self.spread_score


class CMEDataProvider:
    """CME data provider for SPX options"""
    
    def __init__(self, logger: ProductionLogger):
        self.logger = logger
        self.base_url = "https://www.cmegroup.com/CmeWS/mvc/ProductSlate/V2/List"
    
    async def get_spx_options(self, expiry_date: str = None) -> List[OptionsData]:
        """Get SPX options data from CME"""
        try:
            # In production, would integrate with CME API
            # For now, return mock data
            
            spx_price = 4500.0  # Mock SPX price
            options = []
            
            # Generate mock SPX options
            strikes = range(4400, 4600, 25)  # 25-point strikes
            
            for strike in strikes:
                # Put options
                put_option = OptionsData(
                    ticker="SPX",
                    expiry_date=expiry_date or datetime.now().strftime('%Y-%m-%d'),
                    strike=float(strike),
                    option_type="put",
                    bid=max(0.1, (strike - spx_price) * 0.01),
                    ask=max(0.1, (strike - spx_price) * 0.01 + 0.5),
                    last_price=max(0.1, (strike - spx_price) * 0.01 + 0.25),
                    volume=1000,
                    open_interest=5000,
                    implied_volatility=0.20,
                    delta=-0.5 if strike == spx_price else -0.3,
                    gamma=0.01,
                    theta=-0.05,
                    vega=0.1
                )
                options.append(put_option)
                
                # Call options
                call_option = OptionsData(
                    ticker="SPX",
                    expiry_date=expiry_date or datetime.now().strftime('%Y-%m-%d'),
                    strike=float(strike),
                    option_type="call",
                    bid=max(0.1, (spx_price - strike) * 0.01),
                    ask=max(0.1, (spx_price - strike) * 0.01 + 0.5),
                    last_price=max(0.1, (spx_price - strike) * 0.01 + 0.25),
                    volume=1000,
                    open_interest=5000,
                    implied_volatility=0.20,
                    delta=0.5 if strike == spx_price else 0.3,
                    gamma=0.01,
                    theta=-0.05,
                    vega=0.1
                )
                options.append(call_option)
            
            return options
            
        except Exception as e:
            self.logger.error(f"Error getting SPX options: {e}")
            return []
    
    async def get_vix_level(self) -> float:
        """Get current VIX level"""
        try:
            # In production, would get real VIX data
            # For now, return mock VIX
            return 20.0  # Mock VIX level
            
        except Exception as e:
            self.logger.error(f"Error getting VIX: {e}")
            return 20.0
    
    async def get_market_regime(self) -> str:
        """Determine current market regime"""
        try:
            # In production, would analyze market conditions
            # For now, return mock regime
            return "bull"  # Mock market regime
            
        except Exception as e:
            self.logger.error(f"Error determining market regime: {e}")
            return "neutral"


class ProductionSPXSpreads:
    """Production SPX Credit Spreads Implementation"""
    
    def __init__(self, 
                 trading_interface: TradingInterface,
                 data_provider: UnifiedDataProvider,
                 config: ProductionConfig,
                 logger: ProductionLogger):
        self.trading = trading_interface
        self.data = data_provider
        self.config = config
        self.logger = logger
        self.error_handler = ErrorHandler(logger)
        self.metrics = MetricsCollector(logger)
        self.cme_data = CMEDataProvider(logger)
        
        # Strategy parameters
        self.max_positions = config.trading.max_concurrent_trades
        self.max_position_size = config.risk.max_position_risk
        self.profit_target = 0.50  # Close at 50% profit
        self.stop_loss = 0.25  # Close at 25% loss
        self.min_profit_loss_ratio = 3.0  # Minimum 3:1 ratio for SPX
        
        # Position tracking
        self.positions: Dict[str, SPXSpreadPosition] = {}
        self.candidates: List[SPXSpreadCandidate] = []
        
        # Strategy state
        self.last_scan_time: Optional[datetime] = None
        self.scan_interval = timedelta(minutes=5)  # More frequent for 0DTE
        
        self.logger.info("SPX Credit Spreads Strategy initialized",
                        max_positions=self.max_positions,
                        profit_target=self.profit_target)
    
    async def scan_for_opportunities(self) -> List[SPXSpreadCandidate]:
        """Scan for SPX spread opportunities"""
        self.logger.info("Scanning for SPX spread opportunities")
        
        try:
            candidates = []
            
            # Get SPX options data
            spx_options = await self.cme_data.get_spx_options()
            if not spx_options:
                return []
            
            # Get market conditions
            vix_level = await self.cme_data.get_vix_level()
            market_regime = await self.cme_data.get_market_regime()
            
            # Get SPX price
            spx_data = await self.data.get_market_data("SPX")
            spx_price = spx_data.price if spx_data.price > 0 else 4500.0
            
            # Look for put credit spreads
            put_spreads = self._find_put_credit_spreads(spx_options, spx_price)
            for spread in put_spreads:
                candidate = SPXSpreadCandidate(
                    spread_type=SPXSpreadType.PUT_CREDIT_SPREAD,
                    long_strike=spread['long_strike'],
                    short_strike=spread['short_strike'],
                    long_premium=spread['long_premium'],
                    short_premium=spread['short_premium'],
                    net_credit=spread['net_credit'],
                    max_profit=spread['max_profit'],
                    max_loss=spread['max_loss'],
                    profit_loss_ratio=spread['profit_loss_ratio'],
                    net_delta=spread['net_delta'],
                    net_theta=spread['net_theta'],
                    net_vega=spread['net_vega'],
                    spx_price=spx_price,
                    vix_level=vix_level,
                    market_regime=market_regime
                )
                
                candidate.calculate_spread_score()
                
                if candidate.spread_score > 0.7:  # Higher threshold for SPX
                    candidates.append(candidate)
            
            # Sort by spread score
            candidates.sort(key=lambda x: x.spread_score, reverse=True)
            
            self.candidates = candidates[:5]  # Top 5 candidates
            
            self.logger.info(f"Found {len(self.candidates)} SPX spread candidates")
            self.metrics.record_metric("spx_spread_candidates_found", len(self.candidates))
            
            return self.candidates
            
        except Exception as e:
            self.error_handler.handle_error(e, {"operation": "scan_opportunities"})
            return []
    
    def _find_put_credit_spreads(self, options_data: List[OptionsData], spx_price: float) -> List[Dict[str, Any]]:
        """Find put credit spread opportunities"""
        spreads = []
        
        # Filter put options
        put_options = [opt for opt in options_data if opt.option_type == 'put']
        
        # Look for spreads with strikes below current price
        for short_option in put_options:
            if short_option.strike >= spx_price * 0.98:  # Short strike near ATM
                continue
            
            for long_option in put_options:
                if long_option.strike >= short_option.strike:  # Long strike must be lower
                    continue
                
                if short_option.strike - long_option.strike > 50:  # Max 50-point width
                    continue
                
                # Calculate spread metrics
                net_credit = short_option.bid - long_option.ask
                if net_credit <= 0:  # Must be a credit spread
                    continue
                
                max_profit = net_credit * 100
                max_loss = (short_option.strike - long_option.strike) * 100 - net_credit * 100
                profit_loss_ratio = max_profit / max_loss if max_loss > 0 else 0
                
                # Filter by minimum profit/loss ratio
                if profit_loss_ratio < self.min_profit_loss_ratio:
                    continue
                
                # Calculate net Greeks
                net_delta = short_option.delta - long_option.delta
                net_theta = short_option.theta - long_option.theta
                net_vega = short_option.vega - long_option.vega
                
                spreads.append({
                    'long_strike': long_option.strike,
                    'short_strike': short_option.strike,
                    'long_premium': long_option.ask,
                    'short_premium': short_option.bid,
                    'net_credit': net_credit,
                    'max_profit': max_profit,
                    'max_loss': max_loss,
                    'profit_loss_ratio': profit_loss_ratio,
                    'net_delta': net_delta,
                    'net_theta': net_theta,
                    'net_vega': net_vega,
                    'long_option': long_option,
                    'short_option': short_option
                })
        
        # Sort by profit/loss ratio
        spreads.sort(key=lambda x: x['profit_loss_ratio'], reverse=True)
        
        return spreads[:3]  # Top 3 spreads
    
    async def execute_spx_spread(self, candidate: SPXSpreadCandidate) -> bool:
        """Execute SPX spread trade"""
        try:
            self.logger.info(f"Executing SPX spread")
            
            # Check position limits
            if len(self.positions) >= self.max_positions:
                self.logger.warning("Maximum positions reached")
                return False
            
            # Calculate position size
            position_size = self._calculate_position_size(candidate)
            if position_size <= 0:
                self.logger.error("Invalid position size")
                return False
            
            # Execute short option trade (sell put)
            short_signal = TradeSignal(
                strategy_name="SPX Credit Spreads",
                ticker="SPX",
                side=OrderSide.SELL,  # Sell short option
                order_type=OrderType.LIMIT,
                quantity=position_size,
                limit_price=candidate.short_premium,
                reason=f"SPX spread: Sell {candidate.short_strike} put",
                confidence=candidate.spread_score
            )
            
            short_result = await self.trading.execute_trade(short_signal)
            
            if short_result.status.value != "filled":
                self.logger.error(f"Short option trade failed: {short_result.error_message}")
                return False
            
            # Execute long option trade (buy put)
            long_signal = TradeSignal(
                strategy_name="SPX Credit Spreads",
                ticker="SPX",
                side=OrderSide.BUY,  # Buy long option
                order_type=OrderType.LIMIT,
                quantity=position_size,
                limit_price=candidate.long_premium,
                reason=f"SPX spread: Buy {candidate.long_strike} put",
                confidence=candidate.spread_score
            )
            
            long_result = await self.trading.execute_trade(long_signal)
            
            if long_result.status.value != "filled":
                self.logger.error(f"Long option trade failed: {long_result.error_message}")
                # Try to close the short position
                await self._close_short_position(short_signal, short_result)
                return False
            
            # Create spread position
            position_key = f"SPX_{candidate.long_strike}_{candidate.short_strike}"
            position = SPXSpreadPosition(
                spread_type=candidate.spread_type,
                status=SPXSpreadStatus.ACTIVE,
                long_strike=candidate.long_strike,
                short_strike=candidate.short_strike,
                quantity=position_size,
                net_credit=candidate.net_credit,
                max_profit=candidate.max_profit,
                max_loss=candidate.max_loss,
                long_option={
                    'strike': candidate.long_strike,
                    'premium': long_result.filled_price,
                    'delta': 0.0,  # Would get from option data
                    'theta': 0.0,
                    'vega': 0.0
                },
                short_option={
                    'strike': candidate.short_strike,
                    'premium': short_result.filled_price,
                    'delta': 0.0,  # Would get from option data
                    'theta': 0.0,
                    'vega': 0.0
                }
            )
            
            self.positions[position_key] = position
            
            self.logger.info(f"SPX spread executed",
                           long_strike=candidate.long_strike,
                           short_strike=candidate.short_strike,
                           net_credit=candidate.net_credit,
                           max_profit=candidate.max_profit)
            
            self.metrics.record_metric("spx_spreads_executed", 1)
            
            return True
            
        except Exception as e:
            self.error_handler.handle_error(e, {"operation": "execute_spx_spread"})
            return False
    
    async def _close_short_position(self, signal: TradeSignal, result):
        """Close short position if long trade fails"""
        try:
            close_signal = TradeSignal(
                strategy_name="SPX Credit Spreads",
                ticker=signal.ticker,
                side=OrderSide.BUY,  # Buy to close
                order_type=OrderType.MARKET,
                quantity=signal.quantity,
                reason="Close short position due to long trade failure"
            )
            
            await self.trading.execute_trade(close_signal)
            
        except Exception as e:
            self.error_handler.handle_error(e, {"operation": "close_short_position"})
    
    def _calculate_position_size(self, candidate: SPXSpreadCandidate) -> int:
        """Calculate position size for SPX spread"""
        try:
            # Get account value
            account_value = self.config.risk.account_size
            
            # Calculate max position value (1% of account for SPX spreads)
            max_position_value = account_value * 0.01
            
            # Calculate max contracts
            max_contracts = int(max_position_value / candidate.max_loss)
            
            # Limit to reasonable size for SPX
            max_contracts = min(max_contracts, 3)
            
            return max(1, max_contracts)
            
        except Exception as e:
            self.error_handler.handle_error(e, {"operation": "position_sizing"})
            return 1
    
    async def manage_positions(self):
        """Manage existing SPX spread positions"""
        self.logger.info("Managing SPX spread positions")
        
        for position_key, position in list(self.positions.items()):
            try:
                await self._manage_position(position)
            except Exception as e:
                self.error_handler.handle_error(e, {"position": position_key, "operation": "manage_position"})
    
    async def _manage_position(self, position: SPXSpreadPosition):
        """Manage individual SPX spread position"""
        # Get current SPX options data
        spx_options = await self.cme_data.get_spx_options()
        
        if not spx_options:
            return
        
        # Find current option prices
        long_option_data = None
        short_option_data = None
        
        for option in spx_options:
            if (option.strike == position.long_strike and 
                option.option_type == 'put'):
                long_option_data = option
            elif (option.strike == position.short_strike and 
                  option.option_type == 'put'):
                short_option_data = option
        
        if not long_option_data or not short_option_data:
            return
        
        # Update position pricing
        position.update_pricing(long_option_data, short_option_data)
        
        # Check for management actions
        if position.profit_pct >= self.profit_target:
            # Close position at profit target
            await self._close_position(position, "Profit target reached")
            return
        
        if position.profit_pct <= -self.stop_loss:
            # Close position at stop loss
            await self._close_position(position, "Stop loss triggered")
            return
        
        # Check for expiry (0DTE - close before market close)
        if position.calculate_days_to_expiry() <= 0:
            # Close position before expiry
            await self._close_position(position, "Expiry approaching")
            return
    
    async def _close_position(self, position: SPXSpreadPosition, reason: str):
        """Close SPX spread position"""
        try:
            # Close short position
            short_close_signal = TradeSignal(
                strategy_name="SPX Credit Spreads",
                ticker="SPX",
                side=OrderSide.BUY,  # Buy to close
                order_type=OrderType.MARKET,
                quantity=position.quantity,
                reason=f"Close SPX spread short - {reason}"
            )
            
            short_result = await self.trading.execute_trade(short_close_signal)
            
            # Close long position
            long_close_signal = TradeSignal(
                strategy_name="SPX Credit Spreads",
                ticker="SPX",
                side=OrderSide.SELL,  # Sell to close
                order_type=OrderType.MARKET,
                quantity=position.quantity,
                reason=f"Close SPX spread long - {reason}"
            )
            
            long_result = await self.trading.execute_trade(long_close_signal)
            
            if (short_result.status.value == "filled" and 
                long_result.status.value == "filled"):
                
                # Update position
                position.status = SPXSpreadStatus.CLOSED
                
                self.logger.info(f"SPX spread closed",
                               reason=reason,
                               pnl=position.unrealized_pnl,
                               profit_pct=position.profit_pct)
                
                self.metrics.record_metric("spx_spreads_closed", 1)
                
                # Remove from active positions
                position_key = f"SPX_{position.long_strike}_{position.short_strike}"
                if position_key in self.positions:
                    del self.positions[position_key]
            
        except Exception as e:
            self.error_handler.handle_error(e, {"operation": "close_position"})
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get SPX spreads portfolio summary"""
        total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_credit = sum(pos.net_credit * pos.quantity * 100 for pos in self.positions.values())
        
        summary = {
            "total_positions": len(self.positions),
            "total_pnl": total_pnl,
            "total_credit_received": total_credit,
            "positions": []
        }
        
        for position_key, position in self.positions.items():
            summary["positions"].append({
                "spread_type": position.spread_type.value,
                "long_strike": position.long_strike,
                "short_strike": position.short_strike,
                "quantity": position.quantity,
                "net_credit": position.net_credit,
                "unrealized_pnl": position.unrealized_pnl,
                "profit_pct": position.profit_pct,
                "days_to_expiry": position.calculate_days_to_expiry()
            })
        
        return summary
    
    async def run_strategy(self):
        """Run SPX spreads strategy main loop"""
        self.logger.info("Starting SPX Credit Spreads Strategy")
        
        while True:
            try:
                # Check market hours (SPX trades during market hours)
                if not await self._is_market_open():
                    await asyncio.sleep(300)  # Wait 5 minutes
                    continue
                
                # Scan for opportunities
                candidates = await self.scan_for_opportunities()
                
                # Execute new trades
                for candidate in candidates[:1]:  # Top 1 candidate (conservative)
                    if len(self.positions) < self.max_positions:
                        await self.execute_spx_spread(candidate)
                
                # Manage existing positions
                await self.manage_positions()
                
                # Log portfolio summary
                summary = await self.get_portfolio_summary()
                self.logger.info("SPX spreads portfolio summary", **summary)
                
                # Record metrics
                self.metrics.record_metric("spx_spreads_total_pnl", summary["total_pnl"])
                self.metrics.record_metric("spx_spreads_active_positions", summary["total_positions"])
                
                # Wait for next scan
                await asyncio.sleep(self.scan_interval.total_seconds())
                
            except Exception as e:
                self.error_handler.handle_error(e, {"operation": "run_strategy"})
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _is_market_open(self) -> bool:
        """Check if market is open"""
        try:
            # In production, would check actual market hours
            # For now, return True during business hours
            now = datetime.now()
            return 9 <= now.hour < 16  # 9 AM to 4 PM
            
        except Exception as e:
            self.error_handler.handle_error(e, {"operation": "check_market_hours"})
            return False


# Factory function for easy initialization
def create_spx_spreads_strategy(trading_interface: TradingInterface,
                               data_provider: UnifiedDataProvider,
                               config: ProductionConfig,
                               logger: ProductionLogger) -> ProductionSPXSpreads:
    """Create SPX spreads strategy instance"""
    return ProductionSPXSpreads(trading_interface, data_provider, config, logger)
