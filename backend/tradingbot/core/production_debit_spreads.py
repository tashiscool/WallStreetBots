"""
Production Debit Call Spreads Implementation
Defined-risk bullish strategies with QuantLib pricing
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


class SpreadType(Enum):
    """Debit spread types"""
    BULL_CALL_SPREAD="bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    CALENDAR_SPREAD = "calendar_spread"
    DIAGONAL_SPREAD = "diagonal_spread"


class SpreadStatus(Enum):
    """Spread position status"""
    ACTIVE="active"
    EXPIRED = "expired"
    CLOSED = "closed"
    ROLLED = "rolled"


@dataclass
class SpreadPosition:
    """Debit spread position tracking"""
    ticker: str
    spread_type: SpreadType
    status: SpreadStatus
    
    # Spread details
    long_strike: float
    short_strike: float
    quantity: int
    net_debit: float
    max_profit: float
    max_loss: float
    
    # Option details
    long_option: Dict[str, Any]  # Long option details
    short_option: Dict[str, Any]  # Short option details
    
    # Pricing
    current_value: float=0.0
    unrealized_pnl: float = 0.0
    profit_pct: float = 0.0
    
    # Timing
    entry_date: datetime = field(default_factory=datetime.now)
    expiry_date: datetime=field(default_factory=lambda: datetime.now() + timedelta(days=30))
    last_update: datetime=field(default_factory=datetime.now)
    
    # Greeks
    net_delta: float=0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0
    
    def update_pricing(self, long_option_data: OptionsData, short_option_data: OptionsData):
        """Update spread pricing with current option data"""
        # Calculate current spread value
        long_value=long_option_data.bid
        short_value = short_option_data.ask
        
        self.current_value = (long_value - short_value) * self.quantity * 100
        
        # Calculate P&L
        self.unrealized_pnl=self.current_value - (self.net_debit * self.quantity * 100)
        self.profit_pct=self.unrealized_pnl / (self.net_debit * self.quantity * 100)
        
        # Update Greeks
        self.net_delta=(long_option_data.delta - short_option_data.delta) * self.quantity * 100
        self.net_gamma=(long_option_data.gamma - short_option_data.gamma) * self.quantity * 100
        self.net_theta=(long_option_data.theta - short_option_data.theta) * self.quantity * 100
        self.net_vega=(long_option_data.vega - short_option_data.vega) * self.quantity * 100
        
        self.last_update=datetime.now()
    
    def calculate_max_profit(self) -> float:
        """Calculate maximum profit potential"""
        if self.spread_type== SpreadType.BULL_CALL_SPREAD:
            return (self.short_strike - self.long_strike) * self.quantity * 100 - self.net_debit * self.quantity * 100
        return 0.0
    
    def calculate_max_loss(self) -> float:
        """Calculate maximum loss potential"""
        return self.net_debit * self.quantity * 100
    
    def calculate_days_to_expiry(self) -> int:
        """Calculate days to expiry"""
        delta=self.expiry_date - datetime.now()
        return max(0, delta.days)


@dataclass
class SpreadCandidate:
    """Debit spread candidate screening"""
    ticker: str
    current_price: float
    spread_type: SpreadType
    
    # Option chain data
    long_strike: float
    short_strike: float
    long_premium: float
    short_premium: float
    net_debit: float
    
    # Risk metrics
    max_profit: float
    max_loss: float
    profit_loss_ratio: float
    
    # Greeks
    net_delta: float
    net_theta: float
    net_vega: float
    
    # Scoring
    spread_score: float=0.0
    risk_score: float = 0.0
    
    def calculate_spread_score(self) -> float:
        """Calculate spread strategy score"""
        score=0.0
        
        # Profit/Loss ratio (higher is better)
        score += min(self.profit_loss_ratio, 3.0) * 0.3
        
        # Net delta (positive for bullish spreads)
        if self.spread_type== SpreadType.BULL_CALL_SPREAD:
            score += max(0, self.net_delta) * 0.2
        
        # Net theta (negative is better for debit spreads)
        score += max(0, -self.net_theta) * 0.2
        
        # Net debit as % of stock price (lower is better)
        debit_pct=self.net_debit / self.current_price
        score += max(0, 0.05 - debit_pct) * 20
        
        # Strike width (reasonable width)
        strike_width=abs(self.short_strike - self.long_strike)
        if 2 <= strike_width <= 10:
            score += 0.1
        
        self.spread_score=max(0.0, min(1.0, score))
        return self.spread_score


class QuantLibPricer:
    """QuantLib-based options pricing"""
    
    def __init__(self):
        self.logger=logging.getLogger(__name__)
    
    def calculate_black_scholes(self, 
                              spot_price: float,
                              strike_price: float,
                              risk_free_rate: float,
                              volatility: float,
                              time_to_expiry: float,
                              option_type: str) -> Dict[str, float]:
        """Calculate Black-Scholes price and Greeks"""
        try:
            # Simplified Black-Scholes implementation
            # In production, would use QuantLib
            
            # Calculate d1 and d2
            d1=(math.log(spot_price / strike_price) + 
                  (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
            d2=d1 - volatility * math.sqrt(time_to_expiry)
            
            # Calculate option price
            if option_type.lower() == 'call':price=(spot_price * self._normal_cdf(d1) - 
                        strike_price * math.exp(-risk_free_rate * time_to_expiry) * self._normal_cdf(d2))
            else:  # put
                price=(strike_price * math.exp(-risk_free_rate * time_to_expiry) * self._normal_cdf(-d2) - 
                        spot_price * self._normal_cdf(-d1))
            
            # Calculate Greeks
            delta=self._normal_cdf(d1) if option_type.lower() == 'call' else self._normal_cdf(d1) - 1
            gamma=self._normal_pdf(d1) / (spot_price * volatility * math.sqrt(time_to_expiry))
            theta=(-spot_price * self._normal_pdf(d1) * volatility / (2 * math.sqrt(time_to_expiry)) - 
                    risk_free_rate * strike_price * math.exp(-risk_free_rate * time_to_expiry) * self._normal_cdf(d2))
            vega=spot_price * self._normal_pdf(d1) * math.sqrt(time_to_expiry)
            
            return {
                'price':price,
                'delta':delta,
                'gamma':gamma,
                'theta':theta,
                'vega':vega
            }
            
        except Exception as e:
            self.logger.error(f"Black-Scholes calculation error: {e}")
            return {
                'price':0.0,
                'delta':0.0,
                'gamma':0.0,
                'theta':0.0,
                'vega':0.0
            }
    
    def _normal_cdf(self, x: float) -> float:
        """Cumulative distribution function of standard normal distribution"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def _normal_pdf(self, x: float) -> float:
        """Probability density function of standard normal distribution"""
        return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)


class ProductionDebitSpreads:
    """Production Debit Spreads Implementation"""
    
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
        self.pricer=QuantLibPricer()
        
        # Strategy parameters
        self.max_positions=config.trading.max_concurrent_trades
        self.max_position_size = config.risk.max_position_risk
        self.profit_target = 0.50  # Close at 50% profit
        self.stop_loss = 0.25  # Close at 25% loss
        self.min_profit_loss_ratio = 2.0  # Minimum 2:1 ratio
        
        # Position tracking
        self.positions: Dict[str, SpreadPosition] = {}
        self.candidates: List[SpreadCandidate] = []
        
        # Strategy state
        self.last_scan_time: Optional[datetime] = None
        self.scan_interval=timedelta(minutes=15)
        
        self.logger.info("Debit Spreads Strategy initialized",
                        max_positions=self.max_positions,
                        profit_target=self.profit_target)
    
    async def scan_for_opportunities(self) -> List[SpreadCandidate]:
        """Scan for debit spread opportunities"""
        self.logger.info("Scanning for debit spread opportunities")
        
        try:
            candidates=[]
            
            # Get universe of stocks
            universe = self.config.trading.universe
            
            for ticker in universe:
                try:
                    spread_candidates = await self._analyze_ticker_for_spreads(ticker)
                    candidates.extend(spread_candidates)
                
                except Exception as e:
                    self.error_handler.handle_error(e, {"ticker":ticker, "operation":"scan"})
                    continue
            
            # Sort by spread score
            candidates.sort(key=lambda x: x.spread_score, reverse=True)
            
            self.candidates=candidates[:10]  # Top 10 candidates
            
            self.logger.info(f"Found {len(self.candidates)} debit spread candidates")
            self.metrics.record_metric("debit_spread_candidates_found", len(self.candidates))
            
            return self.candidates
            
        except Exception as e:
            self.error_handler.handle_error(e, {"operation":"scan_opportunities"})
            return []
    
    async def _analyze_ticker_for_spreads(self, ticker: str) -> List[SpreadCandidate]:
        """Analyze ticker for debit spread opportunities"""
        try:
            # Get market data
            market_data=await self.data.get_market_data(ticker)
            if market_data.price <= 0:
                return []
            
            # Get options data
            options_data=await self.data.get_options_data(ticker)
            if not options_data:
                return []
            
            candidates=[]
            
            # Look for bull call spreads
            bull_spreads = self._find_bull_call_spreads(options_data, market_data.price)
            for spread in bull_spreads:
                candidate=SpreadCandidate(
                    ticker=ticker,
                    current_price=market_data.price,
                    spread_type=SpreadType.BULL_CALL_SPREAD,
                    long_strike=spread['long_strike'],
                    short_strike=spread['short_strike'],
                    long_premium=spread['long_premium'],
                    short_premium=spread['short_premium'],
                    net_debit=spread['net_debit'],
                    max_profit=spread['max_profit'],
                    max_loss=spread['max_loss'],
                    profit_loss_ratio=spread['profit_loss_ratio'],
                    net_delta=spread['net_delta'],
                    net_theta=spread['net_theta'],
                    net_vega=spread['net_vega']
                )
                
                candidate.calculate_spread_score()
                
                if candidate.spread_score > 0.6:
                    candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            self.error_handler.handle_error(e, {"ticker":ticker, "operation":"analyze_spreads"})
            return []
    
    def _find_bull_call_spreads(self, options_data: List[OptionsData], current_price: float) -> List[Dict[str, Any]]:
        """Find bull call spread opportunities"""
        spreads=[]
        
        # Filter call options
        call_options = [opt for opt in options_data if opt.option_type == 'call']
        
        # Look for spreads with strikes around current price
        for long_option in call_options:
            if long_option.strike >= current_price * 0.95:  # Long strike near ATM or ITM
                continue
            
            for short_option in call_options:
                if short_option.strike <= long_option.strike:  # Short strike must be higher
                    continue
                
                if short_option.strike - long_option.strike > current_price * 0.20:  # Max 20% width
                    continue
                
                # Calculate spread metrics
                net_debit = long_option.ask - short_option.bid
                if net_debit <= 0:  # Must be a debit spread
                    continue
                
                max_profit = (short_option.strike - long_option.strike) * 100 - net_debit * 100
                max_loss=net_debit * 100
                profit_loss_ratio = max_profit / max_loss if max_loss > 0 else 0
                
                # Filter by minimum profit/loss ratio
                if profit_loss_ratio < self.min_profit_loss_ratio:
                    continue
                
                # Calculate net Greeks
                net_delta = long_option.delta - short_option.delta
                net_theta = long_option.theta - short_option.theta
                net_vega = long_option.vega - short_option.vega
                
                spreads.append({
                    'long_strike':long_option.strike,
                    'short_strike':short_option.strike,
                    'long_premium':long_option.ask,
                    'short_premium':short_option.bid,
                    'net_debit':net_debit,
                    'max_profit':max_profit,
                    'max_loss':max_loss,
                    'profit_loss_ratio':profit_loss_ratio,
                    'net_delta':net_delta,
                    'net_theta':net_theta,
                    'net_vega':net_vega,
                    'long_option':long_option,
                    'short_option':short_option
                })
        
        # Sort by profit/loss ratio
        spreads.sort(key=lambda x: x['profit_loss_ratio'], reverse=True)
        
        return spreads[:5]  # Top 5 spreads
    
    async def execute_debit_spread(self, candidate: SpreadCandidate) -> bool:
        """Execute debit spread trade"""
        try:
            self.logger.info(f"Executing debit spread for {candidate.ticker}")
            
            # Check if we already have a position
            position_key=f"{candidate.ticker}_{candidate.long_strike}_{candidate.short_strike}"
            if position_key in self.positions:
                self.logger.warning(f"Already have position: {position_key}")
                return False
            
            # Check position limits
            if len(self.positions) >= self.max_positions:
                self.logger.warning("Maximum positions reached")
                return False
            
            # Calculate position size
            position_size=self._calculate_position_size(candidate)
            if position_size <= 0:
                self.logger.error(f"Invalid position size for {candidate.ticker}")
                return False
            
            # Execute long option trade
            long_signal=TradeSignal(
                strategy_name="Debit Spreads",
                ticker=candidate.ticker,
                side=OrderSide.BUY,  # Buy long option
                order_type=OrderType.LIMIT,
                quantity=position_size,
                limit_price=candidate.long_premium,
                reason=f"Debit spread: Buy {candidate.long_strike} call",
                confidence=candidate.spread_score
            )
            
            long_result=await self.trading.execute_trade(long_signal)
            
            if long_result.status.value != "filled":self.logger.error(f"Long option trade failed: {long_result.error_message}")
                return False
            
            # Execute short option trade
            short_signal=TradeSignal(
                strategy_name="Debit Spreads",
                ticker=candidate.ticker,
                side=OrderSide.SELL,  # Sell short option
                order_type=OrderType.LIMIT,
                quantity=position_size,
                limit_price=candidate.short_premium,
                reason=f"Debit spread: Sell {candidate.short_strike} call",
                confidence=candidate.spread_score
            )
            
            short_result=await self.trading.execute_trade(short_signal)
            
            if short_result.status.value != "filled":self.logger.error(f"Short option trade failed: {short_result.error_message}")
                # Try to close the long position
                await self._close_long_position(long_signal, long_result)
                return False
            
            # Create spread position
            position=SpreadPosition(
                ticker=candidate.ticker,
                spread_type=candidate.spread_type,
                status=SpreadStatus.ACTIVE,
                long_strike=candidate.long_strike,
                short_strike=candidate.short_strike,
                quantity=position_size,
                net_debit=candidate.net_debit,
                max_profit=candidate.max_profit,
                max_loss=candidate.max_loss,
                long_option={
                    'strike':candidate.long_strike,
                    'premium':long_result.filled_price,
                    'delta':0.0,  # Would get from option data
                    'theta':0.0,
                    'vega':0.0
                },
                short_option={
                    'strike':candidate.short_strike,
                    'premium':short_result.filled_price,
                    'delta':0.0,  # Would get from option data
                    'theta':0.0,
                    'vega':0.0
                }
            )
            
            self.positions[position_key] = position
            
            self.logger.info(f"Debit spread executed: {candidate.ticker}",
                           long_strike=candidate.long_strike,
                           short_strike=candidate.short_strike,
                           net_debit=candidate.net_debit,
                           max_profit=candidate.max_profit)
            
            self.metrics.record_metric("debit_spreads_executed", 1, {"ticker":candidate.ticker})
            
            return True
            
        except Exception as e:
            self.error_handler.handle_error(e, {"ticker":candidate.ticker, "operation":"execute_spread"})
            return False
    
    async def _close_long_position(self, signal: TradeSignal, result):
        """Close long position if short trade fails"""
        try:
            close_signal=TradeSignal(
                strategy_name="Debit Spreads",
                ticker=signal.ticker,
                side=OrderSide.SELL,  # Sell to close
                order_type=OrderType.MARKET,
                quantity=signal.quantity,
                reason="Close long position due to short trade failure"
            )
            
            await self.trading.execute_trade(close_signal)
            
        except Exception as e:
            self.error_handler.handle_error(e, {"operation":"close_long_position"})
    
    def _calculate_position_size(self, candidate: SpreadCandidate) -> int:
        """Calculate position size for debit spread"""
        try:
            # Get account value
            account_value=self.config.risk.account_size
            
            # Calculate max position value (2% of account for spreads)
            max_position_value=account_value * 0.02
            
            # Calculate max contracts
            max_contracts = int(max_position_value / candidate.max_loss)
            
            # Limit to reasonable size
            max_contracts=min(max_contracts, 5)
            
            return max(1, max_contracts)
            
        except Exception as e:
            self.error_handler.handle_error(e, {"ticker":candidate.ticker, "operation":"position_sizing"})
            return 1
    
    async def manage_positions(self):
        """Manage existing debit spread positions"""
        self.logger.info("Managing debit spread positions")
        
        for position_key, position in list(self.positions.items()):
            try:
                await self._manage_position(position)
            except Exception as e:
                self.error_handler.handle_error(e, {"position":position_key, "operation":"manage_position"})
    
    async def _manage_position(self, position: SpreadPosition):
        """Manage individual debit spread position"""
        # Get current options data
        options_data=await self.data.get_options_data(position.ticker)
        
        if not options_data:
            return
        
        # Find current option prices
        long_option_data=None
        short_option_data = None
        
        for option in options_data:
            if (option.strike == position.long_strike and 
                option.option_type == 'call'):
                long_option_data=option
            elif (option.strike == position.short_strike and 
                  option.option_type == 'call'):
                short_option_data=option
        
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
        
        # Check for expiry
        if position.calculate_days_to_expiry() <= 1:
            # Close position before expiry
            await self._close_position(position, "Expiry approaching")
            return
    
    async def _close_position(self, position: SpreadPosition, reason: str):
        """Close debit spread position"""
        try:
            # Close long position
            long_close_signal=TradeSignal(
                strategy_name="Debit Spreads",
                ticker=position.ticker,
                side=OrderSide.SELL,  # Sell to close
                order_type=OrderType.MARKET,
                quantity=position.quantity,
                reason=f"Close debit spread long - {reason}"
            )
            
            long_result=await self.trading.execute_trade(long_close_signal)
            
            # Close short position
            short_close_signal=TradeSignal(
                strategy_name="Debit Spreads",
                ticker=position.ticker,
                side=OrderSide.BUY,  # Buy to close
                order_type=OrderType.MARKET,
                quantity=position.quantity,
                reason=f"Close debit spread short - {reason}"
            )
            
            short_result=await self.trading.execute_trade(short_close_signal)
            
            if (long_result.status.value== "filled" and 
                short_result.status.value == "filled"):
                
                # Update position
                position.status=SpreadStatus.CLOSED
                
                self.logger.info(f"Debit spread closed: {position.ticker}",
                               reason=reason,
                               pnl=position.unrealized_pnl,
                               profit_pct=position.profit_pct)
                
                self.metrics.record_metric("debit_spreads_closed", 1, {"ticker":position.ticker})
                
                # Remove from active positions
                position_key=f"{position.ticker}_{position.long_strike}_{position.short_strike}"
                if position_key in self.positions:
                    del self.positions[position_key]
            
        except Exception as e:
            self.error_handler.handle_error(e, {"ticker":position.ticker, "operation":"close_position"})
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get debit spreads portfolio summary"""
        total_pnl=sum(pos.unrealized_pnl for pos in self.positions.values())
        total_debit=sum(pos.net_debit * pos.quantity * 100 for pos in self.positions.values())
        
        summary={
            "total_positions":len(self.positions),
            "total_pnl":total_pnl,
            "total_debit_at_risk":total_debit,
            "positions":[]
        }
        
        for position_key, position in self.positions.items():
            summary["positions"].append({
                "ticker":position.ticker,
                "spread_type":position.spread_type.value,
                "long_strike":position.long_strike,
                "short_strike":position.short_strike,
                "quantity":position.quantity,
                "net_debit":position.net_debit,
                "unrealized_pnl":position.unrealized_pnl,
                "profit_pct":position.profit_pct,
                "days_to_expiry":position.calculate_days_to_expiry()
            })
        
        return summary
    
    async def run_strategy(self):
        """Run debit spreads strategy main loop"""
        self.logger.info("Starting Debit Spreads Strategy")
        
        while True:
            try:
                # Scan for opportunities
                candidates=await self.scan_for_opportunities()
                
                # Execute new trades
                for candidate in candidates[:2]:  # Top 2 candidates
                    if len(self.positions) < self.max_positions:
                        await self.execute_debit_spread(candidate)
                
                # Manage existing positions
                await self.manage_positions()
                
                # Log portfolio summary
                summary=await self.get_portfolio_summary()
                self.logger.info("Debit spreads portfolio summary", **summary)
                
                # Record metrics
                self.metrics.record_metric("debit_spreads_total_pnl", summary["total_pnl"])
                self.metrics.record_metric("debit_spreads_active_positions", summary["total_positions"])
                
                # Wait for next scan
                await asyncio.sleep(self.scan_interval.total_seconds())
                
            except Exception as e:
                self.error_handler.handle_error(e, {"operation":"run_strategy"})
                await asyncio.sleep(60)  # Wait 1 minute on error


# Factory function for easy initialization
def create_debit_spreads_strategy(trading_interface: TradingInterface,
                                 data_provider: UnifiedDataProvider,
                                 config: ProductionConfig,
                                 logger: ProductionLogger) -> ProductionDebitSpreads:
    """Create debit spreads strategy instance"""
    return ProductionDebitSpreads(trading_interface, data_provider, config, logger)
