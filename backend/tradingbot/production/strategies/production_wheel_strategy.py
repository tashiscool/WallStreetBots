"""
Production Wheel Strategy Implementation
Automated premium selling with real broker integration and comprehensive risk management
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any, Union
from decimal import Decimal
import math

# Import production infrastructure
from ..core.production_integration import ProductionIntegrationManager, TradeSignal, OrderType, OrderSide
from ..data.production_data_integration import ReliableDataProvider
from ...options.smart_selection import SmartOptionsSelector, OptionsAnalysis, SelectionCriteria, LiquidityRating
from ...options.pricing_engine import OptionsContract, create_options_pricing_engine
from ...risk.real_time_risk_manager import RealTimeRiskManager


@dataclass
class WheelPosition:
    """Production wheel strategy position tracking"""
    ticker: str
    stage: str  # "cash_secured_put", "assigned_stock", "covered_call", "closed"
    status: str  # "active", "expired", "assigned", "closed", "rolled"
    
    # Position details
    quantity: int
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    
    # Options details (if applicable)
    option_strike: Optional[Decimal] = None
    option_expiry: Optional[date] = None
    option_premium: Optional[Decimal] = None
    option_type: Optional[str] = None  # "put" or "call"
    
    # Tracking
    entry_date: datetime = field(default_factory=datetime.now)
    total_premium_collected: Decimal = field(default_factory=lambda: Decimal('0'))
    cycle_number: int = 1
    annualized_return: Optional[Decimal] = None


@dataclass
class WheelCandidate:
    """Wheel strategy candidate analysis"""
    ticker: str
    current_price: Decimal
    iv_rank: Decimal  # 0-100 IV rank
    
    # Put side analysis
    put_strike: Decimal
    put_premium: Decimal
    put_delta: Decimal
    put_expiry: date
    put_dte: int
    
    # Quality metrics
    liquidity_score: Decimal  # 0-100
    volume_score: Decimal  # Recent volume vs average
    earnings_risk: str  # "low", "medium", "high"
    technical_score: Decimal  # 0-100 technical analysis
    
    # Expected returns
    annualized_return: Decimal
    probability_profit: Decimal
    risk_reward_ratio: Decimal


class ProductionWheelStrategy:
    """
    Production Wheel Strategy
    
    Implements the wheel strategy with:
    1. Cash-secured put selling
    2. Stock assignment management  
    3. Covered call selling
    4. Systematic profit taking and rolling
    """
    
    def __init__(
        self, 
        integration_manager: ProductionIntegrationManager,
        data_provider: ReliableDataProvider,
        config: Dict[str, Any]
    ):
        self.integration_manager = integration_manager
        self.data_provider = data_provider
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.options_selector = SmartOptionsSelector(self.data_provider)
        self.options_engine = create_options_pricing_engine()
        self.risk_manager = RealTimeRiskManager()
        
        # Strategy parameters from config
        self.target_iv_rank = config.get('target_iv_rank', 50)  # Minimum IV rank
        self.target_dte_range = config.get('target_dte_range', (30, 45))  # DTE range
        self.target_delta_range = config.get('target_delta_range', (0.15, 0.30))  # Delta range
        self.max_positions = config.get('max_positions', 10)  # Max concurrent positions
        self.min_premium_dollars = config.get('min_premium_dollars', 50)  # Min premium per contract
        self.profit_target = config.get('profit_target', 0.25)  # 25% profit target
        self.max_loss_pct = config.get('max_loss_pct', 0.50)  # 50% max loss
        self.assignment_buffer_days = config.get('assignment_buffer_days', 7)  # Days before expiry to close
        
        # Watchlist - stocks suitable for wheel strategy
        self.watchlist = config.get('watchlist', [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
            'AMD', 'PLTR', 'F', 'BAC', 'T', 'KO', 'PFE', 'INTC', 'CCL', 'AAL'
        ])
        
        # Active positions tracking
        self.positions: Dict[str, WheelPosition] = {}
        
        self.logger.info("ProductionWheelStrategy initialized")
    
    async def scan_opportunities(self) -> List[TradeSignal]:
        """Scan for new wheel opportunities and position management actions"""
        try:
            signals = []
            
            # Manage existing positions first
            management_signals = await self._manage_existing_positions()
            signals.extend(management_signals)
            
            # Look for new opportunities if we have capacity
            if len(self.positions) < self.max_positions:
                new_signals = await self._scan_new_opportunities()
                signals.extend(new_signals)
            
            self.logger.info(f"Generated {len(signals)} wheel strategy signals")
            return signals
            
        except Exception as e:
            self.logger.error(f"Error scanning wheel opportunities: {e}")
            return []
    
    async def _manage_existing_positions(self) -> List[TradeSignal]:
        """Manage existing wheel positions"""
        signals = []
        
        for ticker, position in self.positions.items():
            try:
                # Get current market data
                market_data = await self.data_provider.get_real_time_quote(ticker)
                if not market_data:
                    continue
                    
                current_price = Decimal(str(market_data.price))
                position.current_price = current_price
                
                # Update unrealized P&L
                if position.stage == "cash_secured_put":
                    # P&L from option premium
                    if position.option_premium:
                        option_value = await self._get_current_option_value(position)
                        position.unrealized_pnl = position.option_premium - option_value
                
                elif position.stage == "assigned_stock":
                    # P&L from stock position
                    stock_pnl = (current_price - position.entry_price) * position.quantity
                    position.unrealized_pnl = position.total_premium_collected + stock_pnl
                
                elif position.stage == "covered_call":
                    # P&L from stock + option
                    stock_pnl = (current_price - position.entry_price) * position.quantity
                    if position.option_premium:
                        option_value = await self._get_current_option_value(position)
                        option_pnl = position.option_premium - option_value
                        position.unrealized_pnl = stock_pnl + option_pnl + position.total_premium_collected
                
                # Check for management actions
                management_signal = await self._check_position_management(position)
                if management_signal:
                    signals.append(management_signal)
                    
            except Exception as e:
                self.logger.error(f"Error managing position {ticker}: {e}")
        
        return signals
    
    async def _check_position_management(self, position: WheelPosition) -> Optional[TradeSignal]:
        """Check if position needs management action"""
        try:
            now = datetime.now()
            
            if position.stage == "cash_secured_put":
                # Check if we should close the put for profit or roll
                if position.option_expiry:
                    days_to_expiry = (position.option_expiry - now.date()).days
                    
                    # Close for profit if we hit target
                    if position.unrealized_pnl >= position.option_premium * self.profit_target:
                        return TradeSignal(
                            ticker=position.ticker,
                            action="BUY_TO_CLOSE",
                            order_type=OrderType.MARKET,
                            side=OrderSide.BUY,
                            quantity=abs(position.quantity),
                            reason=f"Wheel put profit target hit: {position.unrealized_pnl}",
                            strategy="wheel_strategy",
                            metadata={
                                "stage": "close_profitable_put",
                                "position_id": f"{position.ticker}_{position.cycle_number}",
                                "profit_pct": float(position.unrealized_pnl / position.option_premium)
                            }
                        )
                    
                    # Close if too close to expiry to avoid assignment
                    elif days_to_expiry <= self.assignment_buffer_days:
                        return TradeSignal(
                            ticker=position.ticker,
                            action="BUY_TO_CLOSE",
                            order_type=OrderType.MARKET,
                            side=OrderSide.BUY,
                            quantity=abs(position.quantity),
                            reason=f"Wheel put close before assignment: {days_to_expiry} DTE",
                            strategy="wheel_strategy",
                            metadata={
                                "stage": "avoid_assignment",
                                "position_id": f"{position.ticker}_{position.cycle_number}",
                                "days_to_expiry": days_to_expiry
                            }
                        )
            
            elif position.stage == "covered_call":
                # Check if we should close the call for profit
                if position.unrealized_pnl >= position.option_premium * self.profit_target:
                    return TradeSignal(
                        ticker=position.ticker,
                        action="BUY_TO_CLOSE",
                        order_type=OrderType.MARKET,
                        side=OrderSide.BUY,
                        quantity=abs(position.quantity),
                        reason=f"Wheel call profit target hit: {position.unrealized_pnl}",
                        strategy="wheel_strategy",
                        metadata={
                            "stage": "close_profitable_call",
                            "position_id": f"{position.ticker}_{position.cycle_number}",
                            "profit_pct": float(position.unrealized_pnl / position.option_premium) if position.option_premium else 0
                        }
                    )
            
            elif position.stage == "assigned_stock":
                # Look for covered call opportunity
                call_signal = await self._find_covered_call_opportunity(position)
                if call_signal:
                    return call_signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking position management for {position.ticker}: {e}")
            return None
    
    async def _scan_new_opportunities(self) -> List[TradeSignal]:
        """Scan for new cash-secured put opportunities"""
        signals = []
        
        try:
            candidates = []
            
            # Analyze each ticker in watchlist
            for ticker in self.watchlist:
                if ticker in self.positions:
                    continue  # Skip if we already have a position
                
                candidate = await self._analyze_wheel_candidate(ticker)
                if candidate:
                    candidates.append(candidate)
            
            # Sort by best opportunities (highest annualized return with good risk metrics)
            candidates.sort(key=lambda x: x.annualized_return, reverse=True)
            
            # Generate signals for top candidates
            max_new_positions = self.max_positions - len(self.positions)
            for candidate in candidates[:max_new_positions]:
                signal = self._create_cash_secured_put_signal(candidate)
                if signal:
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error scanning new wheel opportunities: {e}")
            return []
    
    async def _analyze_wheel_candidate(self, ticker: str) -> Optional[WheelCandidate]:
        """Analyze a ticker for wheel strategy suitability"""
        try:
            # Get market data
            market_data = await self.data_provider.get_real_time_quote(ticker)
            if not market_data:
                return None
            
            current_price = Decimal(str(market_data.price))
            
            # Get IV rank (simplified - would use real IV rank calculation)
            iv_data = await self.data_provider.get_implied_volatility(ticker)
            if not iv_data or iv_data < self.target_iv_rank:
                return None
            
            iv_rank = Decimal(str(iv_data))
            
            # Find suitable put options
            put_analysis = await self._find_optimal_put(ticker, current_price)
            if not put_analysis:
                return None
            
            # Calculate quality metrics
            liquidity_score = await self._calculate_liquidity_score(ticker)
            volume_score = await self._calculate_volume_score(ticker)
            earnings_risk = await self._assess_earnings_risk(ticker)
            technical_score = await self._calculate_technical_score(ticker)
            
            # Calculate expected returns
            annualized_return = self._calculate_annualized_return(put_analysis)
            probability_profit = Decimal(str(1.0 - abs(put_analysis.greeks.delta))) if put_analysis.greeks else Decimal('0.7')
            
            return WheelCandidate(
                ticker=ticker,
                current_price=current_price,
                iv_rank=iv_rank,
                put_strike=put_analysis.contract.strike,
                put_premium=put_analysis.mid_price,
                put_delta=Decimal(str(put_analysis.greeks.delta)) if put_analysis.greeks else Decimal('-0.25'),
                put_expiry=put_analysis.contract.expiry_date,
                put_dte=(put_analysis.contract.expiry_date - date.today()).days,
                liquidity_score=liquidity_score,
                volume_score=volume_score,
                earnings_risk=earnings_risk,
                technical_score=technical_score,
                annualized_return=annualized_return,
                probability_profit=probability_profit,
                risk_reward_ratio=annualized_return / Decimal('100')  # Simplified risk-reward
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing wheel candidate {ticker}: {e}")
            return None
    
    async def _find_optimal_put(self, ticker: str, current_price: Decimal) -> Optional[OptionsAnalysis]:
        """Find optimal put for cash-secured put strategy"""
        try:
            # Define criteria for cash-secured puts
            criteria = SelectionCriteria(
                target_dte_min=self.target_dte_range[0],
                target_dte_max=self.target_dte_range[1],
                target_delta_min=abs(self.target_delta_range[0]),  # Delta for puts is negative
                target_delta_max=abs(self.target_delta_range[1]),
                min_volume=50,
                min_open_interest=100,
                max_bid_ask_spread_pct=Decimal('0.10'),  # Max 10% spread
                min_premium_dollars=self.min_premium_dollars
            )
            
            # Find optimal put
            put_analysis = await self.options_selector.select_optimal_put_option(
                ticker, current_price, criteria
            )
            
            return put_analysis
            
        except Exception as e:
            self.logger.error(f"Error finding optimal put for {ticker}: {e}")
            return None
    
    async def _find_covered_call_opportunity(self, position: WheelPosition) -> Optional[TradeSignal]:
        """Find covered call opportunity for assigned stock"""
        try:
            # Define criteria for covered calls
            criteria = SelectionCriteria(
                target_dte_min=self.target_dte_range[0],
                target_dte_max=self.target_dte_range[1],
                target_delta_min=self.target_delta_range[0],
                target_delta_max=self.target_delta_range[1],
                min_volume=50,
                min_open_interest=100,
                max_bid_ask_spread_pct=Decimal('0.10'),
                min_premium_dollars=self.min_premium_dollars
            )
            
            # Find optimal call above our cost basis
            min_strike = max(position.entry_price, position.current_price * Decimal('1.02'))  # 2% above current
            
            call_analysis = await self.options_selector.select_optimal_call_option(
                position.ticker, position.current_price, criteria
            )
            
            if call_analysis and call_analysis.contract.strike >= min_strike:
                return TradeSignal(
                    ticker=position.ticker,
                    action="SELL_TO_OPEN",
                    order_type=OrderType.LIMIT,
                    side=OrderSide.SELL,
                    quantity=abs(position.quantity) // 100,  # Convert shares to contracts
                    price=float(call_analysis.mid_price),
                    reason=f"Wheel covered call: {call_analysis.contract.strike} strike",
                    strategy="wheel_strategy",
                    metadata={
                        "stage": "covered_call",
                        "position_id": f"{position.ticker}_{position.cycle_number}",
                        "strike": float(call_analysis.contract.strike),
                        "expiry": call_analysis.contract.expiry_date.isoformat(),
                        "premium": float(call_analysis.mid_price),
                        "delta": float(call_analysis.greeks.delta) if call_analysis.greeks else None
                    }
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding covered call for {position.ticker}: {e}")
            return None
    
    def _create_cash_secured_put_signal(self, candidate: WheelCandidate) -> Optional[TradeSignal]:
        """Create trade signal for cash-secured put"""
        try:
            return TradeSignal(
                ticker=candidate.ticker,
                action="SELL_TO_OPEN",
                order_type=OrderType.LIMIT,
                side=OrderSide.SELL,
                quantity=1,  # 1 contract
                price=float(candidate.put_premium),
                reason=f"Wheel cash-secured put: {candidate.annualized_return:.1f}% annualized",
                strategy="wheel_strategy",
                metadata={
                    "stage": "cash_secured_put",
                    "strike": float(candidate.put_strike),
                    "expiry": candidate.put_expiry.isoformat(),
                    "premium": float(candidate.put_premium),
                    "delta": float(candidate.put_delta),
                    "iv_rank": float(candidate.iv_rank),
                    "annualized_return": float(candidate.annualized_return),
                    "liquidity_score": float(candidate.liquidity_score)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error creating cash-secured put signal for {candidate.ticker}: {e}")
            return None
    
    async def execute_trades(self, signals: List[TradeSignal]) -> List[Dict[str, Any]]:
        """Execute wheel strategy trades"""
        results = []
        
        for signal in signals:
            try:
                # Execute the trade
                result = await self.integration_manager.execute_trade(signal)
                results.append(result)
                
                # Update position tracking
                await self._update_position_after_trade(signal, result)
                
            except Exception as e:
                self.logger.error(f"Error executing wheel trade {signal.ticker}: {e}")
                results.append({"status": "error", "error": str(e)})
        
        return results
    
    async def _update_position_after_trade(self, signal: TradeSignal, result: Dict[str, Any]):
        """Update position tracking after trade execution"""
        try:
            if result.get("status") != "success":
                return
            
            ticker = signal.ticker
            metadata = signal.metadata or {}
            stage = metadata.get("stage", "unknown")
            
            if stage == "cash_secured_put":
                # New cash-secured put position
                self.positions[ticker] = WheelPosition(
                    ticker=ticker,
                    stage="cash_secured_put",
                    status="active",
                    quantity=-signal.quantity * 100,  # Negative for short option
                    entry_price=Decimal(str(signal.price or 0)),
                    current_price=Decimal(str(signal.price or 0)),
                    unrealized_pnl=Decimal('0'),
                    option_strike=Decimal(str(metadata.get("strike", 0))),
                    option_expiry=datetime.fromisoformat(metadata.get("expiry", "")).date(),
                    option_premium=Decimal(str(metadata.get("premium", 0))),
                    option_type="put",
                    total_premium_collected=Decimal(str(metadata.get("premium", 0)))
                )
                
            elif stage == "covered_call":
                # Add covered call to existing position
                if ticker in self.positions:
                    position = self.positions[ticker]
                    position.stage = "covered_call"
                    position.option_strike = Decimal(str(metadata.get("strike", 0)))
                    position.option_expiry = datetime.fromisoformat(metadata.get("expiry", "")).date()
                    position.option_premium = Decimal(str(metadata.get("premium", 0)))
                    position.option_type = "call"
                    position.total_premium_collected += Decimal(str(metadata.get("premium", 0)))
            
            elif stage in ["close_profitable_put", "close_profitable_call", "avoid_assignment"]:
                # Position closed
                if ticker in self.positions:
                    self.positions[ticker].status = "closed"
                    # Could move to historical positions here
            
            self.logger.info(f"Updated wheel position for {ticker}: {stage}")
            
        except Exception as e:
            self.logger.error(f"Error updating position after trade {ticker}: {e}")
    
    async def _get_current_option_value(self, position: WheelPosition) -> Decimal:
        """Get current value of option position"""
        try:
            if not position.option_strike or not position.option_expiry:
                return Decimal('0')
            
            # Get current option price from market
            option_data = await self.data_provider.get_options_chain(
                position.ticker, position.option_expiry
            )
            
            if option_data:
                for contract in option_data:
                    if (contract.strike == position.option_strike and 
                        contract.option_type == position.option_type):
                        return contract.mid_price if hasattr(contract, 'mid_price') else Decimal('0')
            
            return Decimal('0')
            
        except Exception as e:
            self.logger.error(f"Error getting current option value for {position.ticker}: {e}")
            return Decimal('0')
    
    def _calculate_annualized_return(self, analysis: OptionsAnalysis) -> Decimal:
        """Calculate annualized return for wheel strategy"""
        try:
            days_to_expiry = (analysis.contract.expiry_date - date.today()).days
            if days_to_expiry <= 0:
                return Decimal('0')
            
            # Return = (Premium / Strike) * (365 / DTE)
            premium_return = analysis.mid_price / analysis.contract.strike
            annualized = premium_return * Decimal(str(365 / days_to_expiry))
            
            return annualized * Decimal('100')  # Convert to percentage
            
        except Exception as e:
            self.logger.error(f"Error calculating annualized return: {e}")
            return Decimal('0')
    
    async def _calculate_liquidity_score(self, ticker: str) -> Decimal:
        """Calculate liquidity score (0-100)"""
        # Simplified - would analyze bid-ask spreads, volume, open interest
        return Decimal('75')  # Default good liquidity
    
    async def _calculate_volume_score(self, ticker: str) -> Decimal:
        """Calculate volume score vs average"""
        # Simplified - would compare recent volume to average
        return Decimal('100')  # Default normal volume
    
    async def _assess_earnings_risk(self, ticker: str) -> str:
        """Assess earnings announcement risk"""
        # Simplified - would check earnings calendar
        return "low"  # Default low risk
    
    async def _calculate_technical_score(self, ticker: str) -> Decimal:
        """Calculate technical analysis score"""
        # Simplified - would analyze trend, support/resistance, etc.
        return Decimal('70')  # Default neutral technical
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get wheel strategy performance metrics"""
        try:
            total_positions = len(self.positions)
            active_positions = sum(1 for p in self.positions.values() if p.status == "active")
            
            total_unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
            total_premium_collected = sum(p.total_premium_collected for p in self.positions.values())
            
            return {
                "strategy": "wheel_strategy",
                "total_positions": total_positions,
                "active_positions": active_positions,
                "total_unrealized_pnl": float(total_unrealized_pnl),
                "total_premium_collected": float(total_premium_collected),
                "avg_premium_per_position": float(total_premium_collected / total_positions) if total_positions > 0 else 0,
                "positions": {ticker: {
                    "stage": pos.stage,
                    "status": pos.status,
                    "unrealized_pnl": float(pos.unrealized_pnl),
                    "total_premium": float(pos.total_premium_collected),
                    "cycle": pos.cycle_number
                } for ticker, pos in self.positions.items()}
            }
            
        except Exception as e:
            self.logger.error(f"Error getting wheel performance metrics: {e}")
            return {"strategy": "wheel_strategy", "error": str(e)}
    
    async def run_strategy(self):
        """Main strategy execution loop"""
        self.logger.info("Starting Production Wheel Strategy")
        
        try:
            while True:
                # Scan for wheel opportunities
                signals = await self.scan_opportunities()
                
                # Execute trades for signals
                if signals:
                    await self.execute_trades(signals)
                
                # Wait before next scan (wheel strategy runs less frequently)
                await asyncio.sleep(300)  # 5 minutes between scans
                
        except Exception as e:
            self.logger.error(f"Error in wheel strategy main loop: {e}")


def create_production_wheel_strategy(
    integration_manager: ProductionIntegrationManager,
    data_provider: ReliableDataProvider,
    config: Dict[str, Any]
) -> ProductionWheelStrategy:
    """Factory function to create production wheel strategy"""
    return ProductionWheelStrategy(integration_manager, data_provider, config)