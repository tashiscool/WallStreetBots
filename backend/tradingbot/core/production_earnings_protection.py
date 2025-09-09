"""
Production Earnings IV Crush Protection Strategy
Advanced earnings protection with real-time data and sophisticated hedging
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import math
import json

from .production_logging import ProductionLogger
from .production_config import ConfigManager
from .trading_interface import TradingInterface
from .unified_data_provider import UnifiedDataProvider


class EarningsStrategy(Enum):
    """Earnings protection strategies"""
    DEEP_ITM_PROTECTION="deep_itm_protection"
    CALENDAR_SPREAD_PROTECTION = "calendar_spread_protection"
    PROTECTIVE_HEDGE = "protective_hedge"
    VOLATILITY_ARBITRAGE = "volatility_arbitrage"


class EarningsEventType(Enum):
    """Types of earnings events"""
    EARNINGS="earnings"
    GUIDANCE = "guidance"
    DIVIDEND = "dividend"
    SPLIT = "split"
    MERGER = "merger"


@dataclass
class EarningsEvent:
    """Earnings event data"""
    ticker: str
    event_type: EarningsEventType
    event_date: datetime
    announcement_time: str  # "AMC", "BMO", "DURING"
    fiscal_quarter: str
    fiscal_year: int
    eps_estimate: Optional[float] = None
    eps_actual: Optional[float] = None
    revenue_estimate: Optional[float] = None
    revenue_actual: Optional[float] = None
    surprise_pct: Optional[float] = None
    guidance_updated: bool=False
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class IVAnalysis:
    """Implied volatility analysis"""
    ticker: str
    current_iv: float
    historical_iv: float
    iv_percentile: float
    iv_rank: float
    iv_crush_expected: float
    pre_earnings_iv: float
    post_earnings_iv: float
    iv_spike_threshold: float
    analysis_date: datetime=field(default_factory=datetime.now)


@dataclass
class EarningsPosition:
    """Earnings protection position"""
    ticker: str
    strategy: EarningsStrategy
    position_type: str  # "long", "short", "spread"
    quantity: int
    entry_price: float
    current_price: float
    unrealized_pnl: float
    protection_level: float  # Percentage of portfolio protected
    earnings_date: datetime
    days_to_earnings: int
    iv_exposure: float
    delta_exposure: float
    theta_exposure: float
    vega_exposure: float
    max_loss: float
    max_profit: float
    entry_date: datetime=field(default_factory=datetime.now)
    last_update: datetime=field(default_factory=datetime.now)
    status: str="active"


@dataclass
class EarningsCandidate:
    """Earnings protection candidate"""
    ticker: str
    earnings_date: datetime
    days_to_earnings: int
    current_price: float
    iv_rank: float
    iv_percentile: float
    expected_move: float
    protection_cost: float
    protection_ratio: float
    earnings_score: float
    risk_score: float
    strategy_recommended: EarningsStrategy
    last_update: datetime = field(default_factory=datetime.now)


class EarningsDataProvider:
    """Real-time earnings data provider"""
    
    def __init__(self, logger: ProductionLogger):
        self.logger=logger
        self.earnings_cache = {}
        self.iv_cache = {}
    
    async def get_earnings_calendar(self, days_ahead: int=30) -> List[EarningsEvent]:
        """Get upcoming earnings events"""
        try:
            # Mock implementation - in production, integrate with real earnings API
            mock_earnings=[
                EarningsEvent(
                    ticker="AAPL",
                    event_type=EarningsEventType.EARNINGS,
                    event_date=datetime.now() + timedelta(days=5),
                    announcement_time="AMC",
                    fiscal_quarter="Q1",
                    fiscal_year=2024,
                    eps_estimate=2.10,
                    revenue_estimate=120000000000
                ),
                EarningsEvent(
                    ticker="MSFT",
                    event_type=EarningsEventType.EARNINGS,
                    event_date=datetime.now() + timedelta(days=8),
                    announcement_time="AMC",
                    fiscal_quarter="Q1",
                    fiscal_year=2024,
                    eps_estimate=2.85,
                    revenue_estimate=61000000000
                ),
                EarningsEvent(
                    ticker="GOOGL",
                    event_type=EarningsEventType.EARNINGS,
                    event_date=datetime.now() + timedelta(days=12),
                    announcement_time="AMC",
                    fiscal_quarter="Q1",
                    fiscal_year=2024,
                    eps_estimate=1.50,
                    revenue_estimate=85000000000
                )
            ]
            
            self.logger.info(f"Retrieved {len(mock_earnings)} earnings events")
            return mock_earnings
            
        except Exception as e:
            self.logger.error(f"Error fetching earnings calendar: {e}")
            return []
    
    async def get_iv_analysis(self, ticker: str) -> Optional[IVAnalysis]:
        """Get implied volatility analysis for ticker"""
        try:
            # Mock IV analysis - in production, use real options data
            mock_iv=IVAnalysis(
                ticker=ticker,
                current_iv=0.25,
                historical_iv=0.20,
                iv_percentile=0.75,
                iv_rank=0.80,
                iv_crush_expected=0.15,
                pre_earnings_iv=0.30,
                post_earnings_iv=0.15,
                iv_spike_threshold=0.35
            )
            
            self.logger.info(f"Retrieved IV analysis for {ticker}")
            return mock_iv
            
        except Exception as e:
            self.logger.error(f"Error fetching IV analysis for {ticker}: {e}")
            return None
    
    async def get_earnings_history(self, ticker: str, quarters: int=8) -> List[Dict]:
        """Get historical earnings data"""
        try:
            # Mock historical earnings data
            mock_history=[
                {
                    "quarter":"Q4 2023",
                    "eps_estimate":2.05,
                    "eps_actual":2.18,
                    "surprise_pct":0.063,
                    "revenue_estimate":118000000000,
                    "revenue_actual":119600000000,
                    "surprise_pct_revenue":0.014
                },
                {
                    "quarter":"Q3 2023",
                    "eps_estimate":1.95,
                    "eps_actual":1.46,
                    "surprise_pct":-0.251,
                    "revenue_estimate":112000000000,
                    "revenue_actual":109000000000,
                    "surprise_pct_revenue":-0.027
                }
            ]
            
            self.logger.info(f"Retrieved {len(mock_history)} quarters of earnings history for {ticker}")
            return mock_history
            
        except Exception as e:
            self.logger.error(f"Error fetching earnings history for {ticker}: {e}")
            return []


class EarningsProtectionStrategy:
    """Main earnings protection strategy"""
    
    def __init__(self, 
                 trading_interface: TradingInterface,
                 data_provider: UnifiedDataProvider,
                 config: ConfigManager,
                 logger: ProductionLogger):
        self.trading=trading_interface
        self.data = data_provider
        self.config = config
        self.logger = logger
        self.earnings_provider = EarningsDataProvider(logger)
        self.active_positions={}
        self.earnings_candidates = {}
        
        # Strategy parameters
        self.max_earnings_exposure = 0.15  # 15% max exposure to earnings
        self.min_days_to_earnings = 3
        self.max_days_to_earnings = 30
        self.min_iv_rank = 0.60
        self.max_protection_cost = 0.05  # 5% max protection cost
        
        self.logger.info("EarningsProtectionStrategy initialized")
    
    async def scan_for_earnings_opportunities(self) -> List[EarningsCandidate]:
        """Scan for earnings protection opportunities"""
        try:
            self.logger.info("Scanning for earnings protection opportunities")
            
            # Get upcoming earnings events
            earnings_events=await self.earnings_provider.get_earnings_calendar()
            candidates=[]
            
            for event in earnings_events:
                if event.event_type != EarningsEventType.EARNINGS:
                    continue
                
                days_to_earnings = (event.event_date - datetime.now()).days
                
                # Filter by days to earnings
                if not (self.min_days_to_earnings <= days_to_earnings <= self.max_days_to_earnings):
                    continue
                
                # Get IV analysis
                iv_analysis=await self.earnings_provider.get_iv_analysis(event.ticker)
                if not iv_analysis:
                    continue
                
                # Filter by IV rank
                if iv_analysis.iv_rank < self.min_iv_rank:
                    continue
                
                # Get current market data
                market_data=await self.data.get_market_data(event.ticker)
                if not market_data:
                    continue
                
                # Calculate expected move
                expected_move=self._calculate_expected_move(
                    market_data.price, iv_analysis.current_iv, days_to_earnings
                )
                
                # Calculate protection cost
                protection_cost=self._calculate_protection_cost(
                    market_data.price, iv_analysis.current_iv, days_to_earnings
                )
                
                # Filter by protection cost
                if protection_cost > self.max_protection_cost:
                    continue
                
                # Calculate earnings score
                earnings_score=self._calculate_earnings_score(
                    iv_analysis, expected_move, protection_cost, days_to_earnings
                )
                
                # Calculate risk score
                risk_score=self._calculate_risk_score(
                    iv_analysis, expected_move, days_to_earnings
                )
                
                # Determine recommended strategy
                strategy=self._recommend_strategy(iv_analysis, expected_move, days_to_earnings)
                
                candidate=EarningsCandidate(
                    ticker=event.ticker,
                    earnings_date=event.event_date,
                    days_to_earnings=days_to_earnings,
                    current_price=market_data.price,
                    iv_rank=iv_analysis.iv_rank,
                    iv_percentile=iv_analysis.iv_percentile,
                    expected_move=expected_move,
                    protection_cost=protection_cost,
                    protection_ratio=protection_cost / market_data.price,
                    earnings_score=earnings_score,
                    risk_score=risk_score,
                    strategy_recommended=strategy
                )
                
                candidates.append(candidate)
            
            # Sort by earnings score
            candidates.sort(key=lambda x: x.earnings_score, reverse=True)
            
            self.logger.info(f"Found {len(candidates)} earnings protection opportunities")
            return candidates
            
        except Exception as e:
            self.logger.error(f"Error scanning for earnings opportunities: {e}")
            return []
    
    async def execute_earnings_protection(self, candidate: EarningsCandidate) -> Optional[EarningsPosition]:
        """Execute earnings protection strategy"""
        try:
            self.logger.info(f"Executing earnings protection for {candidate.ticker}")
            
            # Check if we already have a position
            if candidate.ticker in self.active_positions:
                self.logger.warning(f"Already have earnings position for {candidate.ticker}")
                return None
            
            # Check portfolio exposure
            if not self._check_portfolio_exposure(candidate):
                self.logger.warning(f"Portfolio exposure limit reached for {candidate.ticker}")
                return None
            
            # Execute strategy based on recommendation
            if candidate.strategy_recommended== EarningsStrategy.DEEP_ITM_PROTECTION:
                position = await self._execute_deep_itm_protection(candidate)
            elif candidate.strategy_recommended== EarningsStrategy.CALENDAR_SPREAD_PROTECTION:
                position = await self._execute_calendar_spread_protection(candidate)
            elif candidate.strategy_recommended== EarningsStrategy.PROTECTIVE_HEDGE:
                position = await self._execute_protective_hedge(candidate)
            elif candidate.strategy_recommended== EarningsStrategy.VOLATILITY_ARBITRAGE:
                position = await self._execute_volatility_arbitrage(candidate)
            else:
                self.logger.error(f"Unknown strategy: {candidate.strategy_recommended}")
                return None
            
            if position:
                self.active_positions[candidate.ticker] = position
                self.logger.info(f"Created earnings protection position for {candidate.ticker}")
            
            return position
            
        except Exception as e:
            self.logger.error(f"Error executing earnings protection for {candidate.ticker}: {e}")
            return None
    
    async def monitor_earnings_positions(self) -> Dict[str, Any]:
        """Monitor active earnings positions"""
        try:
            self.logger.info("Monitoring earnings positions")
            
            monitoring_results={
                "positions_monitored":len(self.active_positions),
                "positions_closed":0,
                "positions_rolled":0,
                "total_pnl":0.0,
                "risk_alerts":[]
            }
            
            positions_to_close=[]
            
            for ticker, position in self.active_positions.items():
                # Update position data
                await self._update_position_data(position)
                
                # Check if earnings has passed
                if position.days_to_earnings <= 0:
                    positions_to_close.append(ticker)
                    continue
                
                # Check for risk alerts
                risk_alerts=self._check_position_risks(position)
                if risk_alerts:
                    monitoring_results["risk_alerts"].extend(risk_alerts)
                
                # Check for profit targets
                if self._check_profit_targets(position):
                    positions_to_close.append(ticker)
                    continue
                
                # Check for roll opportunities
                if self._check_roll_opportunities(position):
                    monitoring_results["positions_rolled"] += 1
                    await self._roll_position(position)
                
                monitoring_results["total_pnl"] += position.unrealized_pnl
            
            # Close positions that need to be closed
            for ticker in positions_to_close:
                await self._close_position(ticker)
                monitoring_results["positions_closed"] += 1
            
            self.logger.info(f"Monitoring complete: {monitoring_results}")
            return monitoring_results
            
        except Exception as e:
            self.logger.error(f"Error monitoring earnings positions: {e}")
            return {"error":str(e)}
    
    def _calculate_expected_move(self, price: float, iv: float, days_to_expiry: int) -> float:
        """Calculate expected move based on IV and time to expiry"""
        time_to_expiry=days_to_expiry / 365.0
        expected_move = price * iv * math.sqrt(time_to_expiry)
        return expected_move
    
    def _calculate_protection_cost(self, price: float, iv: float, days_to_expiry: int) -> float:
        """Calculate protection cost as percentage of stock price"""
        # Simplified calculation - in production, use real options pricing
        time_to_expiry=days_to_expiry / 365.0
        protection_cost = price * iv * math.sqrt(time_to_expiry) * 0.5
        return protection_cost / price
    
    def _calculate_earnings_score(self, iv_analysis: IVAnalysis, 
                                expected_move: float, protection_cost: float, 
                                days_to_earnings: int) -> float:
        """Calculate earnings protection score"""
        score=0.0
        
        # IV rank component (higher is better)
        score += iv_analysis.iv_rank * 0.3
        
        # Expected move component (higher is better)
        move_score=min(expected_move / 0.1, 1.0)  # Cap at 10% move
        score += move_score * 0.25
        
        # Protection cost component (lower is better)
        cost_score=max(0, 1.0 - protection_cost / 0.05)  # Cap at 5% cost
        score += cost_score * 0.25
        
        # Days to earnings component (optimal range)
        if 5 <= days_to_earnings <= 15:
            days_score=1.0
        elif 3 <= days_to_earnings <= 20:
            days_score = 0.8
        else:
            days_score = 0.5
        score += days_score * 0.2
        
        return max(0.0, min(1.0, score))
    
    def _calculate_risk_score(self, iv_analysis: IVAnalysis, 
                            expected_move: float, days_to_earnings: int) -> float:
        """Calculate risk score (lower is better)"""
        risk=0.0
        
        # IV spike risk
        if iv_analysis.current_iv > iv_analysis.iv_spike_threshold:
            risk += 0.3
        
        # Expected move risk
        if expected_move > 0.15:  # 15% move
            risk += 0.3
        
        # Days to earnings risk
        if days_to_earnings < 5:
            risk += 0.2
        elif days_to_earnings > 20:
            risk += 0.1
        
        # IV crush risk
        if iv_analysis.iv_crush_expected > 0.20:  # 20% crush
            risk += 0.2
        
        return max(0.0, min(1.0, risk))
    
    def _recommend_strategy(self, iv_analysis: IVAnalysis, 
                          expected_move: float, days_to_earnings: int) -> EarningsStrategy:
        """Recommend earnings protection strategy"""
        if iv_analysis.iv_rank > 0.80 and expected_move > 0.10:
            return EarningsStrategy.VOLATILITY_ARBITRAGE
        elif days_to_earnings > 15:
            return EarningsStrategy.CALENDAR_SPREAD_PROTECTION
        elif iv_analysis.iv_crush_expected > 0.15:
            return EarningsStrategy.PROTECTIVE_HEDGE
        else:
            return EarningsStrategy.DEEP_ITM_PROTECTION
    
    def _check_portfolio_exposure(self, candidate: EarningsCandidate) -> bool:
        """Check if portfolio exposure limits are respected"""
        # Calculate current earnings exposure
        current_exposure=sum(
            pos.protection_level for pos in self.active_positions.values()
        )
        
        # Add new position exposure
        new_exposure=candidate.protection_ratio
        
        return (current_exposure + new_exposure) <= self.max_earnings_exposure
    
    async def _execute_deep_itm_protection(self, candidate: EarningsCandidate) -> Optional[EarningsPosition]:
        """Execute deep ITM protection strategy"""
        try:
            # Mock implementation - in production, execute real options trades
            position=EarningsPosition(
                ticker=candidate.ticker,
                strategy=EarningsStrategy.DEEP_ITM_PROTECTION,
                position_type="long",
                quantity=100,
                entry_price=candidate.current_price * 0.95,  # 5% below current price
                current_price=candidate.current_price,
                unrealized_pnl=0.0,
                protection_level=candidate.protection_ratio,
                earnings_date=candidate.earnings_date,
                days_to_earnings=candidate.days_to_earnings,
                iv_exposure=0.0,
                delta_exposure=0.8,
                theta_exposure=-0.05,
                vega_exposure=0.0,
                max_loss=candidate.protection_cost * candidate.current_price * 100,
                max_profit=candidate.expected_move * candidate.current_price * 100
            )
            
            self.logger.info(f"Executed deep ITM protection for {candidate.ticker}")
            return position
            
        except Exception as e:
            self.logger.error(f"Error executing deep ITM protection: {e}")
            return None
    
    async def _execute_calendar_spread_protection(self, candidate: EarningsCandidate) -> Optional[EarningsPosition]:
        """Execute calendar spread protection strategy"""
        try:
            # Mock implementation - in production, execute real calendar spreads
            position=EarningsPosition(
                ticker=candidate.ticker,
                strategy=EarningsStrategy.CALENDAR_SPREAD_PROTECTION,
                position_type="spread",
                quantity=10,
                entry_price=candidate.current_price,
                current_price=candidate.current_price,
                unrealized_pnl=0.0,
                protection_level=candidate.protection_ratio,
                earnings_date=candidate.earnings_date,
                days_to_earnings=candidate.days_to_earnings,
                iv_exposure=0.0,
                delta_exposure=0.0,
                theta_exposure=0.1,
                vega_exposure=-0.05,
                max_loss=candidate.protection_cost * candidate.current_price * 100,
                max_profit=candidate.expected_move * candidate.current_price * 100
            )
            
            self.logger.info(f"Executed calendar spread protection for {candidate.ticker}")
            return position
            
        except Exception as e:
            self.logger.error(f"Error executing calendar spread protection: {e}")
            return None
    
    async def _execute_protective_hedge(self, candidate: EarningsCandidate) -> Optional[EarningsPosition]:
        """Execute protective hedge strategy"""
        try:
            # Mock implementation - in production, execute real protective hedges
            position=EarningsPosition(
                ticker=candidate.ticker,
                strategy=EarningsStrategy.PROTECTIVE_HEDGE,
                position_type="hedge",
                quantity=100,
                entry_price=candidate.current_price,
                current_price=candidate.current_price,
                unrealized_pnl=0.0,
                protection_level=candidate.protection_ratio,
                earnings_date=candidate.earnings_date,
                days_to_earnings=candidate.days_to_earnings,
                iv_exposure=0.0,
                delta_exposure=-0.5,
                theta_exposure=-0.02,
                vega_exposure=0.1,
                max_loss=candidate.protection_cost * candidate.current_price * 100,
                max_profit=candidate.expected_move * candidate.current_price * 100
            )
            
            self.logger.info(f"Executed protective hedge for {candidate.ticker}")
            return position
            
        except Exception as e:
            self.logger.error(f"Error executing protective hedge: {e}")
            return None
    
    async def _execute_volatility_arbitrage(self, candidate: EarningsCandidate) -> Optional[EarningsPosition]:
        """Execute volatility arbitrage strategy"""
        try:
            # Mock implementation - in production, execute real volatility arbitrage
            position=EarningsPosition(
                ticker=candidate.ticker,
                strategy=EarningsStrategy.VOLATILITY_ARBITRAGE,
                position_type="arbitrage",
                quantity=20,
                entry_price=candidate.current_price,
                current_price=candidate.current_price,
                unrealized_pnl=0.0,
                protection_level=candidate.protection_ratio,
                earnings_date=candidate.earnings_date,
                days_to_earnings=candidate.days_to_earnings,
                iv_exposure=0.0,
                delta_exposure=0.0,
                theta_exposure=0.05,
                vega_exposure=-0.1,
                max_loss=candidate.protection_cost * candidate.current_price * 100,
                max_profit=candidate.expected_move * candidate.current_price * 100
            )
            
            self.logger.info(f"Executed volatility arbitrage for {candidate.ticker}")
            return position
            
        except Exception as e:
            self.logger.error(f"Error executing volatility arbitrage: {e}")
            return None
    
    async def _update_position_data(self, position: EarningsPosition):
        """Update position data with current market information"""
        try:
            # Get current market data
            market_data=await self.data.get_market_data(position.ticker)
            if market_data:
                position.current_price=market_data.price
                position.last_update = datetime.now()
                
                # Recalculate P&L
                position.unrealized_pnl=self._calculate_position_pnl(position)
                
                # Update days to earnings
                position.days_to_earnings=(position.earnings_date - datetime.now()).days
            
        except Exception as e:
            self.logger.error(f"Error updating position data for {position.ticker}: {e}")
    
    def _calculate_position_pnl(self, position: EarningsPosition) -> float:
        """Calculate position P&L"""
        # Simplified P&L calculation - in production, use real options pricing
        if position.strategy== EarningsStrategy.DEEP_ITM_PROTECTION:
            return (position.current_price - position.entry_price) * position.quantity
        elif position.strategy== EarningsStrategy.CALENDAR_SPREAD_PROTECTION:
            return position.theta_exposure * position.quantity * 10  # Mock theta decay
        elif position.strategy == EarningsStrategy.PROTECTIVE_HEDGE:
            return -position.delta_exposure * (position.current_price - position.entry_price) * position.quantity
        elif position.strategy== EarningsStrategy.VOLATILITY_ARBITRAGE:
            return position.vega_exposure * position.quantity * 5  # Mock vega exposure
        return 0.0
    
    def _check_position_risks(self, position: EarningsPosition) -> List[str]:
        """Check for position risk alerts"""
        alerts=[]
        
        # Check for large unrealized losses
        if position.unrealized_pnl < -position.max_loss * 0.8:
            alerts.append(f"Large unrealized loss for {position.ticker}: ${position.unrealized_pnl:.2f}")
        
        # Check for approaching earnings
        if position.days_to_earnings <= 2:
            alerts.append(f"Earnings approaching for {position.ticker}: {position.days_to_earnings} days")
        
        # Check for high IV exposure
        if abs(position.vega_exposure) > 0.1:
            alerts.append(f"High vega exposure for {position.ticker}: {position.vega_exposure:.3f}")
        
        return alerts
    
    def _check_profit_targets(self, position: EarningsPosition) -> bool:
        """Check if profit targets are met"""
        # Check for profit target (50% of max profit)
        profit_target=position.max_profit * 0.5
        return position.unrealized_pnl >= profit_target
    
    def _check_roll_opportunities(self, position: EarningsPosition) -> bool:
        """Check for roll opportunities"""
        # Roll if we're close to earnings and have profit
        return (position.days_to_earnings <= 3 and 
                position.unrealized_pnl > 0 and 
                position.strategy != EarningsStrategy.VOLATILITY_ARBITRAGE)
    
    async def _roll_position(self, position: EarningsPosition):
        """Roll position to next expiration"""
        try:
            self.logger.info(f"Rolling position for {position.ticker}")
            # Mock implementation - in production, execute roll trades
            position.entry_date=datetime.now()
            position.unrealized_pnl=0.0
            self.logger.info(f"Position rolled for {position.ticker}")
            
        except Exception as e:
            self.logger.error(f"Error rolling position for {position.ticker}: {e}")
    
    async def _close_position(self, ticker: str):
        """Close earnings position"""
        try:
            if ticker in self.active_positions:
                position=self.active_positions.pop(ticker)
                self.logger.info(f"Closed earnings position for {ticker}: P&L ${position.unrealized_pnl:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error closing position for {ticker}: {e}")
    
    async def get_strategy_status(self) -> Dict[str, Any]:
        """Get current strategy status"""
        try:
            total_pnl=sum(pos.unrealized_pnl for pos in self.active_positions.values())
            total_exposure=sum(pos.protection_level for pos in self.active_positions.values())
            
            return {
                "active_positions":len(self.active_positions),
                "total_pnl":total_pnl,
                "total_exposure":total_exposure,
                "max_exposure":self.max_earnings_exposure,
                "positions":[
                    {
                        "ticker":pos.ticker,
                        "strategy":pos.strategy.value,
                        "days_to_earnings":pos.days_to_earnings,
                        "unrealized_pnl":pos.unrealized_pnl,
                        "protection_level":pos.protection_level
                    }
                    for pos in self.active_positions.values()
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting strategy status: {e}")
            return {"error":str(e)}
