"""
Production Momentum Weeklies Scanner
Real-time momentum scanning with weekly options focus
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import math

from .production_logging import ProductionLogger
from .production_config import ConfigManager
from .trading_interface import TradingInterface
from .unified_data_provider import UnifiedDataProvider


class MomentumSignal(Enum):
    """Momentum trading signals"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class MomentumType(Enum):
    """Types of momentum"""
    PRICE_MOMENTUM = "price_momentum"
    VOLUME_MOMENTUM = "volume_momentum"
    EARNINGS_MOMENTUM = "earnings_momentum"
    NEWS_MOMENTUM = "news_momentum"
    TECHNICAL_MOMENTUM = "technical_momentum"


class WeeklyOptionType(Enum):
    """Weekly option types"""
    CALL = "call"
    PUT = "put"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    SPREAD = "spread"


@dataclass
class MomentumData:
    """Momentum analysis data"""
    ticker: str
    current_price: float
    price_change_1d: float
    price_change_5d: float
    price_change_20d: float
    volume_change_1d: float
    volume_change_5d: float
    volume_ratio: float
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    sma_20: float
    sma_50: float
    ema_12: float
    ema_26: float
    bollinger_position: float
    momentum_score: float
    volume_score: float
    technical_score: float
    overall_score: float
    analysis_date: datetime = field(default_factory=datetime.now)


@dataclass
class WeeklyOption:
    """Weekly option data"""
    ticker: str
    option_type: WeeklyOptionType
    strike_price: float
    expiry_date: datetime
    days_to_expiry: int
    bid_price: float
    ask_price: float
    mid_price: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class MomentumCandidate:
    """Momentum trading candidate"""
    ticker: str
    momentum_data: MomentumData
    signal: MomentumSignal
    momentum_type: MomentumType
    confidence: float
    entry_price: float
    target_price: float
    stop_loss: float
    risk_reward_ratio: float
    position_size: int
    recommended_option: Optional[WeeklyOption] = None
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class MomentumPosition:
    """Momentum trading position"""
    ticker: str
    momentum_type: MomentumType
    signal: MomentumSignal
    position_type: str  # "long", "short", "option"
    quantity: int
    entry_price: float
    current_price: float
    unrealized_pnl: float
    target_price: float
    stop_loss: float
    entry_date: datetime
    expiry_date: Optional[datetime] = None
    days_to_expiry: int = 0
    last_update: datetime = field(default_factory=datetime.now)
    status: str = "active"


class MomentumAnalyzer:
    """Momentum analysis engine"""
    
    def __init__(self, logger: ProductionLogger):
        self.logger = logger
    
    def calculate_price_momentum(self, prices: List[float]) -> Dict[str, float]:
        """Calculate price momentum metrics"""
        if len(prices) < 20:
            return {"1d": 0.0, "5d": 0.0, "20d": 0.0}
        
        current_price = prices[-1]
        
        # 1-day momentum
        price_change_1d = (current_price - prices[-2]) / prices[-2] if len(prices) >= 2 else 0.0
        
        # 5-day momentum
        price_change_5d = (current_price - prices[-6]) / prices[-6] if len(prices) >= 6 else 0.0
        
        # 20-day momentum
        price_change_20d = (current_price - prices[-21]) / prices[-21] if len(prices) >= 21 else 0.0
        
        return {
            "1d": price_change_1d,
            "5d": price_change_5d,
            "20d": price_change_20d
        }
    
    def calculate_volume_momentum(self, volumes: List[int]) -> Dict[str, float]:
        """Calculate volume momentum metrics"""
        if len(volumes) < 20:
            return {"1d": 0.0, "5d": 0.0, "ratio": 1.0}
        
        current_volume = volumes[-1]
        
        # 1-day volume change
        volume_change_1d = (current_volume - volumes[-2]) / volumes[-2] if len(volumes) >= 2 and volumes[-2] > 0 else 0.0
        
        # 5-day volume change
        volume_change_5d = (current_volume - volumes[-6]) / volumes[-6] if len(volumes) >= 6 and volumes[-6] > 0 else 0.0
        
        # Volume ratio (current vs 20-day average)
        avg_volume_20d = sum(volumes[-20:]) / 20
        volume_ratio = current_volume / avg_volume_20d if avg_volume_20d > 0 else 1.0
        
        return {
            "1d": volume_change_1d,
            "5d": volume_change_5d,
            "ratio": volume_ratio
        }
    
    def calculate_technical_momentum(self, prices: List[float]) -> Dict[str, float]:
        """Calculate technical momentum indicators"""
        if len(prices) < 50:
            return {"rsi": 50.0, "macd": 0.0, "macd_signal": 0.0, "macd_histogram": 0.0}
        
        # RSI calculation
        rsi = self._calculate_rsi(prices)
        
        # MACD calculation
        macd, macd_signal, macd_histogram = self._calculate_macd(prices)
        
        return {
            "rsi": rsi,
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_histogram": macd_histogram
        }
    
    def calculate_moving_averages(self, prices: List[float]) -> Dict[str, float]:
        """Calculate moving averages"""
        if len(prices) < 50:
            return {"sma_20": prices[-1], "sma_50": prices[-1], "ema_12": prices[-1], "ema_26": prices[-1]}
        
        # Simple moving averages
        sma_20 = sum(prices[-20:]) / 20
        sma_50 = sum(prices[-50:]) / 50
        
        # Exponential moving averages
        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)
        
        return {
            "sma_20": sma_20,
            "sma_50": sma_50,
            "ema_12": ema_12,
            "ema_26": ema_26
        }
    
    def calculate_bollinger_position(self, prices: List[float], period: int = 20, std_dev: float = 2.0) -> float:
        """Calculate position within Bollinger Bands"""
        if len(prices) < period:
            return 0.5
        
        recent_prices = prices[-period:]
        sma = sum(recent_prices) / period
        
        variance = sum((price - sma) ** 2 for price in recent_prices) / period
        std = math.sqrt(variance)
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        if upper_band != lower_band:
            position = (prices[-1] - lower_band) / (upper_band - lower_band)
        else:
            position = 0.5
        
        return max(0.0, min(1.0, position))
    
    def calculate_momentum_score(self, momentum_data: Dict[str, float]) -> float:
        """Calculate overall momentum score"""
        score = 0.0
        
        # Price momentum component (40% weight)
        price_momentum = (momentum_data["1d"] * 0.4 + 
                         momentum_data["5d"] * 0.3 + 
                         momentum_data["20d"] * 0.3)
        score += min(max(price_momentum * 10, -1), 1) * 0.4
        
        # Volume momentum component (30% weight)
        volume_momentum = momentum_data["volume_ratio"]
        if volume_momentum > 1.5:
            score += 0.3
        elif volume_momentum > 1.2:
            score += 0.2
        elif volume_momentum > 1.0:
            score += 0.1
        
        # Technical momentum component (30% weight)
        rsi_score = 0.0
        if 30 <= momentum_data["rsi"] <= 70:
            rsi_score = 0.1
        elif momentum_data["rsi"] > 70:
            rsi_score = 0.05
        
        macd_score = 0.0
        if momentum_data["macd"] > momentum_data["macd_signal"]:
            macd_score = 0.1
        if momentum_data["macd_histogram"] > 0:
            macd_score += 0.1
        
        score += rsi_score + macd_score
        
        return max(0.0, min(1.0, score))
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """Calculate MACD"""
        if len(prices) < slow:
            return 0.0, 0.0, 0.0
        
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        
        macd = ema_fast - ema_slow
        
        # Calculate signal line (EMA of MACD)
        macd_values = [macd] * len(prices)
        signal_line = self._calculate_ema(macd_values, signal)
        
        histogram = macd - signal_line
        
        return macd, signal_line, histogram
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema


class WeeklyOptionsProvider:
    """Weekly options data provider"""
    
    def __init__(self, logger: ProductionLogger):
        self.logger = logger
        self.options_cache = {}
    
    async def get_weekly_options(self, ticker: str, days_to_expiry: int = 7) -> List[WeeklyOption]:
        """Get weekly options for ticker"""
        try:
            # Mock implementation - in production, integrate with real options API
            current_price = 150.0  # Mock current price
            
            # Generate mock weekly options
            options = []
            strikes = [current_price * (1 + i * 0.05) for i in range(-4, 5)]  # ±20% strikes
            
            for strike in strikes:
                # Call option
                call_option = WeeklyOption(
                    ticker=ticker,
                    option_type=WeeklyOptionType.CALL,
                    strike_price=strike,
                    expiry_date=datetime.now() + timedelta(days=days_to_expiry),
                    days_to_expiry=days_to_expiry,
                    bid_price=max(0.01, strike * 0.02),
                    ask_price=max(0.02, strike * 0.025),
                    mid_price=max(0.015, strike * 0.0225),
                    volume=1000,
                    open_interest=5000,
                    implied_volatility=0.25,
                    delta=0.5,
                    gamma=0.01,
                    theta=-0.05,
                    vega=0.1
                )
                options.append(call_option)
                
                # Put option
                put_option = WeeklyOption(
                    ticker=ticker,
                    option_type=WeeklyOptionType.PUT,
                    strike_price=strike,
                    expiry_date=datetime.now() + timedelta(days=days_to_expiry),
                    days_to_expiry=days_to_expiry,
                    bid_price=max(0.01, strike * 0.02),
                    ask_price=max(0.02, strike * 0.025),
                    mid_price=max(0.015, strike * 0.0225),
                    volume=1000,
                    open_interest=5000,
                    implied_volatility=0.25,
                    delta=-0.5,
                    gamma=0.01,
                    theta=-0.05,
                    vega=0.1
                )
                options.append(put_option)
            
            self.logger.info(f"Retrieved {len(options)} weekly options for {ticker}")
            return options
            
        except Exception as e:
            self.logger.error(f"Error fetching weekly options for {ticker}: {e}")
            return []
    
    def find_best_option(self, options: List[WeeklyOption], 
                        momentum_signal: MomentumSignal, 
                        current_price: float) -> Optional[WeeklyOption]:
        """Find best option based on momentum signal"""
        try:
            if not options:
                return None
            
            # Filter options based on momentum signal
            if momentum_signal in [MomentumSignal.STRONG_BUY, MomentumSignal.BUY]:
                # Look for call options or bullish spreads
                call_options = [opt for opt in options if opt.option_type == WeeklyOptionType.CALL]
                if call_options:
                    # Find option with strike closest to current price
                    best_option = min(call_options, key=lambda x: abs(x.strike_price - current_price))
                    return best_option
            
            elif momentum_signal in [MomentumSignal.STRONG_SELL, MomentumSignal.SELL]:
                # Look for put options or bearish spreads
                put_options = [opt for opt in options if opt.option_type == WeeklyOptionType.PUT]
                if put_options:
                    # Find option with strike closest to current price
                    best_option = min(put_options, key=lambda x: abs(x.strike_price - current_price))
                    return best_option
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding best option: {e}")
            return None


class MomentumWeekliesStrategy:
    """Main momentum weeklies strategy"""
    
    def __init__(self, 
                 trading_interface: TradingInterface,
                 data_provider: UnifiedDataProvider,
                 config: ConfigManager,
                 logger: ProductionLogger):
        self.trading = trading_interface
        self.data = data_provider
        self.config = config
        self.logger = logger
        self.momentum_analyzer = MomentumAnalyzer(logger)
        self.options_provider = WeeklyOptionsProvider(logger)
        self.active_positions = {}
        self.momentum_candidates = {}
        
        # Strategy parameters
        self.max_positions = 15
        self.max_position_size = 0.03  # 3% of portfolio per position
        self.min_momentum_score = 0.6
        self.max_hold_days = 7  # Weekly options
        self.stop_loss_pct = 0.15  # 15% stop loss for options
        self.take_profit_pct = 0.30  # 30% take profit for options
        
        self.logger.info("MomentumWeekliesStrategy initialized")
    
    async def scan_for_momentum_opportunities(self) -> List[MomentumCandidate]:
        """Scan for momentum trading opportunities"""
        try:
            self.logger.info("Scanning for momentum opportunities")
            
            # Get universe of stocks to scan
            universe = self.config.trading.universe
            candidates = []
            
            for ticker in universe:
                try:
                    # Get historical data
                    historical_data = await self.data.get_historical_data(ticker, days=50)
                    if not historical_data or len(historical_data) < 20:
                        continue
                    
                    # Perform momentum analysis
                    momentum_data = await self._perform_momentum_analysis(ticker, historical_data)
                    if not momentum_data:
                        continue
                    
                    # Filter by momentum score
                    if momentum_data.overall_score < self.min_momentum_score:
                        continue
                    
                    # Generate momentum signals
                    signal = self._generate_momentum_signal(momentum_data)
                    if signal == MomentumSignal.HOLD:
                        continue
                    
                    # Determine momentum type
                    momentum_type = self._determine_momentum_type(momentum_data)
                    
                    # Create candidate
                    candidate = await self._create_momentum_candidate(
                        ticker, momentum_data, signal, momentum_type
                    )
                    if candidate:
                        candidates.append(candidate)
                
                except Exception as e:
                    self.logger.error(f"Error scanning {ticker}: {e}")
                    continue
            
            # Sort by overall score
            candidates.sort(key=lambda x: x.momentum_data.overall_score, reverse=True)
            
            self.logger.info(f"Found {len(candidates)} momentum opportunities")
            return candidates
            
        except Exception as e:
            self.logger.error(f"Error scanning for momentum opportunities: {e}")
            return []
    
    async def execute_momentum_trade(self, candidate: MomentumCandidate) -> Optional[MomentumPosition]:
        """Execute momentum trading position"""
        try:
            self.logger.info(f"Executing momentum trade for {candidate.ticker}")
            
            # Check if we already have a position
            if candidate.ticker in self.active_positions:
                self.logger.warning(f"Already have momentum position for {candidate.ticker}")
                return None
            
            # Check position limits
            if len(self.active_positions) >= self.max_positions:
                self.logger.warning("Maximum positions reached")
                return None
            
            # Get weekly options
            weekly_options = await self.options_provider.get_weekly_options(candidate.ticker)
            if not weekly_options:
                self.logger.warning(f"No weekly options available for {candidate.ticker}")
                return None
            
            # Find best option
            best_option = self.options_provider.find_best_option(
                weekly_options, candidate.signal, candidate.entry_price
            )
            if not best_option:
                self.logger.warning(f"No suitable option found for {candidate.ticker}")
                return None
            
            # Create position
            position = MomentumPosition(
                ticker=candidate.ticker,
                momentum_type=candidate.momentum_type,
                signal=candidate.signal,
                position_type="option",
                quantity=candidate.position_size,
                entry_price=best_option.mid_price,
                current_price=best_option.mid_price,
                unrealized_pnl=0.0,
                target_price=candidate.target_price,
                stop_loss=candidate.stop_loss,
                entry_date=datetime.now(),
                expiry_date=best_option.expiry_date,
                days_to_expiry=best_option.days_to_expiry
            )
            
            self.active_positions[candidate.ticker] = position
            self.logger.info(f"Created momentum position for {candidate.ticker}")
            
            return position
            
        except Exception as e:
            self.logger.error(f"Error executing momentum trade for {candidate.ticker}: {e}")
            return None
    
    async def monitor_momentum_positions(self) -> Dict[str, Any]:
        """Monitor active momentum positions"""
        try:
            self.logger.info("Monitoring momentum positions")
            
            monitoring_results = {
                "positions_monitored": len(self.active_positions),
                "positions_closed": 0,
                "positions_updated": 0,
                "total_pnl": 0.0,
                "risk_alerts": []
            }
            
            positions_to_close = []
            
            for ticker, position in self.active_positions.items():
                # Update position data
                await self._update_position_data(position)
                
                # Check for exit conditions
                exit_signal = self._check_exit_conditions(position)
                if exit_signal:
                    positions_to_close.append((ticker, exit_signal))
                    continue
                
                # Check for risk alerts
                risk_alerts = self._check_position_risks(position)
                if risk_alerts:
                    monitoring_results["risk_alerts"].extend(risk_alerts)
                
                monitoring_results["positions_updated"] += 1
                monitoring_results["total_pnl"] += position.unrealized_pnl
            
            # Close positions that need to be closed
            for ticker, exit_signal in positions_to_close:
                await self._close_position(ticker, exit_signal)
                monitoring_results["positions_closed"] += 1
            
            self.logger.info(f"Monitoring complete: {monitoring_results}")
            return monitoring_results
            
        except Exception as e:
            self.logger.error(f"Error monitoring momentum positions: {e}")
            return {"error": str(e)}
    
    async def _perform_momentum_analysis(self, ticker: str, historical_data: List[Dict]) -> Optional[MomentumData]:
        """Perform momentum analysis on historical data"""
        try:
            if len(historical_data) < 20:
                return None
            
            # Extract data
            prices = [d["close"] for d in historical_data]
            volumes = [d["volume"] for d in historical_data]
            current_price = prices[-1]
            
            # Calculate momentum metrics
            price_momentum = self.momentum_analyzer.calculate_price_momentum(prices)
            volume_momentum = self.momentum_analyzer.calculate_volume_momentum(volumes)
            technical_momentum = self.momentum_analyzer.calculate_technical_momentum(prices)
            moving_averages = self.momentum_analyzer.calculate_moving_averages(prices)
            bollinger_position = self.momentum_analyzer.calculate_bollinger_position(prices)
            
            # Calculate scores
            momentum_score = self.momentum_analyzer.calculate_momentum_score({
                **price_momentum,
                **volume_momentum,
                **technical_momentum
            })
            
            volume_score = min(volume_momentum["ratio"] / 2.0, 1.0)
            technical_score = (technical_momentum["rsi"] / 100.0 + 
                            (1 if technical_momentum["macd"] > technical_momentum["macd_signal"] else 0)) / 2.0
            
            overall_score = (momentum_score * 0.5 + volume_score * 0.3 + technical_score * 0.2)
            
            momentum_data = MomentumData(
                ticker=ticker,
                current_price=current_price,
                price_change_1d=price_momentum["1d"],
                price_change_5d=price_momentum["5d"],
                price_change_20d=price_momentum["20d"],
                volume_change_1d=volume_momentum["1d"],
                volume_change_5d=volume_momentum["5d"],
                volume_ratio=volume_momentum["ratio"],
                rsi=technical_momentum["rsi"],
                macd=technical_momentum["macd"],
                macd_signal=technical_momentum["macd_signal"],
                macd_histogram=technical_momentum["macd_histogram"],
                sma_20=moving_averages["sma_20"],
                sma_50=moving_averages["sma_50"],
                ema_12=moving_averages["ema_12"],
                ema_26=moving_averages["ema_26"],
                bollinger_position=bollinger_position,
                momentum_score=momentum_score,
                volume_score=volume_score,
                technical_score=technical_score,
                overall_score=overall_score
            )
            
            return momentum_data
            
        except Exception as e:
            self.logger.error(f"Error performing momentum analysis for {ticker}: {e}")
            return None
    
    def _generate_momentum_signal(self, momentum_data: MomentumData) -> MomentumSignal:
        """Generate momentum signal based on analysis"""
        score = momentum_data.overall_score
        
        if score >= 0.8:
            return MomentumSignal.STRONG_BUY
        elif score >= 0.6:
            return MomentumSignal.BUY
        elif score <= 0.2:
            return MomentumSignal.STRONG_SELL
        elif score <= 0.4:
            return MomentumSignal.SELL
        else:
            return MomentumSignal.HOLD
    
    def _determine_momentum_type(self, momentum_data: MomentumData) -> MomentumType:
        """Determine primary momentum type"""
        if momentum_data.volume_ratio > 2.0:
            return MomentumType.VOLUME_MOMENTUM
        elif abs(momentum_data.price_change_5d) > 0.1:
            return MomentumType.PRICE_MOMENTUM
        elif momentum_data.technical_score > 0.7:
            return MomentumType.TECHNICAL_MOMENTUM
        else:
            return MomentumType.PRICE_MOMENTUM
    
    async def _create_momentum_candidate(self, ticker: str, momentum_data: MomentumData, 
                                        signal: MomentumSignal, momentum_type: MomentumType) -> Optional[MomentumCandidate]:
        """Create momentum trading candidate"""
        try:
            # Calculate entry price
            entry_price = momentum_data.current_price
            
            # Calculate target price and stop loss
            if signal in [MomentumSignal.STRONG_BUY, MomentumSignal.BUY]:
                target_price = entry_price * (1 + self.take_profit_pct)
                stop_loss = entry_price * (1 - self.stop_loss_pct)
            else:
                target_price = entry_price * (1 - self.take_profit_pct)
                stop_loss = entry_price * (1 + self.stop_loss_pct)
            
            # Calculate risk/reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(target_price - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # Calculate position size
            position_size = self._calculate_position_size(entry_price, stop_loss)
            
            # Calculate confidence
            confidence = momentum_data.overall_score
            
            candidate = MomentumCandidate(
                ticker=ticker,
                momentum_data=momentum_data,
                signal=signal,
                momentum_type=momentum_type,
                confidence=confidence,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                risk_reward_ratio=risk_reward_ratio,
                position_size=position_size
            )
            
            return candidate
            
        except Exception as e:
            self.logger.error(f"Error creating momentum candidate for {ticker}: {e}")
            return None
    
    def _calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        """Calculate position size based on risk"""
        # Simplified position sizing - in production, use proper risk management
        risk_per_share = abs(entry_price - stop_loss)
        max_risk_amount = 500.0  # $500 max risk per position
        position_size = int(max_risk_amount / risk_per_share) if risk_per_share > 0 else 100
        return min(position_size, 500)  # Cap at 500 shares
    
    async def _update_position_data(self, position: MomentumPosition):
        """Update position data with current market information"""
        try:
            # Get current market data
            market_data = await self.data.get_market_data(position.ticker)
            if market_data:
                position.current_price = market_data.price
                position.last_update = datetime.now()
                
                # Update days to expiry
                if position.expiry_date:
                    position.days_to_expiry = (position.expiry_date - datetime.now()).days
                
                # Recalculate P&L
                position.unrealized_pnl = self._calculate_position_pnl(position)
            
        except Exception as e:
            self.logger.error(f"Error updating position data for {position.ticker}: {e}")
    
    def _calculate_position_pnl(self, position: MomentumPosition) -> float:
        """Calculate position P&L"""
        # Simplified P&L calculation for options
        if position.position_type == "option":
            # Mock option P&L calculation
            price_change = position.current_price - position.entry_price
            return price_change * position.quantity * 100  # Options are per 100 shares
        else:
            # Stock P&L calculation
            return (position.current_price - position.entry_price) * position.quantity
    
    def _check_exit_conditions(self, position: MomentumPosition) -> Optional[str]:
        """Check for exit conditions"""
        # Check stop loss
        if position.position_type == "option":
            if position.current_price <= position.stop_loss:
                return "stop_loss"
        else:
            if position.current_price <= position.stop_loss:
                return "stop_loss"
        
        # Check take profit
        if position.position_type == "option":
            if position.current_price >= position.target_price:
                return "take_profit"
        else:
            if position.current_price >= position.target_price:
                return "take_profit"
        
        # Check expiry
        if position.days_to_expiry <= 0:
            return "expiry"
        
        # Check max hold days
        days_held = (datetime.now() - position.entry_date).days
        if days_held >= self.max_hold_days:
            return "max_hold_days"
        
        return None
    
    def _check_position_risks(self, position: MomentumPosition) -> List[str]:
        """Check for position risk alerts"""
        alerts = []
        
        # Check for large unrealized losses
        if position.unrealized_pnl < -position.quantity * position.entry_price * 0.2:
            alerts.append(f"Large unrealized loss for {position.ticker}: ${position.unrealized_pnl:.2f}")
        
        # Check for approaching expiry
        if position.days_to_expiry <= 1:
            alerts.append(f"Option expiring soon for {position.ticker}: {position.days_to_expiry} days")
        
        # Check for approaching max hold days
        days_held = (datetime.now() - position.entry_date).days
        if days_held >= self.max_hold_days - 1:
            alerts.append(f"Approaching max hold days for {position.ticker}: {days_held} days")
        
        return alerts
    
    async def _close_position(self, ticker: str, exit_signal: str):
        """Close momentum position"""
        try:
            if ticker in self.active_positions:
                position = self.active_positions.pop(ticker)
                self.logger.info(f"Closed momentum position for {ticker}: P&L ${position.unrealized_pnl:.2f}, Signal: {exit_signal}")
            
        except Exception as e:
            self.logger.error(f"Error closing position for {ticker}: {e}")
    
    async def get_strategy_status(self) -> Dict[str, Any]:
        """Get current strategy status"""
        try:
            total_pnl = sum(pos.unrealized_pnl for pos in self.active_positions.values())
            total_exposure = sum(pos.quantity * pos.current_price for pos in self.active_positions.values())
            
            return {
                "active_positions": len(self.active_positions),
                "total_pnl": total_pnl,
                "total_exposure": total_exposure,
                "max_positions": self.max_positions,
                "positions": [
                    {
                        "ticker": pos.ticker,
                        "momentum_type": pos.momentum_type.value,
                        "signal": pos.signal.value,
                        "position_type": pos.position_type,
                        "days_to_expiry": pos.days_to_expiry,
                        "unrealized_pnl": pos.unrealized_pnl
                    }
                    for pos in self.active_positions.values()
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting strategy status: {e}")
            return {"error": str(e)}
