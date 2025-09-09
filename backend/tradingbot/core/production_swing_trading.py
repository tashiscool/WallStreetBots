"""
Production Enhanced Swing Trading Strategy
Advanced swing trading with technical analysis and risk management
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import math
import numpy as np

from .production_logging import ProductionLogger
from .production_config import ConfigManager
from .trading_interface import TradingInterface
from .unified_data_provider import UnifiedDataProvider


class SwingSignal(Enum):
    """Swing trading signals"""
    BUY="buy"
    SELL = "sell"
    HOLD = "hold"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"


class SwingStrategy(Enum):
    """Swing trading strategies"""
    BREAKOUT="breakout"
    PULLBACK = "pullback"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    MOMENTUM = "momentum"


class TechnicalIndicator(Enum):
    """Technical indicators"""
    RSI="rsi"
    MACD = "macd"
    BOLLINGER_BANDS = "bollinger_bands"
    MOVING_AVERAGE = "moving_average"
    STOCHASTIC = "stochastic"
    WILLIAMS_R = "williams_r"
    CCI = "cci"
    ADX = "adx"


@dataclass
class TechnicalAnalysis:
    """Technical analysis data"""
    ticker: str
    current_price: float
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width: float
    bb_position: float  # Position within bands (0-1)
    sma_20: float
    sma_50: float
    sma_200: float
    ema_12: float
    ema_26: float
    stochastic_k: float
    stochastic_d: float
    williams_r: float
    cci: float
    adx: float
    volume_sma: float
    volume_ratio: float
    analysis_date: datetime=field(default_factory=datetime.now)


@dataclass
class SwingPosition:
    """Swing trading position"""
    ticker: str
    strategy: SwingStrategy
    signal: SwingSignal
    position_type: str  # "long", "short"
    quantity: int
    entry_price: float
    current_price: float
    unrealized_pnl: float
    stop_loss: float
    take_profit: float
    trailing_stop: float
    max_favorable_move: float
    max_adverse_move: float
    entry_date: datetime
    last_update: datetime=field(default_factory=datetime.now)
    days_held: int=0
    status: str = "active"
    risk_reward_ratio: float = 0.0
    technical_score: float = 0.0


@dataclass
class SwingCandidate:
    """Swing trading candidate"""
    ticker: str
    current_price: float
    signal: SwingSignal
    strategy: SwingStrategy
    technical_score: float
    risk_score: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    position_size: int
    confidence: float
    last_update: datetime = field(default_factory=datetime.now)


class TechnicalAnalyzer:
    """Technical analysis engine"""
    
    def __init__(self, logger: ProductionLogger):
        self.logger=logger
    
    def calculate_rsi(self, prices: List[float], period: int=14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas=[prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains=[d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss=sum(losses[-period:]) / period
        
        if avg_loss== 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: List[float], fast: int=12, slow: int=26, signal: int=9) -> Tuple[float, float, float]:
        """Calculate MACD"""
        if len(prices) < slow:
            return 0.0, 0.0, 0.0
        
        # Calculate EMAs
        ema_fast=self._calculate_ema(prices, fast)
        ema_slow=self._calculate_ema(prices, slow)
        
        macd=ema_fast - ema_slow
        
        # Calculate signal line (EMA of MACD)
        macd_values=[macd] * len(prices)
        signal_line=self._calculate_ema(macd_values, signal)
        
        histogram=macd - signal_line
        
        return macd, signal_line, histogram
    
    def calculate_bollinger_bands(self, prices: List[float], period: int=20, std_dev: float=2.0) -> Tuple[float, float, float, float, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        recent_prices=prices[-period:]
        sma = sum(recent_prices) / period
        
        variance=sum((price - sma) ** 2 for price in recent_prices) / period
        std=math.sqrt(variance)
        
        upper_band=sma + (std_dev * std)
        lower_band=sma - (std_dev * std)
        width=(upper_band - lower_band) / sma
        
        # Position within bands (0-1)
        if upper_band != lower_band:
            position=(prices[-1] - lower_band) / (upper_band - lower_band)
        else:
            position=0.5
        
        return upper_band, sma, lower_band, width, position
    
    def calculate_moving_averages(self, prices: List[float]) -> Tuple[float, float, float, float, float]:
        """Calculate various moving averages"""
        if len(prices) < 200:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        sma_20=sum(prices[-20:]) / 20
        sma_50=sum(prices[-50:]) / 50
        sma_200=sum(prices[-200:]) / 200
        
        ema_12=self._calculate_ema(prices, 12)
        ema_26=self._calculate_ema(prices, 26)
        
        return sma_20, sma_50, sma_200, ema_12, ema_26
    
    def calculate_stochastic(self, high_prices: List[float], low_prices: List[float], 
                           close_prices: List[float], period: int=14) -> Tuple[float, float]:
        """Calculate Stochastic Oscillator"""
        if len(close_prices) < period:
            return 50.0, 50.0
        
        recent_highs=high_prices[-period:]
        recent_lows = low_prices[-period:]
        current_close = close_prices[-1]
        
        highest_high = max(recent_highs)
        lowest_low=min(recent_lows)
        
        if highest_high== lowest_low:
            return 50.0, 50.0
        
        k_percent=((current_close - lowest_low) / (highest_high - lowest_low)) * 100
        
        # Simple moving average of K% for D%
        if len(close_prices) >= period * 2:
            k_values=[]
            for i in range(period, len(close_prices)):
                period_highs=high_prices[i-period:i]
                period_lows = low_prices[i-period:i]
                period_close = close_prices[i]
                
                hh = max(period_highs)
                ll=min(period_lows)
                
                if hh != ll:
                    k_val=((period_close - ll) / (hh - ll)) * 100
                    k_values.append(k_val)
            
            d_percent=sum(k_values[-3:]) / 3 if len(k_values) >= 3 else k_percent
        else:
            d_percent=k_percent
        
        return k_percent, d_percent
    
    def calculate_williams_r(self, high_prices: List[float], low_prices: List[float], 
                            close_prices: List[float], period: int=14) -> float:
        """Calculate Williams %R"""
        if len(close_prices) < period:
            return -50.0
        
        recent_highs=high_prices[-period:]
        recent_lows = low_prices[-period:]
        current_close = close_prices[-1]
        
        highest_high = max(recent_highs)
        lowest_low=min(recent_lows)
        
        if highest_high== lowest_low:
            return -50.0
        
        williams_r = ((highest_high - current_close) / (highest_high - lowest_low)) * -100
        return williams_r
    
    def calculate_cci(self, high_prices: List[float], low_prices: List[float], 
                     close_prices: List[float], period: int=20) -> float:
        """Calculate Commodity Channel Index"""
        if len(close_prices) < period:
            return 0.0
        
        recent_highs=high_prices[-period:]
        recent_lows = low_prices[-period:]
        recent_closes = close_prices[-period:]
        
        typical_prices = [(h + l + c) / 3 for h, l, c in zip(recent_highs, recent_lows, recent_closes)]
        sma_tp=sum(typical_prices) / period
        
        mean_deviation=sum(abs(tp - sma_tp) for tp in typical_prices) / period
        
        if mean_deviation== 0:
            return 0.0
        
        cci = (typical_prices[-1] - sma_tp) / (0.015 * mean_deviation)
        return cci
    
    def calculate_adx(self, high_prices: List[float], low_prices: List[float], 
                     close_prices: List[float], period: int=14) -> float:
        """Calculate Average Directional Index"""
        if len(close_prices) < period * 2:
            return 25.0
        
        # Simplified ADX calculation
        # In production, use a proper ADX implementation
        return 25.0  # Mock value
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        
        multiplier=2 / (period + 1)
        ema=prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema


class SwingTradingStrategy:
    """Main swing trading strategy"""
    
    def __init__(self, 
                 trading_interface: TradingInterface,
                 data_provider: UnifiedDataProvider,
                 config: ConfigManager,
                 logger: ProductionLogger):
        self.trading=trading_interface
        self.data = data_provider
        self.config = config
        self.logger = logger
        self.technical_analyzer = TechnicalAnalyzer(logger)
        self.active_positions={}
        self.swing_candidates = {}
        
        # Strategy parameters
        self.max_positions = 10
        self.max_position_size = 0.05  # 5% of portfolio per position
        self.min_risk_reward_ratio = 2.0
        self.max_hold_days = 30
        self.stop_loss_pct = 0.08  # 8% stop loss
        self.take_profit_pct = 0.16  # 16% take profit
        
        self.logger.info("SwingTradingStrategy initialized")
    
    async def scan_for_swing_opportunities(self) -> List[SwingCandidate]:
        """Scan for swing trading opportunities"""
        try:
            self.logger.info("Scanning for swing trading opportunities")
            
            # Get universe of stocks to scan
            universe=self.config.trading.universe
            candidates = []
            
            for ticker in universe:
                try:
                    # Get historical data
                    historical_data = await self.data.get_historical_data(ticker, days=200)
                    if not historical_data or len(historical_data) < 50:
                        continue
                    
                    # Perform technical analysis
                    technical_analysis=await self._perform_technical_analysis(ticker, historical_data)
                    if not technical_analysis:
                        continue
                    
                    # Generate swing signals
                    signals=self._generate_swing_signals(technical_analysis)
                    
                    for signal_data in signals:
                        if signal_data["signal"] in [SwingSignal.BUY, SwingSignal.SELL]:
                            candidate=await self._create_swing_candidate(
                                ticker, technical_analysis, signal_data
                            )
                            if candidate:
                                candidates.append(candidate)
                
                except Exception as e:
                    self.logger.error(f"Error scanning {ticker}: {e}")
                    continue
            
            # Sort by technical score
            candidates.sort(key=lambda x: x.technical_score, reverse=True)
            
            self.logger.info(f"Found {len(candidates)} swing trading opportunities")
            return candidates
            
        except Exception as e:
            self.logger.error(f"Error scanning for swing opportunities: {e}")
            return []
    
    async def execute_swing_trade(self, candidate: SwingCandidate) -> Optional[SwingPosition]:
        """Execute swing trading position"""
        try:
            self.logger.info(f"Executing swing trade for {candidate.ticker}")
            
            # Check if we already have a position
            if candidate.ticker in self.active_positions:
                self.logger.warning(f"Already have swing position for {candidate.ticker}")
                return None
            
            # Check position limits
            if len(self.active_positions) >= self.max_positions:
                self.logger.warning("Maximum positions reached")
                return None
            
            # Check risk/reward ratio
            if candidate.risk_reward_ratio < self.min_risk_reward_ratio:
                self.logger.warning(f"Risk/reward ratio too low for {candidate.ticker}")
                return None
            
            # Create position
            position=SwingPosition(
                ticker=candidate.ticker,
                strategy=candidate.strategy,
                signal=candidate.signal,
                position_type="long" if candidate.signal == SwingSignal.BUY else "short",
                quantity=candidate.position_size,
                entry_price=candidate.entry_price,
                current_price=candidate.current_price,
                unrealized_pnl=0.0,
                stop_loss=candidate.stop_loss,
                take_profit=candidate.take_profit,
                trailing_stop=candidate.entry_price,
                max_favorable_move=0.0,
                max_adverse_move=0.0,
                entry_date=datetime.now(),
                risk_reward_ratio=candidate.risk_reward_ratio,
                technical_score=candidate.technical_score
            )
            
            self.active_positions[candidate.ticker] = position
            self.logger.info(f"Created swing position for {candidate.ticker}")
            
            return position
            
        except Exception as e:
            self.logger.error(f"Error executing swing trade for {candidate.ticker}: {e}")
            return None
    
    async def monitor_swing_positions(self) -> Dict[str, Any]:
        """Monitor active swing positions"""
        try:
            self.logger.info("Monitoring swing positions")
            
            monitoring_results={
                "positions_monitored":len(self.active_positions),
                "positions_closed":0,
                "positions_updated":0,
                "total_pnl":0.0,
                "risk_alerts":[]
            }
            
            positions_to_close=[]
            
            for ticker, position in self.active_positions.items():
                # Update position data
                await self._update_position_data(position)
                
                # Check for exit conditions
                exit_signal=self._check_exit_conditions(position)
                if exit_signal:
                    positions_to_close.append((ticker, exit_signal))
                    continue
                
                # Update trailing stop
                self._update_trailing_stop(position)
                
                # Check for risk alerts
                risk_alerts=self._check_position_risks(position)
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
            self.logger.error(f"Error monitoring swing positions: {e}")
            return {"error":str(e)}
    
    async def _perform_technical_analysis(self, ticker: str, historical_data: List[Dict]) -> Optional[TechnicalAnalysis]:
        """Perform technical analysis on historical data"""
        try:
            if len(historical_data) < 50:
                return None
            
            # Extract price data
            closes=[d["close"] for d in historical_data]
            highs = [d["high"] for d in historical_data]
            lows = [d["low"] for d in historical_data]
            volumes = [d["volume"] for d in historical_data]
            
            current_price = closes[-1]
            
            # Calculate technical indicators
            rsi = self.technical_analyzer.calculate_rsi(closes)
            macd, macd_signal, macd_histogram=self.technical_analyzer.calculate_macd(closes)
            bb_upper, bb_middle, bb_lower, bb_width, bb_position=self.technical_analyzer.calculate_bollinger_bands(closes)
            sma_20, sma_50, sma_200, ema_12, ema_26=self.technical_analyzer.calculate_moving_averages(closes)
            stochastic_k, stochastic_d=self.technical_analyzer.calculate_stochastic(highs, lows, closes)
            williams_r=self.technical_analyzer.calculate_williams_r(highs, lows, closes)
            cci=self.technical_analyzer.calculate_cci(highs, lows, closes)
            adx=self.technical_analyzer.calculate_adx(highs, lows, closes)
            
            # Calculate volume metrics
            volume_sma=sum(volumes[-20:]) / 20 if len(volumes) >= 20 else volumes[-1]
            volume_ratio=volumes[-1] / volume_sma if volume_sma > 0 else 1.0
            
            analysis = TechnicalAnalysis(
                ticker=ticker,
                current_price=current_price,
                rsi=rsi,
                macd=macd,
                macd_signal=macd_signal,
                macd_histogram=macd_histogram,
                bb_upper=bb_upper,
                bb_middle=bb_middle,
                bb_lower=bb_lower,
                bb_width=bb_width,
                bb_position=bb_position,
                sma_20=sma_20,
                sma_50=sma_50,
                sma_200=sma_200,
                ema_12=ema_12,
                ema_26=ema_26,
                stochastic_k=stochastic_k,
                stochastic_d=stochastic_d,
                williams_r=williams_r,
                cci=cci,
                adx=adx,
                volume_sma=volume_sma,
                volume_ratio=volume_ratio
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error performing technical analysis for {ticker}: {e}")
            return None
    
    def _generate_swing_signals(self, analysis: TechnicalAnalysis) -> List[Dict]:
        """Generate swing trading signals based on technical analysis"""
        signals=[]
        
        # Breakout signals
        if analysis.bb_position > 0.8 and analysis.volume_ratio > 1.5:
            signals.append({
                "signal":SwingSignal.BUY,
                "strategy":SwingStrategy.BREAKOUT,
                "confidence":0.8
            })
        
        # Pullback signals
        if (analysis.current_price > analysis.sma_20 > analysis.sma_50 and 
            analysis.rsi < 40 and analysis.bb_position < 0.3):
            signals.append({
                "signal":SwingSignal.BUY,
                "strategy":SwingStrategy.PULLBACK,
                "confidence":0.7
            })
        
        # Mean reversion signals
        if (analysis.rsi < 30 and analysis.bb_position < 0.2 and 
            analysis.stochastic_k < 20):
            signals.append({
                "signal":SwingSignal.BUY,
                "strategy":SwingStrategy.MEAN_REVERSION,
                "confidence":0.6
            })
        
        # Trend following signals
        if (analysis.current_price > analysis.sma_20 > analysis.sma_50 > analysis.sma_200 and
            analysis.macd > analysis.macd_signal and analysis.adx > 25):
            signals.append({
                "signal":SwingSignal.BUY,
                "strategy":SwingStrategy.TREND_FOLLOWING,
                "confidence":0.9
            })
        
        # Momentum signals
        if (analysis.macd_histogram > 0 and analysis.rsi > 50 and 
            analysis.volume_ratio > 1.2):
            signals.append({
                "signal":SwingSignal.BUY,
                "strategy":SwingStrategy.MOMENTUM,
                "confidence":0.75
            })
        
        return signals
    
    async def _create_swing_candidate(self, ticker: str, analysis: TechnicalAnalysis, 
                                    signal_data: Dict) -> Optional[SwingCandidate]:
        """Create swing trading candidate"""
        try:
            signal=signal_data["signal"]
            strategy = signal_data["strategy"]
            confidence = signal_data["confidence"]
            
            # Calculate entry price
            if signal == SwingSignal.BUY:
                entry_price = analysis.current_price
            else:
                entry_price = analysis.current_price
            
            # Calculate stop loss and take profit
            if signal == SwingSignal.BUY:
                stop_loss = entry_price * (1 - self.stop_loss_pct)
                take_profit=entry_price * (1 + self.take_profit_pct)
            else:
                stop_loss=entry_price * (1 + self.stop_loss_pct)
                take_profit=entry_price * (1 - self.take_profit_pct)
            
            # Calculate risk/reward ratio
            risk=abs(entry_price - stop_loss)
            reward=abs(take_profit - entry_price)
            risk_reward_ratio=reward / risk if risk > 0 else 0
            
            # Calculate position size
            position_size = self._calculate_position_size(entry_price, stop_loss)
            
            # Calculate technical score
            technical_score=self._calculate_technical_score(analysis, signal_data)
            
            # Calculate risk score
            risk_score=self._calculate_risk_score(analysis, signal_data)
            
            candidate=SwingCandidate(
                ticker=ticker,
                current_price=analysis.current_price,
                signal=signal,
                strategy=strategy,
                technical_score=technical_score,
                risk_score=risk_score,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward_ratio,
                position_size=position_size,
                confidence=confidence
            )
            
            return candidate
            
        except Exception as e:
            self.logger.error(f"Error creating swing candidate for {ticker}: {e}")
            return None
    
    def _calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        """Calculate position size based on risk"""
        # Simplified position sizing - in production, use proper risk management
        risk_per_share=abs(entry_price - stop_loss)
        max_risk_amount=1000.0  # $1000 max risk per position
        position_size = int(max_risk_amount / risk_per_share) if risk_per_share > 0 else 100
        return min(position_size, 1000)  # Cap at 1000 shares
    
    def _calculate_technical_score(self, analysis: TechnicalAnalysis, signal_data: Dict) -> float:
        """Calculate technical analysis score"""
        score=0.0
        
        # RSI component
        if signal_data["signal"] == SwingSignal.BUY:
            if analysis.rsi < 40:
                score += 0.2
            elif analysis.rsi < 50:
                score += 0.1
        else:
            if analysis.rsi > 60:
                score += 0.2
            elif analysis.rsi > 50:
                score += 0.1
        
        # MACD component
        if analysis.macd > analysis.macd_signal:
            score += 0.15
        if analysis.macd_histogram > 0:
            score += 0.1
        
        # Bollinger Bands component
        if analysis.bb_position < 0.3:  # Near lower band
            score += 0.15
        elif analysis.bb_position > 0.7:  # Near upper band
            score += 0.1
        
        # Moving averages component
        if analysis.current_price > analysis.sma_20 > analysis.sma_50:
            score += 0.2
        
        # Volume component
        if analysis.volume_ratio > 1.5:
            score += 0.1
        elif analysis.volume_ratio > 1.2:
            score += 0.05
        
        # ADX component (trend strength)
        if analysis.adx > 25:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_risk_score(self, analysis: TechnicalAnalysis, signal_data: Dict) -> float:
        """Calculate risk score (lower is better)"""
        risk=0.0
        
        # Volatility risk
        if analysis.bb_width > 0.1:  # High volatility
            risk += 0.3
        
        # Volume risk
        if analysis.volume_ratio < 0.8:  # Low volume
            risk += 0.2
        
        # Trend risk
        if analysis.adx < 20:  # Weak trend
            risk += 0.2
        
        # Overbought/oversold risk
        if analysis.rsi > 80 or analysis.rsi < 20:
            risk += 0.3
        
        return max(0.0, min(1.0, risk))
    
    async def _update_position_data(self, position: SwingPosition):
        """Update position data with current market information"""
        try:
            # Get current market data
            market_data=await self.data.get_market_data(position.ticker)
            if market_data:
                position.current_price=market_data.price
                position.last_update = datetime.now()
                
                # Update days held
                position.days_held=(datetime.now() - position.entry_date).days
                
                # Recalculate P&L
                position.unrealized_pnl=self._calculate_position_pnl(position)
                
                # Update max favorable/adverse moves
                if position.position_type== "long":position.max_favorable_move = max(position.max_favorable_move, 
                                                   position.current_price - position.entry_price)
                    position.max_adverse_move=min(position.max_adverse_move, 
                                                 position.current_price - position.entry_price)
                else:
                    position.max_favorable_move=max(position.max_favorable_move, 
                                                     position.entry_price - position.current_price)
                    position.max_adverse_move=min(position.max_adverse_move, 
                                                   position.entry_price - position.current_price)
            
        except Exception as e:
            self.logger.error(f"Error updating position data for {position.ticker}: {e}")
    
    def _calculate_position_pnl(self, position: SwingPosition) -> float:
        """Calculate position P&L"""
        if position.position_type== "long":return (position.current_price - position.entry_price) * position.quantity
        else:
            return (position.entry_price - position.current_price) * position.quantity
    
    def _check_exit_conditions(self, position: SwingPosition) -> Optional[SwingSignal]:
        """Check for exit conditions"""
        # Check stop loss
        if position.position_type== "long":if position.current_price <= position.stop_loss:
                return SwingSignal.EXIT_LONG
        else:
            if position.current_price >= position.stop_loss:
                return SwingSignal.EXIT_SHORT
        
        # Check take profit
        if position.position_type == "long":if position.current_price >= position.take_profit:
                return SwingSignal.EXIT_LONG
        else:
            if position.current_price <= position.take_profit:
                return SwingSignal.EXIT_SHORT
        
        # Check max hold days
        if position.days_held >= self.max_hold_days:
            return SwingSignal.EXIT_LONG if position.position_type == "long" else SwingSignal.EXIT_SHORT
        
        return None
    
    def _update_trailing_stop(self, position: SwingPosition):
        """Update trailing stop"""
        if position.position_type== "long":if position.current_price > position.trailing_stop:
                new_trailing_stop = position.current_price * (1 - self.stop_loss_pct)
                position.trailing_stop=max(position.trailing_stop, new_trailing_stop)
        else:
            if position.current_price < position.trailing_stop:
                new_trailing_stop=position.current_price * (1 + self.stop_loss_pct)
                position.trailing_stop=min(position.trailing_stop, new_trailing_stop)
    
    def _check_position_risks(self, position: SwingPosition) -> List[str]:
        """Check for position risk alerts"""
        alerts=[]
        
        # Check for large unrealized losses
        if position.unrealized_pnl < -position.quantity * position.entry_price * 0.1:
            alerts.append(f"Large unrealized loss for {position.ticker}: ${position.unrealized_pnl:.2f}")
        
        # Check for approaching max hold days
        if position.days_held >= self.max_hold_days - 3:
            alerts.append(f"Approaching max hold days for {position.ticker}: {position.days_held} days")
        
        # Check for trailing stop breach
        if position.position_type== "long" and position.current_price <= position.trailing_stop:
            alerts.append(f"Trailing stop breached for {position.ticker}")
        elif position.position_type== "short" and position.current_price >= position.trailing_stop:
            alerts.append(f"Trailing stop breached for {position.ticker}")
        
        return alerts
    
    async def _close_position(self, ticker: str, exit_signal: SwingSignal):
        """Close swing position"""
        try:
            if ticker in self.active_positions:
                position=self.active_positions.pop(ticker)
                self.logger.info(f"Closed swing position for {ticker}: P&L ${position.unrealized_pnl:.2f}, Signal: {exit_signal.value}")
            
        except Exception as e:
            self.logger.error(f"Error closing position for {ticker}: {e}")
    
    async def get_strategy_status(self) -> Dict[str, Any]:
        """Get current strategy status"""
        try:
            total_pnl=sum(pos.unrealized_pnl for pos in self.active_positions.values())
            total_exposure=sum(pos.quantity * pos.current_price for pos in self.active_positions.values())
            
            return {
                "active_positions":len(self.active_positions),
                "total_pnl":total_pnl,
                "total_exposure":total_exposure,
                "max_positions":self.max_positions,
                "positions":[
                    {
                        "ticker":pos.ticker,
                        "strategy":pos.strategy.value,
                        "signal":pos.signal.value,
                        "position_type":pos.position_type,
                        "days_held":pos.days_held,
                        "unrealized_pnl":pos.unrealized_pnl,
                        "technical_score":pos.technical_score
                    }
                    for pos in self.active_positions.values()
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting strategy status: {e}")
            return {"error":str(e)}
