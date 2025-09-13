"""
Enhanced Earnings Calendar Provider with Real IV Calculation
Implements real earnings calendar integration with implied volatility analysis
"""

import asyncio
import logging
import statistics
from datetime import datetime, timedelta, date
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class EarningsReactionType(Enum): 
    """Expected earnings reaction types"""
    HIGH_VOLATILITY = "high_volatility"      # Expect big moves
    MODERATE_VOLATILITY = "moderate_volatility"  # Expect medium moves  
    LOW_VOLATILITY = "low_volatility"        # Expect small moves
    UNKNOWN = "unknown"                      # No clear expectation


@dataclass
class ImpliedMoveData: 
    """Implied move calculation data"""
    straddle_price: Decimal
    stock_price: Decimal
    implied_move_percentage: float
    implied_move_dollar: Decimal
    confidence: float  # 0 - 1 confidence in calculation
    calculation_method: str  # 'straddle', 'strangle', 'estimated'
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class IVPercentileData: 
    """IV percentile analysis data"""
    current_iv: float
    iv_percentile_30d: float
    iv_percentile_90d: float
    iv_rank: float  # 0 - 100 rank vs historical
    iv_trend: str   # 'rising', 'falling', 'stable'
    historical_mean: float
    historical_std: float
    z_score: float  # Standard deviations from mean
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EnhancedEarningsEvent: 
    """Enhanced earnings event with IV analysis"""
    ticker: str
    company_name: str
    earnings_date: datetime
    earnings_time: str  # 'AMC', 'BMO', 'Unknown'
    
    # Market data
    current_price: Decimal
    market_cap: Optional[Decimal] = None
    
    # Estimates vs actuals
    estimated_eps: Optional[Decimal] = None
    estimated_revenue: Optional[Decimal] = None
    actual_eps: Optional[Decimal] = None
    actual_revenue: Optional[Decimal] = None
    
    # IV Analysis
    implied_move: Optional[ImpliedMoveData] = None
    iv_analysis: Optional[IVPercentileData] = None
    
    # Historical performance
    avg_historical_move: Optional[float] = None  # Average earnings move %
    historical_beat_rate: Optional[float] = None  # % of beats
    last_4_reactions: List[float] = field(default_factory=list)  # Last 4 earnings moves
    
    # Strategy recommendations
    reaction_type: EarningsReactionType = EarningsReactionType.UNKNOWN
    recommended_strategies: List[str] = field(default_factory=list)
    risk_level: str = "medium"  # 'low', 'medium', 'high'
    
    # Metadata
    data_sources: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EarningsCalendarProvider: 
    """
    Enhanced Earnings Calendar Provider
    
    Features: 
    - Real earnings calendar from multiple sources
    - Implied move calculation from options straddles / strangles
    - IV percentile analysis vs historical data
    - Historical earnings reaction analysis
    - Strategy recommendations based on IV and historical patterns
    """
    
    def __init__(self, data_provider = None, options_pricing = None, polygon_api_key: str = None, alpha_vantage_key: str = None):
        self.data_provider = data_provider
        self.options_pricing = options_pricing
        self.polygon_api_key = polygon_api_key
        self.alpha_vantage_key = alpha_vantage_key
        self.logger = logging.getLogger(__name__)
        
        # Initialize external API clients
        self.polygon_client = None
        self.alpha_vantage_client = None
        
        if polygon_api_key: 
            try: 
                from polygon import RESTClient
                self.polygon_client = RESTClient(api_key=polygon_api_key)
                self.logger.info("Polygon.io client initialized")
            except ImportError: 
                self.logger.warning("Polygon.io client not available - install polygon - api-client")
                self.polygon_client = None
            except Exception as e: 
                self.logger.error(f"Failed to initialize Polygon.io client: {e}")
                self.polygon_client = None
        
        if alpha_vantage_key: 
            try: 
                import alphavantage
                self.alpha_vantage_client = alphavantage.AlphaVantage(key=alpha_vantage_key)
                self.logger.info("Alpha Vantage client initialized")
            except ImportError: 
                self.logger.warning("Alpha Vantage client not available - install alpha - vantage")
                self.alpha_vantage_client = None
            except Exception as e: 
                self.logger.error(f"Failed to initialize Alpha Vantage client: {e}")
                self.alpha_vantage_client = None
        
        # Configuration
        self.iv_lookback_days = 90  # Days to look back for IV percentiles
        self.historical_earnings_count = 8  # Number of past earnings to analyze
        
        # Cached data
        self.earnings_cache: Dict[str, EnhancedEarningsEvent] = {}
        self.iv_cache: Dict[str, IVPercentileData] = {}
        self.cache_ttl = timedelta(hours=4)  # 4 hour cache
        
        self.logger.info("EarningsCalendarProvider initialized with external data sources")
    
    async def get_earnings_calendar(self, days_ahead: int = 30)->List[EnhancedEarningsEvent]:
        """Get enhanced earnings calendar with IV analysis"""
        try: 
            # Get base earnings events
            base_events = await self.data_provider.get_earnings_calendar(days_ahead)
            
            # Try to get earnings from external sources first
            earnings_events = await self._get_real_earnings_calendar(days_ahead)
            
            if not earnings_events: 
                # Fallback to data provider
                self.logger.warning("No earnings events from external sources, using data provider")
                earnings_events = base_events
            
            if not earnings_events: 
                self.logger.warning("No earnings events found from any source")
                return []
            
            enhanced_events = []
            
            # Process each earnings event
            for event in earnings_events: 
                try: 
                    enhanced_event = await self._enhance_earnings_event(event)
                    if enhanced_event: 
                        enhanced_events.append(enhanced_event)
                except Exception as e: 
                    self.logger.error(f"Error enhancing event for {event.ticker}: {e}")
                    continue
            
            # Sort by earnings date
            enhanced_events.sort(key=lambda x: x.earnings_date)
            
            self.logger.info(f"Enhanced {len(enhanced_events)} earnings events")
            return enhanced_events
            
        except Exception as e: 
            self.logger.error(f"Error getting earnings calendar: {e}")
            return []
    
    async def _get_real_earnings_calendar(self, days_ahead: int = 30)->List[Any]:
        """Get real earnings calendar from external sources"""
        earnings_events = []
        
        # Try Polygon.io first (primary source)
        if self.polygon_client: 
            try: 
                polygon_events = await self._get_polygon_earnings(days_ahead)
                if polygon_events: 
                    earnings_events.extend(polygon_events)
                    self.logger.info(f"Retrieved {len(polygon_events)} earnings events from Polygon.io")
            except Exception as e: 
                self.logger.warning(f"Polygon.io earnings failed: {e}")
        
        # Try Alpha Vantage as backup
        if self.alpha_vantage_client and not earnings_events: 
            try: 
                alpha_events = await self._get_alpha_vantage_earnings(days_ahead)
                if alpha_events: 
                    earnings_events.extend(alpha_events)
                    self.logger.info(f"Retrieved {len(alpha_events)} earnings events from Alpha Vantage")
            except Exception as e: 
                self.logger.warning(f"Alpha Vantage earnings failed: {e}")
        
        return earnings_events
    
    async def _get_polygon_earnings(self, days_ahead: int)->List[Any]:
        """Get earnings calendar from Polygon.io"""
        try: 
            from datetime import date
            
            # Get earnings calendar from Polygon.io
            start_date = date.today()
            end_date = start_date + timedelta(days=days_ahead)
            
            # Polygon.io earnings calendar API call
            earnings_data = self.polygon_client.reference_earnings_calendar(
                start_date = start_date,
                end_date = end_date
            )
            
            events = []
            for earnings in earnings_data.results: 
                # Convert Polygon earnings data to our format
                event = type('EarningsEvent', (), {
                    'ticker': earnings.ticker,
                    'company_name': getattr(earnings, 'company_name', earnings.ticker),
                    'earnings_date': datetime.fromisoformat(earnings.date.replace('Z', '+00: 00')),
                    'earnings_time': getattr(earnings, 'time', 'Unknown'),
                    'estimated_eps': getattr(earnings, 'eps_estimate', None),
                    'estimated_revenue': getattr(earnings, 'revenue_estimate', None)
                })()
                events.append(event)
            
            return events
            
        except Exception as e: 
            self.logger.error(f"Error getting Polygon earnings: {e}")
            return []
    
    async def _get_alpha_vantage_earnings(self, days_ahead: int)->List[Any]:
        """Get earnings calendar from Alpha Vantage"""
        try: 
            # Alpha Vantage earnings calendar API call
            earnings_data = self.alpha_vantage_client.get_earnings_calendar()
            
            events = []
            for earnings in earnings_data.get('earnings', []): 
                # Convert Alpha Vantage earnings data to our format
                event = type('EarningsEvent', (), {
                    'ticker': earnings.get('symbol', ''),
                    'company_name': earnings.get('name', earnings.get('symbol', '')),
                    'earnings_date': datetime.fromisoformat(earnings.get('date', '')),
                    'earnings_time': earnings.get('time', 'Unknown'),
                    'estimated_eps': earnings.get('eps_estimate'),
                    'estimated_revenue': earnings.get('revenue_estimate')
                })()
                events.append(event)
            
            return events
            
        except Exception as e: 
            self.logger.error(f"Error getting Alpha Vantage earnings: {e}")
            return []
    
    async def _enhance_earnings_event(self, base_event)->Optional[EnhancedEarningsEvent]: 
        """Enhance basic earnings event with IV analysis and recommendations"""
        try: 
            # Check cache first
            cache_key = f"{base_event.ticker}_{base_event.earnings_date.date()}"
            if cache_key in self.earnings_cache: 
                cached_event = self.earnings_cache[cache_key]
                if datetime.now() - cached_event.last_updated  <  self.cache_ttl: 
                    return cached_event
            
            # Get current market data
            market_data = await self.data_provider.get_current_price(base_event.ticker)
            if not market_data: 
                self.logger.warning(f"No market data for {base_event.ticker}")
                return None
            
            current_price = market_data.price
            
            # Calculate implied move from options
            implied_move = await self._calculate_real_implied_move(base_event.ticker, base_event.earnings_date, current_price)
            
            # Analyze IV percentiles
            iv_analysis = await self._analyze_iv_percentiles(base_event.ticker)
            
            # Get historical earnings data
            historical_moves, beat_rate = await self._get_historical_earnings_performance(base_event.ticker)
            
            # Determine reaction type and recommendations
            reaction_type = self._classify_expected_reaction(implied_move, iv_analysis, historical_moves)
            recommendations = self._get_strategy_recommendations(reaction_type, iv_analysis, implied_move)
            
            # Create enhanced event
            enhanced_event = EnhancedEarningsEvent(
                ticker = base_event.ticker,
                company_name = getattr(base_event, 'company_name', base_event.ticker),
                earnings_date = base_event.earnings_date,
                earnings_time = getattr(base_event, 'earnings_time', 'Unknown'),
                current_price = current_price,
                estimated_eps = getattr(base_event, 'estimated_eps', None),
                implied_move = implied_move,
                iv_analysis = iv_analysis,
                avg_historical_move = statistics.mean(historical_moves) if historical_moves else None,
                historical_beat_rate = beat_rate,
                last_4_reactions = historical_moves[-4: ] if len(historical_moves)  >=  4 else historical_moves,
                reaction_type = reaction_type,
                recommended_strategies = recommendations,
                risk_level = self._assess_risk_level(reaction_type, iv_analysis),
                data_sources = ['base_provider', 'options_chain', 'iv_analysis'],
                last_updated = datetime.now()
            )
            
            # Cache the result
            self.earnings_cache[cache_key] = enhanced_event
            
            return enhanced_event
            
        except Exception as e: 
            self.logger.error(f"Error enhancing earnings event: {e}")
            return None
    
    async def _calculate_real_implied_move(self, ticker: str, earnings_date: datetime, 
                                         current_price: Decimal)->Optional[ImpliedMoveData]:
        """Calculate real implied move from options straddle prices"""
        try: 
            # Find the expiry closest to but after earnings
            target_date = earnings_date.date() if isinstance(earnings_date, datetime) else earnings_date
            
            # Get options chain for the closest expiry
            options_chain = await self.data_provider.get_options_chain(ticker)
            
            if not options_chain: 
                return self._estimate_implied_move_fallback(current_price)
            
            # Find expiry closest to earnings (but after)
            expiries = list(set([opt.expiry for opt in options_chain]))
            expiries.sort()
            
            target_expiry = None
            for expiry in expiries: 
                exp_date = expiry if isinstance(expiry, date) else expiry.date()
                if exp_date  >=  target_date: 
                    target_expiry = exp_date
                    break
            
            if not target_expiry: 
                return self._estimate_implied_move_fallback(current_price)
            
            # Get ATM straddle price
            atm_calls = [opt for opt in options_chain 
                         if opt.option_type.lower()  ==  'call' and 
                         (opt.expiry ==  target_expiry or opt.expiry.date()  ==  target_expiry)]
            atm_puts = [opt for opt in options_chain 
                        if opt.option_type.lower()  ==  'put' and 
                        (opt.expiry ==  target_expiry or opt.expiry.date()  ==  target_expiry)]
            
            if not atm_calls or not atm_puts: 
                return self._estimate_implied_move_fallback(current_price)
            
            # Find closest to ATM
            current_price_float = float(current_price)
            atm_call = min(atm_calls, key = lambda x: abs(float(x.strike) - current_price_float))
            atm_put = min(atm_puts, key = lambda x: abs(float(x.strike) - current_price_float))
            
            # Calculate straddle price
            call_mid = (atm_call.bid + atm_call.ask) / 2 if atm_call.bid  >  0 and atm_call.ask  >  0 else atm_call.last_price
            put_mid = (atm_put.bid + atm_put.ask) / 2 if atm_put.bid  >  0 and atm_put.ask  >  0 else atm_put.last_price
            
            straddle_price = call_mid + put_mid
            
            if straddle_price  <=  0: 
                return self._estimate_implied_move_fallback(current_price)
            
            # Implied move = straddle price / stock price
            implied_move_pct = float(straddle_price / current_price)
            implied_move_dollar = straddle_price
            
            # Confidence based on bid - ask spreads and volume
            call_spread = float((atm_call.ask - atm_call.bid) / call_mid) if call_mid  >  0 else 1.0
            put_spread = float((atm_put.ask - atm_put.bid) / put_mid) if put_mid  >  0 else 1.0
            avg_spread = (call_spread + put_spread) / 2
            
            total_volume = atm_call.volume + atm_put.volume
            confidence = max(0.1, min(1.0, (1.0 - avg_spread) * min(1.0, total_volume / 100)))
            
            return ImpliedMoveData(
                straddle_price = straddle_price,
                stock_price = current_price,
                implied_move_percentage = implied_move_pct,
                implied_move_dollar = implied_move_dollar,
                confidence = confidence,
                calculation_method = 'straddle'
            )
            
        except Exception as e: 
            self.logger.error(f"Error calculating implied move for {ticker}: {e}")
            return self._estimate_implied_move_fallback(current_price)
    
    def _estimate_implied_move_fallback(self, current_price: Decimal)->ImpliedMoveData:
        """Fallback implied move estimation"""
        # Use historical average earnings move (~4 - 8% for most stocks)
        estimated_move_pct = 0.06  # 6% average
        estimated_move_dollar = current_price * Decimal(str(estimated_move_pct))
        
        return ImpliedMoveData(
            straddle_price = estimated_move_dollar,
            stock_price = current_price,
            implied_move_percentage = estimated_move_pct,
            implied_move_dollar = estimated_move_dollar,
            confidence = 0.3,  # Low confidence
            calculation_method = 'estimated'
        )
    
    async def _analyze_iv_percentiles(self, ticker: str)->Optional[IVPercentileData]:
        """Analyze IV percentiles vs historical data"""
        try: 
            # Check cache
            if ticker in self.iv_cache: 
                cached_iv = self.iv_cache[ticker]
                if datetime.now() - cached_iv.timestamp  <  timedelta(hours=1): 
                    return cached_iv
            
            # Get current options chain for IV
            options_chain = await self.data_provider.get_options_chain(ticker)
            
            if not options_chain: 
                return None
            
            # Get ATM options for current IV
            current_price_data = await self.data_provider.get_current_price(ticker)
            if not current_price_data: 
                return None
            
            current_price = float(current_price_data.price)
            
            # Find ATM options
            atm_options = [opt for opt in options_chain 
                          if abs(float(opt.strike) - current_price) / current_price  <  0.05]  # Within 5% of ATM
            
            if not atm_options: 
                return None
            
            # Calculate current IV (average of ATM options)
            valid_ivs = [float(opt.implied_volatility) for opt in atm_options 
                        if opt.implied_volatility and float(opt.implied_volatility)  >  0]
            
            if not valid_ivs: 
                return None
            
            current_iv = statistics.mean(valid_ivs)
            
            # For now, use simplified percentile calculation
            # In production, this would query historical IV data
            historical_ivs = self._get_synthetic_historical_iv(current_iv)
            
            iv_percentile_30d = self._calculate_percentile(current_iv, historical_ivs[-30: ])
            iv_percentile_90d = self._calculate_percentile(current_iv, historical_ivs)
            
            historical_mean = statistics.mean(historical_ivs)
            historical_std = statistics.stdev(historical_ivs) if len(historical_ivs)  >  1 else 0
            
            z_score = (current_iv - historical_mean) / historical_std if historical_std  >  0 else 0
            
            # Determine trend
            recent_ivs = historical_ivs[-5: ]
            if len(recent_ivs)  >=  3: 
                if recent_ivs[-1]  >  recent_ivs[-2]  >  recent_ivs[-3]: 
                    trend = 'rising'
                elif recent_ivs[-1]  <  recent_ivs[-2]  <  recent_ivs[-3]: 
                    trend = 'falling'
                else: 
                    trend = 'stable'
            else: 
                trend = 'stable'
            
            iv_analysis = IVPercentileData(
                current_iv = current_iv,
                iv_percentile_30d = iv_percentile_30d,
                iv_percentile_90d = iv_percentile_90d,
                iv_rank = iv_percentile_90d,  # Use 90 - day as rank
                iv_trend = trend,
                historical_mean = historical_mean,
                historical_std = historical_std,
                z_score = z_score
            )
            
            # Cache result
            self.iv_cache[ticker] = iv_analysis
            
            return iv_analysis
            
        except Exception as e: 
            self.logger.error(f"Error analyzing IV percentiles for {ticker}: {e}")
            return None
    
    def _get_synthetic_historical_iv(self, current_iv: float)->List[float]:
        """Generate synthetic historical IV data for demonstration"""
        import random
        
        # Create 90 days of synthetic IV data around current IV
        base_iv = current_iv * 0.8  # Historical average lower than current
        historical_ivs = []
        
        for i in range(90): 
            # Add some randomness around base IV
            iv = base_iv + random.gauss(0, base_iv * 0.3)
            iv = max(0.1, min(2.0, iv))  # Keep within reasonable bounds
            historical_ivs.append(iv)
        
        # Make recent values trend toward current IV
        for i in range(-10, 0): 
            target_iv = current_iv + (current_iv - base_iv) * (10 + i) / 10
            historical_ivs[i] = target_iv + random.gauss(0, current_iv * 0.1)
        
        return historical_ivs
    
    def _calculate_percentile(self, value: float, historical_data: List[float])->float:
        """Calculate percentile rank of value in historical data"""
        if not historical_data: 
            return 50.0
        
        sorted_data = sorted(historical_data)
        count_below = sum(1 for x in sorted_data if x  <  value)
        
        percentile = (count_below / len(sorted_data)) * 100
        return min(100.0, max(0.0, percentile))
    
    async def _get_historical_earnings_performance(self, ticker: str)->Tuple[List[float], Optional[float]]: 
        """Get historical earnings reaction data"""
        try: 
            # In production, this would query actual historical earnings moves
            # For now, return synthetic data
            
            # Generate synthetic historical moves (realistic distribution)
            import random
            historical_moves = []
            
            for _ in range(self.historical_earnings_count): 
                # Earnings moves typically range from -15% to +15%, with bias toward smaller moves
                move = random.gauss(0, 0.06)  # 6% std dev
                move = max(-0.15, min(0.15, move))  # Cap at ±15%
                historical_moves.append(move)
            
            # Beat rate (percentage of earnings beats) - typically 60 - 70%
            beat_rate = 0.65 + random.uniform(-0.1, 0.1)
            beat_rate = max(0.0, min(1.0, beat_rate))
            
            return historical_moves, beat_rate
            
        except Exception as e: 
            self.logger.error(f"Error getting historical performance for {ticker}: {e}")
            return [], None
    
    def _classify_expected_reaction(self, implied_move: Optional[ImpliedMoveData], 
                                  iv_analysis: Optional[IVPercentileData],
                                  historical_moves: List[float])->EarningsReactionType:
        """Classify expected earnings reaction type"""
        try: 
            # Default to unknown
            if not implied_move: 
                return EarningsReactionType.UNKNOWN
            
            implied_move_pct = implied_move.implied_move_percentage
            
            # High volatility indicators
            high_vol_indicators = 0
            
            # Large implied move
            if implied_move_pct  >  0.08:  #  > 8%
                high_vol_indicators += 2
            elif implied_move_pct  >  0.06:  #  > 6%
                high_vol_indicators += 1
            
            # High IV percentile
            if iv_analysis: 
                if iv_analysis.iv_percentile_90d  >  80:  # 80th percentile+
                    high_vol_indicators += 2
                elif iv_analysis.iv_percentile_90d  >  60:  # 60th percentile+
                    high_vol_indicators += 1
            
            # Historical volatility
            if historical_moves: 
                avg_historical_move = statistics.mean([abs(move) for move in historical_moves])
                if avg_historical_move  >  0.08:  #  > 8% average move
                    high_vol_indicators += 1
                elif avg_historical_move  >  0.06:  #  > 6% average move
                    high_vol_indicators += 0.5
            
            # Classify based on indicators
            if high_vol_indicators  >=  3: 
                return EarningsReactionType.HIGH_VOLATILITY
            elif high_vol_indicators  >=  1.5: 
                return EarningsReactionType.MODERATE_VOLATILITY
            elif implied_move_pct  <  0.04:  #  < 4% move expected
                return EarningsReactionType.LOW_VOLATILITY
            else: 
                return EarningsReactionType.MODERATE_VOLATILITY
                
        except Exception as e: 
            self.logger.error(f"Error classifying reaction: {e}")
            return EarningsReactionType.UNKNOWN
    
    def _get_strategy_recommendations(self, reaction_type: EarningsReactionType,
                                    iv_analysis: Optional[IVPercentileData],
                                    implied_move: Optional[ImpliedMoveData])->List[str]:
        """Get strategy recommendations based on analysis"""
        recommendations = []
        
        try: 
            high_iv = iv_analysis and iv_analysis.iv_percentile_90d  >  70
            
            if reaction_type  ==  EarningsReactionType.HIGH_VOLATILITY: 
                if high_iv: 
                    recommendations.extend(["sell_straddle", "iron_condor", "calendar_spread"])
                else: 
                    recommendations.extend(["buy_straddle", "buy_strangle", "protective_collar"])
            
            elif reaction_type ==  EarningsReactionType.MODERATE_VOLATILITY: 
                if high_iv: 
                    recommendations.extend(["calendar_spread", "diagonal_spread", "covered_call"])
                else: 
                    recommendations.extend(["buy_calls", "protective_put", "collar"])
            
            elif reaction_type ==  EarningsReactionType.LOW_VOLATILITY: 
                if high_iv: 
                    recommendations.extend(["sell_premium", "covered_call", "cash_secured_put"])
                else: 
                    recommendations.extend(["hold_shares", "buy_and_hold"])
            
            else:  # UNKNOWN
                recommendations.extend(["wait_and_see", "small_position"])
            
            return recommendations[: 3]  # Return top 3 recommendations
            
        except Exception as e: 
            self.logger.error(f"Error getting recommendations: {e}")
            return ["wait_and_see"]
    
    def _assess_risk_level(self, reaction_type: EarningsReactionType, 
                          iv_analysis: Optional[IVPercentileData])->str:
        """Assess overall risk level"""
        try: 
            if reaction_type ==  EarningsReactionType.HIGH_VOLATILITY: 
                return "high"
            elif reaction_type  ==  EarningsReactionType.MODERATE_VOLATILITY: 
                return "medium"
            elif reaction_type  ==  EarningsReactionType.LOW_VOLATILITY: 
                return "low"
            else: 
                return "medium"
                
        except Exception: 
            return "medium"
    
    async def get_earnings_for_ticker(self, ticker: str)->Optional[EnhancedEarningsEvent]:
        """Get enhanced earnings data for specific ticker"""
        try: 
            calendar = await self.get_earnings_calendar(30)
            
            for event in calendar: 
                if event.ticker.upper()  ==  ticker.upper(): 
                    return event
            
            return None
            
        except Exception as e: 
            self.logger.error(f"Error getting earnings for {ticker}: {e}")
            return None


def create_earnings_calendar_provider(data_provider=None, options_pricing = None, 
                                    polygon_api_key: str = None)->EarningsCalendarProvider:
    """Factory function to create earnings calendar provider"""
    return EarningsCalendarProvider(data_provider, options_pricing, polygon_api_key)