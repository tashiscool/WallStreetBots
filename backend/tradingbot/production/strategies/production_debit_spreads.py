#!/usr / bin/env python3
"""
Production Debit Spreads Strategy
More repeatable than naked calls with reduced theta / IV risk
"""

import asyncio
import logging
import math
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from decimal import Decimal

from ..data.production_data_integration import ReliableDataProvider
from ..core.production_integration import ProductionIntegrationManager, ProductionTradeSignal
from ...models import Stock, Order, Portfolio
from ...risk.real_time_risk_manager import RealTimeRiskManager
from ...options.smart_selection import SmartOptionsSelector
from ...options.pricing_engine import BlackScholesEngine


@dataclass
class SpreadOpportunity: 
    """Production spread opportunity with enhanced metadata"""
    ticker: str
    scan_date: date
    spot_price: float
    trend_strength: float  # 0 - 1 score
    expiry_date: str
    days_to_expiry: int
    long_strike: float
    short_strike: float
    spread_width: float
    long_premium: float
    short_premium: float
    net_debit: float
    max_profit: float
    max_profit_pct: float
    breakeven: float
    prob_profit: float
    risk_reward: float
    iv_rank: float
    volume_score: float
    confidence: float


class ProductionDebitSpreads: 
    """
    Production Debit Spreads Strategy
    
    Strategy Logic: 
    1. Scans for stocks with bullish trend signals
    2. Finds optimal call spread combinations (long + short call)
    3. Targets 20 - 60 DTE with strong risk / reward ratios ( > 1.5)
    4. Manages positions with profit targets and stop losses
    5. Focuses on liquid strikes with tight spreads
    
    Risk Management: 
    - Maximum 10% account risk per spread
    - Maximum 8 concurrent spread positions
    - Stop loss at 50% of premium paid
    - Profit target at 25 - 40% of max spread value
    - Time - based exits before expiration week
    """
    
    def __init__(self, integration_manager, data_provider: ReliableDataProvider, config: dict):
        self.strategy_name = "debit_spreads"
        self.integration_manager = integration_manager
        self.data_provider = data_provider
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.options_selector = SmartOptionsSelector(data_provider)
        self.risk_manager = RealTimeRiskManager()
        self.bs_engine = BlackScholesEngine()
        
        # Strategy configuration
        self.watchlist = config.get('watchlist', [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX",
            "CRM", "ADBE", "ORCL", "AMD", "QCOM", "UBER", "SNOW", "COIN",
            "PLTR", "ROKU", "ZM", "SHOP", "SQ", "PYPL", "TWLO"
        ])
        
        # Risk parameters
        self.max_positions = config.get('max_positions', 8)
        self.max_position_size = config.get('max_position_size', 0.10)  # 10% per spread
        self.min_dte = config.get('min_dte', 20)
        self.max_dte = config.get('max_dte', 60)
        
        # Spread targeting
        self.min_risk_reward = config.get('min_risk_reward', 1.5)
        self.min_trend_strength = config.get('min_trend_strength', 0.6)
        self.max_iv_rank = config.get('max_iv_rank', 80)  # Avoid buying high IV
        self.min_volume_score = config.get('min_volume_score', 0.3)
        
        # Exit criteria
        self.profit_target = config.get('profit_target', 0.30)  # 30% of max profit
        self.stop_loss = config.get('stop_loss', 0.50)  # 50% of premium paid
        self.time_exit_dte = config.get('time_exit_dte', 7)  # Exit 7 days before expiry
        
        # Active positions tracking
        self.active_positions: List[Dict[str, Any]] = []
        
        self.logger.info("ProductionDebitSpreads strategy initialized")
    
    def norm_cdf(self, x: float)->float:
        """Standard normal cumulative distribution function"""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))
    
    def black_scholes_call(self, S: float, K: float, T: float, r: float, sigma: float)->Tuple[float, float]: 
        """Black - Scholes call price and delta"""
        if T  <=  0 or sigma  <=  0: 
            return max(S - K, 0), 1.0 if S  >  K else 0.0
            
        d1 = (math.log(S / K) + (r + 0.5 * sigma*sigma)*T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        call_price = S * self.norm_cdf(d1) - K * math.exp(-r * T) * self.norm_cdf(d2)
        delta = self.norm_cdf(d1)
        
        return max(call_price, 0), delta
    
    async def assess_trend_strength(self, ticker: str)->float:
        """Assess bullish trend strength (0 - 1 score)"""
        try: 
            # Get 60 days of price data
            price_data = await self.data_provider.get_historical_data(ticker, "60d")
            if price_data.empty or len(price_data)  <  50: 
                return 0.5
            
            prices = price_data['close'].values
            volumes = price_data['volume'].values
            current = prices[-1]
            
            scores = []
            
            # 1. Price vs moving averages
            sma_20 = prices[-20: ].mean()
            sma_50 = prices[-50: ].mean()
            
            if current  >  sma_20  >  sma_50: 
                scores.append(0.8)
            elif current  >  sma_20: 
                scores.append(0.6)
            else: 
                scores.append(0.2)
            
            # 2. Recent momentum (10 - day)
            momentum = (current / prices[-10] - 1) * 5  # Scale to 0 - 1
            scores.append(max(0, min(1, momentum)))
            
            # 3. Trend consistency
            direction_changes = 0
            for i in range(len(prices)-10, len(prices)-1): 
                if i  <=  0: 
                    continue
                curr_trend = prices[i + 1]  >  prices[i]
                prev_trend = prices[i]  >  prices[i - 1] if i  >  0 else curr_trend
                if curr_trend  !=  prev_trend: 
                    direction_changes += 1
            
            consistency = max(0, 1 - direction_changes / 10)
            scores.append(consistency)
            
            # 4. Volume trend
            recent_vol = volumes[-10: ].mean()
            past_vol = volumes[-30: -10].mean()
            
            if past_vol  >  0: 
                vol_trend = min(1, recent_vol / past_vol)
                scores.append(vol_trend * 0.5)
            else: 
                scores.append(0.5)
            
            return sum(scores) / len(scores)
            
        except Exception as e: 
            self.logger.error(f"Error assessing trend for {ticker}: {e}")
            return 0.5
    
    async def calculate_iv_rank(self, ticker: str, current_iv: float)->float:
        """Calculate IV rank (current IV vs historical range)"""
        try: 
            # Get 1 year of historical data to estimate IV range
            hist_data = await self.data_provider.get_historical_data(ticker, "1y")
            if hist_data.empty: 
                return 50.0
            
            # Calculate historical volatility as proxy for IV range
            returns = hist_data['close'].pct_change().dropna()
            rolling_vol = returns.rolling(20).std() * math.sqrt(252)
            
            if rolling_vol.empty: 
                return 50.0
            
            iv_min = rolling_vol.min()
            iv_max = rolling_vol.max()
            
            if iv_max ==  iv_min: 
                return 50.0
            
            rank = (current_iv - iv_min) / (iv_max - iv_min) * 100
            return max(0, min(100, rank))
            
        except Exception as e: 
            self.logger.error(f"Error calculating IV rank for {ticker}: {e}")
            return 50.0
    
    async def get_options_chain(self, ticker: str, expiry: str)->Optional[Dict[str, Any]]: 
        """Get filtered options chain for expiry"""
        try: 
            # Get options data from data provider
            options_data = await self.data_provider.get_options_chain(ticker, expiry)
            if not options_data or 'calls' not in options_data: 
                return None
            
            calls = options_data['calls']
            
            # Filter for liquid options
            filtered_calls = []
            for call in calls: 
                volume = call.get('volume', 0)
                open_interest = call.get('open_interest', 0)
                bid = call.get('bid', 0)
                ask = call.get('ask', 0)
                
                if (volume  >=  10 or open_interest  >=  50) and bid  >  0.05 and ask  >  0.05: 
                    call['mid'] = (bid + ask) / 2
                    call['spread'] = ask - bid
                    call['spread_pct'] = call['spread'] / call['mid'] if call['mid']  >  0 else 1.0
                    
                    if call['spread_pct']  <  0.3:  # Reasonable bid - ask spread
                        filtered_calls.append(call)
            
            return {'calls': filtered_calls} if filtered_calls else None
            
        except Exception as e: 
            self.logger.error(f"Error getting options chain for {ticker}: {e}")
            return None
    
    async def find_optimal_spreads(self, ticker: str, spot: float, expiry: str, 
                                 options_data: Dict[str, Any])->List[SpreadOpportunity]: 
        """Find optimal spread combinations"""
        opportunities = []
        calls = options_data['calls']
        
        days_to_exp = (datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()).days
        if days_to_exp  <=  0: 
            return opportunities
        
        # Target strikes around current price
        min_long_strike = spot * 0.95  # Max 5% ITM
        max_long_strike = spot * 1.10  # Max 10% OTM
        
        suitable_calls = [
            call for call in calls
            if min_long_strike  <=  call['strike']  <=  max_long_strike
        ]
        
        if len(suitable_calls)  <  2: 
            return opportunities
        
        # Try different spread widths
        for width in [5, 10, 15, 20]: 
            for long_call in suitable_calls: 
                long_strike = long_call['strike']
                short_strike = long_strike + width
                
                # Find matching short call
                short_calls = [
                    call for call in suitable_calls
                    if abs(call['strike'] - short_strike)  <  0.01
                ]
                
                if not short_calls: 
                    continue
                
                short_call = short_calls[0]
                
                # Calculate spread economics
                long_premium = long_call['mid']
                short_premium = short_call['mid']
                net_debit = long_premium - short_premium
                
                if net_debit  <=  0.10:  # Minimum viable premium
                    continue
                
                max_profit = width - net_debit
                max_profit_pct = (max_profit / net_debit) * 100
                breakeven = long_strike + net_debit
                risk_reward = max_profit / net_debit
                
                # Estimate probability of profit
                if breakeven  <=  spot: 
                    prob_profit = 0.7  # ITM breakeven
                else: 
                    distance_ratio = (breakeven - spot) / spot
                    prob_profit = max(0.1, 0.6 - distance_ratio * 3)
                
                # Volume / liquidity score
                long_vol = long_call.get('volume', 0) + long_call.get('open_interest', 0)
                short_vol = short_call.get('volume', 0) + short_call.get('open_interest', 0)
                volume_score = min(long_vol, short_vol) / 1000
                volume_score = max(0, min(1, volume_score))
                
                # Only include spreads with good risk - reward
                if risk_reward  >=  self.min_risk_reward and max_profit_pct  >=  20: 
                    
                    # Get trend strength and IV rank
                    trend_strength = await self.assess_trend_strength(ticker)
                    
                    # Estimate IV from option price
                    try: 
                        estimated_iv = self.estimate_iv_from_price(
                            spot, long_strike, days_to_exp / 365, long_premium
                        )
                    except: 
                        estimated_iv = 0.25
                    
                    iv_rank = await self.calculate_iv_rank(ticker, estimated_iv)
                    
                    # Calculate confidence score
                    confidence = (
                        (risk_reward / 3.0) * 0.3 +  # Risk - reward component
                        trend_strength * 0.3 +        # Trend component
                        prob_profit * 0.2 +           # Probability component
                        volume_score * 0.1 +          # Liquidity component
                        (1 - iv_rank / 100) * 0.1       # IV component (lower is better)
                    )
                    confidence = max(0, min(1, confidence))
                    
                    opportunity = SpreadOpportunity(
                        ticker = ticker,
                        scan_date = date.today(),
                        spot_price = spot,
                        trend_strength = trend_strength,
                        expiry_date = expiry,
                        days_to_expiry = days_to_exp,
                        long_strike = long_strike,
                        short_strike = short_strike,
                        spread_width = width,
                        long_premium = long_premium,
                        short_premium = short_premium,
                        net_debit = net_debit,
                        max_profit = max_profit,
                        max_profit_pct = max_profit_pct,
                        breakeven = breakeven,
                        prob_profit = prob_profit,
                        risk_reward = risk_reward,
                        iv_rank = iv_rank,
                        volume_score = volume_score,
                        confidence = confidence
                    )
                    
                    opportunities.append(opportunity)
        
        # Sort by confidence score
        opportunities.sort(key = lambda x: x.confidence, reverse = True)
        return opportunities[: 3]  # Top 3 per ticker
    
    def estimate_iv_from_price(self, S: float, K: float, T: float, market_price: float)->float:
        """Estimate implied volatility using Newton - Raphson"""
        try: 
            iv = 0.25  # Initial guess
            
            for _ in range(20): 
                price, delta = self.black_scholes_call(S, K, T, 0.04, iv)
                
                # Calculate vega
                d1 = (math.log(S / K) + (0.04 + 0.5 * iv*iv)*T) / (iv * math.sqrt(T))
                vega = S * math.sqrt(T) * self.norm_cdf(d1) * math.exp(-0.04 * T)
                
                if abs(vega)  <  1e-6: 
                    break
                
                diff = price - market_price
                if abs(diff)  <  0.01: 
                    break
                
                iv -= diff / vega
                iv = max(0.01, min(2.0, iv))
            
            return iv
            
        except: 
            return 0.25
    
    async def scan_spread_opportunities(self)->List[SpreadOpportunity]: 
        """Scan for debit spread opportunities"""
        all_opportunities = []
        
        self.logger.info(f"Scanning {len(self.watchlist)} tickers for debit spreads")
        
        for ticker in self.watchlist: 
            try: 
                # Skip if we already have a position in this ticker
                if any(pos['ticker']  ==  ticker for pos in self.active_positions): 
                    continue
                
                # Get current price
                current_price = await self.data_provider.get_current_price(ticker)
                if not current_price: 
                    continue
                
                # Pre - filter by trend strength
                trend_strength = await self.assess_trend_strength(ticker)
                if trend_strength  <  self.min_trend_strength: 
                    continue
                
                # Get available expiries
                expiries = await self.data_provider.get_option_expiries(ticker)
                if not expiries: 
                    continue
                
                # Filter expiries by DTE range
                valid_expiries = []
                today = date.today()
                
                for exp_str in expiries: 
                    try: 
                        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                        days = (exp_date - today).days
                        if self.min_dte  <=  days  <=  self.max_dte: 
                            valid_expiries.append(exp_str)
                    except: 
                        continue
                
                if not valid_expiries: 
                    continue
                
                # Scan expiries (limit to avoid overload)
                for expiry in valid_expiries[: 3]:
                    options_data = await self.get_options_chain(ticker, expiry)
                    if not options_data: 
                        continue
                    
                    spreads = await self.find_optimal_spreads(ticker, current_price, expiry, options_data)
                    
                    # Filter by additional criteria
                    for spread in spreads: 
                        if (spread.iv_rank  <=  self.max_iv_rank and 
                            spread.volume_score  >=  self.min_volume_score): 
                            all_opportunities.append(spread)
                    
                    if spreads: 
                        self.logger.info(f"Found {len(spreads)} spread opportunities for {ticker}")
                
            except Exception as e: 
                self.logger.error(f"Error scanning {ticker}: {e}")
                continue
        
        # Global ranking by confidence
        all_opportunities.sort(key = lambda x: x.confidence, reverse = True)
        return all_opportunities
    
    async def execute_spread_trade(self, opportunity: SpreadOpportunity)->bool:
        """Execute debit spread trade"""
        try: 
            # Check if we can add more positions
            if len(self.active_positions)  >=  self.max_positions: 
                self.logger.info("Max positions reached, skipping trade")
                return False
            
            # Calculate position size
            portfolio_value = await self.integration_manager.get_portfolio_value()
            max_risk = portfolio_value * self.max_position_size
            
            # Size position based on net debit
            contracts = max(1, int(max_risk / (opportunity.net_debit * 100)))
            contracts = min(contracts, 5)  # Max 5 contracts per spread
            
            # Create long leg trade signal
            long_signal = ProductionTradeSignal(
                symbol = opportunity.ticker,
                action = "BUY",
                quantity = contracts,
                option_type = "CALL",
                strike_price = Decimal(str(opportunity.long_strike)),
                expiration_date = datetime.strptime(opportunity.expiry_date, "%Y-%m-%d").date(),
                premium = Decimal(str(opportunity.long_premium)),
                confidence = opportunity.confidence,
                strategy_name = self.strategy_name,
                signal_strength = opportunity.trend_strength,
                metadata = {
                    'spread_type': 'debit_spread',
                    'leg': 'long',
                    'spread_id': f"{opportunity.ticker}_{opportunity.expiry_date}_{opportunity.long_strike}_{opportunity.short_strike}",
                    'short_strike': opportunity.short_strike,
                    'net_debit': opportunity.net_debit,
                    'max_profit': opportunity.max_profit,
                    'breakeven': opportunity.breakeven,
                    'risk_reward': opportunity.risk_reward
                }
            )
            
            # Create short leg trade signal
            short_signal = ProductionTradeSignal(
                symbol = opportunity.ticker,
                action = "SELL",
                quantity = contracts,
                option_type = "CALL",
                strike_price = Decimal(str(opportunity.short_strike)),
                expiration_date = datetime.strptime(opportunity.expiry_date, "%Y-%m-%d").date(),
                premium = Decimal(str(opportunity.short_premium)),
                confidence = opportunity.confidence,
                strategy_name = self.strategy_name,
                signal_strength = opportunity.trend_strength,
                metadata = {
                    'spread_type': 'debit_spread',
                    'leg': 'short',
                    'spread_id': f"{opportunity.ticker}_{opportunity.expiry_date}_{opportunity.long_strike}_{opportunity.short_strike}",
                    'long_strike': opportunity.long_strike,
                    'net_debit': opportunity.net_debit,
                    'max_profit': opportunity.max_profit,
                    'breakeven': opportunity.breakeven,
                    'risk_reward': opportunity.risk_reward
                }
            )
            
            # Execute both legs
            long_success = await self.integration_manager.execute_trade_signal(long_signal)
            short_success = await self.integration_manager.execute_trade_signal(short_signal)
            
            if long_success and short_success: 
                # Track position
                position = {
                    'ticker': opportunity.ticker,
                    'spread_id': f"{opportunity.ticker}_{opportunity.expiry_date}_{opportunity.long_strike}_{opportunity.short_strike}",
                    'long_signal': long_signal,
                    'short_signal': short_signal,
                    'entry_time': datetime.now(),
                    'contracts': contracts,
                    'net_debit': opportunity.net_debit,
                    'max_profit': opportunity.max_profit,
                    'breakeven': opportunity.breakeven,
                    'profit_target': opportunity.max_profit * self.profit_target,
                    'stop_loss': opportunity.net_debit * self.stop_loss,
                    'expiry_date': datetime.strptime(opportunity.expiry_date, "%Y-%m-%d").date(),
                    'confidence': opportunity.confidence
                }
                
                self.active_positions.append(position)
                
                await self.integration_manager.alert_system.send_alert(
                    "SPREAD_ENTRY",
                    "MEDIUM",
                    f"Debit Spread Entry: {opportunity.ticker} "
                    f"${opportunity.long_strike}/{opportunity.short_strike} "
                    f"{contracts} contracts @ ${opportunity.net_debit: .2f} debit"
                )
                
                self.logger.info(f"Debit spread executed: {opportunity.ticker}")
                return True
            
            return False
            
        except Exception as e: 
            self.logger.error(f"Error executing spread trade for {opportunity.ticker}: {e}")
            return False
    
    async def manage_positions(self): 
        """Manage existing spread positions"""
        positions_to_remove = []
        
        for i, position in enumerate(self.active_positions): 
            try: 
                ticker = position['ticker']
                contracts = position['contracts']
                net_debit = position['net_debit']
                
                # Get current spread value
                long_price = await self.data_provider.get_option_price(
                    ticker,
                    position['long_signal'].strike_price,
                    position['long_signal'].expiration_date,
                    'call'
                )
                
                short_price = await self.data_provider.get_option_price(
                    ticker,
                    position['short_signal'].strike_price,
                    position['short_signal'].expiration_date,
                    'call'
                )
                
                if not (long_price and short_price): 
                    continue
                
                current_spread_value = long_price - short_price
                
                # Calculate P & L
                pnl_per_contract = current_spread_value - net_debit
                total_pnl = pnl_per_contract * contracts * 100
                pnl_pct = pnl_per_contract / net_debit if net_debit  >  0 else 0
                
                # Check exit conditions
                should_exit = False
                exit_reason = ""
                
                # Profit target
                if pnl_per_contract  >=  position['profit_target']: 
                    should_exit = True
                    exit_reason = "PROFIT_TARGET"
                
                # Stop loss
                elif pnl_per_contract  <=  -position['stop_loss']: 
                    should_exit = True
                    exit_reason = "STOP_LOSS"
                
                # Time - based exit
                elif (position['expiry_date'] - date.today()).days  <=  self.time_exit_dte: 
                    should_exit = True
                    exit_reason = "TIME_EXIT"
                
                if should_exit: 
                    # Create exit signals for both legs
                    long_exit = ProductionTradeSignal(
                        symbol = ticker,
                        action = "SELL",
                        quantity = contracts,
                        option_type = "CALL",
                        strike_price = position['long_signal'].strike_price,
                        expiration_date = position['long_signal'].expiration_date,
                        premium = Decimal(str(long_price)),
                        strategy_name = self.strategy_name,
                        metadata = {
                            'spread_type': 'debit_spread',
                            'action': 'close_long',
                            'exit_reason': exit_reason,
                            'pnl_per_contract': pnl_per_contract,
                            'total_pnl': total_pnl
                        }
                    )
                    
                    short_exit = ProductionTradeSignal(
                        symbol = ticker,
                        action = "BUY",
                        quantity = contracts,
                        option_type = "CALL",
                        strike_price = position['short_signal'].strike_price,
                        expiration_date = position['short_signal'].expiration_date,
                        premium = Decimal(str(short_price)),
                        strategy_name = self.strategy_name,
                        metadata = {
                            'spread_type': 'debit_spread',
                            'action': 'close_short',
                            'exit_reason': exit_reason,
                            'pnl_per_contract': pnl_per_contract,
                            'total_pnl': total_pnl
                        }
                    )
                    
                    # Execute exits
                    long_exit_success = await self.integration_manager.execute_trade_signal(long_exit)
                    short_exit_success = await self.integration_manager.execute_trade_signal(short_exit)
                    
                    if long_exit_success and short_exit_success: 
                        await self.integration_manager.alert_system.send_alert(
                            "SPREAD_EXIT",
                            "MEDIUM",
                            f"Debit Spread Exit: {ticker} {exit_reason} "
                            f"P & L: ${total_pnl:.2f} ({pnl_pct: .1%})"
                        )
                        
                        positions_to_remove.append(i)
                        self.logger.info(f"Spread position closed: {ticker} {exit_reason}")
                
            except Exception as e: 
                self.logger.error(f"Error managing position {i}: {e}")
        
        # Remove closed positions
        for i in reversed(positions_to_remove): 
            self.active_positions.pop(i)
    
    async def scan_opportunities(self)->List[ProductionTradeSignal]: 
        """Main strategy execution: scan and generate trade signals"""
        try: 
            # First manage existing positions
            await self.manage_positions()
            
            # Then scan for new opportunities if we have capacity
            if len(self.active_positions)  >=  self.max_positions: 
                return []
            
            # Check market hours
            if not await self.data_provider.is_market_open(): 
                return []
            
            # Scan for spread opportunities
            opportunities = await self.scan_spread_opportunities()
            
            # Execute top opportunities
            trade_signals = []
            max_new_positions = self.max_positions - len(self.active_positions)
            
            for opportunity in opportunities[: max_new_positions]:
                success = await self.execute_spread_trade(opportunity)
                if success: 
                    # Return the long leg signal for tracking
                    trade_signal = ProductionTradeSignal(
                        symbol = opportunity.ticker,
                        action = "BUY",
                        quantity = 1,  # Will be recalculated in execute_trade
                        option_type = "CALL",
                        strike_price = Decimal(str(opportunity.long_strike)),
                        expiration_date = datetime.strptime(opportunity.expiry_date, "%Y-%m-%d").date(),
                        premium = Decimal(str(opportunity.long_premium)),
                        confidence = opportunity.confidence,
                        strategy_name = self.strategy_name,
                        signal_strength = opportunity.trend_strength
                    )
                    trade_signals.append(trade_signal)
            
            return trade_signals
            
        except Exception as e: 
            self.logger.error(f"Error in debit spreads scan: {e}")
            return []
    
    def get_strategy_status(self)->Dict[str, Any]: 
        """Get current strategy status"""
        try: 
            total_pnl = 0.0
            position_details = []
            
            for position in self.active_positions: 
                entry_cost = position['net_debit'] * position['contracts'] * 100
                total_pnl += entry_cost  # This would be updated with current values
                
                position_details.append({
                    'ticker': position['ticker'],
                    'long_strike': float(position['long_signal'].strike_price),
                    'short_strike': float(position['short_signal'].strike_price),
                    'expiry': position['expiry_date'].isoformat(),
                    'contracts': position['contracts'],
                    'net_debit': position['net_debit'],
                    'max_profit': position['max_profit'],
                    'breakeven': position['breakeven'],
                    'confidence': position['confidence'],
                    'days_to_expiry': (position['expiry_date'] - date.today()).days
                })
            
            return {
                'strategy_name': self.strategy_name,
                'is_active': True,
                'active_positions': len(self.active_positions),
                'max_positions': self.max_positions,
                'total_pnl': total_pnl,
                'position_details': position_details,
                'last_scan': datetime.now().isoformat(),
                'config': {
                    'max_positions': self.max_positions,
                    'max_position_size': self.max_position_size,
                    'min_dte': self.min_dte,
                    'max_dte': self.max_dte,
                    'min_risk_reward': self.min_risk_reward,
                    'profit_target': self.profit_target,
                    'stop_loss': self.stop_loss
                }
            }
            
        except Exception as e: 
            self.logger.error(f"Error getting strategy status: {e}")
            return {'strategy_name': self.strategy_name, 'error': str(e)}


    async def run_strategy(self): 
        """Main strategy execution loop"""
        self.logger.info("Starting Production Debit Spreads Strategy")
        
        try: 
            while True: 
                # Scan for debit spread opportunities
                signals = await self.scan_opportunities()
                
                # Execute trades for signals
                if signals: 
                    await self.execute_trades(signals)
                
                # Wait before next scan
                await asyncio.sleep(180)  # 3 minutes between scans
                
        except Exception as e: 
            self.logger.error(f"Error in debit spreads strategy main loop: {e}")


def create_production_debit_spreads(integration_manager, data_provider: ReliableDataProvider, 
                                   config: dict)->ProductionDebitSpreads:
    """Factory function to create ProductionDebitSpreads strategy"""
    return ProductionDebitSpreads(integration_manager, data_provider, config)