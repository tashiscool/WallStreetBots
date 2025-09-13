#!/usr / bin / env python3
"""
Production SPX Credit Spreads Strategy
WSB - style 0DTE / short - term credit spreads with defined risk
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
class CreditSpreadOpportunity: 
    """Production credit spread opportunity"""
    ticker: str
    strategy_type: str  # "put_credit_spread", "call_credit_spread", "iron_condor"
    expiry_date: str
    dte: int
    short_strike: float
    long_strike: float
    spread_width: float
    net_credit: float
    max_profit: float
    max_loss: float
    short_delta: float
    prob_profit: float
    profit_target: float
    break_even_lower: float
    break_even_upper: float
    underlying_price: float
    expected_move: float
    volume_score: float
    # For iron condors
    put_short_strike: Optional[float] = None
    put_long_strike: Optional[float] = None
    call_short_strike: Optional[float] = None
    call_long_strike: Optional[float] = None


class ProductionSPXCreditSpreads: 
    """
    Production SPX Credit Spreads Strategy
    
    Strategy Logic: 
    1. Scans for 0DTE and short - term credit spread opportunities
    2. Targets ~30 delta short strikes for optimal risk / reward
    3. Implements rapid 25% profit - taking discipline
    4. Focuses on SPX / SPY for liquidity and tax advantages
    5. Manages defined - risk spreads with automatic exits
    
    Risk Management: 
    - Maximum 5% account risk per spread complex
    - Maximum 3 concurrent credit spread positions
    - 25% profit target with automatic closing
    - 75% stop loss (protect remaining credit)
    - Time-based exits: close 1 hour before expiration
    - Pin risk management at expiration
    """
    
    def __init__(self, integration_manager, data_provider: ReliableDataProvider, config: dict):
        self.strategy_name="spx_credit_spreads"
        self.integration_manager = integration_manager
        self.data_provider = data_provider
        self.config = config
        self.logger=logging.getLogger(__name__)
        
        # Core components
        self.options_selector=SmartOptionsSelector(data_provider)
        self.risk_manager=RealTimeRiskManager()
        self.bs_engine=BlackScholesEngine()
        
        # Strategy configuration
        self.credit_tickers=config.get('watchlist', [
            "SPY",   # SPDR S & P 500 ETF (preferred for production)
            "QQQ",   # Invesco QQQ ETF
            "IWM",   # Russell 2000 ETF
        ])
        
        # Risk parameters
        self.max_positions=config.get('max_positions', 3)
        self.max_position_size=config.get('max_position_size', 0.05)  # 5% per spread
        self.target_short_delta=config.get('target_short_delta', 0.30)
        self.max_dte=config.get('max_dte', 3)  # Maximum 3 DTE
        
        # Credit spread parameters
        self.min_net_credit=config.get('min_net_credit', 0.10)
        self.min_spread_width=config.get('min_spread_width', 5.0)
        self.max_spread_width=config.get('max_spread_width', 20.0)
        self.min_prob_profit=config.get('min_prob_profit', 60.0)  # 60% min
        
        # Exit criteria
        self.profit_target_pct=config.get('profit_target_pct', 0.25)  # 25%
        self.stop_loss_pct=config.get('stop_loss_pct', 0.75)  # 75% of credit lost
        self.time_exit_minutes=config.get('time_exit_minutes', 60)  # 1 hour before expiry
        
        # Active positions tracking
        self.active_positions: List[Dict[str, Any]] = []
        
        self.logger.info("ProductionSPXCreditSpreads strategy initialized")
    
    def norm_cdf(self, x: float)->float:
        """Standard normal cumulative distribution function"""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))
    
    def black_scholes_put(self, S: float, K: float, T: float, r: float, sigma: float)->Tuple[float, float]: 
        """Black - Scholes put price and delta"""
        if T  <=  0 or sigma  <=  0: 
            return max(K - S, 0), -1.0 if S  <  K else 0.0
            
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        put_price = K * math.exp(-r * T) * self.norm_cdf(-d2) - S * self.norm_cdf(-d1)
        delta = -self.norm_cdf(-d1)
        
        return max(put_price, 0), delta
    
    def black_scholes_call(self, S: float, K: float, T: float, r: float, sigma: float)->Tuple[float, float]: 
        """Black - Scholes call price and delta"""
        if T  <=  0 or sigma  <=  0: 
            return max(S - K, 0), 1.0 if S  >  K else 0.0
            
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        call_price = S * self.norm_cdf(d1) - K * math.exp(-r * T) * self.norm_cdf(d2)
        delta = self.norm_cdf(d1)
        
        return max(call_price, 0), delta
    
    async def get_available_expiries(self, ticker: str)->List[Tuple[str, int]]: 
        """Get available expiries with DTE"""
        try: 
            expiries = await self.data_provider.get_option_expiries(ticker)
            if not expiries: 
                return []
            
            valid_expiries = []
            today = date.today()
            
            for exp_str in expiries: 
                try: 
                    exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                    dte = (exp_date-today).days
                    
                    # Focus on short - term expiries (0 - 3 DTE)
                    if 0  <=  dte  <=  self.max_dte: 
                        valid_expiries.append((exp_str, dte))
                        
                except: 
                    continue
            
            return sorted(valid_expiries, key=lambda x: x[1])  # Sort by DTE
            
        except Exception as e: 
            self.logger.error(f"Error getting expiries for {ticker}: {e}")
            return []
    
    async def get_expected_move(self, ticker: str)->float:
        """Calculate expected daily move from recent volatility"""
        try: 
            # Get 20 days of historical data
            hist_data = await self.data_provider.get_historical_data(ticker, "20d")
            if hist_data.empty or len(hist_data)  <  10: 
                return 0.015  # Default 1.5%
            
            # Calculate daily returns
            returns = hist_data['close'].pct_change().dropna()
            daily_vol = returns.std()
            
            # Expected move is approximately 1 standard deviation
            return min(0.05, max(0.01, daily_vol))
            
        except Exception as e: 
            self.logger.error(f"Error calculating expected move for {ticker}: {e}")
            return 0.015
    
    async def find_target_delta_strike(self, ticker: str, expiry: str, option_type: str, 
                                     target_delta: float, spot_price: float)->Tuple[Optional[float], float, float]: 
        """Find strike closest to target delta"""
        try: 
            # Get options chain
            options_data = await self.data_provider.get_options_chain(ticker, expiry)
            if not options_data: 
                return None, 0.0, 0.0
            
            if option_type ==  "put": options_list=options_data.get('puts', [])
            else: 
                options_list = options_data.get('calls', [])
            
            if not options_list: 
                return None, 0.0, 0.0
            
            # Filter for liquid options
            liquid_options = []
            for option in options_list: 
                volume = option.get('volume', 0)
                open_interest = option.get('open_interest', 0)
                bid = option.get('bid', 0)
                ask = option.get('ask', 0)
                
                if (bid  >  0.05 and ask  >  0.05 and 
                    (volume  >=  10 or open_interest  >=  50)): 
                    liquid_options.append(option)
            
            if not liquid_options: 
                return None, 0.0, 0.0
            
            # Find strike closest to target delta
            best_option = None
            best_delta_diff = float('inf')
            
            # Estimate time to expiry
            exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
            dte = (exp_date-date.today()).days
            time_to_exp = max(0.001, dte / 365.0)  # Minimum time value
            
            for option in liquid_options: 
                strike = option['strike']
                
                # Use actual delta if available, otherwise estimate
                if 'delta' in option and option['delta'] is not None: 
                    actual_delta = abs(option['delta'])
                else: 
                    # Estimate using Black - Scholes
                    iv_estimate = 0.20  # Default IV
                    if option_type  ==  "put": 
                        _, delta=self.black_scholes_put(spot_price, strike, time_to_exp, 0.04, iv_estimate)
                        actual_delta = abs(delta)
                    else: 
                        _, delta=self.black_scholes_call(spot_price, strike, time_to_exp, 0.04, iv_estimate)
                        actual_delta = abs(delta)
                
                delta_diff = abs(actual_delta - target_delta)
                
                if delta_diff  <  best_delta_diff: 
                    best_delta_diff = delta_diff
                    best_option = option
            
            if best_option: 
                strike = best_option['strike']
                premium = (best_option['bid'] + best_option['ask']) / 2
                delta = abs(best_option.get('delta', target_delta))
                
                return strike, delta, premium
            
            return None, 0.0, 0.0
            
        except Exception as e: 
            self.logger.error(f"Error finding target delta strike: {e}")
            return None, 0.0, 0.0
    
    def calculate_spread_metrics(self, short_strike: float, long_strike: float,
                               short_premium: float, long_premium: float)->Tuple[float, float, float]: 
        """Calculate spread financial metrics"""
        spread_width = abs(short_strike-long_strike)
        net_credit = short_premium - long_premium
        max_profit = net_credit
        max_loss = spread_width - net_credit
        
        return net_credit, max_profit, max_loss
    
    async def scan_credit_spread_opportunities(self)->List[CreditSpreadOpportunity]: 
        """Scan for credit spread opportunities"""
        opportunities = []
        
        self.logger.info("Scanning for credit spread opportunities")
        
        for ticker in self.credit_tickers: 
            try: 
                # Skip if we already have a position
                if any(pos['ticker']  ==  ticker for pos in self.active_positions): 
                    continue
                
                # Get current price
                current_price = await self.data_provider.get_current_price(ticker)
                if not current_price: 
                    continue
                
                # Get expected move
                expected_move = await self.get_expected_move(ticker)
                
                # Get available expiries
                expiries = await self.get_available_expiries(ticker)
                if not expiries: 
                    continue
                
                self.logger.info(f"Analyzing {ticker}: ${current_price:.2f}, expected move: ±{expected_move:.1%}")
                
                # Analyze each expiry
                for expiry, dte in expiries[: 2]: # Limit to 2 nearest expiries
                    
                    # PUT CREDIT SPREADS (bullish / neutral bias)
                    put_short_strike, put_short_delta, put_short_premium = \
                        await self.find_target_delta_strike(ticker, expiry, "put", self.target_short_delta, current_price)
                    
                    if put_short_strike and put_short_premium  >  0: 
                        # Calculate spread width (2 - 5% of underlying)
                        spread_width = min(self.max_spread_width, 
                                         max(self.min_spread_width, current_price * 0.03))
                        put_long_strike = put_short_strike-spread_width
                        
                        # Get long put premium
                        _, _, put_long_premium = await self.find_target_delta_strike(
                            ticker, expiry, "put", 0.15, current_price
                        )
                        
                        if put_long_premium  >  0: 
                            net_credit, max_profit, max_loss=self.calculate_spread_metrics(
                                put_short_strike, put_long_strike, put_short_premium, put_long_premium
                            )
                            
                            if net_credit  >=  self.min_net_credit: 
                                prob_profit = (1 - put_short_delta) * 100
                                
                                if prob_profit  >=  self.min_prob_profit: 
                                    opportunity = CreditSpreadOpportunity(
                                        ticker=ticker,
                                        strategy_type = "put_credit_spread",
                                        expiry_date=expiry,
                                        dte=dte,
                                        short_strike=put_short_strike,
                                        long_strike=put_long_strike,
                                        spread_width=spread_width,
                                        net_credit=net_credit,
                                        max_profit=max_profit,
                                        max_loss=max_loss,
                                        short_delta=put_short_delta,
                                        prob_profit=prob_profit,
                                        profit_target = net_credit * self.profit_target_pct,
                                        break_even_lower = put_short_strike-net_credit,
                                        break_even_upper = float('inf'),
                                        underlying_price=current_price,
                                        expected_move=expected_move,
                                        volume_score = 70.0
                                    )
                                    
                                    opportunities.append(opportunity)
                    
                    # CALL CREDIT SPREADS (bearish / neutral bias)
                    call_short_strike, call_short_delta, call_short_premium = \
                        await self.find_target_delta_strike(ticker, expiry, "call", self.target_short_delta, current_price)
                    
                    if call_short_strike and call_short_premium  >  0: 
                        spread_width = min(self.max_spread_width,
                                         max(self.min_spread_width, current_price * 0.03))
                        call_long_strike = call_short_strike+spread_width
                        
                        _, _, call_long_premium = await self.find_target_delta_strike(
                            ticker, expiry, "call", 0.15, current_price
                        )
                        
                        if call_long_premium  >  0: 
                            net_credit, max_profit, max_loss=self.calculate_spread_metrics(
                                call_short_strike, call_long_strike, call_short_premium, call_long_premium
                            )
                            
                            if net_credit  >=  self.min_net_credit: 
                                prob_profit = (1 - call_short_delta) * 100
                                
                                if prob_profit  >=  self.min_prob_profit: 
                                    opportunity = CreditSpreadOpportunity(
                                        ticker=ticker,
                                        strategy_type = "call_credit_spread",
                                        expiry_date=expiry,
                                        dte=dte,
                                        short_strike=call_short_strike,
                                        long_strike=call_long_strike,
                                        spread_width=spread_width,
                                        net_credit=net_credit,
                                        max_profit=max_profit,
                                        max_loss=max_loss,
                                        short_delta=call_short_delta,
                                        prob_profit=prob_profit,
                                        profit_target = net_credit * self.profit_target_pct,
                                        break_even_lower = 0,
                                        break_even_upper = call_short_strike+net_credit,
                                        underlying_price=current_price,
                                        expected_move=expected_move,
                                        volume_score = 70.0
                                    )
                                    
                                    opportunities.append(opportunity)
                
            except Exception as e: 
                self.logger.error(f"Error scanning {ticker}: {e}")
                continue
        
        # Sort by risk - adjusted return
        opportunities.sort(
            key = lambda x: (x.prob_profit / 100.0) * (x.max_profit / max(x.max_loss, 0.01)),
            reverse = True)
        
        return opportunities
    
    async def execute_credit_spread(self, opportunity: CreditSpreadOpportunity)->bool:
        """Execute credit spread trade"""
        try: 
            # Check if we can add more positions
            if len(self.active_positions)  >=  self.max_positions: 
                self.logger.info("Max credit spread positions reached, skipping trade")
                return False
            
            # Calculate position size based on max loss
            portfolio_value = await self.integration_manager.get_portfolio_value()
            max_risk = portfolio_value * self.max_position_size
            
            # Size based on maximum loss of the spread
            contracts = max(1, int(max_risk / (opportunity.max_loss * 100)))
            contracts = min(contracts, 5)  # Max 5 contracts for credit spreads
            
            # Create trade signals for both legs
            short_signal = ProductionTradeSignal(
                symbol = opportunity.ticker,
                action = "SELL",  # Sell short leg
                quantity=contracts,
                option_type = "PUT" if "put" in opportunity.strategy_type else "CALL",
                strike_price = Decimal(str(opportunity.short_strike)),
                expiration_date = datetime.strptime(opportunity.expiry_date, "%Y-%m-%d").date(),
                premium = Decimal(str(opportunity.net_credit * 0.6)),  # Approximate short leg premium
                confidence = min(1.0, opportunity.prob_profit / 100.0),
                strategy_name = self.strategy_name,
                signal_strength = min(1.0, opportunity.prob_profit / 100.0),
                metadata = {
                    'spread_type': opportunity.strategy_type,
                    'leg': 'short',
                    'spread_id': f"{opportunity.ticker}_{opportunity.expiry_date}_{opportunity.short_strike}_{opportunity.long_strike}",
                    'long_strike': opportunity.long_strike,
                    'net_credit': opportunity.net_credit,
                    'max_profit': opportunity.max_profit,
                    'max_loss': opportunity.max_loss,
                    'prob_profit': opportunity.prob_profit,
                    'dte': opportunity.dte
                }
            )
            
            long_signal = ProductionTradeSignal(
                symbol = opportunity.ticker,
                action = "BUY",  # Buy long leg for protection
                quantity=contracts,
                option_type = "PUT" if "put" in opportunity.strategy_type else "CALL",
                strike_price = Decimal(str(opportunity.long_strike)),
                expiration_date = datetime.strptime(opportunity.expiry_date, "%Y-%m-%d").date(),
                premium = Decimal(str(opportunity.net_credit * 0.4)),  # Approximate long leg premium
                confidence = min(1.0, opportunity.prob_profit / 100.0),
                strategy_name = self.strategy_name,
                signal_strength = min(1.0, opportunity.prob_profit / 100.0),
                metadata = {
                    'spread_type': opportunity.strategy_type,
                    'leg': 'long',
                    'spread_id': f"{opportunity.ticker}_{opportunity.expiry_date}_{opportunity.short_strike}_{opportunity.long_strike}",
                    'short_strike': opportunity.short_strike,
                    'net_credit': opportunity.net_credit,
                    'max_profit': opportunity.max_profit,
                    'max_loss': opportunity.max_loss,
                    'prob_profit': opportunity.prob_profit,
                    'dte': opportunity.dte
                }
            )
            
            # Execute both legs
            short_success = await self.integration_manager.execute_trade_signal(short_signal)
            long_success = await self.integration_manager.execute_trade_signal(long_signal)
            
            if short_success and long_success: 
                # Track position
                position = {
                    'ticker': opportunity.ticker,
                    'strategy_type': opportunity.strategy_type,
                    'spread_id': f"{opportunity.ticker}_{opportunity.expiry_date}_{opportunity.short_strike}_{opportunity.long_strike}",
                    'short_signal': short_signal,
                    'long_signal': long_signal,
                    'entry_time': datetime.now(),
                    'contracts': contracts,
                    'net_credit': opportunity.net_credit,
                    'max_profit': opportunity.max_profit,
                    'max_loss': opportunity.max_loss,
                    'profit_target': opportunity.profit_target,
                    'cost_basis': -opportunity.net_credit * contracts * 100,  # Negative for credit received
                    'expiry_date': datetime.strptime(opportunity.expiry_date, "%Y-%m-%d").date(),
                    'dte': opportunity.dte,
                    'prob_profit': opportunity.prob_profit
                }
                
                self.active_positions.append(position)
                
                await self.integration_manager.alert_system.send_alert(
                    "CREDIT_SPREAD_ENTRY",
                    "MEDIUM",
                    f"Credit Spread Entry: {opportunity.ticker} "
                    f"{opportunity.strategy_type.replace('_', ' ').title()} "
                    f"{opportunity.short_strike}/{opportunity.long_strike} "
                    f"{contracts} contracts ${opportunity.net_credit: .2f} credit"
                )
                
                self.logger.info(f"Credit spread executed: {opportunity.ticker}")
                return True
            
            return False
            
        except Exception as e: 
            self.logger.error(f"Error executing credit spread for {opportunity.ticker}: {e}")
            return False
    
    async def manage_positions(self): 
        """Manage existing credit spread positions"""
        positions_to_remove = []
        current_time = datetime.now()
        
        for i, position in enumerate(self.active_positions): 
            try: 
                ticker = position['ticker']
                contracts = position['contracts']
                net_credit = position['net_credit']
                
                # Calculate time to expiry
                time_to_expiry = datetime.combine(position['expiry_date'], datetime.min.time()) - current_time
                minutes_to_expiry = time_to_expiry.total_seconds() / 60
                
                # Get current spread value by checking both legs
                short_price = await self.data_provider.get_option_price(
                    ticker,
                    position['short_signal'].strike_price,
                    position['short_signal'].expiration_date,
                    position['short_signal'].option_type.lower()
                )
                
                long_price = await self.data_provider.get_option_price(
                    ticker,
                    position['long_signal'].strike_price,
                    position['long_signal'].expiration_date,
                    position['long_signal'].option_type.lower()
                )
                
                if not (short_price and long_price): 
                    continue
                
                # Current spread value (what we'd pay to close)
                current_spread_value = short_price-long_price
                
                # P & L calculation (we received credit initially)
                pnl_per_contract = net_credit - current_spread_value
                total_pnl = pnl_per_contract * contracts * 100
                pnl_pct = (pnl_per_contract / net_credit) * 100 if net_credit  >  0 else 0
                
                # Check exit conditions
                should_exit = False
                exit_reason = ""
                
                # 1. Profit target (25% of credit received)
                if pnl_per_contract  >=  position['profit_target']: 
                    should_exit = True
                    exit_reason = "PROFIT_TARGET"
                
                # 2. Stop loss (lost 75% of credit received)
                elif pnl_per_contract  <=  -net_credit * self.stop_loss_pct: 
                    should_exit = True
                    exit_reason = "STOP_LOSS"
                
                # 3. Time-based exit (1 hour before expiry)
                elif minutes_to_expiry  <=  self.time_exit_minutes: 
                    should_exit = True
                    exit_reason = "TIME_EXIT"
                
                # 4. Same day expiry exit (for 0DTE trades)
                elif position['dte']  ==  0 and current_time.hour  >=  15:  # 3pm ET
                    should_exit = True
                    exit_reason = "0DTE_EOD_EXIT"
                
                if should_exit: 
                    # Create exit signals to close both legs
                    short_exit = ProductionTradeSignal(
                        symbol=ticker,
                        action = "BUY",  # Buy back short leg
                        quantity=contracts,
                        option_type = position['short_signal'].option_type,
                        strike_price = position['short_signal'].strike_price,
                        expiration_date = position['short_signal'].expiration_date,
                        premium = Decimal(str(short_price)),
                        strategy_name = self.strategy_name,
                        metadata = {
                            'spread_action': 'close_short',
                            'exit_reason': exit_reason,
                            'pnl_per_contract': pnl_per_contract,
                            'total_pnl': total_pnl,
                            'pnl_pct': pnl_pct,
                            'spread_id': position['spread_id']
                        }
                    )
                    
                    long_exit = ProductionTradeSignal(
                        symbol=ticker,
                        action = "SELL",  # Sell back long leg
                        quantity=contracts,
                        option_type = position['long_signal'].option_type,
                        strike_price = position['long_signal'].strike_price,
                        expiration_date = position['long_signal'].expiration_date,
                        premium = Decimal(str(long_price)),
                        strategy_name = self.strategy_name,
                        metadata = {
                            'spread_action': 'close_long',
                            'exit_reason': exit_reason,
                            'pnl_per_contract': pnl_per_contract,
                            'total_pnl': total_pnl,
                            'pnl_pct': pnl_pct,
                            'spread_id': position['spread_id']
                        }
                    )
                    
                    # Execute exits
                    short_exit_success = await self.integration_manager.execute_trade_signal(short_exit)
                    long_exit_success = await self.integration_manager.execute_trade_signal(long_exit)
                    
                    if short_exit_success and long_exit_success: 
                        await self.integration_manager.alert_system.send_alert(
                            "CREDIT_SPREAD_EXIT",
                            "MEDIUM",
                            f"Credit Spread Exit: {ticker} {exit_reason} "
                            f"P & L: ${total_pnl:.0f} ({pnl_pct: .1%}) "
                            f"Spread value: ${current_spread_value:.2f}"
                        )
                        
                        positions_to_remove.append(i)
                        self.logger.info(f"Credit spread closed: {ticker} {exit_reason}")
                
            except Exception as e: 
                self.logger.error(f"Error managing credit spread position {i}: {e}")
        
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
            
            # Only scan during market hours
            if not await self.data_provider.is_market_open(): 
                return []
            
            # Don't initiate new spreads too close to market close
            current_hour = datetime.now().hour
            if current_hour  >=  15:  # After 3pm ET
                return []
            
            # Scan for credit spread opportunities
            opportunities = await self.scan_credit_spread_opportunities()
            
            # Execute top opportunities
            trade_signals = []
            max_new_positions = self.max_positions - len(self.active_positions)
            
            for opportunity in opportunities[: max_new_positions]:
                success = await self.execute_credit_spread(opportunity)
                if success: 
                    # Return the short leg signal for tracking
                    trade_signal = ProductionTradeSignal(
                        symbol = opportunity.ticker,
                        action = "SELL",
                        quantity = 1,  # Will be recalculated in execute_trade
                        option_type = "PUT" if "put" in opportunity.strategy_type else "CALL",
                        strike_price = Decimal(str(opportunity.short_strike)),
                        expiration_date = datetime.strptime(opportunity.expiry_date, "%Y-%m-%d").date(),
                        premium = Decimal(str(opportunity.net_credit)),
                        confidence = min(1.0, opportunity.prob_profit / 100.0),
                        strategy_name = self.strategy_name,
                        signal_strength = min(1.0, opportunity.prob_profit / 100.0)
                    )
                    trade_signals.append(trade_signal)
            
            return trade_signals
            
        except Exception as e: 
            self.logger.error(f"Error in credit spreads scan: {e}")
            return []
    
    def get_strategy_status(self)->Dict[str, Any]: 
        """Get current strategy status"""
        try: 
            total_credit_received = sum(-pos['cost_basis'] for pos in self.active_positions)
            position_details = []
            
            for position in self.active_positions: 
                time_to_expiry = datetime.combine(position['expiry_date'], datetime.min.time()) - datetime.now()
                hours_to_expiry = time_to_expiry.total_seconds() / 3600
                
                position_details.append({
                    'ticker': position['ticker'],
                    'strategy_type': position['strategy_type'],
                    'short_strike': float(position['short_signal'].strike_price),
                    'long_strike': float(position['long_signal'].strike_price),
                    'expiry': position['expiry_date'].isoformat(),
                    'contracts': position['contracts'],
                    'net_credit': position['net_credit'],
                    'max_profit': position['max_profit'],
                    'max_loss': position['max_loss'],
                    'profit_target': position['profit_target'],
                    'prob_profit': position['prob_profit'],
                    'dte': position['dte'],
                    'hours_to_expiry': round(hours_to_expiry, 1),
                    'entry_time': position['entry_time'].isoformat()
                })
            
            return {
                'strategy_name': self.strategy_name,
                'is_active': True,
                'active_positions': len(self.active_positions),
                'max_positions': self.max_positions,
                'total_credit_received': total_credit_received,
                'position_details': position_details,
                'last_scan': datetime.now().isoformat(),
                'config': {
                    'max_positions': self.max_positions,
                    'max_position_size': self.max_position_size,
                    'target_short_delta': self.target_short_delta,
                    'max_dte': self.max_dte,
                    'profit_target_pct': self.profit_target_pct,
                    'stop_loss_pct': self.stop_loss_pct,
                    'min_net_credit': self.min_net_credit,
                    'min_prob_profit': self.min_prob_profit
                }
            }
            
        except Exception as e: 
            self.logger.error(f"Error getting credit spreads status: {e}")
            return {'strategy_name': self.strategy_name, 'error': str(e)}


    async def run_strategy(self): 
        """Main strategy execution loop"""
        self.logger.info("Starting Production SPX Credit Spreads Strategy")
        
        try: 
            while True: 
                # Scan for SPX credit spread opportunities
                signals = await self.scan_opportunities()
                
                # Execute trades for signals
                if signals: 
                    await self.execute_trades(signals)
                
                # Wait before next scan (0DTE is very active)
                await asyncio.sleep(30)  # 30 seconds between scans for 0DTE
                
        except Exception as e: 
            self.logger.error(f"Error in SPX credit spreads strategy main loop: {e}")


def create_production_spx_credit_spreads(integration_manager, data_provider: ReliableDataProvider, 
                                        config: dict)->ProductionSPXCreditSpreads:
    """Factory function to create ProductionSPXCreditSpreads strategy"""
    return ProductionSPXCreditSpreads(integration_manager, data_provider, config)