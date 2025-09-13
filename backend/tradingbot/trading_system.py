"""
Integrated Trading System
Main orchestrator that ties together all components of the options trading playbook.
"""

from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional

from .options_calculator import OptionsTradeCalculator, TradeCalculation
from .market_regime import SignalGenerator, TechnicalIndicators, MarketSignal, SignalType
from .risk_management import RiskManager, Position, RiskParameters
from .exit_planning import ExitStrategy, ScenarioAnalyzer, ExitSignal
from .alert_system import TradingAlertSystem, ExecutionChecklistManager, AlertType, AlertPriority


@dataclass
class TradingConfig: 
    """Configuration for the trading system"""

    # Account settings
    account_size: float=500000.0
    max_position_risk_pct: float=0.10  # 10% per position
    max_total_risk_pct: float=0.30     # 30% total portfolio risk

    # Universe settings
    target_tickers: List[str] = None

    # Market data settings
    data_refresh_interval: int = 300     # 5 minutes

    # Risk settings
    risk_params: RiskParameters = None

    # Alert settings
    enable_alerts: bool = True
    alert_channels: List[str] = None

    def __post_init__(self): 
        if self.target_tickers is None: 
            self.target_tickers=[
                'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'AVGO', 'AMD', 'TSLA'
            ]

        if self.risk_params is None: 
            self.risk_params=RiskParameters()

        if self.alert_channels is None: 
            self.alert_channels=['desktop', 'webhook']


@dataclass
class SystemState: 
    """Current state of the trading system"""
    is_running: bool = False
    last_scan_time: Optional[datetime] = None
    active_positions: int = 0
    total_portfolio_risk: float=0.0
    alerts_sent_today: int = 0
    errors_today: int = 0


class IntegratedTradingSystem: 
    """
    Main trading system that orchestrates all components.
    Implements the complete 240% options trading playbook.
    """

    def __init__(self, config: TradingConfig=None):
        self.config=config or TradingConfig()
        self.state=SystemState()

        # Initialize all components
        self.options_calculator=OptionsTradeCalculator()
        self.signal_generator=SignalGenerator()
        self.risk_manager=RiskManager(self.config.risk_params)
        self.exit_strategy=ExitStrategy()
        self.scenario_analyzer=ScenarioAnalyzer()
        self.alert_system=TradingAlertSystem()
        self.checklist_manager=ExecutionChecklistManager()

        # Setup logging
        logging.basicConfig(
            level = logging.INFO,
            format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger=logging.getLogger(__name__)

        self.logger.info("Trading system initialized successfully")

    async def start_system(self): 
        """Start the trading system main loop"""
        self.state.is_running = True
        self.logger.info("🚀 Trading system started")

        try: 
            while self.state.is_running: 
                await self._run_scan_cycle()
                await asyncio.sleep(self.config.data_refresh_interval)

        except Exception as e: 
            self.logger.error(f"Critical error in main loop: {e}")
            self.state.errors_today += 1
        finally: 
            self.state.is_running = False
            self.logger.info("Trading system stopped")

    def stop_system(self): 
        """Stop the trading system"""
        self.state.is_running = False
        self.logger.info("Stop signal sent to trading system")

    async def _run_scan_cycle(self): 
        """Run one complete scan cycle"""
        try: 
            self.logger.info("Starting scan cycle...")

            # 1. Get market data (placeholder - integrate with your data source)
            market_data = await self._fetch_market_data()

            # 2. Scan for new opportunities
            await self._scan_for_opportunities(market_data)

            # 3. Monitor existing positions
            await self._monitor_existing_positions(market_data)

            # 4. Update portfolio risk metrics
            await self._update_portfolio_metrics()

            # 5. Check for system maintenance tasks
            await self._run_maintenance_tasks()

            self.state.last_scan_time=datetime.now()

        except Exception as e: 
            self.logger.error(f"Error in scan cycle: {e}")
            self.state.errors_today += 1

    async def _fetch_market_data(self)->Dict: 
        """
        Fetch market data for all target tickers
        This is a placeholder - integrate with your preferred data source
        """
        # Placeholder implementation
        # In production, integrate with: 
        # - Alpaca API
        # - Yahoo Finance
        # - Alpha Vantage
        # - IEX Cloud
        # etc.

        market_data = {}

        for ticker in self.config.target_tickers: 
            # Placeholder data structure
            market_data[ticker] = {
                'current': {
                    'close': 200.0,  # Replace with real data
                    'high': 202.0,
                    'low': 198.0,
                    'volume': 1000000,
                    'ema_20': 198.5,
                    'ema_50': 195.0,
                    'ema_200': 190.0,
                    'rsi': 45.0,
                    'atr': 4.0,
                },
                'previous': {
                    'close': 201.0,
                    'high': 203.0,
                    'low': 199.0,
                    'volume': 900000,
                    'ema_20': 198.0,
                    'ema_50': 194.5,
                    'ema_200': 189.5,
                    'rsi': 48.0,
                    'atr': 3.8,
                },
                'earnings_in_7_days': False,
                'implied_volatility': 0.28
            }

        return market_data

    async def _scan_for_opportunities(self, market_data: Dict):
        """Scan market data for new trading opportunities"""

        for ticker in self.config.target_tickers: 
            if ticker not in market_data: 
                continue

            try: 
                current_data = market_data[ticker]['current']
                previous_data = market_data[ticker]['previous']

                # Convert to technical indicators
                current_indicators = self._create_indicators(current_data)
                previous_indicators = self._create_indicators(previous_data)

                # Generate signal
                signal = self.signal_generator.generate_signal(
                    current_indicators,
                    previous_indicators,
                    earnings_risk = market_data[ticker]['earnings_in_7_days'],
                    macro_risk = False  # You can add macro event detection
                )

                # Process signal
                await self._process_signal(ticker, signal, current_indicators, market_data[ticker])

            except Exception as e: 
                self.logger.error(f"Error scanning {ticker}: {e}")

    def _create_indicators(self, data: Dict)->TechnicalIndicators:
        """Convert raw market data to TechnicalIndicators"""
        return TechnicalIndicators(
            price = data['close'],
            ema_20 = data['ema_20'],
            ema_50 = data['ema_50'],
            ema_200 = data['ema_200'],
            rsi_14 = data['rsi'],
            atr_14 = data['atr'],
            volume = data['volume'],
            high_24h = data['high'],
            low_24h = data['low']
        )

    async def _process_signal(
        self,
        ticker: str,
        signal: MarketSignal,
        indicators: TechnicalIndicators,
        market_data: Dict
    ): 
        """Process trading signal and generate appropriate actions"""

        if signal.signal_type ==  SignalType.BUY and signal.confidence  >  0.7: 
            await self._handle_buy_signal(ticker, signal, indicators, market_data)

        elif signal.signal_type ==  SignalType.HOLD and "setup" in " ".join(signal.reasoning).lower(): 
            await self._handle_setup_signal(ticker, signal, indicators)

    async def _handle_buy_signal(
        self,
        ticker: str,
        signal: MarketSignal,
        indicators: TechnicalIndicators,
        market_data: Dict
    ): 
        """Handle buy signal - calculate trade and send alerts"""

        try: 
            # Calculate trade parameters
            trade_calc = self.options_calculator.calculate_trade(
                ticker=ticker,
                spot_price = indicators.price,
                account_size = self.config.account_size,
                implied_volatility = market_data['implied_volatility'],
                risk_pct = self.config.max_position_risk_pct
            )

            # Check if trade passes risk management
            if self._validate_trade_risk(trade_calc): 
                # Create execution checklist
                checklist_id = self.checklist_manager.create_entry_checklist(ticker, trade_calc)

                # Send alert
                await self._send_entry_alert(ticker, signal, trade_calc, checklist_id)

                self.logger.info(f"🚀 BUY SIGNAL generated for {ticker}")
            else: 
                self.logger.warning(f"❌ Trade for {ticker} rejected by risk management")

        except Exception as e: 
            self.logger.error(f"Error handling buy signal for {ticker}: {e}")

    async def _handle_setup_signal(
        self,
        ticker: str,
        signal: MarketSignal,
        indicators: TechnicalIndicators
    ): 
        """Handle setup signal - watch list addition"""

        # Send setup alert
        if self.config.enable_alerts: 
            from .alert_system import Alert

            alert = Alert(
                alert_type = AlertType.SETUP_DETECTED,
                priority = AlertPriority.MEDIUM,
                ticker=ticker,
                title = f"⚠️ SETUP: {ticker}",
                message = "Pullback setup detected. Monitor for reversal trigger.\n"
                       f"Price: ${indicators.price:.2f} | RSI: {indicators.rsi_14:.0f}",
                data = {
                    "signal_confidence": signal.confidence,
                    "reasoning": signal.reasoning
                }
            )

            self.alert_system.send_alert(alert)

        self.logger.info(f"📋 SETUP detected for {ticker}")

    def _validate_trade_risk(self, trade_calc: TradeCalculation)->bool:
        """Validate trade against risk management rules"""

        # Check individual position size
        if trade_calc.account_risk_pct  >  self.config.max_position_risk_pct * 100: 
            return False

        # Check total portfolio risk
        current_risk = self.risk_manager.calculate_portfolio_risk()
        if current_risk.risk_utilization  >  self.config.max_total_risk_pct: 
            return False

        # Check ticker concentration (simplified)
        # In production, you'd check existing positions in this ticker

        return True

    async def _send_entry_alert(
        self,
        ticker: str,
        signal: MarketSignal,
        trade_calc: TradeCalculation,
        checklist_id: str
    ): 
        """Send entry signal alert"""

        if not self.config.enable_alerts: 
            return

        from .alert_system import Alert

        alert = Alert(
            alert_type = AlertType.ENTRY_SIGNAL,
            priority = AlertPriority.HIGH,
            ticker=ticker,
            title = f"🚀 BUY SIGNAL: {ticker}",
            message = "Bull pullback reversal confirmed!\n"
                   f"Recommended: {trade_calc.recommended_contracts} x ${trade_calc.strike: .0f}C {trade_calc.expiry_date}\n"
                   f"Premium: ${trade_calc.estimated_premium:.2f} | Cost: ${trade_calc.total_cost:,.0f}\n"
                   f"Breakeven: ${trade_calc.breakeven_price:.2f} | Risk: {trade_calc.account_risk_pct:.1f}%\n"
                   f"Checklist: {checklist_id}",
            data = {
                "signal_confidence": signal.confidence,
                "trade_calculation": asdict(trade_calc),
                "checklist_id": checklist_id,
                "reasoning": signal.reasoning
            }
        )

        self.alert_system.send_alert(alert)
        self.state.alerts_sent_today += 1

    async def _monitor_existing_positions(self, market_data: Dict):
        """Monitor existing positions for exit signals"""

        open_positions = [pos for pos in self.risk_manager.positions if pos.status.value  ==  "open"]

        for position in open_positions: 
            ticker = position.ticker

            if ticker not in market_data: 
                continue

            try: 
                current_price = market_data[ticker]['current']['close']
                current_iv = market_data[ticker]['implied_volatility']

                # Analyze exit conditions
                exit_signals = self.exit_strategy.analyze_exit_conditions(
                    position=position,
                    current_spot=current_price,
                    current_iv=current_iv,
                    days_since_entry = (datetime.now() - position.entry_date).days
                )

                if exit_signals: 
                    await self._handle_exit_signals(position, exit_signals)

                # Run scenario analysis periodically
                if datetime.now().hour in [9, 12, 15]:  # 9am, 12pm, 3pm ET
                    await self._run_position_scenario_analysis(position, current_price, current_iv)

            except Exception as e: 
                self.logger.error(f"Error monitoring position {ticker}: {e}")

    async def _handle_exit_signals(self, position: Position, exit_signals: List[ExitSignal]):
        """Handle exit signals for a position"""

        strongest_signal = max(exit_signals, key=lambda x: x.strength.value)

        # Create exit checklist
        self.checklist_manager.create_exit_checklist(
            f"{position.ticker}_{position.entry_date.strftime('%Y % m % d')}",
            position.ticker,
            strongest_signal.reason.value
        )

        # Send exit alert
        if self.config.enable_alerts: 
            self.alert_system.create_exit_alert(
                position.ticker,
                exit_signals,
                asdict(position)
            )

        self.logger.info(f"🎯 EXIT SIGNAL for {position.ticker}: {strongest_signal.reason.value}")

    async def _run_position_scenario_analysis(
        self,
        position: Position,
        current_spot: float,
        current_iv: float
    ): 
        """Run scenario analysis for a position"""

        try: 
            scenarios = self.scenario_analyzer.run_comprehensive_analysis(
                position=position,
                current_spot=current_spot,
                current_iv = current_iv)

            exit_plan = self.scenario_analyzer.generate_exit_plan(scenarios)

            # Log key metrics
            expected_roi = exit_plan['summary']['expected_roi']
            win_rate = exit_plan['summary']['win_rate']

            self.logger.info(f"📊 Scenario analysis for {position.ticker}: "
                           f"Expected ROI: {expected_roi:+.1%}, Win Rate: {win_rate:.1%}")

            # Send risk alert if negative expected value
            if expected_roi  <  -0.2: 
                self.alert_system.create_risk_alert(
                    f"Negative expected value for {position.ticker}: {expected_roi:+.1%}",
                    {"position": asdict(position), "scenarios": exit_plan}
                )

        except Exception as e: 
            self.logger.error(f"Error in scenario analysis for {position.ticker}: {e}")

    async def _update_portfolio_metrics(self): 
        """Update portfolio - level risk metrics"""

        try: 
            portfolio_risk = self.risk_manager.calculate_portfolio_risk()

            # Update state
            self.state.active_positions = len([
                pos for pos in self.risk_manager.positions if pos.status.value  ==  "open"
            ])
            self.state.total_portfolio_risk=portfolio_risk.risk_utilization

            # Check for risk alerts
            if portfolio_risk.risk_utilization  >  self.config.max_total_risk_pct: 
                self.alert_system.create_risk_alert(
                    f"Portfolio risk utilization ({portfolio_risk.risk_utilization: .1%}) "
                    f"exceeds limit ({self.config.max_total_risk_pct: .1%})"
                )

        except Exception as e: 
            self.logger.error(f"Error updating portfolio metrics: {e}")

    async def _run_maintenance_tasks(self): 
        """Run periodic maintenance tasks"""

        # Reset daily counters at midnight
        if datetime.now().hour ==  0 and datetime.now().minute  <  5: 
            self.state.alerts_sent_today = 0
            self.state.errors_today = 0

        # Log system status every hour
        if datetime.now().minute ==  0: 
            self.logger.info("📈 System Status: "
                           f"Active Positions: {self.state.active_positions}, "
                           f"Risk Utilization: {self.state.total_portfolio_risk:.1%}, "
                           f"Alerts Today: {self.state.alerts_sent_today}")

    # Public API methods for manual interaction

    def add_position(self, position: Position)->bool:
        """Manually add a position to the system"""
        return self.risk_manager.add_position(position)

    def get_portfolio_status(self)->Dict: 
        """Get current portfolio status"""
        risk_report = self.risk_manager.generate_risk_report()

        return {
            "system_state": asdict(self.state),
            "portfolio_metrics": risk_report,
            "active_checklists": len(self.checklist_manager.get_active_checklists()),
            "config": asdict(self.config)
        }

    def force_scan(self): 
        """Force an immediate market scan"""
        asyncio.create_task(self._run_scan_cycle())

    def calculate_trade_for_ticker(
        self,
        ticker: str,
        spot_price: float,
        implied_volatility: float
    )->TradeCalculation: 
        """Calculate trade parameters for a specific ticker"""
        return self.options_calculator.calculate_trade(
            ticker=ticker,
            spot_price=spot_price,
            account_size = self.config.account_size,
            implied_volatility=implied_volatility,
            risk_pct = self.config.max_position_risk_pct
        )


if __name__ ==  "__main__": # Test the integrated system
    async def test_system(): 
        print("=== INTEGRATED TRADING SYSTEM TEST===")

        # Create system with custom config
        config = TradingConfig(
            account_size = 500000,
            max_position_risk_pct = 0.10,
            target_tickers = ['GOOGL', 'AAPL', 'MSFT']
        )

        system = IntegratedTradingSystem(config)

        # Test trade calculation
        trade_calc = system.calculate_trade_for_ticker(
            ticker = "GOOGL",
            spot_price = 207.0,
            implied_volatility = 0.28
        )

        print("Trade Calculation for GOOGL: ")
        print(f"  Contracts: {trade_calc.recommended_contracts}")
        print(f"  Strike: ${trade_calc.strike}")
        print(f"  Premium: ${trade_calc.estimated_premium:.2f}")
        print(f"  Total Cost: ${trade_calc.total_cost:,.0f}")
        print(f"  Risk %: {trade_calc.account_risk_pct:.1f}%")

        # Test portfolio status
        status = system.get_portfolio_status()
        print("\nPortfolio Status: ")
        print(f"  Active Positions: {status['system_state']['active_positions']}")
        print(f"  Risk Utilization: {status['system_state']['total_portfolio_risk']:.1%}")

        # Run one scan cycle
        print("\nRunning single scan cycle...")
        await system._run_scan_cycle()
        print("Scan completed successfully!")

    # Run the test
    asyncio.run(test_system())
