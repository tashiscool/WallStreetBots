"""Production 0DTE / Earnings Lotto Scanner
High - risk, high - reward options scanning with volatility analysis
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from .production_config import ConfigManager
from .production_logging import ProductionLogger
from .trading_interface import TradingInterface
from .unified_data_provider import UnifiedDataProvider


class LottoSignal(Enum):
    """Lotto trading signals"""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class LottoType(Enum):
    """Types of lotto plays"""

    ZERO_DTE = "zero_dte"
    EARNINGS_LOTTO = "earnings_lotto"
    VOLATILITY_SPIKE = "volatility_spike"
    GAMMA_SQUEEZE = "gamma_squeeze"
    MEME_STOCK = "meme_stock"


class OptionStrategy(Enum):
    """Option strategies for lotto plays"""

    CALL = "call"
    PUT = "put"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    BUTTERFLY = "butterfly"
    IRON_CONDOR = "iron_condor"


@dataclass
class VolatilityAnalysis:
    """Volatility analysis data"""

    ticker: str
    current_price: float
    implied_volatility: float
    historical_volatility: float
    iv_percentile: float
    iv_rank: float
    vix_level: float
    vix_percentile: float
    expected_move: float
    actual_move: float
    volatility_skew: float
    gamma_exposure: float
    options_volume: int
    put_call_ratio: float
    analysis_date: datetime = field(default_factory=datetime.now)


@dataclass
class LottoOption:
    """Lotto option data"""

    ticker: str
    option_type: OptionStrategy
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
    intrinsic_value: float
    time_value: float
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class LottoCandidate:
    """Lotto trading candidate"""

    ticker: str
    lotto_type: LottoType
    signal: LottoSignal
    volatility_analysis: VolatilityAnalysis
    confidence: float
    entry_price: float
    target_price: float
    stop_loss: float
    risk_reward_ratio: float
    position_size: int
    recommended_option: LottoOption | None = None
    lotto_score: float = 0.0
    risk_score: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class LottoPosition:
    """Lotto trading position"""

    ticker: str
    lotto_type: LottoType
    signal: LottoSignal
    option_strategy: OptionStrategy
    quantity: int
    entry_price: float
    current_price: float
    unrealized_pnl: float
    target_price: float
    stop_loss: float
    entry_date: datetime
    expiry_date: datetime
    days_to_expiry: int
    max_profit: float
    max_loss: float
    last_update: datetime = field(default_factory=datetime.now)
    status: str = "active"


class VolatilityAnalyzer:
    """Volatility analysis engine"""

    def __init__(self, logger: ProductionLogger):
        self.logger = logger

    def calculate_implied_volatility(
        self,
        option_price: float,
        spot_price: float,
        strike_price: float,
        risk_free_rate: float,
        time_to_expiry: float,
        option_type: str,
    ) -> float:
        """Calculate implied volatility using Black - Scholes"""
        try:
            # Simplified IV calculation - in production, use proper numerical methods
            time_to_expiry_years = time_to_expiry / 365.0

            # Rough IV estimation
            if option_type.lower() == "call":
                intrinsic_value = max(0, spot_price - strike_price)
            else:
                intrinsic_value = max(0, strike_price - spot_price)

            time_value = option_price - intrinsic_value

            if time_value <= 0:
                return 0.0

            # Rough IV calculation
            iv = math.sqrt(2 * math.pi / time_to_expiry_years) * time_value / spot_price
            return max(0.0, min(2.0, iv))  # Cap between 0% and 200%

        except Exception as e:
            self.logger.error(f"Error calculating implied volatility: {e}")
            return 0.0

    def calculate_historical_volatility(self, prices: list[float], period: int = 20) -> float:
        """Calculate historical volatility"""
        if len(prices) < period + 1:
            return 0.0

        returns = []
        for i in range(1, len(prices)):
            if prices[i - 1] != 0:
                ret = math.log(prices[i] / prices[i - 1])
                returns.append(ret)

        if len(returns) < period:
            return 0.0

        recent_returns = returns[-period:]
        mean_return = sum(recent_returns) / len(recent_returns)

        variance = sum((ret - mean_return) ** 2 for ret in recent_returns) / len(recent_returns)
        volatility = math.sqrt(variance * 252)  # Annualized

        return volatility

    def calculate_iv_percentile(self, current_iv: float, historical_ivs: list[float]) -> float:
        """Calculate IV percentile"""
        if not historical_ivs:
            return 0.5

        sorted_ivs = sorted(historical_ivs)
        count_below = sum(1 for iv in sorted_ivs if iv < current_iv)
        percentile = count_below / len(sorted_ivs)

        return percentile

    def calculate_iv_rank(self, current_iv: float, historical_ivs: list[float]) -> float:
        """Calculate IV rank"""
        if not historical_ivs:
            return 0.5

        min_iv = min(historical_ivs)
        max_iv = max(historical_ivs)

        if max_iv == min_iv:
            return 0.5

        rank = (current_iv - min_iv) / (max_iv - min_iv)
        return max(0.0, min(1.0, rank))

    def calculate_expected_move(self, spot_price: float, iv: float, time_to_expiry: int) -> float:
        """Calculate expected move"""
        time_to_expiry_years = time_to_expiry / 365.0
        expected_move = spot_price * iv * math.sqrt(time_to_expiry_years)
        return expected_move

    def calculate_volatility_skew(self, call_ivs: list[float], put_ivs: list[float]) -> float:
        """Calculate volatility skew"""
        if not call_ivs or not put_ivs:
            return 0.0

        avg_call_iv = sum(call_ivs) / len(call_ivs)
        avg_put_iv = sum(put_ivs) / len(put_ivs)

        skew = (avg_put_iv - avg_call_iv) / avg_call_iv if avg_call_iv > 0 else 0.0
        return skew

    def calculate_gamma_exposure(self, options_data: list[dict]) -> float:
        """Calculate gamma exposure"""
        total_gamma = 0.0
        total_volume = 0.0

        for option in options_data:
            gamma = option.get("gamma", 0.0)
            volume = option.get("volume", 0)
            total_gamma += gamma * volume
            total_volume += volume

        if total_volume == 0:
            return 0.0

        return total_gamma / total_volume


class LottoOptionsProvider:
    """Lotto options data provider"""

    def __init__(self, logger: ProductionLogger):
        self.logger = logger
        self.options_cache = {}

    async def get_zero_dte_options(self, ticker: str) -> list[LottoOption]:
        """Get 0DTE options for ticker"""
        try:
            # Mock implementation - in production, integrate with real options API
            current_price = 150.0  # Mock current price

            # Generate mock 0DTE options
            options = []
            strikes = [current_price * (1 + i * 0.02) for i in range(-5, 6)]  # ±10% strikes

            for strike in strikes:
                # Call option
                call_option = LottoOption(
                    ticker=ticker,
                    option_type=OptionStrategy.CALL,
                    strike_price=strike,
                    expiry_date=datetime.now() + timedelta(hours=6),  # 0DTE
                    days_to_expiry=0,
                    bid_price=max(0.01, strike * 0.01),
                    ask_price=max(0.02, strike * 0.015),
                    mid_price=max(0.015, strike * 0.0125),
                    volume=5000,
                    open_interest=10000,
                    implied_volatility=0.50,  # High IV for 0DTE
                    delta=0.5,
                    gamma=0.05,
                    theta=-0.2,  # High theta decay
                    vega=0.05,
                    intrinsic_value=max(0, current_price - strike),
                    time_value=max(0.01, strike * 0.01),
                )
                options.append(call_option)

                # Put option
                put_option = LottoOption(
                    ticker=ticker,
                    option_type=OptionStrategy.PUT,
                    strike_price=strike,
                    expiry_date=datetime.now() + timedelta(hours=6),  # 0DTE
                    days_to_expiry=0,
                    bid_price=max(0.01, strike * 0.01),
                    ask_price=max(0.02, strike * 0.015),
                    mid_price=max(0.015, strike * 0.0125),
                    volume=5000,
                    open_interest=10000,
                    implied_volatility=0.50,  # High IV for 0DTE
                    delta=-0.5,
                    gamma=0.05,
                    theta=-0.2,  # High theta decay
                    vega=0.05,
                    intrinsic_value=max(0, strike - current_price),
                    time_value=max(0.01, strike * 0.01),
                )
                options.append(put_option)

            self.logger.info(f"Retrieved {len(options)} 0DTE options for {ticker}")
            return options

        except Exception as e:
            self.logger.error(f"Error fetching 0DTE options for {ticker}: {e}")
            return []

    async def get_earnings_options(self, ticker: str, earnings_date: datetime) -> list[LottoOption]:
        """Get earnings options for ticker"""
        try:
            # Mock implementation - in production, integrate with real options API
            current_price = 150.0  # Mock current price
            days_to_earnings = (earnings_date - datetime.now()).days

            # Generate mock earnings options
            options = []
            strikes = [current_price * (1 + i * 0.05) for i in range(-3, 4)]  # ±15% strikes

            for strike in strikes:
                # Call option
                call_option = LottoOption(
                    ticker=ticker,
                    option_type=OptionStrategy.CALL,
                    strike_price=strike,
                    expiry_date=earnings_date,
                    days_to_expiry=days_to_earnings,
                    bid_price=max(0.01, strike * 0.03),
                    ask_price=max(0.02, strike * 0.035),
                    mid_price=max(0.015, strike * 0.0325),
                    volume=2000,
                    open_interest=5000,
                    implied_volatility=0.40,  # High IV for earnings
                    delta=0.5,
                    gamma=0.03,
                    theta=-0.1,
                    vega=0.1,
                    intrinsic_value=max(0, current_price - strike),
                    time_value=max(0.01, strike * 0.03),
                )
                options.append(call_option)

                # Put option
                put_option = LottoOption(
                    ticker=ticker,
                    option_type=OptionStrategy.PUT,
                    strike_price=strike,
                    expiry_date=earnings_date,
                    days_to_expiry=days_to_earnings,
                    bid_price=max(0.01, strike * 0.03),
                    ask_price=max(0.02, strike * 0.035),
                    mid_price=max(0.015, strike * 0.0325),
                    volume=2000,
                    open_interest=5000,
                    implied_volatility=0.40,  # High IV for earnings
                    delta=-0.5,
                    gamma=0.03,
                    theta=-0.1,
                    vega=0.1,
                    intrinsic_value=max(0, strike - current_price),
                    time_value=max(0.01, strike * 0.03),
                )
                options.append(put_option)

            self.logger.info(f"Retrieved {len(options)} earnings options for {ticker}")
            return options

        except Exception as e:
            self.logger.error(f"Error fetching earnings options for {ticker}: {e}")
            return []

    def find_best_lotto_option(
        self,
        options: list[LottoOption],
        lotto_type: LottoType,
        signal: LottoSignal,
        current_price: float,
    ) -> LottoOption | None:
        """Find best lotto option based on type and signal"""
        try:
            if not options:
                return None

            # Filter options based on lotto type and signal
            if lotto_type == LottoType.ZERO_DTE:
                # For 0DTE, look for options with high gamma and low time value
                filtered_options = [opt for opt in options if opt.days_to_expiry == 0]
                if filtered_options:
                    best_option = max(filtered_options, key=lambda x: x.gamma)
                    return best_option

            elif lotto_type == LottoType.EARNINGS_LOTTO:
                # For earnings, look for options with high vega and reasonable time value
                filtered_options = [opt for opt in options if opt.days_to_expiry <= 7]
                if filtered_options:
                    best_option = max(filtered_options, key=lambda x: x.vega)
                    return best_option

            elif lotto_type == LottoType.VOLATILITY_SPIKE:
                # For volatility spike, look for options with high vega
                best_option = max(options, key=lambda x: x.vega)
                return best_option

            elif lotto_type == LottoType.GAMMA_SQUEEZE:
                # For gamma squeeze, look for options with high gamma
                best_option = max(options, key=lambda x: x.gamma)
                return best_option

            # Default: find option closest to current price
            best_option = min(options, key=lambda x: abs(x.strike_price - current_price))
            return best_option

        except Exception as e:
            self.logger.error(f"Error finding best lotto option: {e}")
            return None


class LottoScannerStrategy:
    """Main lotto scanner strategy"""

    def __init__(
        self,
        trading_interface: TradingInterface,
        data_provider: UnifiedDataProvider,
        config: ConfigManager,
        logger: ProductionLogger,
    ):
        self.trading = trading_interface
        self.data = data_provider
        self.config = config
        self.logger = logger
        self.volatility_analyzer = VolatilityAnalyzer(logger)
        self.options_provider = LottoOptionsProvider(logger)
        self.active_positions = {}
        self.lotto_candidates = {}

        # Strategy parameters
        self.max_positions = 20
        self.max_position_size = 0.02  # 2% of portfolio per position
        self.min_lotto_score = 0.7
        self.max_risk_per_trade = 0.01  # 1% max risk per trade
        self.stop_loss_pct = 0.5  # 50% stop loss for lotto plays
        self.take_profit_pct = 2.0  # 200% take profit for lotto plays

        self.logger.info("LottoScannerStrategy initialized")

    async def scan_for_lotto_opportunities(self) -> list[LottoCandidate]:
        """Scan for lotto trading opportunities"""
        try:
            self.logger.info("Scanning for lotto opportunities")

            # Get universe of stocks to scan
            universe = self.config.trading.universe
            candidates = []

            for ticker in universe:
                try:
                    # Get historical data
                    historical_data = await self.data.get_historical_data(ticker, days=50)
                    if not historical_data or len(historical_data) < 20:
                        continue

                    # Perform volatility analysis
                    volatility_analysis = await self._perform_volatility_analysis(
                        ticker, historical_data
                    )
                    if not volatility_analysis:
                        continue

                    # Check for lotto conditions
                    lotto_type = self._identify_lotto_type(volatility_analysis)
                    if not lotto_type:
                        continue

                    # Generate lotto signals
                    signal = self._generate_lotto_signal(volatility_analysis, lotto_type)
                    if signal == LottoSignal.HOLD:
                        continue

                    # Create candidate
                    candidate = await self._create_lotto_candidate(
                        ticker, volatility_analysis, lotto_type, signal
                    )
                    if candidate:
                        candidates.append(candidate)

                except Exception as e:
                    self.logger.error(f"Error scanning {ticker}: {e}")
                    continue

            # Sort by lotto score
            candidates.sort(key=lambda x: x.lotto_score, reverse=True)

            self.logger.info(f"Found {len(candidates)} lotto opportunities")
            return candidates

        except Exception as e:
            self.logger.error(f"Error scanning for lotto opportunities: {e}")
            return []

    async def execute_lotto_trade(self, candidate: LottoCandidate) -> LottoPosition | None:
        """Execute lotto trading position"""
        try:
            self.logger.info(f"Executing lotto trade for {candidate.ticker}")

            # Check if we already have a position
            if candidate.ticker in self.active_positions:
                self.logger.warning(f"Already have lotto position for {candidate.ticker}")
                return None

            # Check position limits
            if len(self.active_positions) >= self.max_positions:
                self.logger.warning("Maximum positions reached")
                return None

            # Get lotto options
            if candidate.lotto_type == LottoType.ZERO_DTE:
                lotto_options = await self.options_provider.get_zero_dte_options(candidate.ticker)
            elif candidate.lotto_type == LottoType.EARNINGS_LOTTO:
                # Mock earnings date
                earnings_date = datetime.now() + timedelta(days=3)
                lotto_options = await self.options_provider.get_earnings_options(
                    candidate.ticker, earnings_date
                )
            else:
                lotto_options = await self.options_provider.get_zero_dte_options(candidate.ticker)

            if not lotto_options:
                self.logger.warning(f"No lotto options available for {candidate.ticker}")
                return None

            # Find best option
            best_option = self.options_provider.find_best_lotto_option(
                lotto_options, candidate.lotto_type, candidate.signal, candidate.entry_price
            )
            if not best_option:
                self.logger.warning(f"No suitable lotto option found for {candidate.ticker}")
                return None

            # Calculate max profit and loss
            max_profit = best_option.mid_price * self.take_profit_pct
            max_loss = best_option.mid_price * self.stop_loss_pct

            # Create position
            position = LottoPosition(
                ticker=candidate.ticker,
                lotto_type=candidate.lotto_type,
                signal=candidate.signal,
                option_strategy=best_option.option_type,
                quantity=candidate.position_size,
                entry_price=best_option.mid_price,
                current_price=best_option.mid_price,
                unrealized_pnl=0.0,
                target_price=candidate.target_price,
                stop_loss=candidate.stop_loss,
                entry_date=datetime.now(),
                expiry_date=best_option.expiry_date,
                days_to_expiry=best_option.days_to_expiry,
                max_profit=max_profit,
                max_loss=max_loss,
            )

            self.active_positions[candidate.ticker] = position
            self.logger.info(f"Created lotto position for {candidate.ticker}")

            return position

        except Exception as e:
            self.logger.error(f"Error executing lotto trade for {candidate.ticker}: {e}")
            return None

    async def monitor_lotto_positions(self) -> dict[str, Any]:
        """Monitor active lotto positions"""
        try:
            self.logger.info("Monitoring lotto positions")

            monitoring_results = {
                "positions_monitored": len(self.active_positions),
                "positions_closed": 0,
                "positions_updated": 0,
                "total_pnl": 0.0,
                "risk_alerts": [],
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
            self.logger.error(f"Error monitoring lotto positions: {e}")
            return {"error": str(e)}

    async def _perform_volatility_analysis(
        self, ticker: str, historical_data: list[dict]
    ) -> VolatilityAnalysis | None:
        """Perform volatility analysis on historical data"""
        try:
            if len(historical_data) < 20:
                return None

            # Extract data
            prices = [d["close"] for d in historical_data]
            [d["volume"] for d in historical_data]
            current_price = prices[-1]

            # Calculate historical volatility
            historical_vol = self.volatility_analyzer.calculate_historical_volatility(prices)

            # Mock implied volatility (in production, get from options data)
            implied_vol = historical_vol * 1.2  # Mock IV slightly higher than HV

            # Mock IV percentile and rank
            iv_percentile = 0.75  # Mock high IV percentile
            iv_rank = 0.80  # Mock high IV rank

            # Mock VIX data
            vix_level = 25.0  # Mock VIX level
            vix_percentile = 0.70  # Mock VIX percentile

            # Calculate expected move
            expected_move = self.volatility_analyzer.calculate_expected_move(
                current_price, implied_vol, 1
            )

            # Mock actual move
            actual_move = abs(prices[-1] - prices[-2]) if len(prices) >= 2 else 0.0

            # Mock volatility skew
            volatility_skew = 0.1  # Mock positive skew

            # Mock gamma exposure
            gamma_exposure = 0.05  # Mock gamma exposure

            # Mock options data
            options_volume = 10000  # Mock options volume
            put_call_ratio = 0.8  # Mock put / call ratio

            analysis = VolatilityAnalysis(
                ticker=ticker,
                current_price=current_price,
                implied_volatility=implied_vol,
                historical_volatility=historical_vol,
                iv_percentile=iv_percentile,
                iv_rank=iv_rank,
                vix_level=vix_level,
                vix_percentile=vix_percentile,
                expected_move=expected_move,
                actual_move=actual_move,
                volatility_skew=volatility_skew,
                gamma_exposure=gamma_exposure,
                options_volume=options_volume,
                put_call_ratio=put_call_ratio,
            )

            return analysis

        except Exception as e:
            self.logger.error(f"Error performing volatility analysis for {ticker}: {e}")
            return None

    def _identify_lotto_type(self, volatility_analysis: VolatilityAnalysis) -> LottoType | None:
        """Identify lotto type based on volatility analysis"""
        # Check for 0DTE conditions
        if volatility_analysis.iv_rank > 0.8 and volatility_analysis.gamma_exposure > 0.1:
            return LottoType.ZERO_DTE

        # Check for earnings lotto conditions
        if volatility_analysis.iv_percentile > 0.9 and volatility_analysis.vix_level > 30:
            return LottoType.EARNINGS_LOTTO

        # Check for volatility spike conditions
        if volatility_analysis.iv_rank > 0.7 and volatility_analysis.volatility_skew > 0.2:
            return LottoType.VOLATILITY_SPIKE

        # Check for gamma squeeze conditions
        if volatility_analysis.gamma_exposure > 0.15 and volatility_analysis.options_volume > 50000:
            return LottoType.GAMMA_SQUEEZE

        # Check for meme stock conditions
        if volatility_analysis.put_call_ratio < 0.5 and volatility_analysis.options_volume > 100000:
            return LottoType.MEME_STOCK

        return None

    def _generate_lotto_signal(
        self, volatility_analysis: VolatilityAnalysis, lotto_type: LottoType
    ) -> LottoSignal:
        """Generate lotto signal based on analysis"""
        score = 0.0

        # IV rank component
        score += volatility_analysis.iv_rank * 0.3

        # VIX level component
        if volatility_analysis.vix_level > 30:
            score += 0.2
        elif volatility_analysis.vix_level > 25:
            score += 0.1

        # Gamma exposure component
        if volatility_analysis.gamma_exposure > 0.1:
            score += 0.2
        elif volatility_analysis.gamma_exposure > 0.05:
            score += 0.1

        # Options volume component
        if volatility_analysis.options_volume > 100000:
            score += 0.2
        elif volatility_analysis.options_volume > 50000:
            score += 0.1

        # Volatility skew component
        if volatility_analysis.volatility_skew > 0.2:
            score += 0.1

        if score >= 0.8:
            return LottoSignal.STRONG_BUY
        elif score >= 0.6:
            return LottoSignal.BUY
        elif score <= 0.2:
            return LottoSignal.STRONG_SELL
        elif score <= 0.4:
            return LottoSignal.SELL
        else:
            return LottoSignal.HOLD

    async def _create_lotto_candidate(
        self,
        ticker: str,
        volatility_analysis: VolatilityAnalysis,
        lotto_type: LottoType,
        signal: LottoSignal,
    ) -> LottoCandidate | None:
        """Create lotto trading candidate"""
        try:
            # Calculate entry price
            entry_price = volatility_analysis.current_price

            # Calculate target price and stop loss
            if signal in [LottoSignal.STRONG_BUY, LottoSignal.BUY]:
                target_price = entry_price * (1 + self.take_profit_pct)
                stop_loss = entry_price * (1 - self.stop_loss_pct)
            else:
                target_price = entry_price * (1 - self.take_profit_pct)
                stop_loss = entry_price * (1 + self.stop_loss_pct)

            # Calculate risk / reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(target_price - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0

            # Calculate position size
            position_size = self._calculate_position_size(entry_price, stop_loss)

            # Calculate lotto score
            lotto_score = self._calculate_lotto_score(volatility_analysis, lotto_type)

            # Calculate risk score
            risk_score = self._calculate_risk_score(volatility_analysis, lotto_type)

            candidate = LottoCandidate(
                ticker=ticker,
                lotto_type=lotto_type,
                signal=signal,
                volatility_analysis=volatility_analysis,
                confidence=lotto_score,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                risk_reward_ratio=risk_reward_ratio,
                position_size=position_size,
                lotto_score=lotto_score,
                risk_score=risk_score,
            )

            return candidate

        except Exception as e:
            self.logger.error(f"Error creating lotto candidate for {ticker}: {e}")
            return None

    def _calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        """Calculate position size based on risk"""
        # Simplified position sizing - in production, use proper risk management
        risk_per_share = abs(entry_price - stop_loss)
        max_risk_amount = 200.0  # $200 max risk per lotto play
        position_size = int(max_risk_amount / risk_per_share) if risk_per_share > 0 else 100
        return min(position_size, 200)  # Cap at 200 shares

    def _calculate_lotto_score(
        self, volatility_analysis: VolatilityAnalysis, lotto_type: LottoType
    ) -> float:
        """Calculate lotto score"""
        score = 0.0

        # IV rank component (40% weight)
        score += volatility_analysis.iv_rank * 0.4

        # VIX level component (20% weight)
        if volatility_analysis.vix_level > 30:
            score += 0.2
        elif volatility_analysis.vix_level > 25:
            score += 0.1

        # Gamma exposure component (20% weight)
        if volatility_analysis.gamma_exposure > 0.1:
            score += 0.2
        elif volatility_analysis.gamma_exposure > 0.05:
            score += 0.1

        # Options volume component (20% weight)
        if volatility_analysis.options_volume > 100000:
            score += 0.2
        elif volatility_analysis.options_volume > 50000:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _calculate_risk_score(
        self, volatility_analysis: VolatilityAnalysis, lotto_type: LottoType
    ) -> float:
        """Calculate risk score (higher is riskier)"""
        risk = 0.0

        # High IV risk
        if volatility_analysis.iv_rank > 0.9:
            risk += 0.3

        # High VIX risk
        if volatility_analysis.vix_level > 35:
            risk += 0.2

        # High gamma exposure risk
        if volatility_analysis.gamma_exposure > 0.2:
            risk += 0.2

        # High options volume risk
        if volatility_analysis.options_volume > 200000:
            risk += 0.2

        # Lotto type risk
        if lotto_type == LottoType.ZERO_DTE:
            risk += 0.1

        return max(0.0, min(1.0, risk))

    async def _update_position_data(self, position: LottoPosition):
        """Update position data with current market information"""
        try:
            # Get current market data
            market_data = await self.data.get_market_data(position.ticker)
            if market_data:
                position.current_price = market_data.price
                position.last_update = datetime.now()

                # Update days to expiry
                position.days_to_expiry = (position.expiry_date - datetime.now()).days

                # Recalculate P & L
                position.unrealized_pnl = self._calculate_position_pnl(position)

        except Exception as e:
            self.logger.error(f"Error updating position data for {position.ticker}: {e}")

    def _calculate_position_pnl(self, position: LottoPosition) -> float:
        """Calculate position P & L"""
        # Simplified P & L calculation for lotto options
        price_change = position.current_price - position.entry_price
        return price_change * position.quantity * 100  # Options are per 100 shares

    def _check_exit_conditions(self, position: LottoPosition) -> str | None:
        """Check for exit conditions"""
        # Check stop loss
        if position.current_price <= position.stop_loss:
            return "stop_loss"

        # Check take profit
        if position.current_price >= position.target_price:
            return "take_profit"

        # Check expiry
        if position.days_to_expiry <= 0:
            return "expiry"

        # Check max loss
        if position.unrealized_pnl <= -position.max_loss:
            return "max_loss"

        return None

    def _check_position_risks(self, position: LottoPosition) -> list[str]:
        """Check for position risk alerts"""
        alerts = []

        # Check for large unrealized losses
        if position.unrealized_pnl < -position.max_loss * 0.8:
            alerts.append(
                f"Large unrealized loss for {position.ticker}: ${position.unrealized_pnl:.2f}"
            )

        # Check for approaching expiry
        if position.days_to_expiry <= 0:
            alerts.append(f"Option expired for {position.ticker}")

        # Check for approaching max loss
        if position.unrealized_pnl <= -position.max_loss * 0.9:
            alerts.append(
                f"Approaching max loss for {position.ticker}: ${position.unrealized_pnl:.2f}"
            )

        return alerts

    async def _close_position(self, ticker: str, exit_signal: str):
        """Close lotto position"""
        try:
            if ticker in self.active_positions:
                position = self.active_positions.pop(ticker)
                self.logger.info(
                    f"Closed lotto position for {ticker}: P & L ${position.unrealized_pnl:.2f}, Signal: {exit_signal}"
                )

        except Exception as e:
            self.logger.error(f"Error closing position for {ticker}: {e}")

    async def get_strategy_status(self) -> dict[str, Any]:
        """Get current strategy status"""
        try:
            total_pnl = sum(pos.unrealized_pnl for pos in self.active_positions.values())
            total_exposure = sum(
                pos.quantity * pos.current_price for pos in self.active_positions.values()
            )

            return {
                "active_positions": len(self.active_positions),
                "total_pnl": total_pnl,
                "total_exposure": total_exposure,
                "max_positions": self.max_positions,
                "positions": [
                    {
                        "ticker": pos.ticker,
                        "lotto_type": pos.lotto_type.value,
                        "signal": pos.signal.value,
                        "option_strategy": pos.option_strategy.value,
                        "days_to_expiry": pos.days_to_expiry,
                        "unrealized_pnl": pos.unrealized_pnl,
                        "max_profit": pos.max_profit,
                        "max_loss": pos.max_loss,
                    }
                    for pos in self.active_positions.values()
                ],
            }

        except Exception as e:
            self.logger.error(f"Error getting strategy status: {e}")
            return {"error": str(e)}
