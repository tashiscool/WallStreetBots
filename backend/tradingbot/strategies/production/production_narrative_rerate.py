"""Production Narrative Rerate Strategy.

Finds high-conviction narrative trades early, sizes them aggressively but
survivably, and stores the thesis, catalyst, invalidation, and trim plan.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from ...core.trading_interface import OrderSide, OrderType, TradeStatus
from ...production.core.production_integration import ProductionTradeSignal
from ...production.data.production_data_integration import ReliableDataProvider
from ...validation.strategy_signal_integration import StrategySignalMixin, signal_integrator


@dataclass
class NarrativeTheme:
    """Investable market story with explicit ticker and keyword anchors."""

    name: str
    description: str
    tickers: list[str]
    keywords: list[str]
    max_theme_exposure: float = 0.30


@dataclass
class TrimPlan:
    """Predefined profit-taking ladder."""

    first_trim_gain: float = 0.25
    second_trim_gain: float = 0.50
    home_run_trim_gain: float = 1.00
    trim_fraction: float = 0.25

    def as_dict(self) -> dict[str, float]:
        return {
            "first_trim_gain": self.first_trim_gain,
            "second_trim_gain": self.second_trim_gain,
            "home_run_trim_gain": self.home_run_trim_gain,
            "trim_fraction": self.trim_fraction,
        }


@dataclass
class NarrativeRerateSignal:
    """Signal with both numeric score and written trade plan."""

    ticker: str
    theme: str
    current_price: Decimal
    composite_score: float
    narrative_score: float
    momentum_score: float
    rebound_score: float
    sentiment_score: float
    liquidity_score: float
    target_weight: float
    thesis: str
    catalyst: str
    invalidation: str
    invalidation_price: Decimal
    max_loss_pct: float
    trim_plan: TrimPlan
    risk_flags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class ProductionNarrativeRerateStrategy(StrategySignalMixin):
    """Narrative + momentum + risk-controlled concentration strategy.

    Strategy Logic:
    1. Score symbols against emerging themes such as AI, crypto treasury,
       post-panic software, cybersecurity, and quantum.
    2. Require a catalyst/sentiment pulse, technical confirmation, and liquidity.
    3. Generate a written bull case, bear/invalidation case, and trim ladder.
    4. Size concentrated equity positions within portfolio and single-name caps.

    Risk Management:
    - Default 15% maximum single-name target weight
    - Default 50% total strategy exposure
    - No margin by default
    - Hard invalidation price and max-loss metadata on every signal
    - Risk flags reduce position size before order construction
    """

    def __init__(
        self,
        integration_manager,
        data_provider: ReliableDataProvider,
        config: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.strategy_name = "narrative_rerate"
        self.integration_manager = integration_manager
        self.data_provider = data_provider
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        signal_integrator.enhance_strategy_with_validation(self, self.strategy_name)

        self.universe = self.config.get(
            "universe",
            [
                "NVDA",
                "AMD",
                "COIN",
                "CRWD",
                "NET",
                "SHOP",
                "CRM",
                "SNOW",
                "FIG",
                "SBET",
                "PLTR",
                "QBTS",
                "IONQ",
            ],
        )
        self.themes = self._build_themes(self.config.get("themes"))

        self.min_composite_score = self.config.get("min_composite_score", 70.0)
        self.max_positions = self.config.get("max_positions", 5)
        self.max_position_size = self.config.get("max_position_size", 0.15)
        self.max_total_exposure = self.config.get("max_total_exposure", 0.50)
        self.allow_margin = self.config.get("allow_margin", False)
        self.max_gross_exposure = self.config.get(
            "max_gross_exposure", 1.0 if not self.allow_margin else 1.25
        )
        self.min_avg_dollar_volume = Decimal(
            str(self.config.get("min_avg_dollar_volume", 5_000_000))
        )
        self.default_max_loss_pct = self.config.get("max_loss_pct", 0.18)
        self.trim_plan = TrimPlan(**self.config.get("trim_plan", {}))

        self.active_positions: dict[str, dict[str, Any]] = {}

    def _build_themes(self, configured: dict[str, Any] | None) -> dict[str, NarrativeTheme]:
        if configured:
            return {
                key: value
                if isinstance(value, NarrativeTheme)
                else NarrativeTheme(name=value["name"], description=value["description"],
                                    tickers=value.get("tickers", []),
                                    keywords=value.get("keywords", []),
                                    max_theme_exposure=value.get("max_theme_exposure", 0.30))
                for key, value in configured.items()
            }

        return {
            "ai_infrastructure": NarrativeTheme(
                name="AI Infrastructure",
                description="Compute, data, and software platforms leveraged to AI adoption",
                tickers=["NVDA", "AMD", "PLTR", "SNOW", "CRM", "NET"],
                keywords=["ai", "gpu", "compute", "inference", "agent", "datacenter"],
            ),
            "crypto_treasury": NarrativeTheme(
                name="Crypto Treasury",
                description="Public-market wrappers for crypto balance-sheet narratives",
                tickers=["COIN", "SBET", "MSTR"],
                keywords=["ethereum", "bitcoin", "treasury", "staking", "eth", "btc"],
            ),
            "post_panic_software": NarrativeTheme(
                name="Post-Panic Software Rebound",
                description="High-quality software recovering after overdone drawdowns",
                tickers=["FIG", "SHOP", "NET", "CRWD", "SNOW", "CRM"],
                keywords=["guidance", "ai product", "retention", "revenue", "margin"],
            ),
            "cybersecurity": NarrativeTheme(
                name="Cybersecurity",
                description="Security platforms with durable enterprise urgency",
                tickers=["CRWD", "NET", "ZS", "PANW", "OKTA"],
                keywords=["breach", "security", "zero trust", "cloudflare", "endpoint"],
            ),
            "quantum": NarrativeTheme(
                name="Quantum Speculation",
                description="Speculative quantum computing rerate windows",
                tickers=["IONQ", "QBTS", "RGTI"],
                keywords=["quantum", "qubit", "annealing", "error correction"],
            ),
        }

    async def scan_narrative_opportunities(self) -> list[NarrativeRerateSignal]:
        """Scan the configured universe and return ranked qualifying signals."""
        if hasattr(self.data_provider, "is_market_open"):
            try:
                if not await self.data_provider.is_market_open():
                    return []
            except Exception:
                self.logger.warning("Market-open check failed; continuing scan")

        signals: list[NarrativeRerateSignal] = []
        for ticker in self.universe:
            try:
                signal = await self.evaluate_ticker(ticker)
                if signal and signal.composite_score >= self.min_composite_score:
                    signals.append(signal)
            except Exception as exc:
                self.logger.error("Error evaluating %s: %s", ticker, exc)

        signals.sort(key=lambda s: s.composite_score, reverse=True)
        return signals[: self.max_positions]

    async def evaluate_ticker(self, ticker: str) -> NarrativeRerateSignal | None:
        """Build a thesis-ready signal for one ticker when the setup qualifies."""
        prices = await self._get_price_history(ticker, days=90)
        volumes = await self._get_volume_history(ticker, days=30)
        if len(prices) < 30 or len(volumes) < 10:
            return None

        current_price = Decimal(str(prices[-1]))
        if current_price <= 0:
            return None

        theme, theme_score = self._best_theme_for(ticker)
        news_items = await self._get_news_items(ticker)
        sentiment_score, mention_count, matched_keywords = self._score_sentiment(
            news_items, theme
        )
        momentum_score = self._score_momentum(prices)
        rebound_score = self._score_rebound(prices)
        liquidity_score = self._score_liquidity(prices, volumes)
        risk_flags = self._detect_risk_flags(news_items, prices, liquidity_score)

        narrative_score = min(100.0, theme_score + sentiment_score * 0.55 + mention_count * 3)
        composite = (
            narrative_score * 0.35
            + momentum_score * 0.25
            + rebound_score * 0.15
            + sentiment_score * 0.15
            + liquidity_score * 0.10
        )

        for _flag in risk_flags:
            composite -= 5.0
        composite = max(0.0, min(100.0, composite))

        target_weight = self._calculate_target_weight(composite, risk_flags)
        invalidation_price = self._calculate_invalidation_price(prices, current_price)
        catalyst = self._build_catalyst(theme, matched_keywords, mention_count)

        return NarrativeRerateSignal(
            ticker=ticker,
            theme=theme.name,
            current_price=current_price,
            composite_score=round(composite, 2),
            narrative_score=round(narrative_score, 2),
            momentum_score=round(momentum_score, 2),
            rebound_score=round(rebound_score, 2),
            sentiment_score=round(sentiment_score, 2),
            liquidity_score=round(liquidity_score, 2),
            target_weight=target_weight,
            thesis=self._build_thesis(ticker, theme, matched_keywords),
            catalyst=catalyst,
            invalidation=self._build_invalidation(ticker, invalidation_price),
            invalidation_price=invalidation_price,
            max_loss_pct=self.default_max_loss_pct,
            trim_plan=self.trim_plan,
            risk_flags=risk_flags,
            metadata={
                "mention_count": mention_count,
                "matched_keywords": matched_keywords,
                "avg_dollar_volume": float(self._avg_dollar_volume(prices, volumes)),
                "theme_key": self._theme_key(theme),
            },
        )

    async def execute_signal(self, signal: NarrativeRerateSignal) -> bool:
        """Execute a qualified narrative signal as an equity position."""
        if signal.ticker in self.active_positions:
            return False
        if len(self.active_positions) >= self.max_positions:
            return False

        account_info = await self.integration_manager.get_account_info()
        equity = Decimal(str(account_info.get("equity", account_info.get("cash", 0))))
        if equity <= 0:
            return False

        current_strategy_exposure = sum(
            Decimal(str(pos.get("market_value", 0))) for pos in self.active_positions.values()
        )
        max_strategy_value = equity * Decimal(str(self.max_total_exposure))
        remaining_value = max(Decimal("0"), max_strategy_value - current_strategy_exposure)
        target_value = min(
            equity * Decimal(str(signal.target_weight)),
            remaining_value,
            equity * Decimal(str(self.max_gross_exposure)),
        )
        quantity = int(target_value / signal.current_price)
        if quantity <= 0:
            return False

        trade_signal = ProductionTradeSignal(
            strategy_name=self.strategy_name,
            ticker=signal.ticker,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=float(signal.current_price),
            trade_type="stock",
            risk_amount=target_value * Decimal(str(signal.max_loss_pct)),
            expected_return=target_value * Decimal(str(signal.trim_plan.second_trim_gain)),
            confidence=signal.composite_score / 100.0,
            signal_strength=signal.composite_score,
            metadata={
                "theme": signal.theme,
                "thesis": signal.thesis,
                "catalyst": signal.catalyst,
                "invalidation": signal.invalidation,
                "invalidation_price": float(signal.invalidation_price),
                "trim_plan": signal.trim_plan.as_dict(),
                "risk_flags": signal.risk_flags,
                "scores": {
                    "composite": signal.composite_score,
                    "narrative": signal.narrative_score,
                    "momentum": signal.momentum_score,
                    "rebound": signal.rebound_score,
                    "sentiment": signal.sentiment_score,
                    "liquidity": signal.liquidity_score,
                },
            },
        )

        result = await self.integration_manager.execute_trade(trade_signal)
        status = getattr(result, "status", None)
        status_value = getattr(status, "value", status)
        if status_value == TradeStatus.FILLED.value or status == TradeStatus.FILLED:
            self.active_positions[signal.ticker] = {
                "signal": signal,
                "quantity": quantity,
                "entry_price": signal.current_price,
                "market_value": quantity * signal.current_price,
                "trimmed_levels": set(),
            }
            return True
        return False

    async def should_exit_position(self, position: dict[str, Any]) -> dict[str, Any]:
        """Evaluate invalidation, stop, and trim conditions for a live position."""
        signal: NarrativeRerateSignal = position["signal"]
        current_price = await self._get_current_price(signal.ticker)
        entry_price = Decimal(str(position["entry_price"]))
        if entry_price <= 0:
            return {"action": "hold", "reason": "bad_entry_price"}

        gain = (current_price - entry_price) / entry_price
        if current_price <= signal.invalidation_price:
            return {"action": "exit", "reason": "INVALIDATION_PRICE", "gain": float(gain)}
        if gain <= Decimal(str(-signal.max_loss_pct)):
            return {"action": "exit", "reason": "MAX_LOSS", "gain": float(gain)}

        trimmed = position.setdefault("trimmed_levels", set())
        for level_name, threshold in [
            ("first_trim", signal.trim_plan.first_trim_gain),
            ("second_trim", signal.trim_plan.second_trim_gain),
            ("home_run_trim", signal.trim_plan.home_run_trim_gain),
        ]:
            if level_name not in trimmed and gain >= Decimal(str(threshold)):
                return {
                    "action": "trim",
                    "reason": level_name.upper(),
                    "trim_fraction": signal.trim_plan.trim_fraction,
                    "gain": float(gain),
                }

        return {"action": "hold", "reason": "THESIS_INTACT", "gain": float(gain)}

    def _best_theme_for(self, ticker: str) -> tuple[NarrativeTheme, float]:
        fallback = next(iter(self.themes.values()))
        best_theme = fallback
        best_score = 15.0
        for theme in self.themes.values():
            score = 55.0 if ticker in theme.tickers else 15.0
            if score > best_score:
                best_theme = theme
                best_score = score
        return best_theme, best_score

    def _theme_key(self, theme: NarrativeTheme) -> str:
        for key, candidate in self.themes.items():
            if candidate is theme:
                return key
        return theme.name.lower().replace(" ", "_")

    async def _get_price_history(self, ticker: str, days: int) -> list[Decimal]:
        values = await self.data_provider.get_price_history(ticker, days=days)
        return [Decimal(str(v)) for v in values]

    async def _get_volume_history(self, ticker: str, days: int) -> list[int]:
        values = await self.data_provider.get_volume_history(ticker, days=days)
        return [int(v) for v in values]

    async def _get_current_price(self, ticker: str) -> Decimal:
        quote = await self.data_provider.get_current_price(ticker)
        return Decimal(str(getattr(quote, "price", quote)))

    async def _get_news_items(self, ticker: str) -> list[Any]:
        for method_name in ("get_news_items", "get_sentiment_records", "get_recent_news"):
            method = getattr(self.data_provider, method_name, None)
            if method is not None:
                try:
                    return list(await method(ticker))
                except Exception:
                    self.logger.debug("%s failed for %s", method_name, ticker)
        return []

    def _score_sentiment(
        self, news_items: list[Any], theme: NarrativeTheme
    ) -> tuple[float, int, list[str]]:
        if not news_items:
            return 20.0, 0, []

        scores: list[float] = []
        matched: set[str] = set()
        for item in news_items:
            text = self._news_text(item).lower()
            raw_score = getattr(item, "sentiment_score", None)
            if raw_score is None and isinstance(item, dict):
                raw_score = item.get("sentiment_score", item.get("score"))
            if raw_score is not None:
                scores.append(float(raw_score))
            for keyword in theme.keywords:
                if keyword.lower() in text:
                    matched.add(keyword)

        avg_sentiment = sum(scores) / len(scores) if scores else 0.0
        sentiment_component = max(0.0, min(70.0, (avg_sentiment + 1.0) * 35.0))
        keyword_component = min(30.0, len(matched) * 7.5)
        return sentiment_component + keyword_component, len(news_items), sorted(matched)

    def _score_momentum(self, prices: list[Decimal]) -> float:
        current = prices[-1]
        price_20 = prices[-21] if len(prices) > 21 else prices[0]
        sma_50 = sum(prices[-50:]) / Decimal(str(min(50, len(prices))))
        return_20 = (current - price_20) / price_20 if price_20 else Decimal("0")
        score = 45.0 + float(return_20) * 180.0
        if current > sma_50:
            score += 15.0
        return max(0.0, min(100.0, score))

    def _score_rebound(self, prices: list[Decimal]) -> float:
        current = prices[-1]
        high = max(prices)
        low = min(prices[-45:])
        drawdown_from_high = (high - current) / high if high else Decimal("0")
        bounce_from_low = (current - low) / low if low else Decimal("0")
        score = 30.0 + float(drawdown_from_high) * 80.0 + float(bounce_from_low) * 120.0
        return max(0.0, min(100.0, score))

    def _score_liquidity(self, prices: list[Decimal], volumes: list[int]) -> float:
        avg_dollar_volume = self._avg_dollar_volume(prices, volumes)
        if avg_dollar_volume >= self.min_avg_dollar_volume * Decimal("10"):
            return 100.0
        if avg_dollar_volume >= self.min_avg_dollar_volume:
            return 70.0 + float(
                (avg_dollar_volume - self.min_avg_dollar_volume)
                / (self.min_avg_dollar_volume * Decimal("9"))
            ) * 30.0
        return max(0.0, float(avg_dollar_volume / self.min_avg_dollar_volume) * 70.0)

    def _avg_dollar_volume(self, prices: list[Decimal], volumes: list[int]) -> Decimal:
        paired = list(zip(prices[-len(volumes):], volumes))
        if not paired:
            return Decimal("0")
        return sum(price * Decimal(str(volume)) for price, volume in paired) / Decimal(
            str(len(paired))
        )

    def _detect_risk_flags(
        self, news_items: list[Any], prices: list[Decimal], liquidity_score: float
    ) -> list[str]:
        text = " ".join(self._news_text(item).lower() for item in news_items)
        flags: list[str] = []
        if any(term in text for term in ["resale", "pipe", "atm offering", "dilution"]):
            flags.append("dilution_or_resale_risk")
        if liquidity_score < 70:
            flags.append("liquidity_risk")
        current = prices[-1]
        high = max(prices)
        if high and current > high * Decimal("0.95"):
            flags.append("near_recent_high")
        if len(prices) > 5 and prices[-1] < prices[-5]:
            flags.append("short_term_momentum_fading")
        return flags

    def _calculate_target_weight(self, composite: float, risk_flags: list[str]) -> float:
        confidence_scale = max(0.0, min(1.0, (composite - self.min_composite_score) / 30.0))
        target = self.max_position_size * (0.50 + 0.50 * confidence_scale)
        for _flag in risk_flags:
            target *= 0.85
        return round(min(self.max_position_size, max(0.02, target)), 4)

    def _calculate_invalidation_price(
        self, prices: list[Decimal], current_price: Decimal
    ) -> Decimal:
        recent_low = min(prices[-20:])
        max_loss_price = current_price * Decimal(str(1 - self.default_max_loss_pct))
        invalidation = max(recent_low * Decimal("0.98"), max_loss_price)
        return invalidation.quantize(Decimal("0.01"))

    def _build_thesis(
        self, ticker: str, theme: NarrativeTheme, matched_keywords: list[str]
    ) -> str:
        keyword_text = ", ".join(matched_keywords[:4]) or "theme alignment"
        return (
            f"{ticker} can rerate if the market rewards the {theme.name} narrative; "
            f"current evidence centers on {keyword_text}."
        )

    def _build_catalyst(
        self, theme: NarrativeTheme, matched_keywords: list[str], mention_count: int
    ) -> str:
        if matched_keywords:
            return (
                f"{theme.name} catalyst pulse detected across {mention_count} items: "
                f"{', '.join(matched_keywords[:5])}."
            )
        return f"{theme.name} watchlist candidate with limited confirmed catalyst flow."

    def _build_invalidation(self, ticker: str, invalidation_price: Decimal) -> str:
        return (
            f"Exit {ticker} if price closes below {invalidation_price} or if the "
            "core catalyst reverses through dilution, guidance cuts, or narrative decay."
        )

    def _news_text(self, item: Any) -> str:
        if isinstance(item, dict):
            return " ".join(str(item.get(key, "")) for key in ("title", "summary", "body"))
        return " ".join(
            str(getattr(item, key, "")) for key in ("title", "summary", "body")
        )


def create_production_narrative_rerate_strategy(
    integration_manager,
    data_provider: ReliableDataProvider,
    config: dict[str, Any] | None = None,
) -> ProductionNarrativeRerateStrategy:
    """Factory function to create the production narrative rerate strategy."""
    return ProductionNarrativeRerateStrategy(integration_manager, data_provider, config)
