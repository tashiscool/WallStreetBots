"""
Maker/Taker Fee Model & Fee-Aware Execution Optimizer.

Models exchange/broker fee structures and recommends order types
(limit vs market) based on urgency, expected fill probability,
and net fee impact.

Pre-configured schedules for:
- Alpaca (commission-free equities, standard crypto fees)
- Interactive Brokers (tiered and fixed pricing)
- Configurable custom schedules for any venue

References:
- Angel, Harris & Spatt (2015) - Equity Trading in the 21st Century
- Foucault, Kadan & Kandel (2013) - Liquidity Cycles and Make/Take Fees
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AssetClass(Enum):
    """Asset classes for fee differentiation."""
    EQUITY = "equity"
    OPTION = "option"
    CRYPTO = "crypto"
    FUTURE = "future"


class FeeType(Enum):
    """Fee calculation method."""
    PER_SHARE = "per_share"
    PER_CONTRACT = "per_contract"
    PERCENTAGE = "percentage"
    FLAT = "flat"
    ZERO = "zero"


@dataclass(frozen=True)
class FeeSchedule:
    """
    Fee schedule for a single venue/tier combination.

    Supports maker/taker differentiation, per-share and percentage fees,
    minimums and maximums, and regulatory fees.
    """
    venue: str
    tier: str = "default"
    asset_class: AssetClass = AssetClass.EQUITY

    # Maker fees (limit orders that add liquidity)
    maker_fee_type: FeeType = FeeType.PER_SHARE
    maker_fee: float = 0.0  # Per unit or percentage
    maker_min: float = 0.0
    maker_max: float = float('inf')

    # Taker fees (market orders that remove liquidity)
    taker_fee_type: FeeType = FeeType.PER_SHARE
    taker_fee: float = 0.0
    taker_min: float = 0.0
    taker_max: float = float('inf')

    # Regulatory fees (SEC, FINRA TAF, exchange)
    sec_fee_per_dollar: float = 0.0  # SEC fee on sells
    taf_fee_per_share: float = 0.0  # FINRA TAF
    exchange_fee_per_share: float = 0.0

    # Platform/clearing fees
    platform_fee: float = 0.0  # Flat per-trade fee
    clearing_fee_per_share: float = 0.0


# Pre-configured fee schedules
ALPACA_EQUITY = FeeSchedule(
    venue="alpaca",
    tier="default",
    asset_class=AssetClass.EQUITY,
    maker_fee_type=FeeType.ZERO,
    maker_fee=0.0,
    taker_fee_type=FeeType.ZERO,
    taker_fee=0.0,
    # Alpaca passes through regulatory fees
    sec_fee_per_dollar=0.000008,  # $8 per $1M (SEC fee rate, updated periodically)
    taf_fee_per_share=0.000119,  # $0.000119 per share (FINRA TAF)
)

ALPACA_CRYPTO = FeeSchedule(
    venue="alpaca",
    tier="default",
    asset_class=AssetClass.CRYPTO,
    maker_fee_type=FeeType.PERCENTAGE,
    maker_fee=0.0015,  # 15 bps
    taker_fee_type=FeeType.PERCENTAGE,
    taker_fee=0.0025,  # 25 bps
)

IBKR_TIERED_EQUITY = FeeSchedule(
    venue="ibkr",
    tier="tiered",
    asset_class=AssetClass.EQUITY,
    maker_fee_type=FeeType.PER_SHARE,
    maker_fee=-0.0020,  # Rebate of $0.002/share
    maker_min=0.0,
    taker_fee_type=FeeType.PER_SHARE,
    taker_fee=0.0035,  # $0.0035/share
    taker_min=0.35,  # $0.35 minimum per order
    taker_max=0.005,  # 0.5% of trade value (applied as pct, capped separately)
    sec_fee_per_dollar=0.000008,
    taf_fee_per_share=0.000119,
    exchange_fee_per_share=0.003,  # Varies by exchange
    clearing_fee_per_share=0.0002,
)

IBKR_FIXED_EQUITY = FeeSchedule(
    venue="ibkr",
    tier="fixed",
    asset_class=AssetClass.EQUITY,
    maker_fee_type=FeeType.PER_SHARE,
    maker_fee=0.005,  # $0.005/share
    maker_min=1.0,  # $1.00 minimum
    taker_fee_type=FeeType.PER_SHARE,
    taker_fee=0.005,
    taker_min=1.0,
    sec_fee_per_dollar=0.000008,
    taf_fee_per_share=0.000119,
)

IBKR_OPTIONS = FeeSchedule(
    venue="ibkr",
    tier="tiered",
    asset_class=AssetClass.OPTION,
    maker_fee_type=FeeType.PER_CONTRACT,
    maker_fee=0.25,  # $0.25/contract
    maker_min=1.0,
    taker_fee_type=FeeType.PER_CONTRACT,
    taker_fee=0.65,  # $0.65/contract
    taker_min=1.0,
    exchange_fee_per_share=0.30,  # Per-contract exchange fee
    clearing_fee_per_share=0.02,  # OCC clearing
)

# Registry of known schedules
KNOWN_SCHEDULES: Dict[str, FeeSchedule] = {
    "alpaca_equity": ALPACA_EQUITY,
    "alpaca_crypto": ALPACA_CRYPTO,
    "ibkr_tiered_equity": IBKR_TIERED_EQUITY,
    "ibkr_fixed_equity": IBKR_FIXED_EQUITY,
    "ibkr_options": IBKR_OPTIONS,
}


@dataclass
class FeeEstimate:
    """Estimated fee for an order."""
    commission: float  # Commission/exchange fees
    regulatory: float  # SEC + FINRA TAF fees
    total: float  # Total all-in cost
    total_bps: float  # Total as basis points of notional
    is_maker: bool
    breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class OrderTypeRecommendation:
    """Recommendation for limit vs market order."""
    recommended_type: str  # 'limit' or 'market'
    maker_fee_estimate: FeeEstimate
    taker_fee_estimate: FeeEstimate
    fee_savings_bps: float  # Savings from using limit vs market
    fill_probability: float  # Estimated fill probability for limit
    expected_cost_limit: float  # Expected cost (fee + miss risk)
    expected_cost_market: float  # Expected cost (fee + slippage)
    reasoning: str


class FeeModel:
    """
    Fee calculator supporting maker/taker economics.

    Calculates all-in execution costs including commissions,
    exchange fees, regulatory fees, and platform fees.

    Example:
        model = FeeModel(schedule=ALPACA_EQUITY)
        fee = model.calculate_fee(
            quantity=100, price=150.0, side='buy', is_maker=True
        )
        print(f"Total cost: ${fee.total:.4f} ({fee.total_bps:.2f} bps)")
    """

    def __init__(self, schedule: Optional[FeeSchedule] = None):
        """
        Args:
            schedule: Fee schedule to use (defaults to Alpaca equity)
        """
        self.schedule = schedule or ALPACA_EQUITY

    def calculate_fee(
        self,
        quantity: float,
        price: float,
        side: str,
        is_maker: bool = False,
    ) -> FeeEstimate:
        """
        Calculate all-in fee for an order.

        Args:
            quantity: Number of shares/contracts
            price: Execution price
            side: 'buy' or 'sell'
            is_maker: Whether this is a maker (limit) or taker (market) fill

        Returns:
            FeeEstimate with full breakdown
        """
        notional = quantity * price
        breakdown = {}

        # Commission
        if is_maker:
            commission = self._calc_fee_component(
                self.schedule.maker_fee_type,
                self.schedule.maker_fee,
                quantity,
                notional,
                self.schedule.maker_min,
                self.schedule.maker_max,
            )
        else:
            commission = self._calc_fee_component(
                self.schedule.taker_fee_type,
                self.schedule.taker_fee,
                quantity,
                notional,
                self.schedule.taker_min,
                self.schedule.taker_max,
            )
        breakdown['commission'] = commission

        # Regulatory fees
        regulatory = 0.0

        # SEC fee (sells only)
        if side == 'sell' and self.schedule.sec_fee_per_dollar > 0:
            sec_fee = notional * self.schedule.sec_fee_per_dollar
            regulatory += sec_fee
            breakdown['sec_fee'] = sec_fee

        # FINRA TAF
        if self.schedule.taf_fee_per_share > 0:
            taf = quantity * self.schedule.taf_fee_per_share
            regulatory += taf
            breakdown['taf_fee'] = taf

        # Exchange fee
        if self.schedule.exchange_fee_per_share > 0:
            exch = quantity * self.schedule.exchange_fee_per_share
            breakdown['exchange_fee'] = exch
            commission += exch

        # Clearing fee
        if self.schedule.clearing_fee_per_share > 0:
            clearing = quantity * self.schedule.clearing_fee_per_share
            breakdown['clearing_fee'] = clearing
            commission += clearing

        # Platform fee
        if self.schedule.platform_fee > 0:
            breakdown['platform_fee'] = self.schedule.platform_fee
            commission += self.schedule.platform_fee

        total = commission + regulatory
        total_bps = (total / notional * 10000.0) if notional > 0 else 0.0

        return FeeEstimate(
            commission=commission,
            regulatory=regulatory,
            total=total,
            total_bps=total_bps,
            is_maker=is_maker,
            breakdown=breakdown,
        )

    def _calc_fee_component(
        self,
        fee_type: FeeType,
        fee_rate: float,
        quantity: float,
        notional: float,
        fee_min: float,
        fee_max: float,
    ) -> float:
        """Calculate a single fee component."""
        if fee_type == FeeType.ZERO:
            return 0.0
        elif fee_type == FeeType.PER_SHARE or fee_type == FeeType.PER_CONTRACT:
            raw = abs(fee_rate) * quantity
            if fee_rate < 0:
                return -raw  # Rebate
            return max(fee_min, min(raw, fee_max if fee_max < float('inf') else raw))
        elif fee_type == FeeType.PERCENTAGE:
            raw = abs(fee_rate) * notional
            if fee_rate < 0:
                return -raw
            return max(fee_min, min(raw, fee_max if fee_max < float('inf') else raw))
        elif fee_type == FeeType.FLAT:
            return fee_rate
        return 0.0

    def compare_maker_taker(
        self,
        quantity: float,
        price: float,
        side: str,
    ) -> Dict[str, FeeEstimate]:
        """Compare maker vs taker fees for the same order."""
        return {
            'maker': self.calculate_fee(quantity, price, side, is_maker=True),
            'taker': self.calculate_fee(quantity, price, side, is_maker=False),
        }


class FeeOptimizer:
    """
    Recommends limit vs market orders based on fee savings,
    fill probability, and urgency.

    Balances three factors:
    1. Fee savings (maker rebates/lower fees)
    2. Fill probability (limit orders may not fill)
    3. Urgency (time-sensitive trades need market orders)

    Example:
        optimizer = FeeOptimizer(fee_model=FeeModel(IBKR_TIERED_EQUITY))
        rec = optimizer.recommend(
            quantity=1000, price=150.0, side='buy',
            urgency=0.3, spread_bps=2.0
        )
        print(rec.recommended_type, rec.reasoning)
    """

    def __init__(
        self,
        fee_model: Optional[FeeModel] = None,
        default_fill_probability: float = 0.65,
    ):
        """
        Args:
            fee_model: Fee calculator to use
            default_fill_probability: Base fill probability for limit orders
        """
        self.fee_model = fee_model or FeeModel()
        self.default_fill_probability = default_fill_probability

    def recommend(
        self,
        quantity: float,
        price: float,
        side: str,
        urgency: float = 0.5,
        spread_bps: float = 5.0,
        volatility: float = 0.02,
        expected_slippage_bps: float = 2.0,
    ) -> OrderTypeRecommendation:
        """
        Recommend limit vs market order.

        Args:
            quantity: Order size (shares)
            price: Current price
            side: 'buy' or 'sell'
            urgency: 0.0 (patient) to 1.0 (must fill now)
            spread_bps: Current bid-ask spread in bps
            volatility: Recent realized volatility
            expected_slippage_bps: Expected market order slippage

        Returns:
            OrderTypeRecommendation
        """
        maker_fee = self.fee_model.calculate_fee(quantity, price, side, is_maker=True)
        taker_fee = self.fee_model.calculate_fee(quantity, price, side, is_maker=False)

        fee_savings_bps = taker_fee.total_bps - maker_fee.total_bps

        # Estimate fill probability for limit order
        # Higher spread = lower fill probability (wider market)
        # Higher volatility = higher fill probability (more price movement)
        fill_prob = self._estimate_fill_probability(spread_bps, volatility, urgency)

        # Expected cost of limit order:
        # If filled: maker fee
        # If not filled: opportunity cost (estimated as slippage of delayed market order)
        opportunity_cost_bps = expected_slippage_bps * 1.5  # Delay makes it worse
        expected_limit_cost = (
            fill_prob * maker_fee.total_bps +
            (1 - fill_prob) * (taker_fee.total_bps + opportunity_cost_bps)
        )

        # Expected cost of market order:
        # Always fills, pays taker fee + slippage
        expected_market_cost = taker_fee.total_bps + expected_slippage_bps

        # Decision
        if urgency >= 0.9:
            recommended = 'market'
            reasoning = f"High urgency ({urgency:.1f}) requires immediate fill"
        elif urgency <= 0.1 and fee_savings_bps > 0.5:
            recommended = 'limit'
            reasoning = (
                f"Low urgency ({urgency:.1f}), "
                f"saves {fee_savings_bps:.1f}bps in fees"
            )
        elif expected_limit_cost < expected_market_cost:
            recommended = 'limit'
            net_savings = expected_market_cost - expected_limit_cost
            reasoning = (
                f"Limit saves {net_savings:.1f}bps expected cost "
                f"(fill prob {fill_prob:.0%}, fee savings {fee_savings_bps:.1f}bps)"
            )
        else:
            recommended = 'market'
            reasoning = (
                f"Market preferred: expected cost {expected_market_cost:.1f}bps "
                f"vs limit {expected_limit_cost:.1f}bps "
                f"(fill prob only {fill_prob:.0%})"
            )

        return OrderTypeRecommendation(
            recommended_type=recommended,
            maker_fee_estimate=maker_fee,
            taker_fee_estimate=taker_fee,
            fee_savings_bps=fee_savings_bps,
            fill_probability=fill_prob,
            expected_cost_limit=expected_limit_cost,
            expected_cost_market=expected_market_cost,
            reasoning=reasoning,
        )

    def _estimate_fill_probability(
        self,
        spread_bps: float,
        volatility: float,
        urgency: float,
    ) -> float:
        """
        Estimate limit order fill probability.

        Based on empirical factors:
        - Tight spreads -> higher fill probability
        - Higher volatility -> higher fill probability (more price movement)
        - Higher urgency -> we set tighter limits -> lower fill probability
        """
        base = self.default_fill_probability

        # Spread adjustment: tight spread = better fills
        # Calibrated: 1bps spread -> +10%, 10bps -> -10%
        spread_adj = max(-0.2, min(0.2, (5.0 - spread_bps) * 0.02))

        # Volatility adjustment: more vol = more price crossing
        vol_adj = max(-0.1, min(0.15, (volatility - 0.02) * 5.0))

        # Urgency adjustment: higher urgency = tighter limit = lower fill prob
        urgency_adj = -0.3 * urgency

        fill_prob = base + spread_adj + vol_adj + urgency_adj
        return max(0.05, min(0.95, fill_prob))

    def batch_recommend(
        self,
        orders: List[Dict[str, Any]],
    ) -> List[OrderTypeRecommendation]:
        """
        Recommend order types for a batch of orders.

        Args:
            orders: List of dicts with keys: quantity, price, side,
                    and optional: urgency, spread_bps, volatility

        Returns:
            List of recommendations
        """
        return [
            self.recommend(
                quantity=o['quantity'],
                price=o['price'],
                side=o['side'],
                urgency=o.get('urgency', 0.5),
                spread_bps=o.get('spread_bps', 5.0),
                volatility=o.get('volatility', 0.02),
                expected_slippage_bps=o.get('expected_slippage_bps', 2.0),
            )
            for o in orders
        ]
