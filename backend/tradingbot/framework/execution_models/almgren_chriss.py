"""Almgren-Chriss (2000) Optimal Execution with Market Impact.

Computes an optimal liquidation trajectory that balances urgency
(timing risk from price volatility) against market impact costs.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from ..execution_model import ExecutionModel, Order, OrderType, OrderSide, TimeInForce
from ..portfolio_target import PortfolioTarget


class LiquidityBucket(Enum):
    """Asset liquidity classification for impact parameter selection."""
    MEGA_CAP = "mega_cap"       # AAPL, MSFT, top-10 by ADV
    LARGE_CAP = "large_cap"     # S&P 500 typical
    MID_CAP = "mid_cap"
    SMALL_CAP = "small_cap"
    MICRO_CAP = "micro_cap"
    CRYPTO_MAJOR = "crypto_major"  # BTC, ETH
    CRYPTO_ALT = "crypto_alt"


# Empirically calibrated impact parameters per liquidity bucket
# gamma = permanent impact, eta = temporary impact
IMPACT_PARAMS: Dict[LiquidityBucket, Dict[str, float]] = {
    LiquidityBucket.MEGA_CAP:     {'permanent_impact': 0.01,  'temporary_impact': 0.001},
    LiquidityBucket.LARGE_CAP:    {'permanent_impact': 0.05,  'temporary_impact': 0.005},
    LiquidityBucket.MID_CAP:      {'permanent_impact': 0.10,  'temporary_impact': 0.01},
    LiquidityBucket.SMALL_CAP:    {'permanent_impact': 0.30,  'temporary_impact': 0.03},
    LiquidityBucket.MICRO_CAP:    {'permanent_impact': 0.80,  'temporary_impact': 0.10},
    LiquidityBucket.CRYPTO_MAJOR: {'permanent_impact': 0.05,  'temporary_impact': 0.005},
    LiquidityBucket.CRYPTO_ALT:   {'permanent_impact': 0.50,  'temporary_impact': 0.05},
}


def get_impact_params(bucket: LiquidityBucket) -> Dict[str, float]:
    """Return calibrated (gamma, eta) for a liquidity bucket."""
    return IMPACT_PARAMS[bucket].copy()


@dataclass
class AlmgrenChrissConfig:
    """Configuration for Almgren-Chriss optimal execution."""
    total_shares: Decimal
    total_time: float           # execution horizon in minutes
    num_slices: int = 10
    volatility: float = 0.02   # daily volatility (sigma)
    daily_volume: float = 1e6  # average daily volume (ADV)
    permanent_impact: float = 0.1    # gamma — permanent impact coefficient
    temporary_impact: float = 0.01   # eta — temporary impact coefficient
    risk_aversion: float = 1e-6      # lambda — urgency parameter

    def validate(self) -> List[str]:
        """Validate config for unit consistency. Returns list of warnings."""
        warnings = []
        # Volatility sanity: daily vol should be 0.001–0.20 for equities
        if self.volatility > 1.0:
            warnings.append(
                f"volatility={self.volatility} looks like percentage, not decimal "
                f"(expected daily vol ~0.01–0.05 for equities)"
            )
        if self.volatility < 0:
            warnings.append("volatility must be non-negative")
        # Time horizon sanity
        if self.total_time > 390:
            warnings.append(
                f"total_time={self.total_time} minutes exceeds full trading day (390 min)"
            )
        # Participation rate check
        if self.daily_volume > 0:
            shares_per_min = float(self.total_shares) / max(self.total_time, 1)
            vol_per_min = self.daily_volume / 390.0
            participation = shares_per_min / vol_per_min if vol_per_min > 0 else 0
            if participation > 0.25:
                warnings.append(
                    f"Participation rate {participation:.1%} exceeds 25% of volume — "
                    f"market impact model may underestimate costs"
                )
        return warnings


class AlmgrenChrissModel:
    """Almgren-Chriss optimal trajectory calculator.

    Given risk aversion λ, volatility σ, and impact parameters (γ, η),
    the optimal trajectory minimises:

        E[cost] + λ · Var[cost]

    yielding an analytical closed-form solution involving hyperbolic
    functions.
    """

    def compute_optimal_trajectory(self, config: AlmgrenChrissConfig) -> List[Decimal]:
        """Return the number of shares to trade in each slice.

        The trajectory is a list of length *num_slices* whose values sum
        to *config.total_shares*.
        """
        n = config.num_slices
        if n <= 0:
            return []
        if n == 1:
            return [config.total_shares]

        total = float(config.total_shares)
        tau = config.total_time / n  # time per slice in minutes

        kappa = self._kappa(config)

        # Optimal holdings at each time step k = 0 … n
        # x_k = total * sinh(kappa * (n - k) * tau) / sinh(kappa * n * tau)
        denom = math.sinh(kappa * n * tau)
        if abs(denom) < 1e-15:
            # Degenerate — equal slices (TWAP)
            qty = total / n
            return [Decimal(str(round(qty, 6))) for _ in range(n)]

        holdings = []
        for k in range(n + 1):
            x_k = total * math.sinh(kappa * (n - k) * tau) / denom
            holdings.append(x_k)

        # Trade list: n_k = x_{k-1} - x_k
        trades = []
        for k in range(1, n + 1):
            trade = holdings[k - 1] - holdings[k]
            trades.append(Decimal(str(round(trade, 6))))

        # Ensure trades sum exactly to total_shares via residual adjustment
        residual = config.total_shares - sum(trades)
        trades[-1] += residual

        return trades

    def estimate_execution_cost(self, config: AlmgrenChrissConfig) -> Dict[str, float]:
        """Estimate execution cost components.

        Returns
        -------
        dict with keys:
            expected_cost : total expected cost in price units
            variance : variance of execution cost
            is_cost : implementation shortfall (expected + risk penalty)
            timing_risk : sqrt(variance), a.k.a. timing risk
        """
        total = float(config.total_shares)
        n = config.num_slices
        tau = config.total_time / n if n > 0 else config.total_time

        sigma = config.volatility
        gamma = config.permanent_impact
        eta = config.temporary_impact
        lam = config.risk_aversion

        kappa = self._kappa(config)

        # Expected cost (permanent + temporary)
        permanent_cost = 0.5 * gamma * total ** 2
        # Temporary impact cost depends on trajectory
        trajectory = self.compute_optimal_trajectory(config)
        temp_cost = 0.0
        for trade_dec in trajectory:
            trade = float(trade_dec)
            rate = trade / tau if tau > 0 else 0.0
            temp_cost += eta * rate * trade  # eta * (n_k/tau) * n_k

        expected_cost = permanent_cost + temp_cost

        # Variance of cost — proportional to sigma^2 * sum of x_k^2 * tau
        denom = math.sinh(kappa * n * tau)
        variance = 0.0
        if abs(denom) > 1e-15:
            for k in range(n + 1):
                x_k = total * math.sinh(kappa * (n - k) * tau) / denom
                variance += sigma ** 2 * x_k ** 2 * tau

        timing_risk = math.sqrt(max(variance, 0.0))
        is_cost = expected_cost + lam * variance

        return {
            'expected_cost': expected_cost,
            'variance': variance,
            'is_cost': is_cost,
            'timing_risk': timing_risk,
        }

    @staticmethod
    def _kappa(config: AlmgrenChrissConfig) -> float:
        r"""Compute κ = sqrt(λσ² / η).

        κ controls the shape of the optimal trajectory:
        - Large κ → front-loaded (eager) execution
        - Small κ → uniform (TWAP-like) execution
        """
        lam = config.risk_aversion
        sigma = config.volatility
        eta = config.temporary_impact
        if eta <= 0:
            return 0.0
        return math.sqrt(lam * sigma ** 2 / eta)


class AlmgrenChrissExecutionModel(ExecutionModel):
    """Execution model that uses Almgren-Chriss optimal trajectories.

    Slices orders according to the analytically optimal schedule that
    minimises expected cost + risk-aversion-weighted variance.
    """

    def __init__(
        self,
        duration_minutes: float = 30.0,
        num_slices: int = 10,
        volatility: float = 0.02,
        daily_volume: float = 1e6,
        permanent_impact: float = 0.1,
        temporary_impact: float = 0.01,
        risk_aversion: float = 1e-6,
        name: str = "AlmgrenChrissExecution",
        calibrator: Optional['ImpactCalibrator'] = None,
    ):
        super().__init__(name)
        self.duration_minutes = duration_minutes
        self.num_slices = num_slices
        self.volatility = volatility
        self.daily_volume = daily_volume
        self.permanent_impact = permanent_impact
        self.temporary_impact = temporary_impact
        self.risk_aversion = risk_aversion
        self._current_positions: Dict[str, Decimal] = {}
        self._scheduled_slices: List[Dict] = []
        self._model = AlmgrenChrissModel()
        self._calibrator = calibrator

    def set_current_positions(self, positions: Dict[str, Decimal]) -> None:
        """Update current positions."""
        self._current_positions = positions

    def execute(
        self,
        targets: List[PortfolioTarget],
        market_data: Optional[Dict[str, Any]] = None,
    ) -> List[Order]:
        """Generate optimally-sliced orders using Almgren-Chriss.

        Returns only the first slice (slice_index=0) immediately.
        Remaining slices are stored in ``_scheduled_slices`` for later
        dispatch by an external scheduler, matching the VWAP/TWAP pattern.
        """
        orders = []
        now = datetime.now()
        slice_interval = timedelta(minutes=self.duration_minutes / max(self.num_slices, 1))

        for target in targets:
            current_qty = self._current_positions.get(target.symbol, Decimal("0"))
            total_qty = self.calculate_order_quantity(target, current_qty)

            if total_qty == 0:
                continue

            side = self.calculate_order_side(target, current_qty)

            # Override volume/volatility from market data if available
            vol = self.volatility
            adv = self.daily_volume
            if market_data and target.symbol in market_data:
                sym_data = market_data[target.symbol]
                vol = sym_data.get('volatility', vol)
                adv = sym_data.get('daily_volume', adv)

            # Use calibrated impact params if a calibrator is available
            perm_impact = self.permanent_impact
            temp_impact = self.temporary_impact
            if self._calibrator is not None:
                bucket = ImpactCalibrator.classify_bucket(adv, target.symbol)
                params = self._calibrator.get_calibrated_params(bucket)
                perm_impact = params['permanent_impact']
                temp_impact = params['temporary_impact']

            config = AlmgrenChrissConfig(
                total_shares=total_qty,
                total_time=self.duration_minutes,
                num_slices=self.num_slices,
                volatility=vol,
                daily_volume=adv,
                permanent_impact=perm_impact,
                temporary_impact=temp_impact,
                risk_aversion=self.risk_aversion,
            )

            trajectory = self._model.compute_optimal_trajectory(config)

            for i, slice_qty in enumerate(trajectory):
                if slice_qty <= 0:
                    continue

                order = Order(
                    symbol=target.symbol,
                    side=side,
                    quantity=slice_qty.quantize(Decimal("1")),
                    order_type=OrderType.MARKET,
                    source_target_id=target.id,
                    metadata={
                        'slice_index': i,
                        'total_slices': self.num_slices,
                        'algorithm': 'AlmgrenChriss',
                    },
                )
                orders.append(order)

                # Schedule future slices
                if i > 0:
                    self._scheduled_slices.append({
                        'order': order,
                        'scheduled_time': now + (slice_interval * i),
                    })

        # Return only first slice immediately
        immediate_orders = [o for o in orders if o.metadata.get('slice_index', 0) == 0]
        return immediate_orders

    def get_scheduled_slices(self) -> List[Dict]:
        """Get pending scheduled slices."""
        return self._scheduled_slices.copy()


# ---------------------------------------------------------------------------
# Impact Parameter Calibration from Execution History
# ---------------------------------------------------------------------------

@dataclass
class CalibrationRecord:
    """A single execution observation used for impact calibration.

    Includes timestamps so the calibrator can verify the impact observation
    was actually sampled near the target horizon (≈300 s post-fill) and not
    forward-filled or stale from a later bar aggregation.
    """
    symbol: str
    side: str                    # 'buy' or 'sell'
    filled_quantity: float
    daily_volume: float          # ADV at time of trade
    slippage_bps: float          # immediate: (fill - market) / market * 10000
    impact_5min_bps: float       # permanent: price change at 5min post-fill
    fill_timestamp: Optional[datetime] = None
    impact_observed_timestamp: Optional[datetime] = None
    market_cap_bucket: Optional[LiquidityBucket] = None

    def __post_init__(self):
        if self.market_cap_bucket is None:
            self.market_cap_bucket = ImpactCalibrator.classify_bucket(
                self.daily_volume, self.symbol,
            )

    @property
    def impact_lag_seconds(self) -> Optional[float]:
        """Actual seconds between fill and impact observation, or None."""
        if self.fill_timestamp and self.impact_observed_timestamp:
            return (self.impact_observed_timestamp - self.fill_timestamp).total_seconds()
        return None


class ImpactCalibrator:
    """Calibrates Almgren-Chriss impact parameters from execution history.

    Consumes ``CalibrationRecord`` objects (which can be derived from
    ``ToxicFlowDetector.TradeRecord`` or ``SlippagePredictor.ExecutionRecord``)
    and produces per-bucket ``(permanent_impact, temporary_impact)`` estimates
    that replace the hardcoded ``IMPACT_PARAMS`` defaults.

    Records are validated at ingestion time: if both ``fill_timestamp`` and
    ``impact_observed_timestamp`` are present, the actual lag must be within
    ``impact_lag_tolerance`` of the 300 s target horizon.  Records that were
    observed too early (mid not yet settled) or too late (measuring a
    different horizon) are silently rejected to prevent lookahead and
    survivorship-style contamination.

    The calibration approach:
    * **temporary_impact** — OLS slope of ``slippage_bps ~ beta * participation_rate``
    * **permanent_impact** — ``median(impact_5min_bps) / median(participation_rate)``
    """

    MIN_SAMPLES = 30  # per bucket
    IMPACT_HORIZON_SECONDS = 300.0  # 5-minute target horizon

    # ADV thresholds (shares/day) for bucket classification
    _ADV_THRESHOLDS = [
        (50_000_000, LiquidityBucket.MEGA_CAP),
        (10_000_000, LiquidityBucket.LARGE_CAP),
        (2_000_000,  LiquidityBucket.MID_CAP),
        (500_000,    LiquidityBucket.SMALL_CAP),
    ]

    _CRYPTO_PREFIXES = ('BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'DOGE', 'AVAX', 'DOT')
    _CRYPTO_MAJOR = ('BTC', 'ETH')

    def __init__(
        self,
        impact_lag_tolerance: float = 60.0,
    ) -> None:
        """
        Parameters
        ----------
        impact_lag_tolerance : float
            Maximum allowed deviation (in seconds) between the actual
            impact observation lag and the 300 s target horizon.
            Records outside ``[300 - tol, 300 + tol]`` are rejected.
            Set to ``math.inf`` to disable the check.  Default 60 s
            accepts observations sampled between 4 and 6 minutes
            post-fill.
        """
        self._records: List[CalibrationRecord] = []
        self._rejected_count: int = 0
        self._calibrated: Dict[LiquidityBucket, Dict[str, float]] = {}
        self._impact_lag_tolerance = impact_lag_tolerance

    def record(self, rec: CalibrationRecord) -> bool:
        """Add a calibration observation.

        Returns ``True`` if the record was accepted, ``False`` if it was
        rejected because its impact observation lag falls outside the
        tolerance window around the 300 s horizon.  Records without
        timestamps are accepted unconditionally (no data to validate).
        """
        lag = rec.impact_lag_seconds
        if lag is not None:
            lo = self.IMPACT_HORIZON_SECONDS - self._impact_lag_tolerance
            hi = self.IMPACT_HORIZON_SECONDS + self._impact_lag_tolerance
            if lag < lo or lag > hi:
                self._rejected_count += 1
                return False
        self._records.append(rec)
        return True

    @property
    def rejected_count(self) -> int:
        """Number of records rejected due to impact lag violation."""
        return self._rejected_count

    @staticmethod
    def classify_bucket(daily_volume: float, symbol: str = "") -> LiquidityBucket:
        """Classify a symbol into a liquidity bucket by ADV.

        Crypto symbols are detected by common ticker prefixes; equities
        are classified purely by average daily volume.
        """
        sym_upper = symbol.upper()
        # Detect crypto by prefix (handles pairs like BTC-USD, BTCUSD, etc.)
        for prefix in ImpactCalibrator._CRYPTO_PREFIXES:
            if sym_upper.startswith(prefix):
                if any(sym_upper.startswith(m) for m in ImpactCalibrator._CRYPTO_MAJOR):
                    return LiquidityBucket.CRYPTO_MAJOR
                return LiquidityBucket.CRYPTO_ALT

        for threshold, bucket in ImpactCalibrator._ADV_THRESHOLDS:
            if daily_volume >= threshold:
                return bucket
        return LiquidityBucket.MICRO_CAP

    def calibrate(self) -> Dict[LiquidityBucket, Dict[str, float]]:
        """Calibrate impact parameters from recorded execution history.

        Groups records by bucket.  For each bucket with at least
        ``MIN_SAMPLES`` observations:
        * Computes participation_rate = filled_quantity / daily_volume
        * Regresses slippage_bps on participation_rate → slope ≈ temporary_impact
        * permanent_impact = median(impact_5min_bps) / median(participation_rate)

        Buckets with insufficient data retain their ``IMPACT_PARAMS`` defaults.

        Returns
        -------
        dict mapping LiquidityBucket → {permanent_impact, temporary_impact}
        """
        # Group by bucket
        grouped: Dict[LiquidityBucket, List[CalibrationRecord]] = {}
        for rec in self._records:
            bucket = rec.market_cap_bucket
            grouped.setdefault(bucket, []).append(rec)

        result: Dict[LiquidityBucket, Dict[str, float]] = {}

        for bucket in LiquidityBucket:
            records = grouped.get(bucket, [])
            if len(records) < self.MIN_SAMPLES:
                result[bucket] = IMPACT_PARAMS[bucket].copy()
                continue

            participation = np.array([
                r.filled_quantity / r.daily_volume
                for r in records if r.daily_volume > 0
            ])
            slippage = np.array([
                r.slippage_bps for r in records if r.daily_volume > 0
            ])
            impact_5m = np.array([
                r.impact_5min_bps for r in records if r.daily_volume > 0
            ])

            if len(participation) < self.MIN_SAMPLES:
                result[bucket] = IMPACT_PARAMS[bucket].copy()
                continue

            # Temporary impact: OLS  slippage ~ beta * participation_rate
            # beta = Cov(x,y) / Var(x)
            x_mean = participation.mean()
            y_mean = slippage.mean()
            var_x = ((participation - x_mean) ** 2).sum()
            if var_x > 1e-15:
                cov_xy = ((participation - x_mean) * (slippage - y_mean)).sum()
                temp_impact = max(cov_xy / var_x, 1e-6)
            else:
                temp_impact = IMPACT_PARAMS[bucket]['temporary_impact']

            # Permanent impact: median(5min impact) / median(participation)
            med_impact = float(np.median(np.abs(impact_5m)))
            med_part = float(np.median(participation))
            if med_part > 1e-15:
                perm_impact = max(med_impact / med_part, 1e-6)
            else:
                perm_impact = IMPACT_PARAMS[bucket]['permanent_impact']

            result[bucket] = {
                'permanent_impact': perm_impact,
                'temporary_impact': temp_impact,
            }

        self._calibrated = result
        return result

    def get_calibrated_params(self, bucket: LiquidityBucket) -> Dict[str, float]:
        """Return calibrated params for a bucket, falling back to defaults."""
        if bucket in self._calibrated:
            return self._calibrated[bucket].copy()
        return IMPACT_PARAMS[bucket].copy()
