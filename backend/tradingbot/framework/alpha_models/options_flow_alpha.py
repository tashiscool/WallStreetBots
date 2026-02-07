"""
Options Flow Alpha Model — Unusual options activity signal generator.

Analyzes put/call ratios, volume/OI spikes, and block trades to
generate directional insights.
"""

import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional

from ..alpha_model import AlphaModel
from ..insight import Insight, InsightDirection

logger = logging.getLogger(__name__)


class OptionsFlowAlphaModel(AlphaModel):
    """
    Alpha model that detects unusual options flow as a leading indicator.

    Expects option chain data in ``data["options"]`` keyed by symbol::

        data["options"] = {
            "AAPL": OptionChain(
                symbol="AAPL",
                calls=[{strike, volume, open_interest, ...}, ...],
                puts=[...],
            ),
        }

    Signals:
    - High put/call volume ratio → bearish (DOWN)
    - Low put/call volume ratio → bullish (UP)
    - Large block trades → direction depends on call vs put
    - Volume/OI spikes → unusual activity flag
    """

    def __init__(
        self,
        pc_ratio_bullish: float = 0.5,
        pc_ratio_bearish: float = 1.5,
        volume_oi_spike: float = 3.0,
        block_trade_threshold: int = 1000,
        min_total_volume: int = 100,
        name: str = "OptionsFlowAlpha",
    ) -> None:
        super().__init__(name)
        self.pc_ratio_bullish = pc_ratio_bullish
        self.pc_ratio_bearish = pc_ratio_bearish
        self.volume_oi_spike = volume_oi_spike
        self.block_trade_threshold = block_trade_threshold
        self.min_total_volume = min_total_volume

    def generate_insights(
        self,
        data: Dict[str, Any],
        symbols: List[str],
    ) -> List[Insight]:
        """Generate insights from options flow data."""
        insights: List[Insight] = []
        options_data = data.get("options", {})

        for symbol in symbols:
            chain = options_data.get(symbol)
            if chain is None:
                continue

            calls = self._get_contracts(chain, "calls")
            puts = self._get_contracts(chain, "puts")

            if not calls and not puts:
                continue

            analysis = self._analyze_flow(calls, puts)

            if analysis["total_volume"] < self.min_total_volume:
                continue

            direction, confidence, metadata = self._determine_signal(analysis)

            if direction == InsightDirection.FLAT:
                continue

            insights.append(Insight(
                symbol=symbol,
                direction=direction,
                magnitude=0.02,
                confidence=confidence,
                period=timedelta(days=5),
                source_model=self.name,
                metadata=metadata,
            ))

        return insights

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _analyze_flow(
        self,
        calls: List[Dict[str, Any]],
        puts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Analyze call/put flow for signals."""
        call_volume = sum(c.get("volume", 0) for c in calls)
        put_volume = sum(p.get("volume", 0) for p in puts)
        call_oi = sum(c.get("open_interest", 0) for c in calls)
        put_oi = sum(p.get("open_interest", 0) for p in puts)

        total_volume = call_volume + put_volume
        pc_ratio = put_volume / call_volume if call_volume > 0 else 0.0

        # Detect volume/OI spikes (contracts where volume >> OI)
        vol_oi_spikes = 0
        for contract in calls + puts:
            vol = contract.get("volume", 0)
            oi = contract.get("open_interest", 1)
            if oi > 0 and vol / oi > self.volume_oi_spike:
                vol_oi_spikes += 1

        # Detect block trades
        call_blocks = [c for c in calls if c.get("volume", 0) >= self.block_trade_threshold]
        put_blocks = [p for p in puts if p.get("volume", 0) >= self.block_trade_threshold]

        return {
            "call_volume": call_volume,
            "put_volume": put_volume,
            "call_oi": call_oi,
            "put_oi": put_oi,
            "total_volume": total_volume,
            "pc_ratio": pc_ratio,
            "vol_oi_spikes": vol_oi_spikes,
            "call_blocks": len(call_blocks),
            "put_blocks": len(put_blocks),
        }

    def _determine_signal(
        self,
        analysis: Dict[str, Any],
    ) -> tuple:
        """Determine direction and confidence from flow analysis.

        Returns:
            ``(direction, confidence, metadata)``
        """
        pc = analysis["pc_ratio"]
        spikes = analysis["vol_oi_spikes"]
        call_blocks = analysis["call_blocks"]
        put_blocks = analysis["put_blocks"]

        direction = InsightDirection.FLAT
        confidence = 0.0
        signals: List[str] = []

        # Put/call ratio signal
        if pc <= self.pc_ratio_bullish:
            direction = InsightDirection.UP
            confidence = 0.6
            signals.append(f"low_pc_ratio:{pc:.2f}")
        elif pc >= self.pc_ratio_bearish:
            direction = InsightDirection.DOWN
            confidence = 0.6
            signals.append(f"high_pc_ratio:{pc:.2f}")

        # Block trade signal (can amplify or override)
        if call_blocks > put_blocks:
            if direction != InsightDirection.DOWN:
                direction = InsightDirection.UP
                confidence = max(confidence, 0.7)
                signals.append(f"call_blocks:{call_blocks}")
        elif put_blocks > call_blocks:
            if direction != InsightDirection.UP:
                direction = InsightDirection.DOWN
                confidence = max(confidence, 0.7)
                signals.append(f"put_blocks:{put_blocks}")

        # Volume/OI spikes boost confidence
        if spikes > 0:
            confidence = min(confidence + 0.1, 0.95)
            signals.append(f"vol_oi_spikes:{spikes}")

        metadata = {**analysis, "signals": signals}
        return direction, confidence, metadata

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_contracts(chain: Any, side: str) -> List[Dict[str, Any]]:
        """Extract contract list from OptionChain or dict."""
        if isinstance(chain, dict):
            return chain.get(side, [])
        return getattr(chain, side, [])
