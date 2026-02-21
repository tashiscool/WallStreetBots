"""
Technical Indicators Library - Ported from QuantConnect LEAN.

This module provides a comprehensive set of technical indicators
for use in trading strategies.

Ported under Apache 2.0 License from:
https://github.com/QuantConnect/Lean

Original Copyright: QuantConnect Corporation
"""

from .base import Indicator, IndicatorDataPoint, MovingAverageType
from .moving_averages import (
    SimpleMovingAverage,
    ExponentialMovingAverage,
    DoubleExponentialMovingAverage,
    TripleExponentialMovingAverage,
    WilderMovingAverage,
    KaufmanAdaptiveMovingAverage,
    HullMovingAverage,
    WeightedMovingAverage,
    TriangularMovingAverage,
)
from .oscillators import (
    RelativeStrengthIndex,
    Stochastic,
    StochasticRSI,
    CommodityChannelIndex,
    WilliamsPercentR,
    MoneyFlowIndex,
    UltimateOscillator,
)
from .momentum import (
    MACD,
    RateOfChange,
    Momentum,
    AwesomeOscillator,
    PercentagePriceOscillator,
    TrueStrengthIndex,
)
from .volatility import (
    BollingerBands,
    AverageTrueRange,
    StandardDeviation,
    KeltnerChannels,
    DonchianChannel,
    ChoppinessIndex,
)
from .volume import (
    OnBalanceVolume,
    AccumulationDistribution,
    ChaikinMoneyFlow,
    VolumeWeightedAveragePrice,
    ForceIndex,
    NegativeVolumeIndex,
    PositiveVolumeIndex,
)
from .trend import (
    AverageDirectionalIndex,
    ParabolicSAR,
    Aroon,
    IchimokuCloud,
    SuperTrend,
)
from .candlestick import (
    Doji,
    Hammer,
    ShootingStar,
    Engulfing,
    MorningStar,
    EveningStar,
    Harami,
    ThreeWhiteSoldiers,
    ThreeCrows,
    SpinningTop,
)
from .pivot_points import (
    ClassicPivotPoints,
    FibonacciPivotPoints,
    WoodiePivotPoints,
    CamarillaPivotPoints,
)
from .advanced_oscillators import (
    ElderRayBull,
    ElderRayBear,
    DeMarker,
    ConnorsRSI,
    FisherTransform,
    ChandeMomentumOscillator,
    KlingerVolumeOscillator,
)
from .hybrid import (
    VolumeWeightedMovingAverage,
    McGinleyDynamic,
    ZigZag,
    VolumeProfile,
    MarketProfile,
)

__all__ = [
    # Momentum
    "MACD",
    "AccumulationDistribution",
    "Aroon",
    # Trend
    "AverageDirectionalIndex",
    "AverageTrueRange",
    "AwesomeOscillator",
    # Volatility
    "BollingerBands",
    "CamarillaPivotPoints",
    "ChaikinMoneyFlow",
    "ChandeMomentumOscillator",
    "ChoppinessIndex",
    # Pivot Points
    "ClassicPivotPoints",
    "CommodityChannelIndex",
    "ConnorsRSI",
    "DeMarker",
    # Candlestick Patterns
    "Doji",
    "DonchianChannel",
    "DoubleExponentialMovingAverage",
    "ElderRayBear",
    # Advanced Oscillators
    "ElderRayBull",
    "Engulfing",
    "EveningStar",
    "ExponentialMovingAverage",
    "FibonacciPivotPoints",
    "FisherTransform",
    "ForceIndex",
    "Hammer",
    "Harami",
    "HullMovingAverage",
    "IchimokuCloud",
    # Base
    "Indicator",
    "IndicatorDataPoint",
    "KaufmanAdaptiveMovingAverage",
    "KeltnerChannels",
    "KlingerVolumeOscillator",
    "MarketProfile",
    "McGinleyDynamic",
    "Momentum",
    "MoneyFlowIndex",
    "MorningStar",
    "MovingAverageType",
    "NegativeVolumeIndex",
    # Volume
    "OnBalanceVolume",
    "ParabolicSAR",
    "PercentagePriceOscillator",
    "PositiveVolumeIndex",
    "RateOfChange",
    # Oscillators
    "RelativeStrengthIndex",
    "ShootingStar",
    # Moving Averages
    "SimpleMovingAverage",
    "SpinningTop",
    "StandardDeviation",
    "Stochastic",
    "StochasticRSI",
    "SuperTrend",
    "ThreeCrows",
    "ThreeWhiteSoldiers",
    "TriangularMovingAverage",
    "TripleExponentialMovingAverage",
    "TrueStrengthIndex",
    "UltimateOscillator",
    "VolumeProfile",
    "VolumeWeightedAveragePrice",
    # Hybrid
    "VolumeWeightedMovingAverage",
    "WeightedMovingAverage",
    "WilderMovingAverage",
    "WilliamsPercentR",
    "WoodiePivotPoints",
    "ZigZag",
]
