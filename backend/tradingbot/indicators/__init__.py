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
    # Base
    "Indicator",
    "IndicatorDataPoint",
    "MovingAverageType",
    # Moving Averages
    "SimpleMovingAverage",
    "ExponentialMovingAverage",
    "DoubleExponentialMovingAverage",
    "TripleExponentialMovingAverage",
    "WilderMovingAverage",
    "KaufmanAdaptiveMovingAverage",
    "HullMovingAverage",
    "WeightedMovingAverage",
    "TriangularMovingAverage",
    # Oscillators
    "RelativeStrengthIndex",
    "Stochastic",
    "StochasticRSI",
    "CommodityChannelIndex",
    "WilliamsPercentR",
    "MoneyFlowIndex",
    "UltimateOscillator",
    # Momentum
    "MACD",
    "RateOfChange",
    "Momentum",
    "AwesomeOscillator",
    "PercentagePriceOscillator",
    "TrueStrengthIndex",
    # Volatility
    "BollingerBands",
    "AverageTrueRange",
    "StandardDeviation",
    "KeltnerChannels",
    "DonchianChannel",
    "ChoppinessIndex",
    # Volume
    "OnBalanceVolume",
    "AccumulationDistribution",
    "ChaikinMoneyFlow",
    "VolumeWeightedAveragePrice",
    "ForceIndex",
    "NegativeVolumeIndex",
    "PositiveVolumeIndex",
    # Trend
    "AverageDirectionalIndex",
    "ParabolicSAR",
    "Aroon",
    "IchimokuCloud",
    "SuperTrend",
    # Candlestick Patterns
    "Doji",
    "Hammer",
    "ShootingStar",
    "Engulfing",
    "MorningStar",
    "EveningStar",
    "Harami",
    "ThreeWhiteSoldiers",
    "ThreeCrows",
    "SpinningTop",
    # Pivot Points
    "ClassicPivotPoints",
    "FibonacciPivotPoints",
    "WoodiePivotPoints",
    "CamarillaPivotPoints",
    # Advanced Oscillators
    "ElderRayBull",
    "ElderRayBear",
    "DeMarker",
    "ConnorsRSI",
    "FisherTransform",
    "ChandeMomentumOscillator",
    "KlingerVolumeOscillator",
    # Hybrid
    "VolumeWeightedMovingAverage",
    "McGinleyDynamic",
    "ZigZag",
    "VolumeProfile",
    "MarketProfile",
]
