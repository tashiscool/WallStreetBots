"""
Portfolio Management Module.

Provides portfolio rebalancing, drift detection, and allocation management.

Features:
- DriftRebalancer: Auto-rebalance when weights drift from target
- PortfolioAllocator: Manage target allocations
- RebalanceScheduler: Schedule rebalancing on specific days/times
"""

from .drift_rebalancer import (
    DriftRebalancer,
    DriftConfig,
    RebalanceOrder,
    RebalanceResult,
    DriftType,
)

from .allocator import (
    PortfolioAllocator,
    AllocationTarget,
    AllocationResult,
)

__all__ = [
    "DriftRebalancer",
    "DriftConfig",
    "RebalanceOrder",
    "RebalanceResult",
    "DriftType",
    "PortfolioAllocator",
    "AllocationTarget",
    "AllocationResult",
]
