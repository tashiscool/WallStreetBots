"""
Portfolio Construction Model Implementations

Use get_portfolio_model(name) to instantiate by name.
"""

from typing import Any, Dict

from .equal_weight import EqualWeightPortfolioModel
from .insight_weighted import InsightWeightedPortfolioModel
from .risk_parity import RiskParityPortfolioModel
from .kelly import KellyPortfolioModel
from .sector_weighted import SectorWeightedPortfolioModel
from .min_variance import MinVariancePortfolioModel
from .max_diversification import MaxDiversificationPortfolioModel
from .hierarchical_risk_parity import HierarchicalRiskParityModel
from .max_sharpe import MaxSharpePortfolioModel
from .mean_variance import MeanVariancePortfolioModel
from .black_litterman import BlackLittermanPortfolioModel

# Registry: name -> class
PORTFOLIO_MODELS: Dict[str, type] = {
    'equal_weight': EqualWeightPortfolioModel,
    'insight_weighted': InsightWeightedPortfolioModel,
    'risk_parity': RiskParityPortfolioModel,
    'kelly': KellyPortfolioModel,
    'sector_weighted': SectorWeightedPortfolioModel,
    'min_variance': MinVariancePortfolioModel,
    'max_diversification': MaxDiversificationPortfolioModel,
    'hierarchical_risk_parity': HierarchicalRiskParityModel,
    'max_sharpe': MaxSharpePortfolioModel,
    'mean_variance': MeanVariancePortfolioModel,
    'black_litterman': BlackLittermanPortfolioModel,
}


def get_portfolio_model(name: str, **kwargs) -> Any:
    """Instantiate a portfolio construction model by name.

    Args:
        name: Model name (e.g. 'equal_weight', 'max_sharpe', 'hierarchical_risk_parity')
        **kwargs: Passed to model constructor

    Returns:
        Portfolio model instance

    Raises:
        ValueError: If name is not registered
    """
    cls = PORTFOLIO_MODELS.get(name)
    if cls is None:
        available = ", ".join(sorted(PORTFOLIO_MODELS.keys()))
        raise ValueError(f"Unknown portfolio model '{name}'. Available: {available}")
    return cls(**kwargs)


def list_portfolio_models() -> Dict[str, str]:
    """List available portfolio models with their class names."""
    return {name: cls.__name__ for name, cls in PORTFOLIO_MODELS.items()}


__all__ = [
    'EqualWeightPortfolioModel',
    'InsightWeightedPortfolioModel',
    'RiskParityPortfolioModel',
    'KellyPortfolioModel',
    'SectorWeightedPortfolioModel',
    'MinVariancePortfolioModel',
    'MaxDiversificationPortfolioModel',
    'HierarchicalRiskParityModel',
    'MaxSharpePortfolioModel',
    'MeanVariancePortfolioModel',
    'BlackLittermanPortfolioModel',
    'PORTFOLIO_MODELS',
    'get_portfolio_model',
    'list_portfolio_models',
]
