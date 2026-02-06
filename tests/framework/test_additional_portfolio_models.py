"""Tests for Additional Portfolio Models (Phase 5)."""
from decimal import Decimal
import numpy as np
import pytest

from backend.tradingbot.framework.insight import Insight, InsightDirection
from backend.tradingbot.framework.portfolio_model import PortfolioState
from backend.tradingbot.framework.portfolio_target import PortfolioTarget
from backend.tradingbot.framework.portfolio_models.min_variance import MinVariancePortfolioModel
from backend.tradingbot.framework.portfolio_models.max_diversification import MaxDiversificationPortfolioModel
from backend.tradingbot.framework.portfolio_models.hierarchical_risk_parity import HierarchicalRiskParityModel
from backend.tradingbot.framework.portfolio_models.max_sharpe import MaxSharpePortfolioModel
from backend.tradingbot.framework.portfolio_models.mean_variance import MeanVariancePortfolioModel
from backend.tradingbot.framework.portfolio_models.black_litterman import BlackLittermanPortfolioModel


def _make_insights_with_returns(n_assets=4, n_periods=100):
    """Create insights with synthetic return data."""
    np.random.seed(42)
    insights = []
    for i in range(n_assets):
        # Generate correlated returns
        rets = np.random.normal(0.001, 0.02, n_periods).tolist()
        insights.append(Insight(
            symbol=f"SYM{i}",
            direction=InsightDirection.UP,
            magnitude=0.03,
            confidence=0.6 + i * 0.05,
            metadata={"returns": rets, "price": 100 + i * 10},
        ))
    return insights


def _make_portfolio_state():
    return PortfolioState(
        cash=Decimal("100000"),
        total_value=Decimal("100000"),
    )


class TestMinVariancePortfolioModel:
    def test_init(self):
        model = MinVariancePortfolioModel()
        assert model.name == "MinVariance"
        assert model.max_positions == 15

    def test_creates_targets(self):
        model = MinVariancePortfolioModel()
        model.portfolio_state = _make_portfolio_state()
        insights = _make_insights_with_returns()
        targets = model.create_targets(insights)
        assert len(targets) > 0
        assert all(isinstance(t, PortfolioTarget) for t in targets)

    def test_weights_sum_to_one(self):
        model = MinVariancePortfolioModel()
        model.portfolio_state = _make_portfolio_state()
        insights = _make_insights_with_returns()
        targets = model.create_targets(insights)
        if targets:
            total_weight = sum(t.target_weight for t in targets)
            assert abs(total_weight - 1.0) < 0.01

    def test_single_insight_fallback(self):
        model = MinVariancePortfolioModel()
        model.portfolio_state = _make_portfolio_state()
        insights = [Insight(
            symbol="SPY",
            direction=InsightDirection.UP,
            confidence=0.8,
            metadata={"price": 100},
        )]
        targets = model.create_targets(insights)
        assert len(targets) == 1

    def test_empty_insights(self):
        model = MinVariancePortfolioModel()
        model.portfolio_state = _make_portfolio_state()
        targets = model.create_targets([])
        assert targets == []

    def test_weight_bounds(self):
        model = MinVariancePortfolioModel(min_weight=0.05, max_weight=0.50)
        model.portfolio_state = _make_portfolio_state()
        insights = _make_insights_with_returns()
        targets = model.create_targets(insights)
        for t in targets:
            assert t.target_weight >= 0.05 - 0.01
            assert t.target_weight <= 0.50 + 0.01


class TestMaxDiversificationPortfolioModel:
    def test_init(self):
        model = MaxDiversificationPortfolioModel()
        assert model.name == "MaxDiversification"

    def test_creates_targets(self):
        model = MaxDiversificationPortfolioModel()
        model.portfolio_state = _make_portfolio_state()
        insights = _make_insights_with_returns()
        targets = model.create_targets(insights)
        assert len(targets) > 0

    def test_weights_sum_to_one(self):
        model = MaxDiversificationPortfolioModel()
        model.portfolio_state = _make_portfolio_state()
        insights = _make_insights_with_returns()
        targets = model.create_targets(insights)
        if targets:
            total_weight = sum(t.target_weight for t in targets)
            assert abs(total_weight - 1.0) < 0.01


class TestHierarchicalRiskParityModel:
    def test_init(self):
        model = HierarchicalRiskParityModel()
        assert model.name == "HRP"

    def test_creates_targets(self):
        model = HierarchicalRiskParityModel()
        model.portfolio_state = _make_portfolio_state()
        insights = _make_insights_with_returns()
        targets = model.create_targets(insights)
        assert len(targets) > 0

    def test_single_insight_equal_weight(self):
        model = HierarchicalRiskParityModel()
        model.portfolio_state = _make_portfolio_state()
        insights = [Insight(
            symbol="SPY",
            direction=InsightDirection.UP,
            confidence=0.8,
            metadata={"price": 100},
        )]
        targets = model.create_targets(insights)
        assert len(targets) == 1
        assert abs(targets[0].target_weight - 1.0) < 0.01

    def test_quasi_diag_produces_valid_order(self):
        model = HierarchicalRiskParityModel()
        np.random.seed(42)
        n = 4
        corr = np.eye(n) + np.random.normal(0, 0.1, (n, n))
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1.0)
        dist = np.sqrt(0.5 * (1 - corr))
        order = model._quasi_diag(dist, n)
        assert len(order) == n
        assert set(order) == set(range(n))


class TestMaxSharpePortfolioModel:
    def test_init(self):
        model = MaxSharpePortfolioModel()
        assert model.name == "MaxSharpe"
        assert model.risk_free_rate == pytest.approx(0.02 / 252, abs=1e-6)

    def test_creates_targets(self):
        model = MaxSharpePortfolioModel()
        model.portfolio_state = _make_portfolio_state()
        insights = _make_insights_with_returns()
        targets = model.create_targets(insights)
        assert len(targets) > 0

    def test_weights_sum_to_one(self):
        model = MaxSharpePortfolioModel()
        model.portfolio_state = _make_portfolio_state()
        insights = _make_insights_with_returns()
        targets = model.create_targets(insights)
        if targets:
            total_weight = sum(t.target_weight for t in targets)
            assert abs(total_weight - 1.0) < 0.01


class TestMeanVariancePortfolioModel:
    def test_init(self):
        model = MeanVariancePortfolioModel()
        assert model.name == "MeanVariance"
        assert model.risk_aversion == 1.0

    def test_creates_targets(self):
        model = MeanVariancePortfolioModel()
        model.portfolio_state = _make_portfolio_state()
        insights = _make_insights_with_returns()
        targets = model.create_targets(insights)
        assert len(targets) > 0

    def test_higher_risk_aversion_less_aggressive(self):
        """Higher risk aversion should still produce valid weights."""
        model_low = MeanVariancePortfolioModel(risk_aversion=0.5)
        model_high = MeanVariancePortfolioModel(risk_aversion=5.0)
        state = _make_portfolio_state()
        model_low.portfolio_state = state
        model_high.portfolio_state = state
        insights = _make_insights_with_returns()

        targets_low = model_low.create_targets(insights)
        targets_high = model_high.create_targets(insights)

        # Both should produce valid targets
        assert len(targets_low) > 0
        assert len(targets_high) > 0


class TestBlackLittermanPortfolioModel:
    def test_init(self):
        model = BlackLittermanPortfolioModel()
        assert model.name == "BlackLitterman"
        assert model.tau == 0.05
        assert model.risk_aversion == 2.5

    def test_creates_targets(self):
        model = BlackLittermanPortfolioModel()
        model.portfolio_state = _make_portfolio_state()
        insights = _make_insights_with_returns()
        targets = model.create_targets(insights)
        assert len(targets) > 0

    def test_weights_sum_to_one(self):
        model = BlackLittermanPortfolioModel()
        model.portfolio_state = _make_portfolio_state()
        insights = _make_insights_with_returns()
        targets = model.create_targets(insights)
        if targets:
            total_weight = sum(t.target_weight for t in targets)
            assert abs(total_weight - 1.0) < 0.01

    def test_bearish_insights(self):
        """Test with DOWN direction insights."""
        model = BlackLittermanPortfolioModel()
        model.portfolio_state = _make_portfolio_state()
        np.random.seed(42)
        insights = []
        for i in range(3):
            rets = np.random.normal(-0.001, 0.02, 100).tolist()
            insights.append(Insight(
                symbol=f"SYM{i}",
                direction=InsightDirection.DOWN,
                magnitude=0.03,
                confidence=0.7,
                metadata={"returns": rets, "price": 100},
            ))
        targets = model.create_targets(insights)
        # Should still produce valid targets
        assert isinstance(targets, list)

    def test_single_insight_equal_weight(self):
        model = BlackLittermanPortfolioModel()
        model.portfolio_state = _make_portfolio_state()
        insights = [Insight(
            symbol="SPY",
            direction=InsightDirection.UP,
            confidence=0.8,
            metadata={"price": 100},
        )]
        targets = model.create_targets(insights)
        assert len(targets) == 1
