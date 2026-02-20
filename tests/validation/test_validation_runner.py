import pandas as pd

from backend.validation.validation_runner import (
    ValidationRunner,
    _load_strategy_returns_from_csv,
)


class _FactorResult:
    def __init__(self, alpha_significant: bool):
        self.alpha_significant = alpha_significant


def test_generate_validation_summary_handles_mixed_result_types():
    runner = ValidationRunner()
    results = {
        "reality_check": {
            "recommendation": {"significant_strategies": ["swing_trading"]}
        },
        "factor_analysis": {
            "swing_trading": _FactorResult(alpha_significant=True),
            "wheel_strategy": {"alpha_significant": False},
        },
        "regime_testing": {
            "swing_trading": {"edge_is_robust": True},
            "wheel_strategy": {"edge_is_robust": False},
        },
    }

    summary = runner._generate_validation_summary(results)

    assert summary["overall_status"] == "COMPLETED"
    assert any("Significant alpha" in item for item in summary["recommendations"])
    assert any("Regime-robust strategies" in item for item in summary["recommendations"])


def test_load_strategy_returns_from_csv(tmp_path):
    csv_path = tmp_path / "returns.csv"
    frame = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "return": [0.01, -0.005, 0.02],
        }
    )
    frame.to_csv(csv_path, index=False)

    returns = _load_strategy_returns_from_csv(str(csv_path), "test_strategy")

    assert returns is not None
    assert returns.name == "test_strategy"
    assert len(returns) == 3
