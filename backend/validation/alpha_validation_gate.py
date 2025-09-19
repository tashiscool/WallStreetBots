"""Alpha Validation Gate with Data-Driven Scorecard."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class ValidationCriteria:
    """Frozen validation criteria for deployment readiness."""
    # risk-adjusted
    min_sharpe_ratio: float = 1.0
    max_drawdown: float = 0.15
    min_win_rate: float = 0.45
    # alpha validation
    min_factor_adjusted_alpha: float = 0.05   # annualized
    alpha_t_stat_threshold: float = 2.0
    min_regime_consistency: float = 0.70
    # cross-market
    min_success_rate: float = 0.60
    min_universes_validated: int = 2
    # execution quality
    max_slippage_variance: float = 0.10
    min_fill_rate: float = 0.95
    # capital efficiency
    min_return_on_capital: float = 0.15
    max_margin_calls: int = 0


class AlphaValidationGate:
    """Evaluates consolidated validation results against deployment criteria."""

    def __init__(self, criteria: ValidationCriteria | None = None):
        self.criteria = criteria or ValidationCriteria()

    def evaluate_go_no_go(self, results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Evaluate go/no-go decision with comprehensive scorecard."""
        c = self.criteria
        checks = {
            'risk_adjusted_returns.min_sharpe_ratio': (results.get('risk_adjusted_returns', {}).get('min_sharpe_ratio'), c.min_sharpe_ratio, 'min'),
            'risk_adjusted_returns.max_drawdown':     (results.get('risk_adjusted_returns', {}).get('max_drawdown'),     c.max_drawdown,     'max'),
            'risk_adjusted_returns.min_win_rate':     (results.get('risk_adjusted_returns', {}).get('min_win_rate'),     c.min_win_rate,     'min'),

            'alpha_validation.min_factor_adjusted_alpha': (results.get('alpha_validation', {}).get('min_factor_adjusted_alpha'), c.min_factor_adjusted_alpha, 'min'),
            'alpha_validation.alpha_t_stat_threshold':    (results.get('alpha_validation', {}).get('alpha_t_stat_threshold'),    c.alpha_t_stat_threshold,    'min'),
            'alpha_validation.min_regime_consistency':    (results.get('alpha_validation', {}).get('min_regime_consistency'),    c.min_regime_consistency,    'min'),

            'cross_market_validation.min_success_rate':       (results.get('cross_market_validation', {}).get('min_success_rate'),       c.min_success_rate,       'min'),
            'cross_market_validation.min_universes_validated':(results.get('cross_market_validation', {}).get('min_universes_validated'),c.min_universes_validated,'min'),

            'execution_quality.max_slippage_variance': (results.get('execution_quality', {}).get('max_slippage_variance'), c.max_slippage_variance, 'max'),
            'execution_quality.min_fill_rate':         (results.get('execution_quality', {}).get('min_fill_rate'),         c.min_fill_rate,         'min'),

            'capital_efficiency.min_return_on_capital': (results.get('capital_efficiency', {}).get('min_return_on_capital'), c.min_return_on_capital, 'min'),
            'capital_efficiency.max_margin_calls':      (results.get('capital_efficiency', {}).get('max_margin_calls'),      c.max_margin_calls,      'max'),
        }

        eval_map, passed, total = {}, 0, 0
        for key, (val, thr, mode) in checks.items():
            total += 1
            ok = False
            if val is None:
                ok = False
            elif mode == 'min':
                ok = val >= thr
            else:
                ok = val <= thr
            if ok: 
                passed += 1
            eval_map[key] = {'actual': val, 'threshold': thr, 'passed': ok}

        return {
            'overall_recommendation': 'GO' if passed == total else 'NO-GO',
            'deployment_readiness_score': passed / max(total, 1),
            'detailed_evaluation': eval_map,
            'failing_criteria': [k for k, v in eval_map.items() if not v['passed']]
        }


def _test_gate():
    """Mini-test for the validation gate."""
    gate = AlphaValidationGate()
    results = {
        'risk_adjusted_returns': {'min_sharpe_ratio':1.3,'max_drawdown':0.12,'min_win_rate':0.50},
        'alpha_validation': {'min_factor_adjusted_alpha':0.06,'alpha_t_stat_threshold':2.5,'min_regime_consistency':0.75},
        'cross_market_validation': {'min_success_rate':0.7,'min_universes_validated':2},
        'execution_quality': {'max_slippage_variance':0.08,'min_fill_rate':0.98},
        'capital_efficiency': {'min_return_on_capital':0.20,'max_margin_calls':0},
    }
    out = gate.evaluate_go_no_go(results)
    if out['overall_recommendation'] != 'GO':
        raise ValueError(f"Expected GO recommendation, got {out['overall_recommendation']}")
    return out


if __name__ == "__main__":
    _test_gate()
