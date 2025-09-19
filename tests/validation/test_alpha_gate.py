"""Tests for Alpha Validation Gate."""

import pytest
from backend.validation.alpha_validation_gate import AlphaValidationGate, ValidationCriteria


def test_gate_pass():
    """Test validation gate with passing criteria."""
    gate = AlphaValidationGate()
    res = gate.evaluate_go_no_go({
        'risk_adjusted_returns': {'min_sharpe_ratio':1.2,'max_drawdown':0.1,'min_win_rate':0.5},
        'alpha_validation': {'min_factor_adjusted_alpha':0.06,'alpha_t_stat_threshold':2.1,'min_regime_consistency':0.8},
        'cross_market_validation': {'min_success_rate':0.7,'min_universes_validated':2},
        'execution_quality': {'max_slippage_variance':0.05,'min_fill_rate':0.97},
        'capital_efficiency': {'min_return_on_capital':0.2,'max_margin_calls':0},
    })
    assert res['overall_recommendation']=='GO'
    assert res['deployment_readiness_score'] == 1.0


def test_gate_fail():
    """Test validation gate with failing criteria."""
    gate = AlphaValidationGate()
    res = gate.evaluate_go_no_go({
        'risk_adjusted_returns': {'min_sharpe_ratio':0.8,'max_drawdown':0.2,'min_win_rate':0.3},
        'alpha_validation': {'min_factor_adjusted_alpha':0.02,'alpha_t_stat_threshold':1.5,'min_regime_consistency':0.5},
        'cross_market_validation': {'min_success_rate':0.4,'min_universes_validated':1},
        'execution_quality': {'max_slippage_variance':0.15,'min_fill_rate':0.9},
        'capital_efficiency': {'min_return_on_capital':0.1,'max_margin_calls':2},
    })
    assert res['overall_recommendation']=='NO-GO'
    assert res['deployment_readiness_score'] < 1.0
    assert len(res['failing_criteria']) > 0


def test_custom_criteria():
    """Test validation gate with custom criteria."""
    custom_criteria = ValidationCriteria(
        min_sharpe_ratio=2.0,
        max_drawdown=0.1,
        min_win_rate=0.6
    )
    gate = AlphaValidationGate(custom_criteria)
    
    # Test with borderline results
    res = gate.evaluate_go_no_go({
        'risk_adjusted_returns': {'min_sharpe_ratio':1.5,'max_drawdown':0.12,'min_win_rate':0.55},
        'alpha_validation': {'min_factor_adjusted_alpha':0.06,'alpha_t_stat_threshold':2.1,'min_regime_consistency':0.8},
        'cross_market_validation': {'min_success_rate':0.7,'min_universes_validated':2},
        'execution_quality': {'max_slippage_variance':0.05,'min_fill_rate':0.97},
        'capital_efficiency': {'min_return_on_capital':0.2,'max_margin_calls':0},
    })
    
    # Should fail due to custom higher thresholds
    assert res['overall_recommendation']=='NO-GO'


def test_missing_data():
    """Test validation gate with missing data."""
    gate = AlphaValidationGate()
    res = gate.evaluate_go_no_go({
        'risk_adjusted_returns': {'min_sharpe_ratio':1.2},  # Missing other fields
        'alpha_validation': {},  # Empty
        # Missing other categories entirely
    })
    
    assert res['overall_recommendation']=='NO-GO'
    assert res['deployment_readiness_score'] < 1.0
    assert len(res['failing_criteria']) > 0


def test_edge_cases():
    """Test edge cases in validation gate."""
    gate = AlphaValidationGate()
    
    # Test with None values
    res = gate.evaluate_go_no_go({
        'risk_adjusted_returns': {'min_sharpe_ratio':None,'max_drawdown':0.1,'min_win_rate':0.5},
        'alpha_validation': {'min_factor_adjusted_alpha':0.06,'alpha_t_stat_threshold':2.1,'min_regime_consistency':0.8},
        'cross_market_validation': {'min_success_rate':0.7,'min_universes_validated':2},
        'execution_quality': {'max_slippage_variance':0.05,'min_fill_rate':0.97},
        'capital_efficiency': {'min_return_on_capital':0.2,'max_margin_calls':0},
    })
    
    assert res['overall_recommendation']=='NO-GO'
    assert 'risk_adjusted_returns.min_sharpe_ratio' in res['failing_criteria']



