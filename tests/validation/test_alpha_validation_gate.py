"""
Comprehensive Tests for AlphaValidationGate
==========================================

Edge cases, error handling, and boundary condition testing.
"""

import pytest
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import Dict, Any

from backend.validation.alpha_validation_gate import (
    AlphaValidationGate,
    ValidationCriteria
)


class TestValidationCriteria:
    """Test ValidationCriteria dataclass behavior."""

    def test_default_criteria_creation(self):
        """Test default criteria values."""
        criteria = ValidationCriteria()
        assert criteria.min_sharpe_ratio == 1.0
        assert criteria.max_drawdown == 0.15
        assert criteria.min_win_rate == 0.45
        assert criteria.min_factor_adjusted_alpha == 0.05
        assert criteria.alpha_t_stat_threshold == 2.0
        assert criteria.min_regime_consistency == 0.70
        assert criteria.min_success_rate == 0.60
        assert criteria.min_universes_validated == 2
        assert criteria.max_slippage_variance == 0.10
        assert criteria.min_fill_rate == 0.95
        assert criteria.min_return_on_capital == 0.15
        assert criteria.max_margin_calls == 0

    def test_custom_criteria_creation(self):
        """Test custom criteria values."""
        criteria = ValidationCriteria(
            min_sharpe_ratio=1.5,
            max_drawdown=0.10,
            min_win_rate=0.50,
            min_factor_adjusted_alpha=0.08,
            alpha_t_stat_threshold=2.5,
            min_regime_consistency=0.80,
            min_success_rate=0.70,
            min_universes_validated=3,
            max_slippage_variance=0.05,
            min_fill_rate=0.98,
            min_return_on_capital=0.20,
            max_margin_calls=1
        )
        assert criteria.min_sharpe_ratio == 1.5
        assert criteria.max_drawdown == 0.10
        assert criteria.min_win_rate == 0.50
        assert criteria.min_factor_adjusted_alpha == 0.08
        assert criteria.alpha_t_stat_threshold == 2.5
        assert criteria.min_regime_consistency == 0.80
        assert criteria.min_success_rate == 0.70
        assert criteria.min_universes_validated == 3
        assert criteria.max_slippage_variance == 0.05
        assert criteria.min_fill_rate == 0.98
        assert criteria.min_return_on_capital == 0.20
        assert criteria.max_margin_calls == 1

    def test_criteria_immutability(self):
        """Test that ValidationCriteria is frozen/immutable."""
        criteria = ValidationCriteria()
        with pytest.raises(AttributeError):  # FrozenInstanceError in Python 3.7+
            criteria.min_sharpe_ratio = 2.0

    def test_extreme_criteria_values(self):
        """Test criteria with extreme boundary values."""
        criteria = ValidationCriteria(
            min_sharpe_ratio=0.0,
            max_drawdown=1.0,
            min_win_rate=0.0,
            min_factor_adjusted_alpha=-0.10,
            alpha_t_stat_threshold=0.0,
            min_regime_consistency=0.0,
            min_success_rate=0.0,
            min_universes_validated=0,
            max_slippage_variance=1.0,
            min_fill_rate=0.0,
            min_return_on_capital=-0.50,
            max_margin_calls=1000
        )
        assert criteria.min_sharpe_ratio == 0.0
        assert criteria.max_drawdown == 1.0
        assert criteria.min_factor_adjusted_alpha == -0.10


class TestAlphaValidationGateInitialization:
    """Test AlphaValidationGate initialization."""

    def test_default_initialization(self):
        """Test initialization with default criteria."""
        gate = AlphaValidationGate()
        assert isinstance(gate.criteria, ValidationCriteria)
        assert gate.criteria.min_sharpe_ratio == 1.0

    def test_custom_criteria_initialization(self):
        """Test initialization with custom criteria."""
        custom_criteria = ValidationCriteria(min_sharpe_ratio=1.5)
        gate = AlphaValidationGate(criteria=custom_criteria)
        assert gate.criteria.min_sharpe_ratio == 1.5

    def test_none_criteria_initialization(self):
        """Test initialization with None criteria."""
        gate = AlphaValidationGate(criteria=None)
        assert isinstance(gate.criteria, ValidationCriteria)
        assert gate.criteria.min_sharpe_ratio == 1.0


class TestAlphaValidationGateBasicFunctionality:
    """Test basic functionality of AlphaValidationGate."""

    def test_perfect_results_go_decision(self):
        """Test GO decision with perfect results."""
        gate = AlphaValidationGate()
        results = {
            'risk_adjusted_returns': {
                'min_sharpe_ratio': 2.0,
                'max_drawdown': 0.10,
                'min_win_rate': 0.60
            },
            'alpha_validation': {
                'min_factor_adjusted_alpha': 0.10,
                'alpha_t_stat_threshold': 3.0,
                'min_regime_consistency': 0.80
            },
            'cross_market_validation': {
                'min_success_rate': 0.70,
                'min_universes_validated': 3
            },
            'execution_quality': {
                'max_slippage_variance': 0.05,
                'min_fill_rate': 0.98
            },
            'capital_efficiency': {
                'min_return_on_capital': 0.20,
                'max_margin_calls': 0
            }
        }

        decision = gate.evaluate_go_no_go(results)
        assert decision['overall_recommendation'] == 'GO'
        assert decision['deployment_readiness_score'] == 1.0
        assert len(decision['failing_criteria']) == 0

    def test_failing_results_no_go_decision(self):
        """Test NO-GO decision with failing results."""
        gate = AlphaValidationGate()
        results = {
            'risk_adjusted_returns': {
                'min_sharpe_ratio': 0.5,  # Below 1.0 threshold
                'max_drawdown': 0.25,     # Above 0.15 threshold
                'min_win_rate': 0.30      # Below 0.45 threshold
            },
            'alpha_validation': {
                'min_factor_adjusted_alpha': 0.02,  # Below 0.05 threshold
                'alpha_t_stat_threshold': 1.0,      # Below 2.0 threshold
                'min_regime_consistency': 0.50      # Below 0.70 threshold
            },
            'cross_market_validation': {
                'min_success_rate': 0.40,           # Below 0.60 threshold
                'min_universes_validated': 1        # Below 2 threshold
            },
            'execution_quality': {
                'max_slippage_variance': 0.20,      # Above 0.10 threshold
                'min_fill_rate': 0.85                # Below 0.95 threshold
            },
            'capital_efficiency': {
                'min_return_on_capital': 0.05,      # Below 0.15 threshold
                'max_margin_calls': 5                # Above 0 threshold
            }
        }

        decision = gate.evaluate_go_no_go(results)
        assert decision['overall_recommendation'] == 'NO-GO'
        assert decision['deployment_readiness_score'] == 0.0
        assert len(decision['failing_criteria']) == 12  # All criteria fail

    def test_mixed_results_partial_pass(self):
        """Test mixed results with some passing criteria."""
        gate = AlphaValidationGate()
        results = {
            'risk_adjusted_returns': {
                'min_sharpe_ratio': 1.5,   # PASS
                'max_drawdown': 0.25,      # FAIL
                'min_win_rate': 0.50       # PASS
            },
            'alpha_validation': {
                'min_factor_adjusted_alpha': 0.08,  # PASS
                'alpha_t_stat_threshold': 1.5,      # FAIL
                'min_regime_consistency': 0.75      # PASS
            },
            'cross_market_validation': {
                'min_success_rate': 0.65,           # PASS
                'min_universes_validated': 1        # FAIL
            },
            'execution_quality': {
                'max_slippage_variance': 0.08,      # PASS
                'min_fill_rate': 0.98                # PASS
            },
            'capital_efficiency': {
                'min_return_on_capital': 0.18,      # PASS
                'max_margin_calls': 2                # FAIL
            }
        }

        decision = gate.evaluate_go_no_go(results)
        assert decision['overall_recommendation'] == 'NO-GO'
        assert 0.0 < decision['deployment_readiness_score'] < 1.0
        assert len(decision['failing_criteria']) == 4


class TestAlphaValidationGateEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_results_dictionary(self):
        """Test with completely empty results."""
        gate = AlphaValidationGate()
        results = {}

        decision = gate.evaluate_go_no_go(results)
        assert decision['overall_recommendation'] == 'NO-GO'
        assert decision['deployment_readiness_score'] == 0.0
        assert len(decision['failing_criteria']) == 12  # All criteria fail

    def test_partial_empty_results(self):
        """Test with partially empty results."""
        gate = AlphaValidationGate()
        results = {
            'risk_adjusted_returns': {
                'min_sharpe_ratio': 1.5
                # Missing max_drawdown and min_win_rate
            },
            'alpha_validation': {},  # Empty section
            # Missing other sections
        }

        decision = gate.evaluate_go_no_go(results)
        assert decision['overall_recommendation'] == 'NO-GO'
        assert decision['deployment_readiness_score'] < 1.0
        assert len(decision['failing_criteria']) > 0

    def test_none_values_in_results(self):
        """Test with None values in results."""
        gate = AlphaValidationGate()
        results = {
            'risk_adjusted_returns': {
                'min_sharpe_ratio': None,
                'max_drawdown': None,
                'min_win_rate': None
            },
            'alpha_validation': {
                'min_factor_adjusted_alpha': None,
                'alpha_t_stat_threshold': None,
                'min_regime_consistency': None
            },
            'cross_market_validation': {
                'min_success_rate': None,
                'min_universes_validated': None
            },
            'execution_quality': {
                'max_slippage_variance': None,
                'min_fill_rate': None
            },
            'capital_efficiency': {
                'min_return_on_capital': None,
                'max_margin_calls': None
            }
        }

        decision = gate.evaluate_go_no_go(results)
        assert decision['overall_recommendation'] == 'NO-GO'
        assert decision['deployment_readiness_score'] == 0.0
        assert len(decision['failing_criteria']) == 12

    def test_boundary_value_exact_thresholds(self):
        """Test exact threshold boundary values."""
        gate = AlphaValidationGate()
        criteria = ValidationCriteria()

        # Test exact threshold values (should PASS for min, PASS for max)
        results = {
            'risk_adjusted_returns': {
                'min_sharpe_ratio': criteria.min_sharpe_ratio,      # Exactly 1.0 - PASS
                'max_drawdown': criteria.max_drawdown,              # Exactly 0.15 - PASS
                'min_win_rate': criteria.min_win_rate               # Exactly 0.45 - PASS
            },
            'alpha_validation': {
                'min_factor_adjusted_alpha': criteria.min_factor_adjusted_alpha,  # Exactly 0.05 - PASS
                'alpha_t_stat_threshold': criteria.alpha_t_stat_threshold,        # Exactly 2.0 - PASS
                'min_regime_consistency': criteria.min_regime_consistency         # Exactly 0.70 - PASS
            },
            'cross_market_validation': {
                'min_success_rate': criteria.min_success_rate,                    # Exactly 0.60 - PASS
                'min_universes_validated': criteria.min_universes_validated       # Exactly 2 - PASS
            },
            'execution_quality': {
                'max_slippage_variance': criteria.max_slippage_variance,          # Exactly 0.10 - PASS
                'min_fill_rate': criteria.min_fill_rate                           # Exactly 0.95 - PASS
            },
            'capital_efficiency': {
                'min_return_on_capital': criteria.min_return_on_capital,          # Exactly 0.15 - PASS
                'max_margin_calls': criteria.max_margin_calls                     # Exactly 0 - PASS
            }
        }

        decision = gate.evaluate_go_no_go(results)
        assert decision['overall_recommendation'] == 'GO'
        assert decision['deployment_readiness_score'] == 1.0
        assert len(decision['failing_criteria']) == 0

    def test_boundary_value_just_below_thresholds(self):
        """Test values just below thresholds."""
        gate = AlphaValidationGate()
        epsilon = 1e-10

        results = {
            'risk_adjusted_returns': {
                'min_sharpe_ratio': 1.0 - epsilon,    # Just below 1.0 - FAIL
                'max_drawdown': 0.15 + epsilon,       # Just above 0.15 - FAIL
                'min_win_rate': 0.45 - epsilon        # Just below 0.45 - FAIL
            },
            'alpha_validation': {
                'min_factor_adjusted_alpha': 0.05 - epsilon,  # Just below 0.05 - FAIL
                'alpha_t_stat_threshold': 2.0 - epsilon,      # Just below 2.0 - FAIL
                'min_regime_consistency': 0.70 - epsilon      # Just below 0.70 - FAIL
            },
            'cross_market_validation': {
                'min_success_rate': 0.60 - epsilon,           # Just below 0.60 - FAIL
                'min_universes_validated': 1                  # Below 2 - FAIL
            },
            'execution_quality': {
                'max_slippage_variance': 0.10 + epsilon,      # Just above 0.10 - FAIL
                'min_fill_rate': 0.95 - epsilon               # Just below 0.95 - FAIL
            },
            'capital_efficiency': {
                'min_return_on_capital': 0.15 - epsilon,      # Just below 0.15 - FAIL
                'max_margin_calls': 1                         # Above 0 - FAIL
            }
        }

        decision = gate.evaluate_go_no_go(results)
        assert decision['overall_recommendation'] == 'NO-GO'
        assert decision['deployment_readiness_score'] == 0.0
        assert len(decision['failing_criteria']) == 12

    def test_infinity_and_large_values(self):
        """Test handling of infinity and very large values."""
        gate = AlphaValidationGate()
        results = {
            'risk_adjusted_returns': {
                'min_sharpe_ratio': float('inf'),    # Should PASS
                'max_drawdown': float('inf'),         # Should FAIL
                'min_win_rate': 1e10                  # Should PASS
            },
            'alpha_validation': {
                'min_factor_adjusted_alpha': 1e6,    # Should PASS
                'alpha_t_stat_threshold': float('inf'),  # Should PASS
                'min_regime_consistency': 2.0        # Should PASS (clamped at 1.0 is fine)
            },
            'cross_market_validation': {
                'min_success_rate': 1e3,             # Should PASS
                'min_universes_validated': 1000      # Should PASS
            },
            'execution_quality': {
                'max_slippage_variance': 1e-10,      # Should PASS
                'min_fill_rate': 1.5                 # Should PASS (> 0.95)
            },
            'capital_efficiency': {
                'min_return_on_capital': 10.0,       # Should PASS
                'max_margin_calls': -1               # Should PASS (< 0)
            }
        }

        decision = gate.evaluate_go_no_go(results)
        # Most should pass except max_drawdown
        assert decision['deployment_readiness_score'] > 0.9

    def test_negative_values(self):
        """Test handling of negative values."""
        gate = AlphaValidationGate()
        results = {
            'risk_adjusted_returns': {
                'min_sharpe_ratio': -1.0,            # Should FAIL
                'max_drawdown': -0.05,               # Should PASS (negative drawdown is good)
                'min_win_rate': -0.10                # Should FAIL
            },
            'alpha_validation': {
                'min_factor_adjusted_alpha': -0.02,  # Should FAIL
                'alpha_t_stat_threshold': -1.0,      # Should FAIL
                'min_regime_consistency': -0.20      # Should FAIL
            },
            'cross_market_validation': {
                'min_success_rate': -0.30,           # Should FAIL
                'min_universes_validated': -1        # Should FAIL
            },
            'execution_quality': {
                'max_slippage_variance': -0.01,      # Should PASS (negative variance is good)
                'min_fill_rate': -0.50               # Should FAIL
            },
            'capital_efficiency': {
                'min_return_on_capital': -0.20,      # Should FAIL
                'max_margin_calls': -1               # Should PASS (negative margin calls is good)
            }
        }

        decision = gate.evaluate_go_no_go(results)
        # Should have several failures
        assert decision['deployment_readiness_score'] < 0.5


class TestAlphaValidationGateErrorHandling:
    """Test error handling scenarios."""

    def test_malformed_results_structure(self):
        """Test with malformed results structure."""
        gate = AlphaValidationGate()

        # Test with non-dictionary results
        with pytest.raises((AttributeError, TypeError)):
            gate.evaluate_go_no_go("invalid_results")

        with pytest.raises((AttributeError, TypeError)):
            gate.evaluate_go_no_go(None)

        with pytest.raises((AttributeError, TypeError)):
            gate.evaluate_go_no_go(123)

    def test_nested_structure_corruption(self):
        """Test with corrupted nested structure."""
        gate = AlphaValidationGate()

        # Test with non-dictionary nested values
        results = {
            'risk_adjusted_returns': "not_a_dict",
            'alpha_validation': 123,
            'cross_market_validation': [],
            'execution_quality': None,
            'capital_efficiency': {'min_return_on_capital': 0.20}
        }

        try:
            decision = gate.evaluate_go_no_go(results)
            # Should handle gracefully, treating corrupted sections as missing data
            assert decision['overall_recommendation'] == 'NO-GO'
            assert decision['deployment_readiness_score'] < 1.0
        except (AttributeError, TypeError):
            # Also acceptable to raise an error
            pass

    def test_custom_criteria_edge_cases(self):
        """Test edge cases with custom criteria."""
        # Test with all zero thresholds
        zero_criteria = ValidationCriteria(
            min_sharpe_ratio=0.0,
            max_drawdown=0.0,
            min_win_rate=0.0,
            min_factor_adjusted_alpha=0.0,
            alpha_t_stat_threshold=0.0,
            min_regime_consistency=0.0,
            min_success_rate=0.0,
            min_universes_validated=0,
            max_slippage_variance=0.0,
            min_fill_rate=0.0,
            min_return_on_capital=0.0,
            max_margin_calls=0
        )

        gate = AlphaValidationGate(criteria=zero_criteria)

        # Even poor results should pass with zero thresholds
        poor_results = {
            'risk_adjusted_returns': {
                'min_sharpe_ratio': -1.0,
                'max_drawdown': 0.0,  # Exactly at threshold
                'min_win_rate': -0.5
            },
            'alpha_validation': {
                'min_factor_adjusted_alpha': -0.10,
                'alpha_t_stat_threshold': -2.0,
                'min_regime_consistency': -0.50
            },
            'cross_market_validation': {
                'min_success_rate': -0.30,
                'min_universes_validated': -5
            },
            'execution_quality': {
                'max_slippage_variance': 0.0,  # Exactly at threshold
                'min_fill_rate': -0.20
            },
            'capital_efficiency': {
                'min_return_on_capital': -1.0,
                'max_margin_calls': 0  # Exactly at threshold
            }
        }

        decision = gate.evaluate_go_no_go(poor_results)
        # Should have some passes due to zero/exact thresholds
        assert decision['deployment_readiness_score'] > 0.0


class TestAlphaValidationGateDetailedEvaluation:
    """Test detailed evaluation functionality."""

    def test_detailed_evaluation_structure(self):
        """Test structure of detailed evaluation."""
        gate = AlphaValidationGate()
        results = {
            'risk_adjusted_returns': {
                'min_sharpe_ratio': 1.5,
                'max_drawdown': 0.10,
                'min_win_rate': 0.50
            }
        }

        decision = gate.evaluate_go_no_go(results)

        # Check overall structure
        assert 'overall_recommendation' in decision
        assert 'deployment_readiness_score' in decision
        assert 'detailed_evaluation' in decision
        assert 'failing_criteria' in decision

        # Check detailed evaluation structure
        detailed = decision['detailed_evaluation']
        assert isinstance(detailed, dict)

        # Check each evaluation has proper structure
        for evaluation in detailed.values():
            assert 'actual' in evaluation
            assert 'threshold' in evaluation
            assert 'passed' in evaluation
            assert isinstance(evaluation['passed'], bool)

    def test_failing_criteria_list_accuracy(self):
        """Test accuracy of failing criteria list."""
        gate = AlphaValidationGate()
        results = {
            'risk_adjusted_returns': {
                'min_sharpe_ratio': 0.5,    # FAIL
                'max_drawdown': 0.10,       # PASS
                'min_win_rate': 0.30        # FAIL
            },
            'alpha_validation': {
                'min_factor_adjusted_alpha': 0.08,  # PASS
                'alpha_t_stat_threshold': 1.5,      # FAIL
                'min_regime_consistency': 0.75      # PASS
            }
        }

        decision = gate.evaluate_go_no_go(results)
        failing = decision['failing_criteria']

        # Should have exactly the failing criteria
        expected_failing = [
            'risk_adjusted_returns.min_sharpe_ratio',
            'risk_adjusted_returns.min_win_rate',
            'alpha_validation.alpha_t_stat_threshold'
        ]

        for expected in expected_failing:
            assert expected in failing

        # Should not include passing criteria
        assert 'risk_adjusted_returns.max_drawdown' not in failing
        assert 'alpha_validation.min_factor_adjusted_alpha' not in failing
        assert 'alpha_validation.min_regime_consistency' not in failing

    def test_deployment_readiness_score_calculation(self):
        """Test deployment readiness score calculation."""
        gate = AlphaValidationGate()

        # Test with known pass/fail counts
        results_half_pass = {
            'risk_adjusted_returns': {
                'min_sharpe_ratio': 1.5,    # PASS
                'max_drawdown': 0.25,       # FAIL
                'min_win_rate': 0.50        # PASS
            },
            'alpha_validation': {
                'min_factor_adjusted_alpha': 0.02,  # FAIL
                'alpha_t_stat_threshold': 2.5,      # PASS
                'min_regime_consistency': 0.60      # FAIL
            }
        }

        decision = gate.evaluate_go_no_go(results_half_pass)

        # Count actual passes and fails in detailed evaluation
        detailed = decision['detailed_evaluation']
        total_checks = len(detailed)
        passed_checks = sum(1 for eval_data in detailed.values() if eval_data['passed'])
        expected_score = passed_checks / total_checks

        assert abs(decision['deployment_readiness_score'] - expected_score) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])