#!/usr/bin/env python3
"""
Comprehensive tests for AlphaValidationGate to improve coverage.
Tests edge cases, error handling, and advanced scenarios.
"""

import pytest
from unittest.mock import Mock, patch
from backend.validation.alpha_validation_gate import AlphaValidationGate, ValidationCriteria


class TestAlphaValidationGateComprehensive:
    """Comprehensive tests for AlphaValidationGate."""

    def test_custom_criteria_validation(self):
        """Test validation with custom criteria."""
        custom_criteria = ValidationCriteria(
            min_sharpe_ratio=2.0,
            max_drawdown=0.10,
            min_win_rate=0.60,
            min_factor_adjusted_alpha=0.08,
            alpha_t_stat_threshold=3.0,
            min_regime_consistency=0.80,
            min_success_rate=0.70,
            min_universes_validated=3,
            max_slippage_variance=0.05,
            min_fill_rate=0.98,
            min_return_on_capital=0.20,
            max_margin_calls=0
        )
        
        gate = AlphaValidationGate(custom_criteria)
        
        # Test data that passes custom criteria
        results = {
            'risk_adjusted_returns': {
                'min_sharpe_ratio': 2.5,
                'max_drawdown': 0.08,
                'min_win_rate': 0.65
            },
            'alpha_validation': {
                'min_factor_adjusted_alpha': 0.10,
                'alpha_t_stat_threshold': 3.2,
                'min_regime_consistency': 0.85
            },
            'cross_market_validation': {
                'min_success_rate': 0.75,
                'min_universes_validated': 4
            },
            'execution_quality': {
                'max_slippage_variance': 0.03,
                'min_fill_rate': 0.99
            },
            'capital_efficiency': {
                'min_return_on_capital': 0.25,
                'max_margin_calls': 0
            }
        }
        
        result = gate.evaluate_go_no_go(results)
        
        assert result['overall_recommendation'] == 'GO'
        assert result['deployment_readiness_score'] == 1.0
        assert len(result['failing_criteria']) == 0

    def test_partial_failure_scenarios(self):
        """Test scenarios with partial failures."""
        gate = AlphaValidationGate()
        
        # Test with some criteria failing
        results = {
            'risk_adjusted_returns': {
                'min_sharpe_ratio': 0.8,  # Fails (needs 1.0)
                'max_drawdown': 0.20,     # Fails (needs <= 0.15)
                'min_win_rate': 0.50      # Passes
            },
            'alpha_validation': {
                'min_factor_adjusted_alpha': 0.06,  # Passes
                'alpha_t_stat_threshold': 1.5,      # Fails (needs >= 2.0)
                'min_regime_consistency': 0.75     # Passes
            },
            'cross_market_validation': {
                'min_success_rate': 0.65,       # Passes
                'min_universes_validated': 3    # Passes
            },
            'execution_quality': {
                'max_slippage_variance': 0.08,  # Passes
                'min_fill_rate': 0.96          # Passes
            },
            'capital_efficiency': {
                'min_return_on_capital': 0.18,  # Passes
                'max_margin_calls': 1           # Fails (needs <= 0)
            }
        }
        
        result = gate.evaluate_go_no_go(results)
        
        assert result['overall_recommendation'] == 'NO-GO'
        assert result['deployment_readiness_score'] < 1.0
        assert len(result['failing_criteria']) > 0
        assert 'risk_adjusted_returns.min_sharpe_ratio' in result['failing_criteria']
        assert 'risk_adjusted_returns.max_drawdown' in result['failing_criteria']
        assert 'alpha_validation.alpha_t_stat_threshold' in result['failing_criteria']
        assert 'capital_efficiency.max_margin_calls' in result['failing_criteria']

    def test_missing_data_handling(self):
        """Test handling of missing data sections."""
        gate = AlphaValidationGate()
        
        # Test with completely missing sections
        results = {}
        
        result = gate.evaluate_go_no_go(results)
        
        assert result['overall_recommendation'] == 'NO-GO'
        assert result['deployment_readiness_score'] == 0.0
        assert len(result['failing_criteria']) == 12  # All criteria fail

    def test_partial_missing_data(self):
        """Test handling of partially missing data."""
        gate = AlphaValidationGate()
        
        # Test with some sections missing
        results = {
            'risk_adjusted_returns': {
                'min_sharpe_ratio': 1.5,
                'max_drawdown': 0.10,
                'min_win_rate': 0.55
            },
            # Missing alpha_validation section
            'cross_market_validation': {
                'min_success_rate': 0.70,
                'min_universes_validated': 2
            }
            # Missing execution_quality and capital_efficiency sections
        }
        
        result = gate.evaluate_go_no_go(results)
        
        assert result['overall_recommendation'] == 'NO-GO'
        assert result['deployment_readiness_score'] < 1.0
        # Should have failures for missing sections
        assert any('alpha_validation' in criteria for criteria in result['failing_criteria'])
        assert any('execution_quality' in criteria for criteria in result['failing_criteria'])
        assert any('capital_efficiency' in criteria for criteria in result['failing_criteria'])

    def test_edge_case_values(self):
        """Test edge case values at boundaries."""
        gate = AlphaValidationGate()
        
        # Test values exactly at thresholds
        results = {
            'risk_adjusted_returns': {
                'min_sharpe_ratio': 1.0,      # Exactly at threshold
                'max_drawdown': 0.15,         # Exactly at threshold
                'min_win_rate': 0.45          # Exactly at threshold
            },
            'alpha_validation': {
                'min_factor_adjusted_alpha': 0.05,  # Exactly at threshold
                'alpha_t_stat_threshold': 2.0,       # Exactly at threshold
                'min_regime_consistency': 0.70       # Exactly at threshold
            },
            'cross_market_validation': {
                'min_success_rate': 0.60,           # Exactly at threshold
                'min_universes_validated': 2        # Exactly at threshold
            },
            'execution_quality': {
                'max_slippage_variance': 0.10,      # Exactly at threshold
                'min_fill_rate': 0.95               # Exactly at threshold
            },
            'capital_efficiency': {
                'min_return_on_capital': 0.15,     # Exactly at threshold
                'max_margin_calls': 0              # Exactly at threshold
            }
        }
        
        result = gate.evaluate_go_no_go(results)
        
        assert result['overall_recommendation'] == 'GO'
        assert result['deployment_readiness_score'] == 1.0
        assert len(result['failing_criteria']) == 0

    def test_boundary_failure_values(self):
        """Test values just below thresholds."""
        gate = AlphaValidationGate()
        
        # Test values just below thresholds
        results = {
            'risk_adjusted_returns': {
                'min_sharpe_ratio': 0.99,     # Just below threshold
                'max_drawdown': 0.151,        # Just above threshold (should fail)
                'min_win_rate': 0.449         # Just below threshold
            },
            'alpha_validation': {
                'min_factor_adjusted_alpha': 0.049,  # Just below threshold
                'alpha_t_stat_threshold': 1.99,       # Just below threshold
                'min_regime_consistency': 0.699      # Just below threshold
            },
            'cross_market_validation': {
                'min_success_rate': 0.599,           # Just below threshold
                'min_universes_validated': 1         # Just below threshold
            },
            'execution_quality': {
                'max_slippage_variance': 0.101,     # Just above threshold (should fail)
                'min_fill_rate': 0.949              # Just below threshold
            },
            'capital_efficiency': {
                'min_return_on_capital': 0.149,     # Just below threshold
                'max_margin_calls': 1               # Above threshold (should fail)
            }
        }
        
        result = gate.evaluate_go_no_go(results)
        
        assert result['overall_recommendation'] == 'NO-GO'
        assert result['deployment_readiness_score'] < 1.0
        assert len(result['failing_criteria']) > 0

    def test_none_values_handling(self):
        """Test handling of None values in results."""
        gate = AlphaValidationGate()
        
        results = {
            'risk_adjusted_returns': {
                'min_sharpe_ratio': None,     # None value
                'max_drawdown': 0.10,
                'min_win_rate': 0.55
            },
            'alpha_validation': {
                'min_factor_adjusted_alpha': 0.06,
                'alpha_t_stat_threshold': None,  # None value
                'min_regime_consistency': 0.75
            },
            'cross_market_validation': {
                'min_success_rate': 0.70,
                'min_universes_validated': 2
            },
            'execution_quality': {
                'max_slippage_variance': 0.08,
                'min_fill_rate': 0.96
            },
            'capital_efficiency': {
                'min_return_on_capital': 0.18,
                'max_margin_calls': 0
            }
        }
        
        result = gate.evaluate_go_no_go(results)
        
        assert result['overall_recommendation'] == 'NO-GO'
        assert result['deployment_readiness_score'] < 1.0
        assert 'risk_adjusted_returns.min_sharpe_ratio' in result['failing_criteria']
        assert 'alpha_validation.alpha_t_stat_threshold' in result['failing_criteria']

    def test_detailed_evaluation_structure(self):
        """Test the structure of detailed evaluation."""
        gate = AlphaValidationGate()
        
        results = {
            'risk_adjusted_returns': {
                'min_sharpe_ratio': 1.5,
                'max_drawdown': 0.10,
                'min_win_rate': 0.55
            },
            'alpha_validation': {
                'min_factor_adjusted_alpha': 0.06,
                'alpha_t_stat_threshold': 2.5,
                'min_regime_consistency': 0.75
            },
            'cross_market_validation': {
                'min_success_rate': 0.70,
                'min_universes_validated': 2
            },
            'execution_quality': {
                'max_slippage_variance': 0.08,
                'min_fill_rate': 0.96
            },
            'capital_efficiency': {
                'min_return_on_capital': 0.18,
                'max_margin_calls': 0
            }
        }
        
        result = gate.evaluate_go_no_go(results)
        
        # Test structure of detailed evaluation
        assert 'detailed_evaluation' in result
        detailed = result['detailed_evaluation']
        
        # Check that all criteria are present
        expected_keys = [
            'risk_adjusted_returns.min_sharpe_ratio',
            'risk_adjusted_returns.max_drawdown',
            'risk_adjusted_returns.min_win_rate',
            'alpha_validation.min_factor_adjusted_alpha',
            'alpha_validation.alpha_t_stat_threshold',
            'alpha_validation.min_regime_consistency',
            'cross_market_validation.min_success_rate',
            'cross_market_validation.min_universes_validated',
            'execution_quality.max_slippage_variance',
            'execution_quality.min_fill_rate',
            'capital_efficiency.min_return_on_capital',
            'capital_efficiency.max_margin_calls'
        ]
        
        for key in expected_keys:
            assert key in detailed
            assert 'actual' in detailed[key]
            assert 'threshold' in detailed[key]
            assert 'passed' in detailed[key]
            assert isinstance(detailed[key]['passed'], bool)

    def test_deployment_readiness_score_calculation(self):
        """Test deployment readiness score calculation."""
        gate = AlphaValidationGate()
        
        # Test with exactly half criteria passing
        results = {
            'risk_adjusted_returns': {
                'min_sharpe_ratio': 1.5,      # Pass
                'max_drawdown': 0.20,         # Fail
                'min_win_rate': 0.55          # Pass
            },
            'alpha_validation': {
                'min_factor_adjusted_alpha': 0.06,  # Pass
                'alpha_t_stat_threshold': 1.5,      # Fail
                'min_regime_consistency': 0.75      # Pass
            },
            'cross_market_validation': {
                'min_success_rate': 0.70,       # Pass
                'min_universes_validated': 1    # Fail
            },
            'execution_quality': {
                'max_slippage_variance': 0.08,  # Pass
                'min_fill_rate': 0.96          # Pass
            },
            'capital_efficiency': {
                'min_return_on_capital': 0.18,  # Pass
                'max_margin_calls': 1           # Fail
            }
        }
        
        result = gate.evaluate_go_no_go(results)
        
        # Count actual passes: 1.5 (pass), 0.20 (fail), 0.55 (pass), 0.06 (pass), 1.5 (fail), 0.75 (pass), 0.70 (pass), 1 (fail), 0.08 (pass), 0.96 (pass), 0.18 (pass), 1 (fail)
        # Passes: 1.5, 0.55, 0.06, 0.75, 0.70, 0.08, 0.96, 0.18 = 8 passes
        # Failures: 0.20, 1.5, 1, 1 = 4 failures
        # Total: 12 criteria, 8 passes, 4 failures
        expected_score = 8 / 12
        assert abs(result['deployment_readiness_score'] - expected_score) < 0.001

    def test_criteria_immutability(self):
        """Test that ValidationCriteria is immutable."""
        criteria = ValidationCriteria()
        
        # Test that criteria cannot be modified after creation
        with pytest.raises(AttributeError):
            criteria.min_sharpe_ratio = 2.0

    def test_gate_with_none_criteria(self):
        """Test gate initialization with None criteria."""
        gate = AlphaValidationGate(None)
        
        # Should use default criteria
        assert gate.criteria is not None
        assert isinstance(gate.criteria, ValidationCriteria)
        
        # Test that it works normally
        results = {
            'risk_adjusted_returns': {
                'min_sharpe_ratio': 1.5,
                'max_drawdown': 0.10,
                'min_win_rate': 0.55
            },
            'alpha_validation': {
                'min_factor_adjusted_alpha': 0.06,
                'alpha_t_stat_threshold': 2.5,
                'min_regime_consistency': 0.75
            },
            'cross_market_validation': {
                'min_success_rate': 0.70,
                'min_universes_validated': 2
            },
            'execution_quality': {
                'max_slippage_variance': 0.08,
                'min_fill_rate': 0.96
            },
            'capital_efficiency': {
                'min_return_on_capital': 0.18,
                'max_margin_calls': 0
            }
        }
        
        result = gate.evaluate_go_no_go(results)
        assert result['overall_recommendation'] == 'GO'
