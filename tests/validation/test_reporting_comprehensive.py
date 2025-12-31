#!/usr/bin/env python3
"""
Comprehensive tests for reporting module.
Tests all public methods, edge cases, and error handling.
Target: 80%+ coverage
"""

import pytest
import pandas as pd
import numpy as np
import json
import pathlib
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from backend.validation.reporting import ValidationReporter


class TestValidationReporter:
    """Test ValidationReporter class."""

    @pytest.fixture
    def temp_outdir(self, tmp_path):
        """Create temporary output directory."""
        return str(tmp_path / "reports")

    @pytest.fixture
    def reporter(self, temp_outdir):
        """Create reporter instance."""
        return ValidationReporter(outdir=temp_outdir)

    @pytest.fixture
    def sample_equity_data(self):
        """Create sample equity curve data."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        equity = (1 + pd.Series(range(100)) * 0.001).cumprod()
        return pd.Series(equity.values, index=dates)

    @pytest.fixture
    def sample_validation_results(self):
        """Create sample validation results."""
        return {
            'validation_gate': {
                'overall_recommendation': 'GO',
                'deployment_readiness_score': 0.85,
                'detailed_evaluation': {
                    'risk_adjusted_returns.min_sharpe_ratio': {
                        'actual': 1.2,
                        'threshold': 1.0,
                        'passed': True
                    },
                    'alpha_validation.min_factor_adjusted_alpha': {
                        'actual': 0.05,
                        'threshold': 0.05,
                        'passed': True
                    }
                }
            },
            'factor_analysis': {
                'passed': True
            },
            'regime_testing': {
                'passed': True
            }
        }

    def test_initialization_default(self):
        """Test default initialization."""
        reporter = ValidationReporter()

        assert reporter.base.exists()
        assert reporter.artifacts == []

    def test_initialization_custom_outdir(self, temp_outdir):
        """Test initialization with custom output directory."""
        reporter = ValidationReporter(outdir=temp_outdir)

        assert temp_outdir in str(reporter.base)
        assert reporter.base.exists()

    def test_write_json(self, reporter):
        """Test JSON writing."""
        payload = {
            'test_key': 'test_value',
            'numeric': 123,
            'nested': {'inner': 'value'}
        }

        file_path = reporter.write_json('test_output', payload)

        # Verify file exists
        assert pathlib.Path(file_path).exists()

        # Verify content
        with open(file_path, 'r') as f:
            loaded = json.load(f)

        assert loaded['test_key'] == 'test_value'
        assert loaded['numeric'] == 123
        assert file_path in reporter.artifacts

    def test_write_json_with_datetime(self, reporter):
        """Test JSON writing with datetime objects."""
        payload = {
            'timestamp': datetime(2023, 1, 1, 12, 0, 0),
            'data': [1, 2, 3]
        }

        file_path = reporter.write_json('datetime_test', payload)

        # Should convert datetime to string
        with open(file_path, 'r') as f:
            loaded = json.load(f)

        assert '2023' in loaded['timestamp']

    def test_write_csv(self, reporter):
        """Test CSV writing."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })

        file_path = reporter.write_csv('test_data', df)

        # Verify file exists
        assert pathlib.Path(file_path).exists()

        # Verify content
        loaded_df = pd.read_csv(file_path, index_col=0)
        assert len(loaded_df) == 3
        assert 'col1' in loaded_df.columns
        assert file_path in reporter.artifacts

    def test_write_csv_with_index(self, reporter):
        """Test CSV writing with custom index."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5]
        }, index=dates)

        file_path = reporter.write_csv('indexed_data', df)

        loaded_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        assert len(loaded_df) == 5

    def test_write_equity_curve(self, reporter, sample_equity_data):
        """Test equity curve plot generation."""
        file_path = reporter.write_equity_curve('equity_test', sample_equity_data)

        # Verify file exists and is PNG
        assert pathlib.Path(file_path).exists()
        assert file_path.endswith('.png')
        assert file_path in reporter.artifacts

    def test_write_equity_curve_with_benchmark(self, reporter, sample_equity_data):
        """Test equity curve with benchmark."""
        benchmark = sample_equity_data * 0.8

        file_path = reporter.write_equity_curve(
            'equity_with_benchmark',
            sample_equity_data,
            benchmark
        )

        assert pathlib.Path(file_path).exists()

    def test_write_equity_curve_empty_data(self, reporter):
        """Test equity curve with empty data."""
        empty_data = pd.Series(dtype=float)

        # Should handle gracefully
        with pytest.raises(Exception):
            reporter.write_equity_curve('empty_equity', empty_data)

    def test_write_factor_table(self, reporter):
        """Test factor analysis table generation."""
        # Create mock factor results
        class MockResult:
            def __init__(self, alpha, t_stat, r2, n_obs, exposures):
                self.annualized_alpha = alpha
                self.alpha_t_stat = t_stat
                self.r_squared = r2
                self.n_obs = n_obs
                self.factor_exposures = exposures

        factor_results = {
            'strategy1': MockResult(0.06, 2.1, 0.75, 252, {'mkt': 0.8, 'smb': 0.2}),
            'strategy2': MockResult(0.04, 1.5, 0.65, 252, {'mkt': 0.9, 'hml': 0.3})
        }

        file_path = reporter.write_factor_table('factor_table', factor_results)

        # Verify CSV was created
        assert pathlib.Path(file_path).exists()
        assert file_path.endswith('.csv')

        # Load and verify
        df = pd.read_csv(file_path)
        assert len(df) == 2
        assert 'Strategy' in df.columns
        assert 'Alpha (Annualized)' in df.columns

    def test_write_factor_table_empty_results(self, reporter):
        """Test factor table with empty results."""
        file_path = reporter.write_factor_table('empty_factors', {})

        # Should create JSON with error
        assert pathlib.Path(file_path).exists()
        assert file_path.endswith('.json')

    def test_write_factor_table_invalid_results(self, reporter):
        """Test factor table with invalid result objects."""
        invalid_results = {
            'strategy1': {'no_factor_exposures': True}
        }

        file_path = reporter.write_factor_table('invalid_factors', invalid_results)

        # Should handle gracefully
        assert pathlib.Path(file_path).exists()

    def test_write_regime_table(self, reporter):
        """Test regime analysis table generation."""
        class RegimeResult:
            def __init__(self, sharpe, win_rate, avg_ret, max_dd, sample_size):
                self.sharpe_ratio = sharpe
                self.win_rate = win_rate
                self.avg_return = avg_ret
                self.max_drawdown = max_dd
                self.sample_size = sample_size

        regime_results = {
            'strategy1': {
                'regime_results': {
                    'bull_market': RegimeResult(1.5, 0.60, 0.001, -0.10, 100),
                    'bear_market': RegimeResult(0.8, 0.52, 0.0005, -0.15, 80)
                }
            }
        }

        file_path = reporter.write_regime_table('regime_table', regime_results)

        # Verify CSV was created
        assert pathlib.Path(file_path).exists()
        df = pd.read_csv(file_path)
        assert len(df) == 2
        assert 'Regime' in df.columns
        assert 'Sharpe Ratio' in df.columns

    def test_write_regime_table_empty(self, reporter):
        """Test regime table with empty results."""
        file_path = reporter.write_regime_table('empty_regime', {})

        assert pathlib.Path(file_path).exists()
        assert file_path.endswith('.json')

    def test_write_spa_results(self, reporter):
        """Test SPA test results table."""
        class SPAResult:
            def __init__(self, p_val, test_stat, sig, bootstrap):
                self.p_value = p_val
                self.test_statistic = test_stat
                self.is_significant = sig
                self.bootstrap_samples = bootstrap

        spa_results = {
            'spa_results': {
                'strategy1': SPAResult(0.03, 2.1, True, 1000),
                'strategy2': SPAResult(0.12, 1.2, False, 1000)
            }
        }

        file_path = reporter.write_spa_results('spa_table', spa_results)

        assert pathlib.Path(file_path).exists()
        df = pd.read_csv(file_path)
        assert len(df) == 2
        assert 'P-Value' in df.columns

    def test_write_spa_results_empty(self, reporter):
        """Test SPA results with empty data."""
        file_path = reporter.write_spa_results('empty_spa', {})

        assert pathlib.Path(file_path).exists()

    def test_write_summary_html(self, reporter, sample_validation_results):
        """Test HTML summary generation."""
        file_path = reporter.write_summary_html(sample_validation_results)

        # Verify HTML file exists
        assert pathlib.Path(file_path).exists()
        assert file_path.endswith('.html')

        # Verify HTML content
        with open(file_path, 'r') as f:
            html_content = f.read()

        assert '<!DOCTYPE html>' in html_content
        assert 'Validation Summary Report' in html_content
        assert 'GO' in html_content

    def test_write_summary_html_with_equity_curve(self, reporter, sample_validation_results):
        """Test HTML summary with equity curve."""
        equity_png_path = '/path/to/equity_curve.png'

        file_path = reporter.write_summary_html(
            sample_validation_results,
            equity_png_path
        )

        with open(file_path, 'r') as f:
            html_content = f.read()

        assert 'equity_curve.png' in html_content

    def test_write_summary_html_fail_status(self, reporter):
        """Test HTML summary with NO-GO status."""
        fail_results = {
            'validation_gate': {
                'overall_recommendation': 'NO-GO',
                'deployment_readiness_score': 0.45,
                'detailed_evaluation': {
                    'risk_adjusted_returns.min_sharpe_ratio': {
                        'actual': 0.5,
                        'threshold': 1.0,
                        'passed': False
                    }
                }
            }
        }

        file_path = reporter.write_summary_html(fail_results)

        with open(file_path, 'r') as f:
            html_content = f.read()

        assert 'NO-GO' in html_content
        assert 'fail' in html_content

    def test_generate_complete_report(self, reporter, sample_validation_results, sample_equity_data):
        """Test complete report generation."""
        benchmark_data = sample_equity_data * 0.9

        report_path = reporter.generate_complete_report(
            sample_validation_results,
            sample_equity_data,
            benchmark_data
        )

        # Verify report directory exists
        assert pathlib.Path(report_path).exists()

        # Verify artifacts were created
        artifacts = reporter.get_artifacts()
        assert len(artifacts) > 0

        # Check for key files
        artifact_names = [pathlib.Path(a).name for a in artifacts]
        assert 'validation_results.json' in artifact_names
        assert 'summary.html' in artifact_names

    def test_generate_complete_report_minimal(self, reporter, sample_validation_results):
        """Test complete report with minimal data."""
        report_path = reporter.generate_complete_report(sample_validation_results)

        assert pathlib.Path(report_path).exists()
        assert len(reporter.get_artifacts()) > 0

    def test_generate_complete_report_with_factor_analysis(self, reporter):
        """Test complete report including factor analysis."""
        class MockResult:
            def __init__(self):
                self.annualized_alpha = 0.05
                self.alpha_t_stat = 2.0
                self.r_squared = 0.7
                self.n_obs = 252
                self.factor_exposures = {'mkt': 0.8}

        results = {
            'validation_gate': {
                'overall_recommendation': 'GO',
                'deployment_readiness_score': 0.9,
                'detailed_evaluation': {}
            },
            'factor_analysis': {
                'strategy1': MockResult()
            }
        }

        report_path = reporter.generate_complete_report(results)

        artifacts = reporter.get_artifacts()
        artifact_names = [pathlib.Path(a).name for a in artifacts]

        assert 'factor_analysis.csv' in artifact_names

    def test_generate_complete_report_with_regime_testing(self, reporter):
        """Test complete report including regime testing."""
        class RegimeResult:
            def __init__(self):
                self.sharpe_ratio = 1.2
                self.win_rate = 0.55
                self.avg_return = 0.001
                self.max_drawdown = -0.12
                self.sample_size = 100

        results = {
            'validation_gate': {
                'overall_recommendation': 'GO',
                'deployment_readiness_score': 0.9,
                'detailed_evaluation': {}
            },
            'regime_testing': {
                'strategy1': {
                    'regime_results': {
                        'bull_market': RegimeResult()
                    }
                }
            }
        }

        report_path = reporter.generate_complete_report(results)

        artifacts = reporter.get_artifacts()
        artifact_names = [pathlib.Path(a).name for a in artifacts]

        assert 'regime_testing.csv' in artifact_names

    def test_get_report_path(self, reporter):
        """Test getting report path."""
        path = reporter.get_report_path()

        assert isinstance(path, str)
        assert pathlib.Path(path).exists()

    def test_get_artifacts(self, reporter):
        """Test getting artifacts list."""
        # Initially empty
        assert len(reporter.get_artifacts()) == 0

        # Add some artifacts
        reporter.write_json('test1', {'data': 1})
        reporter.write_json('test2', {'data': 2})

        artifacts = reporter.get_artifacts()
        assert len(artifacts) == 2

        # Verify it returns a copy
        artifacts.append('fake_artifact')
        assert len(reporter.get_artifacts()) == 2

    def test_artifacts_tracking(self, reporter):
        """Test that all write methods track artifacts."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        equity = pd.Series([1.0, 1.01, 1.02], index=pd.date_range('2023-01-01', periods=3))

        reporter.write_json('json_test', {'key': 'value'})
        reporter.write_csv('csv_test', df)
        reporter.write_equity_curve('equity_test', equity)

        artifacts = reporter.get_artifacts()
        assert len(artifacts) >= 3

    def test_html_artifacts_section(self, reporter, sample_validation_results):
        """Test that HTML includes artifacts section."""
        # Create some artifacts first
        reporter.write_json('test_artifact', {'data': 1})

        file_path = reporter.write_summary_html(sample_validation_results)

        with open(file_path, 'r') as f:
            html_content = f.read()

        assert 'Artifacts Generated' in html_content
        assert 'test_artifact.json' in html_content

    def test_concurrent_reporters(self, temp_outdir):
        """Test multiple reporter instances."""
        reporter1 = ValidationReporter(outdir=temp_outdir)
        reporter2 = ValidationReporter(outdir=temp_outdir)

        # Different reporters should have different base directories
        assert reporter1.base != reporter2.base

        # Both should be able to write
        reporter1.write_json('r1_data', {'source': 'r1'})
        reporter2.write_json('r2_data', {'source': 'r2'})

        assert len(reporter1.get_artifacts()) == 1
        assert len(reporter2.get_artifacts()) == 1


class TestReportingEdgeCases:
    """Test edge cases and error scenarios."""

    def test_very_long_validation_results(self, tmp_path):
        """Test with very large validation results."""
        reporter = ValidationReporter(outdir=str(tmp_path))

        # Create large nested structure
        large_results = {
            'validation_gate': {
                'overall_recommendation': 'GO',
                'deployment_readiness_score': 0.9,
                'detailed_evaluation': {
                    f'metric_{i}': {
                        'actual': i * 0.1,
                        'threshold': i * 0.05,
                        'passed': True
                    }
                    for i in range(100)
                }
            }
        }

        file_path = reporter.write_json('large_results', large_results)

        # Should handle large files
        assert pathlib.Path(file_path).exists()

        with open(file_path, 'r') as f:
            loaded = json.load(f)
        assert len(loaded['validation_gate']['detailed_evaluation']) == 100

    def test_special_characters_in_data(self, tmp_path):
        """Test handling special characters."""
        reporter = ValidationReporter(outdir=str(tmp_path))

        data = {
            'special': 'test<>&"\'',
            'unicode': 'Test™ • Café',
            'emoji': '📊 📈 🚀'
        }

        file_path = reporter.write_json('special_chars', data)

        with open(file_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        assert loaded['unicode'] == 'Test™ • Café'

    def test_empty_dataframe(self, tmp_path):
        """Test with empty DataFrame."""
        reporter = ValidationReporter(outdir=str(tmp_path))

        empty_df = pd.DataFrame()
        file_path = reporter.write_csv('empty_df', empty_df)

        assert pathlib.Path(file_path).exists()

    def test_missing_validation_gate_key(self, tmp_path):
        """Test HTML generation without validation_gate."""
        reporter = ValidationReporter(outdir=str(tmp_path))

        results = {
            'other_data': {'value': 123}
        }

        file_path = reporter.write_summary_html(results)

        # Should still generate HTML
        assert pathlib.Path(file_path).exists()

    def test_invalid_equity_data_types(self, tmp_path):
        """Test equity curve with invalid data types."""
        reporter = ValidationReporter(outdir=str(tmp_path))

        # Non-numeric data
        invalid_equity = pd.Series(['a', 'b', 'c'], index=pd.date_range('2023-01-01', periods=3))

        with pytest.raises(Exception):
            reporter.write_equity_curve('invalid_equity', invalid_equity)
