"""Validation Reporting with HTML/JSON Artifacts."""

from __future__ import annotations
import json
import pathlib
import datetime
from typing import Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class ValidationReporter:
    """Generates comprehensive validation reports with artifacts."""

    _instance_counter = 0  # Class-level counter for unique directories

    def __init__(self, outdir: str = 'reports/validation'):
        # Use counter to ensure unique directories even within same second
        ValidationReporter._instance_counter += 1
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
        unique_suffix = f"_{ValidationReporter._instance_counter}"
        self.base = pathlib.Path(outdir) / f"{timestamp}{unique_suffix}"
        self.base.mkdir(parents=True, exist_ok=True)
        self.artifacts = []

    def write_json(self, name: str, payload: Dict[str, Any]) -> str:
        """Write JSON artifact."""
        file_path = self.base / f'{name}.json'
        file_path.write_text(json.dumps(payload, indent=2, default=str))
        self.artifacts.append(str(file_path))
        return str(file_path)

    def write_csv(self, name: str, df: pd.DataFrame) -> str:
        """Write CSV artifact."""
        file_path = self.base / f'{name}.csv'
        df.to_csv(file_path, index=True)
        self.artifacts.append(str(file_path))
        return str(file_path)

    def write_equity_curve(self, name: str, equity_data: pd.Series,
                          benchmark_data: Optional[pd.Series] = None) -> str:
        """Generate equity curve plot."""
        if equity_data is None or len(equity_data) == 0:
            raise ValueError("Cannot generate equity curve with empty data")

        # Validate data is numeric
        if not pd.api.types.is_numeric_dtype(equity_data):
            raise TypeError("Equity data must be numeric")

        plt.figure(figsize=(12, 8))
        
        # Plot strategy equity
        plt.plot(equity_data.index, equity_data.values, 
                label='Strategy', linewidth=2, color='blue')
        
        # Plot benchmark if provided
        if benchmark_data is not None:
            plt.plot(benchmark_data.index, benchmark_data.values, 
                    label='Benchmark', linewidth=2, color='red', alpha=0.7)
        
        plt.title('Equity Curve Comparison')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        file_path = self.base / f'{name}.png'
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.artifacts.append(str(file_path))
        return str(file_path)

    def write_factor_table(self, name: str, factor_results: Dict[str, Any]) -> str:
        """Generate factor analysis table."""
        if not factor_results:
            return self.write_json(name, {'error': 'No factor results'})
        
        # Extract factor data
        table_data = []
        for strategy_name, result in factor_results.items():
            if hasattr(result, 'factor_exposures'):
                row = {
                    'Strategy': strategy_name,
                    'Alpha (Annualized)': result.annualized_alpha,
                    'Alpha T-Stat': result.alpha_t_stat,
                    'R-Squared': result.r_squared,
                    'Observations': result.n_obs
                }
                
                # Add factor exposures
                for factor, exposure in result.factor_exposures.items():
                    row[f'{factor}_Exposure'] = exposure
                
                table_data.append(row)
        
        if table_data:
            df = pd.DataFrame(table_data)
            return self.write_csv(name, df)
        else:
            return self.write_json(name, {'error': 'No valid factor data'})

    def write_regime_table(self, name: str, regime_results: Dict[str, Any]) -> str:
        """Generate regime analysis table."""
        if not regime_results:
            return self.write_json(name, {'error': 'No regime results'})

        # Handle non-dict regime_results (e.g., {'passed': True})
        if not isinstance(regime_results, dict):
            return self.write_json(name, {'error': 'Invalid regime results format'})

        table_data = []
        for strategy_name, result in regime_results.items():
            # Skip non-dict entries like 'passed': True
            if not isinstance(result, dict):
                continue
            if 'regime_results' in result:
                for regime_name, regime_result in result['regime_results'].items():
                    row = {
                        'Strategy': strategy_name,
                        'Regime': regime_name,
                        'Sharpe Ratio': regime_result.sharpe_ratio,
                        'Win Rate': regime_result.win_rate,
                        'Avg Return': regime_result.avg_return,
                        'Max Drawdown': regime_result.max_drawdown,
                        'Sample Size': regime_result.sample_size
                    }
                    table_data.append(row)

        if table_data:
            df = pd.DataFrame(table_data)
            return self.write_csv(name, df)
        else:
            return self.write_json(name, {'error': 'No valid regime data'})

    def write_spa_results(self, name: str, spa_results: Dict[str, Any]) -> str:
        """Generate SPA test results table."""
        if not spa_results:
            return self.write_json(name, {'error': 'No SPA results'})
        
        table_data = []
        if 'spa_results' in spa_results:
            for strategy_name, result in spa_results['spa_results'].items():
                row = {
                    'Strategy': strategy_name,
                    'P-Value': result.p_value,
                    'Test Statistic': result.test_statistic,
                    'Significant': result.is_significant,
                    'Bootstrap Samples': result.bootstrap_samples
                }
                table_data.append(row)
        
        if table_data:
            df = pd.DataFrame(table_data)
            return self.write_csv(name, df)
        else:
            return self.write_json(name, {'error': 'No valid SPA data'})

    def write_summary_html(self, validation_results: Dict[str, Any], 
                          equity_png_path: Optional[str] = None) -> str:
        """Generate HTML summary report."""
        html_parts = [
            '<!DOCTYPE html>',
            '<html><head>',
            '<title>Validation Summary Report</title>',
            '<style>',
            'body { font-family: Arial, sans-serif; margin: 40px; }',
            'h1, h2 { color: #333; }',
            'table { border-collapse: collapse; width: 100%; margin: 20px 0; }',
            'th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }',
            'th { background-color: #f2f2f2; }',
            '.pass { color: green; }',
            '.fail { color: red; }',
            '.warning { color: orange; }',
            '</style>',
            '</head><body>',
            '<h1>Validation Summary Report</h1>',
            f'<p><strong>Generated:</strong> {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>'
        ]
        
        # Add equity curve if provided
        if equity_png_path:
            equity_filename = pathlib.Path(equity_png_path).name
            html_parts.extend([
                '<h2>Equity Curve</h2>',
                f'<img src="{equity_filename}" width="900" alt="Equity Curve"/>'
            ])
        
        # Add validation gate results
        if 'validation_gate' in validation_results:
            gate_results = validation_results['validation_gate']
            html_parts.extend([
                '<h2>Validation Gate Results</h2>',
                '<table>',
                '<tr><th>Metric</th><th>Actual</th><th>Threshold</th><th>Status</th></tr>'
            ])
            
            for metric, data in gate_results.get('detailed_evaluation', {}).items():
                status_class = 'pass' if data['passed'] else 'fail'
                status_text = 'PASS' if data['passed'] else 'FAIL'
                html_parts.append(
                    f'<tr><td>{metric}</td><td>{data["actual"]}</td>'
                    f'<td>{data["threshold"]}</td><td class="{status_class}">{status_text}</td></tr>'
                )
            
            html_parts.extend([
                '</table>',
                f'<p><strong>Overall Recommendation:</strong> '
                f'<span class="{"pass" if gate_results.get("overall_recommendation") == "GO" else "fail"}">'
                f'{gate_results.get("overall_recommendation", "UNKNOWN")}</span></p>',
                f'<p><strong>Deployment Readiness Score:</strong> '
                f'{gate_results.get("deployment_readiness_score", 0):.2%}</p>'
            ])
        
        # Add summary statistics
        html_parts.extend([
            '<h2>Summary Statistics</h2>',
            '<table>',
            '<tr><th>Category</th><th>Status</th></tr>'
        ])
        
        for category, status in validation_results.items():
            if category != 'validation_gate':
                status_class = 'pass' if status.get('passed', False) else 'fail'
                status_text = 'PASS' if status.get('passed', False) else 'FAIL'
                html_parts.append(
                    f'<tr><td>{category}</td><td class="{status_class}">{status_text}</td></tr>'
                )
        
        html_parts.extend([
            '</table>',
            '<h2>Artifacts Generated</h2>',
            '<ul>'
        ])
        
        for artifact in self.artifacts:
            artifact_name = pathlib.Path(artifact).name
            html_parts.append(f'<li><a href="{artifact_name}">{artifact_name}</a></li>')
        
        html_parts.extend([
            '</ul>',
            '</body></html>'
        ])
        
        file_path = self.base / 'summary.html'
        file_path.write_text('\n'.join(html_parts))
        self.artifacts.append(str(file_path))
        
        return str(file_path)

    def generate_complete_report(self, validation_results: Dict[str, Any], 
                               equity_data: Optional[pd.Series] = None,
                               benchmark_data: Optional[pd.Series] = None) -> str:
        """Generate complete validation report with all artifacts."""
        
        # Write individual components
        self.write_json('validation_results', validation_results)
        
        if 'factor_analysis' in validation_results:
            self.write_factor_table('factor_analysis', validation_results['factor_analysis'])
        
        if 'regime_testing' in validation_results:
            self.write_regime_table('regime_testing', validation_results['regime_testing'])
        
        if 'reality_check' in validation_results:
            self.write_spa_results('spa_results', validation_results['reality_check'])
        
        # Generate equity curve if data provided
        equity_png_path = None
        if equity_data is not None:
            equity_png_path = self.write_equity_curve('equity_curve', equity_data, benchmark_data)
        
        # Generate HTML summary
        summary_path = self.write_summary_html(validation_results, equity_png_path)
        
        return str(self.base)

    def get_report_path(self) -> str:
        """Get the base report directory path."""
        return str(self.base)

    def get_artifacts(self) -> list[str]:
        """Get list of generated artifacts."""
        return self.artifacts.copy()


# Example usage and testing
if __name__ == "__main__":
    def test_reporting():
        """Test the reporting system."""
        print("=== Validation Reporting Test ===")
        
        # Create test data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        equity_data = pd.Series(
            (1 + pd.Series(range(100)) * 0.001).cumprod(),
            index=dates
        )
        
        benchmark_data = pd.Series(
            (1 + pd.Series(range(100)) * 0.0005).cumprod(),
            index=dates
        )
        
        # Mock validation results
        validation_results = {
            'validation_gate': {
                'overall_recommendation': 'GO',
                'deployment_readiness_score': 0.85,
                'detailed_evaluation': {
                    'risk_adjusted_returns.min_sharpe_ratio': {
                        'actual': 1.2, 'threshold': 1.0, 'passed': True
                    }
                }
            },
            'factor_analysis': {
                'strategy1': type('Result', (), {
                    'annualized_alpha': 0.06,
                    'alpha_t_stat': 2.1,
                    'r_squared': 0.75,
                    'n_obs': 252,
                    'factor_exposures': {'mkt': 0.8, 'smb': 0.2}
                })()
            }
        }
        
        # Generate report
        reporter = ValidationReporter(outdir='/tmp/test_reports')
        report_path = reporter.generate_complete_report(
            validation_results, equity_data, benchmark_data
        )
        
        print(f"Report generated at: {report_path}")
        print(f"Artifacts: {reporter.get_artifacts()}")
    
    test_reporting()

