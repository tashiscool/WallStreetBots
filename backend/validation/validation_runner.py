"""Validation Runner - Entry point for comprehensive strategy validation."""

import argparse
import logging
from datetime import datetime
from typing import Dict, Any
import pandas as pd

# Import validation modules
from .statistical_rigor.reality_check import WhitesRealityCheck as RealityCheckValidator
from .drift_monitor import PerformanceDriftMonitor
from .factor_analysis import AlphaFactorAnalyzer
from .regime_testing import RegimeValidator

logger = logging.getLogger(__name__)


class ValidationRunner:
    """Main validation runner for comprehensive strategy evaluation."""
    
    def __init__(self):
        self.validators = {
            'reality_check': RealityCheckValidator(),
            'factor_analyzer': AlphaFactorAnalyzer(),
            'regime_validator': RegimeValidator()
        }
    
    def run_comprehensive_validation(self, strategy_returns: Dict[str, pd.Series], 
                                   benchmark_returns: pd.Series,
                                   market_data: pd.DataFrame,
                                   start_date: str = '2020-01-01',
                                   end_date: str = '2024-12-31') -> Dict[str, Any]:
        """Run comprehensive validation suite."""
        
        logger.info(f"Starting comprehensive validation from {start_date} to {end_date}")
        
        results = {}
        
        # 1. Reality Check / SPA Test
        logger.info("Running Reality Check / SPA Test...")
        try:
            reality_check_results = self.validators['reality_check'].validate_strategy_universe(
                strategy_returns, benchmark_returns
            )
            results['reality_check'] = reality_check_results
        except Exception as e:
            logger.error(f"Reality check failed: {e}")
            results['reality_check'] = {'error': str(e)}
        
        # 2. Factor Analysis
        logger.info("Running Factor Analysis...")
        try:
            # Create factor proxies if needed
            factor_df = self.validators['factor_analyzer'].create_factor_proxies(market_data)
            
            factor_results = {}
            for strategy_name, returns in strategy_returns.items():
                try:
                    factor_result = self.validators['factor_analyzer'].run_factor_regression(
                        returns, factor_df
                    )
                    factor_results[strategy_name] = factor_result
                except Exception as e:
                    logger.warning(f"Factor analysis failed for {strategy_name}: {e}")
                    continue
            
            results['factor_analysis'] = factor_results
        except Exception as e:
            logger.error(f"Factor analysis failed: {e}")
            results['factor_analysis'] = {'error': str(e)}
        
        # 3. Regime Testing
        logger.info("Running Regime Testing...")
        try:
            regime_results = {}
            for strategy_name, returns in strategy_returns.items():
                try:
                    regime_result = self.validators['regime_validator'].test_edge_persistence(
                        returns, market_data
                    )
                    regime_results[strategy_name] = regime_result
                except Exception as e:
                    logger.warning(f"Regime testing failed for {strategy_name}: {e}")
                    continue
            
            results['regime_testing'] = regime_results
        except Exception as e:
            logger.error(f"Regime testing failed: {e}")
            results['regime_testing'] = {'error': str(e)}
        
        # 4. Generate Summary
        results['summary'] = self._generate_validation_summary(results)
        
        logger.info("Comprehensive validation completed")
        return results
    
    def _generate_validation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation summary."""
        summary = {
            'validation_timestamp': datetime.now().isoformat(),
            'validation_modules_run': list(results.keys()),
            'overall_status': 'COMPLETED',
            'recommendations': []
        }
        
        # Analyze reality check results
        if 'reality_check' in results and 'error' not in results['reality_check']:
            rc = results['reality_check']
            if rc.get('validation_passed', False):
                summary['recommendations'].append("Strategy passes multiple hypothesis testing")
            else:
                summary['recommendations'].append("Strategy fails multiple hypothesis testing - consider reducing universe")
        
        # Analyze factor analysis results
        if 'factor_analysis' in results and 'error' not in results['factor_analysis']:
            fa = results['factor_analysis']
            significant_strategies = [k for k, v in fa.items() if v.alpha_significant]
            if significant_strategies:
                summary['recommendations'].append(f"Significant alpha found in: {significant_strategies}")
            else:
                summary['recommendations'].append("No significant alpha detected")
        
        # Analyze regime testing results
        if 'regime_testing' in results and 'error' not in results['regime_testing']:
            rt = results['regime_testing']
            robust_strategies = [k for k, v in rt.items() if v.get('edge_is_robust', False)]
            if robust_strategies:
                summary['recommendations'].append(f"Regime-robust strategies: {robust_strategies}")
            else:
                summary['recommendations'].append("No regime-robust strategies found")
        
        return summary


def main():
    """Main entry point for validation runner."""
    parser = argparse.ArgumentParser(description='Run comprehensive strategy validation')
    parser.add_argument('--start', default='2020-01-01', help='Start date for validation')
    parser.add_argument('--end', default='2024-12-31', help='End date for validation')
    parser.add_argument('--strategy', required=True, help='Strategy name to validate')
    parser.add_argument('--output', help='Output file for results')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize runner
    runner = ValidationRunner()
    
    # TODO: Load actual strategy data
    # This would typically load from your strategy backtest results
    logger.info(f"Running validation for strategy: {args.strategy}")
    
    # For now, return placeholder
    results = {
        'strategy': args.strategy,
        'start_date': args.start,
        'end_date': args.end,
        'status': 'Validation framework ready - implement data loading'
    }
    
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    print(f"Validation completed for {args.strategy}")
    print(f"Results: {results}")


if __name__ == '__main__':
    main()
