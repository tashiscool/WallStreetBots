"""
Risk Dashboard - 2025 Implementation
Real-time risk monitoring and alerting system
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

from .advanced_var_engine import AdvancedVaREngine, VaRSuite
from .stress_testing_engine import StressTesting2025, StressTestReport
from .ml_risk_predictor import MLRiskPredictor, RiskPrediction

@dataclass
class RiskAlert:
    """Risk alert data structure"""
    alert_id: str
    alert_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    message: str
    timestamp: str
    portfolio_impact: float
    recommended_action: str
    acknowledged: bool = False

@dataclass
class RiskSummary:
    """Comprehensive risk summary"""
    # Core risk metrics
    var_1d: float
    var_5d: float
    cvar_99: float
    
    # Advanced 2025 metrics
    tail_expectation: float
    regime_adjusted_risk: float
    ml_risk_forecast: float
    
    # Alternative data risk signals
    sentiment_risk: float
    options_flow_risk: float
    social_media_risk: float
    
    # Factor risk attribution
    factor_risk_breakdown: Dict[str, float]
    concentration_risk: float
    
    # Stress test results
    stress_test_pnl: float
    scenario_analysis: Dict[str, float]
    
    # Real-time alerts
    active_alerts: List[RiskAlert]
    risk_limit_utilization: Dict[str, float]
    
    # Metadata
    calculation_timestamp: str
    portfolio_value: float
    total_positions: int

class RiskDashboard2025:
    """Real-time risk monitoring dashboard with 2025 features"""
    
    def __init__(self, portfolio_value: float = 100000.0):
        self.portfolio_value = portfolio_value
        
        # Initialize risk engines
        self.var_engine = AdvancedVaREngine(portfolio_value)
        self.stress_tester = StressTesting2025()
        self.ml_predictor = MLRiskPredictor()
        
        # Risk limits
        self.risk_limits = {
            'max_var_1d': 0.05,  # 5% of portfolio
            'max_var_5d': 0.10,  # 10% of portfolio
            'max_cvar_99': 0.08,  # 8% of portfolio
            'max_concentration': 0.20,  # 20% per position
            'max_correlation': 0.80,  # 80% max correlation
            'min_liquidity': 0.10  # 10% minimum liquidity
        }
        
        # Alert system
        self.active_alerts = []
        self.alert_history = []
        
    def generate_risk_summary(self, portfolio: Dict[str, Any]) -> RiskSummary:
        """Generate comprehensive real-time risk summary"""
        
        # Extract portfolio data
        positions = portfolio.get('positions', [])
        strategies = portfolio.get('strategies', {})
        market_data = portfolio.get('market_data', {})
        
        # Calculate core risk metrics
        var_1d, var_5d, cvar_99 = self._calculate_core_risk_metrics(positions, market_data)
        
        # Calculate advanced metrics
        tail_expectation = self._calculate_tail_expectation(positions)
        regime_adjusted_risk = self._calculate_regime_adjusted_risk(market_data)
        ml_risk_forecast = self._calculate_ml_risk_forecast(market_data)
        
        # Calculate alternative data risk signals
        sentiment_risk = self._calculate_sentiment_risk(market_data)
        options_flow_risk = self._calculate_options_flow_risk(market_data)
        social_media_risk = self._calculate_social_media_risk(market_data)
        
        # Calculate factor risk attribution
        factor_risk_breakdown = self._calculate_factor_risk_breakdown(positions)
        concentration_risk = self._calculate_concentration_risk(positions)
        
        # Run stress tests
        stress_test_pnl, scenario_analysis = self._run_stress_tests(portfolio)
        
        # Check for alerts
        self._check_risk_alerts(portfolio)
        
        # Calculate risk limit utilization
        risk_limit_utilization = self._calculate_risk_limit_utilization(
            var_1d, var_5d, cvar_99, concentration_risk
        )
        
        return RiskSummary(
            var_1d=var_1d,
            var_5d=var_5d,
            cvar_99=cvar_99,
            tail_expectation=tail_expectation,
            regime_adjusted_risk=regime_adjusted_risk,
            ml_risk_forecast=ml_risk_forecast,
            sentiment_risk=sentiment_risk,
            options_flow_risk=options_flow_risk,
            social_media_risk=social_media_risk,
            factor_risk_breakdown=factor_risk_breakdown,
            concentration_risk=concentration_risk,
            stress_test_pnl=stress_test_pnl,
            scenario_analysis=scenario_analysis,
            active_alerts=self.active_alerts,
            risk_limit_utilization=risk_limit_utilization,
            calculation_timestamp=datetime.now().isoformat(),
            portfolio_value=self.portfolio_value,
            total_positions=len(positions)
        )
    
    def _calculate_core_risk_metrics(self, positions: List[Dict], market_data: Dict) -> Tuple[float, float, float]:
        """Calculate core VaR and CVaR metrics"""
        
        # Generate sample returns for demonstration
        # In real implementation, this would use actual position returns
        np.random.seed(42)
        sample_returns = np.random.normal(0.001, 0.02, 252)
        
        # Calculate VaR suite
        var_suite = self.var_engine.calculate_var_suite(
            returns=sample_returns,
            confidence_levels=[0.95, 0.99],
            methods=['historical', 'monte_carlo']
        )
        
        # Extract VaR values
        var_1d = var_suite.results.get('historical_95', var_suite.results.get('monte_carlo_95')).var_value
        var_5d = var_1d * np.sqrt(5)  # Scale for 5-day horizon
        cvar_99 = self.var_engine.calculate_cvar(sample_returns, 0.99)
        
        return var_1d, var_5d, cvar_99
    
    def _calculate_tail_expectation(self, positions: List[Dict]) -> float:
        """Calculate tail expectation using EVT"""
        # Simplified tail expectation calculation
        return self.portfolio_value * 0.03  # 3% of portfolio value
    
    def _calculate_regime_adjusted_risk(self, market_data: Dict) -> float:
        """Calculate regime-adjusted risk"""
        # Get regime detection from VaR engine
        sample_returns = np.random.normal(0.001, 0.02, 60)
        regime_info = self.var_engine.detect_regime_and_adjust(sample_returns)
        
        # Adjust risk based on regime
        base_risk = self.portfolio_value * 0.05  # 5% base risk
        adjustment_factor = regime_info.get('adjustment_factor', 1.0)
        
        return base_risk * adjustment_factor
    
    def _calculate_ml_risk_forecast(self, market_data: Dict) -> float:
        """Calculate ML-based risk forecast"""
        # Use ML predictor for risk score
        risk_prediction = self.ml_predictor.predict_risk_score(market_data)
        
        # Convert risk score to dollar amount
        return (risk_prediction.risk_score / 100) * self.portfolio_value
    
    def _calculate_sentiment_risk(self, market_data: Dict) -> float:
        """Calculate sentiment-based risk"""
        sentiment = market_data.get('sentiment', 0)
        
        # Convert sentiment to risk score
        if sentiment < -0.5:
            return self.portfolio_value * 0.08  # 8% risk for very negative sentiment
        elif sentiment < -0.2:
            return self.portfolio_value * 0.05  # 5% risk for negative sentiment
        else:
            return self.portfolio_value * 0.02  # 2% risk for neutral/positive sentiment
    
    def _calculate_options_flow_risk(self, market_data: Dict) -> float:
        """Calculate options flow risk"""
        put_call_ratio = market_data.get('put_call_ratio', 1.0)
        
        # High put/call ratio indicates bearish sentiment
        if put_call_ratio > 1.5:
            return self.portfolio_value * 0.06  # 6% risk
        elif put_call_ratio > 1.2:
            return self.portfolio_value * 0.04  # 4% risk
        else:
            return self.portfolio_value * 0.02  # 2% risk
    
    def _calculate_social_media_risk(self, market_data: Dict) -> float:
        """Calculate social media risk"""
        social_volume = market_data.get('social_volume', 0.5)
        social_sentiment = market_data.get('social_sentiment', 0)
        
        # High volume + negative sentiment = high risk
        risk_score = social_volume * abs(social_sentiment)
        return self.portfolio_value * risk_score * 0.1  # Scale to portfolio
    
    def _calculate_factor_risk_breakdown(self, positions: List[Dict]) -> Dict[str, float]:
        """Calculate factor risk breakdown"""
        # Simplified factor risk calculation
        factors = {
            'market_risk': 0.40,  # 40% market risk
            'sector_risk': 0.25,  # 25% sector risk
            'style_risk': 0.20,   # 20% style risk
            'idiosyncratic_risk': 0.15  # 15% idiosyncratic risk
        }
        
        # Scale by portfolio value
        return {factor: risk * self.portfolio_value for factor, risk in factors.items()}
    
    def _calculate_concentration_risk(self, positions: List[Dict]) -> float:
        """Calculate concentration risk"""
        if not positions:
            return 0.0
        
        # Calculate Herfindahl index for concentration
        total_value = sum(pos.get('value', 0) for pos in positions)
        if total_value == 0:
            return 0.0
        
        weights = [pos.get('value', 0) / total_value for pos in positions]
        herfindahl = sum(w**2 for w in weights)
        
        # Convert to risk amount
        return herfindahl * self.portfolio_value
    
    def _run_stress_tests(self, portfolio: Dict) -> Tuple[float, Dict[str, float]]:
        """Run stress tests and return results"""
        try:
            stress_report = self.stress_tester.run_comprehensive_stress_test(portfolio)
            
            # Get worst case P&L
            worst_pnl = min(result.portfolio_pnl for result in stress_report.results.values())
            
            # Get scenario analysis
            scenario_analysis = {
                scenario: result.portfolio_pnl 
                for scenario, result in stress_report.results.items()
            }
            
            return worst_pnl, scenario_analysis
            
        except Exception as e:
            print(f"Warning: Stress test failed: {e}")
            return 0.0, {}
    
    def _check_risk_alerts(self, portfolio: Dict) -> None:
        """Check for risk limit breaches and generate alerts"""
        # Clear previous alerts
        self.active_alerts = []
        
        # Get current risk metrics
        positions = portfolio.get('positions', [])
        market_data = portfolio.get('market_data', {})
        
        var_1d, var_5d, cvar_99 = self._calculate_core_risk_metrics(positions, market_data)
        concentration_risk = self._calculate_concentration_risk(positions)
        
        # Check VaR limits
        if var_1d > self.risk_limits['max_var_1d'] * self.portfolio_value:
            self._add_alert(
                alert_type="VAR_BREACH",
                severity="HIGH",
                message=f"1-day VaR ${var_1d:,.0f} exceeds limit ${self.risk_limits['max_var_1d'] * self.portfolio_value:,.0f}",
                portfolio_impact=var_1d,
                recommended_action="Reduce position sizes immediately"
            )
        
        # Check concentration limits
        if concentration_risk > self.risk_limits['max_concentration'] * self.portfolio_value:
            self._add_alert(
                alert_type="CONCENTRATION_BREACH",
                severity="MEDIUM",
                message=f"Concentration risk ${concentration_risk:,.0f} exceeds limit",
                portfolio_impact=concentration_risk,
                recommended_action="Diversify portfolio positions"
            )
        
        # Check ML risk forecast
        ml_risk = self._calculate_ml_risk_forecast(market_data)
        if ml_risk > self.portfolio_value * 0.08:  # 8% threshold
            self._add_alert(
                alert_type="ML_RISK_HIGH",
                severity="MEDIUM",
                message=f"ML risk forecast ${ml_risk:,.0f} indicates high risk environment",
                portfolio_impact=ml_risk,
                recommended_action="Consider reducing risk exposure"
            )
    
    def _add_alert(self, alert_type: str, severity: str, message: str, 
                   portfolio_impact: float, recommended_action: str) -> None:
        """Add new risk alert"""
        alert = RiskAlert(
            alert_id=f"{alert_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp=datetime.now().isoformat(),
            portfolio_impact=portfolio_impact,
            recommended_action=recommended_action
        )
        
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
    
    def _calculate_risk_limit_utilization(self, var_1d: float, var_5d: float, 
                                        cvar_99: float, concentration_risk: float) -> Dict[str, float]:
        """Calculate risk limit utilization percentages"""
        return {
            'var_1d': (var_1d / (self.risk_limits['max_var_1d'] * self.portfolio_value)) * 100,
            'var_5d': (var_5d / (self.risk_limits['max_var_5d'] * self.portfolio_value)) * 100,
            'cvar_99': (cvar_99 / (self.risk_limits['max_cvar_99'] * self.portfolio_value)) * 100,
            'concentration': (concentration_risk / (self.risk_limits['max_concentration'] * self.portfolio_value)) * 100
        }
    
    def get_risk_dashboard_data(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Get formatted dashboard data for display"""
        risk_summary = self.generate_risk_summary(portfolio)
        
        return {
            'timestamp': risk_summary.calculation_timestamp,
            'portfolio_value': risk_summary.portfolio_value,
            'total_positions': risk_summary.total_positions,
            
            # Risk metrics
            'risk_metrics': {
                'var_1d': {
                    'value': risk_summary.var_1d,
                    'percentage': (risk_summary.var_1d / risk_summary.portfolio_value) * 100,
                    'limit_utilization': risk_summary.risk_limit_utilization.get('var_1d', 0)
                },
                'var_5d': {
                    'value': risk_summary.var_5d,
                    'percentage': (risk_summary.var_5d / risk_summary.portfolio_value) * 100,
                    'limit_utilization': risk_summary.risk_limit_utilization.get('var_5d', 0)
                },
                'cvar_99': {
                    'value': risk_summary.cvar_99,
                    'percentage': (risk_summary.cvar_99 / risk_summary.portfolio_value) * 100,
                    'limit_utilization': risk_summary.risk_limit_utilization.get('cvar_99', 0)
                }
            },
            
            # Advanced metrics
            'advanced_metrics': {
                'tail_expectation': risk_summary.tail_expectation,
                'regime_adjusted_risk': risk_summary.regime_adjusted_risk,
                'ml_risk_forecast': risk_summary.ml_risk_forecast,
                'concentration_risk': risk_summary.concentration_risk
            },
            
            # Alternative data signals
            'alternative_signals': {
                'sentiment_risk': risk_summary.sentiment_risk,
                'options_flow_risk': risk_summary.options_flow_risk,
                'social_media_risk': risk_summary.social_media_risk
            },
            
            # Factor breakdown
            'factor_breakdown': risk_summary.factor_risk_breakdown,
            
            # Stress test results
            'stress_tests': {
                'worst_case_pnl': risk_summary.stress_test_pnl,
                'scenario_analysis': risk_summary.scenario_analysis
            },
            
            # Alerts
            'alerts': [
                {
                    'id': alert.alert_id,
                    'type': alert.alert_type,
                    'severity': alert.severity,
                    'message': alert.message,
                    'timestamp': alert.timestamp,
                    'portfolio_impact': alert.portfolio_impact,
                    'recommended_action': alert.recommended_action,
                    'acknowledged': alert.acknowledged
                }
                for alert in risk_summary.active_alerts
            ],
            
            # Risk limit utilization
            'risk_limits': risk_summary.risk_limit_utilization
        }
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge a risk alert"""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def get_alert_history(self, limit: int = 50) -> List[Dict]:
        """Get alert history"""
        return [
            {
                'id': alert.alert_id,
                'type': alert.alert_type,
                'severity': alert.severity,
                'message': alert.message,
                'timestamp': alert.timestamp,
                'portfolio_impact': alert.portfolio_impact,
                'recommended_action': alert.recommended_action,
                'acknowledged': alert.acknowledged
            }
            for alert in self.alert_history[-limit:]
        ]

# Example usage and testing
if __name__ == "__main__":
    # Create sample portfolio
    sample_portfolio = {
        'total_value': 100000.0,
        'positions': [
            {'ticker': 'AAPL', 'value': 25000, 'quantity': 100},
            {'ticker': 'TSLA', 'value': 20000, 'quantity': 50},
            {'ticker': 'SPY', 'value': 30000, 'quantity': 100},
            {'ticker': 'QQQ', 'value': 25000, 'quantity': 80}
        ],
        'strategies': {
            'wsb_dip_bot': {'exposure': 0.25},
            'earnings_protection': {'exposure': 0.20},
            'index_baseline': {'exposure': 0.15},
            'momentum_weeklies': {'exposure': 0.20},
            'debit_spreads': {'exposure': 0.10},
            'leaps_tracker': {'exposure': 0.10}
        },
        'market_data': {
            'prices': [100 + i * 0.1 for i in range(100)],
            'volumes': [1000 + i * 10 for i in range(100)],
            'sentiment': -0.3,
            'put_call_ratio': 1.4,
            'social_volume': 0.7,
            'social_sentiment': -0.2
        }
    }
    
    # Initialize risk dashboard
    dashboard = RiskDashboard2025(portfolio_value=100000.0)
    
    # Generate risk summary
    print("Generating Risk Dashboard...")
    print("=" * 50)
    
    dashboard_data = dashboard.get_risk_dashboard_data(sample_portfolio)
    
    # Print risk metrics
    print(f"Portfolio Value: ${dashboard_data['portfolio_value']:,.0f}")
    print(f"Total Positions: {dashboard_data['total_positions']}")
    print(f"Timestamp: {dashboard_data['timestamp']}")
    
    print(f"\nRisk Metrics:")
    for metric, data in dashboard_data['risk_metrics'].items():
        print(f"{metric.upper()}: ${data['value']:,.0f} ({data['percentage']:.1f}%) - {data['limit_utilization']:.1f}% of limit")
    
    print(f"\nAdvanced Metrics:")
    for metric, value in dashboard_data['advanced_metrics'].items():
        print(f"{metric}: ${value:,.0f}")
    
    print(f"\nAlternative Data Signals:")
    for signal, value in dashboard_data['alternative_signals'].items():
        print(f"{signal}: ${value:,.0f}")
    
    print(f"\nFactor Risk Breakdown:")
    for factor, value in dashboard_data['factor_breakdown'].items():
        print(f"{factor}: ${value:,.0f}")
    
    print(f"\nActive Alerts: {len(dashboard_data['alerts'])}")
    for alert in dashboard_data['alerts']:
        print(f"  {alert['severity']}: {alert['message']}")
        print(f"    Action: {alert['recommended_action']}")
    
    print(f"\nRisk Limit Utilization:")
    for limit, utilization in dashboard_data['risk_limits'].items():
        status = "⚠️" if utilization > 80 else "✅"
        print(f"{status} {limit}: {utilization:.1f}%")

