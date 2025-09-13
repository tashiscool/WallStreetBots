"""Stress Testing Engine-2025 Implementation
FCA - compliant stress testing framework with historical and hypothetical scenarios.
"""

import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any

warnings.filterwarnings("ignore")


@dataclass
class StressScenario:
    """Individual stress test scenario."""

    name: str
    description: str
    market_shock: dict[str, float]  # Asset class - >  shock percentage
    duration_days: int
    recovery_days: int
    probability: float = 0.01  # 1% probability


@dataclass
class StressTestResult:
    """Result of stress test scenario."""

    scenario_name: str
    portfolio_pnl: float
    max_drawdown: float
    recovery_time: int
    strategy_breakdown: dict[str, float]
    risk_metrics: dict[str, float]
    passed: bool
    failure_reason: str | None = None


@dataclass
class StressTestReport:
    """Comprehensive stress test report."""

    results: dict[str, StressTestResult]
    compliance_status: str
    recommendations: list[str]
    overall_risk_score: float
    test_date: str


class StressTesting2025:
    """FCA - compliant stress testing framework."""

    def __init__(self):
        self.regulatory_scenarios = self._initialize_regulatory_scenarios()
        self.risk_limits = {
            "max_drawdown": 0.25,  # 25% max drawdown
            "max_daily_loss": 0.10,  # 10% max daily loss
            "min_recovery_time": 750,  # 750 days max recovery (realistic for major crises)
        }

    def _initialize_regulatory_scenarios(self) -> dict[str, StressScenario]:
        """Initialize FCA - compliant stress scenarios."""
        scenarios = {
            "2008_financial_crisis": StressScenario(
                name="2008 Financial Crisis",
                description="Global financial crisis with market collapse",
                market_shock={
                    "equity_market": -0.50,  # 50% decline
                    "credit_spreads": 0.30,  # 30% spread widening
                    "volatility": 2.0,  # 200% vol increase
                    "liquidity": -0.80,  # 80% liquidity reduction
                },
                duration_days=180,
                recovery_days=730,
                probability=0.005,
            ),
            "2010_flash_crash": StressScenario(
                name="2010 Flash Crash",
                description="Rapid market decline and recovery",
                market_shock={
                    "equity_market": -0.20,  # 20% decline
                    "volatility": 3.0,  # 300% vol spike
                    "liquidity": -0.90,  # 90% liquidity loss
                },
                duration_days=1,
                recovery_days=7,
                probability=0.02,
            ),
            "2020_covid_pandemic": StressScenario(
                name="COVID - 19 Pandemic",
                description="Global pandemic market impact",
                market_shock={
                    "equity_market": -0.35,  # 35% decline
                    "volatility": 2.5,  # 250% vol increase
                    "sector_rotation": 0.40,  # 40% sector rotation
                    "liquidity": -0.60,  # 60% liquidity reduction
                },
                duration_days=30,
                recovery_days=120,
                probability=0.01,
            ),
            "interest_rate_shock": StressScenario(
                name="Interest Rate Shock",
                description="Rapid interest rate increase",
                market_shock={
                    "interest_rates": 0.03,  # 3% rate increase
                    "bond_prices": -0.15,  # 15% bond decline
                    "equity_market": -0.20,  # 20% equity decline
                    "volatility": 1.5,  # 150% vol increase
                },
                duration_days=60,
                recovery_days=180,
                probability=0.02,
            ),
            "geopolitical_crisis": StressScenario(
                name="Geopolitical Crisis",
                description="Major geopolitical event",
                market_shock={
                    "equity_market": -0.25,  # 25% decline
                    "oil_prices": 0.50,  # 50% oil increase
                    "volatility": 1.8,  # 180% vol increase
                    "currency": 0.20,  # 20% currency impact
                },
                duration_days=90,
                recovery_days=365,
                probability=0.015,
            ),
            "ai_bubble_burst": StressScenario(
                name="AI Bubble Burst",
                description="Technology / AI bubble collapse",
                market_shock={
                    "tech_stocks": -0.60,  # 60% tech decline
                    "growth_stocks": -0.40,  # 40% growth decline
                    "volatility": 2.2,  # 220% vol increase
                    "sector_rotation": 0.50,  # 50% sector rotation
                },
                duration_days=120,
                recovery_days=540,
                probability=0.01,
            ),
        }

        return scenarios

    def run_comprehensive_stress_test(
        self, portfolio: dict[str, Any], integration_frequency: str = "daily"
    ) -> StressTestReport:
        """Run comprehensive stress tests per 2025 FCA guidelines.

        Args:
            portfolio: Portfolio dictionary with positions and strategies
            integration_frequency: How often to run stress tests

        Returns:
            Comprehensive stress test report
        """
        results = {}
        failed_scenarios = []

        print("Running Comprehensive Stress Tests...")
        print(" = " * 50)

        for scenario_name, scenario in self.regulatory_scenarios.items():
            print(f"Testing scenario: {scenario.name}")

            try:
                scenario_result = self._simulate_portfolio_under_stress(
                    portfolio, scenario
                )
                results[scenario_name] = scenario_result

                if not scenario_result.passed:
                    failed_scenarios.append(scenario_name)
                    print(f"  ❌ FAILED: {scenario_result.failure_reason}")
                else:
                    print(f"  ✅ PASSED: P & L ${scenario_result.portfolio_pnl:,.2f}")

            except Exception as e:
                print(f"  ❌ ERROR: {e!s}")
                results[scenario_name] = StressTestResult(
                    scenario_name=scenario_name,
                    portfolio_pnl=0.0,
                    max_drawdown=0.0,
                    recovery_time=0,
                    strategy_breakdown={},
                    risk_metrics={},
                    passed=False,
                    failure_reason=f"Simulation error: {e!s}",
                )
                failed_scenarios.append(scenario_name)

        # Generate compliance status
        compliance_status = self._check_fca_compliance(results, failed_scenarios)

        # Generate recommendations
        recommendations = self._generate_risk_recommendations(results, failed_scenarios)

        # Calculate overall risk score
        overall_risk_score = self._calculate_overall_risk_score(results)

        return StressTestReport(
            results=results,
            compliance_status=compliance_status,
            recommendations=recommendations,
            overall_risk_score=overall_risk_score,
            test_date=datetime.now().strftime("%Y-%m-%d %H: %M:%S"),
        )

    def _simulate_portfolio_under_stress(
        self, portfolio: dict[str, Any], scenario: StressScenario
    ) -> StressTestResult:
        """Simulate portfolio performance under stress scenario."""
        # Extract portfolio information
        total_value = portfolio.get("total_value", 100000.0)
        strategies = portfolio.get("strategies", {})

        # Calculate strategy - level impacts
        strategy_breakdown = {}
        total_pnl = 0.0

        for strategy_name, strategy_data in strategies.items():
            # Get strategy exposure and sensitivity
            exposure = strategy_data.get("exposure", 0.0)
            sensitivity = self._get_strategy_sensitivity(strategy_name, scenario)

            # Calculate strategy P & L impact
            strategy_pnl = exposure * sensitivity * total_value
            strategy_breakdown[strategy_name] = strategy_pnl
            total_pnl += strategy_pnl

        # Calculate risk metrics
        max_drawdown = abs(min(0, total_pnl / total_value))
        recovery_time = self._estimate_recovery_time(scenario, max_drawdown)

        # Check if scenario passes risk limits
        passed, failure_reason = self._check_risk_limits(
            total_pnl, max_drawdown, recovery_time
        )

        risk_metrics = {
            "portfolio_pnl_pct": (total_pnl / total_value) * 100,
            "max_drawdown_pct": max_drawdown * 100,
            "recovery_time_days": recovery_time,
            "scenario_probability": scenario.probability,
            "expected_loss": total_pnl * scenario.probability,
        }

        return StressTestResult(
            scenario_name=scenario.name,
            portfolio_pnl=total_pnl,
            max_drawdown=max_drawdown,
            recovery_time=recovery_time,
            strategy_breakdown=strategy_breakdown,
            risk_metrics=risk_metrics,
            passed=passed,
            failure_reason=failure_reason,
        )

    def _get_strategy_sensitivity(
        self, strategy_name: str, scenario: StressScenario
    ) -> float:
        """Get strategy sensitivity to stress scenario."""
        # Strategy - specific sensitivities to different market shocks
        sensitivities = {
            "wsb_dip_bot": {
                "equity_market": 1.2,  # High sensitivity to equity moves
                "volatility": 0.8,  # Moderate sensitivity to vol
                "liquidity": 0.6,  # Some liquidity sensitivity
                "sector_rotation": 0.4,
            },
            "earnings_protection": {
                "equity_market": 0.8,
                "volatility": 1.5,  # High vol sensitivity
                "liquidity": 0.3,
                "sector_rotation": 0.6,
            },
            "index_baseline": {
                "equity_market": 1.0,  # Direct market exposure
                "volatility": 0.5,
                "liquidity": 0.2,
                "sector_rotation": 0.1,
            },
            "momentum_weeklies": {
                "equity_market": 1.1,
                "volatility": 1.2,
                "liquidity": 0.7,
                "sector_rotation": 0.8,
            },
            "debit_spreads": {
                "equity_market": 0.9,
                "volatility": 1.8,  # Very high vol sensitivity
                "liquidity": 0.5,
                "sector_rotation": 0.3,
            },
            "leaps_tracker": {
                "equity_market": 1.3,  # High equity sensitivity
                "volatility": 1.0,
                "liquidity": 0.4,
                "sector_rotation": 0.5,
            },
        }

        strategy_sens = sensitivities.get(
            strategy_name,
            {
                "equity_market": 1.0,
                "volatility": 1.0,
                "liquidity": 0.5,
                "sector_rotation": 0.5,
            },
        )

        # Calculate weighted sensitivity based on scenario shocks
        total_sensitivity = 0.0
        total_weight = 0.0

        for shock_type, shock_magnitude in scenario.market_shock.items():
            if shock_type in strategy_sens:
                weight = abs(shock_magnitude)
                total_sensitivity += (
                    strategy_sens[shock_type] * shock_magnitude * weight
                )
                total_weight += weight

        return total_sensitivity / total_weight if total_weight > 0 else 0.0

    def _estimate_recovery_time(
        self, scenario: StressScenario, max_drawdown: float
    ) -> int:
        """Estimate recovery time based on scenario and drawdown."""
        base_recovery = scenario.recovery_days

        # Adjust based on drawdown severity
        if max_drawdown > 0.5:  #  > 50% drawdown
            recovery_multiplier = 2.0
        elif max_drawdown > 0.25:  #  > 25% drawdown
            recovery_multiplier = 1.5
        else:
            recovery_multiplier = 1.0

        return int(base_recovery * recovery_multiplier)

    def _check_risk_limits(
        self, pnl: float, max_drawdown: float, recovery_time: int
    ) -> tuple[bool, str | None]:
        """Check if scenario results exceed risk limits."""
        if max_drawdown > self.risk_limits["max_drawdown"]:
            return (
                False,
                f"Max drawdown {max_drawdown: .1%} exceeds limit {self.risk_limits['max_drawdown']: .1%}",
            )

        if recovery_time > self.risk_limits["min_recovery_time"]:
            return (
                False,
                f"Recovery time {recovery_time} days exceeds limit {self.risk_limits['min_recovery_time']} days",
            )

        return True, None

    def _check_fca_compliance(self, results: dict, failed_scenarios: list[str]) -> str:
        """Check FCA compliance status."""
        total_scenarios = len(results)
        failed_count = len(failed_scenarios)
        pass_rate = (total_scenarios - failed_count) / total_scenarios

        if pass_rate >= 0.8:  # 80% pass rate
            return "COMPLIANT"
        elif pass_rate >= 0.6:  # 60% pass rate
            return "PARTIALLY_COMPLIANT"
        else:
            return "NON_COMPLIANT"

    def _generate_risk_recommendations(
        self, results: dict, failed_scenarios: list[str]
    ) -> list[str]:
        """Generate risk management recommendations."""
        recommendations = []

        if not failed_scenarios:
            recommendations.append(
                "✅ All stress tests passed - portfolio is well - positioned for market stress"
            )
            return recommendations

        # Analyze failure patterns
        high_drawdown_scenarios = []
        long_recovery_scenarios = []

        for scenario_name in failed_scenarios:
            result = results[scenario_name]
            if result.max_drawdown > 0.3:  #  > 30% drawdown
                high_drawdown_scenarios.append(scenario_name)
            if result.recovery_time > 60:  #  > 60 days recovery
                long_recovery_scenarios.append(scenario_name)

        if high_drawdown_scenarios:
            recommendations.append(
                f"⚠️ Reduce position sizes - high drawdowns in {len(high_drawdown_scenarios)} scenarios"
            )

        if long_recovery_scenarios:
            recommendations.append(
                f"⚠️ Improve diversification - long recovery times in {len(long_recovery_scenarios)} scenarios"
            )

        # Strategy - specific recommendations
        strategy_failures = {}
        for scenario_name in failed_scenarios:
            result = results[scenario_name]
            for strategy, pnl in result.strategy_breakdown.items():
                if pnl < -1000:  # Significant loss
                    strategy_failures[strategy] = strategy_failures.get(strategy, 0) + 1

        for strategy, failure_count in strategy_failures.items():
            recommendations.append(
                f"🔧 Review {strategy} strategy - failed in {failure_count} stress scenarios"
            )

        return recommendations

    def _calculate_overall_risk_score(self, results: dict) -> float:
        """Calculate overall portfolio risk score (0 - 100, lower is better)."""
        if not results:
            return 100.0  # Maximum risk if no results

        total_risk = 0.0
        total_weight = 0.0

        for result in results.values():
            # Risk score based on P & L impact and probability
            pnl_impact = abs(result.portfolio_pnl)
            probability = result.risk_metrics.get("scenario_probability", 0.01)

            # Weight by probability
            weight = probability
            risk_score = pnl_impact * weight

            total_risk += risk_score
            total_weight += weight

        # Normalize to 0 - 100 scale
        if total_weight > 0:
            normalized_risk = (total_risk / total_weight) * 1000  # Scale factor
            return min(100.0, max(0.0, normalized_risk))

        return 50.0  # Default moderate risk


# Example usage and testing
if __name__ == "__main__":  # Create sample portfolio
    sample_portfolio = {
        "total_value": 100000.0,
        "strategies": {
            "wsb_dip_bot": {"exposure": 0.25},
            "earnings_protection": {"exposure": 0.20},
            "index_baseline": {"exposure": 0.15},
            "momentum_weeklies": {"exposure": 0.20},
            "debit_spreads": {"exposure": 0.10},
            "leaps_tracker": {"exposure": 0.10},
        },
    }

    # Initialize stress testing engine
    stress_tester = StressTesting2025()

    # Run comprehensive stress test
    print("Running Stress Tests...")
    report = stress_tester.run_comprehensive_stress_test(sample_portfolio)

    # Print results
    print("\nStress Test Report")
    print(" = " * 50)
    print(f"Compliance Status: {report.compliance_status}")
    print(f"Overall Risk Score: {report.overall_risk_score:.1f}/100")
    print(f"Test Date: {report.test_date}")

    print("\nRecommendations: ")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"{i}. {rec}")

    print("\nDetailed Results: ")
    for scenario_name, result in report.results.items():
        status = "✅ PASSED" if result.passed else "❌ FAILED"
        print(
            f"{scenario_name: 25} {status: 10} P & L: ${result.portfolio_pnl: 8,.0f} DD: {result.max_drawdown:.1%}"
        )
