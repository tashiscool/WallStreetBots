"""Comprehensive Phase 4 Testing.

Tests for backtesting, optimization, monitoring, and deployment.
"""

import os
import sys
import unittest
from datetime import datetime
from unittest.mock import Mock, patch

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from backend.tradingbot.core.production_config import ConfigManager
from backend.tradingbot.core.production_logging import ProductionLogger
from backend.tradingbot.phases.phase4_backtesting import (
    BacktestAnalyzer,
    BacktestConfig,
    BacktestEngine,
    BacktestPeriod,
    BacktestResults,
    BacktestTrade,
)
from backend.tradingbot.phases.phase4_deployment import (
    CICDManager,
    DeploymentConfig,
    DeploymentEnvironment,
    DeploymentStatus,
    DockerManager,
    KubernetesManager,
    Phase4Deployment,
)
from backend.tradingbot.phases.phase4_monitoring import (
    AlertLevel,
    AlertRule,
    Metric,
    MetricType,
    Phase4Monitoring,
)
from backend.tradingbot.phases.phase4_optimization import (
    OptimizationConfig,
    OptimizationMethod,
    OptimizationMetric,
    ParameterRange,
    StrategyOptimizer,
)


class TestPhase4Backtesting(unittest.TestCase):
    """Test Phase 4 backtesting functionality."""

    def setUp(self):
        self.logger = Mock(spec=ProductionLogger)
        self.config = Mock(spec=ConfigManager)
        self.config.trading = Mock()
        self.config.trading.universe = ["AAPL", "MSFT", "GOOGL"]

        self.backtest_engine = BacktestEngine(self.config, self.logger)

    def test_backtest_config_creation(self):
        """Test backtest configuration creation."""
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            initial_capital=100000.0,
            commission_per_trade=1.0,
            slippage_per_trade=0.001,
            benchmark_ticker="SPY",
            rebalance_frequency=BacktestPeriod.DAILY,
            risk_free_rate=0.02,
            max_positions=10,
            position_size_limit=0.1,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
        )

        self.assertEqual(config.initial_capital, 100000.0)
        self.assertEqual(config.commission_per_trade, 1.0)
        self.assertEqual(config.rebalance_frequency, BacktestPeriod.DAILY)

    def test_backtest_trade_creation(self):
        """Test backtest trade creation."""
        trade = BacktestTrade(
            ticker="AAPL",
            strategy="test_strategy",
            entry_date=datetime(2024, 1, 1),
            exit_date=datetime(2024, 1, 15),
            entry_price=150.0,
            exit_price=160.0,
            quantity=100,
            side="long",
            pnl=1000.0,
            commission=1.0,
            slippage=0.15,
            net_pnl=998.85,
            holding_period_days=14,
            return_pct=0.0667,
            exit_reason="take_profit",
        )

        self.assertEqual(trade.ticker, "AAPL")
        self.assertEqual(trade.net_pnl, 998.85)
        self.assertEqual(trade.holding_period_days, 14)

    @patch("backend.tradingbot.phase4_backtesting.BacktestEngine._run_simulation")
    async def test_backtest_execution(self, mock_simulation):
        """Test backtest execution."""
        mock_simulation.return_value = None

        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            initial_capital=100000.0,
            commission_per_trade=1.0,
            slippage_per_trade=0.001,
            benchmark_ticker="SPY",
            rebalance_frequency=BacktestPeriod.DAILY,
            risk_free_rate=0.02,
            max_positions=10,
            position_size_limit=0.1,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
        )

        strategy = Mock()

        result = await self.backtest_engine.run_backtest(strategy, config)

        self.assertIsInstance(result, BacktestResults)
        self.assertEqual(result.config, config)
        mock_simulation.assert_called_once()

    def test_backtest_analyzer_report_generation(self):
        """Test backtest analyzer report generation."""
        analyzer = BacktestAnalyzer(self.logger)

        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            initial_capital=100000.0,
            commission_per_trade=1.0,
            slippage_per_trade=0.001,
            benchmark_ticker="SPY",
            rebalance_frequency=BacktestPeriod.DAILY,
            risk_free_rate=0.02,
            max_positions=10,
            position_size_limit=0.1,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
        )

        results = BacktestResults(
            config=config,
            start_date=config.start_date,
            end_date=config.end_date,
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            win_rate=0.6,
            total_return=0.15,
            annualized_return=0.18,
            volatility=0.20,
            sharpe_ratio=0.9,
            max_drawdown=0.05,
            calmar_ratio=3.6,
            profit_factor=1.5,
            avg_win=500.0,
            avg_loss=-300.0,
            total_commission=10.0,
            total_slippage=5.0,
            net_profit=15000.0,
            benchmark_return=0.10,
            alpha=0.08,
            beta=1.2,
        )

        report = analyzer.generate_report(results)

        self.assertIn("summary", report)
        self.assertIn("trading_stats", report)
        self.assertIn("benchmark_comparison", report)
        self.assertEqual(report["summary"]["total_return"], "15.00%")


class TestPhase4Optimization(unittest.TestCase):
    """Test Phase 4 optimization functionality."""

    def setUp(self):
        self.logger = Mock(spec=ProductionLogger)
        self.config = Mock(spec=ConfigManager)
        self.backtest_engine = Mock(spec=BacktestEngine)

        self.optimizer = StrategyOptimizer(self.backtest_engine, self.config, self.logger)

    def test_parameter_range_creation(self):
        """Test parameter range creation."""
        param_range = ParameterRange(
            name="lookback_period", min_value=5, max_value=50, step=5, param_type="int"
        )

        self.assertEqual(param_range.name, "lookback_period")
        self.assertEqual(param_range.min_value, 5)
        self.assertEqual(param_range.max_value, 50)
        self.assertEqual(param_range.param_type, "int")

    def test_optimization_config_creation(self):
        """Test optimization configuration creation."""
        param_ranges = [
            ParameterRange("param1", 1, 10, 1, "int"),
            ParameterRange("param2", 0.1, 1.0, 0.1, "float"),
        ]

        backtest_config = Mock(spec=BacktestConfig)

        config = OptimizationConfig(
            method=OptimizationMethod.GRID_SEARCH,
            metric=OptimizationMetric.SHARPE_RATIO,
            max_iterations=100,
            parameter_ranges=param_ranges,
            backtest_config=backtest_config,
        )

        self.assertEqual(config.method, OptimizationMethod.GRID_SEARCH)
        self.assertEqual(config.metric, OptimizationMetric.SHARPE_RATIO)
        self.assertEqual(len(config.parameter_ranges), 2)

    def test_parameter_combination_generation(self):
        """Test parameter combination generation."""
        param_ranges = [
            ParameterRange("param1", 1, 3, 1, "int"),
            ParameterRange("param2", 0.1, 0.3, 0.1, "float"),
        ]

        combinations = self.optimizer._generate_parameter_combinations(param_ranges)

        self.assertGreater(len(combinations), 0)
        self.assertIsInstance(combinations[0], dict)
        self.assertIn("param1", combinations[0])
        self.assertIn("param2", combinations[0])

    def test_random_parameter_generation(self):
        """Test random parameter generation."""
        param_ranges = [
            ParameterRange("param1", 1, 10, 1, "int"),
            ParameterRange("param2", 0.1, 1.0, 0.1, "float"),
            ParameterRange("param3", True, False, 1, "bool"),
        ]

        params = self.optimizer._generate_random_parameters(param_ranges)

        self.assertIn("param1", params)
        self.assertIn("param2", params)
        self.assertIn("param3", params)
        self.assertIsInstance(params["param1"], int)
        self.assertIsInstance(params["param2"], float)
        self.assertIsInstance(params["param3"], bool)

    def test_score_calculation(self):
        """Test optimization score calculation."""
        mock_results = Mock(spec=BacktestResults)
        mock_results.sharpe_ratio = 1.5
        mock_results.total_return = 0.15
        mock_results.max_drawdown = 0.05

        # Test Sharpe ratio metric
        score = self.optimizer._calculate_score(mock_results, OptimizationMetric.SHARPE_RATIO)
        self.assertEqual(score, 1.5)

        # Test total return metric
        score = self.optimizer._calculate_score(mock_results, OptimizationMetric.TOTAL_RETURN)
        self.assertEqual(score, 0.15)

        # Test max drawdown metric (should be negative)
        score = self.optimizer._calculate_score(mock_results, OptimizationMetric.MAX_DRAWDOWN)
        self.assertEqual(score, -0.05)


class TestPhase4Monitoring(unittest.TestCase):
    """Test Phase 4 monitoring functionality."""

    def setUp(self):
        self.logger = Mock(spec=ProductionLogger)
        self.config = Mock(spec=ConfigManager)

        self.monitoring = Phase4Monitoring(self.config, self.logger)

    def test_metric_creation(self):
        """Test metric creation."""
        metric = Metric(
            name="test_metric",
            value=42.0,
            timestamp=datetime.now(),
            labels={"strategy": "test"},
            metric_type=MetricType.GAUGE,
        )

        self.assertEqual(metric.name, "test_metric")
        self.assertEqual(metric.value, 42.0)
        self.assertEqual(metric.metric_type, MetricType.GAUGE)

    def test_alert_rule_creation(self):
        """Test alert rule creation."""
        rule = AlertRule(
            name="test_alert",
            condition="test_metric",
            threshold=50.0,
            comparison="gt",
            level=AlertLevel.WARNING,
            cooldown_minutes=5,
        )

        self.assertEqual(rule.name, "test_alert")
        self.assertEqual(rule.threshold, 50.0)
        self.assertEqual(rule.level, AlertLevel.WARNING)

    def test_metrics_collector(self):
        """Test metrics collector functionality."""
        collector = self.monitoring.metrics_collector

        # Record a metric
        metric = Metric(
            name="test_metric", value=100.0, timestamp=datetime.now(), metric_type=MetricType.GAUGE
        )

        collector.record_metric(metric)

        # Get metric value
        value = collector.get_metric_value("test_metric")
        self.assertEqual(value, 100.0)

    def test_alert_manager(self):
        """Test alert manager functionality."""
        alert_manager = self.monitoring.alert_manager

        # Add alert rule
        rule = AlertRule(
            name="test_rule",
            condition="test_metric",
            threshold=50.0,
            comparison="gt",
            level=AlertLevel.WARNING,
        )

        alert_manager.add_alert_rule(rule)
        self.assertIn("test_rule", alert_manager.alert_rules)

        # Remove alert rule
        alert_manager.remove_alert_rule("test_rule")
        self.assertNotIn("test_rule", alert_manager.alert_rules)

    def test_dashboard_data(self):
        """Test dashboard data generation."""
        dashboard_data = self.monitoring.get_dashboard_data()

        self.assertIn("timestamp", dashboard_data)
        self.assertIn("system_health", dashboard_data)
        self.assertIn("trading_metrics", dashboard_data)
        self.assertIn("active_alerts", dashboard_data)

    def test_custom_metric_addition(self):
        """Test custom metric addition."""
        self.monitoring.add_custom_metric("custom_metric", 123.45, MetricType.GAUGE)

        value = self.monitoring.metrics_collector.get_metric_value("custom_metric")
        self.assertEqual(value, 123.45)


class TestPhase4Deployment(unittest.TestCase):
    """Test Phase 4 deployment functionality."""

    def setUp(self):
        self.logger = Mock(spec=ProductionLogger)
        self.config = Mock(spec=ConfigManager)

        self.deployment = Phase4Deployment(self.config, self.logger)

    def test_deployment_config_creation(self):
        """Test deployment configuration creation."""
        config = DeploymentConfig(
            environment=DeploymentEnvironment.PRODUCTION,
            version="v1.0.0",
            docker_image="wallstreetbots",
            replicas=3,
            cpu_limit="1000m",
            memory_limit="2Gi",
            health_check_path="/health",
        )

        self.assertEqual(config.environment, DeploymentEnvironment.PRODUCTION)
        self.assertEqual(config.version, "v1.0.0")
        self.assertEqual(config.replicas, 3)

    def test_docker_manager(self):
        """Test Docker manager functionality."""
        docker_manager = DockerManager(self.logger)

        # Test image building (mock)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            result = docker_manager.build_image("Dockerfile", "test - image", "latest")
            self.assertTrue(result)

    def test_kubernetes_manager(self):
        """Test Kubernetes manager functionality."""
        k8s_manager = KubernetesManager(self.logger)

        config = DeploymentConfig(
            environment=DeploymentEnvironment.PRODUCTION,
            version="v1.0.0",
            docker_image="wallstreetbots",
            replicas=3,
            cpu_limit="1000m",
            memory_limit="2Gi",
            health_check_path="/health",
        )

        # Test deployment YAML generation
        yaml = k8s_manager._generate_deployment_yaml(config)
        self.assertIn("apiVersion: apps / v1", yaml)
        self.assertIn("kind: Deployment", yaml)
        self.assertIn("wallstreetbots", yaml)

    def test_cicd_manager(self):
        """Test CI / CD manager functionality."""
        cicd_manager = CICDManager(self.logger)

        # Test test running (mock)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            result = cicd_manager.run_tests()
            self.assertTrue(result)

    @patch("backend.tradingbot.phase4_deployment.DeploymentManager._run_cicd_pipeline")
    @patch("backend.tradingbot.phase4_deployment.DockerManager.build_image")
    @patch("backend.tradingbot.phase4_deployment.DockerManager.push_image")
    @patch("backend.tradingbot.phase4_deployment.KubernetesManager.create_deployment")
    @patch("backend.tradingbot.phase4_deployment.DeploymentManager._health_check")
    async def test_deployment_process(
        self, mock_health, mock_k8s, mock_push, mock_build, mock_cicd
    ):
        """Test deployment process."""
        # Mock all dependencies to return success
        mock_cicd.return_value = True
        mock_build.return_value = True
        mock_push.return_value = True
        mock_k8s.return_value = True
        mock_health.return_value = True

        config = DeploymentConfig(
            environment=DeploymentEnvironment.STAGING,
            version="v1.0.0",
            docker_image="wallstreetbots",
            replicas=1,
            cpu_limit="500m",
            memory_limit="1Gi",
            health_check_path="/health",
        )

        result = await self.deployment.deployment_manager.deploy(config)

        self.assertEqual(result.status, DeploymentStatus.SUCCESS)
        self.assertIsNotNone(result.deployment_id)


class TestPhase4Integration(unittest.TestCase):
    """Test Phase 4 integration functionality."""

    def setUp(self):
        self.logger = Mock(spec=ProductionLogger)
        self.config = Mock(spec=ConfigManager)
        self.config.trading = Mock()
        self.config.trading.universe = ["AAPL", "MSFT", "GOOGL"]

    def test_end_to_end_workflow(self):
        """Test end - to - end Phase 4 workflow."""
        # This would test the complete workflow from backtesting to deployment
        # For now, we'll test that all components can be instantiated together

        # Initialize all Phase 4 components
        from backend.tradingbot.phases.phase4_backtesting import BacktestEngine
        from backend.tradingbot.phases.phase4_deployment import Phase4Deployment
        from backend.tradingbot.phases.phase4_monitoring import Phase4Monitoring
        from backend.tradingbot.phases.phase4_optimization import StrategyOptimizer

        backtest_engine = BacktestEngine(self.config, self.logger)
        optimizer = StrategyOptimizer(backtest_engine, self.config, self.logger)
        monitoring = Phase4Monitoring(self.config, self.logger)
        deployment = Phase4Deployment(self.config, self.logger)

        # Verify all components are initialized
        self.assertIsNotNone(backtest_engine)
        self.assertIsNotNone(optimizer)
        self.assertIsNotNone(monitoring)
        self.assertIsNotNone(deployment)


def run_phase4_tests():
    """Run all Phase 4 tests."""
    print("Running Phase 4 Comprehensive Tests...")
    print(" = " * 50)

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestPhase4Backtesting,
        TestPhase4Optimization,
        TestPhase4Monitoring,
        TestPhase4Deployment,
        TestPhase4Integration,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print("\n" + " = " * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFailures: ")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")

    if result.errors:
        print("\nErrors: ")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")

    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nPhase 4 Tests {'PASSED' if success else 'FAILED'}")

    return success


if __name__ == "__main__":
    success = run_phase4_tests()
    sys.exit(0 if success else 1)
