"""
Chaos Testing Framework for Trading Systems
Implements controlled failure injection to validate system resilience.
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
from contextlib import asynccontextmanager
import psutil
import signal
import subprocess
import numpy as np


class ChaosExperimentType(Enum):
    NETWORK_PARTITION = "network_partition"
    LATENCY_INJECTION = "latency_injection"
    MEMORY_PRESSURE = "memory_pressure"
    CPU_STRESS = "cpu_stress"
    DISK_FILL = "disk_fill"
    PROCESS_KILL = "process_kill"
    DATABASE_FAILURE = "database_failure"
    API_THROTTLING = "api_throttling"
    BROKER_DISCONNECT = "broker_disconnect"
    MARKET_DATA_DELAY = "market_data_delay"


class ExperimentStatus(Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class ChaosExperiment:
    """Definition of a chaos experiment."""
    name: str
    experiment_type: ChaosExperimentType
    description: str
    duration_seconds: int
    target_components: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    safety_checks: List[str] = field(default_factory=list)
    rollback_actions: List[str] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    schedule: Optional[Dict[str, Any]] = None


@dataclass
class ExperimentResult:
    """Result of a chaos experiment."""
    experiment_name: str
    status: ExperimentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    metrics_before: Dict[str, Any] = field(default_factory=dict)
    metrics_during: Dict[str, Any] = field(default_factory=dict)
    metrics_after: Dict[str, Any] = field(default_factory=dict)
    observations: List[str] = field(default_factory=list)
    failures_detected: List[str] = field(default_factory=list)
    recovery_time_seconds: Optional[int] = None
    hypothesis_validated: Optional[bool] = None
    notes: str = ""


class SafetyController:
    """Controls safety mechanisms during chaos experiments."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_experiments: Dict[str, ChaosExperiment] = {}
        self.emergency_stop = False
        self.safety_thresholds = {
            'max_concurrent_experiments': 2,
            'min_system_health_score': 0.7,
            'max_position_exposure_pct': 10.0,  # Max 10% of portfolio at risk
            'trading_hours_only': True
        }

    def can_start_experiment(self, experiment: ChaosExperiment) -> Tuple[bool, str]:
        """Check if experiment can safely start."""
        if self.emergency_stop:
            return False, "Emergency stop activated"

        if len(self.active_experiments) >= self.safety_thresholds['max_concurrent_experiments']:
            return False, f"Too many concurrent experiments: {len(self.active_experiments)}"

        # Check trading hours if required
        if self.safety_thresholds['trading_hours_only']:
            now = datetime.now()
            if now.hour < 9 or now.hour >= 16:  # Outside 9 AM - 4 PM ET
                return False, "Experiments only allowed during trading hours"

        # Check system health
        health_score = self._get_system_health_score()
        if health_score < self.safety_thresholds['min_system_health_score']:
            return False, f"System health too low: {health_score:.2f}"

        # Check position exposure
        exposure_pct = self._get_position_exposure_percentage()
        if exposure_pct > self.safety_thresholds['max_position_exposure_pct']:
            return False, f"Position exposure too high: {exposure_pct:.1f}%"

        return True, "Safety checks passed"

    def register_experiment(self, experiment: ChaosExperiment):
        """Register active experiment for safety tracking."""
        self.active_experiments[experiment.name] = experiment
        self.logger.info(f"Registered chaos experiment: {experiment.name}")

    def unregister_experiment(self, experiment_name: str):
        """Unregister completed experiment."""
        if experiment_name in self.active_experiments:
            del self.active_experiments[experiment_name]
            self.logger.info(f"Unregistered chaos experiment: {experiment_name}")

    def emergency_abort_all(self):
        """Emergency abort all running experiments."""
        self.emergency_stop = True
        self.logger.critical("EMERGENCY STOP: Aborting all chaos experiments")

        for experiment_name in list(self.active_experiments.keys()):
            self.logger.critical(f"Emergency aborting: {experiment_name}")
            # Trigger rollback for each experiment
            # This would be implemented by the specific experiment handlers

    def _get_system_health_score(self) -> float:
        """Calculate overall system health score (0-1)."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_score = max(0, 1 - (cpu_percent / 100))

            # Memory usage
            memory = psutil.virtual_memory()
            memory_score = max(0, 1 - (memory.percent / 100))

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_score = max(0, 1 - (disk.percent / 100))

            # Network connectivity (simplified)
            network_score = 1.0  # Would check broker/market data connections

            # Weighted average
            health_score = (cpu_score * 0.3 + memory_score * 0.3 +
                          disk_score * 0.2 + network_score * 0.2)

            return min(1.0, max(0.0, health_score))

        except Exception as e:
            self.logger.error(f"Error calculating health score: {e}")
            return 0.5  # Conservative default

    def _get_position_exposure_percentage(self) -> float:
        """Get current position exposure as percentage of portfolio."""
        # This would integrate with the actual position management system
        # For now, return a simulated value
        return random.uniform(1.0, 15.0)


class ExperimentExecutor:
    """Executes chaos experiments with proper controls."""

    def __init__(self, safety_controller: SafetyController):
        self.safety_controller = safety_controller
        self.logger = logging.getLogger(__name__)
        self.metrics_collector = MetricsCollector()

    async def run_experiment(self, experiment: ChaosExperiment) -> ExperimentResult:
        """Run a single chaos experiment."""
        result = ExperimentResult(
            experiment_name=experiment.name,
            status=ExperimentStatus.PLANNED,
            start_time=datetime.now()
        )

        try:
            # Safety check
            can_start, reason = self.safety_controller.can_start_experiment(experiment)
            if not can_start:
                result.status = ExperimentStatus.ABORTED
                result.notes = f"Safety check failed: {reason}"
                return result

            # Register experiment
            self.safety_controller.register_experiment(experiment)
            result.status = ExperimentStatus.RUNNING

            # Collect baseline metrics
            result.metrics_before = await self.metrics_collector.collect_metrics()

            # Execute experiment
            await self._execute_experiment_logic(experiment, result)

            # Wait for duration
            await asyncio.sleep(experiment.duration_seconds)

            # Collect during metrics
            result.metrics_during = await self.metrics_collector.collect_metrics()

            # Rollback experiment
            await self._rollback_experiment(experiment)

            # Wait for recovery
            recovery_start = time.time()
            await self._wait_for_recovery(experiment)
            result.recovery_time_seconds = int(time.time() - recovery_start)

            # Collect after metrics
            result.metrics_after = await self.metrics_collector.collect_metrics()

            # Analyze results
            result.hypothesis_validated = self._validate_hypothesis(experiment, result)
            result.status = ExperimentStatus.COMPLETED

        except Exception as e:
            self.logger.error(f"Experiment {experiment.name} failed: {e}")
            result.status = ExperimentStatus.FAILED
            result.notes = str(e)

            # Emergency rollback
            try:
                await self._rollback_experiment(experiment)
            except Exception as rollback_error:
                self.logger.error(f"Rollback failed: {rollback_error}")

        finally:
            result.end_time = datetime.now()
            result.duration_seconds = int((result.end_time - result.start_time).total_seconds())
            self.safety_controller.unregister_experiment(experiment.name)

        return result

    async def _execute_experiment_logic(self, experiment: ChaosExperiment, result: ExperimentResult):
        """Execute the specific chaos experiment logic."""
        experiment_type = experiment.experiment_type

        if experiment_type == ChaosExperimentType.NETWORK_PARTITION:
            await self._inject_network_partition(experiment, result)
        elif experiment_type == ChaosExperimentType.LATENCY_INJECTION:
            await self._inject_latency(experiment, result)
        elif experiment_type == ChaosExperimentType.MEMORY_PRESSURE:
            await self._inject_memory_pressure(experiment, result)
        elif experiment_type == ChaosExperimentType.CPU_STRESS:
            await self._inject_cpu_stress(experiment, result)
        elif experiment_type == ChaosExperimentType.PROCESS_KILL:
            await self._kill_process(experiment, result)
        elif experiment_type == ChaosExperimentType.BROKER_DISCONNECT:
            await self._simulate_broker_disconnect(experiment, result)
        elif experiment_type == ChaosExperimentType.MARKET_DATA_DELAY:
            await self._inject_market_data_delay(experiment, result)
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")

    async def _inject_network_partition(self, experiment: ChaosExperiment, result: ExperimentResult):
        """Inject network partition using iptables."""
        target_hosts = experiment.parameters.get('target_hosts', [])

        for host in target_hosts:
            # Block traffic to specific host
            cmd = f"sudo iptables -A OUTPUT -d {host} -j DROP"
            result.observations.append(f"Blocking traffic to {host}")
            # Would execute command in production

        experiment.rollback_actions.extend([
            f"sudo iptables -D OUTPUT -d {host} -j DROP" for host in target_hosts
        ])

    async def _inject_latency(self, experiment: ChaosExperiment, result: ExperimentResult):
        """Inject network latency using tc (traffic control)."""
        interface = experiment.parameters.get('interface', 'eth0')
        delay_ms = experiment.parameters.get('delay_ms', 100)

        cmd = f"sudo tc qdisc add dev {interface} root netem delay {delay_ms}ms"
        result.observations.append(f"Added {delay_ms}ms latency to {interface}")

        experiment.rollback_actions.append(f"sudo tc qdisc del dev {interface} root")

    async def _inject_memory_pressure(self, experiment: ChaosExperiment, result: ExperimentResult):
        """Inject memory pressure by allocating memory."""
        memory_mb = experiment.parameters.get('memory_mb', 1024)

        # This would be implemented with actual memory allocation
        result.observations.append(f"Allocated {memory_mb}MB of memory")

    async def _inject_cpu_stress(self, experiment: ChaosExperiment, result: ExperimentResult):
        """Inject CPU stress using stress tool."""
        cpu_cores = experiment.parameters.get('cpu_cores', 2)

        cmd = f"stress --cpu {cpu_cores} --timeout {experiment.duration_seconds}s"
        result.observations.append(f"Started CPU stress on {cpu_cores} cores")

    async def _kill_process(self, experiment: ChaosExperiment, result: ExperimentResult):
        """Kill specified process."""
        process_name = experiment.parameters.get('process_name')
        signal_type = experiment.parameters.get('signal', 'SIGTERM')

        result.observations.append(f"Killing process {process_name} with {signal_type}")

        # In production, would find and kill the actual process
        # For safety, we simulate this

    async def _simulate_broker_disconnect(self, experiment: ChaosExperiment, result: ExperimentResult):
        """Simulate broker API disconnection."""
        broker_endpoints = experiment.parameters.get('broker_endpoints', [])

        for endpoint in broker_endpoints:
            result.observations.append(f"Simulating disconnect from {endpoint}")
            # Would inject failures into broker connection layer

    async def _inject_market_data_delay(self, experiment: ChaosExperiment, result: ExperimentResult):
        """Inject delays in market data feed."""
        delay_seconds = experiment.parameters.get('delay_seconds', 5)

        result.observations.append(f"Injecting {delay_seconds}s delay in market data")
        # Would inject delays into market data processing pipeline

    async def _rollback_experiment(self, experiment: ChaosExperiment):
        """Execute rollback actions to restore normal operation."""
        for action in experiment.rollback_actions:
            try:
                self.logger.info(f"Rollback action: {action}")
                # Execute rollback command
                # In production, would actually execute these commands
            except Exception as e:
                self.logger.error(f"Rollback action failed: {action}, error: {e}")

    async def _wait_for_recovery(self, experiment: ChaosExperiment):
        """Wait for system to recover after rollback."""
        max_wait_seconds = 60
        check_interval = 5

        for _ in range(max_wait_seconds // check_interval):
            health_score = self.safety_controller._get_system_health_score()
            if health_score > 0.8:  # System recovered
                return
            await asyncio.sleep(check_interval)

    def _validate_hypothesis(self, experiment: ChaosExperiment, result: ExperimentResult) -> bool:
        """Validate experiment hypothesis based on results."""
        success_criteria = experiment.success_criteria

        if not success_criteria:
            return True  # No specific criteria

        # Check if system maintained minimum availability
        min_availability = success_criteria.get('min_availability_pct', 95)
        # Would calculate actual availability from metrics

        # Check if recovery time was acceptable
        max_recovery_seconds = success_criteria.get('max_recovery_seconds', 30)
        if result.recovery_time_seconds and result.recovery_time_seconds > max_recovery_seconds:
            result.failures_detected.append(f"Recovery took {result.recovery_time_seconds}s > {max_recovery_seconds}s")
            return False

        return len(result.failures_detected) == 0


class MetricsCollector:
    """Collects system metrics during chaos experiments."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics."""
        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'system': await self._collect_system_metrics(),
                'trading': await self._collect_trading_metrics(),
                'network': await self._collect_network_metrics(),
                'application': await self._collect_application_metrics()
            }
            return metrics
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level metrics."""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'load_average': psutil.getloadavg(),
            'process_count': len(psutil.pids())
        }

    async def _collect_trading_metrics(self) -> Dict[str, Any]:
        """Collect trading-specific metrics."""
        # This would integrate with actual trading systems
        return {
            'active_orders': random.randint(0, 10),
            'position_count': random.randint(5, 20),
            'pnl_unrealized': random.uniform(-1000, 1000),
            'risk_exposure': random.uniform(0.1, 0.8),
            'last_execution_latency_ms': random.uniform(10, 200)
        }

    async def _collect_network_metrics(self) -> Dict[str, Any]:
        """Collect network metrics."""
        net_io = psutil.net_io_counters()
        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv,
            'errors_in': net_io.errin,
            'errors_out': net_io.errout
        }

    async def _collect_application_metrics(self) -> Dict[str, Any]:
        """Collect application-specific metrics."""
        # This would integrate with application monitoring
        return {
            'active_strategies': random.randint(3, 8),
            'market_data_latency_ms': random.uniform(5, 50),
            'database_connection_pool': random.randint(5, 20),
            'cache_hit_rate': random.uniform(0.8, 0.99),
            'error_rate_per_minute': random.uniform(0, 5)
        }


class ChaosTestingSuite:
    """Complete chaos testing suite for trading systems."""

    def __init__(self):
        self.safety_controller = SafetyController()
        self.executor = ExperimentExecutor(self.safety_controller)
        self.logger = logging.getLogger(__name__)
        self.experiments: List[ChaosExperiment] = []
        self.results: List[ExperimentResult] = []
        self._setup_standard_experiments()

    def _setup_standard_experiments(self):
        """Setup standard chaos experiments for trading systems."""

        # Network partition experiment
        self.experiments.append(ChaosExperiment(
            name="broker_network_partition",
            experiment_type=ChaosExperimentType.NETWORK_PARTITION,
            description="Simulate network partition to broker API",
            duration_seconds=30,
            target_components=["broker_api"],
            parameters={
                'target_hosts': ['api.broker.com', 'data.broker.com']
            },
            success_criteria={
                'min_availability_pct': 95,
                'max_recovery_seconds': 15
            }
        ))

        # Market data delay experiment
        self.experiments.append(ChaosExperiment(
            name="market_data_delay",
            experiment_type=ChaosExperimentType.MARKET_DATA_DELAY,
            description="Inject delays in market data processing",
            duration_seconds=60,
            target_components=["market_data_feed"],
            parameters={
                'delay_seconds': 5
            },
            success_criteria={
                'max_stale_data_seconds': 10,
                'max_recovery_seconds': 5
            }
        ))

        # Memory pressure experiment
        self.experiments.append(ChaosExperiment(
            name="memory_pressure_test",
            experiment_type=ChaosExperimentType.MEMORY_PRESSURE,
            description="Apply memory pressure to test OOM handling",
            duration_seconds=45,
            target_components=["risk_engine"],
            parameters={
                'memory_mb': 2048
            },
            success_criteria={
                'no_oom_kills': True,
                'max_recovery_seconds': 20
            }
        ))

        # CPU stress experiment
        self.experiments.append(ChaosExperiment(
            name="cpu_stress_test",
            experiment_type=ChaosExperimentType.CPU_STRESS,
            description="Apply CPU stress to test performance degradation",
            duration_seconds=30,
            target_components=["strategy_engine"],
            parameters={
                'cpu_cores': 4
            },
            success_criteria={
                'max_latency_increase_pct': 200,
                'max_recovery_seconds': 10
            }
        ))

    async def run_experiment_by_name(self, experiment_name: str) -> ExperimentResult:
        """Run a specific experiment by name."""
        experiment = next((exp for exp in self.experiments if exp.name == experiment_name), None)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_name}")

        result = await self.executor.run_experiment(experiment)
        self.results.append(result)
        return result

    async def run_all_experiments(self) -> List[ExperimentResult]:
        """Run all configured experiments."""
        results = []
        for experiment in self.experiments:
            try:
                result = await self.executor.run_experiment(experiment)
                results.append(result)
                self.results.append(result)

                # Wait between experiments
                await asyncio.sleep(30)

            except Exception as e:
                self.logger.error(f"Failed to run experiment {experiment.name}: {e}")

        return results

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiment results."""
        if not self.results:
            return {'message': 'No experiments run yet'}

        total_experiments = len(self.results)
        successful_experiments = len([r for r in self.results if r.status == ExperimentStatus.COMPLETED])
        failed_experiments = len([r for r in self.results if r.status == ExperimentStatus.FAILED])

        avg_recovery_time = np.mean([r.recovery_time_seconds for r in self.results
                                   if r.recovery_time_seconds is not None])

        return {
            'total_experiments': total_experiments,
            'successful_experiments': successful_experiments,
            'failed_experiments': failed_experiments,
            'success_rate': successful_experiments / total_experiments if total_experiments > 0 else 0,
            'average_recovery_time_seconds': avg_recovery_time,
            'last_run': max([r.start_time for r in self.results]) if self.results else None,
            'resilience_score': self._calculate_resilience_score()
        }

    def _calculate_resilience_score(self) -> float:
        """Calculate overall system resilience score (0-1)."""
        if not self.results:
            return 0.0

        # Factors: success rate, recovery times, hypothesis validation
        success_rate = len([r for r in self.results if r.status == ExperimentStatus.COMPLETED]) / len(self.results)

        validated_hypotheses = len([r for r in self.results if r.hypothesis_validated])
        hypothesis_rate = validated_hypotheses / len(self.results) if self.results else 0

        # Fast recovery bonus
        fast_recoveries = len([r for r in self.results
                             if r.recovery_time_seconds and r.recovery_time_seconds < 30])
        recovery_rate = fast_recoveries / len(self.results) if self.results else 0

        # Weighted score
        resilience_score = (success_rate * 0.4 + hypothesis_rate * 0.4 + recovery_rate * 0.2)
        return min(1.0, resilience_score)


# Example usage
if __name__ == "__main__":
    import asyncio

    async def run_chaos_testing_demo():
        suite = ChaosTestingSuite()

        print("Starting chaos testing suite...")
        print(f"Configured experiments: {len(suite.experiments)}")

        # Run a single experiment
        print("\nRunning broker network partition experiment...")
        result = await suite.run_experiment_by_name("broker_network_partition")
        print(f"Result: {result.status.value}")
        print(f"Recovery time: {result.recovery_time_seconds}s")

        # Get summary
        summary = suite.get_experiment_summary()
        print("\nChaos testing summary:")
        print(f"Success rate: {summary['success_rate']:.1%}")
        print(f"Resilience score: {summary['resilience_score']:.2f}")

    asyncio.run(run_chaos_testing_demo())