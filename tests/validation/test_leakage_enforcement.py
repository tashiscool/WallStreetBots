"""Tests for AST leakage detection and enforcement modes."""

import logging
import pytest
import pandas as pd

from backend.validation.statistical_rigor.leakage_detection import (
    TemporalGuard,
    LeakageDetectionSuite,
    EnforcementMode,
    LeakageViolationError,
    LeakageViolation,
    SurvivorshipGuard,
)


@pytest.fixture
def guard():
    return TemporalGuard()


class TestASTShiftDetection:
    def test_negative_shift_detected(self, guard):
        source = "df['future'] = df['price'].shift(-1)"
        violations = guard.analyze_source_code(source, "test_func")
        types = [v.violation_type for v in violations]
        assert 'negative_shift_lookahead' in types

    def test_positive_shift_ok(self, guard):
        source = "df['lagged'] = df['price'].shift(1)"
        violations = guard.analyze_source_code(source, "test_func")
        shift_violations = [v for v in violations if 'shift' in v.violation_type]
        assert len(shift_violations) == 0

    def test_shift_minus_two(self, guard):
        source = "x = series.shift(-2)"
        violations = guard.analyze_source_code(source, "test_func")
        types = [v.violation_type for v in violations]
        assert 'negative_shift_lookahead' in types


class TestBfillDetection:
    def test_bfill_detected(self, guard):
        source = "df = df.bfill()"
        violations = guard.analyze_source_code(source, "test_func")
        types = [v.violation_type for v in violations]
        assert 'backfill_leakage' in types

    def test_backfill_method_detected(self, guard):
        source = "df = df.backfill()"
        violations = guard.analyze_source_code(source, "test_func")
        types = [v.violation_type for v in violations]
        assert 'backfill_leakage' in types

    def test_fillna_bfill_detected(self, guard):
        source = "df = df.fillna(method='bfill')"
        violations = guard.analyze_source_code(source, "test_func")
        types = [v.violation_type for v in violations]
        assert 'backfill_leakage' in types

    def test_ffill_ok(self, guard):
        source = "df = df.ffill()"
        violations = guard.analyze_source_code(source, "test_func")
        backfill = [v for v in violations if 'backfill' in v.violation_type]
        assert len(backfill) == 0


class TestCenterTrueDetection:
    def test_rolling_center_true(self, guard):
        source = "df['ma'] = df['price'].rolling(20, center=True).mean()"
        violations = guard.analyze_source_code(source, "test_func")
        types = [v.violation_type for v in violations]
        assert 'rolling_center_leakage' in types

    def test_rolling_center_false_ok(self, guard):
        source = "df['ma'] = df['price'].rolling(20, center=False).mean()"
        violations = guard.analyze_source_code(source, "test_func")
        center = [v for v in violations if 'center' in v.violation_type]
        assert len(center) == 0


class TestEnforcementModes:
    def test_audit_mode_silent(self):
        """AUDIT mode collects violations without logging or raising."""
        suite = LeakageDetectionSuite(enforcement_mode=EnforcementMode.AUDIT)
        # Add a critical violation manually
        suite.temporal_guard.violations.append(LeakageViolation(
            violation_type='test_critical',
            description='test violation',
            severity='critical',
            detected_at=pd.Timestamp.now(),
        ))
        result = suite.validate_strategy(
            strategy=None,
            backtest_start=pd.Timestamp('2020-01-01'),
            backtest_end=pd.Timestamp('2021-01-01'),
            strategy_universe=['AAPL'],
        )
        assert result['passed'] is False
        # No exception raised

    def test_warn_mode_logs(self, caplog):
        """WARN mode should log warnings for violations."""
        suite = LeakageDetectionSuite(enforcement_mode=EnforcementMode.WARN)
        suite.temporal_guard.violations.append(LeakageViolation(
            violation_type='test_warn',
            description='warn test',
            severity='warning',
            detected_at=pd.Timestamp.now(),
        ))
        with caplog.at_level(logging.WARNING):
            suite.validate_strategy(
                strategy=None,
                backtest_start=pd.Timestamp('2020-01-01'),
                backtest_end=pd.Timestamp('2021-01-01'),
                strategy_universe=['AAPL'],
            )
        assert any('warn test' in r.message for r in caplog.records)

    def test_strict_mode_raises_on_critical(self):
        """STRICT mode should raise LeakageViolationError on critical violations."""
        suite = LeakageDetectionSuite(enforcement_mode=EnforcementMode.STRICT)
        suite.temporal_guard.violations.append(LeakageViolation(
            violation_type='test_critical',
            description='critical violation',
            severity='critical',
            detected_at=pd.Timestamp.now(),
        ))
        with pytest.raises(LeakageViolationError, match="critical"):
            suite.validate_strategy(
                strategy=None,
                backtest_start=pd.Timestamp('2020-01-01'),
                backtest_end=pd.Timestamp('2021-01-01'),
                strategy_universe=['AAPL'],
            )

    def test_strict_mode_no_raise_on_warning_only(self):
        """STRICT mode should not raise if only warnings (no critical)."""
        suite = LeakageDetectionSuite(enforcement_mode=EnforcementMode.STRICT)
        suite.temporal_guard.violations.append(LeakageViolation(
            violation_type='test_warning',
            description='just a warning',
            severity='warning',
            detected_at=pd.Timestamp.now(),
        ))
        result = suite.validate_strategy(
            strategy=None,
            backtest_start=pd.Timestamp('2020-01-01'),
            backtest_end=pd.Timestamp('2021-01-01'),
            strategy_universe=['AAPL'],
        )
        # Should not raise — only critical triggers exception
        assert result['passed'] is True  # warnings don't fail


class TestMixedSeverity:
    def test_violation_severity_classification(self, guard):
        source = """
df['future'] = df['price'].shift(-1)
df['filled'] = df['val'].bfill()
df['ma'] = df['price'].rolling(20, center=True).mean()
"""
        violations = guard.analyze_source_code(source, "mixed")
        assert len(violations) >= 3
        assert all(v.severity == 'critical' for v in violations)
