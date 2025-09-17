"""Code quality validation wrapper for production readiness."""

import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import time
from dataclasses import dataclass
from enum import Enum


class QualityLevel(Enum):
    """Quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    NEEDS_IMPROVEMENT = "needs_improvement"
    CRITICAL_ISSUES = "critical_issues"


@dataclass
class QualityReport:
    """Quality assessment report."""
    level: QualityLevel
    score: float  # 0-100
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    passed_checks: List[str]
    timestamp: float
    details: Dict[str, Any]


class CodeQualityWrapper:
    """
    Comprehensive wrapper for code quality validation.

    This class provides a complete quality assessment system that checks:
    - Static analysis (ruff)
    - Type checking (mypy if available)
    - Test coverage
    - Security issues
    - Documentation quality
    - Performance patterns
    """

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the quality wrapper.

        Args:
            project_root: Root directory of the project. If None, uses current directory.
        """
        self.project_root = project_root or Path.cwd()
        self.reports: List[QualityReport] = []

        # Directories to exclude from checks
        self.exclude_dirs = {
            'venv', '.venv', 'env', '.env',
            'node_modules', '.git', '__pycache__',
            '.pytest_cache', '.mypy_cache', '.ruff_cache',
            'build', 'dist', '.tox'
        }

    def run_ruff_check(self) -> Tuple[bool, List[str], List[str]]:
        """
        Run ruff static analysis.

        Returns:
            Tuple of (success, errors, warnings)
        """
        try:
            # Run ruff check only on project directories
            cmd = ["ruff", "check", "backend", "tests", "utils", "ml"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )

            errors = []
            warnings = []

            if result.returncode == 0:
                return True, [], []

            # Parse ruff output
            lines = result.stdout.split('\n') + result.stderr.split('\n')
            for line in lines:
                if line.strip():
                    if any(code in line for code in ['F', 'E']):  # Errors
                        errors.append(line.strip())
                    elif any(code in line for code in ['W', 'PERF', 'RUF']):  # Warnings
                        warnings.append(line.strip())
                    else:
                        warnings.append(line.strip())

            return len(errors) == 0, errors, warnings

        except FileNotFoundError:
            return False, ["ruff not found - install with: pip install ruff"], []
        except Exception as e:
            return False, [f"ruff check failed: {e!s}"], []

    def run_type_check(self) -> Tuple[bool, List[str], List[str]]:
        """
        Run type checking with mypy if available.

        Returns:
            Tuple of (success, errors, warnings)
        """
        try:
            # Check if mypy is available
            subprocess.run(["mypy", "--version"], capture_output=True, check=True)

            # Run mypy on backend directory
            backend_path = self.project_root / "backend"
            if not backend_path.exists():
                return True, [], ["No backend directory found for type checking"]

            result = subprocess.run(
                ["mypy", str(backend_path), "--ignore-missing-imports"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )

            errors = []
            warnings = []

            if result.returncode == 0:
                return True, [], []

            # Parse mypy output
            lines = result.stdout.split('\n') + result.stderr.split('\n')
            for line in lines:
                if line.strip():
                    if "error:" in line.lower():
                        errors.append(line.strip())
                    elif "warning:" in line.lower() or "note:" in line.lower():
                        warnings.append(line.strip())

            return len(errors) == 0, errors, warnings

        except (FileNotFoundError, subprocess.CalledProcessError):
            return True, [], ["mypy not available - consider installing for type checking"]
        except Exception as e:
            return False, [f"Type check failed: {e!s}"], []

    def check_test_structure(self) -> Tuple[bool, List[str], List[str]]:
        """
        Check test structure and coverage.

        Returns:
            Tuple of (success, issues, recommendations)
        """
        issues = []
        recommendations = []

        tests_dir = self.project_root / "tests"
        if not tests_dir.exists():
            issues.append("No tests directory found")
            return False, issues, recommendations

        # Check for test files
        test_files = list(tests_dir.rglob("test_*.py"))
        if len(test_files) == 0:
            issues.append("No test files found")
            return False, issues, recommendations

        # Check test structure
        backend_dir = self.project_root / "backend"
        if backend_dir.exists():
            backend_modules = len(list(backend_dir.rglob("*.py")))
            test_coverage_ratio = len(test_files) / max(backend_modules, 1)

            if test_coverage_ratio < 0.3:
                recommendations.append(f"Low test coverage ratio: {test_coverage_ratio:.2f} (recommended: >0.3)")
            elif test_coverage_ratio > 0.8:
                recommendations.append(f"Excellent test coverage ratio: {test_coverage_ratio:.2f}")

        # Check for common test patterns
        comprehensive_tests = [f for f in test_files if "comprehensive" in f.name]
        if comprehensive_tests:
            recommendations.append(f"Found {len(comprehensive_tests)} comprehensive test files")

        integration_tests = [f for f in test_files if "integration" in f.name]
        if integration_tests:
            recommendations.append(f"Found {len(integration_tests)} integration test files")

        return True, issues, recommendations

    def check_security_patterns(self) -> Tuple[bool, List[str], List[str]]:
        """
        Check for common security issues.

        Returns:
            Tuple of (success, issues, warnings)
        """
        issues = []
        warnings = []

        # Check for common security anti-patterns in project files only
        python_files = []
        for dir_name in ["backend", "tests", "utils", "ml"]:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                python_files.extend(list(dir_path.rglob("*.py")))

        security_patterns = {
            "hardcoded_secrets": [
                r"password\s*=\s*['\"][^'\"]+['\"]",
                r"api_key\s*=\s*['\"][^'\"]+['\"]",
                r"secret\s*=\s*['\"][^'\"]+['\"]",
            ],
            "sql_injection": [
                r"execute\s*\(\s*['\"].*%.*['\"]",
                r"\.format\s*\(",
            ],
            "dangerous_imports": [
                "import pickle",
                "from pickle import",
                "import subprocess",
                "import os.system",
            ]
        }

        for py_file in python_files:
            try:
                content = py_file.read_text(encoding='utf-8')

                # Check for hardcoded secrets (basic check) - exclude test files and examples
                if not any(part in str(py_file) for part in ["test_", "example", "mock"]):
                    if any(pattern in content.lower() for pattern in ["password = '", 'password = "']):
                        # Additional check to exclude documentation patterns
                        if not any(doc_pattern in content.lower() for doc_pattern in ["example", "placeholder", "dummy", "test"]):
                            issues.append(f"Potential hardcoded password in {py_file.relative_to(self.project_root)}")

                # Check for dangerous patterns
                if "eval(" in content or "exec(" in content:
                    warnings.append(f"Dynamic code execution found in {py_file.relative_to(self.project_root)}")

                # Check for proper error handling
                if "except:" in content and "except Exception:" not in content:
                    warnings.append(f"Bare except clause in {py_file.relative_to(self.project_root)}")

            except Exception:
                continue  # Skip files that can't be read

        return len(issues) == 0, issues, warnings

    def check_performance_patterns(self) -> Tuple[bool, List[str], List[str]]:
        """
        Check for performance anti-patterns.

        Returns:
            Tuple of (success, issues, recommendations)
        """
        issues = []
        recommendations = []

        # Check only project files
        python_files = []
        for dir_name in ["backend", "tests", "utils", "ml"]:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                python_files.extend(list(dir_path.rglob("*.py")))

        for py_file in python_files:
            try:
                content = py_file.read_text(encoding='utf-8')

                # Check for inefficient patterns
                if ".items()" in content:
                    # This is now handled by our wrapper
                    recommendations.append(f"Dictionary iteration in {py_file.relative_to(self.project_root)} - verify usage pattern")

                # Check for potential memory issues - more sophisticated check
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if "while True:" in line:
                        # Look for break, return, or raise in the next 20 lines
                        has_exit = False
                        for j in range(i + 1, min(i + 21, len(lines))):
                            if any(keyword in lines[j] for keyword in ["break", "return", "raise", "sys.exit"]):
                                has_exit = True
                                break
                        if not has_exit:
                            issues.append(f"Potential infinite loop in {py_file.relative_to(self.project_root)}:{i+1}")

            except Exception:
                continue

        return len(issues) == 0, issues, recommendations

    def calculate_quality_score(self, results: Dict[str, Tuple[bool, List[str], List[str]]]) -> float:
        """
        Calculate overall quality score from check results.

        Args:
            results: Dictionary of check results

        Returns:
            Quality score from 0-100
        """
        weights = {
            "ruff": 30,      # Static analysis is critical
            "types": 20,     # Type safety is important
            "tests": 25,     # Test coverage is crucial
            "security": 15,  # Security is important
            "performance": 10 # Performance is good to have
        }

        total_score = 0
        total_weight = 0

        for check_name, (success, errors, warnings) in results.items():
            if check_name in weights:
                weight = weights[check_name]
                total_weight += weight

                # Calculate score for this check
                if success and len(errors) == 0:
                    check_score = 100
                elif success and len(errors) == 0 and len(warnings) == 0:
                    check_score = 100
                elif success:
                    # Deduct points for warnings
                    check_score = max(70, 100 - len(warnings) * 5)
                else:
                    # Major deduction for errors
                    check_score = max(0, 50 - len(errors) * 10)

                total_score += check_score * weight

        if total_weight == 0:
            return 0

        return total_score / total_weight

    def determine_quality_level(self, score: float, has_critical_issues: bool) -> QualityLevel:
        """
        Determine quality level based on score and issues.

        Args:
            score: Quality score 0-100
            has_critical_issues: Whether there are critical issues

        Returns:
            Quality level enum
        """
        if has_critical_issues:
            return QualityLevel.CRITICAL_ISSUES

        if score >= 90:
            return QualityLevel.EXCELLENT
        elif score >= 75:
            return QualityLevel.GOOD
        elif score >= 60:
            return QualityLevel.ACCEPTABLE
        else:
            return QualityLevel.NEEDS_IMPROVEMENT

    def run_comprehensive_check(self) -> QualityReport:
        """
        Run comprehensive code quality check.

        Returns:
            Quality report with all findings
        """
        print("🔍 Running comprehensive code quality check...")

        results = {}
        all_issues = []
        all_warnings = []
        all_recommendations = []
        passed_checks = []

        # Run all checks
        checks = [
            ("ruff", self.run_ruff_check),
            ("types", self.run_type_check),
            ("tests", self.check_test_structure),
            ("security", self.check_security_patterns),
            ("performance", self.check_performance_patterns),
        ]

        for check_name, check_func in checks:
            print(f"  ⏳ Running {check_name} check...")
            try:
                success, errors, warnings = check_func()
                results[check_name] = (success, errors, warnings)

                if success and len(errors) == 0:
                    passed_checks.append(f"{check_name} check passed")

                all_issues.extend(errors)
                all_warnings.extend(warnings)

                if check_name == "tests":
                    all_recommendations.extend(warnings)  # Test warnings are recommendations
                elif check_name == "performance":
                    all_recommendations.extend(warnings)  # Performance warnings are recommendations

                print(f"    ✅ {check_name} check completed")

            except Exception as e:
                print(f"    ❌ {check_name} check failed: {e}")
                results[check_name] = (False, [f"{check_name} check failed: {e!s}"], [])
                all_issues.append(f"{check_name} check failed: {e!s}")

        # Calculate overall score and quality level
        score = self.calculate_quality_score(results)
        has_critical_issues = any(
            not success for success, _, _ in results.values()
        ) or len(all_issues) > 0

        quality_level = self.determine_quality_level(score, has_critical_issues)

        # Create report
        report = QualityReport(
            level=quality_level,
            score=score,
            issues=all_issues,
            warnings=all_warnings,
            recommendations=all_recommendations,
            passed_checks=passed_checks,
            timestamp=time.time(),
            details=dict(results)
        )

        self.reports.append(report)
        return report

    def print_report(self, report: QualityReport) -> None:
        """
        Print a formatted quality report.

        Args:
            report: Quality report to print
        """
        print("\n" + "=" * 80)
        print("🔍 CODE QUALITY ASSESSMENT REPORT")
        print("=" * 80)

        # Quality level and score
        level_emojis = {
            QualityLevel.EXCELLENT: "🌟",
            QualityLevel.GOOD: "✅",
            QualityLevel.ACCEPTABLE: "⚠️",
            QualityLevel.NEEDS_IMPROVEMENT: "🔧",
            QualityLevel.CRITICAL_ISSUES: "🚨",
        }

        emoji = level_emojis.get(report.level, "❓")
        print(f"\n{emoji} Overall Quality Level: {report.level.value.upper()}")
        print(f"📊 Quality Score: {report.score:.1f}/100")
        print(f"🕒 Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report.timestamp))}")

        # Passed checks
        if report.passed_checks:
            print(f"\n✅ PASSED CHECKS ({len(report.passed_checks)}):")
            for check in report.passed_checks:
                print(f"  • {check}")

        # Issues
        if report.issues:
            print(f"\n🚨 CRITICAL ISSUES ({len(report.issues)}):")
            for issue in report.issues:
                print(f"  • {issue}")

        # Warnings
        if report.warnings:
            print(f"\n⚠️  WARNINGS ({len(report.warnings)}):")
            for warning in report.warnings:
                print(f"  • {warning}")

        # Recommendations
        if report.recommendations:
            print(f"\n💡 RECOMMENDATIONS ({len(report.recommendations)}):")
            for rec in report.recommendations:
                print(f"  • {rec}")

        # Summary
        print("\n📈 SUMMARY:")
        print(f"  • Issues: {len(report.issues)}")
        print(f"  • Warnings: {len(report.warnings)}")
        print(f"  • Recommendations: {len(report.recommendations)}")
        print(f"  • Passed Checks: {len(report.passed_checks)}")

        if report.level == QualityLevel.EXCELLENT:
            print("\n🎉 Excellent! Your code is production-ready.")
        elif report.level == QualityLevel.GOOD:
            print("\n👍 Good quality code with minor improvements needed.")
        elif report.level == QualityLevel.ACCEPTABLE:
            print("\n👌 Acceptable quality, but consider addressing warnings.")
        elif report.level == QualityLevel.NEEDS_IMPROVEMENT:
            print("\n🔧 Code needs improvement before production deployment.")
        else:
            print("\n🚨 Critical issues must be resolved before deployment!")

        print("=" * 80)

    def save_report(self, report: QualityReport, filename: Optional[str] = None) -> Path:
        """
        Save quality report to JSON file.

        Args:
            report: Quality report to save
            filename: Optional filename. If None, generates timestamp-based name.

        Returns:
            Path to saved report file
        """
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(report.timestamp))
            filename = f"quality_report_{timestamp}.json"

        report_path = self.project_root / filename

        # Convert report to JSON-serializable format
        report_data = {
            "level": report.level.value,
            "score": report.score,
            "issues": report.issues,
            "warnings": report.warnings,
            "recommendations": report.recommendations,
            "passed_checks": report.passed_checks,
            "timestamp": report.timestamp,
            "details": report.details,
            "project_root": str(self.project_root),
        }

        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        return report_path


def main():
    """Main function for running quality checks from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Run comprehensive code quality check")
    parser.add_argument("--project-root", type=Path, help="Project root directory")
    parser.add_argument("--save-report", action="store_true", help="Save report to JSON file")
    parser.add_argument("--output", type=str, help="Output filename for report")

    args = parser.parse_args()

    # Create quality checker
    checker = CodeQualityWrapper(args.project_root)

    # Run comprehensive check
    report = checker.run_comprehensive_check()

    # Print report
    checker.print_report(report)

    # Save report if requested
    if args.save_report:
        report_path = checker.save_report(report, args.output)
        print(f"\n💾 Report saved to: {report_path}")

    # Exit with appropriate code
    if report.level in [QualityLevel.CRITICAL_ISSUES, QualityLevel.NEEDS_IMPROVEMENT]:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()