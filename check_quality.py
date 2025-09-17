#!/usr/bin/env python3
"""
Quality check runner for WallStreetBots project.

This script runs a comprehensive quality assessment of the codebase.
"""

import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent / "utils"))

from code_quality_wrapper import CodeQualityWrapper

def main():
    """Run quality check for the project."""
    print("🚀 WallStreetBots - Code Quality Assessment")
    print("=" * 50)

    # Get project root
    project_root = Path(__file__).parent

    # Create quality checker
    checker = CodeQualityWrapper(project_root)

    # Run comprehensive check
    report = checker.run_comprehensive_check()

    # Print detailed report
    checker.print_report(report)

    # Save report
    report_path = checker.save_report(report)
    print(f"\n💾 Detailed report saved to: {report_path}")

    # Provide specific guidance based on results
    if report.score >= 90:
        print("\n🎉 PRODUCTION READY!")
        print("Your code meets excellent quality standards.")
    elif report.score >= 75:
        print("\n✅ GOOD QUALITY")
        print("Your code is ready for production with minor improvements.")
    elif report.score >= 60:
        print("\n⚠️  ACCEPTABLE")
        print("Consider addressing warnings before production deployment.")
    else:
        print("\n🔧 NEEDS WORK")
        print("Please address issues before production deployment.")

    # Exit with appropriate code
    if report.score < 60 or len(report.issues) > 0:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())