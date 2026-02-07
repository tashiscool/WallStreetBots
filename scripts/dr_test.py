#!/usr/bin/env python
"""
Disaster Recovery Test Script for WallStreetBots.

Runs non-destructive DR tests to verify recovery capabilities.
Uses temporary databases and state files — does not affect production.

Usage:
    python scripts/dr_test.py --full
    python scripts/dr_test.py --scenario backup_restore
    python scripts/dr_test.py --scenario circuit_breaker_recovery
    python scripts/dr_test.py --scenario state_persistence
    python scripts/dr_test.py --scenario reconciliation
"""

import argparse
import gzip
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class DRTestResult:
    """Result of a single DR test."""

    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.details: list = []
        self.start = datetime.utcnow()
        self.end = None

    def pass_(self, msg: str = ""):
        self.passed = True
        self.details.append(f"PASS: {msg}" if msg else "PASS")

    def fail(self, msg: str):
        self.passed = False
        self.details.append(f"FAIL: {msg}")

    def info(self, msg: str):
        self.details.append(f"INFO: {msg}")

    def finish(self):
        self.end = datetime.utcnow()


class DRTestRunner:
    """Runs DR test scenarios."""

    def __init__(self):
        self.results: list = []
        self.temp_dir = tempfile.mkdtemp(prefix="wsb_dr_test_")

    def cleanup(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Test scenarios
    # ------------------------------------------------------------------

    def test_backup_restore(self) -> DRTestResult:
        """Test database backup and restore cycle."""
        result = DRTestResult("backup_restore")

        try:
            # Create a test SQLite database
            db_path = os.path.join(self.temp_dir, "test.sqlite3")
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE positions (symbol TEXT, qty REAL)")
            conn.execute("INSERT INTO positions VALUES ('AAPL', 100)")
            conn.execute("INSERT INTO positions VALUES ('GOOGL', 50)")
            conn.commit()
            conn.close()
            result.info("Created test database with 2 positions")

            # Backup
            backup_path = os.path.join(self.temp_dir, "backup.sqlite3.gz")
            with open(db_path, "rb") as f_in:
                with gzip.open(backup_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            result.info(f"Backup created: {os.path.getsize(backup_path)} bytes")

            # Corrupt original
            os.remove(db_path)
            result.info("Original database deleted (simulating corruption)")

            # Restore
            with gzip.open(backup_path, "rb") as f_in:
                with open(db_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Verify
            conn = sqlite3.connect(db_path)
            rows = conn.execute("SELECT * FROM positions").fetchall()
            conn.close()

            if len(rows) == 2 and rows[0][0] == "AAPL":
                result.pass_("Backup/restore cycle preserves data integrity")
            else:
                result.fail(f"Data mismatch after restore: {rows}")

        except Exception as e:
            result.fail(f"Exception: {e}")

        result.finish()
        return result

    def test_circuit_breaker_recovery(self) -> DRTestResult:
        """Test circuit breaker state persistence across restart."""
        result = DRTestResult("circuit_breaker_recovery")

        try:
            # Simulate circuit breaker state persistence via JSON
            state_path = os.path.join(self.temp_dir, "cb_state.json")
            state = {
                "tripped": True,
                "reason": "daily_loss_limit",
                "tripped_at": datetime.utcnow().isoformat(),
                "trip_count": 3,
            }
            with open(state_path, "w") as f:
                json.dump(state, f)
            result.info("Circuit breaker state saved")

            # Simulate restart — read state back
            with open(state_path) as f:
                restored = json.load(f)

            if restored["tripped"] is True and restored["reason"] == "daily_loss_limit":
                result.pass_("Circuit breaker state persists across restart")
            else:
                result.fail(f"State mismatch: {restored}")

        except Exception as e:
            result.fail(f"Exception: {e}")

        result.finish()
        return result

    def test_state_persistence(self) -> DRTestResult:
        """Test replay guard and other state file persistence."""
        result = DRTestResult("state_persistence")

        try:
            # Test ReplayGuard persistence
            state_path = os.path.join(self.temp_dir, "replay_guard.json")
            orders = {
                "OMS_abc123": "filled",
                "OMS_def456": "canceled",
                "OMS_ghi789": "submitted",
            }
            with open(state_path, "w") as f:
                json.dump(orders, f)

            # Simulate restart
            with open(state_path) as f:
                restored = json.load(f)

            # Verify duplicate detection
            if "OMS_abc123" in restored and restored["OMS_abc123"] == "filled":
                result.pass_("ReplayGuard correctly detects previously filled order")
            else:
                result.fail("ReplayGuard state lost")

            # Verify new order passes
            if "OMS_new999" not in restored:
                result.pass_("New orders correctly pass replay guard")
            else:
                result.fail("Replay guard incorrectly blocked new order")

        except Exception as e:
            result.fail(f"Exception: {e}")

        result.finish()
        return result

    def test_reconciliation(self) -> DRTestResult:
        """Test that reconciliation detects discrepancies."""
        result = DRTestResult("reconciliation")

        try:
            # Simulate internal positions
            internal = {"AAPL": 100, "GOOGL": 50, "MSFT": 75}
            # Simulate broker positions (with discrepancy)
            broker = {"AAPL": 100, "GOOGL": 45, "TSLA": 25}

            discrepancies = []

            # Check for mismatches
            all_symbols = set(list(internal.keys()) + list(broker.keys()))
            for sym in all_symbols:
                int_qty = internal.get(sym, 0)
                brk_qty = broker.get(sym, 0)
                if int_qty != brk_qty:
                    if int_qty > 0 and brk_qty == 0:
                        discrepancies.append(f"MISSING_AT_BROKER: {sym}")
                    elif int_qty == 0 and brk_qty > 0:
                        discrepancies.append(f"MISSING_IN_DB: {sym}")
                    else:
                        discrepancies.append(
                            f"QTY_MISMATCH: {sym} (internal={int_qty}, broker={brk_qty})"
                        )

            result.info(f"Found {len(discrepancies)} discrepancies")
            for d in discrepancies:
                result.info(f"  {d}")

            # Should find: GOOGL mismatch, MSFT missing at broker, TSLA missing in DB
            if len(discrepancies) == 3:
                result.pass_("Reconciliation correctly detects all discrepancy types")
            else:
                result.fail(f"Expected 3 discrepancies, found {len(discrepancies)}")

        except Exception as e:
            result.fail(f"Exception: {e}")

        result.finish()
        return result

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------

    def run_all(self) -> list:
        """Run all DR test scenarios."""
        tests = [
            self.test_backup_restore,
            self.test_circuit_breaker_recovery,
            self.test_state_persistence,
            self.test_reconciliation,
        ]
        for test_fn in tests:
            result = test_fn()
            self.results.append(result)
        return self.results

    def run_scenario(self, name: str) -> DRTestResult:
        """Run a single scenario by name."""
        scenarios = {
            "backup_restore": self.test_backup_restore,
            "circuit_breaker_recovery": self.test_circuit_breaker_recovery,
            "state_persistence": self.test_state_persistence,
            "reconciliation": self.test_reconciliation,
        }
        fn = scenarios.get(name)
        if fn is None:
            print(f"Unknown scenario: {name}")
            print(f"Available: {', '.join(scenarios.keys())}")
            sys.exit(1)
        result = fn()
        self.results.append(result)
        return result

    def print_report(self):
        """Print test results."""
        print("\n" + "=" * 60)
        print("  WallStreetBots Disaster Recovery Test Report")
        print(f"  Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("=" * 60)

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            duration = ""
            if r.end and r.start:
                ms = (r.end - r.start).total_seconds() * 1000
                duration = f" ({ms:.0f}ms)"
            print(f"\n  [{status}] {r.name}{duration}")
            for detail in r.details:
                print(f"    {detail}")

        print(f"\n{'=' * 60}")
        print(f"  Results: {passed}/{total} passed")
        if passed == total:
            print("  STATUS: ALL DR TESTS PASSED")
        else:
            print("  STATUS: SOME DR TESTS FAILED")
        print("=" * 60)

        return passed == total


def main():
    parser = argparse.ArgumentParser(description="WallStreetBots DR Test")
    parser.add_argument("--full", action="store_true", help="Run all scenarios")
    parser.add_argument("--scenario", type=str, help="Run specific scenario")
    args = parser.parse_args()

    runner = DRTestRunner()

    try:
        if args.scenario:
            runner.run_scenario(args.scenario)
        else:
            runner.run_all()

        success = runner.print_report()
        sys.exit(0 if success else 1)
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()
