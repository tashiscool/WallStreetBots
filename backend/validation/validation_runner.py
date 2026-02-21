"""Validation Runner - Entry point for comprehensive strategy validation."""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .factor_analysis import AlphaFactorAnalyzer
from .regime_testing import RegimeValidator
from .statistical_rigor.reality_check import MultipleTestingController

logger = logging.getLogger(__name__)


class ValidationRunner:
    """Main validation runner for comprehensive strategy evaluation."""

    def __init__(self):
        self.validators = {
            "reality_check": MultipleTestingController(),
            "factor_analyzer": AlphaFactorAnalyzer(),
            "regime_validator": RegimeValidator(),
        }

    def run_comprehensive_validation(
        self,
        strategy_returns: dict[str, pd.Series],
        benchmark_returns: pd.Series,
        market_data: pd.DataFrame,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
    ) -> dict[str, Any]:
        """Run the full validation suite."""
        logger.info(
            "Starting comprehensive validation from %s to %s",
            start_date,
            end_date,
        )

        results: dict[str, Any] = {}

        logger.info("Running Reality Check / SPA Test...")
        try:
            results["reality_check"] = self.validators[
                "reality_check"
            ].run_comprehensive_testing(strategy_returns, benchmark_returns)
        except Exception as exc:
            logger.error("Reality check failed: %s", exc)
            results["reality_check"] = {"error": str(exc)}

        logger.info("Running Factor Analysis...")
        try:
            factor_df = self.validators["factor_analyzer"].create_factor_proxies(
                market_data
            )
            factor_results = {}
            for strategy_name, returns in strategy_returns.items():
                try:
                    factor_results[strategy_name] = self.validators[
                        "factor_analyzer"
                    ].run_factor_regression(returns, factor_df)
                except Exception as exc:
                    logger.warning(
                        "Factor analysis failed for %s: %s", strategy_name, exc
                    )
            results["factor_analysis"] = factor_results
        except Exception as exc:
            logger.error("Factor analysis failed: %s", exc)
            results["factor_analysis"] = {"error": str(exc)}

        logger.info("Running Regime Testing...")
        try:
            regime_results = {}
            for strategy_name, returns in strategy_returns.items():
                try:
                    regime_results[strategy_name] = self.validators[
                        "regime_validator"
                    ].test_edge_persistence(returns, market_data)
                except Exception as exc:
                    logger.warning(
                        "Regime testing failed for %s: %s", strategy_name, exc
                    )
            results["regime_testing"] = regime_results
        except Exception as exc:
            logger.error("Regime testing failed: %s", exc)
            results["regime_testing"] = {"error": str(exc)}

        results["summary"] = self._generate_validation_summary(results)
        logger.info("Comprehensive validation completed")
        return results

    def _generate_validation_summary(self, results: dict[str, Any]) -> dict[str, Any]:
        """Generate a compact summary of validation outcomes."""
        summary: dict[str, Any] = {
            "validation_timestamp": datetime.now().isoformat(),
            "validation_modules_run": list(results.keys()),
            "overall_status": "COMPLETED",
            "recommendations": [],
        }

        reality = results.get("reality_check", {})
        if isinstance(reality, dict) and "error" not in reality:
            recommendation = reality.get("recommendation", {})
            significant = recommendation.get("significant_strategies", [])
            if significant:
                summary["recommendations"].append(
                    f"Multiple-testing significant strategies: {significant}"
                )
            else:
                summary["recommendations"].append(
                    "No strategies passed multiple-testing significance checks"
                )

        factor = results.get("factor_analysis", {})
        if isinstance(factor, dict) and "error" not in factor:
            significant_strategies: list[str] = []
            for strategy_name, factor_result in factor.items():
                is_significant = False
                if hasattr(factor_result, "alpha_significant"):
                    is_significant = bool(
                        getattr(factor_result, "alpha_significant", False)
                    )
                elif isinstance(factor_result, dict):
                    is_significant = bool(factor_result.get("alpha_significant", False))
                if is_significant:
                    significant_strategies.append(strategy_name)

            if significant_strategies:
                summary["recommendations"].append(
                    f"Significant alpha detected in: {significant_strategies}"
                )
            else:
                summary["recommendations"].append(
                    "No statistically significant alpha detected"
                )

        regime = results.get("regime_testing", {})
        if isinstance(regime, dict) and "error" not in regime:
            robust_strategies = [
                strategy_name
                for strategy_name, result in regime.items()
                if isinstance(result, dict) and result.get("edge_is_robust", False)
            ]
            if robust_strategies:
                summary["recommendations"].append(
                    f"Regime-robust strategies: {robust_strategies}"
                )
            else:
                summary["recommendations"].append("No regime-robust strategies found")

        return summary


def _setup_django() -> bool:
    """Best-effort Django setup for loading validation data from the database."""
    try:
        import django
        from django.conf import settings

        if not settings.configured:
            os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
        django.setup()
        return True
    except Exception as exc:
        logger.warning("Django setup failed for DB loading: %s", exc)
        return False


def _load_strategy_returns_from_db(
    strategy_name: str, start_date: str, end_date: str
) -> pd.Series | None:
    """Load strategy returns from persisted signal validation outcomes."""
    if not _setup_django():
        return None

    try:
        from django.utils.dateparse import parse_datetime
        from backend.tradingbot.models.models import SignalValidationHistory

        start_dt = parse_datetime(f"{start_date}T00:00:00")
        end_dt = parse_datetime(f"{end_date}T23:59:59")

        queryset = SignalValidationHistory.objects.filter(
            strategy_name=strategy_name,
            trade_pnl_percent__isnull=False,
            outcome_recorded_at__isnull=False,
        )
        if start_dt:
            queryset = queryset.filter(outcome_recorded_at__gte=start_dt)
        if end_dt:
            queryset = queryset.filter(outcome_recorded_at__lte=end_dt)

        records = list(queryset.values("outcome_recorded_at", "trade_pnl_percent"))
        if not records:
            return None

        frame = pd.DataFrame.from_records(records)
        frame["date"] = pd.to_datetime(frame["outcome_recorded_at"]).dt.date
        frame["return"] = frame["trade_pnl_percent"] / 100.0
        daily_returns = frame.groupby("date")["return"].mean()
        daily_returns.index = pd.to_datetime(daily_returns.index)
        daily_returns.name = strategy_name
        return daily_returns.sort_index()
    except Exception as exc:
        logger.warning("Failed to load strategy returns from DB: %s", exc)
        return None


def _load_strategy_returns_from_csv(
    returns_csv: str, strategy_name: str
) -> pd.Series | None:
    """Load strategy returns from CSV with `date` and `return` columns."""
    try:
        frame = pd.read_csv(returns_csv)
        if "date" not in frame or "return" not in frame:
            raise ValueError("CSV must contain `date` and `return` columns")

        returns = pd.Series(
            frame["return"].astype(float).values,
            index=pd.to_datetime(frame["date"]),
            name=strategy_name,
        ).sort_index()
        return returns
    except Exception as exc:
        logger.warning("Failed to load strategy returns from CSV: %s", exc)
        return None


def _load_market_and_benchmark_data(
    start_date: str, end_date: str
) -> tuple[pd.Series, pd.DataFrame]:
    """Load benchmark returns and market regime inputs from market data providers."""
    try:
        import yfinance as yf

        spy = yf.download("SPY", start=start_date, end=end_date, progress=False)
        vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
        tnx = yf.download("^TNX", start=start_date, end=end_date, progress=False)

        if spy.empty:
            raise ValueError("No SPY data returned from provider")

        spy_close = spy["Close"].squeeze()
        if isinstance(spy_close, pd.DataFrame):
            spy_close = spy_close.iloc[:, 0]

        benchmark_returns = spy_close.pct_change().dropna().rename("benchmark")

        market_data = pd.DataFrame(index=spy_close.index)
        market_data["SPY"] = spy_close
        if not vix.empty:
            market_data["VIX"] = vix["Close"].squeeze().reindex(market_data.index).ffill()
        else:
            market_data["VIX"] = 20.0
        if not tnx.empty:
            market_data["DGS10"] = (
                tnx["Close"].squeeze().reindex(market_data.index).ffill() / 10.0
            )
        else:
            market_data["DGS10"] = 4.0

        return benchmark_returns, market_data.ffill().dropna()
    except Exception as exc:
        logger.warning("Market data provider load failed: %s", exc)
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        benchmark_returns = pd.Series(0.0, index=dates, name="benchmark")
        market_data = pd.DataFrame(
            {"SPY": 500.0, "VIX": 20.0, "DGS10": 4.0},
            index=dates,
        )
        return benchmark_returns, market_data


def main():
    """Main entry point for validation runner."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive strategy validation"
    )
    parser.add_argument(
        "--start", default="2020-01-01", help="Start date for validation"
    )
    parser.add_argument("--end", default="2024-12-31", help="End date for validation")
    parser.add_argument("--strategy", required=True, help="Strategy name to validate")
    parser.add_argument(
        "--returns-csv",
        help="Optional CSV file containing strategy returns (columns: date, return)",
    )
    parser.add_argument("--output", help="Output file for results")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    runner = ValidationRunner()

    logger.info("Running validation for strategy: %s", args.strategy)

    strategy_returns = None
    if args.returns_csv:
        strategy_returns = _load_strategy_returns_from_csv(
            args.returns_csv, args.strategy
        )
    if strategy_returns is None:
        strategy_returns = _load_strategy_returns_from_db(
            args.strategy, args.start, args.end
        )

    if strategy_returns is None or strategy_returns.empty:
        raise RuntimeError(
            "No strategy return data available. Provide --returns-csv or persist "
            "SignalValidationHistory trade outcomes for this strategy."
        )

    benchmark_returns, market_data = _load_market_and_benchmark_data(
        args.start, args.end
    )

    results = runner.run_comprehensive_validation(
        strategy_returns={args.strategy: strategy_returns},
        benchmark_returns=benchmark_returns,
        market_data=market_data,
        start_date=args.start,
        end_date=args.end,
    )
    results["metadata"] = {
        "strategy": args.strategy,
        "start_date": args.start,
        "end_date": args.end,
        "returns_points": len(strategy_returns),
        "generated_at": datetime.now().isoformat(),
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(results, output_file, indent=2, default=str)

    print(f"Validation completed for {args.strategy}")
    print(f"Summary: {results.get('summary', {})}")


if __name__ == "__main__":
    main()
