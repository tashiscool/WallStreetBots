"""
Report Delivery Service for WallStreetBots.

Generates PDF performance reports and delivers them via email.
Uses PDFReportGenerator for report creation and Django's EmailMessage
for delivery.
"""

import logging
import uuid
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from django.contrib.auth.models import User
from django.core.mail import EmailMessage
from django.conf import settings
from django.utils import timezone

from backend.tradingbot.analysis.pdf_report import PDFReportConfig, PDFReportGenerator
from backend.tradingbot.analysis.report_templates import ReportTemplates

logger = logging.getLogger(__name__)

# Map report type names to template factory methods
REPORT_TYPE_MAP = {
    'weekly': ReportTemplates.weekly_performance,
    'monthly': ReportTemplates.monthly_detailed,
    'quarterly': ReportTemplates.quarterly_review,
    'yearly': ReportTemplates.year_end_tax,
}


class ReportDeliveryService:
    """
    Service for generating and delivering PDF performance reports.

    Handles report generation from trading data, email delivery with
    PDF attachments, and scheduled report distribution to users.
    """

    def __init__(self):
        self.generator = None  # Lazy-initialized per report config

    def generate_report(
        self,
        user: User,
        report_type: str = 'weekly',
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        strategy_name: Optional[str] = None,
    ) -> bytes:
        """
        Generate a PDF report for a user.

        Args:
            user: Django User to generate report for
            report_type: One of 'weekly', 'monthly', 'quarterly', 'yearly'
            start_date: Report period start date (auto-calculated if None)
            end_date: Report period end date (defaults to today)
            strategy_name: Specific strategy name, or None for all strategies

        Returns:
            PDF file bytes

        Raises:
            ValueError: If report_type is not recognized
        """
        if report_type not in REPORT_TYPE_MAP:
            raise ValueError(
                f"Unknown report type '{report_type}'. "
                f"Valid types: {', '.join(REPORT_TYPE_MAP.keys())}"
            )

        # Get config from template
        config = REPORT_TYPE_MAP[report_type]()

        # Calculate date range if not provided
        end_date = end_date or date.today()
        if start_date is None:
            start_date = self._calculate_start_date(report_type, end_date)

        # Build subtitle with date range
        config.subtitle = f"Period: {start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}"

        # Fetch trading data for the user
        returns, trades, benchmark_returns = self._fetch_trading_data(
            user, start_date, end_date, strategy_name
        )

        # Initialize generator with config
        self.generator = PDFReportGenerator(config)

        # Generate PDF
        display_name = strategy_name or 'All Strategies'
        pdf_bytes = self.generator.generate(
            returns=returns,
            trades=trades,
            benchmark_returns=benchmark_returns,
            strategy_name=display_name,
        )

        return pdf_bytes

    def email_report(
        self,
        user: User,
        pdf_bytes: bytes,
        report_type: str = 'weekly',
    ) -> bool:
        """
        Send a PDF report to a user via email.

        Args:
            user: Django User to send report to
            pdf_bytes: PDF file content as bytes
            report_type: Report type name for the email subject

        Returns:
            True if email was sent successfully, False otherwise
        """
        if not user.email:
            logger.warning(f"User {user.username} has no email address configured")
            return False

        report_label = report_type.title()
        subject = f"WallStreetBots {report_label} Performance Report - {date.today().strftime('%B %d, %Y')}"

        body = (
            f"Hello {user.first_name or user.username},\n\n"
            f"Please find your {report_label.lower()} performance report attached.\n\n"
            f"This report was generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}.\n\n"
            f"Best regards,\n"
            f"WallStreetBots"
        )

        from_email = getattr(settings, 'DEFAULT_FROM_EMAIL', 'reports@wallstreetbots.com')

        try:
            email = EmailMessage(
                subject=subject,
                body=body,
                from_email=from_email,
                to=[user.email],
            )

            # Determine filename
            date_str = date.today().strftime('%Y-%m-%d')
            filename = f"wallstreetbots_{report_type}_report_{date_str}.pdf"

            # Attach PDF
            email.attach(filename, pdf_bytes, 'application/pdf')

            email.send(fail_silently=False)
            logger.info(f"Sent {report_type} report to {user.email}")
            return True

        except Exception as e:
            logger.error(f"Failed to email report to {user.email}: {e}")
            return False

    def schedule_weekly_reports(self) -> Dict[str, Any]:
        """
        Generate and email weekly reports for all users with email_performance_reports enabled.

        Returns:
            Summary dict with counts of successes and failures
        """
        from backend.tradingbot.models.models import UserProfile

        results = {
            'total_users': 0,
            'reports_generated': 0,
            'emails_sent': 0,
            'errors': [],
        }

        # Find users who have opted in to performance report emails
        profiles = UserProfile.objects.filter(
            email_performance_reports=True,
        ).select_related('user')

        results['total_users'] = profiles.count()

        for profile in profiles:
            user = profile.user
            if not user.email:
                results['errors'].append(f"User {user.username}: no email address")
                continue

            try:
                # Generate weekly report
                pdf_bytes = self.generate_report(
                    user=user,
                    report_type='weekly',
                )
                results['reports_generated'] += 1

                # Send email
                sent = self.email_report(
                    user=user,
                    pdf_bytes=pdf_bytes,
                    report_type='weekly',
                )
                if sent:
                    results['emails_sent'] += 1
                else:
                    results['errors'].append(f"User {user.username}: email send failed")

            except Exception as e:
                logger.exception(f"Error generating weekly report for {user.username}")
                results['errors'].append(f"User {user.username}: {str(e)}")

        logger.info(
            f"Weekly reports: {results['emails_sent']}/{results['total_users']} sent, "
            f"{len(results['errors'])} errors"
        )
        return results

    @staticmethod
    def get_report_types() -> List[Dict[str, str]]:
        """
        Return available report templates with descriptions.

        Returns:
            List of dicts with 'id', 'name', and 'description' keys
        """
        return [
            {
                'id': 'weekly',
                'name': 'Weekly Performance',
                'description': 'Compact weekly summary with equity curve, drawdown, trade log, and risk metrics.',
            },
            {
                'id': 'monthly',
                'name': 'Monthly Detailed',
                'description': 'Comprehensive monthly report with all charts, heatmaps, and full trade history.',
            },
            {
                'id': 'quarterly',
                'name': 'Quarterly Review',
                'description': 'In-depth quarterly analysis with complete performance breakdown and extended trade log.',
            },
            {
                'id': 'yearly',
                'name': 'Year-End Tax',
                'description': 'Annual report focused on trade log and tax-relevant information.',
            },
        ]

    def _calculate_start_date(self, report_type: str, end_date: date) -> date:
        """Calculate the start date based on report type."""
        if report_type == 'weekly':
            return end_date - timedelta(days=7)
        elif report_type == 'monthly':
            return end_date - timedelta(days=30)
        elif report_type == 'quarterly':
            return end_date - timedelta(days=90)
        elif report_type == 'yearly':
            return date(end_date.year, 1, 1)
        else:
            return end_date - timedelta(days=7)

    def _fetch_trading_data(
        self,
        user: User,
        start_date: date,
        end_date: date,
        strategy_name: Optional[str] = None,
    ) -> Tuple[pd.Series, Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Fetch trading data from the database for report generation.

        Returns:
            Tuple of (returns Series, trades DataFrame, benchmark_returns Series)
        """
        from backend.tradingbot.models.models import BacktestRun, TradeReasoning

        # Try to build returns from completed backtest runs
        returns = self._build_returns_from_backtests(user, start_date, end_date, strategy_name)

        # Build trades DataFrame from TradeReasoning entries
        trades = self._build_trades_dataframe(user, start_date, end_date, strategy_name)

        # Generate benchmark returns (SPY proxy)
        benchmark_returns = self._get_benchmark_returns(start_date, end_date)

        return returns, trades, benchmark_returns

    def _build_returns_from_backtests(
        self,
        user: User,
        start_date: date,
        end_date: date,
        strategy_name: Optional[str] = None,
    ) -> pd.Series:
        """
        Build a daily returns series from backtest run data or trade history.

        Falls back to synthetic returns derived from trade P&L if no
        backtest equity curve is available.
        """
        from backend.tradingbot.models.models import BacktestRun, TradeReasoning

        # Try to get equity curve from most recent completed backtest
        backtest_filter = {
            'user': user,
            'status': 'completed',
        }
        if strategy_name:
            backtest_filter['strategy_name'] = strategy_name

        latest_backtest = (
            BacktestRun.objects.filter(**backtest_filter)
            .order_by('-completed_at')
            .first()
        )

        if latest_backtest and latest_backtest.equity_curve:
            # equity_curve is stored as JSON: list of {date, value} or similar
            try:
                curve_data = latest_backtest.equity_curve
                if isinstance(curve_data, list) and len(curve_data) > 1:
                    dates = [entry.get('date', entry.get('timestamp', '')) for entry in curve_data]
                    values = [float(entry.get('value', entry.get('equity', 0))) for entry in curve_data]
                    equity = pd.Series(values, index=pd.to_datetime(dates))
                    equity = equity.sort_index()

                    # Filter to date range
                    mask = (equity.index.date >= start_date) & (equity.index.date <= end_date)
                    equity = equity[mask]

                    if len(equity) > 1:
                        returns = equity.pct_change().dropna()
                        returns.name = 'returns'
                        return returns
            except Exception as e:
                logger.warning(f"Failed to parse equity curve from backtest: {e}")

        # Fallback: Build synthetic daily returns from trade P&L
        trade_filter = {
            'entry_time__date__gte': start_date,
            'entry_time__date__lte': end_date,
        }
        if strategy_name:
            trade_filter['strategy_name'] = strategy_name

        trade_reasonings = TradeReasoning.objects.filter(**trade_filter).order_by('entry_time')

        if trade_reasonings.exists():
            # Group P&L by day
            pnl_by_day = {}
            for tr in trade_reasonings:
                day = tr.entry_time.date()
                pnl = float(tr.actual_pnl or 0)
                pnl_by_day[day] = pnl_by_day.get(day, 0) + pnl

            # Convert to returns assuming $100,000 portfolio
            portfolio_value = 100000.0
            dates = sorted(pnl_by_day.keys())
            daily_returns = []
            for d in dates:
                ret = pnl_by_day[d] / portfolio_value
                daily_returns.append(ret)
                portfolio_value += pnl_by_day[d]

            if daily_returns:
                returns = pd.Series(
                    daily_returns,
                    index=pd.DatetimeIndex(dates),
                    name='returns',
                )
                return returns

        # Final fallback: generate date range with zero returns
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        returns = pd.Series(
            np.zeros(len(date_range)),
            index=date_range,
            name='returns',
        )
        return returns

    def _build_trades_dataframe(
        self,
        user: User,
        start_date: date,
        end_date: date,
        strategy_name: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Build a trades DataFrame from TradeReasoning entries."""
        from backend.tradingbot.models.models import TradeReasoning

        trade_filter = {
            'entry_time__date__gte': start_date,
            'entry_time__date__lte': end_date,
        }
        if strategy_name:
            trade_filter['strategy_name'] = strategy_name

        trade_reasonings = TradeReasoning.objects.filter(**trade_filter).order_by('entry_time')

        if not trade_reasonings.exists():
            return None

        rows = []
        for tr in trade_reasonings:
            rows.append({
                'symbol': tr.symbol,
                'side': tr.direction if hasattr(tr, 'direction') else 'long',
                'qty': tr.quantity if hasattr(tr, 'quantity') else 1,
                'entry_price': float(tr.entry_price) if hasattr(tr, 'entry_price') and tr.entry_price else 0,
                'exit_price': float(tr.exit_price) if hasattr(tr, 'exit_price') and tr.exit_price else 0,
                'pnl': float(tr.actual_pnl or 0),
                'entry_date': tr.entry_time.strftime('%Y-%m-%d') if tr.entry_time else '',
                'exit_date': tr.exit_time.strftime('%Y-%m-%d') if hasattr(tr, 'exit_time') and tr.exit_time else '',
            })

        return pd.DataFrame(rows) if rows else None

    def _get_benchmark_returns(
        self,
        start_date: date,
        end_date: date,
    ) -> Optional[pd.Series]:
        """
        Get SPY benchmark returns for the period.

        Attempts to fetch from stored market data; falls back to a
        synthetic approximation if unavailable.
        """
        try:
            from backend.tradingbot.models.models import StrategyPerformanceSnapshot

            # Try to get benchmark data from snapshots
            snapshots = StrategyPerformanceSnapshot.objects.filter(
                strategy_name='index_baseline',
                snapshot_date__gte=start_date,
                snapshot_date__lte=end_date,
                period='daily',
            ).order_by('snapshot_date')

            if snapshots.count() > 5:
                dates = [s.snapshot_date for s in snapshots]
                returns = [float(s.benchmark_return_pct) / 100.0 for s in snapshots]
                return pd.Series(
                    returns,
                    index=pd.DatetimeIndex(dates),
                    name='benchmark',
                )
        except Exception as e:
            logger.debug(f"Could not fetch benchmark data from snapshots: {e}")

        # Synthetic fallback: approximately SPY-like daily returns
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        if len(date_range) == 0:
            return None

        # Generate synthetic SPY-like returns (~10% annual, ~16% vol)
        np.random.seed(42)  # Reproducible for consistent reports
        daily_mean = 0.10 / 252
        daily_std = 0.16 / np.sqrt(252)
        synthetic = np.random.normal(daily_mean, daily_std, len(date_range))

        return pd.Series(
            synthetic,
            index=date_range,
            name='benchmark',
        )
