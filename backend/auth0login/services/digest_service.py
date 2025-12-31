"""
Digest Email Service for WallStreetBots.

Aggregates trading data and sends daily/weekly digest emails to users
based on their email preferences.
"""

import logging
import os
import smtplib
from datetime import datetime, timedelta
from decimal import Decimal
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, Tuple

from django.contrib.auth.models import User
from django.db.models import Avg, Count, F, Q, Sum
from django.template.loader import render_to_string
from django.utils import timezone

logger = logging.getLogger(__name__)


class DigestService:
    """
    Service for generating and sending digest emails.

    Aggregates trading activity, performance metrics, alerts, and positions
    for daily and weekly digests.
    """

    def __init__(self, user: Optional[User] = None):
        """
        Initialize the digest service.

        Args:
            user: Optional user to generate digests for. If None, service
                  can be used for batch operations.
        """
        self.user = user

        # SMTP configuration from environment
        self.smtp_host = os.getenv("ALERT_EMAIL_SMTP_HOST")
        self.smtp_port = int(os.getenv("ALERT_EMAIL_SMTP_PORT", "587"))
        self.smtp_user = os.getenv("ALERT_EMAIL_USER")
        self.smtp_pass = os.getenv("ALERT_EMAIL_PASS")
        self.from_email = os.getenv("ALERT_EMAIL_FROM")

        # Base URL for links in emails
        self.base_url = os.getenv("APP_BASE_URL", "http://localhost:8000")

    def _get_period_bounds(self, digest_type: str) -> Tuple[datetime, datetime]:
        """
        Get the start and end times for a digest period.

        Args:
            digest_type: 'daily' or 'weekly'

        Returns:
            Tuple of (period_start, period_end) datetime objects
        """
        now = timezone.now()

        if digest_type == 'daily':
            # Previous day: midnight to midnight
            period_end = now.replace(hour=0, minute=0, second=0, microsecond=0)
            period_start = period_end - timedelta(days=1)
        else:  # weekly
            # Previous week: Monday to Sunday
            days_since_monday = now.weekday()
            period_end = now.replace(hour=0, minute=0, second=0, microsecond=0)
            period_end = period_end - timedelta(days=days_since_monday)
            period_start = period_end - timedelta(days=7)

        return period_start, period_end

    def aggregate_trades(
        self,
        user: User,
        period_start: datetime,
        period_end: datetime
    ) -> Dict[str, Any]:
        """
        Aggregate trade data for the given period.

        Args:
            user: User to aggregate trades for
            period_start: Start of period
            period_end: End of period

        Returns:
            Dictionary with trade summary and details
        """
        from backend.tradingbot.models.models import TradeSignalSnapshot

        # Get trades in period
        trades = TradeSignalSnapshot.objects.filter(
            created_at__gte=period_start,
            created_at__lt=period_end
        ).order_by('-created_at')

        # Separate buy and sell trades
        buy_trades = trades.filter(direction__in=['buy', 'buy_to_cover'])
        sell_trades = trades.filter(direction__in=['sell', 'sell_short'])

        # Calculate P&L from closed trades (those with exit data)
        closed_trades = trades.filter(exit_price__isnull=False)

        winning_trades = closed_trades.filter(pnl_amount__gt=0)
        losing_trades = closed_trades.filter(pnl_amount__lt=0)

        total_pnl = closed_trades.aggregate(total=Sum('pnl_amount'))['total'] or Decimal('0')

        # Build trade list for detail
        trade_list = []
        for trade in trades[:20]:  # Limit to 20 most recent
            trade_list.append({
                'symbol': trade.symbol,
                'direction': trade.direction,
                'quantity': float(trade.quantity),
                'entry_price': float(trade.entry_price),
                'exit_price': float(trade.exit_price) if trade.exit_price else None,
                'pnl': float(trade.pnl_amount) if trade.pnl_amount else None,
                'strategy': trade.strategy_name,
                'timestamp': trade.created_at.isoformat(),
            })

        win_rate = 0.0
        if closed_trades.count() > 0:
            win_rate = (winning_trades.count() / closed_trades.count()) * 100

        return {
            'summary': {
                'total_trades': trades.count(),
                'buy_trades': buy_trades.count(),
                'sell_trades': sell_trades.count(),
                'closed_trades': closed_trades.count(),
                'winning_trades': winning_trades.count(),
                'losing_trades': losing_trades.count(),
                'total_pnl': float(total_pnl),
                'win_rate': round(win_rate, 2),
            },
            'trades': trade_list,
        }

    def aggregate_alerts(
        self,
        user: User,
        period_start: datetime,
        period_end: datetime
    ) -> Dict[str, Any]:
        """
        Aggregate alerts for the given period.

        Args:
            user: User to aggregate alerts for
            period_start: Start of period
            period_end: End of period

        Returns:
            Dictionary with alert summary and details
        """
        from backend.tradingbot.models.models import CircuitBreakerHistory

        # Get circuit breaker events as a proxy for major alerts
        cb_events = CircuitBreakerHistory.objects.filter(
            timestamp__gte=period_start,
            timestamp__lt=period_end
        ).order_by('-timestamp')

        alert_list = []

        # Circuit breaker events as alerts
        for event in cb_events[:10]:
            alert_list.append({
                'type': 'circuit_breaker',
                'action': event.action,
                'message': event.reason or f"Circuit breaker {event.action}",
                'timestamp': event.timestamp.isoformat(),
                'severity': 'high' if event.action == 'trip' else 'info',
            })

        # Count by type
        type_counts = {}
        for alert in alert_list:
            t = alert['type']
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            'summary': {
                'total_alerts': len(alert_list),
                'high_severity': sum(1 for a in alert_list if a['severity'] == 'high'),
                'by_type': type_counts,
            },
            'alerts': alert_list,
        }

    def aggregate_positions(
        self,
        user: User,
        period_end: datetime
    ) -> Dict[str, Any]:
        """
        Aggregate current position data.

        Args:
            user: User to aggregate positions for
            period_end: End of period (for snapshot time)

        Returns:
            Dictionary with position summary
        """
        from backend.tradingbot.models.models import TradeSignalSnapshot

        # Get open trades (those without exit)
        open_trades = TradeSignalSnapshot.objects.filter(
            exit_price__isnull=True
        ).values('symbol').annotate(
            total_quantity=Sum('quantity'),
            avg_price=Avg('entry_price'),
            trade_count=Count('id')
        )

        positions = []
        total_value = Decimal('0')

        for pos in open_trades:
            value = pos['avg_price'] * pos['total_quantity']
            total_value += value
            positions.append({
                'symbol': pos['symbol'],
                'quantity': float(pos['total_quantity']),
                'avg_price': float(pos['avg_price']),
                'value': float(value),
                'trade_count': pos['trade_count'],
            })

        return {
            'summary': {
                'open_positions': len(positions),
                'total_value': float(total_value),
            },
            'positions': positions[:10],  # Limit to top 10
        }

    def aggregate_performance(
        self,
        user: User,
        period_start: datetime,
        period_end: datetime
    ) -> Dict[str, Any]:
        """
        Aggregate performance metrics for the period.

        Args:
            user: User to aggregate performance for
            period_start: Start of period
            period_end: End of period

        Returns:
            Dictionary with performance metrics
        """
        # Import benchmark service if available
        try:
            from backend.auth0login.services.benchmark import BenchmarkService
            benchmark_svc = BenchmarkService()

            # Get benchmark comparison data
            benchmark_data = benchmark_svc.get_performance_vs_benchmark(
                user,
                period_start,
                period_end
            )

            return {
                'period_return': benchmark_data.get('portfolio_return', 0),
                'benchmark_return': benchmark_data.get('benchmark_return', 0),
                'alpha': benchmark_data.get('alpha', 0),
                'sharpe_ratio': benchmark_data.get('sharpe_ratio'),
                'max_drawdown': benchmark_data.get('max_drawdown'),
            }
        except Exception as e:
            logger.warning(f"Could not get benchmark data: {e}")
            return {
                'period_return': 0,
                'benchmark_return': 0,
                'alpha': 0,
                'sharpe_ratio': None,
                'max_drawdown': None,
            }

    def aggregate_by_strategy(
        self,
        user: User,
        period_start: datetime,
        period_end: datetime
    ) -> List[Dict[str, Any]]:
        """
        Aggregate performance broken down by strategy.

        Args:
            user: User to aggregate for
            period_start: Start of period
            period_end: End of period

        Returns:
            List of strategy performance summaries
        """
        from backend.tradingbot.models.models import TradeSignalSnapshot

        strategy_stats = TradeSignalSnapshot.objects.filter(
            created_at__gte=period_start,
            created_at__lt=period_end
        ).values('strategy_name').annotate(
            total_trades=Count('id'),
            total_pnl=Sum('pnl_amount'),
            winning_trades=Count('id', filter=Q(pnl_amount__gt=0)),
        ).order_by('-total_pnl')

        results = []
        for stat in strategy_stats:
            win_rate = 0
            if stat['total_trades'] > 0 and stat['winning_trades']:
                win_rate = (stat['winning_trades'] / stat['total_trades']) * 100

            results.append({
                'strategy': stat['strategy_name'],
                'total_trades': stat['total_trades'],
                'total_pnl': float(stat['total_pnl'] or 0),
                'winning_trades': stat['winning_trades'] or 0,
                'win_rate': round(win_rate, 2),
            })

        return results

    def generate_digest_data(
        self,
        user: User,
        digest_type: str = 'daily'
    ) -> Dict[str, Any]:
        """
        Generate complete digest data for a user.

        Args:
            user: User to generate digest for
            digest_type: 'daily' or 'weekly'

        Returns:
            Complete digest data dictionary
        """
        period_start, period_end = self._get_period_bounds(digest_type)

        trades = self.aggregate_trades(user, period_start, period_end)
        alerts = self.aggregate_alerts(user, period_start, period_end)
        positions = self.aggregate_positions(user, period_end)
        performance = self.aggregate_performance(user, period_start, period_end)
        by_strategy = self.aggregate_by_strategy(user, period_start, period_end)

        return {
            'digest_type': digest_type,
            'period_start': period_start.isoformat(),
            'period_end': period_end.isoformat(),
            'generated_at': timezone.now().isoformat(),
            'user': {
                'username': user.username,
                'email': user.email,
            },
            'summary': {
                'total_trades': trades['summary']['total_trades'],
                'winning_trades': trades['summary']['winning_trades'],
                'losing_trades': trades['summary']['losing_trades'],
                'total_pnl': trades['summary']['total_pnl'],
                'win_rate': trades['summary']['win_rate'],
                'total_alerts': alerts['summary']['total_alerts'],
                'open_positions': positions['summary']['open_positions'],
            },
            'trades': trades['trades'],
            'alerts': alerts['alerts'],
            'positions': positions['positions'],
            'performance': performance,
            'by_strategy': by_strategy,
        }

    def _get_email_subject(
        self,
        digest_type: str,
        data: Dict[str, Any]
    ) -> str:
        """Generate email subject line."""
        pnl = data['summary']['total_pnl']
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        trades = data['summary']['total_trades']

        if digest_type == 'daily':
            return f"[WSB] Daily Digest: {trades} trades, {pnl_str} P&L"
        else:
            return f"[WSB] Weekly Digest: {trades} trades, {pnl_str} P&L"

    def _render_email_html(
        self,
        digest_type: str,
        data: Dict[str, Any]
    ) -> str:
        """
        Render the HTML email body.

        Args:
            digest_type: 'daily' or 'weekly'
            data: Digest data dictionary

        Returns:
            HTML string
        """
        # Try to use Django template if available
        try:
            template_name = f"emails/{digest_type}_digest.html"
            return render_to_string(template_name, {'data': data})
        except Exception:
            # Fall back to inline HTML
            return self._generate_inline_html(digest_type, data)

    def _render_email_text(
        self,
        digest_type: str,
        data: Dict[str, Any]
    ) -> str:
        """
        Render the plain text email body.

        Args:
            digest_type: 'daily' or 'weekly'
            data: Digest data dictionary

        Returns:
            Plain text string
        """
        lines = []

        # Header
        period_type = "Daily" if digest_type == 'daily' else "Weekly"
        lines.append(f"WallStreetBots {period_type} Digest")
        lines.append("=" * 40)
        lines.append("")

        # Summary
        summary = data['summary']
        lines.append("SUMMARY")
        lines.append("-" * 20)
        lines.append(f"Total Trades: {summary['total_trades']}")
        lines.append(f"Winning: {summary['winning_trades']} | Losing: {summary['losing_trades']}")
        lines.append(f"Win Rate: {summary['win_rate']:.1f}%")
        pnl = summary['total_pnl']
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        lines.append(f"Total P&L: {pnl_str}")
        lines.append(f"Open Positions: {summary['open_positions']}")
        lines.append("")

        # Performance
        perf = data['performance']
        if perf.get('period_return') is not None:
            lines.append("PERFORMANCE vs BENCHMARK")
            lines.append("-" * 20)
            lines.append(f"Your Return: {perf['period_return']:.2f}%")
            lines.append(f"SPY Return: {perf['benchmark_return']:.2f}%")
            lines.append(f"Alpha: {perf['alpha']:.2f}%")
            lines.append("")

        # Recent Trades
        if data['trades']:
            lines.append("RECENT TRADES")
            lines.append("-" * 20)
            for trade in data['trades'][:5]:
                pnl_part = ""
                if trade.get('pnl'):
                    pnl_part = f" (P&L: ${trade['pnl']:.2f})"
                lines.append(
                    f"  {trade['direction'].upper()} {trade['quantity']} "
                    f"{trade['symbol']} @ ${trade['entry_price']:.2f}{pnl_part}"
                )
            lines.append("")

        # Alerts
        if data['alerts']:
            lines.append("ALERTS")
            lines.append("-" * 20)
            for alert in data['alerts'][:5]:
                lines.append(f"  [{alert['severity'].upper()}] {alert['message']}")
            lines.append("")

        # Strategy Breakdown
        if data['by_strategy']:
            lines.append("BY STRATEGY")
            lines.append("-" * 20)
            for strat in data['by_strategy'][:5]:
                pnl = strat['total_pnl']
                pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
                lines.append(
                    f"  {strat['strategy']}: {strat['total_trades']} trades, "
                    f"{strat['win_rate']:.1f}% win rate, {pnl_str}"
                )
            lines.append("")

        # Footer
        lines.append("-" * 40)
        lines.append(f"View full details: {self.base_url}/dashboard")
        lines.append(f"Manage preferences: {self.base_url}/settings")
        lines.append("")

        return "\n".join(lines)

    def _generate_inline_html(
        self,
        digest_type: str,
        data: Dict[str, Any]
    ) -> str:
        """Generate inline HTML when templates aren't available."""
        summary = data['summary']
        pnl = summary['total_pnl']
        pnl_color = "#22c55e" if pnl >= 0 else "#ef4444"
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"

        period_type = "Daily" if digest_type == 'daily' else "Weekly"

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WallStreetBots {period_type} Digest</title>
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f5f5f5; margin: 0; padding: 20px;">
    <div style="max-width: 600px; margin: 0 auto; background-color: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <!-- Header -->
        <div style="background-color: #1e3a5f; color: white; padding: 24px; text-align: center;">
            <h1 style="margin: 0; font-size: 24px;">WallStreetBots</h1>
            <p style="margin: 8px 0 0 0; opacity: 0.9;">{period_type} Trading Digest</p>
        </div>

        <!-- Summary Cards -->
        <div style="padding: 24px;">
            <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 16px; margin-bottom: 24px;">
                <div style="flex: 1; min-width: 120px; background-color: #f8fafc; padding: 16px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 28px; font-weight: bold; color: #1e3a5f;">{summary['total_trades']}</div>
                    <div style="color: #64748b; font-size: 14px;">Total Trades</div>
                </div>
                <div style="flex: 1; min-width: 120px; background-color: #f8fafc; padding: 16px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 28px; font-weight: bold; color: {pnl_color};">{pnl_str}</div>
                    <div style="color: #64748b; font-size: 14px;">Total P&L</div>
                </div>
                <div style="flex: 1; min-width: 120px; background-color: #f8fafc; padding: 16px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 28px; font-weight: bold; color: #1e3a5f;">{summary['win_rate']:.1f}%</div>
                    <div style="color: #64748b; font-size: 14px;">Win Rate</div>
                </div>
            </div>
"""

        # Performance section
        perf = data['performance']
        if perf.get('period_return') is not None:
            alpha_color = "#22c55e" if perf['alpha'] >= 0 else "#ef4444"
            html += f"""
            <div style="margin-bottom: 24px;">
                <h2 style="color: #1e3a5f; font-size: 18px; margin-bottom: 12px;">Performance vs Benchmark</h2>
                <div style="background-color: #f8fafc; padding: 16px; border-radius: 8px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span>Your Return:</span>
                        <span style="font-weight: bold;">{perf['period_return']:.2f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span>SPY Return:</span>
                        <span style="font-weight: bold;">{perf['benchmark_return']:.2f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>Alpha:</span>
                        <span style="font-weight: bold; color: {alpha_color};">{perf['alpha']:.2f}%</span>
                    </div>
                </div>
            </div>
"""

        # Recent trades section
        if data['trades']:
            html += """
            <div style="margin-bottom: 24px;">
                <h2 style="color: #1e3a5f; font-size: 18px; margin-bottom: 12px;">Recent Trades</h2>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="background-color: #f1f5f9;">
                        <th style="padding: 8px; text-align: left; font-size: 12px; color: #64748b;">Symbol</th>
                        <th style="padding: 8px; text-align: left; font-size: 12px; color: #64748b;">Action</th>
                        <th style="padding: 8px; text-align: right; font-size: 12px; color: #64748b;">Qty</th>
                        <th style="padding: 8px; text-align: right; font-size: 12px; color: #64748b;">Price</th>
                        <th style="padding: 8px; text-align: right; font-size: 12px; color: #64748b;">P&L</th>
                    </tr>
"""
            for trade in data['trades'][:5]:
                direction_color = "#22c55e" if trade['direction'] in ['buy', 'buy_to_cover'] else "#ef4444"
                pnl_cell = ""
                if trade.get('pnl') is not None:
                    t_pnl = trade['pnl']
                    t_pnl_color = "#22c55e" if t_pnl >= 0 else "#ef4444"
                    t_pnl_str = f"+${t_pnl:.2f}" if t_pnl >= 0 else f"-${abs(t_pnl):.2f}"
                    pnl_cell = f'<span style="color: {t_pnl_color};">{t_pnl_str}</span>'
                else:
                    pnl_cell = "-"

                html += f"""
                    <tr style="border-bottom: 1px solid #e2e8f0;">
                        <td style="padding: 8px; font-weight: bold;">{trade['symbol']}</td>
                        <td style="padding: 8px; color: {direction_color};">{trade['direction'].upper()}</td>
                        <td style="padding: 8px; text-align: right;">{trade['quantity']:.0f}</td>
                        <td style="padding: 8px; text-align: right;">${trade['entry_price']:.2f}</td>
                        <td style="padding: 8px; text-align: right;">{pnl_cell}</td>
                    </tr>
"""
            html += """
                </table>
            </div>
"""

        # Strategy breakdown
        if data['by_strategy']:
            html += """
            <div style="margin-bottom: 24px;">
                <h2 style="color: #1e3a5f; font-size: 18px; margin-bottom: 12px;">Performance by Strategy</h2>
"""
            for strat in data['by_strategy'][:5]:
                s_pnl = strat['total_pnl']
                s_pnl_color = "#22c55e" if s_pnl >= 0 else "#ef4444"
                s_pnl_str = f"+${s_pnl:.2f}" if s_pnl >= 0 else f"-${abs(s_pnl):.2f}"

                html += f"""
                <div style="background-color: #f8fafc; padding: 12px; border-radius: 6px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: bold;">{strat['strategy']}</span>
                        <span style="color: {s_pnl_color}; font-weight: bold;">{s_pnl_str}</span>
                    </div>
                    <div style="color: #64748b; font-size: 12px; margin-top: 4px;">
                        {strat['total_trades']} trades | {strat['win_rate']:.1f}% win rate
                    </div>
                </div>
"""
            html += """
            </div>
"""

        # Alerts section
        if data['alerts']:
            html += """
            <div style="margin-bottom: 24px;">
                <h2 style="color: #1e3a5f; font-size: 18px; margin-bottom: 12px;">Alerts</h2>
"""
            for alert in data['alerts'][:5]:
                alert_bg = "#fef2f2" if alert['severity'] == 'high' else "#f0f9ff"
                alert_border = "#ef4444" if alert['severity'] == 'high' else "#3b82f6"

                html += f"""
                <div style="background-color: {alert_bg}; border-left: 4px solid {alert_border}; padding: 12px; margin-bottom: 8px;">
                    <div style="font-weight: bold;">{alert['type'].replace('_', ' ').title()}</div>
                    <div style="color: #64748b; font-size: 14px;">{alert['message']}</div>
                </div>
"""
            html += """
            </div>
"""

        # Footer
        html += f"""
            <!-- CTA Button -->
            <div style="text-align: center; margin-bottom: 24px;">
                <a href="{self.base_url}/dashboard" style="display: inline-block; background-color: #1e3a5f; color: white; padding: 12px 32px; border-radius: 6px; text-decoration: none; font-weight: bold;">View Full Dashboard</a>
            </div>
        </div>

        <!-- Footer -->
        <div style="padding: 16px 24px; background-color: #f8fafc; text-align: center; color: #64748b; font-size: 12px;">
            <p style="margin: 0 0 8px 0;">
                <a href="{self.base_url}/settings" style="color: #3b82f6; text-decoration: none;">Manage email preferences</a> |
                <a href="{self.base_url}/api/digest/unsubscribe" style="color: #3b82f6; text-decoration: none;">Unsubscribe</a>
            </p>
            <p style="margin: 0;">WallStreetBots - Automated Trading Platform</p>
        </div>
    </div>
</body>
</html>
"""
        return html

    def send_digest_email(
        self,
        user: User,
        digest_type: str = 'daily',
        data: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Send a digest email to a user.

        Args:
            user: User to send digest to
            digest_type: 'daily' or 'weekly'
            data: Pre-generated digest data (optional, will generate if not provided)

        Returns:
            Tuple of (success, error_message, digest_data)
        """
        from backend.tradingbot.models.models import DigestLog

        # Generate data if not provided
        if data is None:
            data = self.generate_digest_data(user, digest_type)

        period_start, period_end = self._get_period_bounds(digest_type)

        # Check for existing digest for this period
        existing = DigestLog.objects.filter(
            user=user,
            digest_type=digest_type,
            period_start=period_start
        ).first()

        if existing and existing.delivery_status == 'sent':
            return False, "Digest already sent for this period", data

        # Create or update digest log
        digest_log, created = DigestLog.objects.update_or_create(
            user=user,
            digest_type=digest_type,
            period_start=period_start,
            defaults={
                'period_end': period_end,
                'data_snapshot': data,
                'email_subject': self._get_email_subject(digest_type, data),
                'email_recipient': user.email,
                'delivery_status': 'pending',
            }
        )

        # Check SMTP configuration
        if not all([self.smtp_host, self.smtp_user, self.smtp_pass, self.from_email]):
            error_msg = "Email not configured (missing SMTP settings)"
            digest_log.mark_failed(error_msg)
            return False, error_msg, data

        # Check user has email
        if not user.email:
            error_msg = "User has no email address"
            digest_log.mark_failed(error_msg)
            return False, error_msg, data

        # Generate email content
        subject = self._get_email_subject(digest_type, data)
        html_body = self._render_email_html(digest_type, data)
        text_body = self._render_email_text(digest_type, data)

        try:
            # Create multipart message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.from_email
            msg['To'] = user.email

            # Attach plain text and HTML versions
            part1 = MIMEText(text_body, 'plain')
            part2 = MIMEText(html_body, 'html')
            msg.attach(part1)
            msg.attach(part2)

            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.sendmail(self.from_email, [user.email], msg.as_string())

            # Mark as sent
            digest_log.mark_sent()
            logger.info(f"Sent {digest_type} digest to {user.email}")

            return True, "", data

        except Exception as e:
            error_msg = str(e)
            digest_log.mark_failed(error_msg)
            logger.error(f"Failed to send digest to {user.email}: {error_msg}")
            return False, error_msg, data

    def send_digests_for_frequency(
        self,
        digest_type: str,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Send digests to all users who have opted into this frequency.

        Args:
            digest_type: 'daily' or 'weekly'
            dry_run: If True, generate data but don't actually send emails

        Returns:
            Summary of results
        """
        from backend.tradingbot.models.models import UserProfile

        # Map digest type to email_frequency preference
        frequency_map = {
            'daily': 'daily',
            'weekly': 'weekly',
        }

        target_frequency = frequency_map.get(digest_type)
        if not target_frequency:
            return {
                'success': False,
                'error': f"Invalid digest type: {digest_type}",
                'sent': 0,
                'failed': 0,
            }

        # Get users who want this digest frequency
        users = User.objects.filter(
            profile__email_frequency=target_frequency,
            email__isnull=False,
            is_active=True
        ).exclude(email='').select_related('profile')

        results = {
            'digest_type': digest_type,
            'total_users': users.count(),
            'sent': 0,
            'failed': 0,
            'skipped': 0,
            'errors': [],
            'dry_run': dry_run,
        }

        for user in users:
            try:
                if dry_run:
                    # Just generate data
                    data = self.generate_digest_data(user, digest_type)
                    results['sent'] += 1
                    logger.info(f"[DRY RUN] Would send {digest_type} digest to {user.email}")
                else:
                    success, error, _ = self.send_digest_email(user, digest_type)
                    if success:
                        results['sent'] += 1
                    elif "already sent" in error:
                        results['skipped'] += 1
                    else:
                        results['failed'] += 1
                        results['errors'].append({
                            'user': user.username,
                            'error': error,
                        })
            except Exception as e:
                results['failed'] += 1
                results['errors'].append({
                    'user': user.username,
                    'error': str(e),
                })

        return results

    def preview_digest(
        self,
        user: User,
        digest_type: str = 'daily'
    ) -> Dict[str, Any]:
        """
        Generate a preview of what a digest would contain.

        Args:
            user: User to preview digest for
            digest_type: 'daily' or 'weekly'

        Returns:
            Preview data including HTML rendering
        """
        data = self.generate_digest_data(user, digest_type)

        return {
            'data': data,
            'subject': self._get_email_subject(digest_type, data),
            'html': self._render_email_html(digest_type, data),
            'text': self._render_email_text(digest_type, data),
        }
