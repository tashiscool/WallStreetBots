"""
Django management command for sending digest emails.

Usage:
    # Send daily digests to all opted-in users
    python manage.py send_digests --type daily

    # Send weekly digests
    python manage.py send_digests --type weekly

    # Dry run (don't actually send)
    python manage.py send_digests --type daily --dry-run

    # Send to specific user
    python manage.py send_digests --type daily --user admin@example.com

Scheduling with cron:
    # Daily digests at 7:00 AM
    0 7 * * * cd /path/to/project && python manage.py send_digests --type daily

    # Weekly digests on Monday at 8:00 AM
    0 8 * * 1 cd /path/to/project && python manage.py send_digests --type weekly
"""

import logging
from django.core.management.base import BaseCommand, CommandError
from django.contrib.auth.models import User
from django.db.models import Q

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Send daily or weekly digest emails to users'

    def add_arguments(self, parser):
        parser.add_argument(
            '--type',
            type=str,
            choices=['daily', 'weekly'],
            required=True,
            help='Type of digest to send (daily or weekly)'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Generate digests but do not send emails'
        )
        parser.add_argument(
            '--user',
            type=str,
            help='Send digest to a specific user (email or username)'
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force send even if already sent for this period'
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Print detailed output'
        )

    def handle(self, *args, **options):
        from backend.auth0login.services.digest_service import DigestService

        digest_type = options['type']
        dry_run = options['dry_run']
        user_identifier = options.get('user')
        force = options.get('force', False)
        verbose = options.get('verbose', False)

        self.stdout.write(
            self.style.NOTICE(
                f"{'[DRY RUN] ' if dry_run else ''}Starting {digest_type} digest send..."
            )
        )

        service = DigestService()

        if user_identifier:
            # Send to specific user
            try:
                user = User.objects.get(
                    Q(email=user_identifier) | Q(username=user_identifier)
                )
            except User.DoesNotExist:
                raise CommandError(f"User not found: {user_identifier}")
            except User.MultipleObjectsReturned:
                raise CommandError(f"Multiple users match: {user_identifier}")

            self.stdout.write(f"Sending {digest_type} digest to {user.email}...")

            if dry_run:
                data = service.generate_digest_data(user, digest_type)
                self._print_digest_summary(data, verbose)
                self.stdout.write(
                    self.style.SUCCESS(f"[DRY RUN] Would send digest to {user.email}")
                )
            else:
                success, error, data = service.send_digest_email(user, digest_type)
                self._print_digest_summary(data, verbose)

                if success:
                    self.stdout.write(
                        self.style.SUCCESS(f"Successfully sent digest to {user.email}")
                    )
                else:
                    self.stdout.write(
                        self.style.ERROR(f"Failed to send digest: {error}")
                    )
        else:
            # Send to all opted-in users
            results = service.send_digests_for_frequency(digest_type, dry_run=dry_run)

            self.stdout.write("")
            self.stdout.write(self.style.NOTICE("=" * 50))
            self.stdout.write(self.style.NOTICE(f"{digest_type.upper()} DIGEST SUMMARY"))
            self.stdout.write(self.style.NOTICE("=" * 50))
            self.stdout.write(f"Total users: {results['total_users']}")
            self.stdout.write(
                self.style.SUCCESS(f"Sent: {results['sent']}")
            )
            if results['skipped'] > 0:
                self.stdout.write(
                    self.style.WARNING(f"Skipped (already sent): {results['skipped']}")
                )
            if results['failed'] > 0:
                self.stdout.write(
                    self.style.ERROR(f"Failed: {results['failed']}")
                )

            if results['errors'] and verbose:
                self.stdout.write("")
                self.stdout.write(self.style.ERROR("Errors:"))
                for err in results['errors']:
                    self.stdout.write(f"  - {err['user']}: {err['error']}")

            self.stdout.write("")
            if dry_run:
                self.stdout.write(
                    self.style.WARNING(
                        "[DRY RUN] No emails were actually sent. "
                        "Remove --dry-run to send for real."
                    )
                )

    def _print_digest_summary(self, data: dict, verbose: bool = False):
        """Print a summary of the digest data."""
        summary = data.get('summary', {})

        self.stdout.write("")
        self.stdout.write(self.style.NOTICE("-" * 40))
        self.stdout.write(f"Period: {data.get('period_start', 'N/A')[:10]} to {data.get('period_end', 'N/A')[:10]}")
        self.stdout.write(f"Trades: {summary.get('total_trades', 0)}")
        self.stdout.write(f"Win Rate: {summary.get('win_rate', 0):.1f}%")

        pnl = summary.get('total_pnl', 0)
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        if pnl >= 0:
            self.stdout.write(self.style.SUCCESS(f"P&L: {pnl_str}"))
        else:
            self.stdout.write(self.style.ERROR(f"P&L: {pnl_str}"))

        self.stdout.write(f"Alerts: {summary.get('total_alerts', 0)}")
        self.stdout.write(f"Open Positions: {summary.get('open_positions', 0)}")
        self.stdout.write(self.style.NOTICE("-" * 40))

        if verbose:
            # Print trade details
            trades = data.get('trades', [])
            if trades:
                self.stdout.write("")
                self.stdout.write("Recent Trades:")
                for trade in trades[:5]:
                    pnl_part = ""
                    if trade.get('pnl'):
                        pnl_part = f" (P&L: ${trade['pnl']:.2f})"
                    self.stdout.write(
                        f"  {trade['direction'].upper()} {trade['quantity']:.0f} "
                        f"{trade['symbol']} @ ${trade['entry_price']:.2f}{pnl_part}"
                    )

            # Print strategy breakdown
            strategies = data.get('by_strategy', [])
            if strategies:
                self.stdout.write("")
                self.stdout.write("By Strategy:")
                for strat in strategies[:5]:
                    s_pnl = strat.get('total_pnl', 0)
                    s_pnl_str = f"+${s_pnl:.2f}" if s_pnl >= 0 else f"-${abs(s_pnl):.2f}"
                    self.stdout.write(
                        f"  {strat['strategy']}: {strat['total_trades']} trades, "
                        f"{strat.get('win_rate', 0):.1f}% win, {s_pnl_str}"
                    )
