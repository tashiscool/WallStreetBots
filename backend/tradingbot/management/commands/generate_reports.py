"""
Management command to generate PDF performance reports.

Usage:
    # Generate weekly report for all users with email preference enabled
    python manage.py generate_reports --type weekly --email

    # Generate monthly report for a specific user
    python manage.py generate_reports --type monthly --user-id 1

    # Generate quarterly report and save to disk (no email)
    python manage.py generate_reports --type quarterly --user-id 1 --output /tmp/report.pdf

    # Dry run to see which users would receive reports
    python manage.py generate_reports --type weekly --email --dry-run

Recommended scheduling:
    # Weekly on Monday morning
    0 7 * * 1 cd /path/to/project && python manage.py generate_reports --type weekly --email

    # Monthly on the 1st
    0 7 1 * * cd /path/to/project && python manage.py generate_reports --type monthly --email

    # Quarterly
    0 7 1 1,4,7,10 * cd /path/to/project && python manage.py generate_reports --type quarterly --email
"""

import logging
from datetime import datetime

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand, CommandError

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Generate PDF performance reports for users'

    def add_arguments(self, parser):
        parser.add_argument(
            '--type',
            type=str,
            default='weekly',
            choices=['weekly', 'monthly', 'quarterly', 'yearly'],
            help='Report type to generate (default: weekly)',
        )
        parser.add_argument(
            '--user-id',
            type=int,
            default=None,
            help='Generate report for a specific user ID. If omitted, generates for all eligible users.',
        )
        parser.add_argument(
            '--email',
            action='store_true',
            help='Send generated reports via email',
        )
        parser.add_argument(
            '--output',
            type=str,
            default=None,
            help='Save PDF to this file path (only works with --user-id)',
        )
        parser.add_argument(
            '--strategy',
            type=str,
            default=None,
            help='Filter report to a specific strategy name',
        )
        parser.add_argument(
            '--start-date',
            type=str,
            default=None,
            help='Report start date (YYYY-MM-DD). Auto-calculated if omitted.',
        )
        parser.add_argument(
            '--end-date',
            type=str,
            default=None,
            help='Report end date (YYYY-MM-DD). Defaults to today.',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Preview what would happen without generating or sending reports',
        )

    def handle(self, *args, **options):
        from backend.auth0login.services.report_delivery_service import ReportDeliveryService

        report_type = options['type']
        user_id = options['user_id']
        send_email = options['email']
        output_path = options['output']
        strategy_name = options.get('strategy')
        dry_run = options['dry_run']

        # Parse dates
        start_date = None
        end_date = None
        if options['start_date']:
            try:
                start_date = datetime.strptime(options['start_date'], '%Y-%m-%d').date()
            except ValueError:
                raise CommandError('Invalid start date format. Use YYYY-MM-DD.')
        if options['end_date']:
            try:
                end_date = datetime.strptime(options['end_date'], '%Y-%m-%d').date()
            except ValueError:
                raise CommandError('Invalid end date format. Use YYYY-MM-DD.')

        service = ReportDeliveryService()

        self.stdout.write(f"\nPDF Report Generation")
        self.stdout.write(f"{'=' * 50}")
        self.stdout.write(f"Report type: {report_type}")
        self.stdout.write(f"Strategy filter: {strategy_name or 'All strategies'}")

        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN - No reports will be generated or sent'))

        if user_id:
            # Generate for a specific user
            self._generate_for_user(
                service, user_id, report_type, start_date, end_date,
                strategy_name, send_email, output_path, dry_run,
            )
        elif send_email:
            # Batch: generate and email for all eligible users
            self._generate_for_all_users(
                service, report_type, start_date, end_date,
                strategy_name, dry_run,
            )
        else:
            raise CommandError(
                'Please specify either --user-id (for a single user) or '
                '--email (to send to all eligible users).'
            )

        self.stdout.write(f"\n{'=' * 50}")
        self.stdout.write(self.style.SUCCESS('Done.'))

    def _generate_for_user(
        self, service, user_id, report_type, start_date, end_date,
        strategy_name, send_email, output_path, dry_run,
    ):
        """Generate a report for a single user."""
        try:
            user = User.objects.get(pk=user_id)
        except User.DoesNotExist:
            raise CommandError(f'User with ID {user_id} not found.')

        self.stdout.write(f"User: {user.username} ({user.email or 'no email'})")

        if dry_run:
            self.stdout.write(self.style.WARNING(
                f"  Would generate {report_type} report for {user.username}"
            ))
            if send_email and user.email:
                self.stdout.write(self.style.WARNING(
                    f"  Would email report to {user.email}"
                ))
            if output_path:
                self.stdout.write(self.style.WARNING(
                    f"  Would save report to {output_path}"
                ))
            return

        self.stdout.write(f"  Generating {report_type} report...")

        try:
            pdf_bytes = service.generate_report(
                user=user,
                report_type=report_type,
                start_date=start_date,
                end_date=end_date,
                strategy_name=strategy_name,
            )
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"  Error generating report: {e}"))
            logger.exception(f"Report generation failed for user {user.username}")
            return

        self.stdout.write(self.style.SUCCESS(
            f"  Report generated ({len(pdf_bytes):,} bytes)"
        ))

        # Save to file if requested
        if output_path:
            try:
                with open(output_path, 'wb') as f:
                    f.write(pdf_bytes)
                self.stdout.write(self.style.SUCCESS(f"  Saved to {output_path}"))
            except IOError as e:
                self.stdout.write(self.style.ERROR(f"  Failed to save file: {e}"))

        # Email if requested
        if send_email:
            if not user.email:
                self.stdout.write(self.style.WARNING(
                    f"  Cannot email: user {user.username} has no email address"
                ))
                return

            try:
                sent = service.email_report(
                    user=user,
                    pdf_bytes=pdf_bytes,
                    report_type=report_type,
                )
                if sent:
                    self.stdout.write(self.style.SUCCESS(f"  Emailed to {user.email}"))
                else:
                    self.stdout.write(self.style.ERROR(f"  Failed to email to {user.email}"))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"  Email error: {e}"))

    def _generate_for_all_users(
        self, service, report_type, start_date, end_date,
        strategy_name, dry_run,
    ):
        """Generate and email reports for all eligible users."""
        from backend.tradingbot.models.models import UserProfile

        profiles = UserProfile.objects.filter(
            email_performance_reports=True,
        ).select_related('user')

        eligible_count = profiles.count()
        self.stdout.write(f"Eligible users: {eligible_count}")

        if eligible_count == 0:
            self.stdout.write(self.style.WARNING(
                'No users with email_performance_reports enabled.'
            ))
            return

        if dry_run:
            for profile in profiles:
                user = profile.user
                self.stdout.write(f"  Would send to: {user.username} ({user.email or 'no email'})")
            return

        generated = 0
        emailed = 0
        errors = 0

        for profile in profiles:
            user = profile.user
            if not user.email:
                self.stdout.write(self.style.WARNING(
                    f"  Skipping {user.username}: no email address"
                ))
                errors += 1
                continue

            try:
                pdf_bytes = service.generate_report(
                    user=user,
                    report_type=report_type,
                    start_date=start_date,
                    end_date=end_date,
                    strategy_name=strategy_name,
                )
                generated += 1

                sent = service.email_report(
                    user=user,
                    pdf_bytes=pdf_bytes,
                    report_type=report_type,
                )
                if sent:
                    emailed += 1
                    self.stdout.write(f"  Sent to {user.email}")
                else:
                    errors += 1
                    self.stdout.write(self.style.ERROR(
                        f"  Failed to send to {user.email}"
                    ))
            except Exception as e:
                errors += 1
                self.stdout.write(self.style.ERROR(
                    f"  Error for {user.username}: {e}"
                ))
                logger.exception(f"Report generation/send failed for {user.username}")

        self.stdout.write(f"\nSummary:")
        self.stdout.write(f"  Reports generated: {generated}")
        self.stdout.write(f"  Emails sent: {emailed}")
        self.stdout.write(f"  Errors: {errors}")
