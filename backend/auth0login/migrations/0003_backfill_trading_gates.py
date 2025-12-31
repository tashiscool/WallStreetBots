"""
Data migration to backfill TradingGate for existing users.

Logic:
- Users with paper_trading=False (live trading) are grandfathered as approved
- Users with paper_trading=True start their paper trading period now
- Users without Credential records get a gate with paper trading starting now
"""

from django.db import migrations
from django.utils import timezone


def backfill_trading_gates(apps, schema_editor):
    """Create TradingGate records for all existing users."""
    User = apps.get_model('auth', 'User')
    TradingGate = apps.get_model('auth0login', 'TradingGate')
    Credential = apps.get_model('auth0login', 'Credential')

    now = timezone.now()

    for user in User.objects.all():
        # Skip if user already has a TradingGate
        if TradingGate.objects.filter(user=user).exists():
            continue

        # Check if user has credentials (indicating they've set up trading)
        has_credentials = Credential.objects.filter(user=user).exists()

        if has_credentials:
            # User has set up trading - determine their status
            # In production, you'd check their actual trading mode setting
            # For safety, we default to requiring paper trading unless
            # you have a specific field to check

            # Option 1: Conservative - require paper trading for everyone
            TradingGate.objects.create(
                user=user,
                paper_trading_started_at=now,
                paper_trading_days_required=14,
                live_trading_approved=False,
            )
        else:
            # User hasn't set up trading yet - create pending gate
            TradingGate.objects.create(
                user=user,
                paper_trading_started_at=None,  # Will be set when they start
                paper_trading_days_required=14,
                live_trading_approved=False,
            )


def grandfather_live_traders(apps, schema_editor):
    """
    Alternative backfill that grandfathers users who were already live trading.

    Use this if you want to be lenient with existing users.
    Uncomment and modify the main backfill function to use this logic instead.
    """
    User = apps.get_model('auth', 'User')
    TradingGate = apps.get_model('auth0login', 'TradingGate')
    Credential = apps.get_model('auth0login', 'Credential')

    now = timezone.now()

    for user in User.objects.all():
        if TradingGate.objects.filter(user=user).exists():
            continue

        has_credentials = Credential.objects.filter(user=user).exists()

        if has_credentials:
            # Grandfather existing users with credentials as approved
            TradingGate.objects.create(
                user=user,
                paper_trading_started_at=now - timezone.timedelta(days=30),  # Pretend they've been paper trading
                paper_trading_days_required=14,
                live_trading_approved=True,
                live_trading_approved_at=now,
                approval_method='override',
                paper_performance_snapshot={
                    'note': 'Grandfathered during migration',
                    'migrated_at': now.isoformat(),
                },
            )
        else:
            TradingGate.objects.create(
                user=user,
                paper_trading_started_at=None,
                paper_trading_days_required=14,
                live_trading_approved=False,
            )


def reverse_backfill(apps, schema_editor):
    """Remove all TradingGate records (for migration reversal)."""
    TradingGate = apps.get_model('auth0login', 'TradingGate')
    TradingGate.objects.all().delete()


class Migration(migrations.Migration):

    dependencies = [
        ('auth0login', '0002_add_trading_gate'),
    ]

    operations = [
        # Choose ONE of these:
        # Option 1: Conservative - everyone needs to complete paper trading
        migrations.RunPython(backfill_trading_gates, reverse_backfill),

        # Option 2: Lenient - grandfather existing live traders
        # migrations.RunPython(grandfather_live_traders, reverse_backfill),
    ]
