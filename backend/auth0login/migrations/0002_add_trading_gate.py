"""
Migration to add TradingGate model for paper-to-live trading enforcement.
"""

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('auth0login', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='TradingGate',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('paper_trading_started_at', models.DateTimeField(blank=True, help_text='When paper trading was first enabled', null=True)),
                ('paper_trading_days_required', models.IntegerField(default=14, help_text='Minimum days of paper trading required before live trading')),
                ('paper_performance_snapshot', models.JSONField(blank=True, default=dict, help_text='Performance metrics at time of live trading request (P&L, trades, Sharpe, etc.)')),
                ('live_trading_requested_at', models.DateTimeField(blank=True, help_text='When user requested to switch to live trading', null=True)),
                ('live_trading_approved_at', models.DateTimeField(blank=True, help_text='When live trading was approved', null=True)),
                ('live_trading_approved', models.BooleanField(default=False, help_text='Whether live trading has been approved')),
                ('approval_method', models.CharField(blank=True, choices=[('auto', 'Automatic - All criteria met'), ('manual', 'Manual - Approved by admin'), ('override', 'Override - Admin bypass')], help_text='How live trading was approved', max_length=20, null=True)),
                ('live_trading_denied_at', models.DateTimeField(blank=True, help_text='When live trading request was denied (if applicable)', null=True)),
                ('denial_reason', models.TextField(blank=True, help_text='Reason for denial if request was rejected', null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('user', models.OneToOneField(help_text='User this gate applies to', on_delete=django.db.models.deletion.CASCADE, related_name='trading_gate', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': 'Trading Gate',
                'verbose_name_plural': 'Trading Gates',
                'ordering': ['-created_at'],
            },
        ),
        migrations.AddIndex(
            model_name='tradinggate',
            index=models.Index(fields=['user', 'live_trading_approved'], name='auth0login__user_id_7c4d0a_idx'),
        ),
        migrations.AddIndex(
            model_name='tradinggate',
            index=models.Index(fields=['live_trading_approved', 'created_at'], name='auth0login__live_tr_a3e5f9_idx'),
        ),
    ]
