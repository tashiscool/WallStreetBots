"""
Migration to add TradeSignalSnapshot model for trade transparency.
"""

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('models', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='TradeSignalSnapshot',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('trade_id', models.CharField(db_index=True, help_text='Unique trade identifier (alpaca order ID or internal ID)', max_length=100, unique=True)),
                ('strategy_name', models.CharField(db_index=True, help_text='Strategy that generated the trade', max_length=100)),
                ('symbol', models.CharField(db_index=True, help_text='Trading symbol', max_length=20)),
                ('direction', models.CharField(choices=[('buy', 'Buy'), ('sell', 'Sell'), ('buy_to_cover', 'Buy to Cover'), ('sell_short', 'Sell Short')], help_text='Trade direction', max_length=20)),
                ('entry_price', models.DecimalField(decimal_places=4, help_text='Price at entry', max_digits=12)),
                ('quantity', models.DecimalField(decimal_places=4, help_text='Trade quantity', max_digits=12)),
                ('signals_at_entry', models.JSONField(default=dict, help_text='Snapshot of all signals at trade time')),
                ('confidence_score', models.IntegerField(default=50, help_text='Confidence score (0-100) based on signal alignment')),
                ('signals_triggered', models.IntegerField(default=0, help_text='Number of signals that triggered this trade')),
                ('signals_checked', models.IntegerField(default=0, help_text='Total signals checked')),
                ('explanation', models.TextField(blank=True, help_text='Auto-generated explanation of why this trade was triggered')),
                ('similar_historical_trades', models.JSONField(default=list, help_text='List of similar historical trade setups')),
                ('exit_price', models.DecimalField(blank=True, decimal_places=4, help_text='Price at exit', max_digits=12, null=True)),
                ('exit_timestamp', models.DateTimeField(blank=True, help_text='When the position was closed', null=True)),
                ('outcome', models.CharField(blank=True, choices=[('profit', 'Profit'), ('loss', 'Loss'), ('break_even', 'Break Even'), ('open', 'Still Open')], help_text='Trade outcome', max_length=20, null=True)),
                ('pnl_amount', models.DecimalField(blank=True, decimal_places=2, help_text='Profit/loss amount', max_digits=12, null=True)),
                ('pnl_percent', models.DecimalField(blank=True, decimal_places=2, help_text='Profit/loss percentage', max_digits=8, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True, help_text='When the snapshot was captured')),
                ('updated_at', models.DateTimeField(auto_now=True, help_text='Last update')),
                ('order', models.ForeignKey(blank=True, help_text='Associated Order if available', null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='signal_snapshots', to='models.order')),
            ],
            options={
                'verbose_name': 'Trade Signal Snapshot',
                'verbose_name_plural': 'Trade Signal Snapshots',
                'ordering': ['-created_at'],
            },
        ),
        migrations.AddIndex(
            model_name='tradesignalsnapshot',
            index=models.Index(fields=['strategy_name', 'created_at'], name='models_trad_strateg_a1b2c3_idx'),
        ),
        migrations.AddIndex(
            model_name='tradesignalsnapshot',
            index=models.Index(fields=['symbol', 'created_at'], name='models_trad_symbol_d4e5f6_idx'),
        ),
        migrations.AddIndex(
            model_name='tradesignalsnapshot',
            index=models.Index(fields=['confidence_score', 'created_at'], name='models_trad_confide_g7h8i9_idx'),
        ),
        migrations.AddIndex(
            model_name='tradesignalsnapshot',
            index=models.Index(fields=['outcome', 'created_at'], name='models_trad_outcome_j0k1l2_idx'),
        ),
    ]
