"""
Migration to add RiskAssessment model for risk profiling questionnaire.
"""

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('auth0login', '0003_backfill_trading_gates'),
    ]

    operations = [
        migrations.CreateModel(
            name='RiskAssessment',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('responses', models.JSONField(default=dict, help_text='Questionnaire responses with answer values and scores')),
                ('calculated_score', models.IntegerField(default=0, help_text='Total calculated risk score (6-30 range)')),
                ('recommended_profile', models.CharField(choices=[('conservative', 'Conservative'), ('moderate', 'Moderate'), ('aggressive', 'Aggressive')], default='moderate', help_text='System-recommended risk profile based on score', max_length=20)),
                ('selected_profile', models.CharField(blank=True, choices=[('conservative', 'Conservative'), ('moderate', 'Moderate'), ('aggressive', 'Aggressive')], help_text="User's selected profile (if different from recommended)", max_length=20, null=True)),
                ('profile_override', models.BooleanField(default=False, help_text='Whether user selected a different profile than recommended')),
                ('override_acknowledged', models.BooleanField(default=False, help_text='Whether user acknowledged the override warning')),
                ('version', models.IntegerField(default=1, help_text='Questionnaire version used for this assessment')),
                ('completed_at', models.DateTimeField(blank=True, help_text='When the assessment was completed', null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('user', models.ForeignKey(help_text='User who completed this assessment', on_delete=django.db.models.deletion.CASCADE, related_name='risk_assessments', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': 'Risk Assessment',
                'verbose_name_plural': 'Risk Assessments',
                'ordering': ['-created_at'],
            },
        ),
        migrations.AddIndex(
            model_name='riskassessment',
            index=models.Index(fields=['user', 'version'], name='auth0login__user_id_f1a2b3_idx'),
        ),
        migrations.AddIndex(
            model_name='riskassessment',
            index=models.Index(fields=['user', '-completed_at'], name='auth0login__user_id_c4d5e6_idx'),
        ),
    ]
