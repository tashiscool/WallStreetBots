from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User

from .models import Company, Stock
from .models.models import (
    UserProfile, StrategyPortfolio, CircuitBreakerEvent,
    CircuitBreakerState, CircuitBreakerHistory, DigestLog,
    TaxLot, TaxLotSale, StrategyPerformanceSnapshot, CustomStrategy
)


class CompanyAdmin(admin.ModelAdmin):
    list_display = ("name", "ticker")


class StockAdmin(admin.ModelAdmin):
    list_display = ("company",)


class UserProfileInline(admin.StackedInline):
    """Inline UserProfile on User admin page."""
    model = UserProfile
    can_delete = False
    verbose_name_plural = 'Profile'
    fk_name = 'user'
    fieldsets = (
        ('Risk Assessment', {
            'fields': ('risk_tolerance', 'risk_score', 'investment_timeline', 'max_loss_tolerance'),
        }),
        ('Experience', {
            'fields': ('trading_experience', 'options_experience', 'options_level', 'crypto_experience'),
        }),
        ('Trading Mode', {
            'fields': ('is_paper_trading', 'live_trading_approved'),
        }),
        ('Onboarding', {
            'fields': ('onboarding_completed', 'onboarding_step', 'risk_assessment_completed',
                      'brokerage_connected', 'first_strategy_activated'),
        }),
        ('Preferences', {
            'fields': ('timezone', 'email_frequency', 'default_chart_timeframe'),
            'classes': ('collapse',),
        }),
    )


class CustomUserAdmin(BaseUserAdmin):
    """Extended User admin with UserProfile inline and platform roles."""
    inlines = (UserProfileInline,)
    list_display = ('username', 'email', 'first_name', 'last_name', 'is_staff', 'get_trading_mode', 'get_platform_roles')
    list_select_related = ('profile',)
    list_filter = BaseUserAdmin.list_filter + ('groups',)

    def get_trading_mode(self, obj):
        if hasattr(obj, 'profile'):
            return 'Paper' if obj.profile.is_paper_trading else 'Live'
        return 'N/A'
    get_trading_mode.short_description = 'Trading Mode'

    def get_platform_roles(self, obj):
        platform_roles = ['viewer', 'trader', 'risk_manager', 'admin']
        roles = obj.groups.filter(name__in=platform_roles).values_list('name', flat=True)
        return ', '.join(roles) if roles else 'None'
    get_platform_roles.short_description = 'Platform Roles'

    def get_inline_instances(self, request, obj=None):
        if not obj:
            return list()
        return super().get_inline_instances(request, obj)


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    """Admin interface for UserProfile."""
    list_display = (
        'user', 'risk_profile_display', 'trading_mode_display',
        'onboarding_completed', 'brokerage_connected', 'created_at'
    )
    list_filter = (
        'risk_tolerance', 'is_paper_trading', 'live_trading_approved',
        'onboarding_completed', 'risk_assessment_completed', 'brokerage_connected',
        'trading_experience', 'options_experience',
    )
    search_fields = ('user__username', 'user__email')
    readonly_fields = ('created_at', 'updated_at', 'last_active_at', 'last_login_at')
    ordering = ('-created_at',)

    fieldsets = (
        ('User', {
            'fields': ('user',),
        }),
        ('Risk Assessment', {
            'fields': ('risk_tolerance', 'risk_score', 'investment_timeline',
                      'max_loss_tolerance', 'risk_assessment_answers'),
        }),
        ('Financial Context', {
            'fields': ('investable_capital', 'income_source', 'annual_income',
                      'capital_is_risk_capital', 'net_worth'),
        }),
        ('Experience', {
            'fields': ('trading_experience', 'options_experience', 'options_level',
                      'crypto_experience', 'margin_experience', 'shorting_experience'),
        }),
        ('Onboarding Progress', {
            'fields': ('onboarding_completed', 'onboarding_step', 'onboarding_started_at',
                      'onboarding_completed_at', 'risk_assessment_completed',
                      'brokerage_connected', 'first_strategy_activated', 'first_trade_executed'),
        }),
        ('Trading Mode', {
            'fields': ('is_paper_trading', 'paper_trading_start',
                      'live_trading_approved', 'live_trading_approved_at'),
        }),
        ('Preferences', {
            'fields': ('dashboard_layout', 'default_chart_timeframe', 'timezone',
                      'theme', 'compact_mode', 'show_tutorial_hints'),
            'classes': ('collapse',),
        }),
        ('Communication', {
            'fields': ('email_frequency', 'email_trade_alerts', 'email_risk_alerts',
                      'email_performance_reports', 'push_notifications_enabled',
                      'sms_alerts_enabled', 'phone_number'),
            'classes': ('collapse',),
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at', 'last_active_at', 'last_login_at'),
        }),
    )

    def risk_profile_display(self, obj):
        return obj.risk_profile_name
    risk_profile_display.short_description = 'Risk Profile'

    def trading_mode_display(self, obj):
        return 'Paper' if obj.is_paper_trading else 'Live'
    trading_mode_display.short_description = 'Mode'


@admin.register(StrategyPortfolio)
class StrategyPortfolioAdmin(admin.ModelAdmin):
    """Admin interface for Strategy Portfolios."""
    list_display = (
        'name', 'user', 'risk_profile', 'is_active', 'is_template',
        'strategy_count', 'diversification_score', 'created_at'
    )
    list_filter = ('risk_profile', 'is_active', 'is_template')
    search_fields = ('name', 'user__username', 'description')
    readonly_fields = ('created_at', 'updated_at', 'last_rebalanced_at')
    ordering = ('-is_active', '-updated_at')

    fieldsets = (
        ('Basic Info', {
            'fields': ('user', 'name', 'description', 'risk_profile'),
        }),
        ('Status', {
            'fields': ('is_template', 'is_active'),
        }),
        ('Strategies', {
            'fields': ('strategies',),
        }),
        ('Metrics', {
            'fields': ('diversification_score', 'expected_sharpe', 'correlation_matrix', 'performance_metrics'),
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at', 'last_rebalanced_at'),
        }),
    )

    def strategy_count(self, obj):
        return obj.strategy_count
    strategy_count.short_description = 'Strategies'


@admin.register(CircuitBreakerEvent)
class CircuitBreakerEventAdmin(admin.ModelAdmin):
    """Admin interface for Circuit Breaker Events."""
    list_display = (
        'id', 'user', 'breaker_type', 'triggered_at',
        'current_recovery_mode', 'is_active', 'resolved_at'
    )
    list_filter = ('breaker_type', 'current_recovery_mode', 'resolution_method')
    search_fields = ('user__username',)
    readonly_fields = ('triggered_at',)
    ordering = ('-triggered_at',)

    fieldsets = (
        ('Event Info', {
            'fields': ('user', 'breaker_type', 'triggered_at', 'trigger_value', 'trigger_threshold'),
        }),
        ('Recovery', {
            'fields': ('current_recovery_mode', 'recovery_mode_until', 'recovery_stage',
                      'recovery_schedule', 'position_size_multiplier'),
        }),
        ('Recovery Metrics', {
            'fields': ('recovery_trades_count', 'recovery_profitable_trades', 'recovery_pnl'),
        }),
        ('Resolution', {
            'fields': ('is_active', 'resolved_at', 'resolution_method', 'notes'),
        }),
    )


# Re-register User admin with profile inline
admin.site.unregister(User)
admin.site.register(User, CustomUserAdmin)

@admin.register(CircuitBreakerState)
class CircuitBreakerStateAdmin(admin.ModelAdmin):
    """Admin interface for Circuit Breaker State."""
    list_display = (
        'breaker_type', 'user', 'status', 'vix_level',
        'tripped_at', 'can_trade_display', 'updated_at'
    )
    list_filter = ('breaker_type', 'status', 'vix_level')
    search_fields = ('user__username', 'trip_reason')
    readonly_fields = ('created_at', 'updated_at')
    ordering = ('-updated_at',)

    fieldsets = (
        ('Identification', {
            'fields': ('user', 'breaker_type'),
        }),
        ('Current State', {
            'fields': ('status', 'current_value', 'threshold'),
        }),
        ('Trip Information', {
            'fields': ('tripped_at', 'trip_reason', 'cooldown_until'),
        }),
        ('VIX Tracking', {
            'fields': ('vix_level', 'current_vix', 'last_vix_check'),
        }),
        ('Error Tracking', {
            'fields': ('errors_last_minute', 'error_window_start'),
        }),
        ('Equity Tracking', {
            'fields': ('start_of_day_equity', 'current_equity', 'daily_drawdown'),
        }),
        ('Data Freshness', {
            'fields': ('last_data_timestamp', 'last_daily_reset'),
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
        }),
    )

    def can_trade_display(self, obj):
        return 'Yes' if obj.can_trade else 'No'
    can_trade_display.short_description = 'Can Trade'


@admin.register(CircuitBreakerHistory)
class CircuitBreakerHistoryAdmin(admin.ModelAdmin):
    """Admin interface for Circuit Breaker History."""
    list_display = (
        'timestamp', 'get_breaker_type', 'action',
        'previous_status', 'new_status', 'reason'
    )
    list_filter = ('action', 'timestamp')
    search_fields = ('reason', 'state__breaker_type')
    readonly_fields = ('created_at',)
    ordering = ('-timestamp',)
    date_hierarchy = 'timestamp'

    fieldsets = (
        ('Event', {
            'fields': ('state', 'action', 'timestamp'),
        }),
        ('Status Change', {
            'fields': ('previous_status', 'new_status'),
        }),
        ('Values', {
            'fields': ('value_at_action', 'threshold_at_action'),
        }),
        ('Details', {
            'fields': ('reason', 'metadata'),
        }),
        ('Timestamps', {
            'fields': ('created_at',),
        }),
    )

    def get_breaker_type(self, obj):
        return obj.state.breaker_type
    get_breaker_type.short_description = 'Breaker Type'


@admin.register(DigestLog)
class DigestLogAdmin(admin.ModelAdmin):
    """Admin interface for Digest Logs."""
    list_display = (
        'id', 'user', 'digest_type', 'period_start',
        'delivery_status', 'sent_at', 'opened_at'
    )
    list_filter = ('digest_type', 'delivery_status', 'scheduled_at')
    search_fields = ('user__username', 'user__email', 'email_recipient')
    readonly_fields = ('scheduled_at', 'sent_at', 'opened_at', 'clicked_at')
    ordering = ('-scheduled_at',)
    date_hierarchy = 'scheduled_at'

    fieldsets = (
        ('User & Type', {
            'fields': ('user', 'digest_type'),
        }),
        ('Period Coverage', {
            'fields': ('period_start', 'period_end'),
        }),
        ('Delivery Status', {
            'fields': (
                'delivery_status', 'error_message',
                'scheduled_at', 'sent_at'
            ),
        }),
        ('Engagement Tracking', {
            'fields': ('opened_at', 'clicked_at'),
        }),
        ('Email Details', {
            'fields': ('email_subject', 'email_recipient'),
        }),
        ('Data Snapshot', {
            'fields': ('data_snapshot',),
            'classes': ('collapse',),
        }),
    )

    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user')


@admin.register(TaxLot)
class TaxLotAdmin(admin.ModelAdmin):
    """Admin interface for Tax Lots."""
    list_display = (
        'symbol', 'user', 'remaining_quantity', 'cost_basis_per_share',
        'acquired_at', 'is_long_term', 'unrealized_gain', 'is_closed'
    )
    list_filter = ('is_long_term', 'is_closed', 'acquisition_type', 'symbol')
    search_fields = ('symbol', 'user__username', 'order_id')
    readonly_fields = ('created_at', 'updated_at', 'is_long_term', 'days_held')
    ordering = ('-acquired_at',)
    date_hierarchy = 'acquired_at'

    fieldsets = (
        ('Identification', {
            'fields': ('user', 'symbol', 'order_id'),
        }),
        ('Quantity', {
            'fields': ('original_quantity', 'remaining_quantity'),
        }),
        ('Cost Basis', {
            'fields': ('cost_basis_per_share', 'total_cost_basis'),
        }),
        ('Acquisition', {
            'fields': ('acquired_at', 'acquisition_type'),
        }),
        ('Tax Status', {
            'fields': ('is_long_term', 'days_held'),
        }),
        ('Wash Sale', {
            'fields': (
                'wash_sale_adjustment', 'is_wash_sale_replacement',
                'wash_sale_disallowed_loss'
            ),
        }),
        ('Market Data', {
            'fields': (
                'current_price', 'market_value',
                'unrealized_gain', 'unrealized_gain_pct', 'price_updated_at'
            ),
        }),
        ('Status', {
            'fields': ('is_closed', 'closed_at'),
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
        }),
    )

    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user')


@admin.register(TaxLotSale)
class TaxLotSaleAdmin(admin.ModelAdmin):
    """Admin interface for Tax Lot Sales."""
    list_display = (
        'symbol', 'user', 'quantity_sold', 'sale_price',
        'realized_gain', 'is_long_term', 'is_wash_sale', 'sold_at'
    )
    list_filter = ('is_long_term', 'is_gain', 'is_wash_sale', 'lot_selection_method')
    search_fields = ('symbol', 'user__username', 'order_id')
    readonly_fields = ('created_at',)
    ordering = ('-sold_at',)
    date_hierarchy = 'sold_at'

    fieldsets = (
        ('Identification', {
            'fields': ('user', 'tax_lot', 'symbol', 'order_id'),
        }),
        ('Sale Details', {
            'fields': ('quantity_sold', 'sale_price', 'proceeds', 'sold_at'),
        }),
        ('Cost Basis', {
            'fields': ('cost_basis_sold',),
        }),
        ('Gain/Loss', {
            'fields': ('realized_gain', 'is_gain', 'is_long_term'),
        }),
        ('Wash Sale', {
            'fields': (
                'is_wash_sale', 'wash_sale_disallowed',
                'wash_sale_replacement_lot_id'
            ),
        }),
        ('Method', {
            'fields': ('lot_selection_method',),
        }),
        ('Timestamps', {
            'fields': ('created_at',),
        }),
    )

    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user', 'tax_lot')


@admin.register(StrategyPerformanceSnapshot)
class StrategyPerformanceSnapshotAdmin(admin.ModelAdmin):
    """Admin interface for Strategy Performance Snapshots."""
    list_display = (
        'strategy_name', 'snapshot_date', 'period',
        'total_return_pct', 'sharpe_ratio', 'win_rate',
        'rank_by_sharpe', 'trades_count'
    )
    list_filter = ('period', 'snapshot_date', 'strategy_name')
    search_fields = ('strategy_name',)
    readonly_fields = ('created_at', 'updated_at')
    ordering = ('-snapshot_date', 'rank_by_sharpe')
    date_hierarchy = 'snapshot_date'

    fieldsets = (
        ('Identification', {
            'fields': ('strategy_name', 'snapshot_date', 'period'),
        }),
        ('Performance Metrics', {
            'fields': (
                'total_return_pct', 'sharpe_ratio', 'sortino_ratio',
                'max_drawdown_pct', 'win_rate', 'profit_factor'
            ),
        }),
        ('Trading Volume', {
            'fields': (
                'trades_count', 'winning_trades', 'losing_trades',
                'avg_trade_pnl', 'best_trade_pnl', 'worst_trade_pnl',
                'avg_hold_duration_hours'
            ),
        }),
        ('Risk Metrics', {
            'fields': ('volatility', 'var_95', 'calmar_ratio'),
        }),
        ('Benchmark Comparison', {
            'fields': (
                'benchmark_return_pct', 'vs_spy_return',
                'beta', 'alpha', 'correlation_spy'
            ),
        }),
        ('Rankings', {
            'fields': (
                'rank_by_sharpe', 'rank_by_return',
                'rank_by_risk_adjusted', 'total_strategies_ranked'
            ),
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
        }),
    )


@admin.register(CustomStrategy)
class CustomStrategyAdmin(admin.ModelAdmin):
    """Admin interface for Custom Strategies."""
    list_display = (
        'name', 'user', 'universe', 'is_active', 'is_validated',
        'entry_conditions_count', 'exit_conditions_count',
        'has_stop_loss', 'created_at'
    )
    list_filter = ('is_active', 'is_validated', 'is_template', 'is_public', 'universe')
    search_fields = ('name', 'user__username', 'description')
    readonly_fields = ('created_at', 'updated_at', 'last_backtest_at', 'activated_at', 'deactivated_at')
    ordering = ('-updated_at',)
    date_hierarchy = 'created_at'

    fieldsets = (
        ('Basic Info', {
            'fields': ('user', 'name', 'description'),
        }),
        ('Strategy Definition', {
            'fields': ('definition', 'universe', 'custom_symbols'),
            'classes': ('collapse',),
        }),
        ('Status', {
            'fields': ('is_active', 'is_validated', 'validation_errors', 'validation_warnings'),
        }),
        ('Sharing', {
            'fields': ('is_template', 'is_public', 'cloned_from', 'clone_count'),
        }),
        ('Performance', {
            'fields': ('backtest_results', 'live_performance'),
            'classes': ('collapse',),
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at', 'last_backtest_at', 'activated_at', 'deactivated_at'),
        }),
    )

    def entry_conditions_count(self, obj):
        return obj.entry_conditions_count
    entry_conditions_count.short_description = 'Entry'

    def exit_conditions_count(self, obj):
        return obj.exit_conditions_count
    exit_conditions_count.short_description = 'Exit'

    def has_stop_loss(self, obj):
        return 'Yes' if obj.has_stop_loss else 'No'
    has_stop_loss.short_description = 'Stop Loss'

    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user')


# ---------------------------------------------------------------------------
# Audit Log Admin
# ---------------------------------------------------------------------------
from backend.auth0login.audit import AuditLog


@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    """Read-only admin for browsing the audit trail."""
    list_display = ('timestamp', 'event_type', 'severity', 'username', 'description_short', 'ip_address')
    list_filter = ('severity', 'event_type')
    search_fields = ('username', 'description', 'event_type', 'correlation_id')
    readonly_fields = [f.name for f in AuditLog._meta.get_fields() if hasattr(f, 'name')]
    ordering = ('-timestamp',)
    date_hierarchy = 'timestamp'

    def description_short(self, obj):
        return obj.description[:80] + '...' if len(obj.description) > 80 else obj.description
    description_short.short_description = 'Description'

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


# Register base models
admin.site.register(Company, CompanyAdmin)
admin.site.register(Stock, StockAdmin)
