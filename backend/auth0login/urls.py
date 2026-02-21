from django.urls import include, path

from . import views
from . import api_views

urlpatterns = [
    # Authentication
    path("", views.login, name="login"),
    path("logout", views.logout, name="logout"),
    path("", include("django.contrib.auth.urls")),
    path("", include("social_django.urls")),

    # Trading Dashboard
    path("dashboard", views.dashboard, name="dashboard"),
    path("positions", views.positions, name="positions"),
    path("orders", views.orders, name="orders"),

    # Strategy Management
    path("strategies", views.strategies, name="strategies"),
    path("strategy-portfolios", views.strategy_portfolios, name="strategy-portfolios"),
    path("strategies/wsb-dip-bot", views.strategy_wsb_dip_bot, name="strategy-wsb-dip-bot"),
    path("strategies/wheel", views.strategy_wheel, name="strategy-wheel"),
    path("strategies/momentum-weeklies", views.strategy_momentum_weeklies, name="strategy-momentum-weeklies"),
    path("strategies/earnings-protection", views.strategy_earnings_protection, name="strategy-earnings-protection"),
    path("strategies/debit-spreads", views.strategy_debit_spreads, name="strategy-debit-spreads"),
    path("strategies/leaps-tracker", views.strategy_leaps_tracker, name="strategy-leaps-tracker"),
    path("strategies/lotto-scanner", views.strategy_lotto_scanner, name="strategy-lotto-scanner"),
    path("strategies/swing-trading", views.strategy_swing_trading, name="strategy-swing-trading"),
    path("strategies/spx-credit-spreads", views.strategy_spx_credit_spreads, name="strategy-spx-credit-spreads"),
    path("strategies/index-baseline", views.strategy_index_baseline, name="strategy-index-baseline"),
    path("backtesting", views.backtesting, name="backtesting"),

    # Risk & Analytics
    path("risk", views.risk_management, name="risk"),
    path("analytics", views.analytics, name="analytics"),

    # System
    path("alerts", views.alerts, name="alerts"),
    path("system-status", views.system_status, name="system-status"),

    # Account
    path("settings", views.user_settings_page, name="settings"),
    path("user-settings", views.user_settings, name="user-settings"),
    path("machine-learning", views.machine_learning, name="machine-learning"),
    path("setup", views.setup_wizard, name="setup-wizard"),

    # Advanced Features
    path("crypto", views.crypto_trading, name="crypto"),
    path("extended-hours", views.extended_hours, name="extended-hours"),
    path("margin-borrow", views.margin_borrow, name="margin-borrow"),
    path("exotic-spreads", views.exotic_spreads, name="exotic-spreads"),
    path("feature-status", views.feature_status, name="feature-status"),
    path("ml-training", views.ml_training, name="ml-training"),

    # New Dashboard Pages
    path("market-context", views.market_context, name="market-context"),
    path("allocations", views.allocations, name="allocations"),
    path("circuit-breakers", views.circuit_breakers, name="circuit-breakers"),
    path("ml-agents", views.ml_agents, name="ml-agents"),

    # API Endpoints (JSON responses for AJAX)
    path("api/backtest/run", api_views.run_backtest, name="api-run-backtest"),
    path("api/spreads/build", api_views.build_spread, name="api-build-spread"),
    path("api/spreads/suggest", api_views.suggest_spreads, name="api-suggest-spreads"),
    path("api/borrow/locate", api_views.get_locate_quote, name="api-locate-quote"),
    path("api/features", api_views.feature_availability, name="api-features"),
    path("api/alpaca/test", api_views.test_alpaca_connection, name="api-test-alpaca"),
    path("api/wizard/save", api_views.save_wizard_config, name="api-save-wizard"),
    path("api/settings/email/test", api_views.test_email, name="api-test-email"),
    path("api/settings/save", api_views.save_settings, name="api-save-settings"),

    # Trading Gate API (Paper-to-Live Trading Transition)
    path("api/trading-gate/status", api_views.trading_gate_status, name="api-trading-gate-status"),
    path("api/trading-gate/request-live", api_views.trading_gate_request_live, name="api-trading-gate-request-live"),
    path("api/trading-gate/requirements", api_views.trading_gate_requirements, name="api-trading-gate-requirements"),
    path("api/trading-gate/start-paper", api_views.trading_gate_start_paper, name="api-trading-gate-start-paper"),

    # Risk Assessment API (Questionnaire and Profile Determination)
    path("api/risk-assessment/questions", api_views.risk_assessment_questions, name="api-risk-assessment-questions"),
    path("api/risk-assessment/submit", api_views.risk_assessment_submit, name="api-risk-assessment-submit"),
    path("api/risk-assessment/result", api_views.risk_assessment_result, name="api-risk-assessment-result"),
    path("api/risk-assessment/calculate", api_views.risk_assessment_calculate, name="api-risk-assessment-calculate"),

    # Strategy Recommendations API
    path("api/strategy-recommendations", api_views.strategy_recommendations, name="api-strategy-recommendations"),
    path("api/strategy/<str:strategy_id>", api_views.strategy_details, name="api-strategy-details"),

    # Benchmark Comparison API (P&L vs SPY)
    path("api/performance/vs-benchmark", api_views.performance_vs_benchmark, name="api-performance-vs-benchmark"),
    path("api/performance/pnl-with-benchmark", api_views.portfolio_pnl_with_benchmark, name="api-pnl-with-benchmark"),
    path("api/performance/benchmark-chart", api_views.benchmark_chart_data, name="api-benchmark-chart"),

    # Trade Explanation API (Signal Transparency)
    path("api/trades/<str:trade_id>/explanation", api_views.trade_explanation, name="api-trade-explanation"),
    path("api/trades/<str:trade_id>/signals", api_views.trade_signals, name="api-trade-signals"),
    path("api/trades/<str:trade_id>/similar", api_views.trade_similar, name="api-trade-similar"),
    path("api/trades/with-explanations", api_views.trade_list_with_explanations, name="api-trades-with-explanations"),

    # VIX Monitoring API
    path("api/vix/status", api_views.vix_status, name="api-vix-status"),
    path("api/vix/history", api_views.vix_history, name="api-vix-history"),
    path("api/circuit-breaker/status", api_views.circuit_breaker_status, name="api-circuit-breaker-status"),

    # Allocation Management API
    path("api/allocations/", api_views.allocation_list, name="api-allocation-list"),
    path("api/allocations/initialize", api_views.allocation_initialize, name="api-allocation-initialize"),
    path("api/allocations/rebalance", api_views.allocation_rebalance, name="api-allocation-rebalance"),
    path("api/allocations/reconcile", api_views.allocation_reconcile, name="api-allocation-reconcile"),
    path("api/allocations/recalculate", api_views.allocation_recalculate, name="api-allocation-recalculate"),
    path("api/allocations/<str:strategy_name>/", api_views.allocation_detail, name="api-allocation-detail"),
    path("api/allocations/<str:strategy_name>/update", api_views.allocation_update, name="api-allocation-update"),

    # Strategy Configuration API
    path("api/strategies/<str:strategy_name>/config", api_views.strategy_config_save, name="api-strategy-config-save"),
    path("api/strategies/<str:strategy_name>/config/get", api_views.strategy_config_get, name="api-strategy-config-get"),

    # Circuit Breaker Recovery API
    path("api/circuit-breakers/history/", api_views.circuit_breaker_history, name="api-circuit-breaker-history"),
    path("api/circuit-breakers/current/", api_views.circuit_breaker_current, name="api-circuit-breaker-current"),
    path("api/circuit-breakers/timeline/", api_views.circuit_breaker_timeline, name="api-circuit-breaker-timeline"),
    path("api/circuit-breakers/<int:event_id>/advance-recovery/", api_views.circuit_breaker_advance, name="api-circuit-breaker-advance"),
    path("api/circuit-breakers/<int:event_id>/reset/", api_views.circuit_breaker_event_reset, name="api-circuit-breaker-event-reset"),
    # Backward-compatible breaker reset path by breaker type
    path("api/circuit-breakers/<str:breaker_type>/reset/", api_views.circuit_breaker_reset, name="api-circuit-breaker-reset-legacy"),
    path("api/circuit-breakers/<int:event_id>/early-recovery/", api_views.circuit_breaker_early_recovery, name="api-circuit-breaker-early-recovery"),
    path("api/circuit-breakers/<int:event_id>/timeline/", api_views.circuit_breaker_timeline, name="api-circuit-breaker-timeline-detail"),

    # Market Context API
    path("api/market-context/", api_views.market_context, name="api-market-context"),
    path("api/market-context/overview/", api_views.market_overview, name="api-market-overview"),
    path("api/market-context/sectors/", api_views.sector_performance, name="api-sector-performance"),
    path("api/market-context/events/", api_views.holdings_events, name="api-holdings-events"),
    path("api/market-context/calendar/", api_views.economic_calendar, name="api-economic-calendar"),

    # Strategy Portfolio API
    path("api/portfolios/", api_views.portfolio_list, name="api-portfolio-list"),
    path("api/portfolios/create", api_views.portfolio_create, name="api-portfolio-create"),
    path("api/portfolios/templates/", api_views.portfolio_templates, name="api-portfolio-templates"),
    path("api/portfolios/from-template", api_views.portfolio_create_from_template, name="api-portfolio-from-template"),
    path("api/portfolios/strategies/", api_views.available_strategies, name="api-available-strategies"),
    path("api/portfolios/analyze", api_views.analyze_portfolio, name="api-analyze-portfolio"),
    path("api/portfolios/optimize", api_views.optimize_portfolio, name="api-optimize-portfolio"),
    path("api/portfolios/suggest", api_views.suggest_strategies, name="api-suggest-strategies"),
    path("api/portfolios/<int:portfolio_id>/", api_views.portfolio_detail, name="api-portfolio-detail"),
    path("api/portfolios/<int:portfolio_id>/update", api_views.portfolio_update, name="api-portfolio-update"),
    path("api/portfolios/<int:portfolio_id>/delete", api_views.portfolio_delete, name="api-portfolio-delete"),
    path("api/portfolios/<int:portfolio_id>/activate", api_views.portfolio_activate, name="api-portfolio-activate"),
    path("api/portfolios/<int:portfolio_id>/deactivate", api_views.portfolio_deactivate, name="api-portfolio-deactivate"),

    # User Profile API
    path("api/profile/", api_views.user_profile, name="api-user-profile"),
    path("api/profile/update", api_views.update_user_profile, name="api-update-profile"),
    path("api/profile/summary/", api_views.profile_summary, name="api-profile-summary"),
    path("api/profile/onboarding-status/", api_views.profile_onboarding_status, name="api-onboarding-status"),
    path("api/profile/complete-step/", api_views.profile_complete_step, name="api-complete-step"),
    path("api/profile/risk-questions/", api_views.profile_risk_questions, name="api-risk-questions"),
    path("api/profile/risk-assessment/submit", api_views.profile_submit_risk_assessment, name="api-submit-risk-assessment"),
    path("api/profile/recommendations/", api_views.profile_recommendations, name="api-profile-recommendations"),
    path("api/profile/trading-mode/", api_views.profile_switch_trading_mode, name="api-switch-trading-mode"),

    # Circuit Breaker State & History API
    path("api/circuit-breaker-states/", api_views.circuit_breaker_state_list, name="api-circuit-breaker-states"),
    path("api/circuit-breaker-states/initialize/", api_views.circuit_breaker_initialize, name="api-circuit-breaker-initialize"),
    path("api/circuit-breaker-states/daily-reset/", api_views.circuit_breaker_daily_reset, name="api-circuit-breaker-daily-reset"),
    path("api/circuit-breaker-states/<str:breaker_type>/", api_views.circuit_breaker_state_detail, name="api-circuit-breaker-state-detail"),
    path("api/circuit-breaker-states/<str:breaker_type>/reset/", api_views.circuit_breaker_reset, name="api-circuit-breaker-state-reset"),
    path("api/circuit-breaker-history/", api_views.circuit_breaker_history_list, name="api-circuit-breaker-history"),

    # Trade Reasoning API
    path("api/trade-reasoning/", api_views.trade_reasoning_list, name="api-trade-reasoning-list"),
    path("api/trade-reasoning/stats/", api_views.trade_reasoning_stats, name="api-trade-reasoning-stats"),
    path("api/trade-reasoning/<str:trade_id>/", api_views.trade_reasoning_detail, name="api-trade-reasoning-detail"),
    path("api/trade-reasoning/<str:trade_id>/analyze/", api_views.trade_reasoning_analyze, name="api-trade-reasoning-analyze"),
    path("api/trade-reasoning/<str:trade_id>/record-exit/", api_views.trade_reasoning_record_exit, name="api-trade-reasoning-record-exit"),

    # Digest Email API
    path("api/digest/preview", api_views.digest_preview, name="api-digest-preview"),
    path("api/digest/send-test", api_views.digest_send_test, name="api-digest-send-test"),
    path("api/digest/history", api_views.digest_history, name="api-digest-history"),
    path("api/digest/<int:digest_id>/", api_views.digest_detail, name="api-digest-detail"),
    path("api/digest/preferences", api_views.digest_update_preferences, name="api-digest-preferences"),
    path("api/digest/unsubscribe", api_views.digest_unsubscribe, name="api-digest-unsubscribe"),
    path("api/digest/<int:digest_id>/track-open", api_views.digest_track_open, name="api-digest-track-open"),
    path("api/digest/<int:digest_id>/track-click", api_views.digest_track_click, name="api-digest-track-click"),

    # Tax Optimization API
    path("api/tax/lots/", api_views.tax_lots_list, name="api-tax-lots"),
    path("api/tax/lots/<str:symbol>/", api_views.tax_lots_by_symbol, name="api-tax-lots-symbol"),
    path("api/tax/harvesting-opportunities/", api_views.tax_harvesting_opportunities, name="api-tax-harvesting"),
    path("api/tax/preview-sale/", api_views.tax_preview_sale, name="api-tax-preview-sale"),
    path("api/tax/wash-sale-check/<str:symbol>/", api_views.tax_wash_sale_check, name="api-tax-wash-sale-check"),
    path("api/tax/year-summary/", api_views.tax_year_summary, name="api-tax-year-summary"),
    path("api/tax/suggest-lot-selection/", api_views.tax_suggest_lot_selection, name="api-tax-suggest-lot"),
    path("api/tax/create-lot/", api_views.tax_create_lot, name="api-tax-create-lot"),
    path("api/tax/update-prices/", api_views.tax_update_prices, name="api-tax-update-prices"),

    # Tax Optimization UI
    path("tax-optimization", views.tax_optimization, name="tax-optimization"),

    # Strategy Leaderboard UI
    path("leaderboard", views.strategy_leaderboard, name="leaderboard"),

    # Strategy Leaderboard API
    path("api/leaderboard/", api_views.leaderboard, name="api-leaderboard"),
    path("api/leaderboard/compare/", api_views.leaderboard_compare, name="api-leaderboard-compare"),
    path("api/leaderboard/hypothetical/", api_views.leaderboard_hypothetical, name="api-leaderboard-hypothetical"),
    path("api/leaderboard/top-performers/", api_views.leaderboard_top_performers, name="api-leaderboard-top-performers"),
    path("api/leaderboard/strategies/", api_views.leaderboard_all_strategies, name="api-leaderboard-strategies"),
    path("api/leaderboard/strategy/<str:strategy_name>/", api_views.leaderboard_strategy_details, name="api-leaderboard-strategy-details"),
    path("api/leaderboard/strategy/<str:strategy_name>/history/", api_views.leaderboard_strategy_history, name="api-leaderboard-strategy-history"),

    # Strategy Builder UI
    path("strategy-builder", views.strategy_builder, name="strategy-builder"),

    # Custom Strategy API
    path("api/custom-strategies/", api_views.custom_strategies_list, name="api-custom-strategies"),
    path("api/custom-strategies/indicators/", api_views.custom_strategy_indicators, name="api-custom-strategy-indicators"),
    path("api/custom-strategies/templates/", api_views.custom_strategy_templates, name="api-custom-strategy-templates"),
    path("api/custom-strategies/from-template/", api_views.custom_strategy_from_template, name="api-custom-strategy-from-template"),
    path("api/custom-strategies/<int:strategy_id>/", api_views.custom_strategy_detail, name="api-custom-strategy-detail"),
    path("api/custom-strategies/<int:strategy_id>/validate/", api_views.custom_strategy_validate, name="api-custom-strategy-validate"),
    path("api/custom-strategies/<int:strategy_id>/backtest/", api_views.custom_strategy_backtest, name="api-custom-strategy-backtest"),
    path("api/custom-strategies/<int:strategy_id>/activate/", api_views.custom_strategy_activate, name="api-custom-strategy-activate"),
    path("api/custom-strategies/<int:strategy_id>/deactivate/", api_views.custom_strategy_deactivate, name="api-custom-strategy-deactivate"),
    path("api/custom-strategies/<int:strategy_id>/clone/", api_views.custom_strategy_clone, name="api-custom-strategy-clone"),
    path("api/custom-strategies/<int:strategy_id>/preview-signals/", api_views.custom_strategy_preview_signals, name="api-custom-strategy-preview"),

    # Backtest Run Persistence API
    path("api/backtest/runs/", api_views.backtest_runs_list, name="api-backtest-runs"),
    path("api/backtest/runs/<str:run_id>/", api_views.backtest_run_detail, name="api-backtest-run-detail"),
    path("api/backtest/runs/<str:run_id>/trades/", api_views.backtest_run_trades, name="api-backtest-run-trades"),
    path("api/backtest/compare/", api_views.backtest_runs_compare, name="api-backtest-compare"),

    # Parameter Optimization API
    path("api/backtest/optimize/", api_views.optimization_run_start, name="api-optimization-start"),
    path("api/backtest/optimize/list/", api_views.optimization_runs_list, name="api-optimization-list"),
    path("api/backtest/optimize/<str:run_id>/status/", api_views.optimization_run_status, name="api-optimization-status"),
    path("api/backtest/optimize/<str:run_id>/results/", api_views.optimization_run_results, name="api-optimization-results"),

    # Wizard/Onboarding API
    path("api/wizard/session/", api_views.wizard_session, name="api-wizard-session"),
    path("api/wizard/step/<int:step>/", api_views.wizard_step_submit, name="api-wizard-step"),
    path("api/wizard/complete/", api_views.wizard_complete, name="api-wizard-complete"),
    path("api/wizard/skip/", api_views.wizard_skip, name="api-wizard-skip"),
    path("api/wizard/config/", api_views.wizard_config, name="api-wizard-config"),
    path("api/wizard/needs-setup/", api_views.wizard_needs_setup, name="api-wizard-needs-setup"),

    # ML/RL Agent API
    path("api/ml-agents/models/", api_views.ml_models_list, name="api-ml-models-list"),
    path("api/ml-agents/models/create/", api_views.ml_models_create, name="api-ml-models-create"),
    path("api/ml-agents/models/<str:model_id>/", api_views.ml_model_detail, name="api-ml-model-detail"),
    path("api/ml-agents/models/<str:model_id>/train/", api_views.ml_model_train, name="api-ml-model-train"),
    path("api/ml-agents/models/<str:model_id>/status/", api_views.ml_model_status, name="api-ml-model-status"),
    path("api/ml-agents/models/<str:model_id>/update/", api_views.ml_model_update, name="api-ml-model-update"),
    path("api/ml-agents/models/<str:model_id>/delete/", api_views.ml_model_delete, name="api-ml-model-delete"),
    path("api/ml-agents/agents/", api_views.rl_agents_list, name="api-rl-agents-list"),
    path("api/ml-agents/agents/create/", api_views.rl_agents_create, name="api-rl-agents-create"),
    path("api/ml-agents/agents/<str:agent_type>/", api_views.rl_agent_detail, name="api-rl-agent-detail"),
    path("api/ml-agents/agents/<str:agent_type>/train/", api_views.rl_agent_train, name="api-rl-agent-train"),
    path("api/ml-agents/agents/<str:agent_type>/status/", api_views.rl_agent_status, name="api-rl-agent-status"),
    path("api/ml-agents/agents/<str:agent_type>/config/", api_views.rl_agent_config, name="api-rl-agent-config"),
    path("api/ml-agents/training/", api_views.training_jobs_list, name="api-training-jobs-list"),
    path("api/ml-agents/training/<str:job_id>/", api_views.training_job_detail, name="api-training-job-detail"),
    path("api/ml-agents/training/<str:job_id>/cancel/", api_views.training_job_cancel, name="api-training-job-cancel"),
    path("api/ml-agents/training/<str:job_id>/progress/", api_views.training_job_update_progress, name="api-training-job-progress"),

    # Options Payoff Visualization API
    path("api/options/payoff-diagram/", api_views.options_payoff_diagram, name="api-options-payoff"),
    path("api/options/greeks-dashboard/", api_views.options_greeks_dashboard, name="api-options-greeks"),

    # Copy Trading API
    path("api/signals/providers/", api_views.signal_providers_list, name="api-signal-providers"),
    path("api/signals/providers/create/", api_views.signal_provider_create, name="api-signal-provider-create"),
    path("api/signals/providers/<int:provider_id>/", api_views.signal_provider_detail, name="api-signal-provider-detail"),
    path("api/signals/providers/<int:provider_id>/stats/", api_views.signal_provider_stats, name="api-signal-provider-stats"),
    path("api/signals/subscribe/", api_views.signal_subscribe, name="api-signal-subscribe"),
    path("api/signals/unsubscribe/", api_views.signal_unsubscribe, name="api-signal-unsubscribe"),
    path("api/signals/subscriptions/", api_views.signal_subscriptions_list, name="api-signal-subscriptions"),
    path("api/signals/replicate/", api_views.signal_manual_replicate, name="api-signal-replicate"),

    # PDF Report API
    path("api/reports/generate/", api_views.generate_pdf_report, name="api-generate-report"),
    path("api/reports/download/", api_views.download_pdf_report, name="api-download-report"),

    # Strategy Builder API
    path("api/strategy-builder/validate/", api_views.strategy_validate, name="api-strategy-validate"),
    path("api/strategy-builder/backtest/", api_views.strategy_compile_and_backtest, name="api-strategy-compile-backtest"),
    path("api/strategy-builder/indicators/", api_views.strategy_indicators_list, name="api-strategy-indicators"),
    path("api/strategy-builder/presets/", api_views.strategy_presets, name="api-strategy-presets"),

    # DEX Trading API
    path("api/dex/swap/", api_views.dex_swap, name="api-dex-swap"),
    path("api/dex/quote/", api_views.dex_quote, name="api-dex-quote"),
    path("api/dex/balance/", api_views.dex_wallet_balance, name="api-dex-balance"),

    # User Roles & Permissions API (RBAC)
    path("api/roles/", api_views.user_roles, name="api-user-roles"),
    path("api/roles/assign/", api_views.user_roles_assign, name="api-user-roles-assign"),

    # Audit Trail API
    path("api/audit/", api_views.audit_log_list, name="api-audit-log"),
    path("api/audit/summary/", api_views.audit_log_summary, name="api-audit-summary"),
]
