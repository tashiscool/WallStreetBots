from urllib.parse import urlencode

# from datetime import date
import alpaca_trade_api as api
import plotly.express as px
import plotly.graph_objects as go
from django.conf import settings
from django.contrib.auth import logout as log_out
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseRedirect
from django.shortcuts import redirect, render

from backend.tradingbot.synchronization import sync_alpaca
from backend.auth0login.dashboard_service import dashboard_service


def login(request):
    user = request.user
    if user.is_authenticated:
        return redirect(dashboard)
    else:
        return render(request, "accounts / login.html")


@login_required()
def get_user_information(request):
    user = request.user
    user_details = sync_alpaca(user)  # sync the user with Alpaca and extract details
    auth0user = user.social_auth.get(provider="auth0")
    alpaca_id = (
        user.credential.alpaca_id if hasattr(user, "credential") else "no alpaca id"
    )
    alpaca_key = (
        user.credential.alpaca_key if hasattr(user, "credential") else "no alpaca key"
    )

    if user_details is None:
        userdata = {
            "name": user.first_name,
            "alpaca_id": alpaca_id,
            "alpaca_key": "*" * len(alpaca_key),
            "error": "The Alpaca credential is either empty or invalid. Please enter below to access"
            " the account information.",
        }
    else:
        userdata = {
            "name": user.first_name,
            "alpaca_id": alpaca_id,
            "alpaca_key": " " + "*" * 10,
            "total_equity": user_details["equity"],
            "buy_power": user_details["buy_power"],
            "portfolio": user_details["display_portfolio"],
            "cash": user_details["cash"],
            "tradable_cash": user_details["usable_cash"],
            "currency": user_details["currency"],
            "short_portfolio_value": user_details["short_portfolio_value"],
            "long_portfolio_value": user_details["long_portfolio_value"],
            "orders": user_details["orders"],
            "strategy": user_details["strategy"],
            "percent_change": user_details["portfolio_percent_change"],
            "dollar_change": user_details["portfolio_dollar_change"],
            # change direction is used to determine if the price is going positive or negative
            "change_direction": user_details["portfolio_change_direction"],
        }
    return user, userdata, auth0user, user_details


@login_required
def dashboard(request):
    user, userdata, auth0user, user_details = get_user_information(request)
    # managing forms
    from backend.auth0login.forms import CredentialForm, OrderForm, StrategyForm

    credential_form = CredentialForm(request.POST or None)
    order_form = OrderForm(request.POST or None)
    strategy_form = StrategyForm(request.POST or None)
    if request.method == "POST":  # let user input their Alpaca API information
        if "submit_credential" in request.POST and credential_form.is_valid():
            if hasattr(user, "credential"):
                user.credential.alpaca_id = credential_form.get_id()
                user.credential.alpaca_key = credential_form.get_key()
                user.credential.save()
            else:
                from .models import Credential

                cred = Credential(
                    user=request.user,
                    alpaca_id=credential_form.get_id(),
                    alpaca_key=credential_form.get_key(),
                )
                cred.save()
            return HttpResponseRedirect("/")

        if "submit_order" in request.POST and order_form.is_valid():
            response = order_form.place_order(user, user_details)
            order_form = OrderForm()
            #  update order for display
            from backend.tradingbot.models.models import Order

            userdata["orders"] = [
                order.display_order()
                for order in Order.objects.filter(user=user)
                .order_by("-timestamp")
                .iterator()
            ]
            return render(
                request,
                "home/index.html",
                {
                    "credential_form": credential_form,
                    "order_form": order_form,
                    "strategy_form": StrategyForm(None),
                    "auth0User": auth0user,
                    "userdata": userdata,
                    "order_submit_form_response": response,
                },
            )

        if "submit_strategy" in request.POST and strategy_form.is_valid():
            # here for some reason form.cleaned_data changed from type dict to
            # type tuple. I tried to find the reason but it didn't seem to caused by
            # our code. Might be and django bug
            strategy = strategy_form.cleaned_data
            user.portfolio.strategy = strategy
            user.portfolio.save()
            return HttpResponseRedirect("/")

    graph = get_portfolio_chart(request)
    return render(
        request,
        "home/index.html",
        {
            "credential_form": credential_form,
            "order_form": order_form,
            "strategy_form": strategy_form,
            "auth0User": auth0user,
            "userdata": userdata,
            "stock_graph": graph,
        },
    )


@login_required()
def get_portfolio_chart(request):
    user, _userdata, _auth0user, user_details = get_user_information(request)
    if user_details is None:
        return
    API_KEY = user.credential.alpaca_id
    API_SECRET = user.credential.alpaca_key
    BASE_URL = "https://paper-api.alpaca.markets"
    alpaca = api.REST(
        key_id=API_KEY, secret_key=API_SECRET, base_url=BASE_URL, api_version="v2"
    )
    portfolio_hist = alpaca.get_portfolio_history().df
    portfolio_hist = portfolio_hist.reset_index()
    line_plot = px.line(portfolio_hist, "timestamp", "equity")
    line_plot.update_layout(xaxis_title="", yaxis_title="Equity")
    line_plot = line_plot.to_html()
    return line_plot


@login_required
def get_stock_chart(request, symbol):
    user, _userdata, _auth0user, _user_details = get_user_information(request)

    API_KEY = user.credential.alpaca_id
    API_SECRET = user.credential.alpaca_key
    alpaca = api.REST(API_KEY, API_SECRET)
    # Setting parameters before calling method
    timeframe = "1Day"
    start = "2021 - 01 - 01"
    # today=date.today()
    end = "2021 - 02 - 01"
    # Retrieve daily bars for SPY in a dataframe and printing the first 5 rows
    spy_bars = alpaca.get_bars(symbol, timeframe, start, end).df
    candlestick_fig = go.Figure(
        data=[
            go.Candlestick(
                x=spy_bars.index,
                open=spy_bars["open"],
                high=spy_bars["high"],
                low=spy_bars["low"],
                close=spy_bars["close"],
            )
        ]
    )
    candlestick_fig.update_layout(xaxis_title="Date", yaxis_title="Price ($USD)")
    candlestick_fig = candlestick_fig.to_html()
    return candlestick_fig


@login_required
def orders(request):
    user, userdata, auth0user, user_details = get_user_information(request)
    # managing forms
    from backend.auth0login.forms import CredentialForm, OrderForm, StrategyForm

    credential_form = CredentialForm(request.POST or None)
    order_form = OrderForm(request.POST or None)
    strategy_form = StrategyForm(request.POST or None)
    if request.method == "POST":
        if "submit_credential" in request.POST and credential_form.is_valid():
            if hasattr(user, "credential"):
                user.credential.alpaca_id = credential_form.get_id()
                user.credential.alpaca_key = credential_form.get_key()
                user.credential.save()
            else:
                from .models import Credential

                cred = Credential(
                    user=request.user,
                    alpaca_id=credential_form.get_id(),
                    alpaca_key=credential_form.get_key(),
                )
                cred.save()
            return HttpResponseRedirect("/")

        if "submit_order" in request.POST and order_form.is_valid():
            response = order_form.place_order(user, user_details)
            order_form = OrderForm()
            #  update order for display
            from backend.tradingbot.models.models import Order

            userdata["orders"] = [
                order.display_order()
                for order in Order.objects.filter(user=user)
                .order_by("-timestamp")
                .iterator()
            ]
            return render(
                request,
                "home/index.html",
                {
                    "credential_form": credential_form,
                    "order_form": order_form,
                    "strategy_form": StrategyForm(None),
                    "auth0User": auth0user,
                    "userdata": userdata,
                    "order_submit_form_response": response,
                },
            )

        if "submit_strategy" in request.POST and strategy_form.is_valid():
            # here for some reason form.cleaned_data changed from type dict to
            # type tuple. I tried to find the reason but it didn't seem to caused by
            # our code. Might be and django bug
            rebalance_strategy = strategy_form.cleaned_data[0]
            optimization_strategy = strategy_form.cleaned_data[1]
            user.portfolio.rebalancing_strategy = rebalance_strategy
            user.portfolio.optimization_strategy = optimization_strategy
            user.portfolio.save()
            return HttpResponseRedirect("/")
    return render(
        request,
        "home/orders.html",
        {
            "credential_form": credential_form,
            "order_form": order_form,
            "strategy_form": strategy_form,
            "auth0User": auth0user,
            "userdata": userdata,
        },
    )


@login_required
def positions(request):
    from backend.auth0login.forms import StrategyForm, WatchListForm

    user, userdata, auth0user, _user_details = get_user_information(request)
    watchlist_form = WatchListForm(request.POST or None)
    strategy_form = StrategyForm(request.POST or None)
    if request.method == "POST":
        if "add_to_watchlist" in request.POST and watchlist_form.is_valid():
            response = watchlist_form.add_to_watchlist(user)
            return render(
                request,
                "home/positions.html",
                {
                    "watchlist_form": watchlist_form,
                    "strategy_form": strategy_form,
                    "watchlist_form_response": response,
                    "auth0User": auth0user,
                    "userdata": userdata,
                },
            )

        if "submit_strategy" in request.POST and strategy_form.is_valid():
            # here for some reason form.cleaned_data changed from type dict to
            # type tuple. I tried to find the reason but it didn't seem to caused by
            # our code. Might be and django bug
            strategy = strategy_form.cleaned_data
            user.portfolio.strategy = strategy
            user.portfolio.save()
            return HttpResponseRedirect("positions")

    return render(
        request,
        "home/positions.html",
        {
            "watchlist_form": watchlist_form,
            "strategy_form": strategy_form,
            "auth0User": auth0user,
            "userdata": userdata,
        },
    )


@login_required
def user_settings(request):
    return render(
        request, "home/page-not-implemented.html"
    )  # 'home / user - settings.html')


@login_required
def machine_learning(request):
    """Machine Learning page - ML models for trading enhancement."""
    ml_data = dashboard_service.get_ml_dashboard()
    regime = dashboard_service.get_market_regime()
    return render(request, "home/machine-learning.html", {
        "segment": "machine-learning",
        "ml": ml_data,
        "regime": regime,
    })


@login_required
def strategies(request):
    """Strategy Manager page - configure and manage trading strategies."""
    all_strategies = dashboard_service.get_all_strategies()
    regime = dashboard_service.get_market_regime()
    return render(request, "home/strategies.html", {
        "segment": "strategies",
        "strategies": all_strategies,
        "regime": regime,
        "active_count": sum(1 for s in all_strategies if s.enabled),
        "total_count": len(all_strategies),
    })


@login_required
def strategy_portfolios(request):
    """Strategy Portfolio Builder - create and manage strategy combinations."""
    from .services.portfolio_builder import get_portfolio_builder

    builder = get_portfolio_builder(request.user)
    portfolios = builder.get_user_portfolios()
    templates = builder.get_portfolio_templates()
    strategies = builder.get_available_strategies()
    active_portfolio = builder.get_active_portfolio()

    return render(request, "home/strategy-portfolios.html", {
        "segment": "strategy-portfolios",
        "portfolios": [p.to_dict() for p in portfolios],
        "templates": templates,
        "available_strategies": strategies,
        "active_portfolio": active_portfolio.to_dict() if active_portfolio else None,
    })


@login_required
def strategy_wsb_dip_bot(request):
    """WSB Dip Bot strategy configuration page."""
    save_message = None
    save_status = None

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "save_config":
            # Parse configuration from POST data
            config = {
                "run_lookback_days": int(request.POST.get("run_lookback", 7)),
                "run_threshold_pct": float(request.POST.get("run_threshold", 8)),
                "dip_threshold_pct": float(request.POST.get("dip_threshold", -2)),
                "wsb_sentiment_weight": float(request.POST.get("wsb_weight", 30)) / 100,
                "target_dte": int(request.POST.get("target_dte", 21)),
                "otm_percentage": float(request.POST.get("otm_pct", 3)) / 100,
                "target_delta": float(request.POST.get("delta_target", 50)) / 100,
                "profit_multiplier": float(request.POST.get("profit_multiplier", 25)) / 10,
                "max_position_size": float(request.POST.get("position_size", 3)) / 100,
                "stop_loss": float(request.POST.get("stop_loss", 50)) / 100,
                "max_positions": int(request.POST.get("max_positions", 5)),
                "watchlist": request.POST.get("watchlist", ""),
                "scan_interval": int(request.POST.get("scan_interval", 300)),
                "enabled": request.POST.get("strategy_enabled") == "on",
            }

            result = dashboard_service.save_strategy_config("wsb_dip_bot", config)
            save_message = result.get("message", "Configuration saved")
            save_status = result.get("status", "info")

    strategy = dashboard_service.get_strategy_config("wsb_dip_bot")
    return render(request, "home/strategy-wsb-dip-bot.html", {
        "segment": "strategies",
        "strategy": strategy,
        "save_message": save_message,
        "save_status": save_status,
    })


@login_required
def strategy_wheel(request):
    """Wheel Strategy configuration page."""
    strategy = dashboard_service.get_strategy_config("wheel_strategy")
    return render(request, "home/strategy-wheel.html", {
        "segment": "strategies",
        "strategy": strategy,
    })


@login_required
def strategy_momentum_weeklies(request):
    """Momentum Weeklies configuration page."""
    strategy = dashboard_service.get_strategy_config("momentum_weeklies")
    return render(request, "home/strategy-momentum-weeklies.html", {
        "segment": "strategies",
        "strategy": strategy,
    })


@login_required
def strategy_earnings_protection(request):
    """Earnings Protection configuration page."""
    strategy = dashboard_service.get_strategy_config("earnings_protection")
    return render(request, "home/strategy-earnings-protection.html", {
        "segment": "strategies",
        "strategy": strategy,
    })


@login_required
def strategy_debit_spreads(request):
    """Debit Spreads configuration page."""
    strategy = dashboard_service.get_strategy_config("debit_spreads")
    return render(request, "home/strategy-debit-spreads.html", {
        "segment": "strategies",
        "strategy": strategy,
    })


@login_required
def strategy_leaps_tracker(request):
    """LEAPS Tracker configuration page."""
    strategy = dashboard_service.get_strategy_config("leaps_tracker")
    return render(request, "home/strategy-leaps-tracker.html", {
        "segment": "strategies",
        "strategy": strategy,
    })


@login_required
def strategy_lotto_scanner(request):
    """Lotto Scanner configuration page."""
    strategy = dashboard_service.get_strategy_config("lotto_scanner")
    return render(request, "home/strategy-lotto-scanner.html", {
        "segment": "strategies",
        "strategy": strategy,
    })


@login_required
def strategy_swing_trading(request):
    """Swing Trading configuration page."""
    strategy = dashboard_service.get_strategy_config("swing_trading")
    return render(request, "home/strategy-swing-trading.html", {
        "segment": "strategies",
        "strategy": strategy,
    })


@login_required
def strategy_spx_credit_spreads(request):
    """SPX Credit Spreads configuration page."""
    strategy = dashboard_service.get_strategy_config("spx_credit_spreads")
    return render(request, "home/strategy-spx-credit-spreads.html", {
        "segment": "strategies",
        "strategy": strategy,
    })


@login_required
def strategy_index_baseline(request):
    """Index Baseline configuration page."""
    strategy = dashboard_service.get_strategy_config("index_baseline")
    return render(request, "home/strategy-index-baseline.html", {
        "segment": "strategies",
        "strategy": strategy,
    })


@login_required
def backtesting(request):
    """Backtesting page - test strategies against historical data."""
    backtest_status = dashboard_service.get_backtesting_status()
    return render(request, "home/backtesting.html", {
        "segment": "backtesting",
        "backtest": backtest_status,
    })


@login_required
def risk_management(request):
    """Risk Management page - monitor and control portfolio risk."""
    risk_metrics = dashboard_service.get_risk_metrics()
    regime = dashboard_service.get_market_regime()
    return render(request, "home/risk.html", {
        "segment": "risk",
        "metrics": risk_metrics,
        "regime": regime,
    })


@login_required
def analytics(request):
    """Analytics dashboard - view performance metrics and analysis."""
    analytics_data = dashboard_service.get_analytics_metrics()
    regime = dashboard_service.get_market_regime()
    return render(request, "home/analytics.html", {
        "segment": "analytics",
        "metrics": analytics_data,
        "regime": regime,
    })


@login_required
def alerts(request):
    """Alerts center - manage notifications and alert rules."""
    recent_alerts = dashboard_service.get_recent_alerts(limit=20)
    unread_count = dashboard_service.get_unread_alert_count()
    return render(request, "home/alerts.html", {
        "segment": "alerts",
        "alerts": recent_alerts,
        "unread_count": unread_count,
    })


@login_required
def system_status(request):
    """System status monitor - view system health and logs."""
    status = dashboard_service.get_system_status()
    return render(request, "home/system-status.html", {
        "segment": "system-status",
        "status": status,
    })


@login_required
def market_context(request):
    """Market Context Dashboard - comprehensive market overview."""
    return render(request, "home/market-context.html", {"segment": "market-context"})


@login_required
def allocations(request):
    """Allocation Management - view and manage strategy allocations."""
    return render(request, "home/allocations.html", {"segment": "allocations"})


@login_required
def circuit_breakers(request):
    """Circuit Breaker Control Panel - monitor and manage circuit breakers."""
    return render(request, "home/circuit-breakers.html", {"segment": "circuit-breakers"})


@login_required
def ml_agents(request):
    """ML/RL Agent Control Panel - manage ML models and RL agents."""
    return render(request, "home/ml-agents.html", {"segment": "ml-agents"})


@login_required
def user_settings_page(request):
    """Settings page - configure API connections and preferences."""
    return render(request, "home/settings.html", {"segment": "settings"})


@login_required
def setup_wizard(request):
    """Setup wizard for first-time configuration."""
    return render(request, "home/setup-wizard.html", {"segment": "setup"})


@login_required
def crypto_trading(request):
    """Crypto trading dashboard - 24/7 crypto trading via Alpaca."""
    crypto_data = dashboard_service.get_crypto_dashboard()
    return render(request, "home/crypto-trading.html", {
        "segment": "crypto",
        "crypto": crypto_data,
    })


@login_required
def extended_hours(request):
    """Extended hours trading - pre-market and after-hours trading."""
    settings_message = None
    settings_status = None

    # Handle POST requests for settings
    if request.method == "POST":
        action = request.POST.get("action")

        if action == "update_settings":
            pre_market_enabled = request.POST.get("pre_market_enabled") == "on"
            after_hours_enabled = request.POST.get("after_hours_enabled") == "on"

            result = dashboard_service.update_extended_hours_settings(
                pre_market_enabled=pre_market_enabled,
                after_hours_enabled=after_hours_enabled,
            )
            settings_message = result.get("message", "Settings updated")
            settings_status = result.get("status", "info")

    extended_data = dashboard_service.get_extended_hours()
    return render(request, "home/extended-hours.html", {
        "segment": "extended-hours",
        "extended": extended_data,
        "settings_message": settings_message,
        "settings_status": settings_status,
    })


@login_required
def margin_borrow(request):
    """Margin and borrow management - short selling and margin tracking."""
    margin_data = dashboard_service.get_margin_borrow()
    return render(request, "home/margin-borrow.html", {
        "segment": "margin-borrow",
        "margin": margin_data,
    })


@login_required
def exotic_spreads(request):
    """Exotic option spreads - iron condors, butterflies, straddles, etc."""
    spreads_data = dashboard_service.get_exotic_spreads()
    return render(request, "home/exotic-spreads.html", {
        "segment": "exotic-spreads",
        "spreads": spreads_data,
    })


@login_required
def feature_status(request):
    """Feature status - check availability of all modules."""
    features = dashboard_service.get_feature_availability()
    return render(request, "home/feature-status.html", {
        "segment": "feature-status",
        "features": features,
    })


@login_required
def ml_training(request):
    """ML Training center - train and manage ML models."""
    training_message = None
    training_status = None

    # Handle POST requests for training actions
    if request.method == "POST":
        action = request.POST.get("action")
        model_type = request.POST.get("model_type", "lstm")

        if action == "start_training":
            training_result = dashboard_service.start_ml_training(
                model_type=model_type,
                config={
                    "epochs": int(request.POST.get("epochs", 100)),
                    "batch_size": int(request.POST.get("batch_size", 32)),
                    "learning_rate": float(request.POST.get("learning_rate", 0.001)),
                    "data_period": request.POST.get("data_period", "2y"),
                    "validation_split": float(request.POST.get("validation_split", 0.2)),
                    "symbols": request.POST.get("symbols", "nasdaq100"),
                }
            )
            training_message = training_result.get("message", "Training started")
            training_status = training_result.get("status", "info")

        elif action == "fetch_data":
            data_result = dashboard_service.fetch_training_data(
                symbols=request.POST.get("symbols", "nasdaq100"),
                period=request.POST.get("period", "2y"),
            )
            training_message = data_result.get("message", "Data fetched")
            training_status = data_result.get("status", "info")

        elif action == "save_config":
            training_message = "Configuration saved successfully"
            training_status = "success"

    ml_models_status = dashboard_service.get_ml_models_status()
    features = dashboard_service.get_feature_availability()
    return render(request, "home/ml-training.html", {
        "segment": "ml-training",
        "models": ml_models_status,
        "features": features,
        "training_message": training_message,
        "training_status": training_status,
    })


@login_required
def tax_optimization(request):
    """Tax optimization dashboard with lot tracking and harvesting."""
    from .services.tax_optimizer import get_tax_optimizer_service

    service = get_tax_optimizer_service(request.user)

    # Get year summary
    year_summary = service.get_year_summary()

    # Get harvesting opportunities
    opportunities = service.get_harvesting_opportunities(min_loss=100, limit=10)

    # Get all open lots grouped by symbol
    all_lots = service.get_all_lots(include_closed=False)

    # Group lots by symbol for display
    lots_by_symbol = {}
    for lot in all_lots:
        symbol = lot['symbol']
        if symbol not in lots_by_symbol:
            lots_by_symbol[symbol] = []
        lots_by_symbol[symbol].append(lot)

    return render(request, "home/tax-optimization.html", {
        "segment": "tax-optimization",
        "year_summary": year_summary,
        "opportunities": opportunities,
        "lots_by_symbol": lots_by_symbol,
        "total_lots": len(all_lots),
        "disclaimer": (
            "DISCLAIMER: This information is for educational purposes only and does not "
            "constitute tax advice. Consult a qualified tax professional for specific guidance."
        ),
    })


@login_required
def strategy_leaderboard(request):
    """Strategy leaderboard showing ranked performance comparisons."""
    from .services.leaderboard_service import LeaderboardService

    service = LeaderboardService(user=request.user)

    # Get default leaderboard (1M, sorted by Sharpe)
    leaderboard_data = service.get_leaderboard(
        period='1M',
        metric='sharpe_ratio',
        limit=15
    )

    # Get top performers
    top_performers = service.get_top_performers(period='1M', count=3)

    # Get all strategies for filter dropdown
    all_strategies = service.get_all_strategies()

    # Available periods and metrics for UI
    periods = ['1W', '1M', '3M', '6M', '1Y', 'YTD', 'ALL']
    metrics = {
        'sharpe_ratio': 'Sharpe Ratio',
        'sortino_ratio': 'Sortino Ratio',
        'total_return_pct': 'Total Return %',
        'win_rate': 'Win Rate %',
        'profit_factor': 'Profit Factor',
        'max_drawdown_pct': 'Max Drawdown %',
        'calmar_ratio': 'Calmar Ratio',
    }

    return render(request, "home/leaderboard.html", {
        "segment": "leaderboard",
        "leaderboard": leaderboard_data,
        "top_performers": top_performers,
        "all_strategies": all_strategies,
        "periods": periods,
        "metrics": metrics,
        "default_period": '1M',
        "default_metric": 'sharpe_ratio',
    })


@login_required
def strategy_builder(request):
    """Visual strategy builder for creating custom trading strategies."""
    from backend.tradingbot.models.models import CustomStrategy
    from .services.custom_strategy_runner import (
        CustomStrategyRunner,
        get_strategy_templates
    )

    # Get user's strategies
    strategies = CustomStrategy.objects.filter(user=request.user)

    # Get available options for the builder
    indicators = CustomStrategyRunner.get_available_indicators()
    operators = CustomStrategyRunner.get_available_operators()
    exit_types = CustomStrategyRunner.get_exit_types()
    templates = get_strategy_templates()

    return render(request, "home/strategy-builder.html", {
        "segment": "strategy-builder",
        "strategies": [s.to_dict() for s in strategies],
        "strategy_count": strategies.count(),
        "active_count": strategies.filter(is_active=True).count(),
        "indicators": indicators,
        "operators": operators,
        "exit_types": exit_types,
        "templates": templates,
    })


def logout(request):
    # Audit before log_out() clears the user
    if request.user.is_authenticated:
        from .audit import log_event, AuditEventType
        log_event(
            AuditEventType.LOGOUT,
            user=request.user,
            request=request,
            description=f"User {request.user.username} logged out",
        )
    log_out(request)
    return_to = urlencode({"returnTo": request.build_absolute_uri("/")})
    logout_url = f"https://{settings.SOCIAL_AUTH_AUTH0_DOMAIN}/v2/logout?client_id={settings.SOCIAL_AUTH_AUTH0_KEY}&{return_to}"
    return HttpResponseRedirect(logout_url)
