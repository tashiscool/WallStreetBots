"""Synchronization between local database and Alpaca broker.

Uses modern alpaca-py SDK via AlpacaManager.
"""
from datetime import timedelta


def validate_backend():
    from backend.settings import BACKEND_ALPACA_ID, BACKEND_ALPACA_KEY
    from backend.tradingbot.apimanagers import AlpacaManager

    backendapi = AlpacaManager(BACKEND_ALPACA_ID, BACKEND_ALPACA_KEY)
    if not backendapi.validate_api()[0]:
        from django.core.exceptions import ValidationError
        raise ValidationError(backendapi.validate_api()[1])
    return backendapi


def sync_database_company_stock(ticker):
    """Check if company/stock for this ticker exist and sync to database if not already exists.

    Returns the stock and company.
    """
    from backend.tradingbot.models.models import Company, Stock

    if not Company.objects.filter(ticker=ticker).exists():
        company = Company(name=ticker, ticker=ticker)
        company.save()
        stock = Stock(company=company)
        stock.save()
    else:
        company = Company.objects.get(ticker=ticker)
        stock, _ = Stock.objects.get_or_create(company=company)
    return stock, company


def sync_alpaca(user):
    """Sync user related database data with Alpaca.

    Uses the modern alpaca-py SDK via AlpacaManager.
    """
    user_details = {}

    if not hasattr(user, "credential"):
        return user_details

    from backend.tradingbot.apimanagers import AlpacaManager

    api = AlpacaManager(user.credential.alpaca_id, user.credential.alpaca_key)

    if not api.validate_api()[0]:
        print(api.validate_api()[1])
        return user_details

    # Get account information via AlpacaManager
    account = api.get_account()
    if not account:
        print("Failed to get account information")
        return user_details

    user_details["equity"] = str(round(float(account.equity), 2))
    user_details["buy_power"] = str(round(float(account.buying_power), 2))
    user_details["cash"] = str(round(float(account.cash), 2))
    user_details["currency"] = account.currency
    user_details["long_portfolio_value"] = str(
        round(float(account.long_market_value), 2)
    )
    user_details["short_portfolio_value"] = str(
        round(float(account.short_market_value), 2)
    )

    portfolio_value = float(account.portfolio_value)
    last_equity = float(account.last_equity)
    if last_equity != 0:
        pct_change = round((portfolio_value - last_equity) / last_equity, 2)
    else:
        pct_change = 0.0

    user_details["portfolio_percent_change"] = str(pct_change)
    user_details["portfolio_dollar_change"] = str(round(portfolio_value - last_equity))

    if portfolio_value >= last_equity:
        user_details["portfolio_change_direction"] = "positive"
    else:
        user_details["portfolio_change_direction"] = "negative"

    # Get portfolio positions (returned as list of dicts from AlpacaManager)
    positions = api.get_positions()

    # Sync companies/stocks to database
    for position in positions:
        sync_database_company_stock(position["symbol"])

    from backend.tradingbot.models.models import Company, Order, Portfolio, Stock

    if not hasattr(user, "portfolio"):
        new_portfolio = Portfolio(name="default-1", user=user)
        new_portfolio.save()

    # Sync order statuses using AlpacaManager
    alpaca_open_orders = api.get_orders(status="open")

    local_open_orders = Order.objects.filter(user=user, status__in=["pending", "accepted", "new"])
    for order in local_open_orders.iterator():
        if not order.alpaca_order_id:
            continue
        # Find matching Alpaca order
        matched = False
        for alpaca_order in alpaca_open_orders:
            if str(alpaca_order.get("id")) == order.alpaca_order_id:
                alpaca_status = str(alpaca_order.get("status", "")).lower()
                if alpaca_status == "filled":
                    order.status = "filled"
                    if alpaca_order.get("filled_price"):
                        order.filled_avg_price = alpaca_order["filled_price"]
                elif alpaca_status in ("cancelled", "canceled", "expired"):
                    order.status = "cancelled"
                elif alpaca_status in ("accepted", "new", "pending_new"):
                    order.status = "accepted"
                order.save()
                matched = True
                break
        if not matched:
            # Order no longer in Alpaca open orders — check if it was filled or cancelled
            order.status = "cancelled"
            order.save()

    # Calculate usable cash
    usable_cash = float(account.cash)
    for alpaca_order in alpaca_open_orders:
        if alpaca_order.get("side") == "buy":
            success, price = api.get_price(alpaca_order["symbol"])
            if success:
                usable_cash -= price * alpaca_order.get("qty", 0)

    user_details["usable_cash"] = str(round(usable_cash, 2))
    user_details["portfolio"] = positions
    user_details["display_portfolio"] = list(positions)
    user_details["orders"] = list(
        Order.objects.filter(user=user)
        .order_by("-date")
        .values("id", "symbol", "side", "quantity", "price", "status", "order_type", "date")[:50]
    )

    # Update portfolio cash
    if not hasattr(user, "portfolio"):
        port = Portfolio(user=user, name="default-1")
        port.save()

    user_details["strategy"] = user.portfolio.get_strategy_display() if hasattr(user, "portfolio") else "manual"

    return user_details
