from django.core.exceptions import ValidationError

from backend.tradingbot.apimanagers import AlpacaManager
from backend.tradingbot.synchronization import (
    sync_database_company_stock,
    validate_backend,
)


def create_local_order(
    user, ticker, quantity, order_type, transaction_type, status, alpaca_order_id="",
    limit_price=None, stop_price=None
):
    """Create a local Order record in the database."""
    # Map transaction type to side
    if transaction_type in ("buy", "B"):
        side = "buy"
    elif transaction_type in ("sell", "S"):
        side = "sell"
    else:
        raise ValidationError(f"invalid transaction type: {transaction_type}")

    # Normalize order type
    order_type_map = {
        "market": "market",
        "limit": "limit",
        "stop": "stop",
        "stop_limit": "stop_limit",
        "trailing_stop": "trailing_stop",
        "M": "market",
        "L": "limit",
        "S": "stop",
        "ST": "stop_limit",
        "T": "trailing_stop",
    }
    normalized_order_type = order_type_map.get(order_type)
    if not normalized_order_type:
        raise ValidationError(f"invalid order type: {order_type}")

    # Normalize status
    status_map = {
        "A": "accepted",
        "N": "new",
        "F": "filled",
        "C": "cancelled",
        "pending": "pending",
        "accepted": "accepted",
        "new": "new",
        "filled": "filled",
        "cancelled": "cancelled",
    }
    normalized_status = status_map.get(status, "pending")

    stock, _ = sync_database_company_stock(ticker)
    from backend.tradingbot.models.models import Order

    order = Order(
        user=user,
        stock=stock,
        symbol=ticker.upper(),
        side=side,
        order_type=normalized_order_type,
        quantity=quantity,
        status=normalized_status,
        alpaca_order_id=alpaca_order_id,
        limit_price=limit_price,
        stop_price=stop_price,
    )
    order.save()
    return order


def place_general_order(
    user, user_details, ticker, quantity, transaction_type, order_type, time_in_force,
    limit_price=None, stop_price=None
):
    """General place order function with database, margin, and Alpaca synchronization.

    Args:
        user: request.user
        user_details: return from sync_alpaca function
        ticker: Stock symbol
        quantity: Number of shares
        transaction_type: 'buy' or 'sell'
        order_type: 'market', 'limit', or 'stop'
        time_in_force: Time in force (e.g., 'gtc', 'day')
        limit_price: Required for limit orders
        stop_price: Required for stop orders
    """
    backend_api = validate_backend()
    user_api = AlpacaManager(user.credential.alpaca_id, user.credential.alpaca_key)

    # 1. Check if ticker exists and validate price
    check, price = backend_api.get_price(ticker)
    if not check:
        raise ValidationError(
            f"Failed to get price for {ticker}, are you sure that the ticker name is correct?"
        )

    # Validate order based on type
    if transaction_type == "buy":
        buy_order_check(
            order_type=order_type,
            price=price,
            quantity=quantity,
            usable_cash=user_details["usable_cash"],
            limit_price=limit_price,
        )

    # 2. Store order to database
    stock, _ = sync_database_company_stock(ticker)
    from backend.tradingbot.models.models import Order

    order = Order(
        user=user,
        stock=stock,
        symbol=ticker.upper(),
        side=transaction_type,
        order_type=order_type,
        quantity=quantity,
        status="pending",
        limit_price=limit_price,
        stop_price=stop_price,
    )
    order.save()

    # 3. Place order to Alpaca using AlpacaManager methods
    try:
        result = None
        if transaction_type == "buy":
            if order_type == "market":
                result = user_api.market_buy(ticker, int(quantity))
            elif order_type == "limit":
                if not limit_price:
                    raise ValidationError("Limit price required for limit orders")
                result = user_api.market_buy(
                    ticker, int(quantity), order_type="limit", limit_price=float(limit_price)
                )
            elif order_type == "stop":
                if not stop_price:
                    raise ValidationError("Stop price required for stop orders")
                result = user_api.place_stop_loss(ticker, int(quantity), float(stop_price))
        elif transaction_type == "sell":
            if order_type == "market":
                result = user_api.market_sell(ticker, int(quantity))
            elif order_type == "limit":
                if not limit_price:
                    raise ValidationError("Limit price required for limit orders")
                result = user_api.market_sell(
                    ticker, int(quantity), order_type="limit", limit_price=float(limit_price)
                )
            elif order_type == "stop":
                if not stop_price:
                    raise ValidationError("Stop price required for stop orders")
                result = user_api.place_stop_loss(ticker, int(quantity), float(stop_price))

        if not result or "error" in result:
            error_msg = result.get("error", "Unknown error") if result else "Order failed"
            raise ValidationError(error_msg)

        # Update order with Alpaca response
        order.alpaca_order_id = str(result.get("id", ""))
        order.status = "accepted"
        order.save()

    except ValidationError:
        order.delete()
        raise
    except Exception as e:
        order.delete()
        raise ValidationError(str(e)) from e

    return True


def add_stock_to_database(user, ticker):
    """Add a stock to the user's watchlist."""
    ticker = ticker.upper()
    backend_api = validate_backend()
    check, _price = backend_api.get_price(ticker)
    if not check:
        raise ValidationError(
            f"Failed to get price for {ticker}, are you sure that the ticker name is correct?"
        )
    sync_database_company_stock(ticker)


def buy_order_check(order_type, price, quantity, usable_cash, limit_price=None):
    """Validate a buy order before execution."""
    if order_type in ("market", "M"):
        if float(price) * float(quantity) > float(usable_cash):
            raise ValidationError(
                "Not enough cash to perform this operation. Marginal trading is not supported."
            )
    elif order_type in ("limit", "L"):
        check_price = limit_price if limit_price else price
        if float(check_price) * float(quantity) > float(usable_cash):
            raise ValidationError(
                "Not enough cash to perform this operation. Marginal trading is not supported."
            )
    elif order_type in ("stop", "S"):
        if float(price) * float(quantity) > float(usable_cash):
            raise ValidationError(
                "Not enough cash to perform this operation. Marginal trading is not supported."
            )


def sell_order_check(order_type, price, quantity, usable_cash):
    """Validate a sell order before execution.

    Sell orders don't require cash validation.
    The broker will verify the user has the shares to sell.
    """
    pass
