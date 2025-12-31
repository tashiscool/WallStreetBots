from django.core.exceptions import ValidationError

from backend.tradingbot.apimanagers import AlpacaManager
from backend.tradingbot.synchronization import (
    sync_database_company_stock,
    sync_stock_instance,
    validate_backend,
)


def create_local_order(
    user, ticker, quantity, order_type, transaction_type, status, client_order_id="",
    limit_price=None, stop_price=None
):
    # Map transaction type
    if transaction_type == "buy":
        transaction_type = "B"
    elif transaction_type == "sell":
        transaction_type = "S"
    else:
        raise ValidationError("invalid transaction type")

    # Map order type
    order_type_map = {
        "market": "M",
        "limit": "L",
        "stop": "S",
        "stop_limit": "ST",
        "trailing_stop": "T",
    }
    if order_type not in order_type_map:
        raise ValidationError(f"invalid order type: {order_type}")
    order_type = order_type_map[order_type]

    stock, _ = sync_database_company_stock(ticker)
    from backend.tradingbot.models.models import Order

    order = Order(
        user=user,
        stock=stock,
        order_type=order_type,
        quantity=quantity,
        transaction_type=transaction_type,
        status=status,
        client_order_id=client_order_id,
        limit_price=limit_price,
        stop_price=stop_price,
    )
    order.save()


def place_general_order(
    user, user_details, ticker, quantity, transaction_type, order_type, time_in_force,
    limit_price=None, stop_price=None
):
    """General place order function that takes account of database, margin, and alpaca synchronization.

    supports market, limit, and stop buy/sell orders
    user: request.user
    user_details: return from sync_alpaca function
    order_type: 'market', 'limit', or 'stop'
    transaction_type: 'buy' or 'sell'
    limit_price: required for limit orders
    stop_price: required for stop orders
    """
    backend_api = validate_backend()
    user_api = AlpacaManager(user.credential.alpaca_id, user.credential.alpaca_key)

    # Map order type for database storage
    order_type_map = {
        "market": "M",
        "limit": "L",
        "stop": "S",
    }
    db_order_type = order_type_map.get(order_type)
    if not db_order_type:
        raise ValidationError(f"invalid order type: {order_type}")

    # Map transaction type for database storage
    transaction_type_map = {
        "buy": "B",
        "sell": "S",
    }
    db_transaction_type = transaction_type_map.get(transaction_type)
    if not db_transaction_type:
        raise ValidationError(f"invalid transaction type: {transaction_type}")

    # 1. check if ticker exists and check buy / sell availability and errors
    check, price = backend_api.get_price(ticker)
    if not check:
        raise ValidationError(
            f"Failed to get price for {ticker}, are you sure that the ticker name is correct?"
        )

    # Validate order based on type
    if transaction_type == "buy":
        buy_order_check(
            order_type=db_order_type,
            price=price,
            quantity=quantity,
            usable_cash=user_details["usable_cash"],
            limit_price=limit_price,
        )
    elif transaction_type == "sell":
        sell_order_check(
            order_type=db_order_type,
            price=price,
            quantity=quantity,
            usable_cash=user_details["usable_cash"],
        )

    # 2. store order to database
    # 2.1 check if stock and company exists
    stock, _ = sync_database_company_stock(ticker)
    from backend.tradingbot.models.models import Order

    order = Order(
        user=user,
        stock=stock,
        order_type=db_order_type,
        quantity=quantity,
        transaction_type=db_transaction_type,
        status="A",
        limit_price=limit_price,
        stop_price=stop_price,
    )
    order.save()
    client_order_id = order.order_number

    # 3. place order to Alpaca based on order type
    try:
        order_params = {
            "symbol": ticker,
            "qty": float(quantity),
            "side": transaction_type,
            "time_in_force": time_in_force,
            "client_order_id": str(client_order_id),
        }

        if order_type == "market":
            order_params["type"] = "market"
        elif order_type == "limit":
            if not limit_price:
                raise ValidationError("Limit price required for limit orders")
            order_params["type"] = "limit"
            order_params["limit_price"] = float(limit_price)
        elif order_type == "stop":
            if not stop_price:
                raise ValidationError("Stop price required for stop orders")
            order_params["type"] = "stop"
            order_params["stop_price"] = float(stop_price)

        user_api.api.submit_order(**order_params)
    except Exception as e:
        # 4. delete order if not valid.
        order.delete()
        raise ValidationError(e) from e

    return True


def add_stock_to_database(user, ticker):
    # 1. check if ticker exists
    ticker = ticker.upper()
    backend_api = validate_backend()
    check, _price = backend_api.get_price(ticker)
    if not check:
        raise ValidationError(
            f"Failed to get price for {ticker}, are you sure that the ticker name is correct?"
        )
    stock, _ = sync_database_company_stock(ticker)
    sync_stock_instance(user, user.portfolio, stock)


def buy_order_check(order_type, price, quantity, usable_cash, limit_price=None):
    """Validate a buy order before execution.

    Args:
        order_type: 'M' (market), 'L' (limit), 'S' (stop)
        price: Current market price
        quantity: Number of shares
        usable_cash: Available cash for trading
        limit_price: Limit price for limit orders

    Raises:
        ValidationError: If order validation fails
    """
    if order_type == "M":
        # For market orders, check if user has enough cash
        if float(price) * float(quantity) > float(usable_cash):
            raise ValidationError(
                "Not enough cash to perform this operation. Marginal trading is not supported."
            )
    elif order_type == "L":
        # For limit orders, use limit price for cash check
        check_price = limit_price if limit_price else price
        if float(check_price) * float(quantity) > float(usable_cash):
            raise ValidationError(
                "Not enough cash to perform this operation. Marginal trading is not supported."
            )
    elif order_type == "S":
        # For stop orders, check using current market price
        if float(price) * float(quantity) > float(usable_cash):
            raise ValidationError(
                "Not enough cash to perform this operation. Marginal trading is not supported."
            )


def sell_order_check(order_type, price, quantity, usable_cash):
    """Validate a sell order before execution.

    Args:
        order_type: 'M' (market), 'L' (limit), 'S' (stop)
        price: Current market price
        quantity: Number of shares
        usable_cash: Available cash (unused for sell orders)

    Note: For sell orders, we don't need cash validation.
          Position validation should happen at the Alpaca API level.
    """
    # Sell orders don't require cash validation
    # The broker will verify the user has the shares to sell
    pass
