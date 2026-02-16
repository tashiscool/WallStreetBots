from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from rest_framework import status

from .apimanagers import AlpacaManager
from .models import Order, Stock, Company


def index(request):
    return HttpResponse("Hello World, welcome to tradingbot!")


def _get_or_create_stock(ticker):
    """Get or create a Stock record for the given ticker symbol."""
    company, _ = Company.objects.get_or_create(
        ticker=ticker.upper(),
        defaults={"name": ticker.upper()},
    )
    stock, _ = Stock.objects.get_or_create(company=company)
    return stock


@login_required
def stock_trade(request):
    """Handle stock buy/sell orders with market, limit, and stop order types."""
    if request.method != "POST":
        return HttpResponse("Method not allowed", status=status.HTTP_405_METHOD_NOT_ALLOWED)

    user = request.user
    transaction_side = request.POST.get("transaction_side")  # buy or sell
    order_type = request.POST.get("order_type", "market")  # market, limit, stop
    ticker = request.POST.get("ticker")
    quantity_str = request.POST.get("quantity")

    if not ticker or not quantity_str or not transaction_side:
        return HttpResponse(
            "ticker, quantity, and transaction_side are required",
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        quantity = int(quantity_str)
        if quantity <= 0:
            raise ValueError
    except (ValueError, TypeError):
        return HttpResponse("Invalid quantity", status=status.HTTP_400_BAD_REQUEST)

    limit_price = request.POST.get("limit_price")
    stop_price = request.POST.get("stop_price")

    # Parse prices if provided
    try:
        limit_price = float(limit_price) if limit_price else None
        stop_price = float(stop_price) if stop_price else None
    except (ValueError, TypeError):
        return HttpResponse("Invalid price value", status=status.HTTP_400_BAD_REQUEST)

    # Validate inputs
    if transaction_side not in ("buy", "sell"):
        return HttpResponse(
            f"Invalid transaction_side: {transaction_side}",
            status=status.HTTP_400_BAD_REQUEST,
        )

    if order_type not in ("market", "limit", "stop", "stop_limit", "trailing_stop"):
        return HttpResponse(
            f"Invalid order_type: {order_type}",
            status=status.HTTP_400_BAD_REQUEST,
        )

    alpaca_api = AlpacaManager(user.credential.alpaca_id, user.credential.alpaca_key)

    # Get or create stock record
    stock = _get_or_create_stock(ticker)

    # Execute order based on type and side
    result = None
    if transaction_side == "buy":
        if order_type == "market":
            result = alpaca_api.market_buy(ticker, quantity)
        elif order_type == "limit":
            if not limit_price:
                return HttpResponse(
                    "Limit price required for limit orders",
                    status=status.HTTP_400_BAD_REQUEST,
                )
            result = alpaca_api.market_buy(
                ticker, quantity, order_type="limit", limit_price=limit_price
            )
        elif order_type == "stop":
            if not stop_price:
                return HttpResponse(
                    "Stop price required for stop orders",
                    status=status.HTTP_400_BAD_REQUEST,
                )
            result = alpaca_api.place_stop_loss(ticker, quantity, stop_price)
        else:
            return HttpResponse(status=status.HTTP_501_NOT_IMPLEMENTED)

    elif transaction_side == "sell":
        if order_type == "market":
            result = alpaca_api.market_sell(ticker, quantity)
        elif order_type == "limit":
            if not limit_price:
                return HttpResponse(
                    "Limit price required for limit orders",
                    status=status.HTTP_400_BAD_REQUEST,
                )
            result = alpaca_api.market_sell(
                ticker, quantity, order_type="limit", limit_price=limit_price
            )
        elif order_type == "stop":
            if not stop_price:
                return HttpResponse(
                    "Stop price required for stop orders",
                    status=status.HTTP_400_BAD_REQUEST,
                )
            result = alpaca_api.place_stop_loss(ticker, quantity, stop_price)
        else:
            return HttpResponse(status=status.HTTP_501_NOT_IMPLEMENTED)

    # Check result
    if not result or "error" in result:
        error_msg = result.get("error", "Unknown error") if result else "Order failed"
        return HttpResponse(error_msg, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Save order to database with correct field names
    order = Order(
        user=user,
        stock=stock,
        symbol=ticker.upper(),
        quantity=quantity,
        side=transaction_side,
        order_type=order_type,
        status="accepted",
        alpaca_order_id=str(result.get("id", "")),
        limit_price=limit_price,
        stop_price=stop_price,
    )
    order.save()

    return HttpResponse(status=status.HTTP_201_CREATED)


@login_required
def close_position(request):
    """Close a position (full or partial).

    POST parameters:
        symbol: Stock symbol to close
        percentage: Optional percentage to close (0.0-1.0). Default 1.0 (full close)
    """
    if request.method != "POST":
        return HttpResponse(
            "Method not allowed",
            status=status.HTTP_405_METHOD_NOT_ALLOWED,
        )

    user = request.user
    symbol = request.POST.get("symbol")
    percentage = request.POST.get("percentage", "1.0")

    if not symbol:
        return HttpResponse(
            "Symbol is required",
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Parse percentage
    try:
        percentage = float(percentage)
        if percentage <= 0 or percentage > 1:
            return HttpResponse(
                "Percentage must be between 0 and 1",
                status=status.HTTP_400_BAD_REQUEST,
            )
    except ValueError:
        return HttpResponse(
            "Invalid percentage value",
            status=status.HTTP_400_BAD_REQUEST,
        )

    alpaca_api = AlpacaManager(user.credential.alpaca_id, user.credential.alpaca_key)

    # Close the position
    success = alpaca_api.close_position(symbol.upper(), percentage)

    if success:
        return HttpResponse(status=status.HTTP_200_OK)
    else:
        return HttpResponse(
            f"Failed to close position for {symbol}",
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@login_required
def get_position_details(request, symbol):
    """Get details for a specific position."""
    user = request.user
    alpaca_api = AlpacaManager(user.credential.alpaca_id, user.credential.alpaca_key)

    positions = alpaca_api.get_positions()

    for pos in positions:
        if pos["symbol"].upper() == symbol.upper():
            return JsonResponse(pos)

    return HttpResponse(
        f"Position not found: {symbol}",
        status=status.HTTP_404_NOT_FOUND,
    )
