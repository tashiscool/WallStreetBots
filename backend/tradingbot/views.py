from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from rest_framework import status
from decimal import Decimal, InvalidOperation
import json
import re

from .apimanagers import AlpacaManager
from .models import Order, Stock, Company, TradeTransaction

_TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.\-]{0,19}$")


def _normalize_order_status(raw_status):
    """Map broker statuses to supported local Order/TradeTransaction statuses."""
    normalized = str(raw_status or "").strip().lower()
    status_map = {
        "accepted_for_bidding": "accepted",
        "partially_filled": "partially_filled",
        "partially-filled": "partially_filled",
        "canceled": "cancelled",
    }
    normalized = status_map.get(normalized, normalized)
    allowed = {
        "pending",
        "accepted",
        "new",
        "partially_filled",
        "filled",
        "cancelled",
        "rejected",
    }
    return normalized if normalized in allowed else "accepted"


def _extract_transaction_price(result, limit_price, stop_price):
    """Resolve the best available transaction price from broker payload and request."""
    candidates = [
        result.get("filled_price"),
        result.get("limit_price"),
        limit_price,
        stop_price,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            value = Decimal(str(candidate))
        except (InvalidOperation, TypeError, ValueError):
            continue
        if value > 0:
            return value
    return Decimal("0")


def _is_user_paper_trading(user) -> bool:
    """Resolve whether this user should trade in paper mode."""
    try:
        from backend.tradingbot.models.models import UserProfile

        profile = UserProfile.objects.filter(user=user).only("is_paper_trading").first()
        if profile is not None:
            return bool(profile.is_paper_trading)
    except Exception:
        pass

    try:
        from backend.auth0login.models import WizardConfiguration

        config = WizardConfiguration.objects.filter(user=user).only("trading_mode").first()
        if config is not None:
            return config.trading_mode != "live"
    except Exception:
        pass

    return True


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


def _get_request_payload(request):
    """Parse JSON payloads for API requests, falling back to form data."""
    if request.content_type and "application/json" in request.content_type:
        try:
            body = request.body.decode("utf-8") if request.body else "{}"
            payload = json.loads(body)
            if isinstance(payload, dict):
                return payload, None
            return None, "JSON payload must be an object"
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, "Invalid JSON payload"
    return request.POST.dict(), None


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

    alpaca_api = AlpacaManager(
        user.credential.alpaca_id,
        user.credential.alpaca_key,
        paper_trading=_is_user_paper_trading(user),
    )

    # Get or create stock record
    stock = _get_or_create_stock(ticker)

    # Execute order based on type and side
    result = None
    if transaction_side == "buy":
        if order_type == "market":
            result = alpaca_api.market_buy(ticker, quantity, user=user)
        elif order_type == "limit":
            if not limit_price:
                return HttpResponse(
                    "Limit price required for limit orders",
                    status=status.HTTP_400_BAD_REQUEST,
                )
            result = alpaca_api.market_buy(
                ticker, quantity, order_type="limit", limit_price=limit_price, user=user
            )
        elif order_type == "stop":
            if not stop_price:
                return HttpResponse(
                    "Stop price required for stop orders",
                    status=status.HTTP_400_BAD_REQUEST,
                )
            result = alpaca_api.place_stop_loss(ticker, quantity, stop_price, user=user)
        else:
            return HttpResponse(status=status.HTTP_501_NOT_IMPLEMENTED)

    elif transaction_side == "sell":
        if order_type == "market":
            result = alpaca_api.market_sell(ticker, quantity, user=user)
        elif order_type == "limit":
            if not limit_price:
                return HttpResponse(
                    "Limit price required for limit orders",
                    status=status.HTTP_400_BAD_REQUEST,
                )
            result = alpaca_api.market_sell(
                ticker, quantity, order_type="limit", limit_price=limit_price, user=user
            )
        elif order_type == "stop":
            if not stop_price:
                return HttpResponse(
                    "Stop price required for stop orders",
                    status=status.HTTP_400_BAD_REQUEST,
                )
            result = alpaca_api.place_stop_loss(ticker, quantity, stop_price, user=user)
        else:
            return HttpResponse(status=status.HTTP_501_NOT_IMPLEMENTED)

    # Check result
    if not result or "error" in result:
        error_msg = result.get("error", "Unknown error") if result else "Order failed"
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        if result and result.get("live_trading_blocked"):
            status_code = status.HTTP_403_FORBIDDEN
        elif result and result.get("allocation_exceeded"):
            status_code = status.HTTP_400_BAD_REQUEST
        elif result and (
            result.get("recovery_check_failed") or result.get("allocation_check_failed")
        ):
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HttpResponse(error_msg, status=status_code)

    broker_status = _normalize_order_status(result.get("status"))

    # Save order to database with correct field names
    order = Order(
        user=user,
        stock=stock,
        symbol=ticker.upper(),
        quantity=quantity,
        side=transaction_side,
        order_type=order_type,
        status=broker_status if broker_status in dict(Order.STATUS) else "accepted",
        alpaca_order_id=str(result.get("id", "")),
        limit_price=limit_price,
        stop_price=stop_price,
        filled_avg_price=result.get("filled_price"),
    )
    order.save()

    # Persist canonical transaction-side ledger record.
    TradeTransaction.objects.create(
        user=user,
        company=stock.company,
        order=order,
        symbol=ticker.upper(),
        transaction_type=transaction_side.upper(),
        quantity=Decimal(str(quantity)),
        price=_extract_transaction_price(result, limit_price, stop_price),
        status=broker_status,
        metadata={
            "order_type": order_type,
            "alpaca_order_id": str(result.get("id", "")),
            "raw_status": str(result.get("status", "")),
        },
        legacy_reference=str(result.get("id", "")),
    )

    return HttpResponse(status=status.HTTP_201_CREATED)


@login_required
def create_company(request):
    """Create a new Company via API."""
    if request.method != "POST":
        return HttpResponse(
            "Method not allowed", status=status.HTTP_405_METHOD_NOT_ALLOWED
        )

    payload, error = _get_request_payload(request)
    if error:
        return HttpResponse(error, status=status.HTTP_400_BAD_REQUEST)

    ticker = str(payload.get("ticker", "")).strip().upper()
    name = str(payload.get("name", "")).strip()

    if not ticker:
        return HttpResponse(
            "ticker is required", status=status.HTTP_400_BAD_REQUEST
        )
    if not _TICKER_PATTERN.fullmatch(ticker):
        return HttpResponse(
            "Invalid ticker format", status=status.HTTP_400_BAD_REQUEST
        )

    company, created = Company.objects.get_or_create(
        ticker=ticker, defaults={"name": name or ticker}
    )
    Stock.objects.get_or_create(company=company)

    response_status = (
        status.HTTP_201_CREATED if created else status.HTTP_200_OK
    )
    return JsonResponse(
        {
            "ticker": company.ticker,
            "name": company.name,
            "created": created,
        },
        status=response_status,
    )


@login_required
def patch_company(request, ticker):
    """Patch an existing Company record."""
    if request.method != "PATCH":
        return HttpResponse(
            "Method not allowed", status=status.HTTP_405_METHOD_NOT_ALLOWED
        )

    payload, error = _get_request_payload(request)
    if error:
        return HttpResponse(error, status=status.HTTP_400_BAD_REQUEST)

    company = Company.objects.filter(ticker=ticker.upper()).first()
    if company is None:
        return HttpResponse(
            f"Company not found: {ticker}", status=status.HTTP_404_NOT_FOUND
        )

    name = payload.get("name")
    if name is None or not str(name).strip():
        return HttpResponse(
            "name is required for patch", status=status.HTTP_400_BAD_REQUEST
        )

    company.name = str(name).strip()
    company.save(update_fields=["name"])

    return JsonResponse(
        {
            "ticker": company.ticker,
            "name": company.name,
            "updated": True,
        },
        status=status.HTTP_200_OK,
    )


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

    alpaca_api = AlpacaManager(
        user.credential.alpaca_id,
        user.credential.alpaca_key,
        paper_trading=_is_user_paper_trading(user),
    )

    # Close the position
    success = alpaca_api.close_position(symbol.upper(), percentage, user=user)

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
    alpaca_api = AlpacaManager(
        user.credential.alpaca_id,
        user.credential.alpaca_key,
        paper_trading=_is_user_paper_trading(user),
    )

    positions = alpaca_api.get_positions()

    for pos in positions:
        if pos["symbol"].upper() == symbol.upper():
            return JsonResponse(pos)

    return HttpResponse(
        f"Position not found: {symbol}",
        status=status.HTTP_404_NOT_FOUND,
    )
