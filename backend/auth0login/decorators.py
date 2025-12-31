"""
Custom decorators for the trading platform.
"""

import functools
import logging
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required

from .services.trading_gate import trading_gate_service

logger = logging.getLogger(__name__)


def require_live_trading_approved(view_func):
    """
    Decorator that checks if user has live trading approval before
    allowing order submission.

    Usage:
        @login_required
        @require_live_trading_approved
        def submit_order(request):
            # Only reached if live trading is approved
            ...

    Returns 403 with clear message if live trading is not approved.
    """
    @functools.wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        # Ensure user is authenticated
        if not request.user.is_authenticated:
            return JsonResponse({
                'status': 'error',
                'error_code': 'AUTHENTICATION_REQUIRED',
                'message': 'Authentication required.',
            }, status=401)

        # Check live trading approval
        is_allowed, reason = trading_gate_service.is_live_trading_allowed(request.user)

        if not is_allowed:
            logger.warning(
                f"Live trading blocked for user {request.user.username}: {reason}"
            )
            return JsonResponse({
                'status': 'error',
                'error_code': 'LIVE_TRADING_NOT_APPROVED',
                'message': reason,
                'action_required': 'Complete paper trading requirements to unlock live trading.',
                'help_url': '/setup#trading-gate',
            }, status=403)

        return view_func(request, *args, **kwargs)

    return _wrapped_view


def require_paper_or_live_trading(view_func):
    """
    Decorator that allows either paper OR live trading.
    Useful for endpoints that should work in both modes.

    This checks:
    1. If trading in paper mode - always allowed
    2. If trading in live mode - requires approval

    Usage:
        @login_required
        @require_paper_or_live_trading
        def submit_order(request):
            # Check request.paper_mode to determine which mode
            ...
    """
    @functools.wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({
                'status': 'error',
                'error_code': 'AUTHENTICATION_REQUIRED',
                'message': 'Authentication required.',
            }, status=401)

        # Determine if this is a paper trading request
        # Check POST data, GET params, or headers
        is_paper_request = _is_paper_trading_request(request)

        if is_paper_request:
            # Paper trading is always allowed
            request.is_paper_mode = True
            return view_func(request, *args, **kwargs)

        # Live trading - check approval
        is_allowed, reason = trading_gate_service.is_live_trading_allowed(request.user)

        if not is_allowed:
            logger.warning(
                f"Live trading blocked for user {request.user.username}: {reason}"
            )
            return JsonResponse({
                'status': 'error',
                'error_code': 'LIVE_TRADING_NOT_APPROVED',
                'message': reason,
                'suggestion': 'Set paper_trading=true to trade in paper mode while completing requirements.',
            }, status=403)

        request.is_paper_mode = False
        return view_func(request, *args, **kwargs)

    return _wrapped_view


def _is_paper_trading_request(request) -> bool:
    """
    Determine if a request is for paper trading.

    Checks:
    1. POST body for paper_trading field
    2. GET params for paper param
    3. Custom header X-Paper-Trading
    """
    import json

    # Check POST body
    if request.method in ['POST', 'PUT', 'PATCH']:
        try:
            if request.content_type == 'application/json':
                data = json.loads(request.body)
                paper_value = data.get('paper_trading') or data.get('paper')
                if paper_value is not None:
                    return str(paper_value).lower() in ('true', '1', 'yes')
        except (json.JSONDecodeError, AttributeError):
            pass

    # Check GET params
    paper_param = request.GET.get('paper_trading') or request.GET.get('paper')
    if paper_param is not None:
        return str(paper_param).lower() in ('true', '1', 'yes')

    # Check custom header
    header_value = request.headers.get('X-Paper-Trading')
    if header_value is not None:
        return str(header_value).lower() in ('true', '1', 'yes')

    # Default to paper trading if not specified (safe default)
    return True


class LiveTradingRequired:
    """
    Class-based version of require_live_trading_approved for CBVs.

    Usage with Django CBV:
        class OrderCreateView(LiveTradingRequired, CreateView):
            ...
    """

    def dispatch(self, request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({
                'status': 'error',
                'error_code': 'AUTHENTICATION_REQUIRED',
                'message': 'Authentication required.',
            }, status=401)

        is_allowed, reason = trading_gate_service.is_live_trading_allowed(request.user)

        if not is_allowed:
            return JsonResponse({
                'status': 'error',
                'error_code': 'LIVE_TRADING_NOT_APPROVED',
                'message': reason,
                'action_required': 'Complete paper trading requirements.',
            }, status=403)

        return super().dispatch(request, *args, **kwargs)
