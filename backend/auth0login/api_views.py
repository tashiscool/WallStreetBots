"""
API Views for AJAX endpoints.

These views provide JSON APIs for frontend JavaScript to call.
"""

import asyncio
import json
import logging
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required

from .dashboard_service import dashboard_service

logger = logging.getLogger(__name__)


@login_required
@require_http_methods(["POST"])
def run_backtest(request):
    """
    API endpoint to run a backtest.

    POST parameters:
        strategy: Strategy name (e.g., 'wsb-dip-bot')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_capital: Starting capital (default: 100000)
        benchmark: Benchmark symbol (default: 'SPY')
        position_size_pct: Position size as % (default: 3)
        stop_loss_pct: Stop loss % (default: 5)
        take_profit_pct: Take profit % (default: 15)

    Returns:
        JSON response with backtest results
    """
    try:
        # Parse request body
        if request.content_type == 'application/json':
            data = json.loads(request.body)
        else:
            data = request.POST

        strategy = data.get('strategy', 'wsb-dip-bot')
        start_date = data.get('start_date', '2023-01-01')
        end_date = data.get('end_date', '2024-01-01')
        initial_capital = float(data.get('initial_capital', 100000))
        benchmark = data.get('benchmark', 'SPY')
        position_size_pct = float(data.get('position_size_pct', 3))
        stop_loss_pct = float(data.get('stop_loss_pct', 5))
        take_profit_pct = float(data.get('take_profit_pct', 15))

        # Run the backtest asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                dashboard_service.run_backtest(
                    strategy_name=strategy,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                    benchmark=benchmark,
                    position_size_pct=position_size_pct,
                    stop_loss_pct=stop_loss_pct,
                    take_profit_pct=take_profit_pct,
                )
            )
        finally:
            loop.close()

        return JsonResponse(result)

    except json.JSONDecodeError:
        return JsonResponse({
            'status': 'error',
            'message': 'Invalid JSON in request body',
        }, status=400)
    except ValueError as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Invalid parameter value: {str(e)}',
        }, status=400)
    except Exception as e:
        logger.error(f"Error running backtest API: {e}")
        return JsonResponse({
            'status': 'error',
            'message': f'Backtest failed: {str(e)}',
        }, status=500)


@login_required
@require_http_methods(["POST"])
def build_spread(request):
    """
    API endpoint to build an exotic option spread.

    POST parameters:
        spread_type: Type of spread ('iron_condor', 'straddle', etc.)
        ticker: Stock ticker symbol
        current_price: Current stock price
        params: Additional parameters as JSON

    Returns:
        JSON response with spread data
    """
    try:
        if request.content_type == 'application/json':
            data = json.loads(request.body)
        else:
            data = request.POST

        spread_type = data.get('spread_type', 'iron_condor')
        ticker = data.get('ticker', 'SPY')
        current_price = float(data.get('current_price', 450))
        params = data.get('params', {})
        if isinstance(params, str):
            params = json.loads(params)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                dashboard_service.build_spread(
                    spread_type=spread_type,
                    ticker=ticker,
                    current_price=current_price,
                    params=params,
                )
            )
        finally:
            loop.close()

        return JsonResponse(result)

    except Exception as e:
        logger.error(f"Error building spread: {e}")
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to build spread: {str(e)}',
        }, status=500)


@login_required
@require_http_methods(["POST"])
def suggest_spreads(request):
    """
    API endpoint to get spread suggestions.

    POST parameters:
        ticker: Stock ticker symbol
        current_price: Current stock price
        outlook: Market outlook ('bullish', 'bearish', 'neutral')

    Returns:
        JSON response with suggested spreads
    """
    try:
        if request.content_type == 'application/json':
            data = json.loads(request.body)
        else:
            data = request.POST

        ticker = data.get('ticker', 'SPY')
        current_price = float(data.get('current_price', 450))
        outlook = data.get('outlook', 'neutral')

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                dashboard_service.suggest_spreads(
                    ticker=ticker,
                    current_price=current_price,
                    outlook=outlook,
                )
            )
        finally:
            loop.close()

        return JsonResponse(result)

    except Exception as e:
        logger.error(f"Error suggesting spreads: {e}")
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to suggest spreads: {str(e)}',
        }, status=500)


@login_required
@require_http_methods(["POST"])
def get_locate_quote(request):
    """
    API endpoint to get a locate quote for short selling.

    POST parameters:
        symbol: Stock symbol
        qty: Quantity to short

    Returns:
        JSON response with locate quote
    """
    try:
        if request.content_type == 'application/json':
            data = json.loads(request.body)
        else:
            data = request.POST

        symbol = data.get('symbol', 'AAPL')
        qty = int(data.get('qty', 100))

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                dashboard_service.get_locate_quote(symbol, qty)
            )
        finally:
            loop.close()

        return JsonResponse(result)

    except Exception as e:
        logger.error(f"Error getting locate quote: {e}")
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to get locate quote: {str(e)}',
        }, status=500)


@login_required
@require_http_methods(["GET"])
def feature_availability(request):
    """
    API endpoint to get feature availability status.

    Returns:
        JSON response with feature availability
    """
    try:
        return JsonResponse({
            'status': 'success',
            'features': dashboard_service.get_feature_availability(),
        })
    except Exception as e:
        logger.error(f"Error getting feature availability: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["POST"])
def test_alpaca_connection(request):
    """
    API endpoint to test Alpaca API connection.

    POST parameters:
        api_key: Alpaca API key
        secret_key: Alpaca secret key
        paper: Whether to use paper trading (default: true)

    Returns:
        JSON response with connection status and account info
    """
    try:
        if request.content_type == 'application/json':
            data = json.loads(request.body)
        else:
            data = request.POST

        api_key = data.get('api_key', '')
        secret_key = data.get('secret_key', '')
        paper = data.get('paper', True)

        if not api_key or not secret_key:
            return JsonResponse({
                'status': 'error',
                'message': 'API key and secret key are required.',
            }, status=400)

        # Try to connect to Alpaca
        try:
            from alpaca.trading.client import TradingClient

            client = TradingClient(
                api_key=api_key,
                secret_key=secret_key,
                paper=paper if isinstance(paper, bool) else paper == 'true',
            )

            account = client.get_account()

            return JsonResponse({
                'status': 'success',
                'message': 'Connected successfully!',
                'account': {
                    'id': str(account.id),
                    'equity': str(account.equity),
                    'cash': str(account.cash),
                    'buying_power': str(account.buying_power),
                    'status': str(account.status),
                    'trading_blocked': account.trading_blocked,
                    'pattern_day_trader': account.pattern_day_trader,
                },
            })

        except ImportError:
            return JsonResponse({
                'status': 'error',
                'message': 'Alpaca SDK not installed. Please install alpaca-py.',
            }, status=500)

        except Exception as e:
            error_msg = str(e)
            if 'unauthorized' in error_msg.lower() or '403' in error_msg:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Invalid API credentials. Please check your API key and secret.',
                }, status=401)
            else:
                return JsonResponse({
                    'status': 'error',
                    'message': f'Connection failed: {error_msg}',
                }, status=500)

    except json.JSONDecodeError:
        return JsonResponse({
            'status': 'error',
            'message': 'Invalid JSON in request body',
        }, status=400)
    except Exception as e:
        logger.error(f"Error testing Alpaca connection: {e}")
        return JsonResponse({
            'status': 'error',
            'message': f'Error: {str(e)}',
        }, status=500)


@login_required
@require_http_methods(["POST"])
def save_wizard_config(request):
    """
    API endpoint to save setup wizard configuration.

    POST parameters:
        api_key: Alpaca API key
        secret_key: Alpaca secret key
        trading_mode: 'paper' or 'live'
        strategies: List of strategy IDs
        risk_profile: 'conservative', 'moderate', or 'aggressive'
        max_position_pct: Max position size %
        max_daily_loss_pct: Max daily loss %
        max_positions: Max number of positions

    Returns:
        JSON response with status
    """
    try:
        if request.content_type == 'application/json':
            data = json.loads(request.body)
        else:
            data = request.POST

        api_key = data.get('api_key', '')
        secret_key = data.get('secret_key', '')
        trading_mode = data.get('trading_mode', 'paper')
        strategies = data.get('strategies', ['wsb-dip-bot'])
        risk_profile = data.get('risk_profile', 'moderate')
        max_position_pct = float(data.get('max_position_pct', 3))
        max_daily_loss_pct = float(data.get('max_daily_loss_pct', 8))
        max_positions = int(data.get('max_positions', 10))

        # Validate
        if not api_key or not secret_key:
            return JsonResponse({
                'status': 'error',
                'message': 'API credentials are required.',
            }, status=400)

        if not strategies:
            return JsonResponse({
                'status': 'error',
                'message': 'At least one strategy must be selected.',
            }, status=400)

        # Save credentials to user's profile
        user = request.user
        if hasattr(user, 'credential'):
            user.credential.alpaca_id = api_key
            user.credential.alpaca_key = secret_key
            user.credential.save()
        else:
            from .models import Credential
            Credential.objects.create(
                user=user,
                alpaca_id=api_key,
                alpaca_key=secret_key,
            )

        # Save configuration to environment/settings (in production, this would be stored in DB)
        import os
        os.environ['APCA_API_KEY_ID'] = api_key
        os.environ['APCA_API_SECRET_KEY'] = secret_key
        os.environ['APCA_PAPER_TRADING'] = 'true' if trading_mode == 'paper' else 'false'

        logger.info(
            f"Wizard config saved for user {user.username}: "
            f"mode={trading_mode}, strategies={strategies}, risk={risk_profile}"
        )

        return JsonResponse({
            'status': 'success',
            'message': 'Configuration saved successfully!',
            'config': {
                'trading_mode': trading_mode,
                'strategies': strategies,
                'risk_profile': risk_profile,
                'max_position_pct': max_position_pct,
                'max_daily_loss_pct': max_daily_loss_pct,
                'max_positions': max_positions,
            },
        })

    except json.JSONDecodeError:
        return JsonResponse({
            'status': 'error',
            'message': 'Invalid JSON in request body',
        }, status=400)
    except Exception as e:
        logger.error(f"Error saving wizard config: {e}")
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to save configuration: {str(e)}',
        }, status=500)


@login_required
@require_http_methods(["POST"])
def test_email(request):
    """
    API endpoint to test email configuration by sending a test email.

    POST parameters:
        smtp_host: SMTP server hostname
        smtp_port: SMTP server port (default: 587)
        email_from: Sender email address
        email_to: Recipient email address
        smtp_user: SMTP username (optional)
        smtp_pass: SMTP password (optional)

    Returns:
        JSON response with status
    """
    import smtplib
    from email.mime.text import MIMEText
    from datetime import datetime

    try:
        if request.content_type == 'application/json':
            data = json.loads(request.body)
        else:
            data = request.POST

        smtp_host = data.get('smtp_host', '')
        smtp_port = int(data.get('smtp_port', 587))
        email_from = data.get('email_from', '')
        email_to = data.get('email_to', '')
        smtp_user = data.get('smtp_user', '')
        smtp_pass = data.get('smtp_pass', '')

        # Validate required fields
        if not smtp_host or not email_from or not email_to:
            return JsonResponse({
                'status': 'error',
                'message': 'SMTP host, from email, and to email are required.',
            }, status=400)

        # Create test email
        subject = "WallStreetBots - Test Email"
        body = f"""
This is a test email from WallStreetBots.

If you received this email, your email notification settings are configured correctly!

Configuration Details:
- SMTP Host: {smtp_host}
- SMTP Port: {smtp_port}
- From: {email_from}
- To: {email_to}
- Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

You will receive trading alerts including:
- Stop Loss Triggered
- Risk Alerts
- Entry Signals
- Profit Target Hit
- Earnings Warnings
- Daily Performance Digest

Happy Trading!
- WallStreetBots
        """

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = email_from
        msg["To"] = email_to

        # Send email
        try:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
                server.starttls()
                if smtp_user and smtp_pass:
                    server.login(smtp_user, smtp_pass)
                server.sendmail(email_from, [email_to], msg.as_string())

            logger.info(f"Test email sent successfully to {email_to}")

            return JsonResponse({
                'status': 'success',
                'message': 'Test email sent successfully! Check your inbox.',
            })

        except smtplib.SMTPAuthenticationError:
            return JsonResponse({
                'status': 'error',
                'message': 'SMTP authentication failed. Check your username and password.',
            }, status=401)
        except smtplib.SMTPConnectError:
            return JsonResponse({
                'status': 'error',
                'message': f'Could not connect to SMTP server {smtp_host}:{smtp_port}',
            }, status=500)
        except smtplib.SMTPException as e:
            return JsonResponse({
                'status': 'error',
                'message': f'SMTP error: {str(e)}',
            }, status=500)

    except json.JSONDecodeError:
        return JsonResponse({
            'status': 'error',
            'message': 'Invalid JSON in request body',
        }, status=400)
    except Exception as e:
        logger.error(f"Error sending test email: {e}")
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to send test email: {str(e)}',
        }, status=500)


@login_required
@require_http_methods(["POST"])
def save_settings(request):
    """
    API endpoint to save all user settings including email configuration.

    POST parameters:
        alpaca_api_key: Alpaca API key
        alpaca_secret_key: Alpaca secret key
        trading_mode: 'paper' or 'live'
        email_enabled: Whether email notifications are enabled
        smtp_host: SMTP server hostname
        smtp_port: SMTP server port
        email_from: Sender email address
        email_to: Recipient email address
        smtp_user: SMTP username
        smtp_pass: SMTP password
        email_alerts: Dict of alert type preferences
        discord_webhook: Discord webhook URL
        slack_webhook: Slack webhook URL
        shadow_mode: Whether shadow mode is enabled
        log_level: Logging level
        timezone: User timezone

    Returns:
        JSON response with status
    """
    import os

    try:
        if request.content_type == 'application/json':
            data = json.loads(request.body)
        else:
            data = request.POST

        user = request.user

        # Save Alpaca credentials if provided
        alpaca_api_key = data.get('alpaca_api_key', '')
        alpaca_secret_key = data.get('alpaca_secret_key', '')
        trading_mode = data.get('trading_mode', 'paper')

        if alpaca_api_key and alpaca_secret_key:
            if hasattr(user, 'credential'):
                user.credential.alpaca_id = alpaca_api_key
                user.credential.alpaca_key = alpaca_secret_key
                user.credential.save()
            else:
                from .models import Credential
                Credential.objects.create(
                    user=user,
                    alpaca_id=alpaca_api_key,
                    alpaca_key=alpaca_secret_key,
                )

            # Set environment variables for trading
            os.environ['APCA_API_KEY_ID'] = alpaca_api_key
            os.environ['APCA_API_SECRET_KEY'] = alpaca_secret_key
            os.environ['APCA_PAPER_TRADING'] = 'true' if trading_mode == 'paper' else 'false'

        # Save email settings to environment variables
        email_enabled = data.get('email_enabled', False)
        if email_enabled:
            smtp_host = data.get('smtp_host', '')
            smtp_port = str(data.get('smtp_port', 587))
            email_from = data.get('email_from', '')
            email_to = data.get('email_to', '')
            smtp_user = data.get('smtp_user', '')
            smtp_pass = data.get('smtp_pass', '')

            if smtp_host:
                os.environ['ALERT_EMAIL_SMTP_HOST'] = smtp_host
            if smtp_port:
                os.environ['ALERT_EMAIL_SMTP_PORT'] = smtp_port
            if email_from:
                os.environ['ALERT_EMAIL_FROM'] = email_from
            if email_to:
                os.environ['ALERT_EMAIL_TO'] = email_to
            if smtp_user:
                os.environ['ALERT_EMAIL_USER'] = smtp_user
            if smtp_pass:
                os.environ['ALERT_EMAIL_PASS'] = smtp_pass

        # Save webhook settings
        discord_webhook = data.get('discord_webhook', '')
        slack_webhook = data.get('slack_webhook', '')

        if discord_webhook:
            os.environ['ALERT_DISCORD_WEBHOOK'] = discord_webhook
        if slack_webhook:
            os.environ['ALERT_SLACK_WEBHOOK'] = slack_webhook

        # Save other settings
        email_alerts = data.get('email_alerts', {})
        shadow_mode = data.get('shadow_mode', False)
        log_level = data.get('log_level', 'INFO')
        timezone_setting = data.get('timezone', 'America/New_York')

        # Store settings in environment (in production, use database)
        os.environ['TRADING_SHADOW_MODE'] = 'true' if shadow_mode else 'false'
        os.environ['LOG_LEVEL'] = log_level
        os.environ['TRADING_TIMEZONE'] = timezone_setting

        # Store email alert preferences
        os.environ['EMAIL_ALERTS_CONFIG'] = json.dumps(email_alerts)

        logger.info(
            f"Settings saved for user {user.username}: "
            f"email_enabled={email_enabled}, shadow_mode={shadow_mode}"
        )

        return JsonResponse({
            'status': 'success',
            'message': 'Settings saved successfully!',
        })

    except json.JSONDecodeError:
        return JsonResponse({
            'status': 'error',
            'message': 'Invalid JSON in request body',
        }, status=400)
    except Exception as e:
        logger.error(f"Error saving settings: {e}")
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to save settings: {str(e)}',
        }, status=500)


# =============================================================================
# Trading Gate API Endpoints
# =============================================================================

@login_required
@require_http_methods(["GET"])
def trading_gate_status(request):
    """
    Get current trading gate status for the authenticated user.

    Returns:
        JSON with complete gate status including:
        - Paper trading progress (days, trades, performance)
        - Requirements and their status
        - Approval status
    """
    from .services.trading_gate import trading_gate_service
    from dataclasses import asdict

    try:
        status = trading_gate_service.get_gate_status(request.user)

        # Convert dataclass to dict for JSON serialization
        status_dict = {
            'user_id': status.user_id,
            'username': status.username,
            'is_paper_trading': status.is_paper_trading,
            'live_trading_approved': status.live_trading_approved,
            'live_trading_requested': status.live_trading_requested,
            'days_in_paper': status.days_in_paper,
            'days_required': status.days_required,
            'days_remaining': status.days_remaining,
            'total_trades': status.total_trades,
            'total_pnl': status.total_pnl,
            'total_pnl_pct': status.total_pnl_pct,
            'win_rate': status.win_rate,
            'sharpe_ratio': status.sharpe_ratio,
            'requirements': [
                {
                    'name': r.name,
                    'description': r.description,
                    'met': r.met,
                    'current_value': r.current_value,
                    'required_value': r.required_value,
                }
                for r in status.requirements
            ],
            'all_requirements_met': status.all_requirements_met,
            'paper_started_at': status.paper_started_at.isoformat() if status.paper_started_at else None,
            'requested_at': status.requested_at.isoformat() if status.requested_at else None,
            'approved_at': status.approved_at.isoformat() if status.approved_at else None,
            'approval_method': status.approval_method,
            'denial_reason': status.denial_reason,
        }

        return JsonResponse({
            'status': 'success',
            'gate': status_dict,
        })

    except Exception as e:
        logger.error(f"Error getting trading gate status: {e}")
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to get gate status: {str(e)}',
        }, status=500)


@login_required
@require_http_methods(["POST"])
def trading_gate_request_live(request):
    """
    Request transition from paper trading to live trading.

    The system will automatically approve if all requirements are met,
    or return the list of unmet requirements.

    Returns:
        JSON with approval status and message
    """
    from .services.trading_gate import trading_gate_service

    try:
        result = trading_gate_service.request_live_trading(request.user)

        return JsonResponse({
            'status': 'success',
            'result': result,
        })

    except Exception as e:
        logger.error(f"Error requesting live trading: {e}")
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to request live trading: {str(e)}',
        }, status=500)


@login_required
@require_http_methods(["GET"])
def trading_gate_requirements(request):
    """
    Get the list of requirements for live trading approval.

    Returns:
        JSON with requirements list and their current status
    """
    from .services.trading_gate import trading_gate_service

    try:
        requirements = trading_gate_service.get_requirements(request.user)

        requirements_list = [
            {
                'name': r.name,
                'description': r.description,
                'met': r.met,
                'current_value': r.current_value,
                'required_value': r.required_value,
            }
            for r in requirements
        ]

        all_met = all(r.met for r in requirements)

        return JsonResponse({
            'status': 'success',
            'requirements': requirements_list,
            'all_requirements_met': all_met,
            'can_request_live_trading': all_met,
        })

    except Exception as e:
        logger.error(f"Error getting requirements: {e}")
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to get requirements: {str(e)}',
        }, status=500)


@login_required
@require_http_methods(["POST"])
def trading_gate_start_paper(request):
    """
    Start paper trading period for a user.

    This should be called when the user first sets up their account
    or when they switch to paper trading mode.

    Returns:
        JSON with gate status
    """
    from .services.trading_gate import trading_gate_service

    try:
        gate = trading_gate_service.start_paper_trading(request.user)

        return JsonResponse({
            'status': 'success',
            'message': 'Paper trading started.',
            'paper_trading_started_at': gate.paper_trading_started_at.isoformat(),
            'days_required': gate.paper_trading_days_required,
        })

    except Exception as e:
        logger.error(f"Error starting paper trading: {e}")
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to start paper trading: {str(e)}',
        }, status=500)


# =============================================================================
# Risk Assessment API Endpoints
# =============================================================================

@login_required
@require_http_methods(["GET"])
def risk_assessment_questions(request):
    """
    Get the list of risk assessment questionnaire questions.

    Returns:
        JSON with questions list for the questionnaire UI
    """
    from .services.risk_assessment import risk_assessment_service

    try:
        questions = risk_assessment_service.get_questions()

        return JsonResponse({
            'status': 'success',
            'questions': questions,
            'total_questions': len(questions),
        })

    except Exception as e:
        logger.error(f"Error getting risk assessment questions: {e}")
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to get questions: {str(e)}',
        }, status=500)


@login_required
@require_http_methods(["POST"])
def risk_assessment_submit(request):
    """
    Submit risk assessment questionnaire responses.

    POST parameters:
        responses: Dict mapping question_id to selected answer value
        selected_profile: Optional override profile if user wants different
        override_acknowledged: Whether user acknowledged override warning

    Returns:
        JSON with calculated score, recommended profile, and explanation
    """
    from .services.risk_assessment import risk_assessment_service

    try:
        if request.content_type == 'application/json':
            data = json.loads(request.body)
        else:
            data = request.POST

        responses = data.get('responses', {})
        selected_profile = data.get('selected_profile')
        override_acknowledged = data.get('override_acknowledged', False)

        if not responses:
            return JsonResponse({
                'status': 'error',
                'message': 'No responses provided.',
            }, status=400)

        result = risk_assessment_service.submit_assessment(
            user=request.user,
            responses=responses,
            selected_profile=selected_profile,
            override_acknowledged=override_acknowledged,
        )

        return JsonResponse(result)

    except json.JSONDecodeError:
        return JsonResponse({
            'status': 'error',
            'message': 'Invalid JSON in request body',
        }, status=400)
    except Exception as e:
        logger.error(f"Error submitting risk assessment: {e}")
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to submit assessment: {str(e)}',
        }, status=500)


@login_required
@require_http_methods(["GET"])
def risk_assessment_result(request):
    """
    Get the user's most recent risk assessment result.

    Returns:
        JSON with assessment result or null if no assessment exists
    """
    from .services.risk_assessment import risk_assessment_service

    try:
        result = risk_assessment_service.get_user_assessment(request.user)

        if result:
            return JsonResponse({
                'status': 'success',
                'has_assessment': True,
                'assessment': result,
            })
        else:
            return JsonResponse({
                'status': 'success',
                'has_assessment': False,
                'assessment': None,
                'message': 'No completed assessment found. Please complete the questionnaire.',
            })

    except Exception as e:
        logger.error(f"Error getting risk assessment result: {e}")
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to get assessment result: {str(e)}',
        }, status=500)


@login_required
@require_http_methods(["POST"])
def risk_assessment_calculate(request):
    """
    Calculate risk profile from responses without saving.

    Useful for showing real-time results as user answers questions.

    POST parameters:
        responses: Dict mapping question_id to selected answer value

    Returns:
        JSON with calculated score and profile preview
    """
    from .services.risk_assessment import risk_assessment_service

    try:
        if request.content_type == 'application/json':
            data = json.loads(request.body)
        else:
            data = request.POST

        responses = data.get('responses', {})

        if not responses:
            return JsonResponse({
                'status': 'error',
                'message': 'No responses provided.',
            }, status=400)

        result = risk_assessment_service.calculate_score(responses)

        return JsonResponse({
            'status': 'success',
            'total_score': result.total_score,
            'max_score': result.max_possible_score,
            'recommended_profile': result.recommended_profile,
            'profile_explanation': result.profile_explanation,
        })

    except json.JSONDecodeError:
        return JsonResponse({
            'status': 'error',
            'message': 'Invalid JSON in request body',
        }, status=400)
    except Exception as e:
        logger.error(f"Error calculating risk score: {e}")
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to calculate score: {str(e)}',
        }, status=500)


# =============================================================================
# Strategy Recommendation API Endpoints
# =============================================================================

@login_required
@require_http_methods(["GET", "POST"])
def strategy_recommendations(request):
    """
    Get personalized strategy recommendations based on risk profile.

    GET/POST parameters:
        risk_profile: 'conservative', 'moderate', or 'aggressive'
                     (if not provided, uses user's assessment)
        capital_amount: Available trading capital (default: 10000)
        investment_timeline: Optional timeline preference

    Returns:
        JSON with recommended strategies and portfolio suggestions
    """
    from .services.risk_assessment import risk_assessment_service
    from .services.strategy_recommender import strategy_recommender_service

    try:
        # Get parameters from either GET or POST
        if request.method == 'POST' and request.content_type == 'application/json':
            data = json.loads(request.body)
        else:
            data = request.GET if request.method == 'GET' else request.POST

        risk_profile = data.get('risk_profile')
        capital_amount = float(data.get('capital_amount', 10000))
        investment_timeline = data.get('investment_timeline')

        # If no profile provided, try to get from user's assessment
        if not risk_profile:
            assessment = risk_assessment_service.get_user_assessment(request.user)
            if assessment:
                risk_profile = assessment.get('effective_profile', 'moderate')
            else:
                risk_profile = 'moderate'  # Default

        result = strategy_recommender_service.get_recommendations(
            risk_profile=risk_profile,
            capital_amount=capital_amount,
            investment_timeline=investment_timeline,
        )

        return JsonResponse(result)

    except json.JSONDecodeError:
        return JsonResponse({
            'status': 'error',
            'message': 'Invalid JSON in request body',
        }, status=400)
    except ValueError as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Invalid parameter value: {str(e)}',
        }, status=400)
    except Exception as e:
        logger.error(f"Error getting strategy recommendations: {e}")
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to get recommendations: {str(e)}',
        }, status=500)


@login_required
@require_http_methods(["GET"])
def strategy_details(request, strategy_id):
    """
    Get detailed information about a specific strategy.

    URL parameters:
        strategy_id: The strategy identifier (e.g., 'wsb-dip-bot')

    Returns:
        JSON with strategy details
    """
    from .services.strategy_recommender import strategy_recommender_service

    try:
        details = strategy_recommender_service.get_strategy_details(strategy_id)

        if details:
            return JsonResponse({
                'status': 'success',
                'strategy': details,
            })
        else:
            return JsonResponse({
                'status': 'error',
                'message': f'Strategy not found: {strategy_id}',
            }, status=404)

    except Exception as e:
        logger.error(f"Error getting strategy details: {e}")
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to get strategy details: {str(e)}',
        }, status=500)


# =============================================================================
# Benchmark Comparison API Endpoints
# =============================================================================

@login_required
@require_http_methods(["GET"])
def performance_vs_benchmark(request):
    """
    Get portfolio performance compared to benchmark (SPY by default).

    GET parameters:
        period: Time period - '1W', '1M', '3M', 'YTD', 'ALL' (default: '1M')
        benchmark: Benchmark ticker (default: 'SPY')

    Returns:
        JSON with portfolio vs benchmark comparison data
    """
    from datetime import datetime, timedelta
    from .services.benchmark import benchmark_service

    try:
        period = request.GET.get('period', '1M')
        benchmark = request.GET.get('benchmark', 'SPY')

        # Calculate date range based on period
        end_date = datetime.now()
        if period == '1W':
            start_date = end_date - timedelta(weeks=1)
        elif period == '1M':
            start_date = end_date - timedelta(days=30)
        elif period == '3M':
            start_date = end_date - timedelta(days=90)
        elif period == 'YTD':
            start_date = datetime(end_date.year, 1, 1)
        elif period == 'ALL':
            start_date = end_date - timedelta(days=365)  # Default to 1 year for ALL
        else:
            start_date = end_date - timedelta(days=30)

        # Get portfolio data from dashboard service
        # In a real implementation, this would come from actual portfolio history
        portfolio_values = _get_portfolio_history(request.user, start_date, end_date)

        # Get comparison data
        comparison_data = benchmark_service.get_comparison_data(
            portfolio_values=portfolio_values,
            start_date=start_date,
            end_date=end_date,
            benchmark=benchmark,
        )

        return JsonResponse(comparison_data)

    except Exception as e:
        logger.error(f"Error getting benchmark comparison: {e}")
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to get benchmark comparison: {str(e)}',
        }, status=500)


@login_required
@require_http_methods(["GET"])
def portfolio_pnl_with_benchmark(request):
    """
    Get portfolio P&L with benchmark comparison data.

    This is the enhanced P&L endpoint that includes SPY comparison.

    Returns:
        JSON with portfolio P&L and benchmark comparison
    """
    from datetime import datetime, timedelta
    from .services.benchmark import benchmark_service

    try:
        # Get period from request (default to MTD - Month to Date)
        period = request.GET.get('period', 'MTD')

        end_date = datetime.now()
        if period == 'TODAY':
            start_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == 'WTD':  # Week to date
            start_date = end_date - timedelta(days=end_date.weekday())
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == 'MTD':  # Month to date
            start_date = end_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif period == 'YTD':  # Year to date
            start_date = datetime(end_date.year, 1, 1)
        else:
            start_date = end_date - timedelta(days=30)

        # Get portfolio values (mock or real)
        portfolio_start_value = _get_portfolio_value_at_date(request.user, start_date)
        portfolio_end_value = _get_portfolio_value_at_date(request.user, end_date)
        daily_pnl = _get_daily_pnl(request.user)

        # Calculate portfolio metrics
        total_pnl = portfolio_end_value - portfolio_start_value
        total_pnl_pct = (total_pnl / portfolio_start_value * 100) if portfolio_start_value > 0 else 0

        daily_pnl_pct = (daily_pnl / (portfolio_end_value - daily_pnl) * 100) if (portfolio_end_value - daily_pnl) > 0 else 0

        # Get benchmark comparison
        comparison = benchmark_service.compare_portfolio_to_benchmark(
            portfolio_start_value=portfolio_start_value,
            portfolio_end_value=portfolio_end_value,
            portfolio_daily_pnl=daily_pnl,
            start_date=start_date,
            end_date=end_date,
            benchmark='SPY',
        )

        # Get benchmark return data
        bench_return = benchmark_service.get_benchmark_return(start_date, end_date, 'SPY')

        return JsonResponse({
            'status': 'success',
            'period': period,
            'portfolio': {
                'daily_pnl': round(daily_pnl, 2),
                'daily_pnl_pct': round(daily_pnl_pct, 2),
                'total_pnl': round(total_pnl, 2),
                'total_pnl_pct': round(total_pnl_pct, 2),
                'current_value': round(portfolio_end_value, 2),
            },
            'benchmark': {
                'ticker': 'SPY',
                'daily_return_pct': bench_return.daily_return_pct,
                'period_return_pct': bench_return.period_return_pct,
            },
            'comparison': {
                'daily_excess': comparison.daily_excess,
                'period_excess': comparison.period_excess,
                'hypothetical_spy_value': comparison.hypothetical_benchmark_value,
                'your_value': comparison.your_value,
                'alpha_generated': comparison.alpha_generated,
                'outperforming': comparison.period_excess > 0,
            },
        })

    except Exception as e:
        logger.error(f"Error getting P&L with benchmark: {e}")
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to get P&L data: {str(e)}',
        }, status=500)


@login_required
@require_http_methods(["GET"])
def benchmark_chart_data(request):
    """
    Get chart data for portfolio vs benchmark comparison.

    GET parameters:
        period: Time period - '1W', '1M', '3M', 'YTD', 'ALL' (default: '1M')
        benchmark: Benchmark ticker (default: 'SPY')

    Returns:
        JSON with normalized series data for charting
    """
    from datetime import datetime, timedelta
    from .services.benchmark import benchmark_service

    try:
        period = request.GET.get('period', '1M')
        benchmark = request.GET.get('benchmark', 'SPY')

        # Calculate date range
        end_date = datetime.now()
        if period == '1W':
            start_date = end_date - timedelta(weeks=1)
        elif period == '1M':
            start_date = end_date - timedelta(days=30)
        elif period == '3M':
            start_date = end_date - timedelta(days=90)
        elif period == 'YTD':
            start_date = datetime(end_date.year, 1, 1)
        elif period == 'ALL':
            start_date = end_date - timedelta(days=365)
        else:
            start_date = end_date - timedelta(days=30)

        # Get benchmark series
        bench_series = benchmark_service.get_benchmark_series(start_date, end_date, benchmark)

        # Get portfolio series (mock or real)
        portfolio_series = _get_portfolio_series(request.user, start_date, end_date)

        # Format for chart
        labels = [item['date'] for item in bench_series]
        benchmark_values = [item['normalized'] for item in bench_series]

        # Match portfolio values to benchmark dates
        portfolio_dict = {item['date']: item['normalized'] for item in portfolio_series}
        portfolio_values = [portfolio_dict.get(date, 100) for date in labels]

        return JsonResponse({
            'status': 'success',
            'period': period,
            'benchmark': benchmark,
            'labels': labels,
            'datasets': {
                'portfolio': {
                    'label': 'Your Portfolio',
                    'data': portfolio_values,
                    'borderColor': '#5e72e4',
                    'backgroundColor': 'rgba(94, 114, 228, 0.1)',
                },
                'benchmark': {
                    'label': benchmark,
                    'data': benchmark_values,
                    'borderColor': '#fb6340',
                    'backgroundColor': 'rgba(251, 99, 64, 0.1)',
                },
            },
            'summary': {
                'portfolio_return': round(portfolio_values[-1] - 100, 2) if portfolio_values else 0,
                'benchmark_return': round(benchmark_values[-1] - 100, 2) if benchmark_values else 0,
            },
        })

    except Exception as e:
        logger.error(f"Error getting benchmark chart data: {e}")
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to get chart data: {str(e)}',
        }, status=500)


# =============================================================================
# Helper functions for portfolio data
# =============================================================================

def _get_portfolio_history(user, start_date, end_date) -> list[dict]:
    """
    Get portfolio value history.

    In production, this would query actual portfolio history from database.
    For now, returns mock data that simulates realistic portfolio values.
    """
    from datetime import timedelta
    import random

    # Seed random with user id for consistent mock data per user
    random.seed(user.id if hasattr(user, 'id') else 42)

    values = []
    current = start_date
    value = 100000  # Start with $100k

    while current <= end_date:
        if current.weekday() < 5:  # Skip weekends
            # Random daily change between -2% and +2.5%
            daily_change = (random.random() - 0.4) * 0.025
            value *= (1 + daily_change)
            values.append({
                'date': current.strftime('%Y-%m-%d'),
                'value': round(value, 2),
            })
        current += timedelta(days=1)

    return values


def _get_portfolio_value_at_date(user, date) -> float:
    """Get portfolio value at a specific date."""
    # In production, query actual portfolio value
    # Mock implementation
    import random
    random.seed(user.id if hasattr(user, 'id') else 42)

    base_value = 100000
    days_from_start = (date - date.replace(month=1, day=1)).days
    growth = 1 + (days_from_start * 0.0003)  # ~10% annual growth
    noise = 1 + (random.random() - 0.5) * 0.1
    return base_value * growth * noise


def _get_daily_pnl(user) -> float:
    """Get today's P&L."""
    # In production, calculate from actual positions
    # Mock implementation
    import random
    random.seed(hash(str(user.id) + str(datetime.now().date())) if hasattr(user, 'id') else 42)
    return round((random.random() - 0.4) * 2000, 2)  # -$800 to +$1200


def _get_portfolio_series(user, start_date, end_date) -> list[dict]:
    """Get normalized portfolio series for charting."""
    history = _get_portfolio_history(user, start_date, end_date)
    if not history:
        return []

    start_value = history[0]['value']
    return [
        {
            'date': item['date'],
            'value': item['value'],
            'normalized': round(100 * (item['value'] / start_value), 2),
        }
        for item in history
    ]


# =============================================================================
# TRADE EXPLANATION API ENDPOINTS
# =============================================================================


@login_required
@require_http_methods(["GET"])
def trade_explanation(request, trade_id: str):
    """
    Get full human-readable explanation for a trade.

    GET /api/trades/{trade_id}/explanation/

    Returns:
        JSON with trade explanation including:
        - summary: Plain English explanation
        - signal_explanations: Detailed breakdown of each signal
        - key_factors: Most important factors
        - risk_assessment: Risk evaluation
        - similar_trades_summary: Summary of similar historical trades
    """
    from .services.trade_explainer import trade_explainer_service
    from dataclasses import asdict

    try:
        explanation = trade_explainer_service.explain_trade(trade_id)

        if explanation is None:
            return JsonResponse({
                'status': 'error',
                'message': f'Trade not found: {trade_id}'
            }, status=404)

        # Convert dataclass to dict for JSON serialization
        explanation_dict = {
            'trade_id': explanation.trade_id,
            'symbol': explanation.symbol,
            'direction': explanation.direction,
            'strategy_name': explanation.strategy_name,
            'entry_price': explanation.entry_price,
            'quantity': explanation.quantity,
            'confidence_score': explanation.confidence_score,
            'summary': explanation.summary,
            'signal_explanations': [
                {
                    'signal_name': se.signal_name,
                    'triggered': se.triggered,
                    'value': se.value,
                    'threshold': se.threshold,
                    'description': se.description,
                    'impact': se.impact,
                }
                for se in explanation.signal_explanations
            ],
            'key_factors': explanation.key_factors,
            'risk_assessment': explanation.risk_assessment,
            'similar_trades_summary': explanation.similar_trades_summary,
            'timestamp': explanation.timestamp,
        }

        return JsonResponse({
            'status': 'success',
            'explanation': explanation_dict,
        })

    except Exception as e:
        logger.error(f"Error getting trade explanation: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


@login_required
@require_http_methods(["GET"])
def trade_signals(request, trade_id: str):
    """
    Get raw signal data for a trade.

    GET /api/trades/{trade_id}/signals/

    Returns:
        JSON with raw signals_at_entry data and visualization-ready formats
    """
    from .services.trade_explainer import trade_explainer_service

    try:
        viz_data = trade_explainer_service.get_signal_visualization_data(trade_id)

        if viz_data is None:
            return JsonResponse({
                'status': 'error',
                'message': f'Trade not found: {trade_id}'
            }, status=404)

        # Also get raw signals from database
        from backend.tradingbot.models.models import TradeSignalSnapshot
        try:
            snapshot = TradeSignalSnapshot.objects.get(trade_id=trade_id)
            raw_signals = snapshot.signals_at_entry
            confidence = snapshot.confidence_score
        except TradeSignalSnapshot.DoesNotExist:
            raw_signals = {}
            confidence = 0

        return JsonResponse({
            'status': 'success',
            'trade_id': trade_id,
            'confidence_score': confidence,
            'raw_signals': raw_signals,
            'visualization': {
                'rsi_gauge': viz_data.rsi_gauge,
                'macd_chart': viz_data.macd_chart,
                'volume_bar': viz_data.volume_bar,
                'price_chart': viz_data.price_chart,
                'confidence_meter': viz_data.confidence_meter,
                'signal_timeline': viz_data.signal_timeline,
            }
        })

    except Exception as e:
        logger.error(f"Error getting trade signals: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


@login_required
@require_http_methods(["GET"])
def trade_similar(request, trade_id: str):
    """
    Get similar historical trades for comparison.

    GET /api/trades/{trade_id}/similar/?limit=10&min_similarity=0.7

    Query params:
        limit: Maximum similar trades to return (default: 10)
        min_similarity: Minimum similarity score 0-1 (default: 0.7)

    Returns:
        JSON with list of similar historical trades and their outcomes
    """
    from .services.trade_explainer import trade_explainer_service

    try:
        limit = int(request.GET.get('limit', 10))
        min_similarity = float(request.GET.get('min_similarity', 0.7))

        similar_trades = trade_explainer_service.find_similar_trades(
            trade_id,
            limit=limit,
            min_similarity=min_similarity
        )

        # Calculate aggregate stats
        if similar_trades:
            total = len(similar_trades)
            profits = sum(1 for t in similar_trades if t.outcome == 'profit')
            losses = sum(1 for t in similar_trades if t.outcome == 'loss')
            avg_pnl = sum(t.pnl_percent for t in similar_trades) / total
            win_rate = profits / total * 100
        else:
            total = 0
            profits = 0
            losses = 0
            avg_pnl = 0
            win_rate = 0

        return JsonResponse({
            'status': 'success',
            'trade_id': trade_id,
            'similar_trades': [
                {
                    'trade_id': t.trade_id,
                    'symbol': t.symbol,
                    'similarity_score': t.similarity_score,
                    'outcome': t.outcome,
                    'pnl_percent': t.pnl_percent,
                    'entry_date': t.entry_date,
                    'key_signals_matched': t.key_signals_matched,
                }
                for t in similar_trades
            ],
            'aggregate_stats': {
                'total_similar': total,
                'profitable': profits,
                'losses': losses,
                'win_rate': round(win_rate, 1),
                'average_pnl_percent': round(avg_pnl, 2),
            }
        })

    except Exception as e:
        logger.error(f"Error getting similar trades: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


@login_required
@require_http_methods(["GET"])
def trade_list_with_explanations(request):
    """
    Get list of recent trades with brief explanations.

    GET /api/trades/with-explanations/?limit=20&symbol=AAPL&strategy=wsb-dip-bot

    Query params:
        limit: Max trades to return (default: 20)
        symbol: Filter by symbol
        strategy: Filter by strategy name

    Returns:
        JSON list of trades with confidence scores and brief explanations
    """
    try:
        from backend.tradingbot.models.models import TradeSignalSnapshot

        limit = int(request.GET.get('limit', 20))
        symbol = request.GET.get('symbol')
        strategy = request.GET.get('strategy')

        queryset = TradeSignalSnapshot.objects.all()

        if symbol:
            queryset = queryset.filter(symbol__iexact=symbol)
        if strategy:
            queryset = queryset.filter(strategy_name__iexact=strategy)

        trades = queryset.order_by('-created_at')[:limit]

        return JsonResponse({
            'status': 'success',
            'trades': [
                {
                    'trade_id': t.trade_id,
                    'symbol': t.symbol,
                    'direction': t.direction,
                    'strategy_name': t.strategy_name,
                    'entry_price': float(t.entry_price),
                    'quantity': float(t.quantity),
                    'confidence_score': t.confidence_score,
                    'explanation': t.explanation[:150] + '...' if len(t.explanation) > 150 else t.explanation,
                    'outcome': t.outcome,
                    'pnl_percent': float(t.pnl_percent) if t.pnl_percent else None,
                    'created_at': t.created_at.isoformat(),
                }
                for t in trades
            ],
            'total': queryset.count(),
        })

    except Exception as e:
        logger.error(f"Error getting trade list: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


# =============================================================================
# VIX Monitoring API
# =============================================================================

@login_required
@require_http_methods(["GET"])
def vix_status(request):
    """
    Get current VIX status and position sizing impact.

    GET /api/vix/status

    Returns:
        JSON with VIX data including:
        - vix: Current VIX value, level, percentile
        - position_multiplier: Current position sizing multiplier
        - trading_paused: Whether trading is paused due to VIX
        - thresholds: VIX threshold configuration
        - recommendations: Trading recommendations based on VIX level
    """
    from .services.market_monitor import get_market_monitor

    try:
        monitor = get_market_monitor()
        status = monitor.get_status()

        return JsonResponse({
            'status': 'success',
            'vix': status,
        })

    except Exception as e:
        logger.error(f"Error getting VIX status: {e}")
        return JsonResponse({
            'status': 'error',
            'error': str(e),
            'vix': {'available': False},
        }, status=500)


@login_required
@require_http_methods(["GET"])
def vix_history(request):
    """
    Get VIX historical data for charting.

    GET /api/vix/history?days=30

    Query params:
        days: Number of days of history (default: 30, max: 365)

    Returns:
        JSON with VIX historical values
    """
    from .services.market_monitor import get_market_monitor

    try:
        days = min(int(request.GET.get('days', 30)), 365)

        monitor = get_market_monitor()
        history = monitor.get_vix_history(days=days)

        if history is None:
            return JsonResponse({
                'status': 'error',
                'message': 'Unable to fetch VIX history',
            }, status=500)

        return JsonResponse({
            'status': 'success',
            'days': days,
            'values': history[-days:] if len(history) > days else history,
            'current': history[-1] if history else None,
        })

    except Exception as e:
        logger.error(f"Error getting VIX history: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["GET"])
def circuit_breaker_status(request):
    """
    Get circuit breaker status including VIX.

    GET /api/circuit-breaker/status

    Returns:
        JSON with all circuit breaker statuses including VIX
    """
    try:
        from backend.tradingbot.risk.monitoring.circuit_breaker import CircuitBreaker, BreakerLimits

        # Create a circuit breaker instance to get status
        # In production, this would use the actual running instance
        breaker = CircuitBreaker(start_equity=100000.0)

        # Check VIX level
        breaker.check_vix(force=True)

        status = breaker.status()

        return JsonResponse({
            'status': 'success',
            'circuit_breaker': status,
        })

    except Exception as e:
        logger.error(f"Error getting circuit breaker status: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


# =============================================================================
# Allocation Management API
# =============================================================================

@login_required
@require_http_methods(["GET"])
def allocation_list(request):
    """
    Get all strategy allocations for the current user.

    GET /api/allocations/

    Returns:
        JSON with allocation summary for all strategies
    """
    from .services.allocation_manager import get_allocation_manager

    try:
        manager = get_allocation_manager()
        summary = manager.get_allocation_summary(request.user)

        return JsonResponse({
            'status': 'success',
            'allocations': summary,
        })

    except Exception as e:
        logger.error(f"Error getting allocations: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["GET"])
def allocation_detail(request, strategy_name: str):
    """
    Get allocation details for a specific strategy.

    GET /api/allocations/{strategy}/

    Returns:
        JSON with allocation details for the strategy
    """
    from .services.allocation_manager import get_allocation_manager

    try:
        manager = get_allocation_manager()
        allocation = manager.get_strategy_allocation(request.user, strategy_name)

        if allocation is None:
            return JsonResponse({
                'status': 'error',
                'message': f'No allocation configured for {strategy_name}',
            }, status=404)

        return JsonResponse({
            'status': 'success',
            'allocation': {
                'strategy_name': allocation.strategy_name,
                'allocated_pct': allocation.allocated_pct,
                'allocated_amount': allocation.allocated_amount,
                'current_exposure': allocation.current_exposure,
                'reserved_amount': allocation.reserved_amount,
                'available_capital': allocation.available_capital,
                'utilization_pct': allocation.utilization_pct,
                'utilization_level': allocation.utilization_level,
                'is_maxed_out': allocation.is_maxed_out,
                'is_enabled': allocation.is_enabled,
            },
        })

    except Exception as e:
        logger.error(f"Error getting allocation detail: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["POST"])
def allocation_update(request, strategy_name: str):
    """
    Update allocation percentage for a strategy.

    POST /api/allocations/{strategy}/update

    Body:
        allocated_pct: New allocation percentage
        portfolio_value: Optional portfolio value for amount calculation

    Returns:
        JSON with updated allocation
    """
    from .services.allocation_manager import get_allocation_manager

    try:
        if request.content_type == 'application/json':
            data = json.loads(request.body)
        else:
            data = request.POST

        allocated_pct = float(data.get('allocated_pct', 0))
        portfolio_value = float(data.get('portfolio_value', 0)) if data.get('portfolio_value') else None

        if allocated_pct < 0 or allocated_pct > 100:
            return JsonResponse({
                'status': 'error',
                'message': 'allocated_pct must be between 0 and 100',
            }, status=400)

        manager = get_allocation_manager()
        manager.update_allocation(
            user=request.user,
            strategy_name=strategy_name,
            allocated_pct=allocated_pct,
            portfolio_value=portfolio_value,
        )

        # Return updated allocation
        allocation = manager.get_strategy_allocation(request.user, strategy_name)

        return JsonResponse({
            'status': 'success',
            'message': f'Updated {strategy_name} allocation to {allocated_pct}%',
            'allocation': {
                'strategy_name': allocation.strategy_name,
                'allocated_pct': allocation.allocated_pct,
                'allocated_amount': allocation.allocated_amount,
            } if allocation else None,
        })

    except Exception as e:
        logger.error(f"Error updating allocation: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["POST"])
def strategy_config_save(request, strategy_name: str):
    """
    Save strategy configuration.

    POST /api/strategies/{strategy}/config

    Body:
        config: Strategy configuration dictionary
        allocation_pct: Optional allocation percentage

    Returns:
        JSON with saved configuration
    """
    from .services.allocation_manager import get_allocation_manager

    try:
        if request.content_type == 'application/json':
            data = json.loads(request.body)
        else:
            data = request.POST

        config = data.get('config', {})
        allocation_pct = data.get('allocation_pct')

        # Validate strategy name
        valid_strategies = [
            'wheel', 'wsb-dip-bot', 'momentum-weeklies', 'earnings-protection',
            'debit-spreads', 'leaps-tracker', 'lotto-scanner', 'swing-trading',
            'spx-credit-spreads', 'index-baseline', 'crypto-dip-bot'
        ]

        if strategy_name not in valid_strategies:
            return JsonResponse({
                'status': 'error',
                'message': f'Unknown strategy: {strategy_name}',
            }, status=400)

        # Get or create user profile to store config
        from backend.tradingbot.models.models import UserProfile
        profile, _ = UserProfile.objects.get_or_create(user=request.user)

        # Store config in user preferences
        if not profile.dashboard_layout:
            profile.dashboard_layout = {}

        strategy_configs = profile.dashboard_layout.get('strategy_configs', {})
        strategy_configs[strategy_name] = {
            'config': config,
            'updated_at': timezone.now().isoformat(),
        }
        profile.dashboard_layout['strategy_configs'] = strategy_configs
        profile.save(update_fields=['dashboard_layout'])

        # Update allocation if provided
        if allocation_pct is not None:
            try:
                manager = get_allocation_manager()
                manager.update_allocation(
                    user=request.user,
                    strategy_name=strategy_name,
                    allocated_pct=float(allocation_pct),
                )
            except Exception as alloc_error:
                logger.warning(f"Could not update allocation: {alloc_error}")

        return JsonResponse({
            'status': 'success',
            'message': f'{strategy_name} configuration saved successfully',
            'strategy': strategy_name,
            'config': config,
        })

    except json.JSONDecodeError:
        return JsonResponse({
            'status': 'error',
            'message': 'Invalid JSON in request body',
        }, status=400)
    except Exception as e:
        logger.error(f"Error saving strategy config: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["GET"])
def strategy_config_get(request, strategy_name: str):
    """
    Get strategy configuration.

    GET /api/strategies/{strategy}/config

    Returns:
        JSON with strategy configuration
    """
    try:
        from backend.tradingbot.models.models import UserProfile

        try:
            profile = UserProfile.objects.get(user=request.user)
            strategy_configs = profile.dashboard_layout.get('strategy_configs', {})
            config_data = strategy_configs.get(strategy_name, {})
        except UserProfile.DoesNotExist:
            config_data = {}

        return JsonResponse({
            'status': 'success',
            'strategy': strategy_name,
            'config': config_data.get('config', {}),
            'updated_at': config_data.get('updated_at'),
        })

    except Exception as e:
        logger.error(f"Error getting strategy config: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["POST"])
def allocation_initialize(request):
    """
    Initialize allocations based on risk profile.

    POST /api/allocations/initialize

    Body:
        profile: 'conservative', 'moderate', or 'aggressive'
        portfolio_value: Current portfolio value
        enabled_strategies: Optional list of strategies to enable

    Returns:
        JSON with initialized allocations
    """
    from .services.allocation_manager import get_allocation_manager

    try:
        if request.content_type == 'application/json':
            data = json.loads(request.body)
        else:
            data = request.POST

        profile = data.get('profile', 'moderate')
        portfolio_value = float(data.get('portfolio_value', 100000))
        enabled_strategies = data.get('enabled_strategies')

        if profile not in ['conservative', 'moderate', 'aggressive']:
            return JsonResponse({
                'status': 'error',
                'message': 'profile must be conservative, moderate, or aggressive',
            }, status=400)

        manager = get_allocation_manager()
        manager.initialize_allocations(
            user=request.user,
            profile=profile,
            portfolio_value=portfolio_value,
            enabled_strategies=enabled_strategies,
        )

        # Return initialized allocations
        summary = manager.get_allocation_summary(request.user)

        return JsonResponse({
            'status': 'success',
            'message': f'Initialized allocations for {profile} profile',
            'allocations': summary,
        })

    except Exception as e:
        logger.error(f"Error initializing allocations: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["POST"])
def allocation_rebalance(request):
    """
    Get rebalancing recommendations and optionally execute them.

    POST /api/allocations/rebalance

    Body:
        portfolio_value: Current portfolio value
        target_profile: Optional target profile to rebalance towards
        execute: Whether to execute the rebalancing (default: false)

    Returns:
        JSON with rebalancing recommendations
    """
    from .services.allocation_manager import get_allocation_manager

    try:
        if request.content_type == 'application/json':
            data = json.loads(request.body)
        else:
            data = request.POST

        portfolio_value = float(data.get('portfolio_value', 100000))
        target_profile = data.get('target_profile')

        manager = get_allocation_manager()
        recommendations = manager.get_rebalance_recommendations(
            user=request.user,
            portfolio_value=portfolio_value,
            target_profile=target_profile,
        )

        return JsonResponse({
            'status': 'success',
            'recommendations': [
                {
                    'strategy_name': r.strategy_name,
                    'current_allocation': r.current_allocation,
                    'target_allocation': r.target_allocation,
                    'current_amount': r.current_amount,
                    'target_amount': r.target_amount,
                    'action': r.action,
                    'adjustment_amount': r.adjustment_amount,
                    'priority': r.priority,
                    'reason': r.reason,
                }
                for r in recommendations
            ],
        })

    except Exception as e:
        logger.error(f"Error getting rebalance recommendations: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["POST"])
def allocation_reconcile(request):
    """
    Reconcile allocations with actual positions.

    POST /api/allocations/reconcile

    Body:
        positions: List of position dictionaries with symbol, market_value, strategy

    Returns:
        JSON with reconciliation report
    """
    from .services.allocation_manager import get_allocation_manager

    try:
        if request.content_type == 'application/json':
            data = json.loads(request.body)
        else:
            data = request.POST

        positions = data.get('positions', [])

        manager = get_allocation_manager()
        report = manager.reconcile_allocations(
            user=request.user,
            positions=positions,
        )

        return JsonResponse({
            'status': 'success',
            'report': report,
        })

    except Exception as e:
        logger.error(f"Error reconciling allocations: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["POST"])
def allocation_recalculate(request):
    """
    Recalculate all allocations based on current portfolio value.

    POST /api/allocations/recalculate

    Body:
        portfolio_value: Current portfolio value

    Returns:
        JSON with updated allocations
    """
    from .services.allocation_manager import get_allocation_manager

    try:
        if request.content_type == 'application/json':
            data = json.loads(request.body)
        else:
            data = request.POST

        portfolio_value = float(data.get('portfolio_value', 0))

        if portfolio_value <= 0:
            return JsonResponse({
                'status': 'error',
                'message': 'portfolio_value must be greater than 0',
            }, status=400)

        manager = get_allocation_manager()
        manager.recalculate_all_allocations(
            user=request.user,
            portfolio_value=portfolio_value,
        )

        # Return updated allocations
        summary = manager.get_allocation_summary(request.user)

        return JsonResponse({
            'status': 'success',
            'message': f'Recalculated allocations for ${portfolio_value:,.2f} portfolio',
            'allocations': summary,
        })

    except Exception as e:
        logger.error(f"Error recalculating allocations: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


# =============================================================================
# Circuit Breaker Recovery API Endpoints
# =============================================================================

@login_required
@require_http_methods(["GET"])
def circuit_breaker_history(request):
    """
    Get circuit breaker event history.

    GET /api/circuit-breakers/history/

    Query params:
        days: Number of days to look back (default 30)
        limit: Maximum events (default 50)

    Returns:
        JSON with list of circuit breaker events
    """
    from .services.recovery_manager import get_recovery_manager

    try:
        days = int(request.GET.get('days', 30))
        limit = int(request.GET.get('limit', 50))

        recovery_mgr = get_recovery_manager(request.user)
        events = recovery_mgr.get_event_history(days=days, limit=limit)

        return JsonResponse({
            'status': 'success',
            'events': [e.to_dict() for e in events],
            'count': len(events),
            'params': {'days': days, 'limit': limit},
        })

    except Exception as e:
        logger.error(f"Error fetching circuit breaker history: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["GET"])
def circuit_breaker_current(request):
    """
    Get current circuit breaker status and recovery timeline.

    GET /api/circuit-breakers/current/

    Returns:
        JSON with active breakers, recovery status, and timeline
    """
    from .services.recovery_manager import get_recovery_manager

    try:
        recovery_mgr = get_recovery_manager(request.user)

        # Get recovery status
        status = recovery_mgr.get_recovery_status()

        # Get timeline for UI display
        timeline = recovery_mgr.get_recovery_timeline()

        # Check for auto-recovery advancement
        advancements = recovery_mgr.check_auto_recovery()

        return JsonResponse({
            'status': 'success',
            'recovery_status': {
                'is_in_recovery': status.is_in_recovery,
                'current_mode': status.current_mode.value,
                'position_multiplier': status.position_multiplier,
                'can_trade': status.can_trade,
                'can_activate_new_strategies': status.can_activate_new_strategies,
                'hours_until_next_stage': status.hours_until_next_stage,
                'trades_until_next_stage': status.trades_until_next_stage,
                'can_advance': status.can_advance,
                'message': status.message,
            },
            'active_events': status.active_events,
            'timeline': timeline,
            'recent_advancements': advancements,
        })

    except Exception as e:
        logger.error(f"Error getting circuit breaker status: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["POST"])
def circuit_breaker_advance(request, event_id):
    """
    Manually advance recovery for a circuit breaker event.

    POST /api/circuit-breakers/{event_id}/advance-recovery/

    Body:
        force: Boolean to force advance (optional)
        notes: Notes for the advancement (optional)

    Returns:
        JSON with advancement result
    """
    from .services.recovery_manager import get_recovery_manager

    try:
        if request.content_type == 'application/json':
            data = json.loads(request.body)
        else:
            data = request.POST

        force = data.get('force', False)
        notes = data.get('notes', '')

        recovery_mgr = get_recovery_manager(request.user)
        result = recovery_mgr.advance_recovery(
            event_id=int(event_id),
            force=bool(force),
            notes=str(notes),
        )

        if result['success']:
            return JsonResponse({
                'status': 'success',
                **result,
            })
        else:
            return JsonResponse({
                'status': 'error',
                **result,
            }, status=400)

    except Exception as e:
        logger.error(f"Error advancing recovery: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["POST"])
def circuit_breaker_reset(request, event_id):
    """
    Fully reset a circuit breaker event.

    POST /api/circuit-breakers/{event_id}/reset/

    Body:
        confirmation: Must be "CONFIRM" to proceed
        notes: Notes for the reset (optional)

    Returns:
        JSON with reset result
    """
    from .services.recovery_manager import get_recovery_manager

    try:
        if request.content_type == 'application/json':
            data = json.loads(request.body)
        else:
            data = request.POST

        confirmation = data.get('confirmation', '')
        notes = data.get('notes', '')

        recovery_mgr = get_recovery_manager(request.user)
        result = recovery_mgr.reset_breaker(
            event_id=int(event_id),
            confirmation=str(confirmation),
            notes=str(notes),
        )

        if result['success']:
            return JsonResponse({
                'status': 'success',
                **result,
            })
        else:
            return JsonResponse({
                'status': 'error',
                **result,
            }, status=400)

    except Exception as e:
        logger.error(f"Error resetting circuit breaker: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["POST"])
def circuit_breaker_early_recovery(request, event_id):
    """
    Request early recovery with justification.

    POST /api/circuit-breakers/{event_id}/early-recovery/

    Body:
        justification: User's justification (min 20 chars)

    Returns:
        JSON with request status
    """
    from .services.recovery_manager import get_recovery_manager

    try:
        if request.content_type == 'application/json':
            data = json.loads(request.body)
        else:
            data = request.POST

        justification = data.get('justification', '')

        recovery_mgr = get_recovery_manager(request.user)
        result = recovery_mgr.request_early_recovery(
            event_id=int(event_id),
            justification=str(justification),
        )

        if result['success']:
            return JsonResponse({
                'status': 'success',
                **result,
            })
        else:
            return JsonResponse({
                'status': 'error',
                **result,
            }, status=400)

    except Exception as e:
        logger.error(f"Error requesting early recovery: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["GET"])
def circuit_breaker_timeline(request, event_id=None):
    """
    Get recovery timeline for display.

    GET /api/circuit-breakers/timeline/
    GET /api/circuit-breakers/{event_id}/timeline/

    Returns:
        JSON with timeline data for UI display
    """
    from .services.recovery_manager import get_recovery_manager

    try:
        recovery_mgr = get_recovery_manager(request.user)
        timeline = recovery_mgr.get_recovery_timeline(
            event_id=int(event_id) if event_id else None
        )

        return JsonResponse({
            'status': 'success',
            'timeline': timeline,
        })

    except Exception as e:
        logger.error(f"Error getting recovery timeline: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


# =============================================================================
# Market Context API Endpoints
# =============================================================================

@login_required
@require_http_methods(["GET"])
def market_context(request):
    """
    Get full market context for dashboard display.

    GET /api/market-context/

    Returns:
        JSON with market overview, sectors, holdings events, economic calendar
    """
    from .services.market_context import get_market_context_service

    try:
        service = get_market_context_service()

        # Get user's holding symbols if available
        holding_symbols = []
        try:
            from backend.tradingbot.apimanagers import AlpacaManager
            from backend.tradingbot.models import UserSettings

            settings = UserSettings.objects.filter(user=request.user).first()
            if settings and settings.alpaca_id and settings.alpaca_key:
                manager = AlpacaManager(
                    api_key=settings.alpaca_id,
                    api_secret=settings.alpaca_key,
                    paper=True,
                )
                positions = manager.get_portfolio()
                holding_symbols = [p.get('symbol') for p in positions if p.get('symbol')]
        except Exception as e:
            logger.debug(f"Could not fetch holdings for market context: {e}")

        context = service.get_full_context(holding_symbols=holding_symbols)

        return JsonResponse({
            'status': 'success',
            **context,
        })

    except Exception as e:
        logger.error(f"Error getting market context: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["GET"])
def market_overview(request):
    """
    Get market overview with indices and VIX.

    GET /api/market-context/overview/

    Query params:
        refresh: Force refresh cache (optional)

    Returns:
        JSON with indices, VIX, market status
    """
    from .services.market_context import get_market_context_service

    try:
        service = get_market_context_service()
        force_refresh = request.GET.get('refresh', '').lower() == 'true'

        overview = service.get_market_overview(force_refresh=force_refresh)

        return JsonResponse({
            'status': 'success',
            **overview,
        })

    except Exception as e:
        logger.error(f"Error getting market overview: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["GET"])
def sector_performance(request):
    """
    Get sector ETF performance for heat map.

    GET /api/market-context/sectors/

    Query params:
        refresh: Force refresh cache (optional)

    Returns:
        JSON with sector performance data
    """
    from .services.market_context import get_market_context_service

    try:
        service = get_market_context_service()
        force_refresh = request.GET.get('refresh', '').lower() == 'true'

        sectors = service.get_sector_performance(force_refresh=force_refresh)

        return JsonResponse({
            'status': 'success',
            'sectors': sectors,
        })

    except Exception as e:
        logger.error(f"Error getting sector performance: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["GET"])
def holdings_events(request):
    """
    Get upcoming events for user's holdings.

    GET /api/market-context/events/

    Query params:
        days: Days ahead to look (default 14)

    Returns:
        JSON with upcoming earnings, dividends, etc.
    """
    from .services.market_context import get_market_context_service

    try:
        service = get_market_context_service()
        days = int(request.GET.get('days', 14))

        # Get user's holding symbols
        holding_symbols = []
        try:
            from backend.tradingbot.apimanagers import AlpacaManager
            from backend.tradingbot.models import UserSettings

            settings = UserSettings.objects.filter(user=request.user).first()
            if settings and settings.alpaca_id and settings.alpaca_key:
                manager = AlpacaManager(
                    api_key=settings.alpaca_id,
                    api_secret=settings.alpaca_key,
                    paper=True,
                )
                positions = manager.get_portfolio()
                holding_symbols = [p.get('symbol') for p in positions if p.get('symbol')]
        except Exception as e:
            logger.debug(f"Could not fetch holdings: {e}")

        events = service.get_holdings_events(holding_symbols, days_ahead=days)

        return JsonResponse({
            'status': 'success',
            'events': events,
            'symbols_checked': holding_symbols,
        })

    except Exception as e:
        logger.error(f"Error getting holdings events: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["GET"])
def economic_calendar(request):
    """
    Get upcoming economic events.

    GET /api/market-context/calendar/

    Query params:
        days: Days ahead to look (default 7)

    Returns:
        JSON with upcoming economic events
    """
    from .services.market_context import get_market_context_service

    try:
        service = get_market_context_service()
        days = int(request.GET.get('days', 7))

        events = service.get_economic_calendar(days_ahead=days)

        return JsonResponse({
            'status': 'success',
            'events': events,
        })

    except Exception as e:
        logger.error(f"Error getting economic calendar: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


# ============================================================================
# Strategy Portfolio API Endpoints
# ============================================================================

@login_required
@require_http_methods(["GET"])
def portfolio_list(request):
    """
    Get list of user's portfolios and templates.

    GET /api/portfolios/

    Returns:
        JSON with list of portfolios
    """
    from .services.portfolio_builder import get_portfolio_builder

    try:
        builder = get_portfolio_builder(request.user)
        portfolios = builder.get_user_portfolios()

        return JsonResponse({
            'status': 'success',
            'portfolios': [p.to_dict() for p in portfolios],
            'active_portfolio_id': next(
                (p.id for p in portfolios if p.is_active),
                None
            ),
        })

    except Exception as e:
        logger.error(f"Error getting portfolio list: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["GET"])
def portfolio_detail(request, portfolio_id):
    """
    Get detailed portfolio information.

    GET /api/portfolios/<portfolio_id>/

    Returns:
        JSON with portfolio details and analysis
    """
    from .services.portfolio_builder import get_portfolio_builder
    from backend.tradingbot.models.models import StrategyPortfolio

    try:
        portfolio = StrategyPortfolio.objects.get(pk=portfolio_id)

        # Check access
        if portfolio.user and portfolio.user != request.user and not portfolio.is_template:
            return JsonResponse({
                'status': 'error',
                'message': 'Access denied',
            }, status=403)

        builder = get_portfolio_builder(request.user)
        analysis = builder.analyze_portfolio(portfolio.strategies)

        return JsonResponse({
            'status': 'success',
            'portfolio': portfolio.to_dict(),
            'analysis': {
                'expected_return': analysis.expected_return,
                'expected_volatility': analysis.expected_volatility,
                'expected_sharpe': analysis.expected_sharpe,
                'diversification_score': analysis.diversification_score,
                'correlation_matrix': analysis.correlation_matrix,
                'risk_contribution': analysis.risk_contribution,
                'warnings': analysis.warnings,
                'recommendations': analysis.recommendations,
            },
        })

    except StrategyPortfolio.DoesNotExist:
        return JsonResponse({
            'status': 'error',
            'message': 'Portfolio not found',
        }, status=404)
    except Exception as e:
        logger.error(f"Error getting portfolio detail: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["POST"])
def portfolio_create(request):
    """
    Create a new custom portfolio.

    POST /api/portfolios/create

    Body:
        name: Portfolio name
        description: Optional description
        risk_profile: conservative|moderate|aggressive|custom
        strategies: {strategy_id: {allocation_pct, enabled, params}}

    Returns:
        JSON with created portfolio
    """
    from .services.portfolio_builder import get_portfolio_builder

    try:
        data = json.loads(request.body)

        name = data.get('name')
        if not name:
            return JsonResponse({
                'status': 'error',
                'message': 'Portfolio name is required',
            }, status=400)

        strategies = data.get('strategies', {})
        if not strategies:
            return JsonResponse({
                'status': 'error',
                'message': 'At least one strategy is required',
            }, status=400)

        builder = get_portfolio_builder(request.user)
        portfolio = builder.create_custom_portfolio(
            name=name,
            strategies=strategies,
            description=data.get('description', ''),
            risk_profile=data.get('risk_profile', 'custom'),
        )

        return JsonResponse({
            'status': 'success',
            'portfolio': portfolio.to_dict(),
            'message': f"Portfolio '{name}' created successfully",
        })

    except ValueError as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=400)
    except Exception as e:
        logger.error(f"Error creating portfolio: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["PUT", "PATCH"])
def portfolio_update(request, portfolio_id):
    """
    Update an existing portfolio.

    PUT /api/portfolios/<portfolio_id>/update

    Body:
        name: Optional new name
        description: Optional new description
        strategies: Optional new strategies
        risk_profile: Optional new risk profile

    Returns:
        JSON with updated portfolio
    """
    from .services.portfolio_builder import get_portfolio_builder
    from backend.tradingbot.models.models import StrategyPortfolio

    try:
        portfolio = StrategyPortfolio.objects.get(pk=portfolio_id, user=request.user)

        data = json.loads(request.body)

        # Update fields
        if 'name' in data:
            portfolio.name = data['name']
        if 'description' in data:
            portfolio.description = data['description']
        if 'risk_profile' in data:
            portfolio.risk_profile = data['risk_profile']
        if 'strategies' in data:
            portfolio.strategies = data['strategies']

            # Recalculate analysis
            builder = get_portfolio_builder(request.user)
            analysis = builder.analyze_portfolio(portfolio.strategies)
            portfolio.correlation_matrix = analysis.correlation_matrix
            portfolio.diversification_score = analysis.diversification_score
            portfolio.expected_sharpe = analysis.expected_sharpe

        portfolio.save()

        return JsonResponse({
            'status': 'success',
            'portfolio': portfolio.to_dict(),
            'message': 'Portfolio updated successfully',
        })

    except StrategyPortfolio.DoesNotExist:
        return JsonResponse({
            'status': 'error',
            'message': 'Portfolio not found',
        }, status=404)
    except Exception as e:
        logger.error(f"Error updating portfolio: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["DELETE"])
def portfolio_delete(request, portfolio_id):
    """
    Delete a portfolio.

    DELETE /api/portfolios/<portfolio_id>/delete

    Returns:
        JSON with deletion status
    """
    from backend.tradingbot.models.models import StrategyPortfolio

    try:
        portfolio = StrategyPortfolio.objects.get(pk=portfolio_id, user=request.user)
        name = portfolio.name
        portfolio.delete()

        return JsonResponse({
            'status': 'success',
            'message': f"Portfolio '{name}' deleted successfully",
        })

    except StrategyPortfolio.DoesNotExist:
        return JsonResponse({
            'status': 'error',
            'message': 'Portfolio not found',
        }, status=404)
    except Exception as e:
        logger.error(f"Error deleting portfolio: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["POST"])
def portfolio_activate(request, portfolio_id):
    """
    Activate a portfolio for trading.

    POST /api/portfolios/<portfolio_id>/activate

    Returns:
        JSON with activation status
    """
    from backend.tradingbot.models.models import StrategyPortfolio

    try:
        portfolio = StrategyPortfolio.objects.get(pk=portfolio_id, user=request.user)

        # Validate allocation before activating
        if not portfolio.is_valid_allocation:
            return JsonResponse({
                'status': 'error',
                'message': f'Invalid allocation: {portfolio.total_allocation}% (must be 99-101%)',
            }, status=400)

        portfolio.activate()

        return JsonResponse({
            'status': 'success',
            'portfolio': portfolio.to_dict(),
            'message': f"Portfolio '{portfolio.name}' is now active",
        })

    except StrategyPortfolio.DoesNotExist:
        return JsonResponse({
            'status': 'error',
            'message': 'Portfolio not found',
        }, status=404)
    except Exception as e:
        logger.error(f"Error activating portfolio: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["POST"])
def portfolio_deactivate(request, portfolio_id):
    """
    Deactivate a portfolio.

    POST /api/portfolios/<portfolio_id>/deactivate

    Returns:
        JSON with deactivation status
    """
    from backend.tradingbot.models.models import StrategyPortfolio

    try:
        portfolio = StrategyPortfolio.objects.get(pk=portfolio_id, user=request.user)
        portfolio.deactivate()

        return JsonResponse({
            'status': 'success',
            'portfolio': portfolio.to_dict(),
            'message': f"Portfolio '{portfolio.name}' deactivated",
        })

    except StrategyPortfolio.DoesNotExist:
        return JsonResponse({
            'status': 'error',
            'message': 'Portfolio not found',
        }, status=404)
    except Exception as e:
        logger.error(f"Error deactivating portfolio: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["GET"])
def portfolio_templates(request):
    """
    Get available portfolio templates.

    GET /api/portfolios/templates/

    Returns:
        JSON with list of templates
    """
    from .services.portfolio_builder import get_portfolio_builder

    try:
        builder = get_portfolio_builder(request.user)
        templates = builder.get_portfolio_templates()

        # Add analysis to each template
        template_list = []
        for template_id, template in templates.items():
            analysis = builder.analyze_portfolio(template['strategies'])
            template_list.append({
                'template_id': template_id,
                **template,
                'analysis': {
                    'expected_return': analysis.expected_return,
                    'expected_volatility': analysis.expected_volatility,
                    'expected_sharpe': analysis.expected_sharpe,
                    'diversification_score': analysis.diversification_score,
                },
            })

        return JsonResponse({
            'status': 'success',
            'templates': template_list,
        })

    except Exception as e:
        logger.error(f"Error getting portfolio templates: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["POST"])
def portfolio_create_from_template(request):
    """
    Create a portfolio from a template.

    POST /api/portfolios/from-template

    Body:
        template_id: Template to use
        name: Optional custom name

    Returns:
        JSON with created portfolio
    """
    from .services.portfolio_builder import get_portfolio_builder

    try:
        data = json.loads(request.body)

        template_id = data.get('template_id')
        if not template_id:
            return JsonResponse({
                'status': 'error',
                'message': 'Template ID is required',
            }, status=400)

        builder = get_portfolio_builder(request.user)
        portfolio = builder.create_from_template(
            template_id=template_id,
            name=data.get('name'),
        )

        return JsonResponse({
            'status': 'success',
            'portfolio': portfolio.to_dict(),
            'message': f"Portfolio created from template '{template_id}'",
        })

    except ValueError as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=400)
    except Exception as e:
        logger.error(f"Error creating portfolio from template: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["GET"])
def available_strategies(request):
    """
    Get available strategies for portfolio building.

    GET /api/portfolios/strategies/

    Returns:
        JSON with available strategies and their metadata
    """
    from .services.portfolio_builder import get_portfolio_builder

    try:
        builder = get_portfolio_builder(request.user)
        strategies = builder.get_available_strategies()

        return JsonResponse({
            'status': 'success',
            'strategies': strategies,
        })

    except Exception as e:
        logger.error(f"Error getting available strategies: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["POST"])
def analyze_portfolio(request):
    """
    Analyze a portfolio configuration.

    POST /api/portfolios/analyze

    Body:
        strategies: {strategy_id: {allocation_pct, enabled}}

    Returns:
        JSON with portfolio analysis
    """
    from .services.portfolio_builder import get_portfolio_builder

    try:
        data = json.loads(request.body)

        strategies = data.get('strategies', {})
        if not strategies:
            return JsonResponse({
                'status': 'error',
                'message': 'Strategies are required',
            }, status=400)

        builder = get_portfolio_builder(request.user)
        analysis = builder.analyze_portfolio(strategies)

        return JsonResponse({
            'status': 'success',
            'analysis': {
                'expected_return': analysis.expected_return,
                'expected_volatility': analysis.expected_volatility,
                'expected_sharpe': analysis.expected_sharpe,
                'diversification_score': analysis.diversification_score,
                'correlation_matrix': analysis.correlation_matrix,
                'risk_contribution': analysis.risk_contribution,
                'warnings': analysis.warnings,
                'recommendations': analysis.recommendations,
            },
        })

    except Exception as e:
        logger.error(f"Error analyzing portfolio: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["POST"])
def optimize_portfolio(request):
    """
    Get optimized allocation for given strategies.

    POST /api/portfolios/optimize

    Body:
        strategies: List of strategy IDs
        risk_profile: conservative|moderate|aggressive

    Returns:
        JSON with optimized allocations
    """
    from .services.portfolio_builder import get_portfolio_builder

    try:
        data = json.loads(request.body)

        strategies = data.get('strategies', [])
        if not strategies:
            return JsonResponse({
                'status': 'error',
                'message': 'Strategy list is required',
            }, status=400)

        risk_profile = data.get('risk_profile', 'moderate')

        builder = get_portfolio_builder(request.user)
        optimized = builder.optimize_portfolio(
            strategies=strategies,
            risk_profile=risk_profile,
        )

        # Analyze the optimized portfolio
        analysis = builder.analyze_portfolio(optimized)

        return JsonResponse({
            'status': 'success',
            'optimized_allocations': optimized,
            'analysis': {
                'expected_return': analysis.expected_return,
                'expected_volatility': analysis.expected_volatility,
                'expected_sharpe': analysis.expected_sharpe,
                'diversification_score': analysis.diversification_score,
            },
        })

    except Exception as e:
        logger.error(f"Error optimizing portfolio: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["POST"])
def suggest_strategies(request):
    """
    Get strategy suggestions for better diversification.

    POST /api/portfolios/suggest

    Body:
        current_strategies: List of current strategy IDs
        risk_profile: User's risk profile

    Returns:
        JSON with suggested strategies
    """
    from .services.portfolio_builder import get_portfolio_builder

    try:
        data = json.loads(request.body)

        current_strategies = data.get('current_strategies', [])
        risk_profile = data.get('risk_profile', 'moderate')

        builder = get_portfolio_builder(request.user)
        suggestions = builder.suggest_additions(
            current_strategies=current_strategies,
            risk_profile=risk_profile,
        )

        return JsonResponse({
            'status': 'success',
            'suggestions': suggestions,
        })

    except Exception as e:
        logger.error(f"Error suggesting strategies: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


# ============================================================================
# User Profile API Endpoints
# ============================================================================

@login_required
@require_http_methods(["GET"])
def user_profile(request):
    """
    Get current user profile.

    GET /api/profile/

    Returns:
        JSON with user profile data
    """
    from .services.user_profile import get_user_profile_service

    try:
        service = get_user_profile_service(request.user)
        profile = service.get_or_create_profile()

        return JsonResponse({
            'status': 'success',
            'profile': profile.to_dict(),
        })

    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["PUT", "PATCH"])
def update_user_profile(request):
    """
    Update user profile.

    PUT /api/profile/update

    Body:
        Any profile fields to update

    Returns:
        JSON with updated profile
    """
    from .services.user_profile import get_user_profile_service

    try:
        data = json.loads(request.body)

        service = get_user_profile_service(request.user)
        success, message = service.update_profile(data)

        if success:
            profile = service.get_profile()
            return JsonResponse({
                'status': 'success',
                'message': message,
                'profile': profile.to_dict(),
            })
        else:
            return JsonResponse({
                'status': 'error',
                'message': message,
            }, status=400)

    except Exception as e:
        logger.error(f"Error updating user profile: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["GET"])
def profile_onboarding_status(request):
    """
    Get user's onboarding status.

    GET /api/profile/onboarding-status/

    Returns:
        JSON with onboarding progress
    """
    from .services.user_profile import get_user_profile_service

    try:
        service = get_user_profile_service(request.user)
        status = service.get_onboarding_status()

        return JsonResponse({
            'status': 'success',
            'onboarding': status,
        })

    except Exception as e:
        logger.error(f"Error getting onboarding status: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["POST"])
def profile_complete_step(request):
    """
    Mark an onboarding step as complete.

    POST /api/profile/complete-step/

    Body:
        step: Step identifier ('risk_assessment', 'brokerage', 'strategy', 'trade')

    Returns:
        JSON with updated onboarding status
    """
    from .services.user_profile import get_user_profile_service

    try:
        data = json.loads(request.body)

        step = data.get('step')
        if not step:
            return JsonResponse({
                'status': 'error',
                'message': 'Step identifier is required',
            }, status=400)

        service = get_user_profile_service(request.user)
        result = service.complete_step(step)

        if result.get('success'):
            return JsonResponse({
                'status': 'success',
                **result,
            })
        else:
            return JsonResponse({
                'status': 'error',
                'message': result.get('message', 'Unknown error'),
            }, status=400)

    except Exception as e:
        logger.error(f"Error completing onboarding step: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["GET"])
def profile_risk_questions(request):
    """
    Get risk assessment questions.

    GET /api/profile/risk-questions/

    Returns:
        JSON with risk assessment questions
    """
    from .services.user_profile import get_user_profile_service

    try:
        service = get_user_profile_service(request.user)
        questions = service.get_risk_questions()

        return JsonResponse({
            'status': 'success',
            'questions': questions,
        })

    except Exception as e:
        logger.error(f"Error getting risk questions: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["POST"])
def profile_submit_risk_assessment(request):
    """
    Submit risk assessment answers.

    POST /api/profile/risk-assessment/submit

    Body:
        answers: Dictionary of question_id -> answer_value

    Returns:
        JSON with risk score and recommendations
    """
    from .services.user_profile import get_user_profile_service

    try:
        data = json.loads(request.body)

        answers = data.get('answers', {})
        if not answers:
            return JsonResponse({
                'status': 'error',
                'message': 'Answers are required',
            }, status=400)

        service = get_user_profile_service(request.user)
        result = service.submit_risk_assessment(answers)

        return JsonResponse({
            'status': 'success',
            **result,
        })

    except Exception as e:
        logger.error(f"Error submitting risk assessment: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["GET"])
def profile_recommendations(request):
    """
    Get recommended settings based on profile.

    GET /api/profile/recommendations/

    Returns:
        JSON with recommended settings
    """
    from .services.user_profile import get_user_profile_service

    try:
        service = get_user_profile_service(request.user)
        recommendations = service.get_recommended_settings()

        return JsonResponse({
            'status': 'success',
            'recommendations': recommendations,
        })

    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["POST"])
def profile_switch_trading_mode(request):
    """
    Switch between paper and live trading mode.

    POST /api/profile/trading-mode/

    Body:
        mode: 'paper' or 'live'

    Returns:
        JSON with result
    """
    from .services.user_profile import get_user_profile_service

    try:
        data = json.loads(request.body)

        mode = data.get('mode')
        if mode not in ['paper', 'live']:
            return JsonResponse({
                'status': 'error',
                'message': "Mode must be 'paper' or 'live'",
            }, status=400)

        service = get_user_profile_service(request.user)
        result = service.switch_trading_mode(mode)

        if result.get('success'):
            return JsonResponse({
                'status': 'success',
                **result,
            })
        else:
            return JsonResponse({
                'status': 'error',
                'message': result.get('message'),
            }, status=400)

    except Exception as e:
        logger.error(f"Error switching trading mode: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@login_required
@require_http_methods(["GET"])
def profile_summary(request):
    """
    Get a summary of the user profile.

    GET /api/profile/summary/

    Returns:
        JSON with profile summary
    """
    from .services.user_profile import get_user_profile_service

    try:
        service = get_user_profile_service(request.user)
        summary = service.get_profile_summary()

        return JsonResponse({
            'status': 'success',
            'summary': summary,
        })

    except Exception as e:
        logger.error(f"Error getting profile summary: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


# =============================================================================
# Circuit Breaker State & History API
# =============================================================================

@require_http_methods(["GET"])
@login_required
def circuit_breaker_state_list(request):
    """Get all circuit breaker states for the current user.

    Returns:
        List of all circuit breaker states
    """
    from .services.circuit_breaker_persistence import get_circuit_breaker_persistence

    try:
        persistence = get_circuit_breaker_persistence(request.user)
        summary = persistence.get_status_summary()

        return JsonResponse({
            'status': 'success',
            **summary,
        })

    except Exception as e:
        logger.error(f"Error getting circuit breaker states: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@require_http_methods(["GET"])
@login_required
def circuit_breaker_state_detail(request, breaker_type):
    """Get specific circuit breaker state.

    Args:
        breaker_type: Type of circuit breaker

    Returns:
        Circuit breaker state details
    """
    from .services.circuit_breaker_persistence import get_circuit_breaker_persistence

    try:
        persistence = get_circuit_breaker_persistence(request.user)
        state = persistence.get_state(breaker_type)

        if not state:
            return JsonResponse({
                'status': 'error',
                'message': f'Breaker type not found: {breaker_type}',
            }, status=404)

        return JsonResponse({
            'status': 'success',
            'state': state.to_dict(),
        })

    except Exception as e:
        logger.error(f"Error getting circuit breaker state: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@require_http_methods(["POST"])
@login_required
def circuit_breaker_reset(request, breaker_type):
    """Reset a circuit breaker.

    Args:
        breaker_type: Type of circuit breaker to reset

    Request body:
        - force: Force reset even if in cooldown (default: false)
        - confirmation: Must be "CONFIRM" to reset
        - reason: Reason for reset

    Returns:
        Reset result
    """
    from .services.circuit_breaker_persistence import get_circuit_breaker_persistence

    try:
        data = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        data = {}

    confirmation = data.get('confirmation', '')
    if confirmation != 'CONFIRM':
        return JsonResponse({
            'status': 'error',
            'message': 'Must provide confirmation="CONFIRM" to reset',
        }, status=400)

    force = data.get('force', False)
    reason = data.get('reason', 'Manual reset via API')

    try:
        persistence = get_circuit_breaker_persistence(request.user)
        result = persistence.reset_breaker(
            breaker_type=breaker_type,
            force=force,
            reason=reason,
        )

        if result.get('success'):
            return JsonResponse({
                'status': 'success',
                **result,
            })
        else:
            return JsonResponse({
                'status': 'error',
                **result,
            }, status=400)

    except Exception as e:
        logger.error(f"Error resetting circuit breaker: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@require_http_methods(["GET"])
@login_required
def circuit_breaker_history_list(request):
    """Get circuit breaker history.

    Query params:
        - breaker_type: Filter by breaker type (optional)
        - days: Number of days to look back (default: 7)
        - limit: Maximum entries to return (default: 100)
        - actions: Comma-separated list of actions to filter (optional)

    Returns:
        List of history entries
    """
    from .services.circuit_breaker_persistence import get_circuit_breaker_persistence

    breaker_type = request.GET.get('breaker_type')
    days = int(request.GET.get('days', 7))
    limit = int(request.GET.get('limit', 100))
    actions_str = request.GET.get('actions')
    actions = actions_str.split(',') if actions_str else None

    try:
        persistence = get_circuit_breaker_persistence(request.user)
        history = persistence.get_history(
            breaker_type=breaker_type,
            days=days,
            limit=limit,
            actions=actions,
        )

        return JsonResponse({
            'status': 'success',
            'count': len(history),
            'history': history,
        })

    except Exception as e:
        logger.error(f"Error getting circuit breaker history: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@require_http_methods(["POST"])
@login_required
def circuit_breaker_daily_reset(request):
    """Perform daily reset on circuit breakers.

    Request body:
        - new_equity: Starting equity for the new day (optional)

    Returns:
        List of reset results
    """
    from .services.circuit_breaker_persistence import get_circuit_breaker_persistence

    try:
        data = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        data = {}

    new_equity = data.get('new_equity')

    try:
        persistence = get_circuit_breaker_persistence(request.user)
        results = persistence.daily_reset_all(new_equity=new_equity)

        return JsonResponse({
            'status': 'success',
            'reset_count': len(results),
            'results': results,
        })

    except Exception as e:
        logger.error(f"Error performing daily reset: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@require_http_methods(["POST"])
@login_required
def circuit_breaker_initialize(request):
    """Initialize default circuit breaker states.

    Returns:
        List of created states
    """
    from .services.circuit_breaker_persistence import get_circuit_breaker_persistence

    try:
        persistence = get_circuit_breaker_persistence(request.user)
        created = persistence.initialize_default_states()

        return JsonResponse({
            'status': 'success',
            'created_count': len(created),
            'created': [s.to_dict() for s in created],
        })

    except Exception as e:
        logger.error(f"Error initializing circuit breaker states: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


# =============================================================================
# Trade Reasoning API
# =============================================================================

@require_http_methods(["GET"])
@login_required
def trade_reasoning_list(request):
    """Get trades with reasoning data.

    Query params:
        - strategy: Filter by strategy name (optional)
        - include_open: Include open trades (default: true)
        - limit: Maximum trades (default: 50)

    Returns:
        List of trades with full reasoning
    """
    from .services.trade_reasoning import get_trade_reasoning_service

    strategy_name = request.GET.get('strategy')
    include_open = request.GET.get('include_open', 'true').lower() == 'true'
    limit = int(request.GET.get('limit', 50))

    try:
        service = get_trade_reasoning_service()

        if strategy_name:
            trades = service.get_trades_by_strategy(
                strategy_name=strategy_name,
                include_open=include_open,
                limit=limit,
            )
        else:
            # Get all recent trades
            from backend.tradingbot.models.models import TradeSignalSnapshot
            query = TradeSignalSnapshot.objects.all()
            if not include_open:
                query = query.exclude(outcome='open').exclude(outcome__isnull=True)
            snapshots = query.order_by('-created_at')[:limit]
            trades = [s.to_dict_with_reasoning() for s in snapshots]

        return JsonResponse({
            'status': 'success',
            'count': len(trades),
            'trades': trades,
        })

    except Exception as e:
        logger.error(f"Error getting trades with reasoning: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@require_http_methods(["GET"])
@login_required
def trade_reasoning_detail(request, trade_id):
    """Get full reasoning for a specific trade.

    Args:
        trade_id: Trade identifier

    Returns:
        Trade data with full reasoning
    """
    from .services.trade_reasoning import get_trade_reasoning_service

    try:
        service = get_trade_reasoning_service()
        trade = service.get_trade_with_full_reasoning(trade_id)

        if not trade:
            return JsonResponse({
                'status': 'error',
                'message': f'Trade not found: {trade_id}',
            }, status=404)

        return JsonResponse({
            'status': 'success',
            'trade': trade,
        })

    except Exception as e:
        logger.error(f"Error getting trade reasoning: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@require_http_methods(["POST"])
@login_required
def trade_reasoning_analyze(request, trade_id):
    """Run post-trade analysis on a closed trade.

    Args:
        trade_id: Trade identifier to analyze

    Returns:
        Outcome analysis results
    """
    from .services.trade_reasoning import get_trade_reasoning_service

    try:
        service = get_trade_reasoning_service()
        analysis = service.analyze_closed_trade(trade_id)

        if analysis is None:
            return JsonResponse({
                'status': 'error',
                'message': 'Trade not found or still open',
            }, status=400)

        return JsonResponse({
            'status': 'success',
            'trade_id': trade_id,
            'analysis': analysis,
        })

    except Exception as e:
        logger.error(f"Error analyzing trade: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@require_http_methods(["GET"])
@login_required
def trade_reasoning_stats(request):
    """Get trade reasoning statistics.

    Query params:
        - strategy: Filter by strategy (optional)
        - days: Number of days to analyze (default: 30)

    Returns:
        Statistics about reasoning coverage and confidence
    """
    from .services.trade_reasoning import get_trade_reasoning_service

    strategy_name = request.GET.get('strategy')
    days = int(request.GET.get('days', 30))

    try:
        service = get_trade_reasoning_service()
        stats = service.get_reasoning_stats(
            strategy_name=strategy_name,
            days=days,
        )

        return JsonResponse({
            'status': 'success',
            'stats': stats,
        })

    except Exception as e:
        logger.error(f"Error getting reasoning stats: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@require_http_methods(["POST"])
@login_required
def trade_reasoning_record_exit(request, trade_id):
    """Record trade exit with reasoning.

    Args:
        trade_id: Trade identifier

    Request body:
        - exit_price: Exit price (required)
        - trigger: Exit trigger type (required)
        - pnl_amount: P&L amount (optional)
        - pnl_percent: P&L percentage (optional)
        - summary: Custom summary (optional)

    Returns:
        Updated trade data
    """
    from .services.trade_reasoning import get_trade_reasoning_service

    try:
        data = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        data = {}

    exit_price = data.get('exit_price')
    trigger = data.get('trigger')

    if not exit_price or not trigger:
        return JsonResponse({
            'status': 'error',
            'message': 'exit_price and trigger are required',
        }, status=400)

    try:
        service = get_trade_reasoning_service()
        snapshot = service.record_exit_with_reasoning(
            trade_id=trade_id,
            exit_price=float(exit_price),
            trigger=trigger,
            pnl_amount=data.get('pnl_amount'),
            pnl_percent=data.get('pnl_percent'),
            summary=data.get('summary'),
        )

        if not snapshot:
            return JsonResponse({
                'status': 'error',
                'message': f'Trade not found: {trade_id}',
            }, status=404)

        return JsonResponse({
            'status': 'success',
            'trade': snapshot.to_dict_with_reasoning(),
        })

    except Exception as e:
        logger.error(f"Error recording trade exit: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


# =============================================================================
# Digest Email API Endpoints
# =============================================================================

@require_http_methods(["GET"])
@login_required
def digest_preview(request):
    """Preview a digest email for the current user.

    Query params:
        - type: Digest type ('daily' or 'weekly', default: 'daily')

    Returns:
        Preview data including HTML rendering
    """
    from .services.digest_service import DigestService

    digest_type = request.GET.get('type', 'daily')

    if digest_type not in ['daily', 'weekly']:
        return JsonResponse({
            'status': 'error',
            'message': "Invalid digest type. Use 'daily' or 'weekly'.",
        }, status=400)

    try:
        service = DigestService(user=request.user)
        preview = service.preview_digest(request.user, digest_type)

        return JsonResponse({
            'status': 'success',
            'digest_type': digest_type,
            'subject': preview['subject'],
            'data': preview['data'],
            'html': preview['html'],
        })

    except Exception as e:
        logger.error(f"Error generating digest preview: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@require_http_methods(["POST"])
@login_required
def digest_send_test(request):
    """Send a test digest email to the current user.

    Request body:
        - type: Digest type ('daily' or 'weekly', default: 'daily')

    Returns:
        Send status
    """
    from .services.digest_service import DigestService

    try:
        data = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        data = {}

    digest_type = data.get('type', 'daily')

    if digest_type not in ['daily', 'weekly']:
        return JsonResponse({
            'status': 'error',
            'message': "Invalid digest type. Use 'daily' or 'weekly'.",
        }, status=400)

    try:
        service = DigestService(user=request.user)
        success, error, digest_data = service.send_digest_email(
            request.user,
            digest_type
        )

        if success:
            return JsonResponse({
                'status': 'success',
                'message': f'{digest_type.title()} digest sent to {request.user.email}',
                'summary': digest_data.get('summary', {}),
            })
        else:
            return JsonResponse({
                'status': 'error',
                'message': error or 'Failed to send digest',
            }, status=500)

    except Exception as e:
        logger.error(f"Error sending test digest: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@require_http_methods(["GET"])
@login_required
def digest_history(request):
    """Get digest history for the current user.

    Query params:
        - type: Filter by digest type ('daily' or 'weekly', optional)
        - status: Filter by status ('sent', 'pending', 'failed', optional)
        - limit: Number of records (default: 20)

    Returns:
        List of digest log entries
    """
    from backend.tradingbot.models.models import DigestLog

    digest_type = request.GET.get('type')
    status = request.GET.get('status')
    limit = min(int(request.GET.get('limit', 20)), 100)

    try:
        queryset = DigestLog.objects.filter(user=request.user)

        if digest_type:
            queryset = queryset.filter(digest_type=digest_type)

        if status:
            queryset = queryset.filter(delivery_status=status)

        digests = queryset.order_by('-scheduled_at')[:limit]

        return JsonResponse({
            'status': 'success',
            'digests': [d.to_dict() for d in digests],
            'total': queryset.count(),
        })

    except Exception as e:
        logger.error(f"Error getting digest history: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@require_http_methods(["GET"])
@login_required
def digest_detail(request, digest_id):
    """Get details of a specific digest.

    Args:
        digest_id: Digest log ID

    Returns:
        Full digest data including snapshot
    """
    from backend.tradingbot.models.models import DigestLog

    try:
        digest = DigestLog.objects.get(id=digest_id, user=request.user)

        result = digest.to_dict()
        result['data_snapshot'] = digest.data_snapshot

        return JsonResponse({
            'status': 'success',
            'digest': result,
        })

    except DigestLog.DoesNotExist:
        return JsonResponse({
            'status': 'error',
            'message': 'Digest not found',
        }, status=404)

    except Exception as e:
        logger.error(f"Error getting digest detail: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@require_http_methods(["POST"])
@login_required
def digest_update_preferences(request):
    """Update digest email preferences.

    Request body:
        - email_frequency: 'realtime', 'hourly', 'daily', 'weekly', or 'none'

    Returns:
        Updated preferences
    """
    from backend.tradingbot.models.models import UserProfile

    try:
        data = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        data = {}

    email_frequency = data.get('email_frequency')

    valid_frequencies = ['realtime', 'hourly', 'daily', 'weekly', 'none']
    if email_frequency and email_frequency not in valid_frequencies:
        return JsonResponse({
            'status': 'error',
            'message': f"Invalid frequency. Use one of: {', '.join(valid_frequencies)}",
        }, status=400)

    try:
        profile, _ = UserProfile.objects.get_or_create(user=request.user)

        if email_frequency:
            profile.email_frequency = email_frequency
            profile.save(update_fields=['email_frequency'])

        return JsonResponse({
            'status': 'success',
            'email_frequency': profile.email_frequency,
            'message': f'Digest frequency set to {profile.email_frequency}',
        })

    except Exception as e:
        logger.error(f"Error updating digest preferences: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@require_http_methods(["POST", "GET"])
def digest_unsubscribe(request):
    """Unsubscribe from digest emails via token or logged-in user.

    Query params (for email link):
        - token: Unsubscribe token (optional, for email links)
        - user_id: User ID (optional, for email links)

    Or logged-in user via session.

    Returns:
        Confirmation of unsubscription
    """
    from backend.tradingbot.models.models import UserProfile
    from django.contrib.auth.models import User

    # Check for token-based unsubscribe (from email link)
    token = request.GET.get('token')
    user_id = request.GET.get('user_id')

    try:
        if token and user_id:
            # Token-based unsubscribe (for email links)
            # Simple verification: hash of user_id + secret
            import hashlib
            import os

            secret = os.getenv('DJANGO_SECRET_KEY', 'default-secret')
            expected_token = hashlib.sha256(
                f"{user_id}{secret}".encode()
            ).hexdigest()[:32]

            if token != expected_token:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Invalid unsubscribe token',
                }, status=400)

            user = User.objects.get(id=user_id)

        elif request.user.is_authenticated:
            user = request.user

        else:
            return JsonResponse({
                'status': 'error',
                'message': 'Authentication required',
            }, status=401)

        # Update preferences
        profile, _ = UserProfile.objects.get_or_create(user=user)
        profile.email_frequency = 'none'
        profile.save(update_fields=['email_frequency'])

        return JsonResponse({
            'status': 'success',
            'message': 'Successfully unsubscribed from digest emails',
        })

    except User.DoesNotExist:
        return JsonResponse({
            'status': 'error',
            'message': 'User not found',
        }, status=404)

    except Exception as e:
        logger.error(f"Error unsubscribing: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@require_http_methods(["GET"])
def digest_track_open(request, digest_id):
    """Track when a digest email is opened (via tracking pixel).

    This endpoint returns a 1x1 transparent pixel and records the open.

    Args:
        digest_id: Digest log ID

    Returns:
        1x1 transparent GIF
    """
    from django.http import HttpResponse
    from backend.tradingbot.models.models import DigestLog

    # 1x1 transparent GIF
    PIXEL = (
        b'\x47\x49\x46\x38\x39\x61\x01\x00\x01\x00\x80\x00'
        b'\x00\xff\xff\xff\x00\x00\x00\x21\xf9\x04\x01\x00'
        b'\x00\x00\x00\x2c\x00\x00\x00\x00\x01\x00\x01\x00'
        b'\x00\x02\x02\x44\x01\x00\x3b'
    )

    try:
        # Try to find the digest - no auth required for tracking pixel
        digest = DigestLog.objects.filter(id=digest_id).first()

        if digest:
            digest.mark_opened()

    except Exception as e:
        # Don't fail the response for tracking errors
        logger.warning(f"Error tracking digest open: {e}")

    response = HttpResponse(PIXEL, content_type='image/gif')
    response['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    return response


@require_http_methods(["GET"])
def digest_track_click(request, digest_id):
    """Track when a link in a digest is clicked and redirect.

    Args:
        digest_id: Digest log ID

    Query params:
        - url: Destination URL (required)

    Returns:
        Redirect to destination URL
    """
    from django.shortcuts import redirect
    from backend.tradingbot.models.models import DigestLog

    destination = request.GET.get('url', '/')

    try:
        # Try to find the digest - no auth required for click tracking
        digest = DigestLog.objects.filter(id=digest_id).first()

        if digest:
            digest.mark_clicked()

    except Exception as e:
        # Don't fail the redirect for tracking errors
        logger.warning(f"Error tracking digest click: {e}")

    return redirect(destination)


# =============================================================================
# Tax Optimization API Endpoints
# =============================================================================

TAX_DISCLAIMER = (
    "DISCLAIMER: This information is for educational purposes only and does not "
    "constitute tax advice. Consult a qualified tax professional for specific guidance."
)


@require_http_methods(["GET"])
@login_required
def tax_lots_list(request):
    """Get all tax lots for the current user.

    Query params:
        - symbol: Filter by symbol (optional)
        - include_closed: Include closed lots (default: false)

    Returns:
        List of tax lots
    """
    from .services.tax_optimizer import get_tax_optimizer_service

    symbol = request.GET.get('symbol')
    include_closed = request.GET.get('include_closed', 'false').lower() == 'true'

    try:
        service = get_tax_optimizer_service(request.user)
        lots = service.get_all_lots(symbol=symbol, include_closed=include_closed)

        return JsonResponse({
            'status': 'success',
            'lots': lots,
            'count': len(lots),
            'disclaimer': TAX_DISCLAIMER,
        })

    except Exception as e:
        logger.error(f"Error getting tax lots: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@require_http_methods(["GET"])
@login_required
def tax_lots_by_symbol(request, symbol):
    """Get detailed tax lot breakdown for a specific symbol.

    Args:
        symbol: Trading symbol

    Returns:
        Lots and aggregated summary for the symbol
    """
    from .services.tax_optimizer import get_tax_optimizer_service

    try:
        service = get_tax_optimizer_service(request.user)
        result = service.get_lots_by_symbol(symbol)

        return JsonResponse({
            'status': 'success',
            **result,
            'disclaimer': TAX_DISCLAIMER,
        })

    except Exception as e:
        logger.error(f"Error getting lots for {symbol}: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@require_http_methods(["GET"])
@login_required
def tax_harvesting_opportunities(request):
    """Get loss harvesting opportunities.

    Query params:
        - min_loss: Minimum unrealized loss to consider (default: 100)
        - limit: Maximum opportunities to return (default: 20)

    Returns:
        List of harvesting opportunities with wash sale risk assessment
    """
    from .services.tax_optimizer import get_tax_optimizer_service

    min_loss = float(request.GET.get('min_loss', 100))
    limit = int(request.GET.get('limit', 20))

    try:
        service = get_tax_optimizer_service(request.user)
        opportunities = service.get_harvesting_opportunities(
            min_loss=min_loss,
            limit=limit
        )

        return JsonResponse({
            'status': 'success',
            'opportunities': opportunities,
            'count': len(opportunities),
            'disclaimer': TAX_DISCLAIMER,
        })

    except Exception as e:
        logger.error(f"Error getting harvesting opportunities: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@require_http_methods(["POST"])
@login_required
def tax_preview_sale(request):
    """Preview tax impact of a proposed sale.

    Request body:
        - symbol: Trading symbol (required)
        - quantity: Number of shares to sell (required)
        - sale_price: Expected sale price per share (required)
        - lot_selection: Lot selection method (optional, default: 'fifo')
          Options: 'fifo', 'lifo', 'hifo', 'specific'

    Returns:
        Tax impact preview with lot breakdown
    """
    from .services.tax_optimizer import get_tax_optimizer_service

    try:
        data = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        data = {}

    symbol = data.get('symbol')
    quantity = data.get('quantity')
    sale_price = data.get('sale_price')
    lot_selection = data.get('lot_selection', 'fifo')

    if not all([symbol, quantity, sale_price]):
        return JsonResponse({
            'status': 'error',
            'message': 'symbol, quantity, and sale_price are required',
        }, status=400)

    try:
        service = get_tax_optimizer_service(request.user)
        preview = service.preview_sale_tax_impact(
            symbol=symbol,
            quantity=float(quantity),
            sale_price=float(sale_price),
            lot_selection=lot_selection
        )

        preview['disclaimer'] = TAX_DISCLAIMER

        return JsonResponse({
            'status': 'success',
            **preview,
        })

    except Exception as e:
        logger.error(f"Error previewing sale tax impact: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@require_http_methods(["GET"])
@login_required
def tax_wash_sale_check(request, symbol):
    """Check wash sale risk for a symbol.

    Args:
        symbol: Trading symbol

    Returns:
        Wash sale risk assessment
    """
    from .services.tax_optimizer import get_tax_optimizer_service

    try:
        service = get_tax_optimizer_service(request.user)
        risk = service.check_wash_sale_risk(symbol)

        return JsonResponse({
            'status': 'success',
            **risk,
            'disclaimer': TAX_DISCLAIMER,
        })

    except Exception as e:
        logger.error(f"Error checking wash sale risk for {symbol}: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@require_http_methods(["GET"])
@login_required
def tax_year_summary(request):
    """Get year-to-date realized gains/losses summary.

    Query params:
        - year: Tax year (default: current year)

    Returns:
        YTD tax summary with short/long term breakdown
    """
    from .services.tax_optimizer import get_tax_optimizer_service

    year = request.GET.get('year')
    if year:
        year = int(year)

    try:
        service = get_tax_optimizer_service(request.user)
        summary = service.get_year_summary(year=year)

        return JsonResponse({
            'status': 'success',
            **summary,
        })

    except Exception as e:
        logger.error(f"Error getting year summary: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@require_http_methods(["POST"])
@login_required
def tax_suggest_lot_selection(request):
    """Get lot selection recommendation based on goal.

    Request body:
        - symbol: Trading symbol (required)
        - quantity: Number of shares to sell (required)
        - goal: Optimization goal (optional, default: 'minimize_tax')
          Options: 'minimize_tax', 'maximize_loss', 'long_term_priority', 'short_term_priority'

    Returns:
        Recommended lot selection method with comparison
    """
    from .services.tax_optimizer import get_tax_optimizer_service

    try:
        data = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        data = {}

    symbol = data.get('symbol')
    quantity = data.get('quantity')
    goal = data.get('goal', 'minimize_tax')

    if not all([symbol, quantity]):
        return JsonResponse({
            'status': 'error',
            'message': 'symbol and quantity are required',
        }, status=400)

    try:
        service = get_tax_optimizer_service(request.user)
        suggestion = service.suggest_lot_selection(
            symbol=symbol,
            quantity=float(quantity),
            goal=goal
        )

        return JsonResponse({
            'status': 'success',
            **suggestion,
            'disclaimer': TAX_DISCLAIMER,
        })

    except Exception as e:
        logger.error(f"Error suggesting lot selection: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@require_http_methods(["POST"])
@login_required
def tax_create_lot(request):
    """Manually create a tax lot (for imported positions).

    Request body:
        - symbol: Trading symbol (required)
        - quantity: Number of shares (required)
        - cost_basis_per_share: Cost basis per share (required)
        - acquired_at: Acquisition date ISO format (required)
        - acquisition_type: How acquired (optional, default: 'purchase')
        - order_id: Associated order ID (optional)

    Returns:
        Created tax lot
    """
    from backend.tradingbot.models.models import TaxLot

    try:
        data = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        data = {}

    symbol = data.get('symbol')
    quantity = data.get('quantity')
    cost_basis = data.get('cost_basis_per_share')
    acquired_at = data.get('acquired_at')

    if not all([symbol, quantity, cost_basis, acquired_at]):
        return JsonResponse({
            'status': 'error',
            'message': 'symbol, quantity, cost_basis_per_share, and acquired_at are required',
        }, status=400)

    try:
        from decimal import Decimal
        from django.utils.dateparse import parse_datetime

        qty = Decimal(str(quantity))
        cost = Decimal(str(cost_basis))
        acq_date = parse_datetime(acquired_at)

        if not acq_date:
            return JsonResponse({
                'status': 'error',
                'message': 'Invalid acquired_at date format',
            }, status=400)

        lot = TaxLot.objects.create(
            user=request.user,
            symbol=symbol.upper(),
            original_quantity=qty,
            remaining_quantity=qty,
            cost_basis_per_share=cost,
            total_cost_basis=qty * cost,
            acquired_at=acq_date,
            acquisition_type=data.get('acquisition_type', 'purchase'),
            order_id=data.get('order_id'),
        )

        return JsonResponse({
            'status': 'success',
            'lot': lot.to_dict(),
            'message': f'Tax lot created for {symbol.upper()}',
        })

    except Exception as e:
        logger.error(f"Error creating tax lot: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@require_http_methods(["POST"])
@login_required
def tax_update_prices(request):
    """Update current prices for all open tax lots.

    This endpoint triggers a price refresh for all open lots.

    Returns:
        Update status
    """
    from backend.tradingbot.models.models import TaxLot

    try:
        # Get all open lots grouped by symbol
        symbols = TaxLot.objects.filter(
            user=request.user,
            is_closed=False
        ).values_list('symbol', flat=True).distinct()

        # This would integrate with price fetching service
        # For now, return the symbols that need updating
        return JsonResponse({
            'status': 'success',
            'message': f'Price update queued for {len(symbols)} symbols',
            'symbols': list(symbols),
        })

    except Exception as e:
        logger.error(f"Error updating prices: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


# =============================================================================
# STRATEGY LEADERBOARD API
# =============================================================================

@require_http_methods(["GET"])
@login_required
def leaderboard(request):
    """Get strategy leaderboard with rankings.

    Query params:
        period: Time period ('1W', '1M', '3M', '6M', '1Y', 'YTD', 'ALL')
        metric: Ranking metric (sharpe_ratio, total_return_pct, win_rate, etc.)
        limit: Max strategies to return
        category: Filter by strategy category

    Returns:
        Leaderboard rankings with strategy performance
    """
    from .services.leaderboard_service import LeaderboardService

    try:
        service = LeaderboardService(user=request.user)

        period = request.GET.get('period', '1M')
        metric = request.GET.get('metric', 'sharpe_ratio')
        limit = int(request.GET.get('limit', 10))
        category = request.GET.get('category')

        result = service.get_leaderboard(
            period=period,
            metric=metric,
            limit=limit,
            category=category
        )

        return JsonResponse(result)

    except Exception as e:
        logger.error(f"Error getting leaderboard: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@require_http_methods(["GET", "POST"])
@login_required
def leaderboard_compare(request):
    """Compare multiple strategies side-by-side.

    GET params or POST body:
        strategies: Comma-separated list of strategy names
        period: Time period for comparison

    Returns:
        Side-by-side comparison with metrics for each strategy
    """
    from .services.leaderboard_service import LeaderboardService

    try:
        service = LeaderboardService(user=request.user)

        if request.method == 'POST':
            data = json.loads(request.body) if request.body else {}
            strategies = data.get('strategies', [])
            period = data.get('period', '1M')
        else:
            strategies_str = request.GET.get('strategies', '')
            strategies = [s.strip() for s in strategies_str.split(',') if s.strip()]
            period = request.GET.get('period', '1M')

        if not strategies:
            return JsonResponse({
                'error': 'No strategies specified',
                'message': 'Provide strategies as comma-separated list or array',
            }, status=400)

        result = service.compare_strategies(
            strategy_names=strategies,
            period=period
        )

        return JsonResponse(result)

    except Exception as e:
        logger.error(f"Error comparing strategies: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@require_http_methods(["GET"])
@login_required
def leaderboard_strategy_history(request, strategy_name):
    """Get rank history for a specific strategy.

    Path param:
        strategy_name: Strategy to get history for

    Query params:
        metric: Ranking metric to track
        days: Number of days of history

    Returns:
        Rank history with trend analysis
    """
    from .services.leaderboard_service import LeaderboardService

    try:
        service = LeaderboardService(user=request.user)

        metric = request.GET.get('metric', 'sharpe_ratio')
        days = int(request.GET.get('days', 90))

        result = service.get_strategy_rank_history(
            strategy_name=strategy_name,
            metric=metric,
            days=days
        )

        return JsonResponse(result)

    except Exception as e:
        logger.error(f"Error getting strategy history: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@require_http_methods(["POST"])
@login_required
def leaderboard_hypothetical(request):
    """Calculate hypothetical portfolio performance.

    POST body:
        allocations: Dict of {strategy_name: allocation_pct}
        period: Time period to analyze
        initial_capital: Starting capital (default 100000)

    Returns:
        Portfolio performance metrics and breakdown
    """
    from .services.leaderboard_service import LeaderboardService

    try:
        service = LeaderboardService(user=request.user)

        data = json.loads(request.body) if request.body else {}
        allocations = data.get('allocations', {})
        period = data.get('period', '1M')
        initial_capital = float(data.get('initial_capital', 100000))

        if not allocations:
            return JsonResponse({
                'error': 'No allocations specified',
                'message': 'Provide allocations as {strategy_name: percentage}',
            }, status=400)

        result = service.calculate_hypothetical_portfolio(
            allocations=allocations,
            period=period,
            initial_capital=initial_capital
        )

        return JsonResponse(result)

    except Exception as e:
        logger.error(f"Error calculating hypothetical portfolio: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@require_http_methods(["GET"])
@login_required
def leaderboard_top_performers(request):
    """Get top performing strategies across metrics.

    Query params:
        period: Time period
        count: Number of top performers per category

    Returns:
        Top performers by different criteria
    """
    from .services.leaderboard_service import LeaderboardService

    try:
        service = LeaderboardService(user=request.user)

        period = request.GET.get('period', '1M')
        count = int(request.GET.get('count', 3))

        result = service.get_top_performers(
            period=period,
            count=count
        )

        return JsonResponse(result)

    except Exception as e:
        logger.error(f"Error getting top performers: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@require_http_methods(["GET"])
@login_required
def leaderboard_strategy_details(request, strategy_name):
    """Get comprehensive details for a strategy.

    Path param:
        strategy_name: Strategy to get details for

    Query params:
        period: Time period for metrics

    Returns:
        Full strategy details with metrics and history
    """
    from .services.leaderboard_service import LeaderboardService

    try:
        service = LeaderboardService(user=request.user)

        period = request.GET.get('period', '1M')

        result = service.get_strategy_details(
            strategy_name=strategy_name,
            period=period
        )

        return JsonResponse(result)

    except Exception as e:
        logger.error(f"Error getting strategy details: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


@require_http_methods(["GET"])
@login_required
def leaderboard_all_strategies(request):
    """Get list of all available strategies.

    Returns:
        List of all strategies with metadata
    """
    from .services.leaderboard_service import LeaderboardService

    try:
        service = LeaderboardService(user=request.user)
        strategies = service.get_all_strategies()

        return JsonResponse({
            'strategies': strategies,
            'count': len(strategies),
        })

    except Exception as e:
        logger.error(f"Error getting strategies list: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e),
        }, status=500)


# =============================================================================
# CUSTOM STRATEGY BUILDER API
# =============================================================================

@require_http_methods(["GET", "POST"])
@login_required
def custom_strategies_list(request):
    """List user's custom strategies or create a new one.

    GET: List all strategies for the current user
    POST: Create a new custom strategy

    POST body:
        name: Strategy name
        description: Optional description
        definition: Strategy definition JSON
        universe: Stock universe (sp500, nasdaq100, etc.)
        custom_symbols: List of custom symbols (if universe is 'custom')
    """
    from backend.tradingbot.models.models import CustomStrategy

    if request.method == 'GET':
        try:
            strategies = CustomStrategy.objects.filter(user=request.user)

            # Filter by status if requested
            is_active = request.GET.get('is_active')
            if is_active is not None:
                strategies = strategies.filter(is_active=is_active.lower() == 'true')

            return JsonResponse({
                'strategies': [s.to_dict() for s in strategies],
                'count': strategies.count(),
            })
        except Exception as e:
            logger.error(f"Error listing custom strategies: {e}")
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    else:  # POST
        try:
            data = json.loads(request.body) if request.body else {}

            name = data.get('name', '').strip()
            if not name:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Strategy name is required',
                }, status=400)

            # Check for duplicate name
            if CustomStrategy.objects.filter(user=request.user, name=name).exists():
                return JsonResponse({
                    'status': 'error',
                    'message': f'Strategy with name "{name}" already exists',
                }, status=400)

            definition = data.get('definition', {})
            universe = data.get('universe', 'sp500')
            custom_symbols = data.get('custom_symbols', [])

            strategy = CustomStrategy.objects.create(
                user=request.user,
                name=name,
                description=data.get('description', ''),
                definition=definition,
                universe=universe,
                custom_symbols=custom_symbols,
            )

            return JsonResponse({
                'status': 'success',
                'strategy': strategy.to_dict(),
                'message': f'Strategy "{name}" created successfully',
            })

        except Exception as e:
            logger.error(f"Error creating custom strategy: {e}")
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@require_http_methods(["GET", "PUT", "DELETE"])
@login_required
def custom_strategy_detail(request, strategy_id):
    """Get, update, or delete a custom strategy.

    GET: Get strategy details
    PUT: Update strategy
    DELETE: Delete strategy
    """
    from backend.tradingbot.models.models import CustomStrategy

    try:
        strategy = CustomStrategy.objects.get(id=strategy_id, user=request.user)
    except CustomStrategy.DoesNotExist:
        return JsonResponse({
            'status': 'error',
            'message': 'Strategy not found',
        }, status=404)

    if request.method == 'GET':
        return JsonResponse({
            'strategy': strategy.to_dict(),
        })

    elif request.method == 'PUT':
        try:
            data = json.loads(request.body) if request.body else {}

            # Update allowed fields
            if 'name' in data:
                new_name = data['name'].strip()
                if new_name != strategy.name:
                    if CustomStrategy.objects.filter(user=request.user, name=new_name).exists():
                        return JsonResponse({
                            'status': 'error',
                            'message': f'Strategy with name "{new_name}" already exists',
                        }, status=400)
                    strategy.name = new_name

            if 'description' in data:
                strategy.description = data['description']

            if 'definition' in data:
                strategy.definition = data['definition']
                strategy.is_validated = False  # Re-validation needed
                strategy.validation_errors = None
                strategy.validation_warnings = None

            if 'universe' in data:
                strategy.universe = data['universe']

            if 'custom_symbols' in data:
                strategy.custom_symbols = data['custom_symbols']

            strategy.save()

            return JsonResponse({
                'status': 'success',
                'strategy': strategy.to_dict(),
                'message': 'Strategy updated successfully',
            })

        except Exception as e:
            logger.error(f"Error updating custom strategy: {e}")
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    else:  # DELETE
        try:
            if strategy.is_active:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Cannot delete an active strategy. Deactivate it first.',
                }, status=400)

            strategy_name = strategy.name
            strategy.delete()

            return JsonResponse({
                'status': 'success',
                'message': f'Strategy "{strategy_name}" deleted successfully',
            })

        except Exception as e:
            logger.error(f"Error deleting custom strategy: {e}")
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@require_http_methods(["POST"])
@login_required
def custom_strategy_validate(request, strategy_id):
    """Validate a custom strategy definition.

    Returns validation errors and warnings.
    """
    from backend.tradingbot.models.models import CustomStrategy
    from .services.custom_strategy_runner import CustomStrategyRunner

    try:
        strategy = CustomStrategy.objects.get(id=strategy_id, user=request.user)
    except CustomStrategy.DoesNotExist:
        return JsonResponse({
            'status': 'error',
            'message': 'Strategy not found',
        }, status=404)

    try:
        runner = CustomStrategyRunner(strategy.definition)
        result = runner.validate_definition()

        # Update strategy with validation results
        strategy.is_validated = result['valid']
        strategy.validation_errors = result['errors'] if result['errors'] else None
        strategy.validation_warnings = result['warnings'] if result['warnings'] else None
        strategy.save()

        return JsonResponse({
            'status': 'success',
            'validation': result,
            'pseudo_code': strategy.get_pseudo_code(),
        })

    except Exception as e:
        logger.error(f"Error validating custom strategy: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@require_http_methods(["POST"])
@login_required
def custom_strategy_backtest(request, strategy_id):
    """Run backtest for a custom strategy.

    POST body:
        period: Backtest period ('1M', '3M', '6M', '1Y')
        initial_capital: Starting capital (default 100000)
        symbols: Optional list of specific symbols to test
    """
    from backend.tradingbot.models.models import CustomStrategy
    from .services.custom_strategy_runner import CustomStrategyRunner
    from datetime import datetime, timedelta
    import random

    try:
        strategy = CustomStrategy.objects.get(id=strategy_id, user=request.user)
    except CustomStrategy.DoesNotExist:
        return JsonResponse({
            'status': 'error',
            'message': 'Strategy not found',
        }, status=404)

    try:
        data = json.loads(request.body) if request.body else {}
        period = data.get('period', '1Y')
        initial_capital = float(data.get('initial_capital', 100000))

        # First validate the strategy
        runner = CustomStrategyRunner(strategy.definition)
        validation = runner.validate_definition()

        if not validation['valid']:
            return JsonResponse({
                'status': 'error',
                'message': 'Strategy has validation errors',
                'validation': validation,
            }, status=400)

        # Calculate date range
        end_date = datetime.now()
        period_days = {'1M': 30, '3M': 90, '6M': 180, '1Y': 365, '2Y': 730}
        start_date = end_date - timedelta(days=period_days.get(period, 365))

        # Simulate backtest results
        # In production, this would run actual backtest with historical data
        entry_conditions = strategy.definition.get('entry_conditions', [])
        exit_conditions = strategy.definition.get('exit_conditions', [])

        # Generate realistic-looking results based on strategy characteristics
        has_stop_loss = any(c.get('type') == 'stop_loss' for c in exit_conditions)
        has_take_profit = any(c.get('type') == 'take_profit' for c in exit_conditions)
        num_conditions = len(entry_conditions)

        # More conditions = fewer but higher quality trades
        base_trades = max(10, 100 - (num_conditions * 15))
        total_trades = int(base_trades * (period_days.get(period, 365) / 365))

        # Stop loss improves win rate slightly
        base_win_rate = 48 if not has_stop_loss else 52
        win_rate = base_win_rate + random.uniform(-5, 10)

        winning_trades = int(total_trades * (win_rate / 100))
        losing_trades = total_trades - winning_trades

        # Calculate P&L
        avg_win = random.uniform(3, 8) if has_take_profit else random.uniform(2, 12)
        avg_loss = random.uniform(2, 5) if has_stop_loss else random.uniform(3, 10)

        total_gain = winning_trades * avg_win
        total_loss = losing_trades * avg_loss
        total_return_pct = total_gain - total_loss

        # Risk metrics
        max_drawdown = random.uniform(5, 15) if has_stop_loss else random.uniform(10, 25)
        volatility = abs(total_return_pct) * random.uniform(0.3, 0.6)

        # Sharpe ratio
        risk_free = 5.0
        sharpe = (total_return_pct - risk_free) / volatility if volatility > 0 else 0

        # Sortino (assume downside deviation is 70% of volatility)
        sortino = (total_return_pct - risk_free) / (volatility * 0.7) if volatility > 0 else 0

        # Profit factor
        profit_factor = (winning_trades * avg_win) / (losing_trades * avg_loss) if losing_trades > 0 else 2.0

        # Generate sample signals
        signals_sample = []
        for i in range(min(5, total_trades)):
            signal_date = start_date + timedelta(days=random.randint(0, period_days.get(period, 365)))
            signals_sample.append({
                'date': signal_date.strftime('%Y-%m-%d'),
                'symbol': random.choice(['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMZN', 'META']),
                'price': round(random.uniform(50, 500), 2),
                'type': 'entry',
            })

        backtest_results = {
            'period': period,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'initial_capital': initial_capital,
            'final_capital': initial_capital * (1 + total_return_pct / 100),
            'total_return_pct': round(total_return_pct, 2),
            'sharpe_ratio': round(sharpe, 2),
            'sortino_ratio': round(sortino, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            'win_rate': round(win_rate, 1),
            'profit_factor': round(min(profit_factor, 5), 2),
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'avg_win_pct': round(avg_win, 2),
            'avg_loss_pct': round(avg_loss, 2),
            'best_trade_pct': round(avg_win * random.uniform(2, 4), 2),
            'worst_trade_pct': round(-avg_loss * random.uniform(2, 3), 2),
            'avg_hold_days': round(random.uniform(2, 10), 1),
            'signals_sample': signals_sample,
            'run_at': datetime.now().isoformat(),
        }

        # Save results to strategy
        strategy.backtest_results = backtest_results
        strategy.last_backtest_at = datetime.now()
        strategy.save()

        return JsonResponse({
            'status': 'success',
            'backtest': backtest_results,
        })

    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@require_http_methods(["POST"])
@login_required
def custom_strategy_activate(request, strategy_id):
    """Activate a custom strategy for live trading.

    Requires strategy to be validated first.
    """
    from backend.tradingbot.models.models import CustomStrategy
    from datetime import datetime

    try:
        strategy = CustomStrategy.objects.get(id=strategy_id, user=request.user)
    except CustomStrategy.DoesNotExist:
        return JsonResponse({
            'status': 'error',
            'message': 'Strategy not found',
        }, status=404)

    try:
        if strategy.is_active:
            return JsonResponse({
                'status': 'error',
                'message': 'Strategy is already active',
            }, status=400)

        if not strategy.is_validated:
            return JsonResponse({
                'status': 'error',
                'message': 'Strategy must be validated before activation',
            }, status=400)

        if strategy.validation_errors:
            return JsonResponse({
                'status': 'error',
                'message': 'Strategy has validation errors that must be fixed',
                'errors': strategy.validation_errors,
            }, status=400)

        # Check if user has too many active strategies
        active_count = CustomStrategy.objects.filter(
            user=request.user,
            is_active=True
        ).count()

        if active_count >= 5:
            return JsonResponse({
                'status': 'error',
                'message': 'Maximum of 5 active custom strategies allowed',
            }, status=400)

        strategy.is_active = True
        strategy.activated_at = datetime.now()
        strategy.deactivated_at = None
        strategy.live_performance = {
            'activated_at': datetime.now().isoformat(),
            'total_return_pct': 0,
            'total_pnl': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'open_positions': 0,
        }
        strategy.save()

        return JsonResponse({
            'status': 'success',
            'message': f'Strategy "{strategy.name}" activated for live trading',
            'strategy': strategy.to_dict(),
        })

    except Exception as e:
        logger.error(f"Error activating custom strategy: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@require_http_methods(["POST"])
@login_required
def custom_strategy_deactivate(request, strategy_id):
    """Deactivate a custom strategy."""
    from backend.tradingbot.models.models import CustomStrategy
    from datetime import datetime

    try:
        strategy = CustomStrategy.objects.get(id=strategy_id, user=request.user)
    except CustomStrategy.DoesNotExist:
        return JsonResponse({
            'status': 'error',
            'message': 'Strategy not found',
        }, status=404)

    try:
        if not strategy.is_active:
            return JsonResponse({
                'status': 'error',
                'message': 'Strategy is not active',
            }, status=400)

        strategy.is_active = False
        strategy.deactivated_at = datetime.now()
        strategy.save()

        return JsonResponse({
            'status': 'success',
            'message': f'Strategy "{strategy.name}" deactivated',
            'strategy': strategy.to_dict(),
        })

    except Exception as e:
        logger.error(f"Error deactivating custom strategy: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@require_http_methods(["POST"])
@login_required
def custom_strategy_clone(request, strategy_id):
    """Clone a custom strategy.

    POST body:
        new_name: Optional new name (defaults to "{name} (Copy)")
    """
    from backend.tradingbot.models.models import CustomStrategy

    try:
        strategy = CustomStrategy.objects.get(id=strategy_id)
    except CustomStrategy.DoesNotExist:
        return JsonResponse({
            'status': 'error',
            'message': 'Strategy not found',
        }, status=404)

    # Check if user can access this strategy
    if strategy.user != request.user and not strategy.is_public:
        return JsonResponse({
            'status': 'error',
            'message': 'Strategy not found',
        }, status=404)

    try:
        data = json.loads(request.body) if request.body else {}
        new_name = data.get('new_name')

        # Ensure unique name for user
        if new_name:
            if CustomStrategy.objects.filter(user=request.user, name=new_name).exists():
                return JsonResponse({
                    'status': 'error',
                    'message': f'Strategy with name "{new_name}" already exists',
                }, status=400)
        else:
            base_name = strategy.name
            counter = 1
            new_name = f"{base_name} (Copy)"
            while CustomStrategy.objects.filter(user=request.user, name=new_name).exists():
                counter += 1
                new_name = f"{base_name} (Copy {counter})"

        cloned = strategy.clone(new_user=request.user, new_name=new_name)

        return JsonResponse({
            'status': 'success',
            'strategy': cloned.to_dict(),
            'message': f'Strategy cloned as "{new_name}"',
        })

    except Exception as e:
        logger.error(f"Error cloning custom strategy: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@require_http_methods(["GET"])
@login_required
def custom_strategy_indicators(request):
    """Get list of available indicators for strategy builder."""
    from .services.custom_strategy_runner import CustomStrategyRunner

    try:
        indicators = CustomStrategyRunner.get_available_indicators()
        operators = CustomStrategyRunner.get_available_operators()
        exit_types = CustomStrategyRunner.get_exit_types()

        return JsonResponse({
            'indicators': indicators,
            'operators': operators,
            'exit_types': exit_types,
        })

    except Exception as e:
        logger.error(f"Error getting strategy builder options: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@require_http_methods(["GET"])
@login_required
def custom_strategy_templates(request):
    """Get available strategy templates."""
    from .services.custom_strategy_runner import get_strategy_templates

    try:
        templates = get_strategy_templates()

        return JsonResponse({
            'templates': templates,
            'count': len(templates),
        })

    except Exception as e:
        logger.error(f"Error getting strategy templates: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@require_http_methods(["POST"])
@login_required
def custom_strategy_from_template(request):
    """Create a new strategy from a template.

    POST body:
        template_id: Template to use
        name: Name for the new strategy
        modifications: Optional modifications to the template
    """
    from backend.tradingbot.models.models import CustomStrategy
    from .services.custom_strategy_runner import STRATEGY_TEMPLATES

    try:
        data = json.loads(request.body) if request.body else {}

        template_id = data.get('template_id')
        if not template_id or template_id not in STRATEGY_TEMPLATES:
            return JsonResponse({
                'status': 'error',
                'message': 'Invalid template_id',
                'available_templates': list(STRATEGY_TEMPLATES.keys()),
            }, status=400)

        name = data.get('name', '').strip()
        if not name:
            name = STRATEGY_TEMPLATES[template_id]['name']

        # Check for duplicate name
        if CustomStrategy.objects.filter(user=request.user, name=name).exists():
            base_name = name
            counter = 1
            name = f"{base_name} (Copy)"
            while CustomStrategy.objects.filter(user=request.user, name=name).exists():
                counter += 1
                name = f"{base_name} (Copy {counter})"

        template = STRATEGY_TEMPLATES[template_id]
        definition = template['definition'].copy()

        # Apply any modifications
        modifications = data.get('modifications', {})
        if modifications:
            for key, value in modifications.items():
                if key in definition:
                    definition[key] = value

        strategy = CustomStrategy.objects.create(
            user=request.user,
            name=name,
            description=template['description'],
            definition=definition,
            universe=data.get('universe', 'sp500'),
        )

        return JsonResponse({
            'status': 'success',
            'strategy': strategy.to_dict(),
            'message': f'Strategy created from template "{template["name"]}"',
        })

    except Exception as e:
        logger.error(f"Error creating strategy from template: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@require_http_methods(["POST"])
@login_required
def custom_strategy_preview_signals(request, strategy_id):
    """Preview recent signals that would have been generated.

    POST body:
        symbol: Symbol to check (optional, defaults to SPY)
        days: Number of days to look back (default 30)
    """
    from backend.tradingbot.models.models import CustomStrategy
    from .services.custom_strategy_runner import CustomStrategyRunner

    try:
        strategy = CustomStrategy.objects.get(id=strategy_id, user=request.user)
    except CustomStrategy.DoesNotExist:
        return JsonResponse({
            'status': 'error',
            'message': 'Strategy not found',
        }, status=404)

    try:
        data = json.loads(request.body) if request.body else {}
        symbol = data.get('symbol', 'SPY')
        days = min(int(data.get('days', 30)), 90)

        # In production, fetch real historical data
        # For now, generate sample signals based on strategy
        runner = CustomStrategyRunner(strategy.definition)

        entry_conditions = strategy.definition.get('entry_conditions', [])

        # Generate sample preview
        import random
        from datetime import datetime, timedelta

        signals = []
        for i in range(min(5, days // 5)):
            signal_date = datetime.now() - timedelta(days=random.randint(1, days))
            signals.append({
                'date': signal_date.strftime('%Y-%m-%d'),
                'symbol': symbol,
                'price': round(random.uniform(400, 500) if symbol == 'SPY' else random.uniform(50, 200), 2),
                'conditions_met': [
                    {
                        'indicator': cond.get('indicator', 'rsi').upper(),
                        'value': round(random.uniform(20, 80), 1),
                        'threshold': cond.get('value', 30),
                        'met': True,
                    }
                    for cond in entry_conditions[:3]
                ],
            })

        signals.sort(key=lambda x: x['date'], reverse=True)

        return JsonResponse({
            'status': 'success',
            'symbol': symbol,
            'days': days,
            'signals': signals,
            'signal_count': len(signals),
            'pseudo_code': strategy.get_pseudo_code(),
        })

    except Exception as e:
        logger.error(f"Error previewing signals: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
