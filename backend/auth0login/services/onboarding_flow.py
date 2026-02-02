"""Onboarding flow state machine for the setup wizard.

Manages the 5-step wizard flow with:
- Step validation and processing
- Resume/skip functionality
- First-time user detection
- Post-setup broker validation
- Email confirmation
"""

import logging
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from django.contrib.auth.models import User
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.utils import timezone
from django.conf import settings

logger = logging.getLogger(__name__)


class WizardStep(Enum):
    """Wizard steps with their numeric values."""
    BROKER_CONNECTION = 1
    TRADING_MODE = 2
    STRATEGY_SELECTION = 3
    RISK_PROFILE = 4
    REVIEW_LAUNCH = 5


class WizardState(Enum):
    """Possible states for the wizard."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    ERROR = "error"


@dataclass
class StepResult:
    """Result of processing a wizard step."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None
    next_step: Optional[WizardStep] = None


@dataclass
class StepValidation:
    """Validation result for a step."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


# Valid strategies that can be selected
VALID_STRATEGIES = {
    'wsb-dip-bot',
    'wheel',
    'momentum-weeklies',
    'index-baseline',
    'lotto-scanner',
    'earnings-protection',
    'debit-spreads',
    'leaps-tracker',
    'swing-trading',
    'spx-credit-spreads',
}


class OnboardingFlowService:
    """State machine for the onboarding wizard flow.

    This service manages the 5-step wizard:
    1. Broker Connection - Connect Alpaca API credentials
    2. Trading Mode - Choose Paper vs Live trading
    3. Strategy Selection - Choose trading strategies
    4. Risk Profile - Complete risk assessment
    5. Review & Launch - Confirm and start trading
    """

    def __init__(self, user: User):
        """Initialize the onboarding service for a user.

        Args:
            user: The Django User instance
        """
        self.user = user
        self.logger = logging.getLogger(__name__)

    def get_or_create_session(self):
        """Get or create an onboarding session for the user.

        Returns:
            OnboardingSession instance
        """
        from backend.auth0login.models import OnboardingSession
        return OnboardingSession.get_active_session(self.user)

    def get_current_state(self) -> Dict[str, Any]:
        """Get the current state of the wizard.

        Returns:
            Dict with session info, current step, progress, etc.
        """
        session = self.get_or_create_session()

        return {
            'session_id': session.session_id,
            'current_step': session.current_step,
            'steps_completed': session.steps_completed,
            'progress_percentage': session.progress_percentage,
            'status': session.status,
            'step_data': session.step_data,
            'is_resumable': session.is_resumable,
            'can_skip': self._can_skip_to_end(),
            'started_at': session.started_at.isoformat() if session.started_at else None,
            'last_activity_at': session.last_activity_at.isoformat() if session.last_activity_at else None,
        }

    def validate_step(self, step: int, data: Dict[str, Any]) -> StepValidation:
        """Validate data for a specific step.

        Args:
            step: Step number (1-5)
            data: Data submitted for the step

        Returns:
            StepValidation with is_valid, errors, warnings
        """
        validators = {
            1: self._validate_broker_connection,
            2: self._validate_trading_mode,
            3: self._validate_strategy_selection,
            4: self._validate_risk_profile,
            5: self._validate_review_launch,
        }

        validator = validators.get(step)
        if not validator:
            return StepValidation(False, [f"Unknown step: {step}"], [])

        return validator(data)

    def process_step(self, step: int, data: Dict[str, Any]) -> StepResult:
        """Process a wizard step.

        Args:
            step: Step number (1-5)
            data: Data submitted for the step

        Returns:
            StepResult with success status, message, and next step
        """
        # First validate
        validation = self.validate_step(step, data)
        if not validation.is_valid:
            return StepResult(
                success=False,
                message="Validation failed",
                errors=validation.errors
            )

        # Process based on step
        processors = {
            1: self._process_broker_connection,
            2: self._process_trading_mode,
            3: self._process_strategy_selection,
            4: self._process_risk_profile,
            5: self._process_review_launch,
        }

        processor = processors.get(step)
        if not processor:
            return StepResult(
                success=False,
                message=f"Unknown step: {step}",
                errors=[f"No processor for step {step}"]
            )

        result = processor(data)

        # Update session on success
        if result.success:
            session = self.get_or_create_session()
            session.complete_step(step, data)

        return result

    def complete_wizard(self, final_data: Dict[str, Any]) -> StepResult:
        """Complete the wizard and create the final configuration.

        Args:
            final_data: Complete wizard data from all steps

        Returns:
            StepResult indicating success/failure
        """
        session = self.get_or_create_session()

        try:
            from backend.auth0login.models import WizardConfiguration, Credential

            # Merge all step data
            all_data = {}
            for step_num, step_data in session.step_data.items():
                if isinstance(step_data, dict):
                    all_data.update(step_data)
            all_data.update(final_data)

            # Get or create credential
            credential, _ = Credential.objects.get_or_create(user=self.user)

            # Delete any existing wizard config
            WizardConfiguration.objects.filter(user=self.user).delete()

            # Create new configuration
            config = WizardConfiguration.objects.create(
                user=self.user,
                trading_mode=all_data.get('trading_mode', 'paper'),
                selected_strategies=all_data.get('strategies', []),
                risk_profile=all_data.get('risk_profile', 'moderate'),
                max_position_pct=all_data.get('max_position_pct', 3.0),
                max_daily_loss_pct=all_data.get('max_daily_loss_pct', 8.0),
                max_positions=all_data.get('max_positions', 10),
            )

            # Validate broker connection
            validation_result = self._validate_broker_live(credential)
            config.broker_validated = validation_result.get('success', False)
            if config.broker_validated:
                config.broker_validated_at = timezone.now()
            config.save()

            # Mark session as completed
            session.mark_completed(config)

            # Start paper trading if enabled
            if config.trading_mode == 'paper':
                self._start_paper_trading()

            # Send confirmation email
            email_sent = self._send_confirmation_email(config)

            return StepResult(
                success=True,
                message="Wizard completed successfully!",
                data={
                    'config_id': config.id,
                    'broker_validated': config.broker_validated,
                    'email_sent': email_sent,
                    'trading_mode': config.trading_mode,
                }
            )

        except Exception as e:
            self.logger.error(f"Error completing wizard: {e}", exc_info=True)
            return StepResult(
                success=False,
                message=f"Failed to complete wizard: {str(e)}",
                errors=[str(e)]
            )

    def skip_wizard(self) -> StepResult:
        """Allow returning users to skip the wizard.

        Returns:
            StepResult indicating success/failure
        """
        if not self._can_skip_to_end():
            return StepResult(
                success=False,
                message="Cannot skip wizard - no existing configuration",
                errors=["First-time users must complete the wizard"]
            )

        session = self.get_or_create_session()
        session.status = 'skipped'
        session.save()

        return StepResult(
            success=True,
            message="Wizard skipped - using existing configuration"
        )

    def reset_wizard(self) -> StepResult:
        """Reset the wizard to start over.

        Returns:
            StepResult indicating success
        """
        from backend.auth0login.models import OnboardingSession

        # Mark current session as abandoned
        current_sessions = OnboardingSession.objects.filter(
            user=self.user,
            status='in_progress'
        )
        current_sessions.update(status='abandoned')

        return StepResult(
            success=True,
            message="Wizard reset - you can start over"
        )

    def _can_skip_to_end(self) -> bool:
        """Check if user can skip the wizard (has existing valid config).

        Returns:
            True if user has a valid configuration
        """
        from backend.auth0login.models import WizardConfiguration

        return WizardConfiguration.objects.filter(
            user=self.user,
            is_active=True,
            broker_validated=True
        ).exists()

    # Step Validators

    def _validate_broker_connection(self, data: Dict) -> StepValidation:
        """Validate broker connection step."""
        errors = []
        warnings = []

        api_key = data.get('api_key', '').strip()
        secret_key = data.get('secret_key', '').strip()

        if not api_key:
            errors.append("API key is required")
        elif len(api_key) < 10:
            errors.append("API key is too short")
        elif not (api_key.startswith('PK') or api_key.startswith('AK') or api_key.startswith('CK')):
            warnings.append("API key format looks unusual - verify it's correct")

        if not secret_key:
            errors.append("Secret key is required")
        elif len(secret_key) < 20:
            errors.append("Secret key appears too short")

        return StepValidation(len(errors) == 0, errors, warnings)

    def _validate_trading_mode(self, data: Dict) -> StepValidation:
        """Validate trading mode step."""
        errors = []
        mode = data.get('trading_mode', '')

        if mode not in ['paper', 'live']:
            errors.append("Trading mode must be 'paper' or 'live'")

        return StepValidation(len(errors) == 0, errors, [])

    def _validate_strategy_selection(self, data: Dict) -> StepValidation:
        """Validate strategy selection step."""
        errors = []
        warnings = []
        strategies = data.get('strategies', [])

        if not strategies:
            errors.append("At least one strategy must be selected")

        invalid = set(strategies) - VALID_STRATEGIES
        if invalid:
            errors.append(f"Invalid strategies: {', '.join(invalid)}")

        if len(strategies) > 5:
            warnings.append("Selecting more than 5 strategies may be overwhelming")

        return StepValidation(len(errors) == 0, errors, warnings)

    def _validate_risk_profile(self, data: Dict) -> StepValidation:
        """Validate risk profile step."""
        errors = []

        risk_profile = data.get('risk_profile')
        if not risk_profile:
            errors.append("Risk profile is required")
        elif risk_profile not in ['conservative', 'moderate', 'aggressive']:
            errors.append("Invalid risk profile")

        return StepValidation(len(errors) == 0, errors, [])

    def _validate_review_launch(self, data: Dict) -> StepValidation:
        """Validate review and launch step."""
        errors = []

        if not data.get('terms_accepted', False):
            errors.append("Terms of service must be accepted")

        return StepValidation(len(errors) == 0, errors, [])

    # Step Processors

    def _process_broker_connection(self, data: Dict) -> StepResult:
        """Process broker connection step."""
        from backend.auth0login.models import Credential

        try:
            credential, _ = Credential.objects.get_or_create(user=self.user)

            # Use the encryption-enabled setters
            credential.alpaca_api_key = data['api_key']
            credential.alpaca_secret_key = data['secret_key']
            credential.save()

            # Test the connection
            validation = self._validate_broker_live(credential)

            if validation.get('success'):
                credential.mark_validated()

                return StepResult(
                    success=True,
                    message="Broker connected successfully!",
                    data={
                        'equity': validation.get('equity'),
                        'cash': validation.get('cash'),
                        'status': validation.get('status'),
                    },
                    next_step=WizardStep.TRADING_MODE
                )
            else:
                credential.mark_invalid(validation.get('error', 'Unknown error'))

                return StepResult(
                    success=False,
                    message=f"Connection failed: {validation.get('error', 'Unknown error')}",
                    errors=[validation.get('error', 'Connection failed')]
                )

        except Exception as e:
            self.logger.error(f"Broker connection error: {e}")
            return StepResult(
                success=False,
                message=f"Error: {str(e)}",
                errors=[str(e)]
            )

    def _process_trading_mode(self, data: Dict) -> StepResult:
        """Process trading mode step."""
        return StepResult(
            success=True,
            message="Trading mode selected",
            data={'trading_mode': data['trading_mode']},
            next_step=WizardStep.STRATEGY_SELECTION
        )

    def _process_strategy_selection(self, data: Dict) -> StepResult:
        """Process strategy selection step."""
        return StepResult(
            success=True,
            message="Strategies selected",
            data={'strategies': data['strategies']},
            next_step=WizardStep.RISK_PROFILE
        )

    def _process_risk_profile(self, data: Dict) -> StepResult:
        """Process risk profile step."""
        # Optionally save to RiskAssessment model
        try:
            from backend.auth0login.models import RiskAssessment

            # Create or update risk assessment
            assessment, _ = RiskAssessment.objects.get_or_create(
                user=self.user,
                defaults={'responses': {}}
            )

            if data.get('questionnaire_responses'):
                assessment.responses = data['questionnaire_responses']
                assessment.calculated_score = data.get('risk_score', 15)

            assessment.recommended_profile = data.get('risk_profile', 'moderate')
            assessment.selected_profile = data.get('risk_profile', 'moderate')
            assessment.completed_at = timezone.now()
            assessment.save()

        except Exception as e:
            self.logger.warning(f"Could not save risk assessment: {e}")

        return StepResult(
            success=True,
            message="Risk profile set",
            data={
                'risk_profile': data.get('risk_profile'),
                'max_position_pct': data.get('max_position_pct', 3.0),
                'max_daily_loss_pct': data.get('max_daily_loss_pct', 8.0),
                'max_positions': data.get('max_positions', 10),
            },
            next_step=WizardStep.REVIEW_LAUNCH
        )

    def _process_review_launch(self, data: Dict) -> StepResult:
        """Process review and launch step - triggers wizard completion."""
        return self.complete_wizard(data)

    def _validate_broker_live(self, credential) -> Dict:
        """Actually test the broker connection.

        Args:
            credential: Credential model instance

        Returns:
            Dict with success status and account info or error
        """
        try:
            from alpaca.trading.client import TradingClient

            # Get decrypted credentials
            api_key = credential.alpaca_api_key
            secret_key = credential.alpaca_secret_key

            if not api_key or not secret_key:
                return {
                    'success': False,
                    'error': 'Credentials not found'
                }

            client = TradingClient(
                api_key=api_key,
                secret_key=secret_key,
                paper=True  # Always test against paper first
            )

            account = client.get_account()

            return {
                'success': True,
                'account_id': str(account.id),
                'equity': str(account.equity),
                'cash': str(account.cash),
                'buying_power': str(account.buying_power),
                'status': str(account.status),
            }

        except Exception as e:
            self.logger.error(f"Broker validation error: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _start_paper_trading(self):
        """Initialize paper trading for the user."""
        try:
            from backend.auth0login.models import TradingGate

            gate, created = TradingGate.objects.get_or_create(user=self.user)
            if created or not gate.paper_trading_started_at:
                gate.start_paper_trading()
                self.logger.info(f"Started paper trading for user {self.user.username}")

        except Exception as e:
            self.logger.error(f"Failed to start paper trading: {e}")

    def _send_confirmation_email(self, config) -> bool:
        """Send setup confirmation email.

        Args:
            config: WizardConfiguration instance

        Returns:
            True if email sent successfully
        """
        try:
            subject = "Welcome to WallStreetBots - Setup Complete!"

            context = {
                'user': self.user,
                'config': config,
                'trading_mode': 'Paper Trading' if config.trading_mode == 'paper' else 'Live Trading',
                'strategies': config.selected_strategies,
                'risk_profile': config.risk_profile.title(),
                'site_url': getattr(settings, 'SITE_URL', 'http://localhost:8000'),
            }

            # Try to render HTML template
            try:
                html_message = render_to_string('emails/setup_complete.html', context)
            except Exception:
                html_message = None

            # Plain text fallback
            plain_message = f"""
Welcome to WallStreetBots, {self.user.username}!

Your trading setup is complete. Here's a summary:

Trading Mode: {context['trading_mode']}
Risk Profile: {context['risk_profile']}
Selected Strategies: {', '.join(config.selected_strategies)}

{'Your paper trading period has begun. You have 14 days to practice before live trading is enabled.' if config.trading_mode == 'paper' else 'You are now ready to start live trading.'}

Dashboard: {context['site_url']}/dashboard

Happy trading!
The WallStreetBots Team
"""

            send_mail(
                subject=subject,
                message=plain_message,
                html_message=html_message,
                from_email=getattr(settings, 'DEFAULT_FROM_EMAIL', 'noreply@wallstreetbots.com'),
                recipient_list=[self.user.email],
                fail_silently=True,
            )

            config.email_confirmation_sent = True
            config.email_confirmation_sent_at = timezone.now()
            config.save(update_fields=['email_confirmation_sent', 'email_confirmation_sent_at'])

            self.logger.info(f"Setup confirmation email sent to {self.user.email}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to send confirmation email: {e}")
            return False


def is_first_time_user(user: User) -> bool:
    """Check if user is a first-time user (no completed onboarding).

    Args:
        user: Django User instance

    Returns:
        True if user has never completed the wizard
    """
    from backend.auth0login.models import OnboardingSession, WizardConfiguration

    # Check for active wizard configuration
    has_config = WizardConfiguration.objects.filter(
        user=user,
        is_active=True
    ).exists()

    # Check for completed onboarding session
    has_completed_session = OnboardingSession.objects.filter(
        user=user,
        status='completed'
    ).exists()

    return not (has_config or has_completed_session)


def get_onboarding_service(user: User) -> OnboardingFlowService:
    """Factory function to get onboarding service for a user.

    Args:
        user: Django User instance

    Returns:
        OnboardingFlowService instance
    """
    return OnboardingFlowService(user)
