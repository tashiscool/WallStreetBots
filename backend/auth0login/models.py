from django.contrib.auth.models import User
from django.db import models
from django.utils import timezone

from backend.tradingbot.models.models import Portfolio


class Credential(models.Model):
    """Stores the user's Alpaca API key and secret key with encryption.

    Credentials are encrypted at rest using Fernet symmetric encryption.
    The encryption is transparent through the alpaca_api_key and
    alpaca_secret_key properties.
    """

    # Increased length for encrypted values
    ALPACA_ID_MAX_LENGTH = 500
    ALPACA_KEY_MAX_LENGTH = 500

    # User relationship
    user = models.OneToOneField(
        User,
        help_text="Associated user",
        on_delete=models.CASCADE
    )

    # Encrypted credential fields (stored as encrypted base64 strings)
    # Use _encrypted suffix to indicate these are not plaintext
    alpaca_id = models.CharField(
        max_length=ALPACA_ID_MAX_LENGTH,
        help_text="Encrypted Alpaca API key",
        blank=True,
        default=''
    )
    alpaca_key = models.CharField(
        max_length=ALPACA_KEY_MAX_LENGTH,
        help_text="Encrypted Alpaca secret key",
        blank=True,
        default=''
    )

    # Key metadata for rotation tracking
    key_version = models.IntegerField(
        default=1,
        help_text="Encryption key version for rotation"
    )
    last_rotated_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When credentials were last rotated"
    )

    # Validation status
    is_valid = models.BooleanField(
        default=False,
        help_text="Whether credentials have been validated"
    )
    last_validated_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When credentials were last validated"
    )
    validation_error = models.TextField(
        blank=True,
        default='',
        help_text="Last validation error message if any"
    )

    # Audit fields
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["user"]

    def __str__(self):
        return f"Credential for {self.user}"

    @property
    def alpaca_api_key(self) -> str:
        """Get decrypted Alpaca API key."""
        if not self.alpaca_id:
            return ''
        try:
            from backend.auth0login.services.credential_encryption import decrypt_credential
            # Check if value is encrypted (for migration from plaintext)
            from backend.auth0login.services.credential_encryption import get_encryption_service
            service = get_encryption_service()
            if service.is_encrypted(self.alpaca_id):
                return decrypt_credential(self.alpaca_id)
            # Return as-is if not encrypted (legacy plaintext)
            return self.alpaca_id
        except Exception:
            # Return empty on decryption failure
            return ''

    @alpaca_api_key.setter
    def alpaca_api_key(self, value: str):
        """Set and encrypt Alpaca API key."""
        if not value:
            self.alpaca_id = ''
            return
        from backend.auth0login.services.credential_encryption import encrypt_credential
        self.alpaca_id = encrypt_credential(value)

    @property
    def alpaca_secret_key(self) -> str:
        """Get decrypted Alpaca secret key."""
        if not self.alpaca_key:
            return ''
        try:
            from backend.auth0login.services.credential_encryption import decrypt_credential
            from backend.auth0login.services.credential_encryption import get_encryption_service
            service = get_encryption_service()
            if service.is_encrypted(self.alpaca_key):
                return decrypt_credential(self.alpaca_key)
            # Return as-is if not encrypted (legacy plaintext)
            return self.alpaca_key
        except Exception:
            return ''

    @alpaca_secret_key.setter
    def alpaca_secret_key(self, value: str):
        """Set and encrypt Alpaca secret key."""
        if not value:
            self.alpaca_key = ''
            return
        from backend.auth0login.services.credential_encryption import encrypt_credential
        self.alpaca_key = encrypt_credential(value)

    def rotate_credentials(self, new_api_key: str, new_secret_key: str):
        """Rotate credentials with new values."""
        self.alpaca_api_key = new_api_key
        self.alpaca_secret_key = new_secret_key
        self.key_version += 1
        self.last_rotated_at = timezone.now()
        self.is_valid = False  # Requires re-validation
        self.save()

    def mark_validated(self):
        """Mark credentials as validated."""
        self.is_valid = True
        self.last_validated_at = timezone.now()
        self.validation_error = ''
        self.save(update_fields=['is_valid', 'last_validated_at', 'validation_error', 'updated_at'])

    def mark_invalid(self, error: str):
        """Mark credentials as invalid with error message."""
        self.is_valid = False
        self.validation_error = error
        self.save(update_fields=['is_valid', 'validation_error', 'updated_at'])

    @property
    def has_credentials(self) -> bool:
        """Check if both credentials are set."""
        return bool(self.alpaca_id and self.alpaca_key)


class BotInstance(models.Model):
    """An instance of a bot."""

    name = models.CharField(max_length=100, blank=False, help_text="Bot Name")
    portfolio = models.OneToOneField(
        Portfolio,
        blank=True,
        help_text="Associated portfolio",
        on_delete=models.CASCADE,
    )
    user = models.ForeignKey(
        User, help_text="Associated user", on_delete=models.CASCADE
    )
    bot = None  # To Be Completed

    # Metadata
    class Meta:
        ordering = ["user"]

    # Methods
    def __str__(self):
        return f"Bot: {self.name} \n User: {self.user!s} \n Portfolio: {self.portfolio.name}"


class TradingGate(models.Model):
    """Controls paper-to-live trading transition for user safety.

    This model enforces a mandatory paper trading period before users
    can trade with real money. It tracks paper trading duration,
    performance, and approval status for live trading.
    """

    APPROVAL_METHOD_CHOICES = [
        ('auto', 'Automatic - All criteria met'),
        ('manual', 'Manual - Approved by admin'),
        ('override', 'Override - Admin bypass'),
    ]

    # User relationship
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='trading_gate',
        help_text="User this gate applies to"
    )

    # Paper trading tracking
    paper_trading_started_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When paper trading was first enabled"
    )
    paper_trading_days_required = models.IntegerField(
        default=14,
        help_text="Minimum days of paper trading required before live trading"
    )

    # Performance snapshot at time of live trading request
    paper_performance_snapshot = models.JSONField(
        default=dict,
        blank=True,
        help_text="Performance metrics at time of live trading request (P&L, trades, Sharpe, etc.)"
    )

    # Live trading approval tracking
    live_trading_requested_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When user requested to switch to live trading"
    )
    live_trading_approved_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When live trading was approved"
    )
    live_trading_approved = models.BooleanField(
        default=False,
        help_text="Whether live trading has been approved"
    )
    approval_method = models.CharField(
        max_length=20,
        choices=APPROVAL_METHOD_CHOICES,
        null=True,
        blank=True,
        help_text="How live trading was approved"
    )

    # Denial tracking
    live_trading_denied_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When live trading request was denied (if applicable)"
    )
    denial_reason = models.TextField(
        null=True,
        blank=True,
        help_text="Reason for denial if request was rejected"
    )

    # Audit fields
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Trading Gate"
        verbose_name_plural = "Trading Gates"
        indexes = [
            models.Index(fields=['user', 'live_trading_approved']),
            models.Index(fields=['live_trading_approved', 'created_at']),
        ]

    def __str__(self):
        status = "Approved" if self.live_trading_approved else "Paper Only"
        return f"TradingGate for {self.user.username}: {status}"

    @property
    def days_in_paper_trading(self) -> int:
        """Calculate days spent in paper trading mode."""
        if not self.paper_trading_started_at:
            return 0
        delta = timezone.now() - self.paper_trading_started_at
        return delta.days

    @property
    def days_remaining(self) -> int:
        """Days remaining before minimum paper trading period is met."""
        remaining = self.paper_trading_days_required - self.days_in_paper_trading
        return max(0, remaining)

    @property
    def paper_trading_duration_met(self) -> bool:
        """Check if minimum paper trading duration has been met."""
        return self.days_in_paper_trading >= self.paper_trading_days_required

    def start_paper_trading(self):
        """Mark the start of paper trading period."""
        if not self.paper_trading_started_at:
            self.paper_trading_started_at = timezone.now()
            self.save()

    def request_live_trading(self, performance_snapshot: dict = None):
        """User requests transition to live trading."""
        self.live_trading_requested_at = timezone.now()
        if performance_snapshot:
            self.paper_performance_snapshot = performance_snapshot
        # Reset any previous denial
        self.live_trading_denied_at = None
        self.denial_reason = None
        self.save()

    def approve_live_trading(self, method: str = 'auto'):
        """Approve live trading for this user."""
        self.live_trading_approved = True
        self.live_trading_approved_at = timezone.now()
        self.approval_method = method
        self.save()

    def deny_live_trading(self, reason: str):
        """Deny live trading request with reason."""
        self.live_trading_denied_at = timezone.now()
        self.denial_reason = reason
        self.live_trading_requested_at = None  # Clear request
        self.save()

    def revoke_live_trading(self, reason: str = None):
        """Revoke previously approved live trading (e.g., for violations)."""
        self.live_trading_approved = False
        self.live_trading_approved_at = None
        self.approval_method = None
        if reason:
            self.denial_reason = f"Revoked: {reason}"
        self.save()


class RiskAssessment(models.Model):
    """Stores user risk assessment questionnaire responses and calculated profile.

    This model captures the user's answers to risk tolerance questions,
    calculates a risk score, and recommends an appropriate risk profile.
    """

    RISK_PROFILE_CHOICES = [
        ('conservative', 'Conservative'),
        ('moderate', 'Moderate'),
        ('aggressive', 'Aggressive'),
    ]

    # Current questionnaire version - increment when questions change
    CURRENT_VERSION = 1

    # User relationship
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='risk_assessments',
        help_text="User who completed this assessment"
    )

    # Questionnaire responses stored as JSON
    # Format: {"question_id": {"answer": "value", "score": 1-5}, ...}
    responses = models.JSONField(
        default=dict,
        help_text="Questionnaire responses with answer values and scores"
    )

    # Calculated score (weighted sum of all question scores)
    calculated_score = models.IntegerField(
        default=0,
        help_text="Total calculated risk score (6-30 range)"
    )

    # Recommended profile based on score
    recommended_profile = models.CharField(
        max_length=20,
        choices=RISK_PROFILE_CHOICES,
        default='moderate',
        help_text="System-recommended risk profile based on score"
    )

    # User's chosen profile (may differ from recommended if they override)
    selected_profile = models.CharField(
        max_length=20,
        choices=RISK_PROFILE_CHOICES,
        null=True,
        blank=True,
        help_text="User's selected profile (if different from recommended)"
    )

    # Track if user overrode the recommendation
    profile_override = models.BooleanField(
        default=False,
        help_text="Whether user selected a different profile than recommended"
    )
    override_acknowledged = models.BooleanField(
        default=False,
        help_text="Whether user acknowledged the override warning"
    )

    # Questionnaire version for future updates
    version = models.IntegerField(
        default=CURRENT_VERSION,
        help_text="Questionnaire version used for this assessment"
    )

    # Timestamps
    completed_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the assessment was completed"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Risk Assessment"
        verbose_name_plural = "Risk Assessments"
        indexes = [
            models.Index(fields=['user', 'version']),
            models.Index(fields=['user', '-completed_at']),
        ]

    def __str__(self):
        status = "Completed" if self.completed_at else "In Progress"
        return f"RiskAssessment for {self.user.username}: {self.recommended_profile} ({status})"

    @property
    def effective_profile(self) -> str:
        """Return the profile that should be used (selected or recommended)."""
        return self.selected_profile if self.selected_profile else self.recommended_profile

    @property
    def is_complete(self) -> bool:
        """Check if assessment has been completed."""
        return self.completed_at is not None

    @classmethod
    def get_latest_for_user(cls, user):
        """Get the most recent completed assessment for a user."""
        return cls.objects.filter(
            user=user,
            completed_at__isnull=False
        ).order_by('-completed_at').first()


class WizardConfiguration(models.Model):
    """Persistent storage for wizard configuration.

    This model stores all configuration from the setup wizard,
    replacing the environment variable approach for persistence.
    """

    TRADING_MODE_CHOICES = [
        ('paper', 'Paper Trading'),
        ('live', 'Live Trading'),
    ]

    RISK_PROFILE_CHOICES = [
        ('conservative', 'Conservative'),
        ('moderate', 'Moderate'),
        ('aggressive', 'Aggressive'),
    ]

    # User relationship
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='wizard_config',
        help_text="User this configuration belongs to"
    )

    # Trading Configuration
    trading_mode = models.CharField(
        max_length=10,
        choices=TRADING_MODE_CHOICES,
        default='paper',
        help_text="Paper or live trading mode"
    )

    # Selected Strategies (stored as JSON array)
    selected_strategies = models.JSONField(
        default=list,
        help_text="List of strategy IDs selected by user"
    )

    # Risk Configuration
    risk_profile = models.CharField(
        max_length=20,
        choices=RISK_PROFILE_CHOICES,
        default='moderate',
        help_text="User's risk profile"
    )
    max_position_pct = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        default=3.00,
        help_text="Maximum position size percentage"
    )
    max_daily_loss_pct = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        default=8.00,
        help_text="Maximum daily loss percentage"
    )
    max_positions = models.IntegerField(
        default=10,
        help_text="Maximum number of open positions"
    )

    # Risk Assessment Reference
    risk_assessment = models.ForeignKey(
        RiskAssessment,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='wizard_configs',
        help_text="Associated risk assessment"
    )

    # Configuration Status
    is_active = models.BooleanField(
        default=True,
        help_text="Whether this configuration is currently active"
    )
    broker_validated = models.BooleanField(
        default=False,
        help_text="Whether broker connection has been validated"
    )
    broker_validated_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When broker connection was last validated"
    )

    # Email Confirmation
    email_confirmation_sent = models.BooleanField(
        default=False,
        help_text="Whether setup confirmation email was sent"
    )
    email_confirmation_sent_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When confirmation email was sent"
    )

    # Audit Fields
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Wizard Configuration"
        verbose_name_plural = "Wizard Configurations"
        indexes = [
            models.Index(fields=['user', 'is_active']),
            models.Index(fields=['created_at']),
        ]

    def __str__(self):
        return f"WizardConfig for {self.user.username}: {self.trading_mode} ({self.risk_profile})"

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'trading_mode': self.trading_mode,
            'selected_strategies': self.selected_strategies,
            'risk_profile': self.risk_profile,
            'max_position_pct': float(self.max_position_pct),
            'max_daily_loss_pct': float(self.max_daily_loss_pct),
            'max_positions': self.max_positions,
            'is_active': self.is_active,
            'broker_validated': self.broker_validated,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class OnboardingSession(models.Model):
    """Tracks wizard session state for resume/skip functionality.

    This model enables users to resume the wizard where they left off
    and tracks overall onboarding progress.
    """

    SESSION_STATUS_CHOICES = [
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('abandoned', 'Abandoned'),
        ('skipped', 'Skipped'),
    ]

    # User relationship
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='onboarding_sessions',
        help_text="User this session belongs to"
    )

    # Session identifier (for multi-device support)
    session_id = models.CharField(
        max_length=100,
        unique=True,
        db_index=True,
        help_text="Unique session identifier"
    )

    # Progress Tracking
    current_step = models.IntegerField(
        default=1,
        help_text="Current wizard step (1-5)"
    )
    steps_completed = models.JSONField(
        default=list,
        help_text="List of completed step numbers"
    )
    step_data = models.JSONField(
        default=dict,
        help_text="Data collected at each step (for resume)"
    )

    # Status
    status = models.CharField(
        max_length=20,
        choices=SESSION_STATUS_CHOICES,
        default='in_progress',
        help_text="Current session status"
    )

    # Timing
    started_at = models.DateTimeField(
        auto_now_add=True,
        help_text="When the session started"
    )
    last_activity_at = models.DateTimeField(
        auto_now=True,
        help_text="Last activity timestamp"
    )
    completed_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the session was completed"
    )

    # Step-level timing (for analytics)
    step_timings = models.JSONField(
        default=dict,
        help_text="Time spent on each step in seconds"
    )

    # Final Configuration (linked after completion)
    wizard_configuration = models.OneToOneField(
        WizardConfiguration,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='onboarding_session',
        help_text="Final configuration created from this session"
    )

    # User Agent / Device Info (for analytics)
    user_agent = models.CharField(
        max_length=500,
        blank=True,
        help_text="Browser user agent"
    )
    ip_address = models.GenericIPAddressField(
        null=True,
        blank=True,
        help_text="Client IP address"
    )

    class Meta:
        ordering = ['-started_at']
        verbose_name = "Onboarding Session"
        verbose_name_plural = "Onboarding Sessions"
        indexes = [
            models.Index(fields=['user', 'status']),
            models.Index(fields=['session_id']),
            models.Index(fields=['status', 'started_at']),
        ]

    def __str__(self):
        return f"OnboardingSession {self.session_id[:8]}... for {self.user.username}: Step {self.current_step}"

    @property
    def progress_percentage(self) -> int:
        """Calculate completion percentage."""
        return int((len(self.steps_completed) / 5) * 100)

    @property
    def is_resumable(self) -> bool:
        """Check if session can be resumed."""
        from datetime import timedelta
        if self.status != 'in_progress':
            return False
        # Session is stale after 7 days
        stale_threshold = timezone.now() - timedelta(days=7)
        return self.last_activity_at > stale_threshold

    def complete_step(self, step: int, data: dict = None):
        """Mark a step as completed and store its data."""
        if step not in self.steps_completed:
            self.steps_completed = self.steps_completed + [step]
        if data:
            self.step_data[str(step)] = data
        self.current_step = max(self.current_step, step + 1)
        self.save()

    def mark_completed(self, config: 'WizardConfiguration'):
        """Mark session as completed."""
        self.status = 'completed'
        self.completed_at = timezone.now()
        self.wizard_configuration = config
        self.save()

    @classmethod
    def get_active_session(cls, user):
        """Get or create active session for user."""
        import uuid

        session = cls.objects.filter(
            user=user,
            status='in_progress'
        ).first()

        if session and session.is_resumable:
            return session
        elif session:
            # Mark stale session as abandoned
            session.status = 'abandoned'
            session.save()

        # Create new session
        return cls.objects.create(
            user=user,
            session_id=str(uuid.uuid4())
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            'session_id': self.session_id,
            'current_step': self.current_step,
            'steps_completed': self.steps_completed,
            'step_data': self.step_data,
            'status': self.status,
            'progress_percentage': self.progress_percentage,
            'is_resumable': self.is_resumable,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'last_activity_at': self.last_activity_at.isoformat() if self.last_activity_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
        }
