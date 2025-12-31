from django.contrib.auth.models import User
from django.db import models
from django.utils import timezone

from backend.tradingbot.models.models import Portfolio


class Credential(models.Model):
    """stores the user's Alpaca API key and secret key."""

    ALPACA_ID_MAX_LENGTH = 100
    ALPACA_KEY_MAX_LENGTH = 100
    # Fields
    user = models.OneToOneField(
        User, help_text="Associated user", on_delete=models.CASCADE
    )
    alpaca_id = models.CharField(
        max_length=ALPACA_ID_MAX_LENGTH, help_text="Enter your Alpaca id"
    )
    alpaca_key = models.CharField(
        max_length=ALPACA_KEY_MAX_LENGTH, help_text="Enter your Alpaca key"
    )

    # Metadata
    class Meta:
        ordering = ["user"]

    # Methods
    def __str__(self):
        return "Credential for " + str(self.user)


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
