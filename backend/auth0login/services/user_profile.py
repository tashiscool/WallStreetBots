"""User Profile Service

Manages user profiles, onboarding progress, risk assessment, and personalization.
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from django.utils import timezone
from django.db import transaction

logger = logging.getLogger(__name__)


# Risk Assessment Questions
RISK_ASSESSMENT_QUESTIONS = [
    {
        'id': 'timeline',
        'question': 'What is your investment time horizon?',
        'description': 'How long do you plan to keep your money invested?',
        'options': [
            {'value': 1, 'label': 'Less than 1 year', 'description': 'Short-term needs'},
            {'value': 2, 'label': '1-3 years', 'description': 'Medium-term goals'},
            {'value': 3, 'label': '3-5 years', 'description': 'Moderate time horizon'},
            {'value': 4, 'label': '5-10 years', 'description': 'Long-term investing'},
            {'value': 5, 'label': '10+ years', 'description': 'Very long-term'},
        ],
    },
    {
        'id': 'loss_reaction',
        'question': 'If your portfolio dropped 20% in a month, what would you do?',
        'description': 'This helps us understand your emotional response to losses.',
        'options': [
            {'value': 1, 'label': 'Sell everything immediately', 'description': 'Protect remaining capital'},
            {'value': 2, 'label': 'Sell some positions', 'description': 'Reduce exposure'},
            {'value': 3, 'label': 'Hold and wait', 'description': 'Wait for recovery'},
            {'value': 4, 'label': 'Buy more if possible', 'description': 'Opportunity to add'},
            {'value': 5, 'label': 'Significantly increase positions', 'description': 'Aggressive buying'},
        ],
    },
    {
        'id': 'experience',
        'question': 'How would you describe your trading experience?',
        'description': 'Your experience level helps us recommend appropriate strategies.',
        'options': [
            {'value': 1, 'label': 'Complete beginner', 'description': 'Never traded before'},
            {'value': 2, 'label': 'Some experience', 'description': 'Made a few trades'},
            {'value': 3, 'label': 'Moderate experience', 'description': '1-3 years active trading'},
            {'value': 4, 'label': 'Experienced', 'description': '3-5 years active trading'},
            {'value': 5, 'label': 'Expert', 'description': '5+ years, professional level'},
        ],
    },
    {
        'id': 'income_stability',
        'question': 'How stable is your income?',
        'description': 'This affects how much risk you can take.',
        'options': [
            {'value': 1, 'label': 'Very unstable', 'description': 'Irregular or uncertain income'},
            {'value': 2, 'label': 'Somewhat unstable', 'description': 'Variable income'},
            {'value': 3, 'label': 'Moderately stable', 'description': 'Generally predictable'},
            {'value': 4, 'label': 'Stable', 'description': 'Consistent salary'},
            {'value': 5, 'label': 'Very stable', 'description': 'Secure, growing income'},
        ],
    },
    {
        'id': 'capital_importance',
        'question': 'How important is preserving this capital?',
        'description': 'Can you afford to lose some or all of this investment?',
        'options': [
            {'value': 1, 'label': 'Cannot afford any loss', 'description': 'Need every dollar'},
            {'value': 2, 'label': 'Need most of it', 'description': 'Important savings'},
            {'value': 3, 'label': 'Prefer to preserve', 'description': 'Would like to keep most'},
            {'value': 4, 'label': 'Some loss acceptable', 'description': 'Part of overall wealth'},
            {'value': 5, 'label': 'Can afford to lose', 'description': 'True risk capital'},
        ],
    },
    {
        'id': 'goal',
        'question': 'What is your primary investment goal?',
        'description': 'This helps us align strategy recommendations.',
        'options': [
            {'value': 1, 'label': 'Capital preservation', 'description': 'Protect against loss'},
            {'value': 2, 'label': 'Income generation', 'description': 'Regular cash flow'},
            {'value': 3, 'label': 'Balanced growth', 'description': 'Moderate growth with safety'},
            {'value': 4, 'label': 'Growth', 'description': 'Build wealth over time'},
            {'value': 5, 'label': 'Aggressive growth', 'description': 'Maximum returns'},
        ],
    },
]

# Onboarding Steps
ONBOARDING_STEPS = [
    {
        'step': 1,
        'name': 'Welcome',
        'description': 'Get started with WallStreetBots',
        'field': None,
        'required': False,
    },
    {
        'step': 2,
        'name': 'Risk Assessment',
        'description': 'Complete the risk questionnaire',
        'field': 'risk_assessment_completed',
        'required': True,
    },
    {
        'step': 3,
        'name': 'Connect Brokerage',
        'description': 'Connect your Alpaca account',
        'field': 'brokerage_connected',
        'required': True,
    },
    {
        'step': 4,
        'name': 'Choose Strategy',
        'description': 'Activate your first strategy',
        'field': 'first_strategy_activated',
        'required': True,
    },
    {
        'step': 5,
        'name': 'Review & Start',
        'description': 'Review settings and start trading',
        'field': 'onboarding_completed',
        'required': True,
    },
]


class UserProfileService:
    """Service for managing user profiles and onboarding."""

    def __init__(self, user):
        """Initialize service with user.

        Args:
            user: Django User instance
        """
        self.user = user
        self.logger = logging.getLogger(__name__)

    def get_or_create_profile(self):
        """Get or create user profile.

        Returns:
            UserProfile instance
        """
        from backend.tradingbot.models.models import UserProfile

        profile, created = UserProfile.objects.get_or_create(user=self.user)

        if created:
            self.logger.info(f"Created new profile for user {self.user.username}")
            # Set initial values
            profile.onboarding_started_at = timezone.now()
            profile.paper_trading_start = timezone.now()
            profile.save()

        return profile

    def get_profile(self):
        """Get user profile.

        Returns:
            UserProfile instance or None
        """
        from backend.tradingbot.models.models import UserProfile

        try:
            return UserProfile.objects.get(user=self.user)
        except UserProfile.DoesNotExist:
            return self.get_or_create_profile()

    def update_profile(self, data: dict) -> Tuple[bool, str]:
        """Update user profile with provided data.

        Args:
            data: Dictionary of fields to update

        Returns:
            Tuple of (success, message)
        """
        profile = self.get_or_create_profile()

        # Fields that can be updated
        allowed_fields = [
            # Risk Assessment
            'risk_tolerance', 'investment_timeline', 'max_loss_tolerance',
            # Financial Context
            'investable_capital', 'income_source', 'annual_income',
            'capital_is_risk_capital', 'net_worth',
            # Experience
            'trading_experience', 'options_experience', 'options_level',
            'crypto_experience', 'margin_experience', 'shorting_experience',
            # Preferences
            'dashboard_layout', 'default_chart_timeframe', 'timezone',
            'theme', 'compact_mode', 'show_tutorial_hints',
            # Communication
            'email_frequency', 'email_trade_alerts', 'email_risk_alerts',
            'email_performance_reports', 'push_notifications_enabled',
            'sms_alerts_enabled', 'phone_number',
        ]

        try:
            for field, value in data.items():
                if field in allowed_fields:
                    setattr(profile, field, value)

            profile.save()
            self.logger.info(f"Updated profile for user {self.user.username}")
            return True, "Profile updated successfully"

        except Exception as e:
            self.logger.error(f"Error updating profile: {e}")
            return False, str(e)

    def get_risk_questions(self) -> List[dict]:
        """Get risk assessment questions.

        Returns:
            List of question dictionaries
        """
        return RISK_ASSESSMENT_QUESTIONS

    def submit_risk_assessment(self, answers: dict) -> dict:
        """Submit risk assessment answers and calculate score.

        Args:
            answers: Dictionary of question_id -> answer_value

        Returns:
            Dictionary with score and risk tolerance
        """
        profile = self.get_or_create_profile()

        # Validate answers
        valid_questions = {q['id'] for q in RISK_ASSESSMENT_QUESTIONS}
        filtered_answers = {
            k: v for k, v in answers.items()
            if k in valid_questions
        }

        if len(filtered_answers) < len(RISK_ASSESSMENT_QUESTIONS):
            missing = valid_questions - set(filtered_answers.keys())
            self.logger.warning(f"Missing answers: {missing}")

        # Calculate risk score
        risk_score = profile.calculate_risk_score(filtered_answers)

        # Get recommended settings
        recommendations = profile.get_recommended_settings()

        return {
            'risk_score': risk_score,
            'risk_tolerance': profile.risk_tolerance,
            'risk_tolerance_display': profile.risk_profile_name,
            'recommendations': recommendations,
            'message': f"Risk assessment complete. Your risk profile is {profile.risk_profile_name}.",
        }

    def get_onboarding_status(self) -> dict:
        """Get current onboarding status.

        Returns:
            Dictionary with onboarding progress
        """
        profile = self.get_or_create_profile()

        # Determine current step
        current_step = 1
        steps_status = []

        for step_info in ONBOARDING_STEPS:
            field = step_info.get('field')

            if field:
                is_complete = getattr(profile, field, False)
            else:
                is_complete = step_info['step'] <= profile.onboarding_step

            steps_status.append({
                **step_info,
                'completed': is_complete,
                'current': step_info['step'] == profile.onboarding_step,
            })

            if not is_complete and step_info['required']:
                current_step = step_info['step']
                break

        # Calculate overall progress
        completed_steps = sum(1 for s in steps_status if s['completed'])
        total_steps = len(ONBOARDING_STEPS)
        progress_percentage = int((completed_steps / total_steps) * 100)

        return {
            'current_step': profile.onboarding_step,
            'total_steps': total_steps,
            'progress_percentage': progress_percentage,
            'is_complete': profile.onboarding_completed,
            'steps': steps_status,
            'started_at': profile.onboarding_started_at.isoformat() if profile.onboarding_started_at else None,
            'completed_at': profile.onboarding_completed_at.isoformat() if profile.onboarding_completed_at else None,
            'next_action': self._get_next_action(profile),
        }

    def _get_next_action(self, profile) -> dict:
        """Get the next required action for onboarding.

        Args:
            profile: UserProfile instance

        Returns:
            Dictionary with next action info
        """
        if profile.onboarding_completed:
            return {
                'action': 'complete',
                'title': 'Onboarding Complete',
                'description': 'You can start trading!',
                'url': '/dashboard',
            }

        if not profile.risk_assessment_completed:
            return {
                'action': 'risk_assessment',
                'title': 'Complete Risk Assessment',
                'description': 'Answer a few questions to determine your risk profile',
                'url': '/setup#risk-assessment',
            }

        if not profile.brokerage_connected:
            return {
                'action': 'connect_brokerage',
                'title': 'Connect Brokerage',
                'description': 'Link your Alpaca account to start trading',
                'url': '/setup#brokerage',
            }

        if not profile.first_strategy_activated:
            return {
                'action': 'activate_strategy',
                'title': 'Activate Strategy',
                'description': 'Choose and activate your first trading strategy',
                'url': '/strategies',
            }

        return {
            'action': 'finish',
            'title': 'Finish Setup',
            'description': 'Review and complete your setup',
            'url': '/setup#review',
        }

    def complete_step(self, step: str) -> dict:
        """Mark an onboarding step as complete.

        Args:
            step: Step identifier

        Returns:
            Dictionary with updated status
        """
        profile = self.get_or_create_profile()

        try:
            profile.complete_onboarding_step(step)
            self.logger.info(f"Completed onboarding step '{step}' for user {self.user.username}")

            return {
                'success': True,
                'message': f"Step '{step}' completed",
                'onboarding_status': self.get_onboarding_status(),
            }

        except Exception as e:
            self.logger.error(f"Error completing step: {e}")
            return {
                'success': False,
                'message': str(e),
            }

    def mark_brokerage_connected(self) -> dict:
        """Mark brokerage as connected.

        Returns:
            Dictionary with status
        """
        return self.complete_step('brokerage')

    def mark_strategy_activated(self) -> dict:
        """Mark first strategy as activated.

        Returns:
            Dictionary with status
        """
        return self.complete_step('strategy')

    def mark_trade_executed(self) -> dict:
        """Mark first trade as executed.

        Returns:
            Dictionary with status
        """
        return self.complete_step('trade')

    def get_recommended_settings(self) -> dict:
        """Get recommended settings based on profile.

        Returns:
            Dictionary of recommended settings
        """
        profile = self.get_or_create_profile()
        return profile.get_recommended_settings()

    def switch_trading_mode(self, mode: str) -> dict:
        """Switch between paper and live trading mode.

        Args:
            mode: 'paper' or 'live'

        Returns:
            Dictionary with result
        """
        profile = self.get_or_create_profile()

        if mode == 'live':
            if not profile.live_trading_approved:
                return {
                    'success': False,
                    'message': 'Live trading has not been approved for this account',
                }

            profile.is_paper_trading = False
            profile.save()

            return {
                'success': True,
                'message': 'Switched to live trading mode',
                'trading_mode': 'live',
            }

        elif mode == 'paper':
            profile.is_paper_trading = True
            if not profile.paper_trading_start:
                profile.paper_trading_start = timezone.now()
            profile.save()

            return {
                'success': True,
                'message': 'Switched to paper trading mode',
                'trading_mode': 'paper',
            }

        else:
            return {
                'success': False,
                'message': f"Invalid mode: {mode}",
            }

    def update_last_active(self):
        """Update last active timestamp."""
        profile = self.get_or_create_profile()
        profile.update_last_active()

    def update_last_login(self):
        """Update last login timestamp."""
        profile = self.get_or_create_profile()
        profile.last_login_at = timezone.now()
        profile.save(update_fields=['last_login_at'])

    def get_profile_summary(self) -> dict:
        """Get a summary of the user profile.

        Returns:
            Dictionary with profile summary
        """
        profile = self.get_or_create_profile()

        return {
            'username': self.user.username,
            'email': self.user.email,
            'risk_profile': profile.risk_profile_name,
            'trading_mode': profile.trading_mode,
            'onboarding_complete': profile.onboarding_completed,
            'onboarding_progress': profile.onboarding_progress,
            'experience_level': profile.trading_experience or 'Not set',
            'member_since': profile.created_at.isoformat() if profile.created_at else None,
            'last_active': profile.last_active_at.isoformat() if profile.last_active_at else None,
        }


def get_user_profile_service(user) -> UserProfileService:
    """Factory function to get user profile service.

    Args:
        user: Django User instance

    Returns:
        UserProfileService instance
    """
    return UserProfileService(user=user)


def ensure_profile_exists(user):
    """Ensure a user profile exists.

    Args:
        user: Django User instance

    Returns:
        UserProfile instance
    """
    service = UserProfileService(user=user)
    return service.get_or_create_profile()
