"""Comprehensive tests for UserProfileService."""

import pytest
from unittest.mock import Mock, patch
from django.contrib.auth.models import User
from django.utils import timezone

from backend.auth0login.services.user_profile import (
    UserProfileService,
    get_user_profile_service,
    ensure_profile_exists,
    RISK_ASSESSMENT_QUESTIONS,
    ONBOARDING_STEPS
)


@pytest.fixture
def user(db):  # db fixture ensures database access
    return User.objects.create_user(username='testuser', email='test@example.com')


@pytest.fixture
def service(user):
    return UserProfileService(user=user)


@pytest.mark.django_db
class TestInit:
    """Test initialization."""

    def test_init_with_user(self, user):
        service = UserProfileService(user)
        assert service.user == user


@pytest.mark.django_db
class TestGetOrCreateProfile:
    """Test profile creation."""

    @patch('backend.tradingbot.models.models.UserProfile')
    def test_get_or_create_profile_creates(self, mock_model, service, user):
        mock_profile = Mock()
        mock_model.objects.get_or_create.return_value = (mock_profile, True)

        result = service.get_or_create_profile()
        assert result == mock_profile

    @patch('backend.tradingbot.models.models.UserProfile')
    def test_get_or_create_profile_gets(self, mock_model, service, user):
        mock_profile = Mock()
        mock_model.objects.get_or_create.return_value = (mock_profile, False)

        result = service.get_or_create_profile()
        assert result == mock_profile


@pytest.mark.django_db
class TestUpdateProfile:
    """Test profile updates."""

    @patch('backend.tradingbot.models.models.UserProfile')
    def test_update_profile_success(self, mock_model, service, user):
        mock_profile = Mock()
        mock_model.objects.get_or_create.return_value = (mock_profile, False)

        data = {'risk_tolerance': 'moderate', 'email_frequency': 'daily'}
        success, message = service.update_profile(data)

        assert success is True


@pytest.mark.django_db
class TestRiskAssessment:
    """Test risk assessment."""

    def test_get_risk_questions(self, service):
        questions = service.get_risk_questions()
        assert len(questions) == len(RISK_ASSESSMENT_QUESTIONS)

    @patch('backend.tradingbot.models.models.UserProfile')
    def test_submit_risk_assessment(self, mock_model, service, user):
        mock_profile = Mock()
        mock_profile.calculate_risk_score.return_value = 50
        mock_profile.risk_tolerance = 'moderate'
        mock_profile.risk_profile_name = 'Moderate'
        mock_profile.get_recommended_settings.return_value = {}
        mock_model.objects.get_or_create.return_value = (mock_profile, False)

        answers = {'timeline': 3, 'loss_reaction': 3}
        result = service.submit_risk_assessment(answers)

        assert 'risk_score' in result


@pytest.mark.django_db
class TestOnboardingStatus:
    """Test onboarding status."""

    @patch('backend.tradingbot.models.models.UserProfile')
    def test_get_onboarding_status(self, mock_model, service, user):
        mock_profile = Mock()
        mock_profile.onboarding_step = 1
        mock_profile.onboarding_completed = False
        mock_profile.risk_assessment_completed = False
        mock_profile.brokerage_connected = False
        mock_profile.first_strategy_activated = False
        mock_profile.onboarding_started_at = timezone.now()
        mock_profile.onboarding_completed_at = None
        mock_model.objects.get_or_create.return_value = (mock_profile, False)

        result = service.get_onboarding_status()

        assert 'current_step' in result
        assert 'steps' in result


@pytest.mark.django_db
class TestCompleteStep:
    """Test completing onboarding steps."""

    @patch('backend.tradingbot.models.models.UserProfile')
    def test_complete_step(self, mock_model, service, user):
        mock_profile = Mock()
        mock_profile.complete_onboarding_step = Mock()
        mock_profile.onboarding_step = 2
        mock_profile.onboarding_completed = False
        mock_model.objects.get_or_create.return_value = (mock_profile, False)

        result = service.complete_step('risk_assessment')
        assert result['success'] is True


@pytest.mark.django_db
class TestTradingMode:
    """Test trading mode switching."""

    @patch('backend.tradingbot.models.models.UserProfile')
    def test_switch_to_paper_mode(self, mock_model, service, user):
        mock_profile = Mock()
        mock_profile.is_paper_trading = True
        mock_profile.paper_trading_start = None
        mock_model.objects.get_or_create.return_value = (mock_profile, False)

        result = service.switch_trading_mode('paper')
        assert result['success'] is True

    @patch('backend.tradingbot.models.models.UserProfile')
    def test_switch_to_live_mode_approved(self, mock_model, service, user):
        mock_profile = Mock()
        mock_profile.live_trading_approved = True
        mock_profile.is_paper_trading = False
        mock_model.objects.get_or_create.return_value = (mock_profile, False)

        result = service.switch_trading_mode('live')
        assert result['success'] is True


@pytest.mark.django_db
class TestGetProfileSummary:
    """Test profile summary."""

    @patch('backend.tradingbot.models.models.UserProfile')
    def test_get_profile_summary(self, mock_model, service, user):
        mock_profile = Mock()
        mock_profile.risk_profile_name = 'Moderate'
        mock_profile.trading_mode = 'paper'
        mock_profile.onboarding_completed = True
        mock_profile.onboarding_progress = 100
        mock_profile.trading_experience = 'intermediate'
        mock_profile.created_at = timezone.now()
        mock_profile.last_active_at = timezone.now()
        mock_model.objects.get_or_create.return_value = (mock_profile, False)

        result = service.get_profile_summary()
        assert 'username' in result
        assert 'risk_profile' in result
