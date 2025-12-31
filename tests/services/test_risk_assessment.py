"""Comprehensive tests for RiskAssessmentService."""

import pytest
from unittest.mock import Mock, patch
from django.contrib.auth.models import User
from django.utils import timezone

from backend.auth0login.services.risk_assessment import (
    RiskAssessmentService,
    QuestionnaireResult,
    RISK_QUESTIONS,
    risk_assessment_service
)


@pytest.fixture
def service():
    return RiskAssessmentService()


@pytest.fixture
def user():
    return User.objects.create_user(username='testuser', email='test@example.com')


class TestInit:
    """Test initialization."""

    def test_init(self):
        service = RiskAssessmentService()
        assert service.questions == RISK_QUESTIONS


class TestGetQuestions:
    """Test getting questions."""

    def test_get_questions(self, service):
        questions = service.get_questions()
        assert len(questions) > 0
        assert all('id' in q for q in questions)


class TestCalculateScore:
    """Test score calculation."""

    def test_calculate_score_conservative(self, service):
        responses = {
            'loss_reaction': 'sell_all',
            'timeline': 'less_1_year',
            'savings_portion': 'emergency',
            'experience': 'none',
            'holding_period': 'months',
            'monitoring': 'rarely'
        }
        result = service.calculate_score(responses)
        assert result.recommended_profile == 'conservative'

    def test_calculate_score_aggressive(self, service):
        responses = {
            'loss_reaction': 'buy_more',
            'timeline': '5_plus_years',
            'savings_portion': 'play_money',
            'experience': 'active',
            'holding_period': 'minutes',
            'monitoring': 'constantly'
        }
        result = service.calculate_score(responses)
        assert result.recommended_profile == 'aggressive'


class TestSubmitAssessment:
    """Test submitting assessment."""

    @patch('backend.auth0login.services.risk_assessment.RiskAssessment')
    def test_submit_assessment(self, mock_model, service, user):
        responses = {'loss_reaction': 'hold', 'timeline': '3_5_years'}

        mock_assessment = Mock()
        mock_assessment.id = 1
        mock_model.objects.create.return_value = mock_assessment

        result = service.submit_assessment(user, responses)
        assert result['status'] == 'success'


class TestGetUserAssessment:
    """Test getting user assessment."""

    @patch('backend.auth0login.services.risk_assessment.RiskAssessment')
    def test_get_user_assessment_found(self, mock_model, service, user):
        mock_assessment = Mock()
        mock_assessment.id = 1
        mock_assessment.calculated_score = 50
        mock_assessment.recommended_profile = 'moderate'
        mock_assessment.selected_profile = None
        mock_assessment.effective_profile = 'moderate'
        mock_assessment.profile_override = False
        mock_assessment.override_acknowledged = False
        mock_assessment.responses = {}
        mock_assessment.completed_at = timezone.now()
        mock_assessment.version = 1
        mock_model.get_latest_for_user.return_value = mock_assessment

        result = service.get_user_assessment(user)
        assert result is not None

    @patch('backend.auth0login.services.risk_assessment.RiskAssessment')
    def test_get_user_assessment_not_found(self, mock_model, service, user):
        mock_model.get_latest_for_user.return_value = None
        result = service.get_user_assessment(user)
        assert result is None


class TestProfileOverrideWarning:
    """Test profile override warnings."""

    def test_check_profile_override_warning_conservative_to_aggressive(self, service):
        warning = service.check_profile_override_warning('conservative', 'aggressive')
        assert warning is not None
        assert 'significantly more aggressive' in warning

    def test_check_profile_override_warning_moderate_to_aggressive(self, service):
        warning = service.check_profile_override_warning('moderate', 'aggressive')
        assert warning is not None

    def test_check_profile_override_warning_no_warning(self, service):
        warning = service.check_profile_override_warning('moderate', 'moderate')
        assert warning is None
