"""
Comprehensive tests for OnboardingFlowService.

Tests wizard session management, step processing, validation, and completion.
Target: 80%+ coverage.
"""
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from dataclasses import dataclass


class TestWizardEnums(unittest.TestCase):
    """Test wizard enum definitions."""

    def test_wizard_step_values(self):
        """Test that wizard steps have correct values."""
        from backend.auth0login.services.onboarding_flow import WizardStep

        self.assertEqual(WizardStep.BROKER_CONNECTION.value, 1)
        self.assertEqual(WizardStep.TRADING_MODE.value, 2)
        self.assertEqual(WizardStep.STRATEGY_SELECTION.value, 3)
        self.assertEqual(WizardStep.RISK_PROFILE.value, 4)
        self.assertEqual(WizardStep.REVIEW_LAUNCH.value, 5)

    def test_wizard_state_values(self):
        """Test wizard state enum values."""
        from backend.auth0login.services.onboarding_flow import WizardState

        self.assertEqual(WizardState.NOT_STARTED.value, 'not_started')
        self.assertEqual(WizardState.IN_PROGRESS.value, 'in_progress')
        self.assertEqual(WizardState.COMPLETED.value, 'completed')
        self.assertEqual(WizardState.ABANDONED.value, 'abandoned')
        self.assertEqual(WizardState.ERROR.value, 'error')


class TestDataClasses(unittest.TestCase):
    """Test wizard data classes."""

    def test_step_result_creation(self):
        """Test StepResult dataclass."""
        from backend.auth0login.services.onboarding_flow import StepResult, WizardStep

        result = StepResult(
            success=True,
            message="Step completed",
            data={'key': 'value'},
            next_step=WizardStep.TRADING_MODE
        )

        self.assertTrue(result.success)
        self.assertEqual(result.message, "Step completed")
        self.assertEqual(result.data['key'], 'value')
        self.assertEqual(result.next_step, WizardStep.TRADING_MODE)

    def test_step_result_with_errors(self):
        """Test StepResult with errors."""
        from backend.auth0login.services.onboarding_flow import StepResult

        result = StepResult(
            success=False,
            message="Validation failed",
            errors=['Error 1', 'Error 2']
        )

        self.assertFalse(result.success)
        self.assertEqual(len(result.errors), 2)

    def test_step_validation_creation(self):
        """Test StepValidation dataclass."""
        from backend.auth0login.services.onboarding_flow import StepValidation

        validation = StepValidation(
            is_valid=True,
            errors=[],
            warnings=['Consider using paper trading first']
        )

        self.assertTrue(validation.is_valid)
        self.assertEqual(len(validation.errors), 0)
        self.assertEqual(len(validation.warnings), 1)


class TestValidStrategies(unittest.TestCase):
    """Test valid strategies constant."""

    def test_valid_strategies_set(self):
        """Test VALID_STRATEGIES contains expected values."""
        from backend.auth0login.services.onboarding_flow import VALID_STRATEGIES

        expected = {
            'wsb-dip-bot', 'wheel', 'momentum-weeklies', 'index-baseline',
            'lotto-scanner', 'earnings-protection', 'debit-spreads',
            'leaps-tracker', 'swing-trading', 'spx-credit-spreads'
        }

        self.assertEqual(VALID_STRATEGIES, expected)


class TestOnboardingFlowServiceInit(unittest.TestCase):
    """Test OnboardingFlowService initialization."""

    def test_service_initialization(self):
        """Test service can be initialized with a user."""
        from backend.auth0login.services.onboarding_flow import OnboardingFlowService

        mock_user = Mock()
        mock_user.id = 1
        mock_user.username = 'testuser'

        service = OnboardingFlowService(mock_user)

        self.assertEqual(service.user, mock_user)


class TestBrokerConnectionValidation(unittest.TestCase):
    """Test broker connection step validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_user = Mock()
        self.mock_user.id = 1

        from backend.auth0login.services.onboarding_flow import OnboardingFlowService
        self.service = OnboardingFlowService(self.mock_user)

    def test_valid_broker_credentials(self):
        """Test validation with valid credentials."""
        data = {
            'api_key': 'PKXXXXXXXXXXXXXXXX',
            'secret_key': 'secretkey1234567890123456789012'
        }

        validation = self.service._validate_broker_connection(data)

        self.assertTrue(validation.is_valid)
        self.assertEqual(len(validation.errors), 0)

    def test_missing_api_key(self):
        """Test validation with missing API key."""
        data = {
            'api_key': '',
            'secret_key': 'secretkey1234567890123456789012'
        }

        validation = self.service._validate_broker_connection(data)

        self.assertFalse(validation.is_valid)
        self.assertIn('API key is required', validation.errors)

    def test_short_api_key(self):
        """Test validation with too short API key."""
        data = {
            'api_key': 'PK123',
            'secret_key': 'secretkey1234567890123456789012'
        }

        validation = self.service._validate_broker_connection(data)

        self.assertFalse(validation.is_valid)
        self.assertIn('API key is too short', validation.errors)

    def test_missing_secret_key(self):
        """Test validation with missing secret key."""
        data = {
            'api_key': 'PKXXXXXXXXXXXXXXXX',
            'secret_key': ''
        }

        validation = self.service._validate_broker_connection(data)

        self.assertFalse(validation.is_valid)
        self.assertIn('Secret key is required', validation.errors)

    def test_short_secret_key(self):
        """Test validation with too short secret key."""
        data = {
            'api_key': 'PKXXXXXXXXXXXXXXXX',
            'secret_key': 'short'
        }

        validation = self.service._validate_broker_connection(data)

        self.assertFalse(validation.is_valid)
        self.assertIn('Secret key appears too short', validation.errors)

    def test_unusual_api_key_format_warning(self):
        """Test warning for unusual API key format."""
        data = {
            'api_key': 'XXXXXXXXXXXXXXXXXXXX',  # Doesn't start with PK/AK/CK
            'secret_key': 'secretkey1234567890123456789012'
        }

        validation = self.service._validate_broker_connection(data)

        self.assertTrue(validation.is_valid)
        self.assertGreater(len(validation.warnings), 0)


class TestTradingModeValidation(unittest.TestCase):
    """Test trading mode step validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_user = Mock()

        from backend.auth0login.services.onboarding_flow import OnboardingFlowService
        self.service = OnboardingFlowService(self.mock_user)

    def test_valid_paper_mode(self):
        """Test validation with paper trading mode."""
        validation = self.service._validate_trading_mode({'trading_mode': 'paper'})
        self.assertTrue(validation.is_valid)

    def test_valid_live_mode(self):
        """Test validation with live trading mode."""
        validation = self.service._validate_trading_mode({'trading_mode': 'live'})
        self.assertTrue(validation.is_valid)

    def test_invalid_mode(self):
        """Test validation with invalid mode."""
        validation = self.service._validate_trading_mode({'trading_mode': 'invalid'})
        self.assertFalse(validation.is_valid)
        self.assertIn("Trading mode must be 'paper' or 'live'", validation.errors)

    def test_missing_mode(self):
        """Test validation with missing mode."""
        validation = self.service._validate_trading_mode({})
        self.assertFalse(validation.is_valid)


class TestStrategySelectionValidation(unittest.TestCase):
    """Test strategy selection step validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_user = Mock()

        from backend.auth0login.services.onboarding_flow import OnboardingFlowService
        self.service = OnboardingFlowService(self.mock_user)

    def test_valid_single_strategy(self):
        """Test validation with single valid strategy."""
        validation = self.service._validate_strategy_selection({
            'strategies': ['wsb-dip-bot']
        })
        self.assertTrue(validation.is_valid)

    def test_valid_multiple_strategies(self):
        """Test validation with multiple valid strategies."""
        validation = self.service._validate_strategy_selection({
            'strategies': ['wsb-dip-bot', 'wheel', 'swing-trading']
        })
        self.assertTrue(validation.is_valid)

    def test_empty_strategies(self):
        """Test validation with no strategies selected."""
        validation = self.service._validate_strategy_selection({
            'strategies': []
        })
        self.assertFalse(validation.is_valid)
        self.assertIn('At least one strategy must be selected', validation.errors)

    def test_invalid_strategy(self):
        """Test validation with invalid strategy name."""
        validation = self.service._validate_strategy_selection({
            'strategies': ['invalid-strategy']
        })
        self.assertFalse(validation.is_valid)
        self.assertTrue(any('Invalid strategies' in e for e in validation.errors))

    def test_mixed_valid_invalid_strategies(self):
        """Test validation with mix of valid and invalid strategies."""
        validation = self.service._validate_strategy_selection({
            'strategies': ['wsb-dip-bot', 'fake-strategy']
        })
        self.assertFalse(validation.is_valid)

    def test_many_strategies_warning(self):
        """Test warning when selecting more than 5 strategies."""
        validation = self.service._validate_strategy_selection({
            'strategies': [
                'wsb-dip-bot', 'wheel', 'momentum-weeklies',
                'index-baseline', 'lotto-scanner', 'swing-trading'
            ]
        })
        self.assertTrue(validation.is_valid)
        self.assertGreater(len(validation.warnings), 0)


class TestRiskProfileValidation(unittest.TestCase):
    """Test risk profile step validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_user = Mock()

        from backend.auth0login.services.onboarding_flow import OnboardingFlowService
        self.service = OnboardingFlowService(self.mock_user)

    def test_valid_conservative_profile(self):
        """Test validation with conservative profile."""
        validation = self.service._validate_risk_profile({
            'risk_profile': 'conservative'
        })
        self.assertTrue(validation.is_valid)

    def test_valid_moderate_profile(self):
        """Test validation with moderate profile."""
        validation = self.service._validate_risk_profile({
            'risk_profile': 'moderate'
        })
        self.assertTrue(validation.is_valid)

    def test_valid_aggressive_profile(self):
        """Test validation with aggressive profile."""
        validation = self.service._validate_risk_profile({
            'risk_profile': 'aggressive'
        })
        self.assertTrue(validation.is_valid)

    def test_missing_profile(self):
        """Test validation with missing profile."""
        validation = self.service._validate_risk_profile({})
        self.assertFalse(validation.is_valid)
        self.assertIn('Risk profile is required', validation.errors)

    def test_invalid_profile(self):
        """Test validation with invalid profile."""
        validation = self.service._validate_risk_profile({
            'risk_profile': 'extreme'
        })
        self.assertFalse(validation.is_valid)
        self.assertIn('Invalid risk profile', validation.errors)


class TestReviewLaunchValidation(unittest.TestCase):
    """Test review and launch step validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_user = Mock()

        from backend.auth0login.services.onboarding_flow import OnboardingFlowService
        self.service = OnboardingFlowService(self.mock_user)

    def test_valid_with_terms_accepted(self):
        """Test validation with terms accepted."""
        validation = self.service._validate_review_launch({
            'terms_accepted': True
        })
        self.assertTrue(validation.is_valid)

    def test_invalid_without_terms(self):
        """Test validation without terms accepted."""
        validation = self.service._validate_review_launch({
            'terms_accepted': False
        })
        self.assertFalse(validation.is_valid)
        self.assertIn('Terms of service must be accepted', validation.errors)


class TestValidateStep(unittest.TestCase):
    """Test the validate_step dispatcher method."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_user = Mock()

        from backend.auth0login.services.onboarding_flow import OnboardingFlowService
        self.service = OnboardingFlowService(self.mock_user)

    def test_validate_step_1(self):
        """Test validate_step routes to broker validation."""
        validation = self.service.validate_step(1, {
            'api_key': 'PKXXXXXXXXXXXXXXXX',
            'secret_key': 'secretkey1234567890123456789012'
        })
        self.assertTrue(validation.is_valid)

    def test_validate_step_2(self):
        """Test validate_step routes to trading mode validation."""
        validation = self.service.validate_step(2, {'trading_mode': 'paper'})
        self.assertTrue(validation.is_valid)

    def test_validate_step_3(self):
        """Test validate_step routes to strategy validation."""
        validation = self.service.validate_step(3, {'strategies': ['wsb-dip-bot']})
        self.assertTrue(validation.is_valid)

    def test_validate_step_4(self):
        """Test validate_step routes to risk profile validation."""
        validation = self.service.validate_step(4, {'risk_profile': 'moderate'})
        self.assertTrue(validation.is_valid)

    def test_validate_step_5(self):
        """Test validate_step routes to review validation."""
        validation = self.service.validate_step(5, {'terms_accepted': True})
        self.assertTrue(validation.is_valid)

    def test_validate_invalid_step(self):
        """Test validate_step with invalid step number."""
        validation = self.service.validate_step(99, {})
        self.assertFalse(validation.is_valid)
        self.assertIn('Unknown step: 99', validation.errors)


class TestProcessTradingMode(unittest.TestCase):
    """Test trading mode step processing."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_user = Mock()

        from backend.auth0login.services.onboarding_flow import OnboardingFlowService, WizardStep
        self.service = OnboardingFlowService(self.mock_user)
        self.WizardStep = WizardStep

    def test_process_paper_mode(self):
        """Test processing paper trading mode."""
        result = self.service._process_trading_mode({'trading_mode': 'paper'})

        self.assertTrue(result.success)
        self.assertEqual(result.data['trading_mode'], 'paper')
        self.assertEqual(result.next_step, self.WizardStep.STRATEGY_SELECTION)

    def test_process_live_mode(self):
        """Test processing live trading mode."""
        result = self.service._process_trading_mode({'trading_mode': 'live'})

        self.assertTrue(result.success)
        self.assertEqual(result.data['trading_mode'], 'live')


class TestProcessStrategySelection(unittest.TestCase):
    """Test strategy selection step processing."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_user = Mock()

        from backend.auth0login.services.onboarding_flow import OnboardingFlowService, WizardStep
        self.service = OnboardingFlowService(self.mock_user)
        self.WizardStep = WizardStep

    def test_process_strategies(self):
        """Test processing strategy selection."""
        strategies = ['wsb-dip-bot', 'wheel']
        result = self.service._process_strategy_selection({'strategies': strategies})

        self.assertTrue(result.success)
        self.assertEqual(result.data['strategies'], strategies)
        self.assertEqual(result.next_step, self.WizardStep.RISK_PROFILE)


class TestHelperFunctions(unittest.TestCase):
    """Test module-level helper functions."""

    @patch('backend.auth0login.models.OnboardingSession')
    @patch('backend.auth0login.models.WizardConfiguration')
    def test_is_first_time_user_no_config(self, mock_config, mock_session):
        """Test is_first_time_user returns True when no config exists."""
        from backend.auth0login.services.onboarding_flow import is_first_time_user

        mock_config.objects.filter.return_value.exists.return_value = False
        mock_session.objects.filter.return_value.exists.return_value = False

        mock_user = Mock()
        result = is_first_time_user(mock_user)

        self.assertTrue(result)

    @patch('backend.auth0login.models.OnboardingSession')
    @patch('backend.auth0login.models.WizardConfiguration')
    def test_is_first_time_user_has_config(self, mock_config, mock_session):
        """Test is_first_time_user returns False when config exists."""
        from backend.auth0login.services.onboarding_flow import is_first_time_user

        mock_config.objects.filter.return_value.exists.return_value = True

        mock_user = Mock()
        result = is_first_time_user(mock_user)

        self.assertFalse(result)

    def test_get_onboarding_service(self):
        """Test get_onboarding_service factory function."""
        from backend.auth0login.services.onboarding_flow import (
            get_onboarding_service,
            OnboardingFlowService
        )

        mock_user = Mock()
        service = get_onboarding_service(mock_user)

        self.assertIsInstance(service, OnboardingFlowService)
        self.assertEqual(service.user, mock_user)


class TestSkipWizard(unittest.TestCase):
    """Test wizard skip functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_user = Mock()

        from backend.auth0login.services.onboarding_flow import OnboardingFlowService
        self.service = OnboardingFlowService(self.mock_user)

    @patch.object(
        __import__('backend.auth0login.services.onboarding_flow', fromlist=['OnboardingFlowService']).OnboardingFlowService,
        '_can_skip_to_end'
    )
    @patch.object(
        __import__('backend.auth0login.services.onboarding_flow', fromlist=['OnboardingFlowService']).OnboardingFlowService,
        'get_or_create_session'
    )
    def test_skip_wizard_allowed(self, mock_get_session, mock_can_skip):
        """Test skipping wizard when allowed."""
        mock_can_skip.return_value = True
        mock_session = Mock()
        mock_get_session.return_value = mock_session

        result = self.service.skip_wizard()

        self.assertTrue(result.success)
        self.assertEqual(mock_session.status, 'skipped')

    @patch.object(
        __import__('backend.auth0login.services.onboarding_flow', fromlist=['OnboardingFlowService']).OnboardingFlowService,
        '_can_skip_to_end'
    )
    def test_skip_wizard_not_allowed(self, mock_can_skip):
        """Test skipping wizard when not allowed."""
        mock_can_skip.return_value = False

        result = self.service.skip_wizard()

        self.assertFalse(result.success)
        self.assertIn('Cannot skip wizard', result.message)


if __name__ == '__main__':
    unittest.main()
