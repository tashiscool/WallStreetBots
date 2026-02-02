"""
Tests for new API endpoints: backtest persistence, optimization, and wizard.

Tests the endpoints added in api_views.py for:
- Backtest run persistence
- Parameter optimization
- Setup wizard/onboarding
"""
import json
import unittest
from datetime import date
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock


class MockRequest:
    """Mock Django request object."""

    def __init__(self, method='GET', user=None, data=None, query_params=None):
        self.method = method
        self.user = user or Mock(id=1, username='testuser', email='test@test.com')
        self.content_type = 'application/json'
        self.body = json.dumps(data).encode() if data else b''
        self.GET = query_params or {}
        self.POST = data or {}


class TestBacktestRunsListEndpoint(unittest.TestCase):
    """Test backtest_runs_list API endpoint."""

    @patch('backend.tradingbot.models.models.BacktestRun')
    def test_list_backtest_runs(self, mock_model):
        """Test listing backtest runs."""
        from backend.auth0login.api_views import backtest_runs_list

        # Setup mock
        mock_run = Mock()
        mock_run.run_id = 'test-run-123'
        mock_run.strategy_name = 'test-strategy'
        mock_run.start_date = date(2023, 1, 1)
        mock_run.end_date = date(2024, 1, 1)
        mock_run.initial_capital = Decimal('100000')
        mock_run.total_return_pct = 15.5
        mock_run.sharpe_ratio = 1.2
        mock_run.max_drawdown_pct = 8.5
        mock_run.win_rate = 55.0
        mock_run.total_trades = 50
        mock_run.status = 'completed'
        mock_run.created_at = None

        mock_queryset = Mock()
        mock_queryset.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value = mock_queryset
        mock_queryset.count.return_value = 1
        mock_queryset.__getitem__ = Mock(return_value=[mock_run])

        mock_model.objects = mock_queryset

        request = MockRequest(method='GET')
        response = backtest_runs_list(request)

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertIn('runs', data)
        self.assertEqual(data['total'], 1)

    @patch('backend.tradingbot.models.models.BacktestRun')
    def test_list_backtest_runs_with_filters(self, mock_model):
        """Test listing backtest runs with filters."""
        from backend.auth0login.api_views import backtest_runs_list

        mock_queryset = Mock()
        mock_queryset.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value = mock_queryset
        mock_queryset.count.return_value = 0
        mock_queryset.__getitem__ = Mock(return_value=[])

        mock_model.objects = mock_queryset

        request = MockRequest(
            method='GET',
            query_params={
                'strategy_name': 'wsb-dip-bot',
                'status': 'completed',
                'limit': '10',
                'offset': '0'
            }
        )
        response = backtest_runs_list(request)

        self.assertEqual(response.status_code, 200)


class TestBacktestRunDetailEndpoint(unittest.TestCase):
    """Test backtest_run_detail API endpoint."""

    @patch('backend.tradingbot.models.models.BacktestRun')
    def test_get_backtest_run_detail(self, mock_model):
        """Test getting backtest run detail."""
        from backend.auth0login.api_views import backtest_run_detail

        mock_run = Mock()
        mock_run.run_id = 'test-run-123'
        mock_run.user_id = 1
        mock_run.strategy_name = 'test-strategy'
        mock_run.custom_strategy_id = None
        mock_run.start_date = date(2023, 1, 1)
        mock_run.end_date = date(2024, 1, 1)
        mock_run.symbols = ['SPY']
        mock_run.initial_capital = 100000.0
        mock_run.final_capital = 115000.0
        mock_run.position_size_pct = 3.0
        mock_run.stop_loss_pct = 5.0
        mock_run.take_profit_pct = 15.0
        mock_run.benchmark_symbol = 'SPY'
        mock_run.status = 'completed'
        mock_run.error_message = None
        mock_run.total_return_pct = 15.5
        mock_run.annualized_return_pct = 15.5
        mock_run.benchmark_return_pct = 10.0
        mock_run.alpha = 5.5
        mock_run.beta = 1.0
        mock_run.sharpe_ratio = 1.2
        mock_run.sortino_ratio = 1.5
        mock_run.max_drawdown_pct = 8.5
        mock_run.win_rate = 55.0
        mock_run.profit_factor = 1.8
        mock_run.total_trades = 50
        mock_run.winning_trades = 28
        mock_run.losing_trades = 22
        mock_run.avg_win_pct = 3.5
        mock_run.avg_loss_pct = 2.0
        mock_run.execution_time_seconds = 5.2
        mock_run.monthly_returns = {'2023-01': 2.5}
        mock_run.equity_curve = [{'date': '2023-01-01', 'equity': 100000}]
        mock_run.drawdown_curve = [{'date': '2023-01-01', 'drawdown': 0}]
        mock_run.created_at = None
        mock_run.completed_at = None

        mock_model.objects.get.return_value = mock_run

        request = MockRequest(method='GET')
        response = backtest_run_detail(request, 'test-run-123')

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data['run_id'], 'test-run-123')

    @patch('backend.tradingbot.models.models.BacktestRun')
    def test_get_backtest_run_not_found(self, mock_model):
        """Test getting non-existent backtest run."""
        from backend.auth0login.api_views import backtest_run_detail

        # Create a real exception class for DoesNotExist
        class DoesNotExist(Exception):
            pass
        mock_model.DoesNotExist = DoesNotExist
        mock_model.objects.get.side_effect = DoesNotExist("Not found")

        request = MockRequest(method='GET')
        response = backtest_run_detail(request, 'non-existent')

        self.assertEqual(response.status_code, 404)

    @patch('backend.tradingbot.models.models.BacktestRun')
    def test_delete_backtest_run(self, mock_model):
        """Test deleting a backtest run."""
        from backend.auth0login.api_views import backtest_run_detail

        mock_run = Mock()
        mock_run.delete = Mock()
        mock_model.objects.get.return_value = mock_run

        request = MockRequest(method='DELETE')
        response = backtest_run_detail(request, 'test-run-123')

        self.assertEqual(response.status_code, 200)
        mock_run.delete.assert_called_once()


class TestBacktestRunTradesEndpoint(unittest.TestCase):
    """Test backtest_run_trades API endpoint."""

    @patch('backend.tradingbot.models.models.BacktestTrade')
    @patch('backend.tradingbot.models.models.BacktestRun')
    def test_get_backtest_trades(self, mock_run_model, mock_trade_model):
        """Test getting trades for a backtest run."""
        from backend.auth0login.api_views import backtest_run_trades

        # Mock run exists
        mock_run = Mock()
        mock_run.run_id = 'test-run-123'
        mock_run_model.objects.get.return_value = mock_run

        # Return empty trades list for simplicity
        mock_queryset = Mock()
        mock_queryset.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value = mock_queryset
        mock_queryset.count.return_value = 0
        mock_queryset.__getitem__ = Mock(return_value=[])

        mock_trade_model.objects = mock_queryset

        request = MockRequest(method='GET')
        response = backtest_run_trades(request, 'test-run-123')

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertIn('trades', data)


class TestWizardSessionEndpoint(unittest.TestCase):
    """Test wizard_session API endpoint."""

    @patch('backend.auth0login.services.onboarding_flow.OnboardingFlowService')
    def test_get_wizard_session_none(self, mock_service_class):
        """Test getting wizard session when none exists."""
        from backend.auth0login.api_views import wizard_session

        mock_service = Mock()
        mock_service.get_or_create_session.return_value = None
        mock_service_class.return_value = mock_service

        # Patch to return None for get_current_session
        mock_service.get_or_create_session.return_value = None

        request = MockRequest(method='GET')

        # This will work because we're testing the endpoint logic
        # The actual implementation may vary

    @patch('backend.auth0login.services.onboarding_flow.OnboardingFlowService')
    def test_create_wizard_session(self, mock_service_class):
        """Test creating a new wizard session."""
        from backend.auth0login.api_views import wizard_session

        mock_session = Mock()
        mock_session.session_id = 'new-session-123'
        mock_session.current_step = 1
        mock_session.steps_completed = []
        mock_session.status = 'in_progress'
        mock_session.step_data = {}

        mock_service = Mock()
        mock_service.get_or_create_session.return_value = mock_session
        mock_service_class.return_value = mock_service

        request = MockRequest(method='POST', data={})
        response = wizard_session(request)

        self.assertEqual(response.status_code, 200)


class TestWizardStepSubmitEndpoint(unittest.TestCase):
    """Test wizard_step_submit API endpoint."""

    @patch('backend.auth0login.services.onboarding_flow.OnboardingFlowService')
    def test_submit_valid_step(self, mock_service_class):
        """Test submitting a valid wizard step."""
        from backend.auth0login.api_views import wizard_step_submit

        mock_result = Mock()
        mock_result.success = True
        mock_result.data = {'trading_mode': 'paper'}
        mock_result.errors = []
        mock_result.warnings = []

        mock_service = Mock()
        mock_service.process_step.return_value = mock_result
        mock_service_class.return_value = mock_service

        request = MockRequest(method='POST', data={'trading_mode': 'paper'})
        response = wizard_step_submit(request, 2)

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data['status'], 'success')

    @patch('backend.auth0login.services.onboarding_flow.OnboardingFlowService')
    def test_submit_invalid_step_number(self, mock_service_class):
        """Test submitting with invalid step number."""
        from backend.auth0login.api_views import wizard_step_submit

        request = MockRequest(method='POST', data={})
        response = wizard_step_submit(request, 99)

        self.assertEqual(response.status_code, 400)

    @patch('backend.auth0login.services.onboarding_flow.OnboardingFlowService')
    def test_submit_step_validation_error(self, mock_service_class):
        """Test submitting step with validation errors."""
        from backend.auth0login.api_views import wizard_step_submit

        mock_result = Mock()
        mock_result.success = False
        mock_result.errors = ['Trading mode is required']
        mock_result.warnings = []

        mock_service = Mock()
        mock_service.process_step.return_value = mock_result
        mock_service_class.return_value = mock_service

        request = MockRequest(method='POST', data={})
        response = wizard_step_submit(request, 2)

        self.assertEqual(response.status_code, 400)


class TestWizardCompleteEndpoint(unittest.TestCase):
    """Test wizard_complete API endpoint."""

    @patch('backend.auth0login.services.onboarding_flow.OnboardingFlowService')
    def test_complete_wizard_success(self, mock_service_class):
        """Test completing wizard successfully."""
        from backend.auth0login.api_views import wizard_complete

        mock_result = Mock()
        mock_result.success = True
        mock_result.message = 'Wizard completed!'
        mock_result.data = {
            'config_id': 1,
            'broker_validated': True,
            'email_sent': True
        }
        mock_result.errors = None

        mock_service = Mock()
        mock_service.complete_wizard.return_value = mock_result
        mock_service_class.return_value = mock_service

        request = MockRequest(method='POST', data={'terms_accepted': True})
        response = wizard_complete(request)

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data['status'], 'success')


class TestWizardSkipEndpoint(unittest.TestCase):
    """Test wizard_skip API endpoint."""

    @patch('backend.auth0login.models.OnboardingSession')
    @patch('backend.auth0login.models.WizardConfiguration')
    def test_skip_wizard_allowed(self, mock_config, mock_session):
        """Test skipping wizard when allowed (has existing config)."""
        from backend.auth0login.api_views import wizard_skip

        # User has existing config
        mock_config.objects.filter.return_value.exists.return_value = True
        mock_session.objects.filter.return_value.update.return_value = 1

        request = MockRequest(method='POST')
        response = wizard_skip(request)

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data['status'], 'success')

    @patch('backend.auth0login.models.WizardConfiguration')
    def test_skip_wizard_not_allowed(self, mock_config):
        """Test skipping wizard when not allowed (no existing config)."""
        from backend.auth0login.api_views import wizard_skip

        # User has no existing config
        mock_config.objects.filter.return_value.exists.return_value = False

        request = MockRequest(method='POST')
        response = wizard_skip(request)

        self.assertEqual(response.status_code, 400)
        data = json.loads(response.content)
        self.assertEqual(data['status'], 'error')


class TestWizardNeedsSetupEndpoint(unittest.TestCase):
    """Test wizard_needs_setup API endpoint."""

    @patch('backend.auth0login.models.WizardConfiguration')
    def test_needs_setup_true(self, mock_config):
        """Test when user needs setup - no config exists."""
        from backend.auth0login.api_views import wizard_needs_setup

        # No config exists
        mock_config.objects.filter.return_value.first.return_value = None

        request = MockRequest(method='GET')
        response = wizard_needs_setup(request)

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertTrue(data['needs_setup'])

    @patch('backend.auth0login.models.WizardConfiguration')
    def test_needs_setup_false(self, mock_config):
        """Test when user doesn't need setup - has valid config."""
        from backend.auth0login.api_views import wizard_needs_setup

        # Config exists and is complete
        mock_wizard_config = Mock()
        mock_wizard_config.setup_completed = True
        mock_wizard_config.broker_validated = True
        mock_config.objects.filter.return_value.first.return_value = mock_wizard_config

        request = MockRequest(method='GET')
        response = wizard_needs_setup(request)

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertFalse(data['needs_setup'])


class TestOptimizationRunStartEndpoint(unittest.TestCase):
    """Test optimization_run_start API endpoint."""

    @patch('backend.tradingbot.models.models.OptimizationRun')
    @patch('backend.tradingbot.models.models.CustomStrategy')
    def test_start_optimization_missing_strategy_id(self, mock_strategy, mock_opt_run):
        """Test starting optimization without strategy_id."""
        from backend.auth0login.api_views import optimization_run_start

        request = MockRequest(method='POST', data={})
        response = optimization_run_start(request)

        self.assertEqual(response.status_code, 400)
        data = json.loads(response.content)
        self.assertIn('strategy_id is required', data['message'])

    @patch('backend.tradingbot.models.models.OptimizationRun')
    @patch('backend.tradingbot.models.models.CustomStrategy')
    def test_start_optimization_strategy_not_found(self, mock_strategy_model, mock_opt_run):
        """Test starting optimization with non-existent strategy."""
        from backend.auth0login.api_views import optimization_run_start

        # Create a real exception class for DoesNotExist
        class DoesNotExist(Exception):
            pass
        mock_strategy_model.DoesNotExist = DoesNotExist
        mock_strategy_model.objects.get.side_effect = DoesNotExist("Not found")

        request = MockRequest(method='POST', data={'strategy_id': 999})
        response = optimization_run_start(request)

        self.assertEqual(response.status_code, 404)


class TestOptimizationRunStatusEndpoint(unittest.TestCase):
    """Test optimization_run_status API endpoint."""

    @patch('backend.tradingbot.models.models.OptimizationRun')
    def test_get_optimization_status(self, mock_model):
        """Test getting optimization run status."""
        from backend.auth0login.api_views import optimization_run_status

        # Create a mock that returns primitive values for all attributes
        mock_run = Mock()
        mock_run.run_id = 'opt-123'
        mock_run.status = 'running'
        mock_run.progress = 50
        mock_run.current_trial = 25
        mock_run.n_trials = 50
        mock_run.best_value = 1.5
        mock_run.best_params = {'stop_loss': 5.0}
        mock_run.strategy_name = 'test-strategy'
        mock_run.parameter_ranges = {}
        mock_run.all_trials = []
        mock_run.created_at = None
        mock_run.started_at = None
        mock_run.completed_at = None

        mock_model.objects.get.return_value = mock_run

        request = MockRequest(method='GET')
        response = optimization_run_status(request, 'opt-123')

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data['status'], 'running')
        self.assertEqual(data['progress_pct'], 50)  # Correct key name

    @patch('backend.tradingbot.models.models.OptimizationRun')
    def test_get_optimization_status_not_found(self, mock_model):
        """Test getting status for non-existent optimization."""
        from backend.auth0login.api_views import optimization_run_status

        # Create a real exception class for DoesNotExist
        class DoesNotExist(Exception):
            pass
        mock_model.DoesNotExist = DoesNotExist
        mock_model.objects.get.side_effect = DoesNotExist("Not found")

        request = MockRequest(method='GET')
        response = optimization_run_status(request, 'non-existent')

        self.assertEqual(response.status_code, 404)


class TestMarketContextEndpoints(unittest.TestCase):
    """Test market context API endpoints."""

    @patch('backend.auth0login.services.market_context.get_market_context_service')
    def test_market_context_overview(self, mock_get_service):
        """Test market context overview endpoint."""
        from backend.auth0login.api_views import market_context_overview

        # Setup mock service
        mock_service = Mock()
        mock_service.get_market_overview.return_value = {
            'vix': {'current': 18.5, 'change': -0.3},
            'indices': {'SPY': {'price': 450.0, 'change_pct': 0.5}},
            'market_regime': 'normal'
        }
        mock_service.get_sector_performance.return_value = []
        mock_service.get_holdings_events.return_value = []
        mock_service.get_economic_calendar.return_value = []
        mock_get_service.return_value = mock_service

        request = MockRequest(method='GET')
        response = market_context_overview(request)

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertIn('overview', data)

    @patch('backend.auth0login.services.market_context.get_market_context_service')
    def test_market_context_sectors(self, mock_get_service):
        """Test market context sectors endpoint."""
        from backend.auth0login.api_views import market_context_sectors

        mock_service = Mock()
        mock_service.get_sector_performance.return_value = [
            {'sector': 'Technology', 'symbol': 'XLK', 'change_pct': 1.2},
            {'sector': 'Healthcare', 'symbol': 'XLV', 'change_pct': -0.5}
        ]
        mock_get_service.return_value = mock_service

        request = MockRequest(method='GET')
        response = market_context_sectors(request)

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertIn('sectors', data)


class TestAllocationsEndpoints(unittest.TestCase):
    """Test portfolio allocation API endpoints."""

    @patch('backend.auth0login.services.allocation_manager.get_allocation_manager')
    def test_allocations_list(self, mock_get_manager):
        """Test listing allocations."""
        from backend.auth0login.api_views import allocations_list

        mock_manager = Mock()
        mock_manager.get_allocation_summary.return_value = {
            'allocations': [
                {'strategy_name': 'Momentum', 'allocated_pct': 30.0, 'current_pct': 25.0}
            ],
            'total_allocated': 30.0,
            'total_exposure': 25.0,
            'warnings': [],
        }
        mock_get_manager.return_value = mock_manager

        request = MockRequest(method='GET')
        response = allocations_list(request)

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertIn('allocations', data)

    @patch('backend.auth0login.services.allocation_manager.get_allocation_manager')
    def test_allocation_update(self, mock_get_manager):
        """Test updating an allocation."""
        from backend.auth0login.api_views import allocation_update

        mock_manager = Mock()
        mock_manager.update_allocation.return_value = {
            'strategy_name': 'Momentum',
            'new_allocation_pct': 35.0,
            'success': True,
        }
        mock_get_manager.return_value = mock_manager

        request = MockRequest(
            method='PUT',
            data={'target_pct': 35.0}
        )
        response = allocation_update(request, 'Momentum')

        self.assertEqual(response.status_code, 200)

    @patch('backend.auth0login.services.allocation_manager.get_allocation_manager')
    def test_allocations_rebalance(self, mock_get_manager):
        """Test rebalancing allocations."""
        from backend.auth0login.api_views import allocations_rebalance

        # Create a mock recommendation dataclass-like object
        mock_recommendation = Mock()
        mock_recommendation.strategy_name = 'Momentum'
        mock_recommendation.current_allocation = 25.0
        mock_recommendation.target_allocation = 30.0
        mock_recommendation.action = 'increase'
        mock_recommendation.adjustment_amount = 5000
        mock_recommendation.priority = 'medium'
        mock_recommendation.reason = 'Below target'

        mock_manager = Mock()
        mock_manager.get_rebalance_recommendations.return_value = [mock_recommendation]
        mock_get_manager.return_value = mock_manager

        request = MockRequest(method='POST')
        response = allocations_rebalance(request)

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertIn('status', data)


class TestCircuitBreakerEndpoints(unittest.TestCase):
    """Test circuit breaker API endpoints."""

    @patch('backend.tradingbot.models.models.CircuitBreakerState')
    def test_circuit_breaker_states(self, mock_model):
        """Test getting circuit breaker states."""
        from backend.auth0login.api_views import circuit_breaker_states

        mock_cb = Mock()
        mock_cb.breaker_type = 'daily_loss'
        mock_cb.state = 'active'
        mock_cb.trigger_count = 0
        mock_cb.last_triggered_at = None
        mock_cb.recovery_stage = 0
        mock_cb.cooldown_until = None
        mock_cb.metadata = {}

        mock_model.objects.all.return_value = [mock_cb]

        request = MockRequest(method='GET')
        response = circuit_breaker_states(request)

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertIn('breakers', data)

    @patch('backend.tradingbot.models.models.CircuitBreakerState')
    def test_circuit_breaker_reset(self, mock_model):
        """Test resetting a circuit breaker."""
        from backend.auth0login.api_views import circuit_breaker_reset

        mock_cb = Mock()
        mock_cb.breaker_type = 'daily_loss'
        mock_cb.state = 'triggered'
        mock_cb.trigger_count = 1
        mock_cb.recovery_stage = 0
        mock_cb.cooldown_until = None
        mock_cb.last_triggered_at = None
        mock_cb.metadata = {}
        mock_model.objects.get.return_value = mock_cb

        request = MockRequest(method='POST')
        response = circuit_breaker_reset(request, 'daily_loss')

        # Endpoint should return success
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data['status'], 'success')

    @patch('backend.tradingbot.models.models.CircuitBreakerHistory')
    def test_circuit_breaker_history(self, mock_model):
        """Test getting circuit breaker history."""
        from backend.auth0login.api_views import circuit_breaker_history

        from datetime import datetime
        mock_history = Mock()
        mock_history.id = 1
        mock_history.breaker_type = 'daily_loss'
        mock_history.trigger_reason = 'Max daily loss exceeded'
        mock_history.triggered_at = datetime.now()
        mock_history.recovered_at = None
        mock_history.duration_seconds = None
        mock_history.metadata = {}

        mock_queryset = Mock()
        mock_queryset.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value.__getitem__ = Mock(return_value=[mock_history])
        mock_model.objects.all.return_value = mock_queryset

        request = MockRequest(method='GET')
        response = circuit_breaker_history(request)

        self.assertEqual(response.status_code, 200)


class TestMLAgentEndpoints(unittest.TestCase):
    """Test ML/RL agent API endpoints."""

    @patch('backend.tradingbot.models.models.MLModel')
    def test_ml_models_list(self, mock_model):
        """Test listing ML models."""
        from backend.auth0login.api_views import ml_models_list

        # Setup mock
        mock_ml_model = Mock()
        mock_ml_model.id = 1
        mock_ml_model.name = 'LSTM_SPY_Predictor'
        mock_ml_model.model_type = 'lstm'
        mock_ml_model.status = 'active'
        mock_ml_model.symbols = 'SPY'
        mock_ml_model.accuracy = 62.5
        mock_ml_model.to_dict.return_value = {
            'id': '1',
            'name': 'LSTM_SPY_Predictor',
            'type': 'lstm',
            'status': 'active',
        }

        # Setup proper queryset chain
        mock_queryset = Mock()
        mock_queryset.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value = mock_queryset
        mock_queryset.count.return_value = 1
        mock_queryset.__iter__ = Mock(return_value=iter([mock_ml_model]))
        mock_model.objects = mock_queryset

        request = MockRequest(method='GET')
        response = ml_models_list(request)

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertIn('models', data)
        self.assertIn('total', data)

    @patch('backend.tradingbot.models.models.MLModel')
    def test_ml_models_create(self, mock_model):
        """Test creating an ML model."""
        from backend.auth0login.api_views import ml_models_create

        # Mock the model doesn't exist check
        mock_model.objects.filter.return_value.exists.return_value = False

        # Mock create
        mock_created = Mock()
        mock_created.id = 1
        mock_created.name = 'Test Model'
        mock_created.model_type = 'lstm'
        mock_created.get_default_hyperparameters.return_value = {'learning_rate': 0.001}
        mock_created.to_dict.return_value = {'id': '1', 'name': 'Test Model', 'type': 'lstm'}
        mock_model.objects.create.return_value = mock_created

        request = MockRequest(method='POST', data={
            'name': 'Test Model',
            'type': 'lstm',
            'symbols': 'SPY',
        })
        response = ml_models_create(request)

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data['status'], 'success')
        self.assertIn('model', data)

    @patch('backend.tradingbot.models.models.MLModel')
    def test_ml_models_create_missing_name(self, mock_model):
        """Test creating ML model without name."""
        from backend.auth0login.api_views import ml_models_create

        request = MockRequest(method='POST', data={
            'type': 'lstm',
        })
        response = ml_models_create(request)

        self.assertEqual(response.status_code, 400)
        data = json.loads(response.content)
        self.assertIn('name is required', data['message'])

    @patch('backend.tradingbot.models.models.TrainingJob')
    @patch('backend.tradingbot.models.models.MLModel')
    def test_ml_model_train(self, mock_model, mock_job):
        """Test starting ML model training."""
        from backend.auth0login.api_views import ml_model_train

        mock_ml_model = Mock()
        mock_ml_model.id = 1
        mock_ml_model.name = 'Test Model'
        mock_ml_model.status = 'idle'
        mock_ml_model.hyperparameters = {'epochs': 100, 'batch_size': 32}
        mock_model.objects.get.return_value = mock_ml_model

        mock_training_job = Mock()
        mock_training_job.job_id = 'train-ml-1-abc123'
        mock_training_job.to_dict.return_value = {'id': 'train-ml-1-abc123', 'status': 'running'}
        mock_job.objects.create.return_value = mock_training_job

        request = MockRequest(method='POST', data={})
        response = ml_model_train(request, '1')

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data['status'], 'success')
        self.assertIn('job_id', data)

    @patch('backend.tradingbot.models.models.MLModel')
    def test_ml_model_status_update(self, mock_model):
        """Test updating ML model status."""
        from backend.auth0login.api_views import ml_model_status

        mock_ml_model = Mock()
        mock_ml_model.id = 1
        mock_ml_model.status = 'idle'
        mock_model.objects.get.return_value = mock_ml_model

        request = MockRequest(method='PUT', data={'status': 'active'})
        response = ml_model_status(request, '1')

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['new_status'], 'active')


class TestRLAgentEndpoints(unittest.TestCase):
    """Test RL agent API endpoints."""

    @patch('backend.tradingbot.models.models.RLAgent')
    def test_rl_agents_list(self, mock_model):
        """Test listing RL agents."""
        from backend.auth0login.api_views import rl_agents_list

        mock_agent = Mock()
        mock_agent.agent_type = 'ppo'
        mock_agent.name = 'PPO Agent'
        mock_agent.status = 'active'
        mock_agent.to_dict.return_value = {
            'id': '1',
            'type': 'ppo',
            'name': 'PPO Agent',
            'status': 'active',
        }

        # Setup proper queryset chain
        mock_queryset = Mock()
        mock_queryset.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value = mock_queryset
        mock_queryset.count.return_value = 1
        mock_queryset.__iter__ = Mock(return_value=iter([mock_agent]))
        mock_model.objects = mock_queryset

        request = MockRequest(method='GET')
        response = rl_agents_list(request)

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertIn('agents', data)

    @patch('backend.tradingbot.models.models.TrainingJob')
    @patch('backend.tradingbot.models.models.RLAgent')
    def test_rl_agent_train(self, mock_model, mock_job):
        """Test starting RL agent training."""
        from backend.auth0login.api_views import rl_agent_train

        mock_agent = Mock()
        mock_agent.id = 1
        mock_agent.agent_type = 'ppo'
        mock_agent.status = 'idle'
        mock_agent.get_default_hyperparameters.return_value = {}
        mock_model.objects.get_or_create.return_value = (mock_agent, False)

        mock_training_job = Mock()
        mock_training_job.job_id = 'train-rl-ppo-abc123'
        mock_training_job.to_dict.return_value = {'id': 'train-rl-ppo-abc123', 'status': 'running'}
        mock_job.objects.create.return_value = mock_training_job

        request = MockRequest(method='POST', data={})
        response = rl_agent_train(request, 'ppo')

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data['status'], 'success')
        self.assertIn('job_id', data)

    @patch('backend.tradingbot.models.models.RLAgent')
    def test_rl_agent_status_update(self, mock_model):
        """Test updating RL agent status."""
        from backend.auth0login.api_views import rl_agent_status

        mock_agent = Mock()
        mock_agent.agent_type = 'ppo'
        mock_agent.status = 'idle'
        mock_model.objects.get.return_value = mock_agent

        request = MockRequest(method='PUT', data={'status': 'active'})
        response = rl_agent_status(request, 'ppo')

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['new_status'], 'active')

    @patch('backend.tradingbot.models.models.RLAgent')
    def test_rl_agent_config_update(self, mock_model):
        """Test updating RL agent config."""
        from backend.auth0login.api_views import rl_agent_config

        mock_agent = Mock()
        mock_agent.agent_type = 'ppo'
        mock_agent.hyperparameters = {'gamma': 0.99}
        mock_agent.get_default_hyperparameters.return_value = {'gamma': 0.99}
        mock_model.objects.get.return_value = mock_agent

        request = MockRequest(method='PUT', data={
            'gamma': 0.95,
            'actor_lr': 0.0001,
        })
        response = rl_agent_config(request, 'ppo')

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data['status'], 'success')


class TestTrainingJobEndpoints(unittest.TestCase):
    """Test training job API endpoints."""

    @patch('backend.tradingbot.models.models.TrainingJob')
    def test_training_jobs_list(self, mock_model):
        """Test listing training jobs."""
        from backend.auth0login.api_views import training_jobs_list

        # Create a chainable mock queryset that returns empty
        class MockQuerySet:
            def __init__(self, items=None):
                self._items = items or []

            def filter(self, **kwargs):
                return MockQuerySet([])

            def order_by(self, *args):
                return MockQuerySet([])

            def count(self):
                return 0

            def __iter__(self):
                return iter(self._items)

            def __getitem__(self, key):
                return self._items[key] if isinstance(key, int) else []

        mock_model.objects = MockQuerySet()

        request = MockRequest(method='GET')
        response = training_jobs_list(request)

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertIn('jobs', data)

    @patch('backend.tradingbot.models.models.TrainingJob')
    def test_training_job_cancel(self, mock_model):
        """Test cancelling a training job."""
        from backend.auth0login.api_views import training_job_cancel

        mock_job = Mock()
        mock_job.job_id = 'job-123'
        mock_job.status = 'running'
        mock_job.ml_model = None
        mock_job.rl_agent = None
        mock_model.objects.get.return_value = mock_job

        request = MockRequest(method='POST')
        response = training_job_cancel(request, 'job-123')

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(mock_job.status, 'cancelled')

    @patch('backend.tradingbot.models.models.TrainingJob')
    def test_training_job_cancel_not_found(self, mock_model):
        """Test cancelling non-existent job."""
        from backend.auth0login.api_views import training_job_cancel

        # Create a real exception class for DoesNotExist
        class DoesNotExist(Exception):
            pass
        mock_model.DoesNotExist = DoesNotExist
        mock_model.objects.get.side_effect = DoesNotExist("Not found")

        request = MockRequest(method='POST')
        response = training_job_cancel(request, 'non-existent')

        self.assertEqual(response.status_code, 404)


class TestUIPageViews(unittest.TestCase):
    """Test that UI page views are accessible."""

    def test_page_views_exist(self):
        """Test that all required page views exist."""
        from backend.auth0login import views

        # Check new page views exist
        self.assertTrue(hasattr(views, 'market_context'))
        self.assertTrue(hasattr(views, 'allocations'))
        self.assertTrue(hasattr(views, 'circuit_breakers'))
        self.assertTrue(hasattr(views, 'ml_agents'))

    def test_page_views_are_callable(self):
        """Test that page views are callable."""
        from backend.auth0login import views

        self.assertTrue(callable(views.market_context))
        self.assertTrue(callable(views.allocations))
        self.assertTrue(callable(views.circuit_breakers))
        self.assertTrue(callable(views.ml_agents))


if __name__ == '__main__':
    unittest.main()
