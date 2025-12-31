"""Comprehensive tests for PortfolioBuilderService."""

import pytest
from unittest.mock import Mock, patch
from decimal import Decimal
from django.contrib.auth.models import User

from backend.auth0login.services.portfolio_builder import (
    PortfolioBuilderService,
    AVAILABLE_STRATEGIES,
    PORTFOLIO_TEMPLATES,
    get_portfolio_builder
)


@pytest.fixture
def user():
    return User.objects.create_user(username='testuser', email='test@example.com')


@pytest.fixture
def service(user):
    return PortfolioBuilderService(user=user)


class TestInit:
    """Test initialization."""

    def test_init_with_user(self, user):
        service = PortfolioBuilderService(user=user)
        assert service.user == user

    def test_init_without_user(self):
        service = PortfolioBuilderService()
        assert service.user is None


class TestGetAvailableStrategies:
    """Test getting available strategies."""

    def test_get_available_strategies(self, service):
        strategies = service.get_available_strategies()
        assert len(strategies) > 0
        assert 'wsb-dip-bot' in strategies


class TestGetPortfolioTemplates:
    """Test getting templates."""

    def test_get_portfolio_templates(self, service):
        templates = service.get_portfolio_templates()
        assert len(templates) > 0
        assert 'conservative-income' in templates


class TestGetTemplateDetails:
    """Test getting template details."""

    @patch.object(PortfolioBuilderService, 'analyze_portfolio')
    def test_get_template_details_found(self, mock_analyze, service):
        mock_analyze.return_value = Mock(
            expected_return=0.12,
            expected_volatility=0.15,
            expected_sharpe=1.2,
            diversification_score=75.0,
            correlation_matrix={},
            risk_contribution={},
            warnings=[],
            recommendations=[]
        )

        result = service.get_template_details('conservative-income')
        assert result is not None
        assert result['template_id'] == 'conservative-income'

    def test_get_template_details_not_found(self, service):
        result = service.get_template_details('nonexistent')
        assert result is None


class TestCreateFromTemplate:
    """Test creating portfolio from template."""

    @patch('backend.auth0login.services.portfolio_builder.StrategyPortfolio')
    @patch.object(PortfolioBuilderService, 'analyze_portfolio')
    def test_create_from_template(self, mock_analyze, mock_model, service, user):
        mock_analyze.return_value = Mock(
            correlation_matrix={},
            diversification_score=75.0,
            expected_sharpe=1.2
        )
        mock_portfolio = Mock()
        mock_model.objects.create.return_value = mock_portfolio

        result = service.create_from_template('conservative-income')
        assert result == mock_portfolio

    @patch('backend.auth0login.services.portfolio_builder.StrategyPortfolio')
    def test_create_from_template_not_found(self, mock_model, service):
        with pytest.raises(ValueError):
            service.create_from_template('nonexistent')


class TestCreateCustomPortfolio:
    """Test creating custom portfolio."""

    @patch('backend.auth0login.services.portfolio_builder.StrategyPortfolio')
    @patch.object(PortfolioBuilderService, 'analyze_portfolio')
    def test_create_custom_portfolio(self, mock_analyze, mock_model, service, user):
        mock_analyze.return_value = Mock(
            correlation_matrix={},
            diversification_score=70.0,
            expected_sharpe=1.1
        )
        mock_portfolio = Mock()
        mock_model.objects.create.return_value = mock_portfolio

        strategies = {
            'wheel': {'allocation_pct': 50, 'enabled': True},
            'wsb-dip-bot': {'allocation_pct': 50, 'enabled': True}
        }

        result = service.create_custom_portfolio('My Portfolio', strategies)
        assert result == mock_portfolio

    @patch('backend.auth0login.services.portfolio_builder.StrategyPortfolio')
    def test_create_custom_portfolio_invalid_strategy(self, mock_model, service):
        strategies = {'invalid-strategy': {'allocation_pct': 100, 'enabled': True}}

        with pytest.raises(ValueError):
            service.create_custom_portfolio('Test', strategies)


class TestAnalyzePortfolio:
    """Test portfolio analysis."""

    def test_analyze_portfolio_basic(self, service):
        strategies = {
            'wheel': {'allocation_pct': 50, 'enabled': True},
            'wsb-dip-bot': {'allocation_pct': 50, 'enabled': True}
        }

        result = service.analyze_portfolio(strategies)

        assert result.expected_return > 0
        assert result.expected_volatility > 0
        assert result.diversification_score >= 0
        assert isinstance(result.warnings, list)

    def test_analyze_portfolio_under_allocated(self, service):
        strategies = {'wheel': {'allocation_pct': 50, 'enabled': True}}

        result = service.analyze_portfolio(strategies)

        assert any('below 95%' in w for w in result.warnings)

    def test_analyze_portfolio_over_allocated(self, service):
        strategies = {
            'wheel': {'allocation_pct': 60, 'enabled': True},
            'wsb-dip-bot': {'allocation_pct': 60, 'enabled': True}
        }

        result = service.analyze_portfolio(strategies)

        assert any('exceeds 100%' in w for w in result.warnings)


class TestBuildCorrelationMatrix:
    """Test correlation matrix building."""

    def test_build_correlation_matrix(self, service):
        strategies = {
            'wheel': {'allocation_pct': 50, 'enabled': True},
            'wsb-dip-bot': {'allocation_pct': 50, 'enabled': True}
        }

        result = service._build_correlation_matrix(strategies)

        assert 'wheel' in result
        assert 'wsb-dip-bot' in result
        assert result['wheel']['wheel'] == 1.0


class TestCalculateDiversificationScore:
    """Test diversification score calculation."""

    def test_calculate_diversification_score_single_strategy(self, service):
        strategies = {'wheel': {'allocation_pct': 100, 'enabled': True}}
        corr_matrix = service._build_correlation_matrix(strategies)

        score = service._calculate_diversification_score(strategies, corr_matrix)

        assert score == 50.0

    def test_calculate_diversification_score_multiple_strategies(self, service):
        strategies = {
            'wheel': {'allocation_pct': 33, 'enabled': True},
            'wsb-dip-bot': {'allocation_pct': 33, 'enabled': True},
            'momentum-weeklies': {'allocation_pct': 34, 'enabled': True}
        }
        corr_matrix = service._build_correlation_matrix(strategies)

        score = service._calculate_diversification_score(strategies, corr_matrix)

        assert score > 0


class TestOptimizePortfolio:
    """Test portfolio optimization."""

    def test_optimize_portfolio_conservative(self, service):
        strategies = ['wheel', 'spx-credit-spreads', 'leaps-tracker']

        result = service.optimize_portfolio(strategies, 'conservative', Decimal('10000'))

        assert len(result) == 3
        assert all(s in result for s in strategies)
        total_allocation = sum(c['allocation_pct'] for c in result.values())
        assert 95 <= total_allocation <= 100

    def test_optimize_portfolio_aggressive(self, service):
        strategies = ['wsb-dip-bot', 'momentum-weeklies', 'lotto-scanner']

        result = service.optimize_portfolio(strategies, 'aggressive', Decimal('10000'))

        assert len(result) == 3

    def test_optimize_portfolio_empty(self, service):
        result = service.optimize_portfolio([], 'moderate')
        assert result == {}


class TestSuggestAdditions:
    """Test strategy addition suggestions."""

    def test_suggest_additions_missing_groups(self, service):
        current = ['wsb-dip-bot', 'momentum-weeklies']

        suggestions = service.suggest_additions(current, 'moderate')

        assert isinstance(suggestions, list)
        assert len(suggestions) <= 3

    def test_suggest_additions_complete_portfolio(self, service):
        current = list(AVAILABLE_STRATEGIES.keys())

        suggestions = service.suggest_additions(current, 'moderate')

        assert len(suggestions) == 0


class TestGetPortfolios:
    """Test getting user portfolios."""

    @patch('backend.auth0login.services.portfolio_builder.StrategyPortfolio')
    def test_get_user_portfolios(self, mock_model, service, user):
        mock_qs = Mock()
        mock_qs.order_by.return_value = []
        mock_model.objects.filter.return_value = mock_qs

        result = service.get_user_portfolios()

        assert isinstance(result, list)

    @patch('backend.auth0login.services.portfolio_builder.StrategyPortfolio')
    def test_get_active_portfolio(self, mock_model, service, user):
        mock_portfolio = Mock()
        mock_model.objects.filter.return_value.first.return_value = mock_portfolio

        result = service.get_active_portfolio()

        assert result == mock_portfolio


class TestFactoryFunction:
    """Test factory function."""

    def test_get_portfolio_builder(self, user):
        service = get_portfolio_builder(user)
        assert isinstance(service, PortfolioBuilderService)
        assert service.user == user
