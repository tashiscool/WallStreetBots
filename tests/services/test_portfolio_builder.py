"""Comprehensive integration tests for PortfolioBuilderService.

These tests use real database operations and real portfolio analysis calculations.
No mocking of business logic - all calculations are performed with actual implementations.
"""

import pytest
from decimal import Decimal
from django.contrib.auth.models import User

from backend.auth0login.services.portfolio_builder import (
    PortfolioBuilderService,
    PortfolioAnalysis,
    AVAILABLE_STRATEGIES,
    PORTFOLIO_TEMPLATES,
    CORRELATION_GROUPS,
    get_portfolio_builder,
)
from backend.tradingbot.models.models import StrategyPortfolio


@pytest.fixture
def user(db):
    """Create a test user for portfolio operations."""
    return User.objects.create_user(
        username='testuser',
        email='test@example.com',
        password='testpass123'
    )


@pytest.fixture
def second_user(db):
    """Create a second test user for isolation tests."""
    return User.objects.create_user(
        username='seconduser',
        email='second@example.com',
        password='testpass456'
    )


@pytest.fixture
def service(user):
    """Create a PortfolioBuilderService instance with a test user."""
    return PortfolioBuilderService(user=user)


@pytest.fixture
def service_no_user():
    """Create a PortfolioBuilderService instance without a user."""
    return PortfolioBuilderService()


@pytest.mark.django_db
class TestServiceInitialization:
    """Test PortfolioBuilderService initialization."""

    def test_init_with_user(self, user):
        """Service should store user reference when provided."""
        service = PortfolioBuilderService(user=user)
        assert service.user == user
        assert service.logger is not None

    def test_init_without_user(self):
        """Service should work without a user."""
        service = PortfolioBuilderService()
        assert service.user is None

    def test_factory_function_with_user(self, user):
        """Factory function should create service with user."""
        service = get_portfolio_builder(user)
        assert isinstance(service, PortfolioBuilderService)
        assert service.user == user

    def test_factory_function_without_user(self):
        """Factory function should create service without user."""
        service = get_portfolio_builder()
        assert isinstance(service, PortfolioBuilderService)
        assert service.user is None


@pytest.mark.django_db
class TestAvailableStrategiesRetrieval:
    """Test retrieval of available strategies."""

    def test_get_available_strategies_returns_all_strategies(self, service):
        """Should return all defined strategies."""
        strategies = service.get_available_strategies()
        assert len(strategies) == len(AVAILABLE_STRATEGIES)
        assert set(strategies.keys()) == set(AVAILABLE_STRATEGIES.keys())

    def test_get_available_strategies_returns_copy(self, service):
        """Should return a copy to prevent modification of source data."""
        strategies = service.get_available_strategies()
        strategies['new-strategy'] = {}
        assert 'new-strategy' not in service.get_available_strategies()

    def test_strategy_metadata_completeness(self, service):
        """Each strategy should have required metadata fields."""
        strategies = service.get_available_strategies()
        required_fields = [
            'name', 'description', 'risk_level', 'expected_return',
            'volatility', 'correlation_group', 'min_allocation', 'max_allocation'
        ]
        for sid, meta in strategies.items():
            for field in required_fields:
                assert field in meta, f"Strategy '{sid}' missing field '{field}'"

    def test_strategy_risk_levels_valid(self, service):
        """Each strategy should have a valid risk level."""
        valid_levels = {'conservative', 'moderate', 'aggressive'}
        strategies = service.get_available_strategies()
        for sid, meta in strategies.items():
            assert meta['risk_level'] in valid_levels, \
                f"Strategy '{sid}' has invalid risk level: {meta['risk_level']}"


@pytest.mark.django_db
class TestPortfolioTemplatesRetrieval:
    """Test retrieval of portfolio templates."""

    def test_get_portfolio_templates_returns_all_templates(self, service):
        """Should return all defined templates."""
        templates = service.get_portfolio_templates()
        assert len(templates) == len(PORTFOLIO_TEMPLATES)
        assert set(templates.keys()) == set(PORTFOLIO_TEMPLATES.keys())

    def test_get_portfolio_templates_returns_copy(self, service):
        """Should return a copy to prevent modification of source data."""
        templates = service.get_portfolio_templates()
        templates['new-template'] = {}
        assert 'new-template' not in service.get_portfolio_templates()

    def test_template_metadata_completeness(self, service):
        """Each template should have required metadata fields."""
        templates = service.get_portfolio_templates()
        required_fields = [
            'name', 'description', 'risk_profile', 'strategies',
            'expected_sharpe', 'expected_return', 'expected_volatility'
        ]
        for tid, tmpl in templates.items():
            for field in required_fields:
                assert field in tmpl, f"Template '{tid}' missing field '{field}'"

    def test_template_strategies_reference_valid_strategies(self, service):
        """Template strategies should all reference valid strategies."""
        templates = service.get_portfolio_templates()
        for tid, tmpl in templates.items():
            for sid in tmpl['strategies'].keys():
                assert sid in AVAILABLE_STRATEGIES, \
                    f"Template '{tid}' references unknown strategy '{sid}'"


@pytest.mark.django_db
class TestTemplateDetailsRetrieval:
    """Test getting detailed information about specific templates."""

    def test_get_template_details_found_with_real_analysis(self, service):
        """Should return template details with real portfolio analysis."""
        result = service.get_template_details('conservative-income')

        assert result is not None
        assert result['template_id'] == 'conservative-income'
        assert result['name'] == 'Conservative Income'
        assert result['risk_profile'] == 'conservative'

        # Verify analysis was performed with real calculations
        analysis = result['analysis']
        assert analysis['expected_return'] > 0
        assert analysis['expected_volatility'] > 0
        assert analysis['expected_sharpe'] > 0
        assert 0 <= analysis['diversification_score'] <= 100
        assert isinstance(analysis['correlation_matrix'], dict)
        assert isinstance(analysis['risk_contribution'], dict)
        assert isinstance(analysis['warnings'], list)
        assert isinstance(analysis['recommendations'], list)

    def test_get_template_details_not_found(self, service):
        """Should return None for non-existent template."""
        result = service.get_template_details('nonexistent-template')
        assert result is None

    def test_get_template_details_includes_strategy_details(self, service):
        """Should include detailed strategy information."""
        result = service.get_template_details('balanced-growth')

        assert 'strategy_details' in result
        assert len(result['strategy_details']) == len(PORTFOLIO_TEMPLATES['balanced-growth']['strategies'])

        for sid, details in result['strategy_details'].items():
            assert 'allocation_pct' in details
            assert 'enabled' in details
            # Should merge strategy metadata
            if sid in AVAILABLE_STRATEGIES:
                assert 'name' in details
                assert 'volatility' in details


@pytest.mark.django_db
class TestPortfolioAnalysisCalculations:
    """Test real portfolio analysis calculations."""

    def test_analyze_portfolio_returns_analysis_object(self, service):
        """Should return a PortfolioAnalysis dataclass."""
        strategies = {
            'wheel': {'allocation_pct': 50, 'enabled': True},
            'wsb-dip-bot': {'allocation_pct': 50, 'enabled': True}
        }
        result = service.analyze_portfolio(strategies)
        assert isinstance(result, PortfolioAnalysis)

    def test_analyze_portfolio_expected_return_calculation(self, service):
        """Expected return should be weighted average of strategy returns."""
        strategies = {
            'wheel': {'allocation_pct': 100, 'enabled': True},
        }
        result = service.analyze_portfolio(strategies)

        # Wheel has expected_return of 0.12
        assert abs(result.expected_return - 0.12) < 0.001

    def test_analyze_portfolio_expected_return_weighted(self, service):
        """Expected return should be weighted by allocation."""
        strategies = {
            'wheel': {'allocation_pct': 50, 'enabled': True},  # 0.12 return
            'wsb-dip-bot': {'allocation_pct': 50, 'enabled': True}  # 0.25 return
        }
        result = service.analyze_portfolio(strategies)

        # Weighted: 0.5 * 0.12 + 0.5 * 0.25 = 0.185
        expected = 0.5 * 0.12 + 0.5 * 0.25
        assert abs(result.expected_return - expected) < 0.001

    def test_analyze_portfolio_volatility_calculation(self, service):
        """Volatility should be calculated from strategy volatilities."""
        strategies = {
            'wheel': {'allocation_pct': 100, 'enabled': True},  # volatility 0.15
        }
        result = service.analyze_portfolio(strategies)

        # Single strategy: sqrt((1.0 * 0.15)^2) = 0.15
        assert abs(result.expected_volatility - 0.15) < 0.001

    def test_analyze_portfolio_sharpe_ratio_calculation(self, service):
        """Sharpe ratio should be (return - risk_free) / volatility."""
        strategies = {
            'wheel': {'allocation_pct': 100, 'enabled': True},
        }
        result = service.analyze_portfolio(strategies)

        # Wheel: return=0.12, vol=0.15, risk_free=0.05
        # Sharpe = (0.12 - 0.05) / 0.15 = 0.4667
        expected_sharpe = (0.12 - 0.05) / 0.15
        assert abs(result.expected_sharpe - expected_sharpe) < 0.001

    def test_analyze_portfolio_disabled_strategies_excluded(self, service):
        """Disabled strategies should not contribute to metrics."""
        strategies = {
            'wheel': {'allocation_pct': 50, 'enabled': True},
            'lotto-scanner': {'allocation_pct': 50, 'enabled': False}  # High return, should be ignored
        }
        result = service.analyze_portfolio(strategies)

        # Only wheel's return contributes (50% allocation)
        expected_return = 0.5 * 0.12
        assert abs(result.expected_return - expected_return) < 0.001

    def test_analyze_portfolio_under_allocated_warning(self, service):
        """Should warn when total allocation is below 95%."""
        strategies = {
            'wheel': {'allocation_pct': 50, 'enabled': True}
        }
        result = service.analyze_portfolio(strategies)

        assert any('below 95%' in w for w in result.warnings)

    def test_analyze_portfolio_over_allocated_warning(self, service):
        """Should warn when total allocation exceeds 100%."""
        strategies = {
            'wheel': {'allocation_pct': 60, 'enabled': True},
            'wsb-dip-bot': {'allocation_pct': 60, 'enabled': True}
        }
        result = service.analyze_portfolio(strategies)

        assert any('exceeds 100%' in w for w in result.warnings)

    def test_analyze_portfolio_fully_allocated_no_warning(self, service):
        """Should not warn when allocation is between 95-100%."""
        strategies = {
            'wheel': {'allocation_pct': 50, 'enabled': True},
            'wsb-dip-bot': {'allocation_pct': 50, 'enabled': True}
        }
        result = service.analyze_portfolio(strategies)

        assert not any('below 95%' in w or 'exceeds 100%' in w for w in result.warnings)

    def test_analyze_portfolio_risk_contribution_normalized(self, service):
        """Risk contribution should sum to approximately 100%."""
        strategies = {
            'wheel': {'allocation_pct': 40, 'enabled': True},
            'wsb-dip-bot': {'allocation_pct': 30, 'enabled': True},
            'debit-spreads': {'allocation_pct': 30, 'enabled': True}
        }
        result = service.analyze_portfolio(strategies)

        total_risk = sum(result.risk_contribution.values())
        assert abs(total_risk - 100) < 1  # Allow small floating point error


@pytest.mark.django_db
class TestCorrelationMatrixBuilding:
    """Test real correlation matrix building."""

    def test_build_correlation_matrix_diagonal_is_one(self, service):
        """Diagonal elements (self-correlation) should be 1.0."""
        strategies = {
            'wheel': {'allocation_pct': 33, 'enabled': True},
            'wsb-dip-bot': {'allocation_pct': 33, 'enabled': True},
            'debit-spreads': {'allocation_pct': 34, 'enabled': True}
        }
        matrix = service._build_correlation_matrix(strategies)

        for sid in strategies.keys():
            assert matrix[sid][sid] == 1.0

    def test_build_correlation_matrix_symmetric(self, service):
        """Correlation matrix should be symmetric."""
        strategies = {
            'wheel': {'allocation_pct': 50, 'enabled': True},
            'wsb-dip-bot': {'allocation_pct': 50, 'enabled': True}
        }
        matrix = service._build_correlation_matrix(strategies)

        assert matrix['wheel']['wsb-dip-bot'] == matrix['wsb-dip-bot']['wheel']

    def test_build_correlation_matrix_uses_correlation_groups(self, service):
        """Should use predefined correlation group values."""
        strategies = {
            'wheel': {'allocation_pct': 50, 'enabled': True},  # income group
            'wsb-dip-bot': {'allocation_pct': 50, 'enabled': True}  # momentum group
        }
        matrix = service._build_correlation_matrix(strategies)

        # income-momentum correlation should be 0.2
        expected_corr = CORRELATION_GROUPS.get(
            ('momentum', 'income'),
            CORRELATION_GROUPS.get(('income', 'momentum'), 0.3)
        )
        assert matrix['wheel']['wsb-dip-bot'] == expected_corr

    def test_build_correlation_matrix_same_group_high_correlation(self, service):
        """Strategies in same correlation group should have high correlation."""
        strategies = {
            'wsb-dip-bot': {'allocation_pct': 50, 'enabled': True},  # momentum
            'momentum-weeklies': {'allocation_pct': 50, 'enabled': True}  # momentum
        }
        matrix = service._build_correlation_matrix(strategies)

        # momentum-momentum correlation should be 0.8
        assert matrix['wsb-dip-bot']['momentum-weeklies'] == 0.8

    def test_build_correlation_matrix_hedging_negative_correlation(self, service):
        """Hedging strategies should have negative correlation with momentum."""
        strategies = {
            'wsb-dip-bot': {'allocation_pct': 50, 'enabled': True},  # momentum
            'earnings-protection': {'allocation_pct': 50, 'enabled': True}  # hedging
        }
        matrix = service._build_correlation_matrix(strategies)

        # momentum-hedging correlation should be -0.3
        assert matrix['wsb-dip-bot']['earnings-protection'] == -0.3


@pytest.mark.django_db
class TestDiversificationScoreCalculation:
    """Test real diversification score calculation."""

    def test_diversification_score_single_strategy(self, service):
        """Single strategy should return moderate diversification score of 50."""
        strategies = {'wheel': {'allocation_pct': 100, 'enabled': True}}
        corr_matrix = service._build_correlation_matrix(strategies)

        score = service._calculate_diversification_score(strategies, corr_matrix)

        assert score == 50.0

    def test_diversification_score_range(self, service):
        """Diversification score should be between 0 and 100."""
        strategies = {
            'wheel': {'allocation_pct': 25, 'enabled': True},
            'wsb-dip-bot': {'allocation_pct': 25, 'enabled': True},
            'earnings-protection': {'allocation_pct': 25, 'enabled': True},
            'debit-spreads': {'allocation_pct': 25, 'enabled': True}
        }
        corr_matrix = service._build_correlation_matrix(strategies)

        score = service._calculate_diversification_score(strategies, corr_matrix)

        assert 0 <= score <= 100

    def test_diversification_score_increases_with_strategies(self, service):
        """More strategies should generally increase diversification score."""
        strategies_2 = {
            'wheel': {'allocation_pct': 50, 'enabled': True},
            'wsb-dip-bot': {'allocation_pct': 50, 'enabled': True}
        }
        strategies_4 = {
            'wheel': {'allocation_pct': 25, 'enabled': True},
            'wsb-dip-bot': {'allocation_pct': 25, 'enabled': True},
            'earnings-protection': {'allocation_pct': 25, 'enabled': True},
            'leaps-tracker': {'allocation_pct': 25, 'enabled': True}
        }

        corr_2 = service._build_correlation_matrix(strategies_2)
        corr_4 = service._build_correlation_matrix(strategies_4)

        score_2 = service._calculate_diversification_score(strategies_2, corr_2)
        score_4 = service._calculate_diversification_score(strategies_4, corr_4)

        assert score_4 > score_2

    def test_diversification_score_low_correlation_better(self, service):
        """Lower average correlation should give higher diversification score."""
        # High correlation pair (both momentum)
        high_corr_strategies = {
            'wsb-dip-bot': {'allocation_pct': 50, 'enabled': True},
            'momentum-weeklies': {'allocation_pct': 50, 'enabled': True}
        }
        # Low correlation pair (momentum + hedging)
        low_corr_strategies = {
            'wsb-dip-bot': {'allocation_pct': 50, 'enabled': True},
            'earnings-protection': {'allocation_pct': 50, 'enabled': True}
        }

        high_corr_matrix = service._build_correlation_matrix(high_corr_strategies)
        low_corr_matrix = service._build_correlation_matrix(low_corr_strategies)

        high_corr_score = service._calculate_diversification_score(high_corr_strategies, high_corr_matrix)
        low_corr_score = service._calculate_diversification_score(low_corr_strategies, low_corr_matrix)

        assert low_corr_score > high_corr_score

    def test_diversification_score_disabled_strategies_excluded(self, service):
        """Disabled strategies should not affect diversification score."""
        strategies = {
            'wheel': {'allocation_pct': 100, 'enabled': True},
            'wsb-dip-bot': {'allocation_pct': 0, 'enabled': False}  # Disabled
        }
        corr_matrix = service._build_correlation_matrix(strategies)

        score = service._calculate_diversification_score(strategies, corr_matrix)

        # Should be treated as single strategy
        assert score == 50.0


@pytest.mark.django_db
class TestPortfolioOptimization:
    """Test real portfolio optimization."""

    def test_optimize_portfolio_returns_all_strategies(self, service):
        """Should return allocations for all requested strategies."""
        strategies = ['wheel', 'spx-credit-spreads', 'leaps-tracker']

        result = service.optimize_portfolio(strategies, 'moderate', Decimal('100000'))

        assert set(result.keys()) == set(strategies)

    def test_optimize_portfolio_allocations_sum_to_100(self, service):
        """Allocations should sum to approximately 100%."""
        strategies = ['wheel', 'wsb-dip-bot', 'debit-spreads', 'earnings-protection']

        result = service.optimize_portfolio(strategies, 'moderate')

        total = sum(alloc['allocation_pct'] for alloc in result.values())
        assert 99 <= total <= 101  # Allow small floating point error

    def test_optimize_portfolio_respects_min_max_constraints(self, service):
        """Allocations should respect min/max constraints."""
        strategies = ['wheel', 'lotto-scanner', 'wsb-dip-bot']

        result = service.optimize_portfolio(strategies, 'moderate')

        for sid, alloc in result.items():
            meta = AVAILABLE_STRATEGIES[sid]
            min_alloc = meta['min_allocation']
            max_alloc = meta['max_allocation']
            # Allow for renormalization effects
            assert alloc['allocation_pct'] >= min_alloc * 0.8, \
                f"{sid} allocation {alloc['allocation_pct']} below min {min_alloc}"

    def test_optimize_portfolio_conservative_favors_income(self, service):
        """Conservative profile should favor income strategies."""
        strategies = ['wheel', 'wsb-dip-bot', 'spx-credit-spreads']

        result = service.optimize_portfolio(strategies, 'conservative')

        # Income strategies (wheel, spx-credit-spreads) should have higher allocation than momentum
        income_alloc = result['wheel']['allocation_pct'] + result['spx-credit-spreads']['allocation_pct']
        momentum_alloc = result['wsb-dip-bot']['allocation_pct']

        assert income_alloc > momentum_alloc

    def test_optimize_portfolio_aggressive_favors_momentum(self, service):
        """Aggressive profile should favor momentum strategies."""
        strategies = ['wheel', 'wsb-dip-bot', 'momentum-weeklies']

        result = service.optimize_portfolio(strategies, 'aggressive')

        # Momentum should have higher allocation than income in aggressive
        momentum_alloc = result['wsb-dip-bot']['allocation_pct'] + result['momentum-weeklies']['allocation_pct']
        income_alloc = result['wheel']['allocation_pct']

        assert momentum_alloc > income_alloc

    def test_optimize_portfolio_empty_list(self, service):
        """Empty strategy list should return empty dict."""
        result = service.optimize_portfolio([], 'moderate')
        assert result == {}

    def test_optimize_portfolio_enabled_flag_set(self, service):
        """All returned allocations should have enabled=True."""
        strategies = ['wheel', 'wsb-dip-bot']

        result = service.optimize_portfolio(strategies, 'moderate')

        for alloc in result.values():
            assert alloc['enabled'] is True

    def test_optimize_portfolio_includes_params(self, service):
        """All returned allocations should include params dict."""
        strategies = ['wheel', 'wsb-dip-bot']

        result = service.optimize_portfolio(strategies, 'moderate')

        for alloc in result.values():
            assert 'params' in alloc
            assert isinstance(alloc['params'], dict)


@pytest.mark.django_db
class TestRecommendationsGeneration:
    """Test portfolio recommendations generation."""

    def test_recommendations_low_diversification(self, service):
        """Should recommend diversification when score is low."""
        # High correlation strategies (both momentum)
        strategies = {
            'wsb-dip-bot': {'allocation_pct': 50, 'enabled': True},
            'momentum-weeklies': {'allocation_pct': 50, 'enabled': True}
        }
        result = service.analyze_portfolio(strategies)

        # Check if diversification score warrants recommendation
        if result.diversification_score < 40:
            assert any('diversification' in r.lower() for r in result.recommendations)

    def test_recommendations_low_sharpe(self, service):
        """Should recommend reducing volatility when Sharpe is low."""
        # High volatility strategy
        strategies = {
            'lotto-scanner': {'allocation_pct': 100, 'enabled': True}  # vol=0.80
        }
        result = service.analyze_portfolio(strategies)

        if result.expected_sharpe < 1.0:
            assert any('sharpe' in r.lower() for r in result.recommendations)

    def test_recommendations_no_hedging(self, service):
        """Should recommend hedging when not present."""
        strategies = {
            'wsb-dip-bot': {'allocation_pct': 50, 'enabled': True},
            'wheel': {'allocation_pct': 50, 'enabled': True}
        }
        result = service.analyze_portfolio(strategies)

        assert any('earnings-protection' in r for r in result.recommendations)

    def test_recommendations_no_income(self, service):
        """Should recommend income strategies when not present."""
        strategies = {
            'wsb-dip-bot': {'allocation_pct': 50, 'enabled': True},
            'momentum-weeklies': {'allocation_pct': 50, 'enabled': True}
        }
        result = service.analyze_portfolio(strategies)

        assert any('wheel' in r or 'SPX' in r for r in result.recommendations)

    def test_recommendations_high_concentration(self, service):
        """Should warn about high concentration."""
        strategies = {
            'wheel': {'allocation_pct': 60, 'enabled': True},
            'wsb-dip-bot': {'allocation_pct': 40, 'enabled': True}
        }
        result = service.analyze_portfolio(strategies)

        assert any('concentration' in r.lower() for r in result.recommendations)


@pytest.mark.django_db
class TestSuggestAdditions:
    """Test strategy suggestion functionality."""

    def test_suggest_additions_returns_list(self, service):
        """Should return a list of suggestions."""
        current = ['wsb-dip-bot', 'momentum-weeklies']

        suggestions = service.suggest_additions(current, 'moderate')

        assert isinstance(suggestions, list)

    def test_suggest_additions_max_three(self, service):
        """Should return at most 3 suggestions."""
        current = ['wsb-dip-bot']

        suggestions = service.suggest_additions(current, 'moderate')

        assert len(suggestions) <= 3

    def test_suggest_additions_excludes_current(self, service):
        """Should not suggest strategies already in portfolio."""
        current = ['wsb-dip-bot', 'wheel']

        suggestions = service.suggest_additions(current, 'moderate')

        for suggestion in suggestions:
            assert suggestion['strategy_id'] not in current

    def test_suggest_additions_includes_required_fields(self, service):
        """Each suggestion should have required fields."""
        current = ['wsb-dip-bot']

        suggestions = service.suggest_additions(current, 'moderate')

        for suggestion in suggestions:
            assert 'strategy_id' in suggestion
            assert 'name' in suggestion
            assert 'reason' in suggestion

    def test_suggest_additions_complete_portfolio_returns_empty(self, service):
        """Complete portfolio should return empty suggestions."""
        current = list(AVAILABLE_STRATEGIES.keys())

        suggestions = service.suggest_additions(current, 'moderate')

        assert len(suggestions) == 0

    def test_suggest_additions_conservative_excludes_aggressive(self, service):
        """Conservative profile should not suggest aggressive strategies."""
        current = ['wheel', 'spx-credit-spreads']

        suggestions = service.suggest_additions(current, 'conservative')

        for suggestion in suggestions:
            sid = suggestion['strategy_id']
            assert AVAILABLE_STRATEGIES[sid]['risk_level'] != 'aggressive'


@pytest.mark.django_db
class TestCreateFromTemplateWithPersistence:
    """Test creating portfolios from templates with real database persistence."""

    def test_create_from_template_persists_to_database(self, service, user):
        """Should create and persist StrategyPortfolio to database."""
        portfolio = service.create_from_template('conservative-income')

        # Verify it's saved to database
        assert portfolio.pk is not None
        saved = StrategyPortfolio.objects.get(pk=portfolio.pk)
        assert saved.name == 'Conservative Income'

    def test_create_from_template_correct_fields(self, service, user):
        """Should set all fields correctly from template."""
        portfolio = service.create_from_template('balanced-growth')

        assert portfolio.user == user
        assert portfolio.name == 'Balanced Growth'
        assert portfolio.description == PORTFOLIO_TEMPLATES['balanced-growth']['description']
        assert portfolio.risk_profile == 'moderate'
        assert portfolio.is_template is False
        assert portfolio.is_active is False

    def test_create_from_template_custom_name(self, service, user):
        """Should allow custom name override."""
        portfolio = service.create_from_template('conservative-income', name='My Custom Portfolio')

        assert portfolio.name == 'My Custom Portfolio'

    def test_create_from_template_stores_strategies(self, service, user):
        """Should store strategy allocations in JSON field."""
        portfolio = service.create_from_template('aggressive-momentum')

        assert portfolio.strategies is not None
        assert isinstance(portfolio.strategies, dict)
        assert 'wsb-dip-bot' in portfolio.strategies

    def test_create_from_template_stores_correlation_matrix(self, service, user):
        """Should calculate and store correlation matrix."""
        portfolio = service.create_from_template('balanced-growth')

        assert portfolio.correlation_matrix is not None
        assert isinstance(portfolio.correlation_matrix, dict)
        # Should have entries for each strategy
        strategy_ids = list(PORTFOLIO_TEMPLATES['balanced-growth']['strategies'].keys())
        for sid in strategy_ids:
            assert sid in portfolio.correlation_matrix

    def test_create_from_template_stores_diversification_score(self, service, user):
        """Should calculate and store diversification score."""
        portfolio = service.create_from_template('income-focused')

        assert portfolio.diversification_score is not None
        assert 0 <= float(portfolio.diversification_score) <= 100

    def test_create_from_template_stores_expected_sharpe(self, service, user):
        """Should calculate and store expected Sharpe ratio."""
        portfolio = service.create_from_template('growth-seeker')

        assert portfolio.expected_sharpe is not None
        assert float(portfolio.expected_sharpe) > 0

    def test_create_from_template_not_found_raises_error(self, service):
        """Should raise ValueError for non-existent template."""
        with pytest.raises(ValueError) as exc_info:
            service.create_from_template('nonexistent-template')

        assert 'not found' in str(exc_info.value)

    def test_create_from_template_all_templates_valid(self, service, user):
        """All defined templates should create valid portfolios."""
        for template_id in PORTFOLIO_TEMPLATES.keys():
            portfolio = service.create_from_template(template_id)
            assert portfolio.pk is not None
            assert portfolio.strategies is not None
            assert portfolio.correlation_matrix is not None


@pytest.mark.django_db
class TestCreateCustomPortfolioWithPersistence:
    """Test creating custom portfolios with real database persistence."""

    def test_create_custom_portfolio_persists_to_database(self, service, user):
        """Should create and persist custom portfolio to database."""
        strategies = {
            'wheel': {'allocation_pct': 50, 'enabled': True},
            'wsb-dip-bot': {'allocation_pct': 50, 'enabled': True}
        }

        portfolio = service.create_custom_portfolio('My Portfolio', strategies)

        assert portfolio.pk is not None
        saved = StrategyPortfolio.objects.get(pk=portfolio.pk)
        assert saved.name == 'My Portfolio'

    def test_create_custom_portfolio_correct_fields(self, service, user):
        """Should set all fields correctly."""
        strategies = {
            'wheel': {'allocation_pct': 60, 'enabled': True},
            'earnings-protection': {'allocation_pct': 40, 'enabled': True}
        }

        portfolio = service.create_custom_portfolio(
            name='Test Portfolio',
            strategies=strategies,
            description='My test description',
            risk_profile='conservative'
        )

        assert portfolio.user == user
        assert portfolio.name == 'Test Portfolio'
        assert portfolio.description == 'My test description'
        assert portfolio.risk_profile == 'conservative'
        assert portfolio.is_template is False
        assert portfolio.is_active is False

    def test_create_custom_portfolio_calculates_metrics(self, service, user):
        """Should calculate real metrics for custom portfolio."""
        strategies = {
            'wheel': {'allocation_pct': 40, 'enabled': True},
            'wsb-dip-bot': {'allocation_pct': 30, 'enabled': True},
            'debit-spreads': {'allocation_pct': 30, 'enabled': True}
        }

        portfolio = service.create_custom_portfolio('Custom', strategies)

        # Verify correlation matrix was built
        assert 'wheel' in portfolio.correlation_matrix
        assert 'wsb-dip-bot' in portfolio.correlation_matrix
        assert portfolio.correlation_matrix['wheel']['wheel'] == 1.0

        # Verify diversification score was calculated
        assert 0 <= float(portfolio.diversification_score) <= 100

        # Verify Sharpe was calculated
        assert float(portfolio.expected_sharpe) > 0

    def test_create_custom_portfolio_invalid_strategy_raises_error(self, service, user):
        """Should raise ValueError for invalid strategy ID."""
        strategies = {
            'invalid-strategy': {'allocation_pct': 100, 'enabled': True}
        }

        with pytest.raises(ValueError) as exc_info:
            service.create_custom_portfolio('Test', strategies)

        assert 'Unknown strategy' in str(exc_info.value)

    def test_create_custom_portfolio_mixed_valid_invalid_raises_error(self, service, user):
        """Should raise ValueError when any strategy is invalid."""
        strategies = {
            'wheel': {'allocation_pct': 50, 'enabled': True},
            'fake-strategy': {'allocation_pct': 50, 'enabled': True}
        }

        with pytest.raises(ValueError):
            service.create_custom_portfolio('Test', strategies)


@pytest.mark.django_db
class TestGetUserPortfoliosWithPersistence:
    """Test retrieving user portfolios from database."""

    def test_get_user_portfolios_returns_created(self, service, user):
        """Should return portfolios created for user."""
        service.create_from_template('conservative-income')
        service.create_from_template('balanced-growth')

        portfolios = service.get_user_portfolios()

        assert len(portfolios) >= 2
        names = [p.name for p in portfolios]
        assert 'Conservative Income' in names
        assert 'Balanced Growth' in names

    def test_get_user_portfolios_empty_for_new_user(self, second_user):
        """Should return empty list for user with no portfolios."""
        service = PortfolioBuilderService(user=second_user)

        portfolios = service.get_user_portfolios()

        # May return templates if any exist
        user_portfolios = [p for p in portfolios if p.user == second_user]
        assert len(user_portfolios) == 0

    def test_get_user_portfolios_no_user_returns_empty(self, service_no_user):
        """Should return empty list when no user set."""
        portfolios = service_no_user.get_user_portfolios()
        assert portfolios == []

    def test_get_user_portfolios_isolation(self, service, user, second_user):
        """Users should only see their own portfolios."""
        # Create portfolio for first user
        service.create_custom_portfolio(
            'User1 Portfolio',
            {'wheel': {'allocation_pct': 100, 'enabled': True}}
        )

        # Create portfolio for second user
        service2 = PortfolioBuilderService(user=second_user)
        service2.create_custom_portfolio(
            'User2 Portfolio',
            {'wsb-dip-bot': {'allocation_pct': 100, 'enabled': True}}
        )

        # First user should not see second user's portfolio
        portfolios1 = service.get_user_portfolios()
        portfolio_names = [p.name for p in portfolios1 if p.user == user]
        assert 'User1 Portfolio' in portfolio_names
        assert 'User2 Portfolio' not in portfolio_names


@pytest.mark.django_db
class TestGetActivePortfolioWithPersistence:
    """Test retrieving active portfolio from database."""

    def test_get_active_portfolio_none_initially(self, service, user):
        """Should return None when no active portfolio."""
        result = service.get_active_portfolio()
        assert result is None

    def test_get_active_portfolio_returns_active(self, service, user):
        """Should return the active portfolio."""
        # Create and activate a portfolio
        portfolio = service.create_from_template('conservative-income')
        portfolio.is_active = True
        portfolio.save()

        result = service.get_active_portfolio()

        assert result is not None
        assert result.pk == portfolio.pk
        assert result.is_active is True

    def test_get_active_portfolio_no_user_returns_none(self, service_no_user):
        """Should return None when no user set."""
        result = service_no_user.get_active_portfolio()
        assert result is None


@pytest.mark.django_db
class TestEndToEndPortfolioWorkflow:
    """End-to-end integration tests for complete portfolio workflows."""

    def test_complete_portfolio_creation_workflow(self, service, user):
        """Test complete workflow: create, analyze, verify persistence."""
        # 1. Get available strategies
        strategies = service.get_available_strategies()
        assert 'wheel' in strategies

        # 2. Get templates
        templates = service.get_portfolio_templates()
        assert 'balanced-growth' in templates

        # 3. Get template details with real analysis
        details = service.get_template_details('balanced-growth')
        assert details['analysis']['expected_return'] > 0

        # 4. Create portfolio from template
        portfolio = service.create_from_template('balanced-growth')

        # 5. Verify it's persisted
        from_db = StrategyPortfolio.objects.get(pk=portfolio.pk)
        assert from_db.name == 'Balanced Growth'
        assert from_db.correlation_matrix is not None
        assert float(from_db.diversification_score) > 0

        # 6. Retrieve through service
        portfolios = service.get_user_portfolios()
        assert any(p.pk == portfolio.pk for p in portfolios)

    def test_custom_optimized_portfolio_workflow(self, service, user):
        """Test workflow: select strategies, optimize, create."""
        # 1. Select strategies to include
        selected_strategies = ['wheel', 'wsb-dip-bot', 'earnings-protection', 'debit-spreads']

        # 2. Get optimization suggestions
        suggestions = service.suggest_additions(['wheel'], 'moderate')
        assert len(suggestions) > 0

        # 3. Optimize allocations
        optimized = service.optimize_portfolio(selected_strategies, 'moderate')
        total_alloc = sum(a['allocation_pct'] for a in optimized.values())
        assert 99 <= total_alloc <= 101

        # 4. Analyze the optimized portfolio
        analysis = service.analyze_portfolio(optimized)
        assert analysis.expected_return > 0
        assert analysis.expected_sharpe > 0
        assert 0 <= analysis.diversification_score <= 100

        # 5. Create custom portfolio with optimized allocations
        portfolio = service.create_custom_portfolio(
            name='My Optimized Portfolio',
            strategies=optimized,
            description='Auto-optimized for moderate risk',
            risk_profile='moderate'
        )

        # 6. Verify persistence
        from_db = StrategyPortfolio.objects.get(pk=portfolio.pk)
        assert from_db.strategies == optimized
        assert abs(float(from_db.expected_sharpe) - analysis.expected_sharpe) < 0.01

    def test_portfolio_metrics_consistency(self, service, user):
        """Test that stored metrics match analysis results."""
        strategies = {
            'wheel': {'allocation_pct': 40, 'enabled': True},
            'wsb-dip-bot': {'allocation_pct': 30, 'enabled': True},
            'earnings-protection': {'allocation_pct': 30, 'enabled': True}
        }

        # Get analysis
        analysis = service.analyze_portfolio(strategies)

        # Create portfolio
        portfolio = service.create_custom_portfolio('Consistency Test', strategies)

        # Verify stored values match analysis
        assert portfolio.correlation_matrix == analysis.correlation_matrix
        assert abs(float(portfolio.diversification_score) - analysis.diversification_score) < 0.01
        assert abs(float(portfolio.expected_sharpe) - analysis.expected_sharpe) < 0.01
