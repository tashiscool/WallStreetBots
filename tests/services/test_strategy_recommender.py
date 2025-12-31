"""Comprehensive tests for StrategyRecommenderService."""

import pytest
from backend.auth0login.services.strategy_recommender import (
    StrategyRecommenderService,
    STRATEGIES,
    strategy_recommender_service
)


@pytest.fixture
def service():
    return StrategyRecommenderService()


class TestInit:
    """Test initialization."""

    def test_init(self):
        service = StrategyRecommenderService()
        assert service.strategies == STRATEGIES


class TestGetRecommendations:
    """Test getting recommendations."""

    def test_get_recommendations_conservative(self, service):
        result = service.get_recommendations('conservative', 10000)
        assert result['status'] == 'success'
        assert result['profile'] == 'conservative'
        assert 'highly_recommended' in result

    def test_get_recommendations_moderate(self, service):
        result = service.get_recommendations('moderate', 10000)
        assert result['status'] == 'success'

    def test_get_recommendations_aggressive(self, service):
        result = service.get_recommendations('aggressive', 10000)
        assert result['status'] == 'success'

    def test_get_recommendations_insufficient_capital(self, service):
        result = service.get_recommendations('moderate', 500)
        assert len(result['excluded']) > 0


class TestGetStrategyDetails:
    """Test getting strategy details."""

    def test_get_strategy_details_found(self, service):
        result = service.get_strategy_details('wheel')
        assert result is not None
        assert result['id'] == 'wheel'

    def test_get_strategy_details_not_found(self, service):
        result = service.get_strategy_details('nonexistent')
        assert result is None


class TestHelperMethods:
    """Test helper methods."""

    def test_is_adjacent_profile(self, service):
        assert service._is_adjacent_profile('conservative', ['moderate'])
        assert not service._is_adjacent_profile('conservative', ['aggressive'])

    def test_get_allocation_suggestion(self, service):
        strategy = STRATEGIES[0]
        result = service._get_allocation_suggestion('moderate', strategy, 10000)
        assert '%' in result
