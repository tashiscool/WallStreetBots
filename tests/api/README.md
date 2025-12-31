# API Views Test Suite

This directory contains comprehensive tests for the Django API views in `backend/auth0login/api_views.py`.

## Test Files

- **`test_api_views_comprehensive.py`** (2,744 lines): Main comprehensive test suite with 140 test cases
- **`test_trading_gate.py`** (existing): Trading Gate specific tests
- **`test_email_settings.py`** (existing): Email settings tests
- **`TEST_COVERAGE_SUMMARY.md`**: Detailed summary of test coverage

## Running Tests

### Run All API Tests
```bash
python -m pytest tests/api/ -v
```

### Run Only Comprehensive Test Suite
```bash
python -m pytest tests/api/test_api_views_comprehensive.py -v
```

### Run Specific Test Class
```bash
# Test only backtest endpoints
python -m pytest tests/api/test_api_views_comprehensive.py::TestBacktestAPI -v

# Test only trading gate endpoints
python -m pytest tests/api/test_api_views_comprehensive.py::TestTradingGateAPI -v

# Test only portfolio endpoints
python -m pytest tests/api/test_api_views_comprehensive.py::TestPortfolioAPI -v
```

### Run Specific Test Method
```bash
python -m pytest tests/api/test_api_views_comprehensive.py::TestBacktestAPI::test_run_backtest_success -v
```

### Run with Coverage Report
```bash
# Generate coverage report for api_views.py only
python -m pytest tests/api/test_api_views_comprehensive.py \
  --cov=backend/auth0login/api_views \
  --cov-report=html \
  --cov-report=term-missing

# View the HTML report
open htmlcov/index.html
```

### Run Fast (Skip Slow Tests)
```bash
python -m pytest tests/api/ -v -m "not slow"
```

### Run with Detailed Output
```bash
python -m pytest tests/api/test_api_views_comprehensive.py -vv --tb=short
```

## Test Organization

### Test Classes by Category

1. **TestBacktestAPI** - Backtesting endpoints (6 tests)
2. **TestBuildSpreadAPI** - Options spread building (3 tests)
3. **TestTradingGateAPI** - Trading gate system (10 tests)
4. **TestRiskAssessmentAPI** - Risk questionnaire (12 tests)
5. **TestStrategyRecommendationsAPI** - Strategy recommendations (10 tests)
6. **TestAllocationManagementAPI** - Capital allocation (14 tests)
7. **TestVIXMonitoringAPI** - VIX monitoring (4 tests)
8. **TestCircuitBreakerAPI** - Circuit breaker system (8 tests)
9. **TestPortfolioAPI** - Portfolio management (9 tests)
10. **TestLeaderboardAPI** - Strategy leaderboard (5 tests)
11. **TestCustomStrategyAPI** - Custom strategies (9 tests)
12. **TestMarketContextAPI** - Market data (4 tests)
13. **TestTaxOptimizationAPI** - Tax optimization (5 tests)
14. **TestUserProfileAPI** - User profiles (4 tests)
15. **TestDigestAPI** - Email digests (4 tests)
16. **TestTradeExplanationAPI** - Trade reasoning (3 tests)
17. **TestBenchmarkComparisonAPI** - Benchmark comparison (2 tests)
18. **TestAlpacaConnectionAPI** - Alpaca API testing (5 tests)
19. **TestWizardConfigAPI** - Setup wizard (2 tests)
20. **TestEmailAPI** - Email notifications (5 tests)
21. **TestSaveSettingsAPI** - User settings (2 tests)
22. **TestFeatureAvailabilityAPI** - Feature flags (2 tests)
23. **Additional endpoint tests** (12 tests)

## Test Patterns

Each test typically follows this pattern:

```python
class TestEndpointAPI(TestCase):
    """Test endpoint API endpoints."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = Client()
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        self.client.login(username='testuser', password='testpass123')

    @patch('module.service')
    def test_endpoint_success(self, mock_service):
        """Test successful endpoint call."""
        # Setup mock
        mock_service.method.return_value = expected_result

        # Make request
        response = self.client.post('/api/endpoint', data=...)

        # Assert
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'success')

    @patch('module.service')
    def test_endpoint_error(self, mock_service):
        """Test endpoint error handling."""
        mock_service.method.side_effect = Exception("Error")

        response = self.client.post('/api/endpoint', data=...)

        self.assertEqual(response.status_code, 500)
```

## Common Test Scenarios

Each endpoint is typically tested for:

1. **Success Case** - Normal operation with valid data
2. **Invalid JSON** - Malformed request body
3. **Missing Parameters** - Required fields not provided
4. **Service Exceptions** - Backend service failures
5. **Authentication** - Unauthenticated access denied
6. **Authorization** - Permission checks
7. **Value Errors** - Invalid parameter values
8. **Edge Cases** - Boundary conditions

## Mocking Strategy

The tests mock external dependencies:

- **Dashboard Service**: `dashboard_service.method()`
- **Trading Services**: Imported locally in view functions
- **Alpaca API**: `alpaca.trading.client.TradingClient`
- **Email (SMTP)**: `smtplib.SMTP`
- **Database Models**: Django ORM models

## Coverage Goals

- **Target Coverage**: 80%+ of api_views.py
- **Current Test Count**: 140 test cases
- **Endpoint Coverage**: 119 functions in api_views.py
- **Test Density**: 1.18 tests per endpoint

## Known Issues

### Mock Path Corrections Needed

Many services are imported inside view functions, requiring specific patch paths:

```python
# In api_views.py
def my_view(request):
    from .services.my_service import my_service_instance
    result = my_service_instance.do_something()

# In tests - WRONG
@patch('backend.auth0login.api_views.my_service_instance')

# In tests - CORRECT
@patch('backend.auth0login.services.my_service.my_service_instance')
```

### URL Route Issues

Some tests may fail if URL patterns are not configured. Check `urls.py` for proper route definitions.

## Debugging Failed Tests

### View Detailed Errors
```bash
python -m pytest tests/api/test_api_views_comprehensive.py::TestName::test_name -vv --tb=long
```

### Check Coverage for Specific Function
```bash
python -m pytest tests/api/ --cov=backend/auth0login/api_views --cov-report=term-missing | grep "function_name"
```

### Run Tests in Interactive Mode
```bash
python -m pytest tests/api/test_api_views_comprehensive.py --pdb
```

### Check Logs During Test
```bash
python -m pytest tests/api/test_api_views_comprehensive.py -v -s
```

## Contributing

When adding new tests:

1. Follow the existing test class organization
2. Use descriptive test names: `test_endpoint_scenario`
3. Include docstrings explaining what is being tested
4. Test both success and error cases
5. Mock external dependencies properly
6. Verify with coverage report

## Next Steps

To achieve 80%+ coverage:

1. Fix mock import paths for locally-imported services
2. Add tests for missing endpoints (30+ remaining)
3. Add edge case tests for complex workflows
4. Add integration tests for multi-step operations
5. Set up CI/CD coverage tracking

## Resources

- [Django Testing Documentation](https://docs.djangoproject.com/en/stable/topics/testing/)
- [pytest-django](https://pytest-django.readthedocs.io/)
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
- [Coverage.py](https://coverage.readthedocs.io/)
