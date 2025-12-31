# Quick Start Guide - API Tests

## Run Tests Now

```bash
# Run all comprehensive API tests
python -m pytest tests/api/test_api_views_comprehensive.py -v

# Run with coverage report
python -m pytest tests/api/test_api_views_comprehensive.py \
  --cov=backend/auth0login/api_views \
  --cov-report=term-missing \
  --cov-report=html

# View coverage report
open htmlcov/index.html
```

## What's Included

### File Created
`tests/api/test_api_views_comprehensive.py` (2,744 lines)

### Test Coverage
- **140 test cases** covering all major API endpoints
- **23 test classes** organized by functional area
- **119 API functions** in api_views.py targeted

### Test Categories

| Category | Tests | Endpoints Covered |
|----------|-------|-------------------|
| Backtest API | 6 | run_backtest |
| Options Trading | 3 | build_spread, suggest_spreads |
| Trading Gate | 10 | status, requirements, request_live, start_paper |
| Risk Assessment | 12 | questions, submit, result, calculate |
| Strategy Recommendations | 10 | recommendations, details |
| Allocation Management | 14 | list, detail, update, initialize, rebalance |
| VIX Monitoring | 4 | status, history |
| Circuit Breaker | 8 | status, history, advance, reset |
| Portfolio Management | 9 | CRUD operations, templates |
| Leaderboard | 5 | rankings, comparisons |
| Custom Strategies | 9 | CRUD, validate, backtest |
| Market Context | 4 | context, overview, sectors |
| Tax Optimization | 5 | lots, harvesting, wash sales |
| User Profile | 4 | profile, onboarding |
| Digest/Email | 4 | preview, send, history |
| Trade Explanation | 3 | explanation, signals |
| Benchmark Comparison | 2 | performance comparison |
| Alpaca Connection | 5 | connection testing |
| Wizard Config | 2 | setup configuration |
| Email Notifications | 5 | SMTP testing |
| Settings | 2 | save settings |
| Feature Availability | 2 | feature flags |
| Additional | 12 | various endpoints |

## Current Status

- **Total Tests**: 140
- **Currently Passing**: 8 (backtest and settings tests)
- **Needs Mock Fixes**: 132 tests
- **Coverage Target**: 80%+

## Quick Test Examples

### Test a Specific Category
```bash
# Test backtest endpoints (currently passing)
python -m pytest tests/api/test_api_views_comprehensive.py::TestBacktestAPI -v

# Test trading gate endpoints
python -m pytest tests/api/test_api_views_comprehensive.py::TestTradingGateAPI -v

# Test portfolio endpoints
python -m pytest tests/api/test_api_views_comprehensive.py::TestPortfolioAPI -v
```

### Test a Specific Scenario
```bash
# Test successful backtest
python -m pytest tests/api/test_api_views_comprehensive.py::TestBacktestAPI::test_run_backtest_success -v

# Test error handling
python -m pytest tests/api/test_api_views_comprehensive.py::TestBacktestAPI::test_run_backtest_service_exception -v
```

## Test Structure

Each test class follows this pattern:

```python
class TestCategoryAPI(TestCase):
    def setUp(self):
        # Create test user and login

    @patch('service.module')
    def test_endpoint_success(self, mock_service):
        # Test happy path

    @patch('service.module')
    def test_endpoint_error(self, mock_service):
        # Test error handling

    def test_endpoint_invalid_json(self):
        # Test validation
```

## Next Steps

1. **Fix Mock Paths**: Update patch decorators to match local imports
2. **Add Missing Tests**: Cover remaining 30+ endpoints
3. **Run Coverage**: Verify 80%+ coverage achieved
4. **CI Integration**: Add to continuous integration pipeline

## Documentation

- Full summary: `TEST_COVERAGE_SUMMARY.md`
- Detailed guide: `README.md`
- Test file: `test_api_views_comprehensive.py`
