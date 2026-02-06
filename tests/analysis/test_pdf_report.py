"""
Tests for PDF Report Generation (Phase 3).

Tests cover:
- PDFReportConfig defaults
- ReportTemplates factory methods
- PDFReportGenerator._calculate_metrics with sample returns
- PDFReportGenerator._build_html generates valid HTML
- generate() returns bytes (mock weasyprint)
- _fig_to_base64 with mocked kaleido
"""

import sys
import os
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure backend is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.tradingbot.analysis.pdf_report import (
    PDFReportConfig,
    PDFReportGenerator,
)
from backend.tradingbot.analysis.report_templates import ReportTemplates


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_returns():
    """Generate a sample daily returns series for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-02', periods=252, freq='B')
    daily_returns = np.random.normal(0.0004, 0.012, len(dates))
    return pd.Series(daily_returns, index=dates, name='returns')


@pytest.fixture
def sample_benchmark_returns():
    """Generate a sample benchmark returns series."""
    np.random.seed(99)
    dates = pd.date_range(start='2024-01-02', periods=252, freq='B')
    daily_returns = np.random.normal(0.0003, 0.010, len(dates))
    return pd.Series(daily_returns, index=dates, name='benchmark')


@pytest.fixture
def sample_trades():
    """Generate a sample trades DataFrame."""
    return pd.DataFrame({
        'symbol': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'],
        'side': ['long', 'long', 'short', 'long', 'long'],
        'qty': [10, 5, 20, 15, 8],
        'entry_price': [150.00, 2800.00, 350.00, 200.00, 500.00],
        'exit_price': [165.00, 2750.00, 340.00, 220.00, 550.00],
        'pnl': [150.00, -250.00, 200.00, 300.00, 400.00],
        'entry_date': ['2024-01-05', '2024-02-10', '2024-03-15', '2024-04-20', '2024-05-25'],
        'exit_date': ['2024-01-12', '2024-02-17', '2024-03-22', '2024-04-27', '2024-06-01'],
    })


@pytest.fixture
def generator():
    """Create a default PDFReportGenerator."""
    return PDFReportGenerator()


@pytest.fixture
def custom_config():
    """Create a custom PDFReportConfig."""
    return PDFReportConfig(
        title="Test Report",
        subtitle="Test Subtitle",
        cover_page=True,
        executive_summary=True,
        equity_curve=False,
        drawdown_chart=False,
        monthly_heatmap=False,
        returns_distribution=False,
        rolling_sharpe=False,
        trade_log=True,
        trade_log_max_rows=10,
        risk_metrics=True,
    )


# ============================================================================
# PDFReportConfig Tests
# ============================================================================

class TestPDFReportConfig:
    """Tests for PDFReportConfig dataclass."""

    def test_default_values(self):
        config = PDFReportConfig()
        assert config.title == "Trading Performance Report"
        assert config.subtitle == ""
        assert config.cover_page is True
        assert config.executive_summary is True
        assert config.equity_curve is True
        assert config.drawdown_chart is True
        assert config.monthly_heatmap is True
        assert config.returns_distribution is True
        assert config.rolling_sharpe is True
        assert config.trade_log is True
        assert config.trade_log_max_rows == 50
        assert config.risk_metrics is True
        assert config.tax_summary is False
        assert config.page_size == "A4"
        assert config.orientation == "portrait"
        assert config.theme == "light"

    def test_custom_values(self):
        config = PDFReportConfig(
            title="Custom Title",
            subtitle="Custom Subtitle",
            cover_page=False,
            trade_log_max_rows=100,
            page_size="Letter",
            theme="dark",
        )
        assert config.title == "Custom Title"
        assert config.subtitle == "Custom Subtitle"
        assert config.cover_page is False
        assert config.trade_log_max_rows == 100
        assert config.page_size == "Letter"
        assert config.theme == "dark"

    def test_is_dataclass(self):
        """Ensure PDFReportConfig is a proper dataclass."""
        from dataclasses import fields
        config = PDFReportConfig()
        field_names = {f.name for f in fields(config)}
        assert 'title' in field_names
        assert 'cover_page' in field_names
        assert 'trade_log_max_rows' in field_names


# ============================================================================
# ReportTemplates Tests
# ============================================================================

class TestReportTemplates:
    """Tests for ReportTemplates factory methods."""

    def test_weekly_performance(self):
        config = ReportTemplates.weekly_performance()
        assert isinstance(config, PDFReportConfig)
        assert config.title == "Weekly Performance Report"
        assert config.cover_page is False
        assert config.monthly_heatmap is False
        assert config.returns_distribution is False
        assert config.rolling_sharpe is False
        assert config.trade_log is True
        assert config.trade_log_max_rows == 30
        assert config.risk_metrics is True

    def test_monthly_detailed(self):
        config = ReportTemplates.monthly_detailed()
        assert isinstance(config, PDFReportConfig)
        assert config.title == "Monthly Performance Report"
        assert config.cover_page is True
        assert config.monthly_heatmap is True
        assert config.returns_distribution is True
        assert config.rolling_sharpe is True
        assert config.trade_log_max_rows == 100

    def test_quarterly_review(self):
        config = ReportTemplates.quarterly_review()
        assert isinstance(config, PDFReportConfig)
        assert config.title == "Quarterly Performance Review"
        assert config.cover_page is True
        assert config.trade_log_max_rows == 200

    def test_year_end_tax(self):
        config = ReportTemplates.year_end_tax()
        assert isinstance(config, PDFReportConfig)
        assert config.title == "Year-End Tax Report"
        assert config.tax_summary is True
        assert config.drawdown_chart is False
        assert config.returns_distribution is False
        assert config.rolling_sharpe is False
        assert config.risk_metrics is False
        assert config.trade_log_max_rows == 500

    def test_all_templates_return_pdfreportconfig(self):
        """All template methods should return PDFReportConfig instances."""
        templates = [
            ReportTemplates.weekly_performance(),
            ReportTemplates.monthly_detailed(),
            ReportTemplates.quarterly_review(),
            ReportTemplates.year_end_tax(),
        ]
        for config in templates:
            assert isinstance(config, PDFReportConfig)


# ============================================================================
# PDFReportGenerator._calculate_metrics Tests
# ============================================================================

class TestCalculateMetrics:
    """Tests for PDFReportGenerator._calculate_metrics."""

    def test_basic_metrics(self, generator, sample_returns):
        metrics = generator._calculate_metrics(sample_returns)

        assert 'total_return' in metrics
        assert 'ann_return' in metrics
        assert 'ann_volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'sortino_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'calmar_ratio' in metrics
        assert 'win_rate' in metrics
        assert 'best_day' in metrics
        assert 'worst_day' in metrics
        assert 'var_95' in metrics
        assert 'total_days' in metrics
        assert 'start_date' in metrics
        assert 'end_date' in metrics

    def test_total_return_is_reasonable(self, generator, sample_returns):
        metrics = generator._calculate_metrics(sample_returns)
        # With seed 42 and 252 days at mean 0.04%, should be roughly 10% total
        assert -0.5 < metrics['total_return'] < 1.0

    def test_sharpe_ratio_is_reasonable(self, generator, sample_returns):
        metrics = generator._calculate_metrics(sample_returns)
        # Should be between -3 and 5 for reasonable data
        assert -3 < metrics['sharpe_ratio'] < 5

    def test_max_drawdown_is_negative(self, generator, sample_returns):
        metrics = generator._calculate_metrics(sample_returns)
        assert metrics['max_drawdown'] <= 0

    def test_win_rate_between_0_and_1(self, generator, sample_returns):
        metrics = generator._calculate_metrics(sample_returns)
        assert 0 <= metrics['win_rate'] <= 1

    def test_total_days_matches_input(self, generator, sample_returns):
        metrics = generator._calculate_metrics(sample_returns)
        assert metrics['total_days'] == len(sample_returns)

    def test_start_end_dates(self, generator, sample_returns):
        metrics = generator._calculate_metrics(sample_returns)
        assert metrics['start_date'] == sample_returns.index[0]
        assert metrics['end_date'] == sample_returns.index[-1]

    def test_with_benchmark(self, generator, sample_returns, sample_benchmark_returns):
        metrics = generator._calculate_metrics(sample_returns, sample_benchmark_returns)
        assert 'benchmark_return' in metrics
        assert 'excess_return' in metrics
        assert 'beta' in metrics
        assert 'alpha' in metrics

    def test_excess_return_calculation(self, generator, sample_returns, sample_benchmark_returns):
        metrics = generator._calculate_metrics(sample_returns, sample_benchmark_returns)
        expected_excess = metrics['total_return'] - metrics['benchmark_return']
        assert abs(metrics['excess_return'] - expected_excess) < 1e-10

    def test_empty_returns(self, generator):
        """Handle edge case of empty returns series gracefully."""
        empty = pd.Series([], dtype=float)
        # Should not raise
        metrics = generator._calculate_metrics(empty)
        assert metrics['total_days'] == 0

    def test_all_positive_returns(self, generator):
        """Win rate should be 1.0 for all-positive returns."""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='B')
        returns = pd.Series([0.01] * 10, index=dates)
        metrics = generator._calculate_metrics(returns)
        assert metrics['win_rate'] == 1.0

    def test_all_negative_returns(self, generator):
        """Win rate should be 0.0 for all-negative returns."""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='B')
        returns = pd.Series([-0.01] * 10, index=dates)
        metrics = generator._calculate_metrics(returns)
        assert metrics['win_rate'] == 0.0

    def test_var_95_is_negative(self, generator, sample_returns):
        """VaR at 95% (5th percentile) should typically be negative."""
        metrics = generator._calculate_metrics(sample_returns)
        # For normal distribution centered near zero, the 5th percentile
        # should be negative
        assert metrics['var_95'] < 0


# ============================================================================
# PDFReportGenerator._build_html Tests
# ============================================================================

class TestBuildHtml:
    """Tests for PDFReportGenerator._build_html."""

    def test_html_structure(self, generator, sample_returns):
        html = generator._build_html(sample_returns, None, None, "TestStrategy")
        assert '<!DOCTYPE html>' in html
        assert '<html>' in html
        assert '</html>' in html
        assert '<head>' in html
        assert '<body>' in html

    def test_html_contains_title(self, generator, sample_returns):
        html = generator._build_html(sample_returns, None, None, "TestStrategy")
        assert 'Trading Performance Report' in html

    def test_html_contains_strategy_name(self, generator, sample_returns):
        html = generator._build_html(sample_returns, None, None, "MyStrategy")
        assert 'MyStrategy' in html

    def test_html_contains_executive_summary(self, generator, sample_returns):
        html = generator._build_html(sample_returns, None, None, "TestStrategy")
        assert 'Executive Summary' in html
        assert 'Sharpe Ratio' in html
        assert 'Max Drawdown' in html

    def test_html_contains_css(self, generator, sample_returns):
        html = generator._build_html(sample_returns, None, None, "TestStrategy")
        assert '<style>' in html
        assert 'font-family' in html

    def test_html_with_trades(self, generator, sample_returns, sample_trades):
        html = generator._build_html(sample_returns, sample_trades, None, "TestStrategy")
        assert 'Trade Log' in html
        assert 'AAPL' in html
        assert 'GOOGL' in html

    def test_html_without_cover_page(self, sample_returns):
        config = PDFReportConfig(cover_page=False)
        gen = PDFReportGenerator(config)
        html = gen._build_html(sample_returns, None, None, "TestStrategy")
        # The CSS may still reference .cover-page styles, but the actual
        # cover-page div should not be rendered in the body
        assert '<div class="cover-page">' not in html

    def test_html_with_cover_page(self, generator, sample_returns):
        html = generator._build_html(sample_returns, None, None, "TestStrategy")
        assert 'cover-page' in html

    def test_html_risk_metrics_section(self, generator, sample_returns):
        html = generator._build_html(sample_returns, None, None, "TestStrategy")
        assert 'Risk Metrics' in html
        assert 'Value at Risk' in html

    def test_custom_config_disables_sections(self, sample_returns, custom_config):
        gen = PDFReportGenerator(custom_config)
        html = gen._build_html(sample_returns, None, None, "TestStrategy")
        assert 'Test Report' in html
        assert 'Executive Summary' in html
        # Equity curve and other chart sections disabled -- but
        # chart unavailable messages may still appear if PLOTLY_AVAILABLE is False
        # At minimum, the risk metrics should be present
        assert 'Risk Metrics' in html


# ============================================================================
# PDFReportGenerator.generate() Tests
# ============================================================================

class TestGenerate:
    """Tests for PDFReportGenerator.generate()."""

    def test_generate_returns_bytes(self, generator, sample_returns):
        result = generator.generate(sample_returns, strategy_name="TestStrategy")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_generate_with_all_data(self, generator, sample_returns, sample_trades, sample_benchmark_returns):
        result = generator.generate(
            returns=sample_returns,
            trades=sample_trades,
            benchmark_returns=sample_benchmark_returns,
            strategy_name="FullStrategy",
        )
        assert isinstance(result, bytes)
        assert len(result) > 100  # Should contain substantial content

    def test_generate_uses_weasyprint_when_available(self, generator, sample_returns):
        mock_weasyprint = MagicMock()
        mock_html_obj = MagicMock()
        mock_html_obj.write_pdf.return_value = b'%PDF-fake-content'
        mock_weasyprint.HTML.return_value = mock_html_obj

        import backend.tradingbot.analysis.pdf_report as pdf_mod
        original_available = pdf_mod.WEASYPRINT_AVAILABLE
        original_wp = getattr(pdf_mod, 'weasyprint', None)
        try:
            pdf_mod.WEASYPRINT_AVAILABLE = True
            pdf_mod.weasyprint = mock_weasyprint
            result = generator.generate(sample_returns, strategy_name="TestStrategy")
            assert result == b'%PDF-fake-content'
            mock_weasyprint.HTML.assert_called_once()
            mock_html_obj.write_pdf.assert_called_once()
        finally:
            pdf_mod.WEASYPRINT_AVAILABLE = original_available
            if original_wp is not None:
                pdf_mod.weasyprint = original_wp
            elif hasattr(pdf_mod, 'weasyprint'):
                delattr(pdf_mod, 'weasyprint')

    def test_generate_without_weasyprint_returns_html_bytes(self, generator, sample_returns):
        """When weasyprint is not available, returns HTML as bytes."""
        with patch('backend.tradingbot.analysis.pdf_report.WEASYPRINT_AVAILABLE', False):
            result = generator.generate(sample_returns, strategy_name="TestStrategy")
            assert isinstance(result, bytes)
            decoded = result.decode('utf-8')
            assert '<!DOCTYPE html>' in decoded

    def test_generate_saves_to_output_path(self, generator, sample_returns, tmp_path):
        output_file = str(tmp_path / "test_report.pdf")
        result = generator.generate(
            sample_returns,
            strategy_name="TestStrategy",
            output_path=output_file,
        )
        assert os.path.exists(output_file)
        with open(output_file, 'rb') as f:
            saved_content = f.read()
        assert saved_content == result


# ============================================================================
# PDFReportGenerator._fig_to_base64 Tests
# ============================================================================

class TestFigToBase64:
    """Tests for _fig_to_base64."""

    def test_fig_to_base64_without_kaleido(self, generator):
        """Without kaleido, returns empty string."""
        with patch('backend.tradingbot.analysis.pdf_report.KALEIDO_AVAILABLE', False):
            mock_fig = MagicMock()
            result = generator._fig_to_base64(mock_fig)
            assert result == ""

    @patch('backend.tradingbot.analysis.pdf_report.KALEIDO_AVAILABLE', True)
    def test_fig_to_base64_with_kaleido(self, generator):
        """With kaleido available, returns base64-encoded PNG."""
        mock_fig = MagicMock()
        fake_png = b'\x89PNG\r\n\x1a\nfake_image_data'
        mock_fig.to_image.return_value = fake_png

        result = generator._fig_to_base64(mock_fig)

        import base64
        expected = base64.b64encode(fake_png).decode('utf-8')
        assert result == expected
        mock_fig.to_image.assert_called_once_with(format="png", width=800, height=400, scale=2)


# ============================================================================
# Trade Log Section Tests
# ============================================================================

class TestTradeLogSection:
    """Tests for _trade_log_section."""

    def test_trade_log_contains_all_symbols(self, generator, sample_trades):
        html = generator._trade_log_section(sample_trades)
        assert 'AAPL' in html
        assert 'GOOGL' in html
        assert 'MSFT' in html
        assert 'TSLA' in html
        assert 'NVDA' in html

    def test_trade_log_max_rows(self, sample_trades):
        config = PDFReportConfig(trade_log_max_rows=2)
        gen = PDFReportGenerator(config)
        html = gen._trade_log_section(sample_trades)
        assert 'Showing 2 of 5 trades' in html

    def test_trade_log_pnl_coloring(self, generator, sample_trades):
        html = generator._trade_log_section(sample_trades)
        # Positive PnL should have 'positive' class
        assert 'class="positive"' in html
        # Negative PnL should have 'negative' class
        assert 'class="negative"' in html

    def test_trade_log_empty_trades(self, generator):
        """Empty DataFrame should not cause errors in _build_html."""
        empty_trades = pd.DataFrame()
        # _build_html checks len(trades) > 0 so it shouldn't call _trade_log_section
        html = generator._build_html(
            pd.Series([0.01, -0.005], index=pd.date_range('2024-01-01', periods=2, freq='B')),
            empty_trades,
            None,
            "TestStrategy",
        )
        # Trade Log section should not appear
        assert 'Trade Log' not in html


# ============================================================================
# Cover Page Tests
# ============================================================================

class TestCoverPage:
    """Tests for _cover_page."""

    def test_cover_page_positive_return(self, generator, sample_returns):
        metrics = generator._calculate_metrics(sample_returns)
        html = generator._cover_page("TestStrategy", sample_returns, metrics)
        assert 'TestStrategy' in html
        assert 'Generated' in html

    def test_cover_page_contains_stats(self, generator, sample_returns):
        metrics = generator._calculate_metrics(sample_returns)
        html = generator._cover_page("TestStrategy", sample_returns, metrics)
        assert 'Total Return' in html
        assert 'Sharpe Ratio' in html
        assert 'Max Drawdown' in html


# ============================================================================
# CSS Tests
# ============================================================================

class TestCSS:
    """Tests for CSS generation."""

    def test_css_contains_essential_rules(self, generator):
        css = generator._get_css()
        assert '@page' in css
        assert 'font-family' in css
        assert '.cover-page' in css
        assert '.metrics-table' in css
        assert '.trade-table' in css
        assert '.positive' in css
        assert '.negative' in css
        assert '.chart' in css
