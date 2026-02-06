"""Pre-configured report templates for common use cases."""

from .pdf_report import PDFReportConfig


class ReportTemplates:
    """Factory methods for common report configurations."""

    @staticmethod
    def weekly_performance() -> PDFReportConfig:
        return PDFReportConfig(
            title="Weekly Performance Report",
            cover_page=False,
            executive_summary=True,
            equity_curve=True,
            drawdown_chart=True,
            monthly_heatmap=False,
            returns_distribution=False,
            rolling_sharpe=False,
            trade_log=True,
            trade_log_max_rows=30,
            risk_metrics=True,
            tax_summary=False,
        )

    @staticmethod
    def monthly_detailed() -> PDFReportConfig:
        return PDFReportConfig(
            title="Monthly Performance Report",
            cover_page=True,
            executive_summary=True,
            equity_curve=True,
            drawdown_chart=True,
            monthly_heatmap=True,
            returns_distribution=True,
            rolling_sharpe=True,
            trade_log=True,
            trade_log_max_rows=100,
            risk_metrics=True,
            tax_summary=False,
        )

    @staticmethod
    def quarterly_review() -> PDFReportConfig:
        return PDFReportConfig(
            title="Quarterly Performance Review",
            cover_page=True,
            executive_summary=True,
            equity_curve=True,
            drawdown_chart=True,
            monthly_heatmap=True,
            returns_distribution=True,
            rolling_sharpe=True,
            trade_log=True,
            trade_log_max_rows=200,
            risk_metrics=True,
            tax_summary=False,
        )

    @staticmethod
    def year_end_tax() -> PDFReportConfig:
        return PDFReportConfig(
            title="Year-End Tax Report",
            cover_page=True,
            executive_summary=True,
            equity_curve=True,
            drawdown_chart=False,
            monthly_heatmap=True,
            returns_distribution=False,
            rolling_sharpe=False,
            trade_log=True,
            trade_log_max_rows=500,
            risk_metrics=False,
            tax_summary=True,
        )
