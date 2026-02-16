from django.apps import AppConfig


class TradingbotConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "backend.tradingbot"

    def ready(self):
        # Production strategies are managed by ProductionStrategyManager,
        # not by APScheduler. No automatic trading pipeline is started here.
        pass
