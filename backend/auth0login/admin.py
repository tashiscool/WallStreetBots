from django.contrib import admin

from .models import BotInstance, Credential

# Import models with fallback for testing
try:
    from ..tradingbot.models import Order, Portfolio
    TRADING_MODELS_AVAILABLE = True
except ImportError:
    Order = Portfolio = None
    TRADING_MODELS_AVAILABLE = False


class CredentialAdmin(admin.ModelAdmin):
    list_display = ("user", "alpaca_id", "alpaca_key")


class BotInstanceAdmin(admin.ModelAdmin):
    list_display = ("name", "portfolio", "user", "bot")


# Register auth0login models
admin.site.register(Credential, CredentialAdmin)
admin.site.register(BotInstance, BotInstanceAdmin)


# Register trading models if available
if TRADING_MODELS_AVAILABLE and Order is not None:
    class OrderAdmin(admin.ModelAdmin):
        list_display = (
            "bot",
            "stock",
            "side",
            "quantity",
            "price",
            "status",
            "date",
        )
    admin.site.register(Order, OrderAdmin)

if TRADING_MODELS_AVAILABLE and Portfolio is not None:
    class PortfolioAdmin(admin.ModelAdmin):
        list_display = ("name", "user", "strategy")
    admin.site.register(Portfolio, PortfolioAdmin)
