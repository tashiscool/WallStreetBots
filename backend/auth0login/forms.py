from typing import ClassVar

from django import forms

# from django.core.exceptions import ValidationError


class CredentialForm(forms.Form):
    """credential for user."""

    alpaca_id = forms.CharField(help_text="Your Alpaca ID")
    alpaca_key = forms.CharField(help_text="Your Alpaca Key")

    def get_id(self):
        return self.cleaned_data["alpaca_id"]

    def get_key(self):
        return self.cleaned_data["alpaca_key"]


class OrderForm(forms.Form):
    """manual orders from user."""

    ORDERTYPES: ClassVar[list[tuple[str, str]]] = [
        ("market", "Market"),
        ("limit", "Limit"),
        ("stop", "Stop"),
    ]
    TRANSACTIONTYPES: ClassVar[list[tuple[str, str]]] = [
        ("buy", "Buy"),
        ("sell", "Sell"),
    ]
    TIMEINFORCE: ClassVar[list[tuple[str, str]]] = [
        ("day", "Day"),
        ("gtc", "Good Until Canceled"),
    ]
    ticker = forms.CharField(help_text="Stock ticker")
    order_type = forms.ChoiceField(choices=ORDERTYPES, help_text="Order Type")
    transaction_type = forms.ChoiceField(
        choices=TRANSACTIONTYPES, help_text="Transaction Type"
    )
    quantity = forms.DecimalField(decimal_places=2, help_text="Quantity")
    limit_price = forms.DecimalField(
        decimal_places=2, required=False, help_text="Limit Price (for limit orders)"
    )
    stop_price = forms.DecimalField(
        decimal_places=2, required=False, help_text="Stop Price (for stop orders)"
    )
    time_in_force = forms.ChoiceField(choices=TIMEINFORCE, help_text="Time in force")

    def clean(self):
        """Validate that required prices are provided for limit/stop orders."""
        cleaned_data = super().clean()
        order_type = cleaned_data.get("order_type")
        limit_price = cleaned_data.get("limit_price")
        stop_price = cleaned_data.get("stop_price")

        if order_type == "limit" and not limit_price:
            raise forms.ValidationError("Limit price is required for limit orders")
        if order_type == "stop" and not stop_price:
            raise forms.ValidationError("Stop price is required for stop orders")

        return cleaned_data

    def place_order(self, user, user_details):
        ticker = self.cleaned_data["ticker"].upper()
        order_type = self.cleaned_data["order_type"]
        transaction_type = self.cleaned_data["transaction_type"]
        quantity = self.cleaned_data["quantity"]
        limit_price = self.cleaned_data.get("limit_price")
        stop_price = self.cleaned_data.get("stop_price")
        time_in_force = self.cleaned_data["time_in_force"]
        from backend.tradingbot.apiutility import place_general_order

        try:
            place_general_order(
                user=user,
                user_details=user_details,
                ticker=ticker,
                quantity=quantity,
                order_type=order_type,
                transaction_type=transaction_type,
                time_in_force=time_in_force,
                limit_price=limit_price,
                stop_price=stop_price,
            )
            return "Order placed successfully"
        except Exception as e:
            return str(e)


class StrategyForm(forms.Form):
    STRATEGY = [
        ("manual", "Manual portfolio management"),
        (
            "hmm_naive_even_split",
            "HMM model prediction + Even split portfolio",
        ),  # HMMNaiveStrategy()
        (
            "ma_sharp_ratio_monte_carlo",
            "Moving average+Sharpe ratio Monte Carlo simulation",
        ),  # MonteCarloMASharpeRatioStrategy()
        (
            "hmm_sharp_ratio_monte_carlo",
            "HMM model prediction + Sharpe ratio Monte Carlo simulation",
        ),
    ]
    strategy = forms.ChoiceField(
        choices=STRATEGY, help_text="Portfolio Rebalancing Strategy"
    )

    def clean(self):
        """Validate and return normalized strategy form data."""
        cleaned_data = super().clean()
        strategy = cleaned_data.get("strategy")

        valid_strategies = {choice[0] for choice in self.STRATEGY}
        if strategy and strategy not in valid_strategies:
            raise forms.ValidationError("Invalid strategy selected")

        return cleaned_data


class WatchListForm(forms.Form):
    """manual orders from user."""

    ticker = forms.CharField(help_text="Stock ticker")

    def add_to_watchlist(self, user):
        from backend.tradingbot.apiutility import add_stock_to_database

        ticker = self.cleaned_data["ticker"].upper()
        try:
            add_stock_to_database(user=user, ticker=ticker)
            return "Added to Watchlist successfully"
        except Exception as e:
            return str(e)
