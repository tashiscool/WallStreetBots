"""Essential Django models for the trading bot system."""

from django.contrib.auth.models import User
from django.db import models


class Portfolio(models.Model):
    """Portfolio for a user."""
    
    STRATEGY = [
        ("manual", "Manual portfolio management"),
        (
            "hmm_naive_even_split",
            "HMM model prediction + Even split portfolio",
        ),
        (
            "ma_sharp_ratio_monte_carlo",
            "Moving average+Sharpe ratio Monte Carlo simulation",
        ),
        (
            "hmm_sharp_ratio_monte_carlo",
            "HMM model prediction + Sharpe ratio Monte Carlo simulation",
        ),
    ]
    
    name = models.CharField(max_length=100, blank=False, help_text="Portfolio name")
    user = models.OneToOneField(
        User, help_text="Associated user", on_delete=models.CASCADE
    )
    strategy = models.CharField(
        max_length=100, choices=STRATEGY, default="manual", help_text="Strategy"
    )

    # Metadata
    class Meta:
        ordering = ["name"]

    # Methods
    def __str__(self):
        return f"Portfolio: {self.name} \n User: {self.user!s} \n Strategy: {self.strategy}"


class Bot(models.Model):
    """Bot entity."""

    name = models.CharField(max_length=100, blank=False, help_text="Bot name")
    user = models.ForeignKey(
        User, help_text="Associated user", on_delete=models.CASCADE
    )
    portfolio = models.ForeignKey(
        Portfolio, help_text="Associated portfolio", on_delete=models.CASCADE
    )

    # Metadata
    class Meta:
        ordering = ["name"]

    # Methods
    def __str__(self):
        return f"Bot: {self.name} \n User: {self.user!s} \n Portfolio: {self.portfolio.name}"


class Company(models.Model):
    """Company entity."""

    name = models.TextField()
    ticker = models.CharField(max_length=255, primary_key=True)

    # Metadata
    class Meta:
        ordering = ["ticker"]

    # Methods
    def __str__(self):
        return f"Company: {self.name} \n Ticker: {self.ticker}"


class Stock(models.Model):
    """Stock entity."""

    company = models.ForeignKey(
        Company, help_text="Associated company", on_delete=models.CASCADE
    )
    price = models.DecimalField(max_digits=10, decimal_places=2)
    date = models.DateField(auto_now=False, auto_now_add=False)

    # Metadata
    class Meta:
        ordering = ["date"]

    # Methods
    def __str__(self):
        return f"Stock: {self.company.ticker} \n Price: {self.price} \n Date: {self.date}"


class Order(models.Model):
    """Order entity."""

    SIDE = [("buy", "Buy"), ("sell", "Sell")]
    STATUS = [
        ("pending", "Pending"),
        ("filled", "Filled"),
        ("cancelled", "Cancelled"),
    ]

    bot = models.ForeignKey(Bot, help_text="Associated bot", on_delete=models.CASCADE)
    stock = models.ForeignKey(
        Stock, help_text="Associated stock", on_delete=models.CASCADE
    )
    side = models.CharField(max_length=10, choices=SIDE, help_text="Order side")
    quantity = models.IntegerField(help_text="Order quantity")
    price = models.DecimalField(max_digits=10, decimal_places=2, help_text="Order price")
    status = models.CharField(
        max_length=20, choices=STATUS, default="pending", help_text="Order status"
    )
    date = models.DateField(auto_now=False, auto_now_add=False)

    # Metadata
    class Meta:
        ordering = ["date"]

    # Methods
    def __str__(self):
        return f"Order: {self.side} {self.quantity} {self.stock.company.ticker} @ {self.price} \n Status: {self.status} \n Date: {self.date}"
