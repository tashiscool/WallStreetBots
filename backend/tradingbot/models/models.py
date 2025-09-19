"""Essential Django models for the trading bot system."""

from django.contrib.auth.models import User
from django.db import models
from django.utils import timezone
import json


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


class ValidationRun(models.Model):
    """Validation run record for tracking validation history."""
    
    VALIDATION_STATES = [
        ('HEALTHY', 'Healthy'),
        ('THROTTLE', 'Throttled'),
        ('HALT', 'Halted'),
    ]
    
    run_id = models.CharField(max_length=100, unique=True, help_text="Unique validation run identifier")
    strategy_name = models.CharField(max_length=100, help_text="Strategy being validated")
    timestamp = models.DateTimeField(default=timezone.now, help_text="Validation run timestamp")
    validation_state = models.CharField(max_length=20, choices=VALIDATION_STATES, help_text="Validation state")
    overall_recommendation = models.CharField(max_length=10, help_text="GO or NO-GO recommendation")
    deployment_readiness_score = models.FloatField(help_text="Deployment readiness score (0-1)")
    validation_results = models.JSONField(help_text="Detailed validation results")
    gate_result = models.JSONField(help_text="Validation gate evaluation result")
    failing_criteria = models.JSONField(default=list, help_text="List of failing criteria")
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['strategy_name', 'timestamp']),
            models.Index(fields=['validation_state', 'timestamp']),
        ]
    
    def __str__(self):
        return f"Validation Run {self.run_id}: {self.strategy_name} - {self.overall_recommendation} ({self.deployment_readiness_score:.2%})"


class SignalValidationMetrics(models.Model):
    """Signal validation performance metrics."""
    
    strategy_name = models.CharField(max_length=100, help_text="Strategy name")
    timestamp = models.DateTimeField(default=timezone.now, help_text="Metrics timestamp")
    total_signals = models.IntegerField(help_text="Total signals processed")
    validated_signals = models.IntegerField(help_text="Signals that passed validation")
    rejected_signals = models.IntegerField(help_text="Signals that failed validation")
    average_validation_score = models.FloatField(help_text="Average validation score")
    validation_latency_ms = models.FloatField(help_text="Average validation latency in milliseconds")
    false_positive_rate = models.FloatField(help_text="False positive rate")
    false_negative_rate = models.FloatField(help_text="False negative rate")
    precision = models.FloatField(help_text="Precision metric")
    recall = models.FloatField(help_text="Recall metric")
    f1_score = models.FloatField(help_text="F1 score")
    overall_health = models.CharField(max_length=20, help_text="Overall validation health")
    issues = models.JSONField(default=list, help_text="Identified issues")
    recommendations = models.JSONField(default=list, help_text="Recommendations")
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['strategy_name', 'timestamp']),
            models.Index(fields=['overall_health', 'timestamp']),
        ]
    
    def __str__(self):
        return f"Signal Metrics {self.strategy_name}: {self.overall_health} ({self.total_signals} signals)"


class DataQualityMetrics(models.Model):
    """Data quality metrics with validation state awareness."""
    
    timestamp = models.DateTimeField(default=timezone.now, help_text="Metrics timestamp")
    validation_state = models.CharField(max_length=20, help_text="Validation state at time of measurement")
    overall_score = models.FloatField(help_text="Overall data quality score (0-1)")
    source_scores = models.JSONField(help_text="Quality scores by data source")
    recommendations = models.JSONField(default=list, help_text="Data quality recommendations")
    data_age_seconds = models.FloatField(help_text="Age of most recent data in seconds")
    stale_data_count = models.IntegerField(help_text="Number of stale data points")
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['validation_state', 'timestamp']),
            models.Index(fields=['overall_score', 'timestamp']),
        ]
    
    def __str__(self):
        return f"Data Quality {self.validation_state}: {self.overall_score:.2%} score"


class ValidationParameterRegistry(models.Model):
    """Frozen parameter registry for reproducible validation runs."""
    
    strategy_name = models.CharField(max_length=100, help_text="Strategy name")
    frozen_params = models.JSONField(help_text="Frozen parameter set")
    random_seed = models.IntegerField(help_text="Random seed used")
    requirements_sha256 = models.CharField(max_length=64, help_text="SHA256 of requirements.txt")
    python_version = models.CharField(max_length=50, help_text="Python version")
    git_commit = models.CharField(max_length=40, help_text="Git commit hash")
    created_at = models.DateTimeField(default=timezone.now, help_text="Creation timestamp")
    file_path = models.CharField(max_length=500, help_text="Path to frozen parameter file")
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['strategy_name', 'created_at']),
            models.Index(fields=['git_commit']),
        ]
    
    def __str__(self):
        return f"Frozen Params {self.strategy_name}: {self.git_commit[:8]}"

