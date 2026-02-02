"""Essential Django models for the trading bot system."""

from datetime import date

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
    price = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    date = models.DateField(default=date.today)

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

    bot = models.ForeignKey(Bot, help_text="Associated bot", on_delete=models.CASCADE, null=True, blank=True)
    stock = models.ForeignKey(
        Stock, help_text="Associated stock", on_delete=models.CASCADE, null=True, blank=True
    )
    side = models.CharField(max_length=10, choices=SIDE, help_text="Order side", default="buy")
    quantity = models.IntegerField(help_text="Order quantity", default=0)
    price = models.DecimalField(max_digits=10, decimal_places=2, help_text="Order price", default=0)
    status = models.CharField(
        max_length=20, choices=STATUS, default="pending", help_text="Order status"
    )
    date = models.DateField(default=date.today)

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


class TradeSignalSnapshot(models.Model):
    """Capture signal state at the moment of trade execution for transparency.

    This model stores a snapshot of all technical indicators and signals
    that were evaluated when a trade was triggered, enabling users to
    understand "why" a trade was made.
    """

    DIRECTION_CHOICES = [
        ('buy', 'Buy'),
        ('sell', 'Sell'),
        ('buy_to_cover', 'Buy to Cover'),
        ('sell_short', 'Sell Short'),
    ]

    # Trade identification
    trade_id = models.CharField(
        max_length=100,
        unique=True,
        db_index=True,
        help_text="Unique trade identifier (alpaca order ID or internal ID)"
    )
    order = models.ForeignKey(
        'Order',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='signal_snapshots',
        help_text="Associated Order if available"
    )

    # Trade context
    strategy_name = models.CharField(
        max_length=100,
        db_index=True,
        help_text="Strategy that generated the trade"
    )
    symbol = models.CharField(
        max_length=20,
        db_index=True,
        help_text="Trading symbol"
    )
    direction = models.CharField(
        max_length=20,
        choices=DIRECTION_CHOICES,
        help_text="Trade direction"
    )
    entry_price = models.DecimalField(
        max_digits=12,
        decimal_places=4,
        help_text="Price at entry"
    )
    quantity = models.DecimalField(
        max_digits=12,
        decimal_places=4,
        help_text="Trade quantity"
    )

    # Signal snapshot at entry time
    signals_at_entry = models.JSONField(
        default=dict,
        help_text="""Snapshot of all signals at trade time. Structure:
        {
            "rsi": {"value": 28, "threshold": 30, "triggered": true, "period": 14},
            "macd": {"value": -0.5, "signal": -0.3, "histogram": -0.2, "crossover": true},
            "volume": {"current": 5000000, "average": 2000000, "ratio": 2.5},
            "price_action": {"change_pct": -3.2, "from_sma20": -5.1, "from_sma50": -8.2},
            "bollinger": {"upper": 155.2, "lower": 148.3, "position": "below_lower"},
            "stochastic": {"k": 15, "d": 18, "oversold": true},
            "atr": {"value": 2.5, "percent": 1.8},
            "trend": {"short_term": "bearish", "medium_term": "bearish", "long_term": "bullish"}
        }"""
    )

    # Confidence and scoring
    confidence_score = models.IntegerField(
        default=50,
        help_text="Confidence score (0-100) based on signal alignment"
    )
    signals_triggered = models.IntegerField(
        default=0,
        help_text="Number of signals that triggered this trade"
    )
    signals_checked = models.IntegerField(
        default=0,
        help_text="Total signals checked"
    )

    # Human-readable explanation
    explanation = models.TextField(
        blank=True,
        help_text="Auto-generated explanation of why this trade was triggered"
    )

    # Similar historical trades
    similar_historical_trades = models.JSONField(
        default=list,
        help_text="""List of similar historical trade setups:
        [
            {"trade_id": "abc123", "similarity": 0.92, "outcome": "profit", "pnl_pct": 5.2},
            {"trade_id": "def456", "similarity": 0.87, "outcome": "loss", "pnl_pct": -2.1}
        ]"""
    )

    # NEW: Structured entry reasoning
    entry_reasoning = models.JSONField(
        null=True,
        blank=True,
        help_text="""Structured reasoning for trade entry:
        {
            "summary": "RSI oversold + MACD bullish crossover on high volume",
            "signals": {
                "rsi": {"value": 28, "threshold": 30, "met": true},
                "macd": {"histogram": 0.5, "crossover": true, "met": true},
                "volume": {"ratio": 2.3, "threshold": 1.5, "met": true}
            },
            "confidence": 85,
            "market_context": {
                "spy_trend": "up",
                "vix": 18.5,
                "sector_performance": 1.2
            },
            "timestamp": "2024-01-15T10:30:00Z"
        }"""
    )

    # NEW: Structured exit reasoning
    exit_reasoning = models.JSONField(
        null=True,
        blank=True,
        help_text="""Structured reasoning for trade exit:
        {
            "summary": "Profit target reached",
            "trigger": "take_profit",
            "held_duration": "3 days",
            "held_duration_hours": 72,
            "signals_at_exit": {
                "rsi": {"value": 72, "overbought": true},
                "price_vs_entry": {"pct_gain": 5.2}
            },
            "timestamp": "2024-01-18T14:45:00Z"
        }"""
    )

    # NEW: Post-trade outcome analysis
    outcome_analysis = models.JSONField(
        null=True,
        blank=True,
        help_text="""Post-trade analysis filled after exit:
        {
            "pnl": 150.00,
            "pnl_pct": 3.2,
            "vs_hold": 1.5,
            "timing_score": "good",
            "entry_timing_analysis": {
                "optimal_entry_price": 148.50,
                "actual_vs_optimal_pct": 0.5,
                "entered_early_or_late": "slightly_early"
            },
            "exit_timing_analysis": {
                "max_price_during_hold": 158.20,
                "captured_pct_of_max": 82.5
            },
            "similar_trades_avg_pnl": 2.8,
            "similar_trades_win_rate": 0.65,
            "notes": "Exited slightly early, could have captured 0.5% more",
            "analyzed_at": "2024-01-18T15:00:00Z"
        }"""
    )

    # Outcome tracking (filled in later)
    exit_price = models.DecimalField(
        max_digits=12,
        decimal_places=4,
        null=True,
        blank=True,
        help_text="Price at exit"
    )
    exit_timestamp = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the position was closed"
    )
    outcome = models.CharField(
        max_length=20,
        null=True,
        blank=True,
        choices=[
            ('profit', 'Profit'),
            ('loss', 'Loss'),
            ('break_even', 'Break Even'),
            ('open', 'Still Open'),
        ],
        help_text="Trade outcome"
    )
    pnl_amount = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Profit/loss amount"
    )
    pnl_percent = models.DecimalField(
        max_digits=8,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Profit/loss percentage"
    )

    # Timestamps
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="When the snapshot was captured"
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="Last update"
    )

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['strategy_name', 'created_at']),
            models.Index(fields=['symbol', 'created_at']),
            models.Index(fields=['confidence_score', 'created_at']),
            models.Index(fields=['outcome', 'created_at']),
        ]
        verbose_name = "Trade Signal Snapshot"
        verbose_name_plural = "Trade Signal Snapshots"

    def __str__(self):
        return f"{self.symbol} {self.direction} | {self.strategy_name} | Confidence: {self.confidence_score}%"

    def get_triggered_signals(self) -> list:
        """Return list of signals that triggered this trade."""
        triggered = []
        for signal_name, signal_data in self.signals_at_entry.items():
            if isinstance(signal_data, dict) and signal_data.get('triggered'):
                triggered.append({
                    'name': signal_name,
                    'data': signal_data
                })
        return triggered

    def calculate_similarity(self, other_snapshot) -> float:
        """Calculate similarity score with another snapshot."""
        if not isinstance(other_snapshot, TradeSignalSnapshot):
            return 0.0

        score = 0.0
        weights = 0.0

        # Compare key signals
        my_signals = self.signals_at_entry
        other_signals = other_snapshot.signals_at_entry

        # RSI similarity
        if 'rsi' in my_signals and 'rsi' in other_signals:
            rsi_diff = abs(my_signals['rsi'].get('value', 50) - other_signals['rsi'].get('value', 50))
            score += (1 - min(rsi_diff / 50, 1)) * 0.25
            weights += 0.25

        # Volume ratio similarity
        if 'volume' in my_signals and 'volume' in other_signals:
            my_ratio = my_signals['volume'].get('ratio', 1)
            other_ratio = other_signals['volume'].get('ratio', 1)
            ratio_diff = abs(my_ratio - other_ratio) / max(my_ratio, other_ratio, 1)
            score += (1 - min(ratio_diff, 1)) * 0.2
            weights += 0.2

        # Price action similarity
        if 'price_action' in my_signals and 'price_action' in other_signals:
            my_change = my_signals['price_action'].get('change_pct', 0)
            other_change = other_signals['price_action'].get('change_pct', 0)
            change_diff = abs(my_change - other_change)
            score += (1 - min(change_diff / 10, 1)) * 0.25
            weights += 0.25

        # MACD similarity
        if 'macd' in my_signals and 'macd' in other_signals:
            if my_signals['macd'].get('crossover') == other_signals['macd'].get('crossover'):
                score += 0.15
            weights += 0.15

        # Same direction bonus
        if self.direction == other_snapshot.direction:
            score += 0.15
            weights += 0.15

        return score / weights if weights > 0 else 0.0

    def record_exit(
        self,
        exit_price: float,
        pnl_amount: float = None,
        pnl_percent: float = None,
        exit_reasoning: dict = None
    ):
        """Record trade exit and calculate outcome.

        Args:
            exit_price: Price at exit
            pnl_amount: Profit/loss amount
            pnl_percent: Profit/loss percentage
            exit_reasoning: Structured exit reasoning dict
        """
        from decimal import Decimal
        self.exit_price = Decimal(str(exit_price))
        self.exit_timestamp = timezone.now()

        if pnl_amount is not None:
            self.pnl_amount = Decimal(str(pnl_amount))
        if pnl_percent is not None:
            self.pnl_percent = Decimal(str(pnl_percent))

        if exit_reasoning is not None:
            self.exit_reasoning = exit_reasoning

        # Determine outcome
        if self.pnl_percent is not None:
            if self.pnl_percent > 0.5:
                self.outcome = 'profit'
            elif self.pnl_percent < -0.5:
                self.outcome = 'loss'
            else:
                self.outcome = 'break_even'

        self.save()

    def set_entry_reasoning(
        self,
        summary: str,
        signals: dict,
        confidence: int,
        market_context: dict = None
    ):
        """Set structured entry reasoning.

        Args:
            summary: Human-readable summary (e.g., "RSI oversold + MACD crossover")
            signals: Dict of signal states {signal_name: {value, threshold, met, ...}}
            confidence: Confidence score 0-100
            market_context: Optional market context (VIX, SPY trend, sector performance)
        """
        self.entry_reasoning = {
            'summary': summary,
            'signals': signals,
            'confidence': confidence,
            'market_context': market_context or {},
            'timestamp': timezone.now().isoformat(),
        }
        self.confidence_score = confidence
        self.save()

    def set_exit_reasoning(
        self,
        summary: str,
        trigger: str,
        signals_at_exit: dict = None
    ):
        """Set structured exit reasoning.

        Args:
            summary: Human-readable summary (e.g., "Profit target reached")
            trigger: Exit trigger type (take_profit, stop_loss, trailing_stop, signal, manual)
            signals_at_exit: Optional dict of signals at exit time
        """
        # Calculate held duration
        if self.created_at:
            held_delta = timezone.now() - self.created_at
            held_hours = held_delta.total_seconds() / 3600
            held_duration = self._format_duration(held_hours)
        else:
            held_hours = 0
            held_duration = "unknown"

        self.exit_reasoning = {
            'summary': summary,
            'trigger': trigger,
            'held_duration': held_duration,
            'held_duration_hours': round(held_hours, 2),
            'signals_at_exit': signals_at_exit or {},
            'timestamp': timezone.now().isoformat(),
        }
        self.save()

    def set_outcome_analysis(self, analysis: dict):
        """Set post-trade outcome analysis.

        Args:
            analysis: Dict with analysis fields (pnl, pnl_pct, vs_hold, timing_score, etc.)
        """
        analysis['analyzed_at'] = timezone.now().isoformat()
        self.outcome_analysis = analysis
        self.save()

    def _format_duration(self, hours: float) -> str:
        """Format duration in hours to human-readable string."""
        if hours < 1:
            return f"{int(hours * 60)} minutes"
        elif hours < 24:
            return f"{hours:.1f} hours"
        elif hours < 48:
            return "1 day"
        else:
            days = int(hours / 24)
            return f"{days} days"

    @property
    def reasoning_summary(self) -> str:
        """Get combined reasoning summary for display."""
        parts = []
        if self.entry_reasoning and self.entry_reasoning.get('summary'):
            parts.append(f"Entry: {self.entry_reasoning['summary']}")
        if self.exit_reasoning and self.exit_reasoning.get('summary'):
            parts.append(f"Exit: {self.exit_reasoning['summary']}")
        return " | ".join(parts) if parts else self.explanation or "No reasoning captured"

    @property
    def exit_trigger_type(self) -> str:
        """Get the exit trigger type if available."""
        if self.exit_reasoning:
            return self.exit_reasoning.get('trigger', 'unknown')
        return 'unknown'

    @property
    def held_duration_display(self) -> str:
        """Get formatted held duration."""
        if self.exit_reasoning:
            return self.exit_reasoning.get('held_duration', 'unknown')
        if self.exit_timestamp and self.created_at:
            hours = (self.exit_timestamp - self.created_at).total_seconds() / 3600
            return self._format_duration(hours)
        return 'still open'

    def to_dict_with_reasoning(self) -> dict:
        """Convert to dictionary with full reasoning data."""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'strategy_name': self.strategy_name,
            'entry_price': float(self.entry_price),
            'quantity': float(self.quantity),
            'confidence_score': self.confidence_score,
            'signals_triggered': self.signals_triggered,
            'signals_checked': self.signals_checked,
            'explanation': self.explanation,
            'reasoning_summary': self.reasoning_summary,
            # Entry details
            'entry_reasoning': self.entry_reasoning,
            'signals_at_entry': self.signals_at_entry,
            # Exit details
            'exit_price': float(self.exit_price) if self.exit_price else None,
            'exit_timestamp': self.exit_timestamp.isoformat() if self.exit_timestamp else None,
            'exit_reasoning': self.exit_reasoning,
            'exit_trigger': self.exit_trigger_type,
            'held_duration': self.held_duration_display,
            # Outcome
            'outcome': self.outcome,
            'pnl_amount': float(self.pnl_amount) if self.pnl_amount else None,
            'pnl_percent': float(self.pnl_percent) if self.pnl_percent else None,
            'outcome_analysis': self.outcome_analysis,
            # Similar trades
            'similar_historical_trades': self.similar_historical_trades,
            # Timestamps
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class SignalValidationHistory(models.Model):
    """Store individual signal validation results for analysis.

    GAP FIX: This model implements the Database Integration Gap - providing
    persistence for individual signal validation results for historical analysis
    and performance feedback loops.

    This tracks each signal that goes through validation, capturing:
    - The signal's strength score and quality grade
    - The recommended action (execute, reduce_size, wait, reject)
    - Position sizing recommendations
    - Whether the signal was actually executed
    - Optional trade outcome for accuracy tracking
    """

    strategy_name = models.CharField(
        max_length=100,
        db_index=True,
        help_text="Name of the strategy that generated the signal"
    )
    symbol = models.CharField(
        max_length=20,
        db_index=True,
        help_text="Trading symbol (e.g., AAPL, SPY)"
    )
    signal_type = models.CharField(
        max_length=50,
        help_text="Type of signal (e.g., BUY_CALL, SELL_PUT, BREAKOUT)"
    )
    strength_score = models.FloatField(
        help_text="Signal strength score (0-100)"
    )
    quality_grade = models.CharField(
        max_length=2,
        help_text="Quality grade (A, B, C, D, F)"
    )
    recommended_action = models.CharField(
        max_length=20,
        help_text="Recommended action (execute, reduce_size, wait, reject)"
    )
    suggested_position_size = models.FloatField(
        default=1.0,
        help_text="Suggested position size multiplier (0.0-1.0)"
    )
    confidence_level = models.FloatField(
        default=0.5,
        help_text="Confidence level in the validation (0.0-1.0)"
    )
    validation_details = models.JSONField(
        default=dict,
        help_text="Detailed validation metrics and criteria results"
    )

    # Execution tracking
    was_executed = models.BooleanField(
        default=False,
        help_text="Whether the signal was actually executed"
    )
    execution_timestamp = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the signal was executed (if applicable)"
    )
    actual_position_size = models.FloatField(
        null=True,
        blank=True,
        help_text="Actual position size used"
    )

    # Trade outcome tracking (for accuracy feedback loop)
    trade_outcome = models.CharField(
        max_length=20,
        null=True,
        blank=True,
        help_text="Trade outcome (profit, loss, break_even, pending)"
    )
    trade_pnl = models.FloatField(
        null=True,
        blank=True,
        help_text="Trade P&L if closed"
    )
    trade_pnl_percent = models.FloatField(
        null=True,
        blank=True,
        help_text="Trade P&L percentage"
    )
    outcome_recorded_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the trade outcome was recorded"
    )

    # Timestamps
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="When the signal was validated"
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="Last update timestamp"
    )

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['strategy_name', 'created_at']),
            models.Index(fields=['symbol', 'created_at']),
            models.Index(fields=['quality_grade', 'created_at']),
            models.Index(fields=['recommended_action', 'created_at']),
            models.Index(fields=['was_executed', 'created_at']),
            models.Index(fields=['trade_outcome', 'created_at']),
        ]
        verbose_name = "Signal Validation History"
        verbose_name_plural = "Signal Validation Histories"

    def __str__(self):
        return (
            f"{self.strategy_name} | {self.symbol} | "
            f"Score: {self.strength_score:.1f} | Grade: {self.quality_grade} | "
            f"Action: {self.recommended_action}"
        )

    def record_execution(self, actual_size: float = None):
        """Record that this signal was executed."""
        self.was_executed = True
        self.execution_timestamp = timezone.now()
        if actual_size is not None:
            self.actual_position_size = actual_size
        self.save()

    def record_outcome(self, outcome: str, pnl: float = None, pnl_percent: float = None):
        """Record the trade outcome for accuracy tracking.

        Args:
            outcome: Trade outcome (profit, loss, break_even, pending)
            pnl: Absolute P&L value
            pnl_percent: P&L as a percentage
        """
        self.trade_outcome = outcome
        self.trade_pnl = pnl
        self.trade_pnl_percent = pnl_percent
        self.outcome_recorded_at = timezone.now()
        self.save()

    @classmethod
    def calculate_validation_accuracy(cls, strategy_name: str = None, days: int = 30):
        """Calculate validation accuracy based on trade outcomes.

        This method helps close the performance feedback loop by analyzing
        how well validation predictions correlate with actual trade outcomes.

        Args:
            strategy_name: Filter by strategy (optional)
            days: Number of days to analyze

        Returns:
            Dictionary with accuracy metrics
        """
        from datetime import timedelta

        cutoff = timezone.now() - timedelta(days=days)

        queryset = cls.objects.filter(
            was_executed=True,
            trade_outcome__isnull=False,
            created_at__gte=cutoff,
        )

        if strategy_name:
            queryset = queryset.filter(strategy_name=strategy_name)

        if not queryset.exists():
            return {'error': 'No completed trades found for analysis'}

        # Calculate accuracy metrics
        total_trades = queryset.count()
        profitable_trades = queryset.filter(trade_outcome='profit').count()
        losing_trades = queryset.filter(trade_outcome='loss').count()

        # Grade breakdown
        grade_results = {}
        for grade in ['A', 'B', 'C', 'D', 'F']:
            grade_trades = queryset.filter(quality_grade=grade)
            grade_total = grade_trades.count()
            if grade_total > 0:
                grade_profits = grade_trades.filter(trade_outcome='profit').count()
                grade_results[grade] = {
                    'total': grade_total,
                    'profitable': grade_profits,
                    'win_rate': grade_profits / grade_total,
                }

        return {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'losing_trades': losing_trades,
            'overall_win_rate': profitable_trades / total_trades if total_trades > 0 else 0,
            'by_grade': grade_results,
            'analysis_period_days': days,
        }


class StrategyAllocationLimit(models.Model):
    """Per-strategy capital allocation limits with enforcement tracking.

    This model tracks capital allocation limits for each trading strategy,
    enabling enforcement of per-strategy capital limits to prevent any
    single strategy from consuming too much capital.
    """

    STRATEGY_CHOICES = [
        ('wsb-dip-bot', 'WSB Dip Bot'),
        ('wheel', 'Wheel Strategy'),
        ('momentum-weeklies', 'Momentum Weeklies'),
        ('earnings-protection', 'Earnings Protection'),
        ('debit-spreads', 'Debit Spreads'),
        ('leaps-tracker', 'LEAPS Tracker'),
        ('lotto-scanner', 'Lotto Scanner'),
        ('swing-trading', 'Swing Trading'),
        ('spx-credit-spreads', 'SPX Credit Spreads'),
        ('index-baseline', 'Index Baseline'),
        ('crypto-dip-bot', 'Crypto Dip Bot'),
    ]

    # User association
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='strategy_allocations',
        help_text="User who owns this allocation"
    )

    # Strategy identification
    strategy_name = models.CharField(
        max_length=100,
        db_index=True,
        help_text="Strategy identifier (e.g., wsb-dip-bot)"
    )

    # Allocation settings
    allocated_pct = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        help_text="Allocated percentage of portfolio (e.g., 20.00 for 20%)"
    )
    allocated_amount = models.DecimalField(
        max_digits=14,
        decimal_places=2,
        default=0,
        help_text="Allocated dollar amount (calculated from portfolio value)"
    )

    # Current usage tracking
    current_exposure = models.DecimalField(
        max_digits=14,
        decimal_places=2,
        default=0,
        help_text="Current dollar exposure (sum of open positions)"
    )
    reserved_amount = models.DecimalField(
        max_digits=14,
        decimal_places=2,
        default=0,
        help_text="Amount reserved for pending orders"
    )

    # Metadata
    is_enabled = models.BooleanField(
        default=True,
        help_text="Whether allocation limits are enforced for this strategy"
    )
    last_updated = models.DateTimeField(
        auto_now=True,
        help_text="Last update timestamp"
    )
    last_reconciled = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Last reconciliation timestamp"
    )

    class Meta:
        unique_together = ['user', 'strategy_name']
        ordering = ['-allocated_pct']
        indexes = [
            models.Index(fields=['user', 'strategy_name']),
            models.Index(fields=['strategy_name', 'last_updated']),
        ]

    def __str__(self):
        return f"{self.strategy_name}: {self.allocated_pct}% (${self.current_exposure}/{self.allocated_amount})"

    @property
    def available_capital(self):
        """Calculate available capital for new positions."""
        return max(0, self.allocated_amount - self.current_exposure - self.reserved_amount)

    @property
    def utilization_pct(self):
        """Calculate utilization percentage."""
        if self.allocated_amount <= 0:
            return 0
        return float((self.current_exposure + self.reserved_amount) / self.allocated_amount * 100)

    @property
    def is_maxed_out(self):
        """Check if strategy has reached its allocation limit."""
        return self.available_capital <= 0

    @property
    def utilization_level(self):
        """Get utilization level category."""
        util = self.utilization_pct
        if util >= 100:
            return 'maxed_out'
        elif util >= 90:
            return 'critical'
        elif util >= 80:
            return 'warning'
        elif util >= 50:
            return 'moderate'
        else:
            return 'low'

    def can_allocate(self, amount):
        """Check if the specified amount can be allocated."""
        from decimal import Decimal
        return Decimal(str(amount)) <= self.available_capital

    def reserve(self, amount):
        """Reserve capital for a pending order.

        Args:
            amount: Amount to reserve

        Returns:
            True if reservation successful, False otherwise
        """
        from decimal import Decimal
        amount = Decimal(str(amount))

        if amount > self.available_capital:
            return False

        self.reserved_amount += amount
        self.save()
        return True

    def release_reservation(self, amount):
        """Release a reservation (order cancelled or failed)."""
        from decimal import Decimal
        amount = Decimal(str(amount))
        self.reserved_amount = max(0, self.reserved_amount - amount)
        self.save()

    def add_exposure(self, amount):
        """Add exposure when a position is opened."""
        from decimal import Decimal
        amount = Decimal(str(amount))
        self.current_exposure += amount
        # Release any corresponding reservation
        self.reserved_amount = max(0, self.reserved_amount - amount)
        self.save()

    def remove_exposure(self, amount):
        """Remove exposure when a position is closed."""
        from decimal import Decimal
        amount = Decimal(str(amount))
        self.current_exposure = max(0, self.current_exposure - amount)
        self.save()

    def recalculate_allocated_amount(self, portfolio_value):
        """Recalculate allocated amount based on current portfolio value."""
        from decimal import Decimal
        portfolio_value = Decimal(str(portfolio_value))
        self.allocated_amount = portfolio_value * (self.allocated_pct / 100)
        self.save()

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            'strategy_name': self.strategy_name,
            'allocated_pct': float(self.allocated_pct),
            'allocated_amount': float(self.allocated_amount),
            'current_exposure': float(self.current_exposure),
            'reserved_amount': float(self.reserved_amount),
            'available_capital': float(self.available_capital),
            'utilization_pct': self.utilization_pct,
            'utilization_level': self.utilization_level,
            'is_maxed_out': self.is_maxed_out,
            'is_enabled': self.is_enabled,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
        }


class AllocationReservation(models.Model):
    """Track pending reservations for orders not yet filled.

    This allows us to reserve capital when an order is submitted,
    then release or convert to exposure when the order completes.
    """

    RESERVATION_STATUS = [
        ('pending', 'Pending'),
        ('filled', 'Filled'),
        ('cancelled', 'Cancelled'),
        ('expired', 'Expired'),
    ]

    allocation = models.ForeignKey(
        StrategyAllocationLimit,
        on_delete=models.CASCADE,
        related_name='reservations',
        help_text="Associated strategy allocation"
    )
    order_id = models.CharField(
        max_length=100,
        unique=True,
        db_index=True,
        help_text="Alpaca order ID or internal order reference"
    )
    symbol = models.CharField(
        max_length=20,
        help_text="Trading symbol"
    )
    amount = models.DecimalField(
        max_digits=14,
        decimal_places=2,
        help_text="Reserved amount"
    )
    status = models.CharField(
        max_length=20,
        choices=RESERVATION_STATUS,
        default='pending',
        help_text="Reservation status"
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Reservation creation time"
    )
    resolved_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the reservation was resolved"
    )

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['allocation', 'status']),
            models.Index(fields=['order_id']),
        ]

    def __str__(self):
        return f"Reservation {self.order_id}: ${self.amount} ({self.status})"

    def resolve(self, status: str):
        """Resolve the reservation.

        Args:
            status: 'filled' or 'cancelled' or 'expired'
        """
        self.status = status
        self.resolved_at = timezone.now()
        self.save()


class CircuitBreakerEvent(models.Model):
    """Track circuit breaker trigger events for history and recovery management.

    This model records when circuit breakers trigger, enabling:
    - Historical analysis of trigger patterns
    - Graduated recovery tracking
    - Performance-based recovery conditions
    - Audit trail of all breaker events
    """

    BREAKER_TYPE_CHOICES = [
        ('daily_loss', 'Daily Loss Limit'),
        ('vix_critical', 'VIX Critical'),
        ('vix_extreme', 'VIX Extreme'),
        ('error_rate', 'Error Rate Spike'),
        ('stale_data', 'Stale Data'),
        ('consecutive_loss', 'Consecutive Losses'),
        ('position_limit', 'Position Count Limit'),
        ('margin_call', 'Margin Call Risk'),
        ('manual', 'Manual Trigger'),
    ]

    RESOLUTION_METHOD_CHOICES = [
        ('auto_time', 'Auto Time-Based'),
        ('auto_condition', 'Auto Condition-Based'),
        ('auto_hybrid', 'Auto Hybrid (Time + Condition)'),
        ('manual', 'Manual Reset'),
        ('override', 'Admin Override'),
        ('pending', 'Pending Resolution'),
    ]

    RECOVERY_MODE_CHOICES = [
        ('paused', 'Paused'),
        ('restricted', 'Restricted'),
        ('cautious', 'Cautious'),
        ('normal', 'Normal'),
    ]

    # User association
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='circuit_breaker_events',
        help_text="User affected by this circuit breaker event"
    )

    # Breaker identification
    breaker_type = models.CharField(
        max_length=50,
        choices=BREAKER_TYPE_CHOICES,
        db_index=True,
        help_text="Type of circuit breaker that triggered"
    )

    # Trigger details
    triggered_at = models.DateTimeField(
        default=timezone.now,
        db_index=True,
        help_text="When the circuit breaker was triggered"
    )
    trigger_value = models.DecimalField(
        max_digits=14,
        decimal_places=4,
        help_text="The value that caused the trigger (e.g., loss amount, VIX level)"
    )
    trigger_threshold = models.DecimalField(
        max_digits=14,
        decimal_places=4,
        help_text="The threshold that was exceeded"
    )
    trigger_context = models.JSONField(
        default=dict,
        blank=True,
        help_text="Additional context (positions, market conditions, etc.)"
    )

    # Resolution tracking
    resolved_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the circuit breaker was resolved (full trading resumed)"
    )
    resolution_method = models.CharField(
        max_length=20,
        choices=RESOLUTION_METHOD_CHOICES,
        default='pending',
        help_text="How the breaker was resolved"
    )

    # Recovery tracking
    current_recovery_mode = models.CharField(
        max_length=20,
        choices=RECOVERY_MODE_CHOICES,
        default='paused',
        help_text="Current recovery mode for this event"
    )
    recovery_mode_until = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When full trading should resume (for time-based recovery)"
    )
    recovery_schedule = models.JSONField(
        default=list,
        blank=True,
        help_text="Recovery schedule with stages and conditions"
    )
    recovery_stage = models.IntegerField(
        default=0,
        help_text="Current stage in recovery schedule (0-indexed)"
    )
    last_stage_advance = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the recovery stage last advanced"
    )

    # Performance tracking during recovery
    recovery_trades_count = models.IntegerField(
        default=0,
        help_text="Number of trades executed during recovery"
    )
    recovery_profitable_trades = models.IntegerField(
        default=0,
        help_text="Number of profitable trades during recovery"
    )
    recovery_pnl = models.DecimalField(
        max_digits=14,
        decimal_places=2,
        default=0,
        help_text="P&L during recovery period"
    )

    # Admin notes
    notes = models.TextField(
        blank=True,
        help_text="Admin notes or justification for actions"
    )

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-triggered_at']
        indexes = [
            models.Index(fields=['user', 'triggered_at']),
            models.Index(fields=['breaker_type', 'triggered_at']),
            models.Index(fields=['user', 'resolved_at']),
            models.Index(fields=['current_recovery_mode']),
        ]

    def __str__(self):
        status = "Active" if not self.resolved_at else "Resolved"
        return f"{self.breaker_type} @ {self.triggered_at.strftime('%Y-%m-%d %H:%M')} ({status})"

    @property
    def is_active(self):
        """Check if this circuit breaker event is still active."""
        return self.resolved_at is None

    @property
    def is_trading_allowed(self):
        """Check if trading is allowed under current recovery mode."""
        return self.current_recovery_mode != 'paused'

    @property
    def position_size_multiplier(self):
        """Get position size multiplier based on recovery mode."""
        multipliers = {
            'paused': 0.0,
            'restricted': 0.25,
            'cautious': 0.5,
            'normal': 1.0,
        }
        return multipliers.get(self.current_recovery_mode, 0.0)

    @property
    def duration_hours(self):
        """Get duration of this event in hours."""
        end_time = self.resolved_at or timezone.now()
        delta = end_time - self.triggered_at
        return delta.total_seconds() / 3600

    @property
    def recovery_win_rate(self):
        """Calculate win rate during recovery period."""
        if self.recovery_trades_count == 0:
            return None
        return self.recovery_profitable_trades / self.recovery_trades_count

    @property
    def can_advance_recovery(self):
        """Check if recovery can advance to next stage."""
        if not self.recovery_schedule or not self.is_active:
            return False

        if self.recovery_stage >= len(self.recovery_schedule) - 1:
            return False  # Already at final stage

        next_stage = self.recovery_schedule[self.recovery_stage + 1]
        now = timezone.now()

        # Check time requirement
        if 'hours_after_trigger' in next_stage:
            required_hours = next_stage['hours_after_trigger']
            if self.duration_hours < required_hours:
                return False

        # Check performance requirement
        if 'min_profitable_trades' in next_stage:
            if self.recovery_profitable_trades < next_stage['min_profitable_trades']:
                return False

        if 'min_win_rate' in next_stage and self.recovery_trades_count > 0:
            if self.recovery_win_rate < next_stage['min_win_rate']:
                return False

        return True

    def advance_recovery_stage(self, force=False, notes=''):
        """Advance to the next recovery stage.

        Args:
            force: Force advance even if conditions not met
            notes: Notes for the advancement

        Returns:
            bool: True if advanced, False otherwise
        """
        if not self.is_active:
            return False

        if not force and not self.can_advance_recovery:
            return False

        if self.recovery_stage >= len(self.recovery_schedule) - 1:
            # Final stage reached - resolve the event
            self.resolve(method='auto_time' if not force else 'manual', notes=notes)
            return True

        self.recovery_stage += 1
        self.last_stage_advance = timezone.now()

        # Update recovery mode from schedule
        if self.recovery_schedule:
            new_stage = self.recovery_schedule[self.recovery_stage]
            self.current_recovery_mode = new_stage.get('mode', 'cautious')

        if notes:
            self.notes += f"\n[{timezone.now().isoformat()}] Advanced to stage {self.recovery_stage}: {notes}"

        self.save()
        return True

    def record_recovery_trade(self, is_profitable: bool, pnl: float = 0):
        """Record a trade during recovery period.

        Args:
            is_profitable: Whether the trade was profitable
            pnl: P&L of the trade
        """
        self.recovery_trades_count += 1
        if is_profitable:
            self.recovery_profitable_trades += 1
        self.recovery_pnl += pnl
        self.save()

    def resolve(self, method='manual', notes=''):
        """Resolve the circuit breaker event.

        Args:
            method: Resolution method
            notes: Resolution notes
        """
        self.resolved_at = timezone.now()
        self.resolution_method = method
        self.current_recovery_mode = 'normal'

        if notes:
            self.notes += f"\n[{timezone.now().isoformat()}] Resolved ({method}): {notes}"

        self.save()

    def request_early_recovery(self, justification: str):
        """Request early recovery with justification.

        Args:
            justification: User's justification for early recovery

        Returns:
            dict with request status
        """
        self.notes += f"\n[{timezone.now().isoformat()}] Early recovery requested: {justification}"
        self.save()

        return {
            'request_submitted': True,
            'event_id': self.id,
            'current_mode': self.current_recovery_mode,
            'message': 'Early recovery request submitted for review',
        }

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'breaker_type': self.breaker_type,
            'breaker_type_display': self.get_breaker_type_display(),
            'triggered_at': self.triggered_at.isoformat(),
            'trigger_value': float(self.trigger_value),
            'trigger_threshold': float(self.trigger_threshold),
            'trigger_context': self.trigger_context,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolution_method': self.resolution_method,
            'resolution_method_display': self.get_resolution_method_display(),
            'is_active': self.is_active,
            'current_recovery_mode': self.current_recovery_mode,
            'current_recovery_mode_display': self.get_current_recovery_mode_display(),
            'recovery_mode_until': self.recovery_mode_until.isoformat() if self.recovery_mode_until else None,
            'recovery_stage': self.recovery_stage,
            'recovery_schedule': self.recovery_schedule,
            'position_size_multiplier': self.position_size_multiplier,
            'duration_hours': round(self.duration_hours, 2),
            'recovery_trades_count': self.recovery_trades_count,
            'recovery_profitable_trades': self.recovery_profitable_trades,
            'recovery_pnl': float(self.recovery_pnl),
            'recovery_win_rate': self.recovery_win_rate,
            'can_advance_recovery': self.can_advance_recovery,
            'notes': self.notes,
        }


class CircuitBreakerState(models.Model):
    """Persists current circuit breaker state for restart persistence.

    This model maintains the real-time state of each circuit breaker type,
    allowing the system to restore state after restarts and maintain
    trading restrictions across sessions.

    Features:
    - Per-user, per-breaker-type state
    - Survives application restarts
    - Daily reset capability
    - VIX tracking with levels
    - Error rate tracking
    """

    STATUS_CHOICES = [
        ('ok', 'OK'),
        ('warning', 'Warning'),
        ('triggered', 'Triggered'),
        ('critical', 'Critical'),
    ]

    VIX_LEVEL_CHOICES = [
        ('none', 'None'),
        ('elevated', 'Elevated'),
        ('extreme', 'Extreme'),
        ('critical', 'Critical'),
    ]

    # User association (null for global breakers)
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='circuit_breaker_states',
        null=True,
        blank=True,
        help_text="User for this breaker state (null for global)"
    )

    # Breaker identification
    breaker_type = models.CharField(
        max_length=50,
        db_index=True,
        help_text="Type of circuit breaker (daily_loss, vix, error_rate, stale_data, global)"
    )

    # Current state
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='ok',
        db_index=True,
        help_text="Current breaker status"
    )

    # Trigger details
    current_value = models.DecimalField(
        max_digits=14,
        decimal_places=4,
        null=True,
        blank=True,
        help_text="Current measured value (drawdown %, VIX level, error count)"
    )
    threshold = models.DecimalField(
        max_digits=14,
        decimal_places=4,
        null=True,
        blank=True,
        help_text="Threshold that triggers the breaker"
    )

    # Trip information
    tripped_at = models.DateTimeField(
        null=True,
        blank=True,
        db_index=True,
        help_text="When the breaker was tripped"
    )
    trip_reason = models.CharField(
        max_length=255,
        blank=True,
        help_text="Reason the breaker tripped"
    )
    cooldown_until = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When cooldown ends and reset is possible"
    )

    # VIX-specific tracking
    vix_level = models.CharField(
        max_length=20,
        choices=VIX_LEVEL_CHOICES,
        default='none',
        help_text="Current VIX trip level"
    )
    current_vix = models.DecimalField(
        max_digits=8,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Current VIX value"
    )
    last_vix_check = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When VIX was last checked"
    )

    # Error rate tracking
    errors_last_minute = models.IntegerField(
        default=0,
        help_text="Error count in current minute window"
    )
    error_window_start = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Start of current error window"
    )

    # Equity tracking (for drawdown calculation)
    start_of_day_equity = models.DecimalField(
        max_digits=14,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Equity at start of trading day"
    )
    current_equity = models.DecimalField(
        max_digits=14,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Current equity value"
    )
    daily_drawdown = models.DecimalField(
        max_digits=8,
        decimal_places=4,
        null=True,
        blank=True,
        help_text="Current daily drawdown as decimal (0.05 = 5%)"
    )

    # Data freshness tracking
    last_data_timestamp = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Timestamp of last fresh data"
    )

    # Daily reset tracking
    last_daily_reset = models.DateField(
        null=True,
        blank=True,
        help_text="Date of last daily reset"
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ['user', 'breaker_type']
        ordering = ['breaker_type', 'user']
        indexes = [
            models.Index(fields=['user', 'breaker_type']),
            models.Index(fields=['status', 'tripped_at']),
            models.Index(fields=['breaker_type', 'status']),
        ]

    def __str__(self):
        user_str = self.user.username if self.user else 'Global'
        return f"{self.breaker_type} ({user_str}): {self.status}"

    @property
    def is_tripped(self):
        """Check if breaker is currently tripped."""
        return self.status in ('triggered', 'critical')

    @property
    def is_in_cooldown(self):
        """Check if still in cooldown period."""
        if not self.cooldown_until:
            return False
        return timezone.now() < self.cooldown_until

    @property
    def can_trade(self):
        """Check if trading is allowed."""
        if not self.is_tripped:
            return True
        # Allow trading if cooldown has passed
        return not self.is_in_cooldown

    @property
    def cooldown_remaining_seconds(self):
        """Get remaining cooldown time in seconds."""
        if not self.cooldown_until:
            return 0
        remaining = (self.cooldown_until - timezone.now()).total_seconds()
        return max(0, remaining)

    @property
    def position_size_multiplier(self):
        """Get position size multiplier based on VIX level."""
        if self.status in ('triggered', 'critical'):
            return 0.0
        multipliers = {
            'none': 1.0,
            'elevated': 0.5,
            'extreme': 0.25,
            'critical': 0.0,
        }
        return multipliers.get(self.vix_level, 1.0)

    def trip(self, reason: str, cooldown_seconds: int = 1800):
        """Trip the circuit breaker.

        Args:
            reason: Reason for tripping
            cooldown_seconds: Cooldown period (default 30 min)
        """
        self.status = 'triggered'
        self.tripped_at = timezone.now()
        self.trip_reason = reason
        self.cooldown_until = timezone.now() + timedelta(seconds=cooldown_seconds)
        self.save()

    def reset(self, force: bool = False):
        """Reset the circuit breaker.

        Args:
            force: Force reset even if in cooldown

        Returns:
            bool: True if reset, False if still in cooldown
        """
        if not force and self.is_in_cooldown:
            return False

        self.status = 'ok'
        self.tripped_at = None
        self.trip_reason = ''
        self.cooldown_until = None
        self.errors_last_minute = 0
        self.error_window_start = None
        self.save()
        return True

    def daily_reset(self, new_equity: float = None):
        """Perform daily reset of breaker state.

        Args:
            new_equity: Starting equity for the new day
        """
        today = timezone.now().date()
        if self.last_daily_reset == today:
            return  # Already reset today

        # Reset daily counters
        self.errors_last_minute = 0
        self.error_window_start = None
        self.daily_drawdown = Decimal('0')

        if new_equity:
            self.start_of_day_equity = Decimal(str(new_equity))
            self.current_equity = Decimal(str(new_equity))

        # Don't auto-clear trips - those need explicit resolution
        self.last_daily_reset = today
        self.save()

    def update_equity(self, current_equity: float):
        """Update current equity and calculate drawdown.

        Args:
            current_equity: Current portfolio equity

        Returns:
            float: Current drawdown as decimal
        """
        self.current_equity = Decimal(str(current_equity))

        if self.start_of_day_equity and self.start_of_day_equity > 0:
            dd = 1 - (self.current_equity / self.start_of_day_equity)
            self.daily_drawdown = max(Decimal('0'), dd)
        else:
            self.daily_drawdown = Decimal('0')

        self.save()
        return float(self.daily_drawdown)

    def update_vix(self, vix_value: float):
        """Update VIX tracking.

        Args:
            vix_value: Current VIX value
        """
        self.current_vix = Decimal(str(vix_value))
        self.last_vix_check = timezone.now()

        # Determine VIX level (using default thresholds)
        if vix_value >= 45:
            self.vix_level = 'critical'
        elif vix_value >= 35:
            self.vix_level = 'extreme'
        elif vix_value >= 25:
            self.vix_level = 'elevated'
        else:
            self.vix_level = 'none'

        self.save()

    def mark_error(self, max_per_minute: int = 5):
        """Mark an error occurrence.

        Args:
            max_per_minute: Maximum errors before tripping

        Returns:
            bool: True if breaker tripped from this error
        """
        now = timezone.now()

        # Reset window if more than a minute old
        if self.error_window_start:
            if (now - self.error_window_start).total_seconds() > 60:
                self.errors_last_minute = 0
                self.error_window_start = now
        else:
            self.error_window_start = now

        self.errors_last_minute += 1
        self.save()

        if self.errors_last_minute >= max_per_minute:
            self.trip(f"error-rate-{self.errors_last_minute}")
            return True

        return False

    def mark_data_fresh(self):
        """Mark data as fresh."""
        self.last_data_timestamp = timezone.now()
        self.save(update_fields=['last_data_timestamp', 'updated_at'])

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            'breaker_type': self.breaker_type,
            'status': self.status,
            'is_tripped': self.is_tripped,
            'can_trade': self.can_trade,
            'current_value': float(self.current_value) if self.current_value else None,
            'threshold': float(self.threshold) if self.threshold else None,
            'tripped_at': self.tripped_at.isoformat() if self.tripped_at else None,
            'trip_reason': self.trip_reason,
            'cooldown_until': self.cooldown_until.isoformat() if self.cooldown_until else None,
            'cooldown_remaining_seconds': self.cooldown_remaining_seconds,
            'vix': {
                'level': self.vix_level,
                'current_value': float(self.current_vix) if self.current_vix else None,
                'last_check': self.last_vix_check.isoformat() if self.last_vix_check else None,
                'position_multiplier': self.position_size_multiplier,
            },
            'daily_drawdown': float(self.daily_drawdown) if self.daily_drawdown else None,
            'start_of_day_equity': float(self.start_of_day_equity) if self.start_of_day_equity else None,
            'current_equity': float(self.current_equity) if self.current_equity else None,
            'errors_last_minute': self.errors_last_minute,
            'last_daily_reset': self.last_daily_reset.isoformat() if self.last_daily_reset else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class CircuitBreakerHistory(models.Model):
    """Lightweight history of circuit breaker state changes.

    While CircuitBreakerEvent tracks full recovery workflows,
    this model tracks all state transitions for audit and analysis.
    """

    ACTION_CHOICES = [
        ('trip', 'Tripped'),
        ('reset', 'Reset'),
        ('warning', 'Warning'),
        ('daily_reset', 'Daily Reset'),
        ('vix_change', 'VIX Level Change'),
        ('equity_update', 'Equity Update'),
        ('error_spike', 'Error Spike'),
        ('cooldown_end', 'Cooldown Ended'),
    ]

    # Link to state
    state = models.ForeignKey(
        CircuitBreakerState,
        on_delete=models.CASCADE,
        related_name='history',
        help_text="Circuit breaker state this entry relates to"
    )

    # Action details
    action = models.CharField(
        max_length=30,
        choices=ACTION_CHOICES,
        db_index=True,
        help_text="Type of state change"
    )
    timestamp = models.DateTimeField(
        default=timezone.now,
        db_index=True,
        help_text="When this action occurred"
    )

    # State snapshot at time of action
    previous_status = models.CharField(
        max_length=20,
        blank=True,
        help_text="Status before the action"
    )
    new_status = models.CharField(
        max_length=20,
        blank=True,
        help_text="Status after the action"
    )

    # Values at time of action
    value_at_action = models.DecimalField(
        max_digits=14,
        decimal_places=4,
        null=True,
        blank=True,
        help_text="Relevant value when action occurred"
    )
    threshold_at_action = models.DecimalField(
        max_digits=14,
        decimal_places=4,
        null=True,
        blank=True,
        help_text="Threshold at time of action"
    )

    # Additional context
    reason = models.CharField(
        max_length=255,
        blank=True,
        help_text="Reason or description for the action"
    )
    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text="Additional context data"
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['state', 'timestamp']),
            models.Index(fields=['action', 'timestamp']),
        ]
        verbose_name_plural = "Circuit breaker histories"

    def __str__(self):
        return f"{self.state.breaker_type} - {self.action} @ {self.timestamp.strftime('%Y-%m-%d %H:%M')}"

    @classmethod
    def log_action(cls, state: CircuitBreakerState, action: str, reason: str = '',
                   value: float = None, threshold: float = None,
                   previous_status: str = '', new_status: str = '',
                   metadata: dict = None):
        """Create a history entry for a state change.

        Args:
            state: CircuitBreakerState instance
            action: Action type
            reason: Reason for the action
            value: Value at time of action
            threshold: Threshold at time of action
            previous_status: Status before action
            new_status: Status after action
            metadata: Additional context

        Returns:
            CircuitBreakerHistory instance
        """
        return cls.objects.create(
            state=state,
            action=action,
            reason=reason,
            value_at_action=Decimal(str(value)) if value is not None else None,
            threshold_at_action=Decimal(str(threshold)) if threshold is not None else None,
            previous_status=previous_status,
            new_status=new_status or state.status,
            metadata=metadata or {},
        )

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'breaker_type': self.state.breaker_type,
            'action': self.action,
            'action_display': self.get_action_display(),
            'timestamp': self.timestamp.isoformat(),
            'previous_status': self.previous_status,
            'new_status': self.new_status,
            'value_at_action': float(self.value_at_action) if self.value_at_action else None,
            'threshold_at_action': float(self.threshold_at_action) if self.threshold_at_action else None,
            'reason': self.reason,
            'metadata': self.metadata,
        }


class StrategyPortfolio(models.Model):
    """
    User-defined portfolio combining multiple strategies with allocations.
    Enables pre-built portfolio templates and custom strategy combinations.
    """
    RISK_PROFILE_CHOICES = [
        ('conservative', 'Conservative'),
        ('moderate', 'Moderate'),
        ('aggressive', 'Aggressive'),
        ('custom', 'Custom'),
    ]

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='strategy_portfolios',
        null=True,
        blank=True,
        help_text='User who owns this portfolio (null for system templates)'
    )
    name = models.CharField(
        max_length=100,
        help_text='User-friendly portfolio name'
    )
    description = models.TextField(
        blank=True,
        help_text='Description of portfolio strategy and goals'
    )
    risk_profile = models.CharField(
        max_length=20,
        choices=RISK_PROFILE_CHOICES,
        default='moderate',
        help_text='Risk profile classification'
    )
    is_template = models.BooleanField(
        default=False,
        help_text='True if this is a system template available to all users'
    )
    is_active = models.BooleanField(
        default=False,
        help_text='True if this portfolio is currently active for trading'
    )

    # Strategy allocations as JSONField
    # Format: {"strategy_id": {"allocation_pct": 25.0, "enabled": true, "params": {}}, ...}
    strategies = models.JSONField(
        default=dict,
        help_text='Strategy allocations and configurations'
    )

    # Portfolio metrics (calculated periodically)
    performance_metrics = models.JSONField(
        default=dict,
        blank=True,
        help_text='Combined performance metrics across strategies'
    )

    # Correlation and diversification analysis
    correlation_matrix = models.JSONField(
        default=dict,
        blank=True,
        help_text='Correlation matrix between strategies'
    )
    diversification_score = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        default=0,
        help_text='Diversification score (0-100)'
    )
    expected_sharpe = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        null=True,
        blank=True,
        help_text='Expected Sharpe ratio based on historical analysis'
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_rebalanced_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text='Last time portfolio allocations were rebalanced'
    )

    class Meta:
        verbose_name = 'Strategy Portfolio'
        verbose_name_plural = 'Strategy Portfolios'
        ordering = ['-is_active', '-updated_at']
        constraints = [
            # Each user can only have one active portfolio
            models.UniqueConstraint(
                fields=['user'],
                condition=models.Q(is_active=True),
                name='unique_active_portfolio_per_user'
            )
        ]

    def __str__(self):
        owner = self.user.username if self.user else 'System'
        active = ' (Active)' if self.is_active else ''
        return f"{self.name} - {owner}{active}"

    @property
    def total_allocation(self) -> float:
        """Calculate total allocation percentage across all strategies."""
        return sum(
            s.get('allocation_pct', 0)
            for s in self.strategies.values()
            if s.get('enabled', True)
        )

    @property
    def strategy_count(self) -> int:
        """Count enabled strategies in portfolio."""
        return sum(
            1 for s in self.strategies.values()
            if s.get('enabled', True)
        )

    @property
    def is_valid_allocation(self) -> bool:
        """Check if total allocation is valid (close to 100%)."""
        total = self.total_allocation
        return 99.0 <= total <= 101.0

    def get_strategy_allocation(self, strategy_id: str) -> dict:
        """Get allocation details for a specific strategy.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Allocation dict or empty dict if not found
        """
        return self.strategies.get(strategy_id, {})

    def set_strategy_allocation(
        self,
        strategy_id: str,
        allocation_pct: float,
        enabled: bool = True,
        params: dict = None
    ):
        """Set or update allocation for a strategy.

        Args:
            strategy_id: Strategy identifier
            allocation_pct: Percentage allocation (0-100)
            enabled: Whether strategy is enabled
            params: Optional strategy-specific parameters
        """
        self.strategies[strategy_id] = {
            'allocation_pct': allocation_pct,
            'enabled': enabled,
            'params': params or {},
            'updated_at': timezone.now().isoformat(),
        }
        self.save()

    def remove_strategy(self, strategy_id: str):
        """Remove a strategy from the portfolio.

        Args:
            strategy_id: Strategy to remove
        """
        if strategy_id in self.strategies:
            del self.strategies[strategy_id]
            self.save()

    def activate(self):
        """Activate this portfolio, deactivating any other active portfolio for user."""
        if self.user:
            # Deactivate other portfolios for this user
            StrategyPortfolio.objects.filter(
                user=self.user,
                is_active=True
            ).exclude(pk=self.pk).update(is_active=False)

        self.is_active = True
        self.save()

    def deactivate(self):
        """Deactivate this portfolio."""
        self.is_active = False
        self.save()

    def update_performance_metrics(self, metrics: dict):
        """Update portfolio performance metrics.

        Args:
            metrics: Dict with performance data
        """
        self.performance_metrics = {
            **self.performance_metrics,
            **metrics,
            'updated_at': timezone.now().isoformat(),
        }
        self.save()

    def update_correlation_analysis(
        self,
        correlation_matrix: dict,
        diversification_score: float,
        expected_sharpe: float = None
    ):
        """Update correlation and diversification analysis.

        Args:
            correlation_matrix: Strategy correlation matrix
            diversification_score: Diversification score (0-100)
            expected_sharpe: Optional expected Sharpe ratio
        """
        self.correlation_matrix = correlation_matrix
        self.diversification_score = diversification_score
        if expected_sharpe is not None:
            self.expected_sharpe = expected_sharpe
        self.save()

    def clone_for_user(self, user) -> 'StrategyPortfolio':
        """Create a copy of this portfolio for a user.

        Args:
            user: User to clone portfolio for

        Returns:
            New StrategyPortfolio instance
        """
        return StrategyPortfolio.objects.create(
            user=user,
            name=f"{self.name} (Copy)",
            description=self.description,
            risk_profile=self.risk_profile,
            is_template=False,
            is_active=False,
            strategies=self.strategies.copy(),
            performance_metrics={},
            correlation_matrix=self.correlation_matrix.copy(),
            diversification_score=self.diversification_score,
            expected_sharpe=self.expected_sharpe,
        )

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'risk_profile': self.risk_profile,
            'risk_profile_display': self.get_risk_profile_display(),
            'is_template': self.is_template,
            'is_active': self.is_active,
            'strategies': self.strategies,
            'strategy_count': self.strategy_count,
            'total_allocation': self.total_allocation,
            'is_valid_allocation': self.is_valid_allocation,
            'performance_metrics': self.performance_metrics,
            'correlation_matrix': self.correlation_matrix,
            'diversification_score': float(self.diversification_score),
            'expected_sharpe': float(self.expected_sharpe) if self.expected_sharpe else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'last_rebalanced_at': self.last_rebalanced_at.isoformat() if self.last_rebalanced_at else None,
            'user_id': self.user_id,
        }


class UserProfile(models.Model):
    """
    Comprehensive user profile for trading preferences, risk assessment, and onboarding.
    One-to-one relationship with Django User model.
    """

    # Risk Tolerance Choices
    RISK_TOLERANCE_CHOICES = [
        (1, 'Very Conservative'),
        (2, 'Conservative'),
        (3, 'Moderate'),
        (4, 'Aggressive'),
        (5, 'Very Aggressive'),
    ]

    # Investment Timeline Choices
    INVESTMENT_TIMELINE_CHOICES = [
        ('short', 'Short-term (< 1 year)'),
        ('medium', 'Medium-term (1-5 years)'),
        ('long', 'Long-term (5-10 years)'),
        ('very_long', 'Very Long-term (10+ years)'),
    ]

    # Income Source Choices
    INCOME_SOURCE_CHOICES = [
        ('employment', 'Employment'),
        ('self_employed', 'Self-Employed'),
        ('trading', 'Full-time Trading'),
        ('retirement', 'Retirement'),
        ('other', 'Other'),
    ]

    # Trading Experience Choices
    TRADING_EXPERIENCE_CHOICES = [
        ('none', 'No Experience'),
        ('beginner', 'Beginner (< 1 year)'),
        ('intermediate', 'Intermediate (1-3 years)'),
        ('advanced', 'Advanced (3-5 years)'),
        ('expert', 'Expert (5+ years)'),
    ]

    # Email Frequency Choices
    EMAIL_FREQUENCY_CHOICES = [
        ('realtime', 'Real-time'),
        ('hourly', 'Hourly'),
        ('daily', 'Daily Digest'),
        ('weekly', 'Weekly'),
        ('none', 'None'),
    ]

    # Chart Timeframe Choices
    CHART_TIMEFRAME_CHOICES = [
        ('1D', '1 Day'),
        ('1W', '1 Week'),
        ('1M', '1 Month'),
        ('3M', '3 Months'),
        ('6M', '6 Months'),
        ('1Y', '1 Year'),
        ('YTD', 'Year to Date'),
        ('ALL', 'All Time'),
    ]

    # Core relationship
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='profile',
        help_text='The Django user this profile belongs to'
    )

    # =========================================================================
    # Risk Assessment
    # =========================================================================
    risk_tolerance = models.IntegerField(
        choices=RISK_TOLERANCE_CHOICES,
        null=True,
        blank=True,
        help_text='User risk tolerance level (1-5)'
    )
    risk_score = models.IntegerField(
        null=True,
        blank=True,
        help_text='Calculated risk score from questionnaire (1-100)'
    )
    investment_timeline = models.CharField(
        max_length=20,
        choices=INVESTMENT_TIMELINE_CHOICES,
        null=True,
        blank=True,
        help_text='Investment time horizon'
    )
    max_loss_tolerance = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        null=True,
        blank=True,
        help_text='Maximum acceptable loss percentage (e.g., 10.00 for 10%)'
    )
    risk_assessment_answers = models.JSONField(
        default=dict,
        blank=True,
        help_text='Stored answers from risk questionnaire'
    )

    # =========================================================================
    # Financial Context
    # =========================================================================
    investable_capital = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        null=True,
        blank=True,
        help_text='Total investable capital'
    )
    income_source = models.CharField(
        max_length=20,
        choices=INCOME_SOURCE_CHOICES,
        null=True,
        blank=True,
        help_text='Primary income source'
    )
    annual_income = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        null=True,
        blank=True,
        help_text='Annual income for suitability'
    )
    capital_is_risk_capital = models.BooleanField(
        null=True,
        help_text='Whether user can afford to lose this capital'
    )
    net_worth = models.DecimalField(
        max_digits=14,
        decimal_places=2,
        null=True,
        blank=True,
        help_text='Total net worth for suitability'
    )

    # =========================================================================
    # Experience
    # =========================================================================
    trading_experience = models.CharField(
        max_length=20,
        choices=TRADING_EXPERIENCE_CHOICES,
        null=True,
        blank=True,
        help_text='Level of trading experience'
    )
    options_experience = models.BooleanField(
        default=False,
        help_text='Has options trading experience'
    )
    options_level = models.IntegerField(
        null=True,
        blank=True,
        help_text='Options trading level (1-4)'
    )
    crypto_experience = models.BooleanField(
        default=False,
        help_text='Has cryptocurrency trading experience'
    )
    margin_experience = models.BooleanField(
        default=False,
        help_text='Has margin trading experience'
    )
    shorting_experience = models.BooleanField(
        default=False,
        help_text='Has short selling experience'
    )

    # =========================================================================
    # Onboarding Progress
    # =========================================================================
    onboarding_completed = models.BooleanField(
        default=False,
        help_text='Whether onboarding is complete'
    )
    onboarding_step = models.IntegerField(
        default=1,
        help_text='Current onboarding step (1-5)'
    )
    onboarding_started_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text='When user started onboarding'
    )
    onboarding_completed_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text='When user completed onboarding'
    )
    risk_assessment_completed = models.BooleanField(
        default=False,
        help_text='Risk assessment questionnaire completed'
    )
    brokerage_connected = models.BooleanField(
        default=False,
        help_text='Brokerage account connected'
    )
    first_strategy_activated = models.BooleanField(
        default=False,
        help_text='At least one strategy has been activated'
    )
    first_trade_executed = models.BooleanField(
        default=False,
        help_text='First trade has been executed'
    )

    # =========================================================================
    # Trading Mode
    # =========================================================================
    is_paper_trading = models.BooleanField(
        default=True,
        help_text='Whether user is in paper trading mode'
    )
    paper_trading_start = models.DateTimeField(
        null=True,
        blank=True,
        help_text='When paper trading started'
    )
    live_trading_approved = models.BooleanField(
        default=False,
        help_text='Whether user has been approved for live trading'
    )
    live_trading_approved_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text='When live trading was approved'
    )

    # =========================================================================
    # Preferences
    # =========================================================================
    dashboard_layout = models.JSONField(
        default=dict,
        blank=True,
        help_text='Custom dashboard widget layout preferences'
    )
    default_chart_timeframe = models.CharField(
        max_length=10,
        choices=CHART_TIMEFRAME_CHOICES,
        default='1M',
        help_text='Default chart timeframe'
    )
    timezone = models.CharField(
        max_length=50,
        default='America/New_York',
        help_text='User timezone for display'
    )
    theme = models.CharField(
        max_length=20,
        default='light',
        help_text='UI theme preference'
    )
    compact_mode = models.BooleanField(
        default=False,
        help_text='Use compact UI mode'
    )
    show_tutorial_hints = models.BooleanField(
        default=True,
        help_text='Show tutorial hints in UI'
    )

    # =========================================================================
    # Communication Preferences
    # =========================================================================
    email_frequency = models.CharField(
        max_length=20,
        choices=EMAIL_FREQUENCY_CHOICES,
        default='realtime',
        help_text='Email notification frequency'
    )
    email_trade_alerts = models.BooleanField(
        default=True,
        help_text='Receive trade execution alerts'
    )
    email_risk_alerts = models.BooleanField(
        default=True,
        help_text='Receive risk/circuit breaker alerts'
    )
    email_performance_reports = models.BooleanField(
        default=True,
        help_text='Receive periodic performance reports'
    )
    push_notifications_enabled = models.BooleanField(
        default=False,
        help_text='Enable push notifications'
    )
    sms_alerts_enabled = models.BooleanField(
        default=False,
        help_text='Enable SMS alerts'
    )
    phone_number = models.CharField(
        max_length=20,
        null=True,
        blank=True,
        help_text='Phone number for SMS alerts'
    )

    # =========================================================================
    # Timestamps
    # =========================================================================
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text='When profile was created'
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text='When profile was last updated'
    )
    last_active_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text='Last user activity timestamp'
    )
    last_login_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text='Last login timestamp'
    )

    class Meta:
        verbose_name = 'User Profile'
        verbose_name_plural = 'User Profiles'
        ordering = ['-created_at']

    def __str__(self):
        return f"Profile: {self.user.username}"

    @property
    def onboarding_progress(self) -> dict:
        """Calculate onboarding progress percentage and status."""
        steps = {
            'risk_assessment': self.risk_assessment_completed,
            'brokerage_connected': self.brokerage_connected,
            'first_strategy': self.first_strategy_activated,
            'first_trade': self.first_trade_executed,
        }
        completed = sum(1 for v in steps.values() if v)
        total = len(steps)
        return {
            'completed_steps': completed,
            'total_steps': total,
            'percentage': int((completed / total) * 100),
            'steps': steps,
            'is_complete': self.onboarding_completed,
        }

    @property
    def risk_profile_name(self) -> str:
        """Get human-readable risk profile name."""
        if self.risk_tolerance:
            return dict(self.RISK_TOLERANCE_CHOICES).get(self.risk_tolerance, 'Unknown')
        return 'Not Assessed'

    @property
    def trading_mode(self) -> str:
        """Get current trading mode."""
        return 'paper' if self.is_paper_trading else 'live'

    def update_last_active(self):
        """Update last active timestamp."""
        self.last_active_at = timezone.now()
        self.save(update_fields=['last_active_at'])

    def complete_onboarding_step(self, step: str):
        """Mark an onboarding step as complete.

        Args:
            step: Step name ('risk_assessment', 'brokerage', 'strategy', 'trade')
        """
        step_map = {
            'risk_assessment': ('risk_assessment_completed', 2),
            'brokerage': ('brokerage_connected', 3),
            'strategy': ('first_strategy_activated', 4),
            'trade': ('first_trade_executed', 5),
        }

        if step in step_map:
            field, next_step = step_map[step]
            setattr(self, field, True)
            if self.onboarding_step < next_step:
                self.onboarding_step = next_step

            # Check if all steps complete
            if all([
                self.risk_assessment_completed,
                self.brokerage_connected,
                self.first_strategy_activated,
            ]):
                self.onboarding_completed = True
                self.onboarding_completed_at = timezone.now()

            self.save()

    def calculate_risk_score(self, answers: dict) -> int:
        """Calculate risk score from questionnaire answers.

        Args:
            answers: Dictionary of question_id -> answer_value

        Returns:
            Risk score (1-100)
        """
        # Store answers
        self.risk_assessment_answers = answers

        # Simple scoring algorithm - weight answers
        score = 0
        max_score = 0

        weights = {
            'timeline': 20,
            'loss_reaction': 25,
            'experience': 15,
            'income_stability': 15,
            'capital_importance': 25,
        }

        for question, weight in weights.items():
            if question in answers:
                # Assume answers are 1-5 scale
                answer_value = answers.get(question, 3)
                score += (answer_value / 5) * weight
                max_score += weight

        # Normalize to 1-100
        if max_score > 0:
            normalized_score = int((score / max_score) * 100)
        else:
            normalized_score = 50  # Default moderate

        self.risk_score = normalized_score

        # Map score to risk tolerance
        if normalized_score <= 20:
            self.risk_tolerance = 1
        elif normalized_score <= 40:
            self.risk_tolerance = 2
        elif normalized_score <= 60:
            self.risk_tolerance = 3
        elif normalized_score <= 80:
            self.risk_tolerance = 4
        else:
            self.risk_tolerance = 5

        self.risk_assessment_completed = True
        self.save()

        return normalized_score

    def get_recommended_settings(self) -> dict:
        """Get recommended system settings based on profile.

        Returns:
            Dictionary of recommended settings
        """
        recommendations = {
            'max_position_size_pct': 5.0,
            'max_portfolio_risk_pct': 20.0,
            'recommended_strategies': [],
            'avoid_strategies': [],
        }

        # Adjust based on risk tolerance
        if self.risk_tolerance == 1:  # Very Conservative
            recommendations['max_position_size_pct'] = 2.0
            recommendations['max_portfolio_risk_pct'] = 10.0
            recommendations['recommended_strategies'] = ['wheel', 'spx-credit-spreads']
            recommendations['avoid_strategies'] = ['lotto-scanner', 'momentum-weeklies']
        elif self.risk_tolerance == 2:  # Conservative
            recommendations['max_position_size_pct'] = 3.0
            recommendations['max_portfolio_risk_pct'] = 15.0
            recommendations['recommended_strategies'] = ['wheel', 'leaps-tracker']
            recommendations['avoid_strategies'] = ['lotto-scanner']
        elif self.risk_tolerance == 3:  # Moderate
            recommendations['max_position_size_pct'] = 5.0
            recommendations['max_portfolio_risk_pct'] = 20.0
            recommendations['recommended_strategies'] = ['swing-trading', 'debit-spreads']
        elif self.risk_tolerance == 4:  # Aggressive
            recommendations['max_position_size_pct'] = 7.0
            recommendations['max_portfolio_risk_pct'] = 30.0
            recommendations['recommended_strategies'] = ['wsb-dip-bot', 'momentum-weeklies']
        else:  # Very Aggressive
            recommendations['max_position_size_pct'] = 10.0
            recommendations['max_portfolio_risk_pct'] = 40.0
            recommendations['recommended_strategies'] = ['wsb-dip-bot', 'momentum-weeklies', 'lotto-scanner']

        # Adjust for experience
        if self.trading_experience in ['none', 'beginner']:
            recommendations['max_position_size_pct'] = min(
                recommendations['max_position_size_pct'],
                3.0
            )
            recommendations['recommended_strategies'] = ['wheel', 'index-baseline']

        # Adjust for options experience
        if not self.options_experience:
            recommendations['avoid_strategies'].extend([
                'momentum-weeklies', 'earnings-protection', 'exotic-spreads'
            ])

        return recommendations

    def to_dict(self) -> dict:
        """Convert profile to dictionary for API responses."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'username': self.user.username,
            'email': self.user.email,

            # Risk Assessment
            'risk_tolerance': self.risk_tolerance,
            'risk_tolerance_display': self.risk_profile_name,
            'risk_score': self.risk_score,
            'investment_timeline': self.investment_timeline,
            'max_loss_tolerance': float(self.max_loss_tolerance) if self.max_loss_tolerance else None,

            # Financial Context
            'investable_capital': float(self.investable_capital) if self.investable_capital else None,
            'income_source': self.income_source,
            'capital_is_risk_capital': self.capital_is_risk_capital,

            # Experience
            'trading_experience': self.trading_experience,
            'options_experience': self.options_experience,
            'options_level': self.options_level,
            'crypto_experience': self.crypto_experience,
            'margin_experience': self.margin_experience,
            'shorting_experience': self.shorting_experience,

            # Onboarding
            'onboarding_completed': self.onboarding_completed,
            'onboarding_step': self.onboarding_step,
            'onboarding_progress': self.onboarding_progress,
            'risk_assessment_completed': self.risk_assessment_completed,
            'brokerage_connected': self.brokerage_connected,
            'first_strategy_activated': self.first_strategy_activated,
            'first_trade_executed': self.first_trade_executed,

            # Trading Mode
            'is_paper_trading': self.is_paper_trading,
            'trading_mode': self.trading_mode,
            'live_trading_approved': self.live_trading_approved,

            # Preferences
            'dashboard_layout': self.dashboard_layout,
            'default_chart_timeframe': self.default_chart_timeframe,
            'timezone': self.timezone,
            'theme': self.theme,
            'compact_mode': self.compact_mode,
            'show_tutorial_hints': self.show_tutorial_hints,

            # Communication
            'email_frequency': self.email_frequency,
            'email_trade_alerts': self.email_trade_alerts,
            'email_risk_alerts': self.email_risk_alerts,
            'email_performance_reports': self.email_performance_reports,
            'push_notifications_enabled': self.push_notifications_enabled,
            'sms_alerts_enabled': self.sms_alerts_enabled,

            # Timestamps
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'last_active_at': self.last_active_at.isoformat() if self.last_active_at else None,
        }


class DigestLog(models.Model):
    """
    Track sent digest emails for auditing and preventing duplicates.
    Records what data was included in each digest for transparency.
    """

    # Digest Type Choices
    DIGEST_TYPE_CHOICES = [
        ('daily', 'Daily Digest'),
        ('weekly', 'Weekly Digest'),
    ]

    # Delivery Status Choices
    DELIVERY_STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('sent', 'Sent'),
        ('failed', 'Failed'),
        ('bounced', 'Bounced'),
    ]

    # Core relationship
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='digest_logs',
        help_text='The user this digest was sent to'
    )

    # Digest type and timing
    digest_type = models.CharField(
        max_length=20,
        choices=DIGEST_TYPE_CHOICES,
        help_text='Type of digest (daily or weekly)'
    )
    period_start = models.DateTimeField(
        help_text='Start of the period covered by this digest'
    )
    period_end = models.DateTimeField(
        help_text='End of the period covered by this digest'
    )

    # Delivery tracking
    scheduled_at = models.DateTimeField(
        auto_now_add=True,
        help_text='When the digest was scheduled/created'
    )
    sent_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text='When the digest was actually sent'
    )
    delivery_status = models.CharField(
        max_length=20,
        choices=DELIVERY_STATUS_CHOICES,
        default='pending',
        help_text='Current delivery status'
    )
    error_message = models.TextField(
        blank=True,
        default='',
        help_text='Error message if delivery failed'
    )

    # Engagement tracking
    opened_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text='When the user opened the email (via tracking pixel)'
    )
    clicked_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text='When the user clicked a link in the email'
    )

    # Data snapshot - what was included in the digest
    data_snapshot = models.JSONField(
        default=dict,
        help_text="""Snapshot of data included in the digest:
        {
            "summary": {
                "total_trades": 15,
                "winning_trades": 10,
                "losing_trades": 5,
                "total_pnl": 1250.50,
                "win_rate": 66.67
            },
            "trades": [
                {"symbol": "AAPL", "action": "BUY", "quantity": 10, "pnl": 150.00}
            ],
            "alerts": [
                {"type": "risk", "message": "Portfolio drawdown exceeded 5%"}
            ],
            "positions": {
                "open_count": 5,
                "total_value": 25000.00
            },
            "performance": {
                "period_return": 2.5,
                "benchmark_return": 1.8,
                "alpha": 0.7
            }
        }"""
    )

    # Email metadata
    email_subject = models.CharField(
        max_length=255,
        blank=True,
        default='',
        help_text='Subject line of the sent email'
    )
    email_recipient = models.EmailField(
        blank=True,
        default='',
        help_text='Email address the digest was sent to'
    )

    class Meta:
        ordering = ['-scheduled_at']
        indexes = [
            models.Index(fields=['user', 'digest_type', 'period_start']),
            models.Index(fields=['delivery_status', 'scheduled_at']),
            models.Index(fields=['digest_type', 'sent_at']),
        ]
        # Prevent duplicate digests for same user/type/period
        constraints = [
            models.UniqueConstraint(
                fields=['user', 'digest_type', 'period_start'],
                name='unique_user_digest_period'
            )
        ]

    def __str__(self):
        return f"{self.digest_type.title()} Digest for {self.user.username} ({self.period_start.date()})"

    def mark_sent(self):
        """Mark the digest as successfully sent."""
        self.delivery_status = 'sent'
        self.sent_at = timezone.now()
        self.save(update_fields=['delivery_status', 'sent_at'])

    def mark_failed(self, error_message: str):
        """Mark the digest as failed with an error message."""
        self.delivery_status = 'failed'
        self.error_message = error_message
        self.save(update_fields=['delivery_status', 'error_message'])

    def mark_opened(self):
        """Record when user opened the email."""
        if not self.opened_at:
            self.opened_at = timezone.now()
            self.save(update_fields=['opened_at'])

    def mark_clicked(self):
        """Record when user clicked a link."""
        if not self.clicked_at:
            self.clicked_at = timezone.now()
            self.save(update_fields=['clicked_at'])

    def get_summary(self) -> dict:
        """Get summary metrics from the data snapshot."""
        return self.data_snapshot.get('summary', {})

    def get_trades(self) -> list:
        """Get trades from the data snapshot."""
        return self.data_snapshot.get('trades', [])

    def get_alerts(self) -> list:
        """Get alerts from the data snapshot."""
        return self.data_snapshot.get('alerts', [])

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'digest_type': self.digest_type,
            'period_start': self.period_start.isoformat() if self.period_start else None,
            'period_end': self.period_end.isoformat() if self.period_end else None,
            'scheduled_at': self.scheduled_at.isoformat() if self.scheduled_at else None,
            'sent_at': self.sent_at.isoformat() if self.sent_at else None,
            'delivery_status': self.delivery_status,
            'error_message': self.error_message,
            'opened_at': self.opened_at.isoformat() if self.opened_at else None,
            'clicked_at': self.clicked_at.isoformat() if self.clicked_at else None,
            'email_subject': self.email_subject,
            'email_recipient': self.email_recipient,
            'summary': self.get_summary(),
            'trade_count': len(self.get_trades()),
            'alert_count': len(self.get_alerts()),
        }


class TaxLot(models.Model):
    """
    Track individual tax lots for cost basis and wash sale compliance.
    Each purchase creates a new lot; partial sales reduce remaining_quantity.

    DISCLAIMER: This is for informational purposes only and does not constitute
    tax advice. Users should consult qualified tax professionals.
    """

    # Acquisition Type Choices
    ACQUISITION_TYPE_CHOICES = [
        ('purchase', 'Purchase'),
        ('dividend_reinvest', 'Dividend Reinvestment'),
        ('transfer_in', 'Transfer In'),
        ('gift', 'Gift'),
        ('inheritance', 'Inheritance'),
        ('exercise', 'Option Exercise'),
        ('rsu_vest', 'RSU Vest'),
        ('split', 'Stock Split'),
    ]

    # Lot Selection Method Choices (for reference)
    LOT_SELECTION_CHOICES = [
        ('fifo', 'First In, First Out'),
        ('lifo', 'Last In, First Out'),
        ('hifo', 'Highest In, First Out'),
        ('lifo_gain', 'Lowest Gain First'),
        ('specific', 'Specific Lot'),
    ]

    # Core relationship
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='tax_lots',
        help_text='Owner of this tax lot'
    )

    # Lot identification
    symbol = models.CharField(
        max_length=20,
        db_index=True,
        help_text='Trading symbol'
    )
    order_id = models.CharField(
        max_length=100,
        null=True,
        blank=True,
        help_text='Original order ID that created this lot'
    )

    # Quantity tracking
    original_quantity = models.DecimalField(
        max_digits=18,
        decimal_places=6,
        help_text='Original quantity purchased'
    )
    remaining_quantity = models.DecimalField(
        max_digits=18,
        decimal_places=6,
        help_text='Remaining quantity after partial sales'
    )

    # Cost basis
    cost_basis_per_share = models.DecimalField(
        max_digits=18,
        decimal_places=6,
        help_text='Cost basis per share'
    )
    total_cost_basis = models.DecimalField(
        max_digits=18,
        decimal_places=6,
        help_text='Total cost basis for this lot'
    )

    # Acquisition details
    acquired_at = models.DateTimeField(
        help_text='Date and time lot was acquired'
    )
    acquisition_type = models.CharField(
        max_length=30,
        choices=ACQUISITION_TYPE_CHOICES,
        default='purchase',
        help_text='How the shares were acquired'
    )

    # Tax status
    is_long_term = models.BooleanField(
        default=False,
        help_text='Whether held for more than 1 year (computed)'
    )
    days_held = models.IntegerField(
        default=0,
        help_text='Number of days held (computed)'
    )

    # Wash sale tracking
    wash_sale_adjustment = models.DecimalField(
        max_digits=18,
        decimal_places=6,
        default=0,
        help_text='Disallowed loss added to cost basis due to wash sale'
    )
    is_wash_sale_replacement = models.BooleanField(
        default=False,
        help_text='Whether this lot is a wash sale replacement purchase'
    )
    wash_sale_disallowed_loss = models.DecimalField(
        max_digits=18,
        decimal_places=6,
        default=0,
        help_text='Total disallowed loss from wash sales on this lot'
    )

    # Current market data (updated periodically)
    current_price = models.DecimalField(
        max_digits=18,
        decimal_places=6,
        null=True,
        blank=True,
        help_text='Current market price'
    )
    unrealized_gain = models.DecimalField(
        max_digits=18,
        decimal_places=6,
        null=True,
        blank=True,
        help_text='Unrealized gain/loss (can be negative)'
    )
    unrealized_gain_pct = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        null=True,
        blank=True,
        help_text='Unrealized gain/loss percentage'
    )
    market_value = models.DecimalField(
        max_digits=18,
        decimal_places=6,
        null=True,
        blank=True,
        help_text='Current market value of remaining shares'
    )
    price_updated_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text='When current_price was last updated'
    )

    # Status
    is_closed = models.BooleanField(
        default=False,
        help_text='Whether all shares have been sold'
    )
    closed_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text='When the lot was fully closed'
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['symbol', 'acquired_at']
        indexes = [
            models.Index(fields=['symbol', 'acquired_at']),
            models.Index(fields=['user', 'symbol']),
            models.Index(fields=['user', 'is_closed']),
            models.Index(fields=['acquired_at']),
        ]

    def __str__(self):
        return f"{self.symbol} - {self.remaining_quantity:.2f} shares @ ${self.cost_basis_per_share:.2f}"

    def save(self, *args, **kwargs):
        """Update computed fields before saving."""
        # Calculate days held and long-term status
        if self.acquired_at:
            self.days_held = (timezone.now() - self.acquired_at).days
            self.is_long_term = self.days_held > 365

        # Check if closed
        if self.remaining_quantity <= 0:
            self.is_closed = True
            if not self.closed_at:
                self.closed_at = timezone.now()

        super().save(*args, **kwargs)

    def update_market_data(self, current_price: float):
        """Update market data and unrealized gain calculations."""
        from decimal import Decimal

        self.current_price = Decimal(str(current_price))
        self.market_value = self.remaining_quantity * self.current_price

        # Adjusted cost basis includes wash sale adjustments
        adjusted_cost = (
            self.cost_basis_per_share * self.remaining_quantity
        ) + self.wash_sale_adjustment

        self.unrealized_gain = self.market_value - adjusted_cost

        if adjusted_cost > 0:
            self.unrealized_gain_pct = (
                (self.unrealized_gain / adjusted_cost) * 100
            )
        else:
            self.unrealized_gain_pct = Decimal('0')

        self.price_updated_at = timezone.now()
        self.save(update_fields=[
            'current_price', 'market_value', 'unrealized_gain',
            'unrealized_gain_pct', 'price_updated_at'
        ])

    def get_adjusted_cost_basis(self) -> 'Decimal':
        """Get cost basis adjusted for wash sales."""
        return (
            self.cost_basis_per_share * self.remaining_quantity
        ) + self.wash_sale_adjustment

    def get_holding_period_status(self) -> str:
        """Get human-readable holding period status."""
        if self.is_long_term:
            return f"Long-term ({self.days_held} days)"
        else:
            days_until_long = 366 - self.days_held
            return f"Short-term ({self.days_held} days, {days_until_long} until long-term)"

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'symbol': self.symbol,
            'order_id': self.order_id,
            'original_quantity': float(self.original_quantity),
            'remaining_quantity': float(self.remaining_quantity),
            'cost_basis_per_share': float(self.cost_basis_per_share),
            'total_cost_basis': float(self.total_cost_basis),
            'acquired_at': self.acquired_at.isoformat() if self.acquired_at else None,
            'acquisition_type': self.acquisition_type,
            'is_long_term': self.is_long_term,
            'days_held': self.days_held,
            'holding_period_status': self.get_holding_period_status(),
            'wash_sale_adjustment': float(self.wash_sale_adjustment),
            'is_wash_sale_replacement': self.is_wash_sale_replacement,
            'current_price': float(self.current_price) if self.current_price else None,
            'unrealized_gain': float(self.unrealized_gain) if self.unrealized_gain else None,
            'unrealized_gain_pct': float(self.unrealized_gain_pct) if self.unrealized_gain_pct else None,
            'market_value': float(self.market_value) if self.market_value else None,
            'price_updated_at': self.price_updated_at.isoformat() if self.price_updated_at else None,
            'is_closed': self.is_closed,
            'closed_at': self.closed_at.isoformat() if self.closed_at else None,
        }


class TaxLotSale(models.Model):
    """
    Record of a sale against a specific tax lot.
    Used for tracking realized gains/losses and wash sale compliance.
    """

    # Relationships
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='tax_lot_sales',
        help_text='User who made the sale'
    )
    tax_lot = models.ForeignKey(
        TaxLot,
        on_delete=models.CASCADE,
        related_name='sales',
        help_text='The tax lot being sold from'
    )

    # Sale details
    symbol = models.CharField(
        max_length=20,
        db_index=True,
        help_text='Trading symbol'
    )
    quantity_sold = models.DecimalField(
        max_digits=18,
        decimal_places=6,
        help_text='Number of shares sold from this lot'
    )
    sale_price = models.DecimalField(
        max_digits=18,
        decimal_places=6,
        help_text='Price per share at sale'
    )
    proceeds = models.DecimalField(
        max_digits=18,
        decimal_places=6,
        help_text='Total sale proceeds'
    )
    sold_at = models.DateTimeField(
        help_text='Date and time of sale'
    )
    order_id = models.CharField(
        max_length=100,
        null=True,
        blank=True,
        help_text='Sell order ID'
    )

    # Cost basis at sale
    cost_basis_sold = models.DecimalField(
        max_digits=18,
        decimal_places=6,
        help_text='Cost basis of shares sold'
    )

    # Realized gain/loss
    realized_gain = models.DecimalField(
        max_digits=18,
        decimal_places=6,
        help_text='Realized gain/loss (proceeds - cost basis)'
    )
    is_gain = models.BooleanField(
        help_text='True if gain, False if loss'
    )
    is_long_term = models.BooleanField(
        help_text='Whether this was a long-term holding'
    )

    # Wash sale tracking
    is_wash_sale = models.BooleanField(
        default=False,
        help_text='Whether this sale triggered a wash sale'
    )
    wash_sale_disallowed = models.DecimalField(
        max_digits=18,
        decimal_places=6,
        default=0,
        help_text='Amount of loss disallowed due to wash sale'
    )
    wash_sale_replacement_lot_id = models.IntegerField(
        null=True,
        blank=True,
        help_text='ID of the replacement lot if wash sale'
    )

    # Lot selection method used
    lot_selection_method = models.CharField(
        max_length=20,
        default='fifo',
        help_text='Method used to select this lot'
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-sold_at']
        indexes = [
            models.Index(fields=['symbol', 'sold_at']),
            models.Index(fields=['user', 'sold_at']),
            models.Index(fields=['is_wash_sale']),
        ]

    def __str__(self):
        gain_str = "gain" if self.is_gain else "loss"
        return f"{self.symbol} - Sold {self.quantity_sold:.2f} @ ${self.sale_price:.2f} ({gain_str}: ${abs(self.realized_gain):.2f})"

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'tax_lot_id': self.tax_lot_id,
            'symbol': self.symbol,
            'quantity_sold': float(self.quantity_sold),
            'sale_price': float(self.sale_price),
            'proceeds': float(self.proceeds),
            'sold_at': self.sold_at.isoformat() if self.sold_at else None,
            'order_id': self.order_id,
            'cost_basis_sold': float(self.cost_basis_sold),
            'realized_gain': float(self.realized_gain),
            'is_gain': self.is_gain,
            'is_long_term': self.is_long_term,
            'is_wash_sale': self.is_wash_sale,
            'wash_sale_disallowed': float(self.wash_sale_disallowed),
            'lot_selection_method': self.lot_selection_method,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class StrategyPerformanceSnapshot(models.Model):
    """
    Daily/weekly/monthly performance snapshot for strategy leaderboard.
    Captures key performance metrics at regular intervals for ranking and comparison.
    """

    PERIOD_CHOICES = [
        ('daily', 'Daily'),
        ('weekly', 'Weekly'),
        ('monthly', 'Monthly'),
    ]

    # Identification
    strategy_name = models.CharField(
        max_length=100,
        db_index=True,
        help_text='Name of the strategy'
    )
    snapshot_date = models.DateField(
        db_index=True,
        help_text='Date of the snapshot'
    )
    period = models.CharField(
        max_length=10,
        choices=PERIOD_CHOICES,
        default='daily',
        help_text='Snapshot period type'
    )

    # Performance metrics
    total_return_pct = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        default=0,
        help_text='Total return percentage for the period'
    )
    sharpe_ratio = models.DecimalField(
        max_digits=8,
        decimal_places=4,
        default=0,
        help_text='Sharpe ratio (risk-adjusted return)'
    )
    sortino_ratio = models.DecimalField(
        max_digits=8,
        decimal_places=4,
        null=True,
        blank=True,
        help_text='Sortino ratio (downside risk-adjusted return)'
    )
    max_drawdown_pct = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        default=0,
        help_text='Maximum drawdown percentage'
    )
    win_rate = models.DecimalField(
        max_digits=6,
        decimal_places=2,
        default=0,
        help_text='Win rate percentage'
    )
    profit_factor = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        default=0,
        help_text='Gross profit / gross loss ratio'
    )

    # Volume metrics
    trades_count = models.IntegerField(
        default=0,
        help_text='Number of trades in the period'
    )
    winning_trades = models.IntegerField(
        default=0,
        help_text='Number of winning trades'
    )
    losing_trades = models.IntegerField(
        default=0,
        help_text='Number of losing trades'
    )
    avg_hold_duration_hours = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True,
        help_text='Average holding duration in hours'
    )

    # Risk metrics
    volatility = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        null=True,
        blank=True,
        help_text='Annualized volatility'
    )
    var_95 = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        null=True,
        blank=True,
        help_text='Value at Risk (95% confidence)'
    )
    calmar_ratio = models.DecimalField(
        max_digits=8,
        decimal_places=4,
        null=True,
        blank=True,
        help_text='Calmar ratio (return / max drawdown)'
    )

    # Benchmark comparison
    benchmark_return_pct = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        default=0,
        help_text='Benchmark (SPY) return for same period'
    )
    vs_spy_return = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        default=0,
        help_text='Excess return vs SPY'
    )
    beta = models.DecimalField(
        max_digits=8,
        decimal_places=4,
        null=True,
        blank=True,
        help_text='Beta coefficient vs market'
    )
    alpha = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        null=True,
        blank=True,
        help_text="Jensen's alpha"
    )
    correlation_spy = models.DecimalField(
        max_digits=6,
        decimal_places=4,
        null=True,
        blank=True,
        help_text='Correlation with SPY'
    )

    # P&L metrics
    total_pnl = models.DecimalField(
        max_digits=18,
        decimal_places=2,
        default=0,
        help_text='Total profit/loss amount'
    )
    avg_win = models.DecimalField(
        max_digits=18,
        decimal_places=2,
        default=0,
        help_text='Average winning trade amount'
    )
    avg_loss = models.DecimalField(
        max_digits=18,
        decimal_places=2,
        default=0,
        help_text='Average losing trade amount'
    )

    # Ranking (calculated)
    rank_by_sharpe = models.IntegerField(
        null=True,
        blank=True,
        help_text='Rank by Sharpe ratio for this date/period'
    )
    rank_by_return = models.IntegerField(
        null=True,
        blank=True,
        help_text='Rank by total return for this date/period'
    )
    rank_by_risk_adjusted = models.IntegerField(
        null=True,
        blank=True,
        help_text='Rank by risk-adjusted score (composite)'
    )

    # Metadata
    equity_curve = models.JSONField(
        default=list,
        blank=True,
        help_text='Daily equity values for the period'
    )
    monthly_returns = models.JSONField(
        default=dict,
        blank=True,
        help_text='Monthly return breakdown'
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-snapshot_date', 'strategy_name']
        unique_together = ['strategy_name', 'snapshot_date', 'period']
        indexes = [
            models.Index(fields=['snapshot_date', 'period']),
            models.Index(fields=['strategy_name', 'period']),
            models.Index(fields=['sharpe_ratio']),
            models.Index(fields=['total_return_pct']),
        ]

    def __str__(self):
        return f"{self.strategy_name} - {self.snapshot_date} ({self.period})"

    def get_risk_adjusted_score(self) -> float:
        """Calculate composite risk-adjusted score."""
        from decimal import Decimal

        # Weighted composite: 40% Sharpe, 30% Return, 30% (1 - DD)
        sharpe_component = float(self.sharpe_ratio or 0) * 0.4
        return_component = float(self.total_return_pct or 0) * 0.003  # Normalize
        dd_component = (1 - abs(float(self.max_drawdown_pct or 0) / 100)) * 0.3

        return sharpe_component + return_component + dd_component

    def get_trend_vs_previous(self, metric: str = 'sharpe_ratio') -> str:
        """Get trend direction compared to previous snapshot."""
        previous = StrategyPerformanceSnapshot.objects.filter(
            strategy_name=self.strategy_name,
            period=self.period,
            snapshot_date__lt=self.snapshot_date
        ).order_by('-snapshot_date').first()

        if not previous:
            return 'new'

        current_val = getattr(self, metric, 0) or 0
        previous_val = getattr(previous, metric, 0) or 0

        if current_val > previous_val * 1.05:
            return 'up'
        elif current_val < previous_val * 0.95:
            return 'down'
        else:
            return 'flat'

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'strategy_name': self.strategy_name,
            'snapshot_date': self.snapshot_date.isoformat() if self.snapshot_date else None,
            'period': self.period,
            'total_return_pct': float(self.total_return_pct or 0),
            'sharpe_ratio': float(self.sharpe_ratio or 0),
            'sortino_ratio': float(self.sortino_ratio) if self.sortino_ratio else None,
            'max_drawdown_pct': float(self.max_drawdown_pct or 0),
            'win_rate': float(self.win_rate or 0),
            'profit_factor': float(self.profit_factor or 0),
            'trades_count': self.trades_count,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'avg_hold_duration_hours': float(self.avg_hold_duration_hours) if self.avg_hold_duration_hours else None,
            'volatility': float(self.volatility) if self.volatility else None,
            'var_95': float(self.var_95) if self.var_95 else None,
            'calmar_ratio': float(self.calmar_ratio) if self.calmar_ratio else None,
            'benchmark_return_pct': float(self.benchmark_return_pct or 0),
            'vs_spy_return': float(self.vs_spy_return or 0),
            'beta': float(self.beta) if self.beta else None,
            'alpha': float(self.alpha) if self.alpha else None,
            'correlation_spy': float(self.correlation_spy) if self.correlation_spy else None,
            'total_pnl': float(self.total_pnl or 0),
            'avg_win': float(self.avg_win or 0),
            'avg_loss': float(self.avg_loss or 0),
            'rank_by_sharpe': self.rank_by_sharpe,
            'rank_by_return': self.rank_by_return,
            'rank_by_risk_adjusted': self.rank_by_risk_adjusted,
            'risk_adjusted_score': self.get_risk_adjusted_score(),
            'trend': self.get_trend_vs_previous(),
        }

    def to_leaderboard_entry(self, rank: int) -> dict:
        """Convert to leaderboard entry format."""
        return {
            'rank': rank,
            'strategy': self.strategy_name,
            'total_return_pct': float(self.total_return_pct or 0),
            'sharpe_ratio': float(self.sharpe_ratio or 0),
            'max_drawdown_pct': float(self.max_drawdown_pct or 0),
            'win_rate': float(self.win_rate or 0),
            'trades_count': self.trades_count,
            'vs_spy_return': float(self.vs_spy_return or 0),
            'trend': self.get_trend_vs_previous(),
            'risk_adjusted_score': self.get_risk_adjusted_score(),
        }


class CustomStrategy(models.Model):
    """
    User-created custom trading strategy with visual builder support.

    Stores strategy definition as JSON for flexible condition building.
    Supports entry/exit conditions, position sizing, filters, and stock universe.
    """

    # Universe choices for stock selection
    UNIVERSE_CHOICES = [
        ('sp500', 'S&P 500'),
        ('nasdaq100', 'Nasdaq 100'),
        ('dow30', 'Dow 30'),
        ('russell2000', 'Russell 2000'),
        ('custom', 'Custom List'),
        ('all', 'All US Stocks'),
    ]

    # Position sizing type choices
    SIZING_CHOICES = [
        ('fixed_percent', 'Fixed Percentage'),
        ('fixed_dollar', 'Fixed Dollar Amount'),
        ('kelly', 'Kelly Criterion'),
        ('equal_weight', 'Equal Weight'),
        ('volatility_adjusted', 'Volatility Adjusted'),
    ]

    # Owner
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='custom_strategies'
    )

    # Basic info
    name = models.CharField(max_length=100, db_index=True)
    description = models.TextField(blank=True)

    # Strategy definition as JSON
    # Structure documented in class docstring
    definition = models.JSONField(default=dict)
    """
    Strategy definition structure:
    {
        "entry_conditions": [
            {"indicator": "rsi", "operator": "less_than", "value": 30, "period": 14},
            {"indicator": "macd_histogram", "operator": "crosses_above", "value": 0}
        ],
        "entry_logic": "all",  # "all" (AND) or "any" (OR)
        "exit_conditions": [
            {"type": "take_profit", "value": 10},
            {"type": "stop_loss", "value": 5},
            {"type": "trailing_stop", "value": 7},
            {"type": "time_based", "days": 5},
            {"type": "indicator", "indicator": "rsi", "operator": "greater_than", "value": 70}
        ],
        "exit_logic": "any",
        "position_sizing": {
            "type": "fixed_percent",
            "value": 5,
            "max_positions": 5
        },
        "filters": {
            "min_price": 10,
            "max_price": null,
            "min_volume": 1000000,
            "sectors": ["technology", "healthcare"],
            "exclude_earnings_window": 5,
            "min_market_cap": null
        },
        "universe": "sp500",
        "custom_symbols": ["AAPL", "MSFT"]
    }
    """

    # Universe settings
    universe = models.CharField(
        max_length=20,
        choices=UNIVERSE_CHOICES,
        default='sp500'
    )
    custom_symbols = models.JSONField(
        default=list,
        blank=True,
        help_text='Custom list of symbols when universe is "custom"'
    )

    # Status
    is_active = models.BooleanField(
        default=False,
        help_text='Whether strategy is actively trading'
    )
    is_validated = models.BooleanField(
        default=False,
        help_text='Whether strategy has passed validation'
    )
    validation_errors = models.JSONField(
        null=True,
        blank=True,
        help_text='List of validation errors if any'
    )
    validation_warnings = models.JSONField(
        null=True,
        blank=True,
        help_text='List of validation warnings'
    )

    # Template/sharing
    is_template = models.BooleanField(
        default=False,
        help_text='Whether this is a template strategy'
    )
    is_public = models.BooleanField(
        default=False,
        help_text='Whether this strategy is publicly visible'
    )
    cloned_from = models.ForeignKey(
        'self',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='clones'
    )
    clone_count = models.PositiveIntegerField(default=0)

    # Performance tracking
    backtest_results = models.JSONField(
        null=True,
        blank=True,
        help_text='Most recent backtest results'
    )
    """
    Backtest results structure:
    {
        "period": "1Y",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "total_return_pct": 15.5,
        "sharpe_ratio": 1.2,
        "sortino_ratio": 1.5,
        "max_drawdown_pct": 8.3,
        "win_rate": 62.5,
        "profit_factor": 1.8,
        "total_trades": 48,
        "winning_trades": 30,
        "losing_trades": 18,
        "avg_trade_pnl": 125.50,
        "best_trade_pnl": 850.00,
        "worst_trade_pnl": -320.00,
        "avg_hold_days": 3.5,
        "signals_generated": [...],
        "equity_curve": [...],
        "run_at": "2024-12-31T10:30:00Z"
    }
    """

    live_performance = models.JSONField(
        null=True,
        blank=True,
        help_text='Live trading performance metrics'
    )
    """
    Live performance structure:
    {
        "activated_at": "2024-12-01T09:30:00Z",
        "total_return_pct": 5.2,
        "total_pnl": 2600.00,
        "total_trades": 12,
        "winning_trades": 8,
        "open_positions": 2,
        "last_trade_at": "2024-12-30T14:45:00Z",
        "current_drawdown_pct": 1.5
    }
    """

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_backtest_at = models.DateTimeField(null=True, blank=True)
    activated_at = models.DateTimeField(null=True, blank=True)
    deactivated_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-updated_at']
        verbose_name = 'Custom Strategy'
        verbose_name_plural = 'Custom Strategies'
        unique_together = ['user', 'name']
        indexes = [
            models.Index(fields=['user', 'is_active']),
            models.Index(fields=['is_template', 'is_public']),
        ]

    def __str__(self):
        return f"{self.name} ({self.user.username})"

    @property
    def entry_conditions_count(self) -> int:
        """Number of entry conditions defined."""
        return len(self.definition.get('entry_conditions', []))

    @property
    def exit_conditions_count(self) -> int:
        """Number of exit conditions defined."""
        return len(self.definition.get('exit_conditions', []))

    @property
    def has_stop_loss(self) -> bool:
        """Check if strategy has stop loss configured."""
        exit_conditions = self.definition.get('exit_conditions', [])
        return any(c.get('type') == 'stop_loss' for c in exit_conditions)

    @property
    def has_take_profit(self) -> bool:
        """Check if strategy has take profit configured."""
        exit_conditions = self.definition.get('exit_conditions', [])
        return any(c.get('type') == 'take_profit' for c in exit_conditions)

    @property
    def position_size_type(self) -> str:
        """Get position sizing type."""
        sizing = self.definition.get('position_sizing', {})
        return sizing.get('type', 'fixed_percent')

    @property
    def max_positions(self) -> int:
        """Get maximum concurrent positions."""
        sizing = self.definition.get('position_sizing', {})
        return sizing.get('max_positions', 5)

    def get_pseudo_code(self) -> str:
        """
        Generate human-readable pseudo-code from definition.

        Returns:
            String representation like "BUY when RSI < 30 AND MACD crosses above 0"
        """
        entry_conditions = self.definition.get('entry_conditions', [])
        entry_logic = self.definition.get('entry_logic', 'all')
        exit_conditions = self.definition.get('exit_conditions', [])

        # Format entry conditions
        entry_parts = []
        for cond in entry_conditions:
            indicator = cond.get('indicator', '').upper()
            operator = cond.get('operator', '')
            value = cond.get('value', '')
            period = cond.get('period')

            if period:
                indicator = f"{indicator}({period})"

            op_map = {
                'less_than': '<',
                'greater_than': '>',
                'less_equal': '<=',
                'greater_equal': '>=',
                'equals': '=',
                'crosses_above': 'crosses above',
                'crosses_below': 'crosses below',
            }
            op_str = op_map.get(operator, operator)

            entry_parts.append(f"{indicator} {op_str} {value}")

        logic_word = 'AND' if entry_logic == 'all' else 'OR'
        entry_str = f" {logic_word} ".join(entry_parts)

        # Format exit conditions
        exit_parts = []
        for cond in exit_conditions:
            cond_type = cond.get('type', '')
            if cond_type == 'take_profit':
                exit_parts.append(f"Take Profit at {cond.get('value')}%")
            elif cond_type == 'stop_loss':
                exit_parts.append(f"Stop Loss at {cond.get('value')}%")
            elif cond_type == 'trailing_stop':
                exit_parts.append(f"Trailing Stop {cond.get('value')}%")
            elif cond_type == 'time_based':
                exit_parts.append(f"Hold max {cond.get('days')} days")
            elif cond_type == 'indicator':
                indicator = cond.get('indicator', '').upper()
                operator = cond.get('operator', '')
                value = cond.get('value', '')
                op_map = {'greater_than': '>', 'less_than': '<'}
                exit_parts.append(f"{indicator} {op_map.get(operator, operator)} {value}")

        result = f"BUY when {entry_str}" if entry_str else "BUY (no conditions)"

        if exit_parts:
            exit_logic = self.definition.get('exit_logic', 'any')
            exit_word = 'OR' if exit_logic == 'any' else 'AND'
            result += f"\nSELL when {f' {exit_word} '.join(exit_parts)}"

        return result

    def get_validation_summary(self) -> dict:
        """Get validation status summary."""
        return {
            'is_validated': self.is_validated,
            'errors': self.validation_errors or [],
            'warnings': self.validation_warnings or [],
            'error_count': len(self.validation_errors or []),
            'warning_count': len(self.validation_warnings or []),
        }

    def get_backtest_summary(self) -> dict:
        """Get backtest results summary."""
        if not self.backtest_results:
            return {'has_backtest': False}

        br = self.backtest_results
        return {
            'has_backtest': True,
            'period': br.get('period'),
            'total_return_pct': br.get('total_return_pct'),
            'sharpe_ratio': br.get('sharpe_ratio'),
            'max_drawdown_pct': br.get('max_drawdown_pct'),
            'win_rate': br.get('win_rate'),
            'total_trades': br.get('total_trades'),
            'run_at': br.get('run_at'),
        }

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'definition': self.definition,
            'universe': self.universe,
            'custom_symbols': self.custom_symbols,
            'is_active': self.is_active,
            'is_validated': self.is_validated,
            'validation_summary': self.get_validation_summary(),
            'is_template': self.is_template,
            'is_public': self.is_public,
            'clone_count': self.clone_count,
            'backtest_summary': self.get_backtest_summary(),
            'live_performance': self.live_performance,
            'pseudo_code': self.get_pseudo_code(),
            'entry_conditions_count': self.entry_conditions_count,
            'exit_conditions_count': self.exit_conditions_count,
            'has_stop_loss': self.has_stop_loss,
            'has_take_profit': self.has_take_profit,
            'position_size_type': self.position_size_type,
            'max_positions': self.max_positions,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'last_backtest_at': self.last_backtest_at.isoformat() if self.last_backtest_at else None,
            'activated_at': self.activated_at.isoformat() if self.activated_at else None,
        }

    def clone(self, new_user: User = None, new_name: str = None) -> 'CustomStrategy':
        """
        Clone this strategy for another user.

        Args:
            new_user: User to clone for (defaults to same user)
            new_name: Name for cloned strategy (defaults to "{name} (Copy)")

        Returns:
            New CustomStrategy instance
        """
        clone = CustomStrategy(
            user=new_user or self.user,
            name=new_name or f"{self.name} (Copy)",
            description=self.description,
            definition=self.definition.copy(),
            universe=self.universe,
            custom_symbols=self.custom_symbols.copy() if self.custom_symbols else [],
            cloned_from=self,
        )
        clone.save()

        # Update clone count on original
        self.clone_count += 1
        self.save(update_fields=['clone_count'])

        return clone


class BacktestRun(models.Model):
    """Persisted backtest run with full results for history and comparison.

    Stores complete backtest configuration, metrics, equity curve, and trades
    for later analysis and comparison between runs.
    """

    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    # Identification
    run_id = models.CharField(
        max_length=100,
        unique=True,
        db_index=True,
        help_text="Unique identifier for this backtest run"
    )
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='backtest_runs',
        help_text="User who ran this backtest"
    )
    name = models.CharField(
        max_length=200,
        blank=True,
        help_text="User-defined name for this run"
    )

    # Configuration
    strategy_name = models.CharField(
        max_length=100,
        db_index=True,
        help_text="Strategy that was backtested"
    )
    custom_strategy = models.ForeignKey(
        CustomStrategy,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='backtest_runs',
        help_text="Custom strategy if applicable"
    )
    start_date = models.DateField(help_text="Backtest start date")
    end_date = models.DateField(help_text="Backtest end date")
    initial_capital = models.DecimalField(
        max_digits=15,
        decimal_places=2,
        help_text="Starting capital for backtest"
    )
    benchmark = models.CharField(
        max_length=20,
        default='SPY',
        help_text="Benchmark symbol for comparison"
    )

    # Parameters (JSON blob for flexibility)
    parameters = models.JSONField(
        default=dict,
        help_text="Strategy parameters used in this run"
    )
    """
    Parameters structure:
    {
        "position_size_pct": 3.0,
        "stop_loss_pct": 5.0,
        "take_profit_pct": 15.0,
        "max_positions": 10,
        "commission_per_trade": 0,
        "slippage_pct": 0.05
    }
    """

    # Status
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending',
        help_text="Current status of the backtest"
    )
    progress = models.IntegerField(
        default=0,
        help_text="Progress percentage (0-100)"
    )
    error_message = models.TextField(
        blank=True,
        help_text="Error message if backtest failed"
    )

    # Summary Metrics
    total_return_pct = models.FloatField(null=True, help_text="Total return percentage")
    benchmark_return_pct = models.FloatField(null=True, help_text="Benchmark return percentage")
    alpha = models.FloatField(null=True, help_text="Alpha vs benchmark")
    beta = models.FloatField(null=True, help_text="Beta vs benchmark")
    sharpe_ratio = models.FloatField(null=True, help_text="Sharpe ratio")
    sortino_ratio = models.FloatField(null=True, help_text="Sortino ratio")
    calmar_ratio = models.FloatField(null=True, help_text="Calmar ratio")
    max_drawdown_pct = models.FloatField(null=True, help_text="Maximum drawdown percentage")
    win_rate = models.FloatField(null=True, help_text="Win rate percentage")
    profit_factor = models.FloatField(null=True, help_text="Profit factor")
    total_trades = models.IntegerField(null=True, help_text="Total number of trades")
    winning_trades = models.IntegerField(null=True, help_text="Number of winning trades")
    losing_trades = models.IntegerField(null=True, help_text="Number of losing trades")
    avg_win = models.DecimalField(max_digits=12, decimal_places=2, null=True, help_text="Average winning trade")
    avg_loss = models.DecimalField(max_digits=12, decimal_places=2, null=True, help_text="Average losing trade")
    avg_hold_days = models.FloatField(null=True, help_text="Average holding period in days")
    final_equity = models.DecimalField(max_digits=15, decimal_places=2, null=True, help_text="Final portfolio value")
    total_pnl = models.DecimalField(max_digits=15, decimal_places=2, null=True, help_text="Total profit/loss")

    # Detailed Results (JSON)
    monthly_returns = models.JSONField(
        default=dict,
        help_text="Monthly returns {YYYY-MM: return_pct}"
    )
    equity_curve = models.JSONField(
        default=list,
        help_text="Daily equity snapshots [{date, equity, benchmark}]"
    )
    drawdown_curve = models.JSONField(
        default=list,
        help_text="Daily drawdown values [{date, drawdown_pct}]"
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    # Organization
    tags = models.JSONField(default=list, help_text="Tags for filtering")
    is_favorite = models.BooleanField(default=False, help_text="User marked as favorite")
    notes = models.TextField(blank=True, help_text="User notes about this run")

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Backtest Run'
        verbose_name_plural = 'Backtest Runs'
        indexes = [
            models.Index(fields=['user', 'strategy_name', 'created_at']),
            models.Index(fields=['status', 'created_at']),
            models.Index(fields=['user', 'is_favorite']),
        ]

    def __str__(self):
        return f"Backtest {self.run_id}: {self.strategy_name} ({self.start_date} to {self.end_date})"

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            'run_id': self.run_id,
            'name': self.name,
            'strategy_name': self.strategy_name,
            'custom_strategy_id': self.custom_strategy_id,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'initial_capital': float(self.initial_capital) if self.initial_capital else None,
            'benchmark': self.benchmark,
            'parameters': self.parameters,
            'status': self.status,
            'progress': self.progress,
            'error_message': self.error_message,
            'summary': {
                'total_return_pct': self.total_return_pct,
                'benchmark_return_pct': self.benchmark_return_pct,
                'alpha': self.alpha,
                'sharpe_ratio': self.sharpe_ratio,
                'sortino_ratio': self.sortino_ratio,
                'max_drawdown_pct': self.max_drawdown_pct,
                'win_rate': self.win_rate,
                'profit_factor': self.profit_factor,
                'total_trades': self.total_trades,
                'avg_hold_days': self.avg_hold_days,
                'final_equity': float(self.final_equity) if self.final_equity else None,
                'total_pnl': float(self.total_pnl) if self.total_pnl else None,
            },
            'monthly_returns': self.monthly_returns,
            'equity_curve': self.equity_curve,
            'drawdown_curve': self.drawdown_curve,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'is_favorite': self.is_favorite,
            'tags': self.tags,
        }


class BacktestTrade(models.Model):
    """Individual trade from a backtest run."""

    DIRECTION_CHOICES = [
        ('long', 'Long'),
        ('short', 'Short'),
    ]

    STATUS_CHOICES = [
        ('open', 'Open'),
        ('closed', 'Closed'),
        ('stopped_out', 'Stopped Out'),
        ('take_profit', 'Take Profit'),
        ('time_exit', 'Time Exit'),
    ]

    backtest_run = models.ForeignKey(
        BacktestRun,
        on_delete=models.CASCADE,
        related_name='trades',
        help_text="Parent backtest run"
    )
    trade_id = models.CharField(max_length=50, help_text="Trade identifier within the run")
    symbol = models.CharField(max_length=20, db_index=True, help_text="Traded symbol")
    direction = models.CharField(max_length=10, choices=DIRECTION_CHOICES, help_text="Trade direction")

    # Entry
    entry_date = models.DateField(help_text="Entry date")
    entry_price = models.DecimalField(max_digits=12, decimal_places=4, help_text="Entry price")
    quantity = models.IntegerField(help_text="Number of shares/contracts")

    # Exit
    exit_date = models.DateField(null=True, blank=True, help_text="Exit date")
    exit_price = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True, help_text="Exit price")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='open', help_text="Trade status")

    # P&L
    pnl = models.DecimalField(max_digits=12, decimal_places=2, default=0, help_text="Profit/loss in dollars")
    pnl_pct = models.FloatField(default=0, help_text="Profit/loss percentage")
    commission = models.DecimalField(max_digits=10, decimal_places=2, default=0, help_text="Commission paid")
    slippage = models.DecimalField(max_digits=10, decimal_places=2, default=0, help_text="Slippage cost")

    # Metadata
    entry_reason = models.CharField(max_length=200, blank=True, help_text="Signal that triggered entry")
    exit_reason = models.CharField(max_length=200, blank=True, help_text="Reason for exit")

    class Meta:
        ordering = ['-entry_date']
        verbose_name = 'Backtest Trade'
        verbose_name_plural = 'Backtest Trades'
        indexes = [
            models.Index(fields=['backtest_run', 'symbol']),
            models.Index(fields=['backtest_run', 'entry_date']),
        ]

    def __str__(self):
        return f"{self.direction.upper()} {self.symbol} @ {self.entry_price}"

    @property
    def hold_days(self) -> int:
        """Calculate holding period in days."""
        if self.exit_date and self.entry_date:
            return (self.exit_date - self.entry_date).days
        return 0

    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl > 0

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_date': self.entry_date.isoformat() if self.entry_date else None,
            'entry_price': float(self.entry_price) if self.entry_price else None,
            'quantity': self.quantity,
            'exit_date': self.exit_date.isoformat() if self.exit_date else None,
            'exit_price': float(self.exit_price) if self.exit_price else None,
            'status': self.status,
            'pnl': float(self.pnl),
            'pnl_pct': self.pnl_pct,
            'hold_days': self.hold_days,
            'is_winner': self.is_winner,
            'entry_reason': self.entry_reason,
            'exit_reason': self.exit_reason,
        }


class OptimizationRun(models.Model):
    """Parameter optimization run results using Optuna."""

    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ]

    LOSS_FUNCTION_CHOICES = [
        ('sharpe', 'Maximize Sharpe Ratio'),
        ('sortino', 'Maximize Sortino Ratio'),
        ('calmar', 'Maximize Calmar Ratio'),
        ('return', 'Maximize Total Return'),
        ('profit_factor', 'Maximize Profit Factor'),
        ('win_rate', 'Maximize Win Rate'),
        ('min_drawdown', 'Minimize Max Drawdown'),
    ]

    SAMPLER_CHOICES = [
        ('tpe', 'TPE (Tree-structured Parzen Estimator)'),
        ('random', 'Random Search'),
        ('cmaes', 'CMA-ES'),
        ('grid', 'Grid Search'),
    ]

    # Identification
    run_id = models.CharField(
        max_length=100,
        unique=True,
        db_index=True,
        help_text="Unique identifier for this optimization run"
    )
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='optimization_runs',
        help_text="User who ran this optimization"
    )
    name = models.CharField(max_length=200, blank=True, help_text="User-defined name")

    # Strategy
    strategy_name = models.CharField(max_length=100, help_text="Strategy being optimized")
    custom_strategy = models.ForeignKey(
        CustomStrategy,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='optimization_runs'
    )

    # Configuration
    start_date = models.DateField(help_text="Backtest period start")
    end_date = models.DateField(help_text="Backtest period end")
    initial_capital = models.DecimalField(max_digits=15, decimal_places=2)

    # Optimization settings
    loss_function = models.CharField(
        max_length=50,
        choices=LOSS_FUNCTION_CHOICES,
        default='sharpe',
        help_text="Metric to optimize"
    )
    n_trials = models.IntegerField(default=100, help_text="Number of optimization trials")
    sampler = models.CharField(
        max_length=20,
        choices=SAMPLER_CHOICES,
        default='tpe',
        help_text="Optimization algorithm"
    )
    parameter_ranges = models.JSONField(
        default=dict,
        help_text="Parameter ranges for optimization"
    )
    """
    Parameter ranges structure:
    {
        "position_size_pct": {"min": 1, "max": 10, "step": 0.5},
        "stop_loss_pct": {"min": 2, "max": 15, "step": 1},
        "take_profit_pct": {"min": 5, "max": 30, "step": 2},
        "rsi_period": {"min": 7, "max": 21, "step": 1}
    }
    """

    # Status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    progress = models.IntegerField(default=0, help_text="Progress 0-100")
    current_trial = models.IntegerField(default=0, help_text="Current trial number")

    # Results
    best_params = models.JSONField(default=dict, help_text="Best parameters found")
    best_value = models.FloatField(null=True, help_text="Best objective value")
    best_sharpe = models.FloatField(null=True, help_text="Sharpe ratio with best params")
    best_return_pct = models.FloatField(null=True, help_text="Return with best params")
    best_drawdown_pct = models.FloatField(null=True, help_text="Max drawdown with best params")

    # All trials for visualization
    all_trials = models.JSONField(
        default=list,
        help_text="All trial results [{params, value, metrics}]"
    )
    parameter_importance = models.JSONField(
        default=dict,
        help_text="Parameter importance scores"
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Optimization Run'
        verbose_name_plural = 'Optimization Runs'

    def __str__(self):
        return f"Optimization {self.run_id}: {self.strategy_name} ({self.status})"

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            'run_id': self.run_id,
            'name': self.name,
            'strategy_name': self.strategy_name,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'loss_function': self.loss_function,
            'n_trials': self.n_trials,
            'sampler': self.sampler,
            'parameter_ranges': self.parameter_ranges,
            'status': self.status,
            'progress': self.progress,
            'current_trial': self.current_trial,
            'best_params': self.best_params,
            'best_value': self.best_value,
            'best_sharpe': self.best_sharpe,
            'best_return_pct': self.best_return_pct,
            'best_drawdown_pct': self.best_drawdown_pct,
            'parameter_importance': self.parameter_importance,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
        }


class WalkForwardRun(models.Model):
    """Walk-forward analysis run for strategy robustness testing."""

    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    # Identification
    run_id = models.CharField(max_length=100, unique=True, db_index=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='walkforward_runs')
    name = models.CharField(max_length=200, blank=True)

    # Strategy
    strategy_name = models.CharField(max_length=100)
    custom_strategy = models.ForeignKey(
        CustomStrategy,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='walkforward_runs'
    )

    # Configuration
    start_date = models.DateField()
    end_date = models.DateField()
    initial_capital = models.DecimalField(max_digits=15, decimal_places=2)
    train_window_days = models.IntegerField(default=90, help_text="Training window in days")
    test_window_days = models.IntegerField(default=30, help_text="Out-of-sample test window in days")
    step_days = models.IntegerField(default=30, help_text="Step size between windows")
    parameters = models.JSONField(default=dict, help_text="Fixed parameters for testing")

    # Status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    progress = models.IntegerField(default=0)

    # Results
    total_windows = models.IntegerField(default=0, help_text="Total number of walk-forward windows")
    windows_completed = models.IntegerField(default=0)

    # Aggregated metrics
    avg_is_return = models.FloatField(null=True, help_text="Average in-sample return")
    avg_oos_return = models.FloatField(null=True, help_text="Average out-of-sample return")
    avg_is_sharpe = models.FloatField(null=True, help_text="Average in-sample Sharpe")
    avg_oos_sharpe = models.FloatField(null=True, help_text="Average out-of-sample Sharpe")
    robustness_ratio = models.FloatField(null=True, help_text="OOS/IS performance ratio")
    stability_score = models.FloatField(null=True, help_text="Performance stability across windows")

    # Detailed window results
    window_results = models.JSONField(
        default=list,
        help_text="Results for each walk-forward window"
    )
    """
    Window results structure:
    [{
        "window": 1,
        "train_start": "2023-01-01",
        "train_end": "2023-03-31",
        "test_start": "2023-04-01",
        "test_end": "2023-04-30",
        "is_return": 5.2,
        "oos_return": 3.1,
        "is_sharpe": 1.5,
        "oos_sharpe": 1.1,
        "best_params": {...}
    }]
    """

    recommendations = models.JSONField(default=list, help_text="Recommendations based on analysis")
    warnings = models.JSONField(default=list, help_text="Warnings about strategy robustness")

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Walk-Forward Run'
        verbose_name_plural = 'Walk-Forward Runs'

    def __str__(self):
        return f"WalkForward {self.run_id}: {self.strategy_name}"


# ============================================================================
# ML/RL Agent Models
# ============================================================================

class MLModel(models.Model):
    """Machine Learning model configuration and state."""

    MODEL_TYPES = [
        ('lstm', 'LSTM - Sequence Prediction'),
        ('cnn', 'CNN - Pattern Recognition'),
        ('transformer', 'Transformer - Attention Model'),
        ('hmm', 'HMM - Regime Detection'),
        ('xgboost', 'XGBoost - Gradient Boosting'),
        ('random_forest', 'Random Forest'),
    ]

    MODEL_STATUS = [
        ('idle', 'Idle'),
        ('active', 'Active'),
        ('training', 'Training'),
        ('error', 'Error'),
    ]

    PREDICTION_TARGETS = [
        ('direction', 'Price Direction'),
        ('returns', 'Returns'),
        ('volatility', 'Volatility'),
        ('regime', 'Market Regime'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='ml_models')
    name = models.CharField(max_length=100)
    model_type = models.CharField(max_length=20, choices=MODEL_TYPES, default='lstm')
    status = models.CharField(max_length=20, choices=MODEL_STATUS, default='idle')

    # Target configuration
    symbols = models.CharField(max_length=500, help_text="Comma-separated symbols")
    prediction_target = models.CharField(max_length=20, choices=PREDICTION_TARGETS, default='direction')
    lookback_period = models.IntegerField(default=60, help_text="Days of historical data")

    # Hyperparameters (stored as JSON for flexibility)
    hyperparameters = models.JSONField(default=dict, help_text="Model hyperparameters")
    """
    Default hyperparameters structure:
    {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "hidden_layers": 2,
        "hidden_units": 64,
        "dropout": 0.2,
        "optimizer": "adam",
        "loss_function": "binary_crossentropy"
    }
    """

    # Performance metrics
    accuracy = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    predictions_count = models.IntegerField(default=0)

    # Training history
    accuracy_history = models.JSONField(default=list)
    loss_history = models.JSONField(default=list)
    validation_history = models.JSONField(default=list)

    # Model artifact storage
    model_path = models.CharField(max_length=500, blank=True, help_text="Path to saved model file")
    model_version = models.IntegerField(default=1)

    # Feature configuration
    features_config = models.JSONField(default=dict, help_text="Feature engineering config")
    feature_importance = models.JSONField(default=dict, help_text="Feature importance scores")

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_trained = models.DateTimeField(null=True, blank=True)
    last_prediction = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-updated_at']
        unique_together = ['user', 'name']

    def __str__(self):
        return f"{self.name} ({self.model_type})"

    def to_dict(self):
        return {
            'id': str(self.id),
            'name': self.name,
            'type': self.model_type,
            'status': self.status,
            'symbols': self.symbols,
            'prediction_target': self.prediction_target,
            'lookback_period': self.lookback_period,
            'hyperparameters': self.hyperparameters,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'predictions_count': self.predictions_count,
            'accuracy_history': self.accuracy_history,
            'loss_history': self.loss_history,
            'feature_importance': self.feature_importance,
            'model_version': self.model_version,
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }

    def get_default_hyperparameters(self):
        """Return default hyperparameters based on model type."""
        defaults = {
            'lstm': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'hidden_layers': 2,
                'hidden_units': 64,
                'dropout': 0.2,
                'sequence_length': 20,
            },
            'cnn': {
                'learning_rate': 0.001,
                'batch_size': 64,
                'epochs': 50,
                'filters': [32, 64, 128],
                'kernel_size': 3,
                'dropout': 0.25,
            },
            'transformer': {
                'learning_rate': 0.0001,
                'batch_size': 32,
                'epochs': 100,
                'num_heads': 4,
                'ff_dim': 128,
                'num_layers': 2,
                'dropout': 0.1,
            },
            'hmm': {
                'n_states': 3,
                'n_iterations': 100,
                'covariance_type': 'full',
            },
        }
        return defaults.get(self.model_type, defaults['lstm'])


class RLAgent(models.Model):
    """Reinforcement Learning agent configuration and state."""

    AGENT_TYPES = [
        ('ppo', 'PPO - Proximal Policy Optimization'),
        ('ddpg', 'DDPG - Deep Deterministic Policy Gradient'),
        ('a2c', 'A2C - Advantage Actor-Critic'),
        ('sac', 'SAC - Soft Actor-Critic'),
        ('td3', 'TD3 - Twin Delayed DDPG'),
    ]

    AGENT_STATUS = [
        ('idle', 'Idle'),
        ('active', 'Active'),
        ('training', 'Training'),
        ('error', 'Error'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='rl_agents')
    name = models.CharField(max_length=100)
    agent_type = models.CharField(max_length=20, choices=AGENT_TYPES, default='ppo')
    status = models.CharField(max_length=20, choices=AGENT_STATUS, default='idle')

    # Environment configuration
    symbols = models.CharField(max_length=500, default='SPY', help_text="Trading symbols")
    initial_capital = models.DecimalField(max_digits=15, decimal_places=2, default=100000)
    max_position_size = models.FloatField(default=1.0, help_text="Max position as fraction of portfolio")
    transaction_cost = models.FloatField(default=0.001, help_text="Transaction cost as fraction")

    # Hyperparameters
    hyperparameters = models.JSONField(default=dict)
    """
    Default hyperparameters structure:
    {
        "actor_lr": 0.0003,
        "critic_lr": 0.0003,
        "gamma": 0.99,
        "tau": 0.005,
        "buffer_size": 100000,
        "batch_size": 64,
        "update_frequency": 1,
        "clip_ratio": 0.2,  # PPO specific
        "entropy_coef": 0.01,
        "risk_penalty": 0.1
    }
    """

    # Training state
    total_episodes = models.IntegerField(default=0)
    total_timesteps = models.IntegerField(default=0)

    # Performance metrics
    avg_reward = models.FloatField(null=True, blank=True)
    sharpe_ratio = models.FloatField(null=True, blank=True)
    sortino_ratio = models.FloatField(null=True, blank=True)
    max_drawdown = models.FloatField(null=True, blank=True)
    win_rate = models.FloatField(null=True, blank=True)
    avg_return = models.FloatField(null=True, blank=True)
    total_return = models.FloatField(null=True, blank=True)

    # History
    reward_history = models.JSONField(default=list)
    returns_history = models.JSONField(default=list)
    episode_lengths = models.JSONField(default=list)

    # Model storage
    actor_path = models.CharField(max_length=500, blank=True)
    critic_path = models.CharField(max_length=500, blank=True)
    model_version = models.IntegerField(default=1)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_trained = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-updated_at']
        unique_together = ['user', 'name']

    def __str__(self):
        return f"{self.name} ({self.agent_type})"

    def to_dict(self):
        return {
            'id': str(self.id),
            'name': self.name,
            'type': self.agent_type,
            'status': self.status,
            'symbols': self.symbols,
            'episodes': self.total_episodes,
            'timesteps': self.total_timesteps,
            'avg_reward': self.avg_reward,
            'sharpe': self.sharpe_ratio,
            'sortino': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'avg_return': self.avg_return,
            'total_return': self.total_return,
            'reward_history': self.reward_history[-100:] if self.reward_history else [],
            'returns_history': self.returns_history[-100:] if self.returns_history else [],
            'hyperparameters': self.hyperparameters,
            'model_version': self.model_version,
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }

    def get_default_hyperparameters(self):
        """Return default hyperparameters based on agent type."""
        defaults = {
            'ppo': {
                'actor_lr': 0.0003,
                'critic_lr': 0.0003,
                'gamma': 0.99,
                'clip_ratio': 0.2,
                'entropy_coef': 0.01,
                'value_coef': 0.5,
                'max_grad_norm': 0.5,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
            },
            'ddpg': {
                'actor_lr': 0.0001,
                'critic_lr': 0.001,
                'gamma': 0.99,
                'tau': 0.005,
                'buffer_size': 100000,
                'batch_size': 64,
                'noise_std': 0.1,
                'noise_clip': 0.5,
            },
            'sac': {
                'actor_lr': 0.0003,
                'critic_lr': 0.0003,
                'gamma': 0.99,
                'tau': 0.005,
                'alpha': 0.2,
                'buffer_size': 100000,
                'batch_size': 256,
            },
        }
        return defaults.get(self.agent_type, defaults['ppo'])


class TrainingJob(models.Model):
    """Track ML model and RL agent training jobs."""

    JOB_STATUS = [
        ('queued', 'Queued'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ]

    JOB_TYPES = [
        ('ml_model', 'ML Model Training'),
        ('rl_agent', 'RL Agent Training'),
        ('optimization', 'Hyperparameter Optimization'),
    ]

    job_id = models.CharField(max_length=100, unique=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='training_jobs')
    job_type = models.CharField(max_length=20, choices=JOB_TYPES)
    status = models.CharField(max_length=20, choices=JOB_STATUS, default='queued')

    # Reference to model/agent being trained
    ml_model = models.ForeignKey(MLModel, on_delete=models.CASCADE, null=True, blank=True, related_name='training_jobs')
    rl_agent = models.ForeignKey(RLAgent, on_delete=models.CASCADE, null=True, blank=True, related_name='training_jobs')

    # Training configuration
    training_config = models.JSONField(default=dict)
    """
    Config structure:
    {
        "epochs": 100,
        "batch_size": 32,
        "validation_split": 0.2,
        "early_stopping": true,
        "patience": 10,
        "data_start": "2020-01-01",
        "data_end": "2024-01-01"
    }
    """

    # Progress tracking
    progress = models.FloatField(default=0, help_text="Progress 0-100")
    current_epoch = models.IntegerField(default=0)
    total_epochs = models.IntegerField(default=100)
    current_loss = models.FloatField(null=True, blank=True)
    current_metric = models.FloatField(null=True, blank=True)

    # Training metrics (updated during training)
    metrics_history = models.JSONField(default=list)
    """
    Metrics history structure:
    [{
        "epoch": 1,
        "train_loss": 0.5,
        "val_loss": 0.6,
        "train_accuracy": 0.6,
        "val_accuracy": 0.55,
        "timestamp": "2024-01-01T10:00:00"
    }]
    """

    # Results
    final_metrics = models.JSONField(default=dict, help_text="Final training metrics")
    best_epoch = models.IntegerField(null=True, blank=True)
    best_metric_value = models.FloatField(null=True, blank=True)

    # Error tracking
    error_message = models.TextField(blank=True)
    error_traceback = models.TextField(blank=True)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    # Computed fields
    @property
    def duration_seconds(self):
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            from django.utils import timezone
            return (timezone.now() - self.started_at).total_seconds()
        return None

    @property
    def eta_seconds(self):
        if self.progress > 0 and self.started_at:
            elapsed = self.duration_seconds
            if elapsed:
                return (elapsed / self.progress) * (100 - self.progress)
        return None

    @property
    def model_name(self):
        if self.ml_model:
            return self.ml_model.name
        elif self.rl_agent:
            return self.rl_agent.name
        return "Unknown"

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Training Job {self.job_id}: {self.model_name}"

    def to_dict(self):
        return {
            'id': self.job_id,
            'model_name': self.model_name,
            'type': self.job_type,
            'status': self.status,
            'progress': self.progress,
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'current_loss': self.current_loss,
            'eta': self._format_eta(),
            'started': self.started_at.isoformat() if self.started_at else None,
            'duration': self._format_duration(),
            'final_metrics': self.final_metrics,
            'error_message': self.error_message if self.status == 'failed' else None,
        }

    def _format_duration(self):
        seconds = self.duration_seconds
        if not seconds:
            return None
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"

    def _format_eta(self):
        seconds = self.eta_seconds
        if not seconds:
            return "Calculating..."
        minutes = int(seconds // 60)
        if minutes > 60:
            hours = minutes // 60
            minutes = minutes % 60
            return f"{hours}h {minutes}m"
        return f"{minutes} minutes"


# Signal to auto-create UserProfile when User is created
from django.db.models.signals import post_save
from django.dispatch import receiver


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    """Create UserProfile when a new User is created."""
    if created:
        UserProfile.objects.create(user=instance)


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    """Ensure UserProfile is saved when User is saved."""
    if hasattr(instance, 'profile'):
        instance.profile.save()

