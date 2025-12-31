"""
Ensemble Alpha Model

Combines multiple alpha models for more robust signals.
"""

from datetime import timedelta
from typing import Any, Dict, List, Optional
import numpy as np

from ..alpha_model import AlphaModel
from ..insight import Insight, InsightDirection


class EnsembleAlphaModel(AlphaModel):
    """
    Alpha model that combines signals from multiple alpha models.

    Aggregation methods:
    - 'majority': Signal when majority of models agree
    - 'unanimous': Signal only when all models agree
    - 'weighted': Weight by model confidence/accuracy
    - 'any': Signal when any model generates insight
    """

    def __init__(
        self,
        alpha_models: List[AlphaModel],
        aggregation_method: str = 'majority',
        min_agreement: float = 0.5,  # For weighted method
        use_model_accuracy: bool = True,  # Weight by historical accuracy
        insight_period: timedelta = timedelta(days=5),
        name: str = "EnsembleAlpha",
    ):
        """
        Initialize Ensemble Alpha Model.

        Args:
            alpha_models: List of alpha models to combine
            aggregation_method: How to combine signals
            min_agreement: Minimum weighted agreement for signal
            use_model_accuracy: Weight models by historical accuracy
            insight_period: How long insights are valid
            name: Model name
        """
        super().__init__(name)
        self.alpha_models = alpha_models
        self.aggregation_method = aggregation_method
        self.min_agreement = min_agreement
        self.use_model_accuracy = use_model_accuracy
        self.insight_period = insight_period

    def generate_insights(
        self,
        data: Dict[str, Any],
        symbols: List[str],
    ) -> List[Insight]:
        """Generate ensemble insights from multiple models."""
        # Collect insights from all models
        all_insights: Dict[str, List[Insight]] = {symbol: [] for symbol in symbols}

        for model in self.alpha_models:
            model_insights = model.generate_insights(data, symbols)
            for insight in model_insights:
                all_insights[insight.symbol].append(insight)

        # Aggregate insights
        if self.aggregation_method == 'majority':
            return self._aggregate_majority(all_insights)
        elif self.aggregation_method == 'unanimous':
            return self._aggregate_unanimous(all_insights)
        elif self.aggregation_method == 'weighted':
            return self._aggregate_weighted(all_insights)
        elif self.aggregation_method == 'any':
            return self._aggregate_any(all_insights)
        else:
            return self._aggregate_majority(all_insights)

    def _aggregate_majority(
        self,
        all_insights: Dict[str, List[Insight]],
    ) -> List[Insight]:
        """Aggregate by majority vote."""
        results = []
        num_models = len(self.alpha_models)
        threshold = num_models // 2 + 1

        for symbol, insights in all_insights.items():
            if len(insights) < threshold:
                continue

            # Count directions
            up_count = sum(1 for i in insights if i.direction == InsightDirection.UP)
            down_count = sum(1 for i in insights if i.direction == InsightDirection.DOWN)

            if up_count >= threshold:
                # Majority says UP
                avg_magnitude = np.mean([i.magnitude for i in insights if i.is_long])
                avg_confidence = np.mean([i.confidence for i in insights if i.is_long])

                results.append(Insight(
                    symbol=symbol,
                    direction=InsightDirection.UP,
                    magnitude=avg_magnitude,
                    confidence=avg_confidence,
                    period=self.insight_period,
                    source_model=self.name,
                    metadata={
                        'agreement': up_count / num_models,
                        'models_agreed': up_count,
                        'total_models': num_models,
                    },
                ))

            elif down_count >= threshold:
                # Majority says DOWN
                avg_magnitude = np.mean([i.magnitude for i in insights if i.is_short])
                avg_confidence = np.mean([i.confidence for i in insights if i.is_short])

                results.append(Insight(
                    symbol=symbol,
                    direction=InsightDirection.DOWN,
                    magnitude=avg_magnitude,
                    confidence=avg_confidence,
                    period=self.insight_period,
                    source_model=self.name,
                    metadata={
                        'agreement': down_count / num_models,
                        'models_agreed': down_count,
                        'total_models': num_models,
                    },
                ))

        return results

    def _aggregate_unanimous(
        self,
        all_insights: Dict[str, List[Insight]],
    ) -> List[Insight]:
        """Aggregate requiring unanimous agreement."""
        results = []
        num_models = len(self.alpha_models)

        for symbol, insights in all_insights.items():
            if len(insights) != num_models:
                continue  # Not all models have opinion

            directions = [i.direction for i in insights]

            if all(d == InsightDirection.UP for d in directions):
                avg_magnitude = np.mean([i.magnitude for i in insights])
                avg_confidence = np.mean([i.confidence for i in insights])

                results.append(Insight(
                    symbol=symbol,
                    direction=InsightDirection.UP,
                    magnitude=avg_magnitude,
                    confidence=min(avg_confidence * 1.2, 0.95),  # Boost for unanimity
                    period=self.insight_period,
                    source_model=self.name,
                    metadata={'agreement': 1.0, 'unanimous': True},
                ))

            elif all(d == InsightDirection.DOWN for d in directions):
                avg_magnitude = np.mean([i.magnitude for i in insights])
                avg_confidence = np.mean([i.confidence for i in insights])

                results.append(Insight(
                    symbol=symbol,
                    direction=InsightDirection.DOWN,
                    magnitude=avg_magnitude,
                    confidence=min(avg_confidence * 1.2, 0.95),
                    period=self.insight_period,
                    source_model=self.name,
                    metadata={'agreement': 1.0, 'unanimous': True},
                ))

        return results

    def _aggregate_weighted(
        self,
        all_insights: Dict[str, List[Insight]],
    ) -> List[Insight]:
        """Aggregate using weighted voting by confidence and accuracy."""
        results = []

        for symbol, insights in all_insights.items():
            if not insights:
                continue

            # Calculate weighted scores for each direction
            up_weight = 0.0
            down_weight = 0.0
            total_weight = 0.0

            up_insights = []
            down_insights = []

            for insight in insights:
                # Weight by confidence
                weight = insight.confidence

                # Optionally weight by model accuracy
                if self.use_model_accuracy:
                    model = self._find_model(insight.source_model)
                    if model and model.state.accuracy > 0:
                        weight *= (0.5 + model.state.accuracy)

                total_weight += weight

                if insight.direction == InsightDirection.UP:
                    up_weight += weight
                    up_insights.append((insight, weight))
                elif insight.direction == InsightDirection.DOWN:
                    down_weight += weight
                    down_insights.append((insight, weight))

            # Check if enough agreement
            if total_weight == 0:
                continue

            up_ratio = up_weight / total_weight
            down_ratio = down_weight / total_weight

            if up_ratio >= self.min_agreement:
                # Weighted average of UP insights
                total_w = sum(w for _, w in up_insights)
                avg_magnitude = sum(i.magnitude * w for i, w in up_insights) / total_w
                avg_confidence = sum(i.confidence * w for i, w in up_insights) / total_w

                results.append(Insight(
                    symbol=symbol,
                    direction=InsightDirection.UP,
                    magnitude=avg_magnitude,
                    confidence=avg_confidence,
                    period=self.insight_period,
                    source_model=self.name,
                    metadata={'weighted_agreement': up_ratio},
                ))

            elif down_ratio >= self.min_agreement:
                total_w = sum(w for _, w in down_insights)
                avg_magnitude = sum(i.magnitude * w for i, w in down_insights) / total_w
                avg_confidence = sum(i.confidence * w for i, w in down_insights) / total_w

                results.append(Insight(
                    symbol=symbol,
                    direction=InsightDirection.DOWN,
                    magnitude=avg_magnitude,
                    confidence=avg_confidence,
                    period=self.insight_period,
                    source_model=self.name,
                    metadata={'weighted_agreement': down_ratio},
                ))

        return results

    def _aggregate_any(
        self,
        all_insights: Dict[str, List[Insight]],
    ) -> List[Insight]:
        """Pass through any insight (highest confidence wins for conflicts)."""
        results = []

        for symbol, insights in all_insights.items():
            if not insights:
                continue

            # Group by direction
            up_insights = [i for i in insights if i.is_long]
            down_insights = [i for i in insights if i.is_short]

            # Take highest confidence insight for each direction
            if up_insights:
                best_up = max(up_insights, key=lambda x: x.confidence)
                results.append(Insight(
                    symbol=symbol,
                    direction=InsightDirection.UP,
                    magnitude=best_up.magnitude,
                    confidence=best_up.confidence,
                    period=self.insight_period,
                    source_model=self.name,
                    metadata={
                        'original_source': best_up.source_model,
                        'num_agreeing': len(up_insights),
                    },
                ))

            if down_insights:
                best_down = max(down_insights, key=lambda x: x.confidence)
                results.append(Insight(
                    symbol=symbol,
                    direction=InsightDirection.DOWN,
                    magnitude=best_down.magnitude,
                    confidence=best_down.confidence,
                    period=self.insight_period,
                    source_model=self.name,
                    metadata={
                        'original_source': best_down.source_model,
                        'num_agreeing': len(down_insights),
                    },
                ))

        return results

    def _find_model(self, name: str) -> Optional[AlphaModel]:
        """Find alpha model by name."""
        for model in self.alpha_models:
            if model.name == name:
                return model
        return None

    def on_securities_changed(
        self,
        added: List[str],
        removed: List[str],
    ) -> None:
        """Propagate universe changes to all sub-models."""
        super().on_securities_changed(added, removed)
        for model in self.alpha_models:
            model.on_securities_changed(added, removed)
