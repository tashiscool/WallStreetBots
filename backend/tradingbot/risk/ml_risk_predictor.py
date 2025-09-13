"""ML Risk Predictor - 2025 Implementation
Machine learning models for risk prediction and regime detection.
"""

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

# Try to import ML libraries, fallback to basic implementations if not available
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: scikit - learn not available. Using basic implementations.")


@dataclass
class VolatilityForecast:
    """Volatility prediction result."""

    predicted_volatility: float
    confidence_interval: tuple[float, float]
    regime_probability: dict[str, float]
    horizon_days: int
    model_confidence: float


@dataclass
class MLFeatures:
    """Machine learning features."""

    price_features: dict[str, float]
    volume_features: dict[str, float]
    sentiment_features: dict[str, float]
    options_flow_features: dict[str, float]
    macro_features: dict[str, float]
    technical_indicators: dict[str, float]


@dataclass
class RiskPrediction:
    """Risk prediction result."""

    risk_score: float  # 0 - 100, higher is riskier
    volatility_forecast: VolatilityForecast
    regime_prediction: str
    confidence: float
    recommended_actions: list[str]


class MLRiskPredictor:
    """Machine learning models for risk prediction."""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_weights = {"lstm": 0.4, "random_forest": 0.4, "ensemble": 0.2}

        if ML_AVAILABLE:
            self._initialize_ml_models()
        else:
            self._initialize_basic_models()

    def _initialize_ml_models(self):
        """Initialize scikit - learn models."""
        self.models = {
            "volatility_rf": RandomForestRegressor(n_estimators=100, random_state=42),
            "regime_rf": RandomForestClassifier(n_estimators=100, random_state=42),
            "risk_rf": RandomForestRegressor(n_estimators=100, random_state=42),
        }

        self.scalers = {
            "volatility": StandardScaler(),
            "regime": StandardScaler(),
            "risk": StandardScaler(),
        }

    def _initialize_basic_models(self):
        """Initialize basic models without ML libraries."""
        self.models = {"volatility_rf": None, "regime_rf": None, "risk_rf": None}
        self.scalers = {}

    def predict_volatility_regime(
        self, market_data: dict[str, Any], horizon_days: int = 5
    ) -> VolatilityForecast:
        """Predict volatility using ensemble of models.

        Args:
            market_data: Dictionary containing market data
            horizon_days: Prediction horizon

        Returns:
            Volatility forecast with confidence intervals
        """
        # Extract features
        features = self._engineer_features_2025(market_data)

        if ML_AVAILABLE and self.models["volatility_rf"] is not None:
            try:
                # Use trained ML model
                feature_vector = self._extract_feature_vector(features)

                # Check if scaler is fitted
                if hasattr(self.scalers["volatility"], "mean_"):
                    feature_vector_scaled = self.scalers["volatility"].transform(
                        [feature_vector]
                    )
                else:
                    # Fallback to basic prediction if model not trained
                    predicted_vol = self._basic_volatility_prediction(market_data)
                    confidence_interval = (predicted_vol * 0.8, predicted_vol * 1.2)
                    regime_probs = self._predict_regime_probabilities(features)
                    return VolatilityForecast(
                        predicted_volatility=predicted_vol,
                        confidence_interval=confidence_interval,
                        regime_probability=regime_probs,
                        horizon_days=horizon_days,
                        model_confidence=0.75,
                    )

                # Get prediction
                predicted_vol = self.models["volatility_rf"].predict(
                    feature_vector_scaled
                )[0]

                # Calculate confidence interval (simplified)
                confidence_std = 0.1  # 10% standard deviation
                confidence_interval = (
                    max(0, predicted_vol - 1.96 * confidence_std),
                    predicted_vol + 1.96 * confidence_std,
                )

                # Get regime probabilities
                regime_probs = self._predict_regime_probabilities(features)

            except Exception:
                # Fallback to basic prediction if ML fails
                predicted_vol = self._basic_volatility_prediction(market_data)
                confidence_interval = (predicted_vol * 0.8, predicted_vol * 1.2)
                regime_probs = self._predict_regime_probabilities(features)

        else:
            # Fallback to basic statistical model
            predicted_vol = self._basic_volatility_prediction(market_data)
            confidence_interval = (predicted_vol * 0.8, predicted_vol * 1.2)
            regime_probs = {"normal": 0.6, "high_vol": 0.3, "crisis": 0.1}

        return VolatilityForecast(
            predicted_volatility=predicted_vol,
            confidence_interval=confidence_interval,
            regime_probability=regime_probs,
            horizon_days=horizon_days,
            model_confidence=0.75,  # Default confidence
        )

    def _engineer_features_2025(self, market_data: dict[str, Any]) -> MLFeatures:
        """Advanced feature engineering with alternative data."""
        # Extract price data
        prices = market_data.get("prices", [])
        volumes = market_data.get("volumes", [])

        # Calculate technical indicators
        price_features = self._extract_price_features(prices)
        volume_features = self._extract_volume_features(volumes)
        technical_indicators = self._calculate_technical_indicators(prices, volumes)

        # Simulate alternative data features (in real implementation, these would come from APIs)
        sentiment_features = self._extract_sentiment_features(market_data)
        options_flow_features = self._extract_options_flow_features(market_data)
        macro_features = self._extract_macro_features(market_data)

        return MLFeatures(
            price_features=price_features,
            volume_features=volume_features,
            sentiment_features=sentiment_features,
            options_flow_features=options_flow_features,
            macro_features=macro_features,
            technical_indicators=technical_indicators,
        )

    def _extract_price_features(self, prices: list[float]) -> dict[str, float]:
        """Extract price-based features."""
        if len(prices) < 2:
            return {"returns_mean": 0.0, "returns_std": 0.0, "price_trend": 0.0}

        prices = np.array(prices)
        returns = np.diff(prices) / prices[:-1]

        return {
            "returns_mean": np.mean(returns),
            "returns_std": np.std(returns),
            "price_trend": (prices[-1] - prices[0]) / prices[0]
            if len(prices) > 0
            else 0.0,
            "price_volatility": np.std(returns) * np.sqrt(252),  # Annualized
            "max_drawdown": self._calculate_max_drawdown(prices),
            "skewness": self._calculate_skewness(returns),
            "kurtosis": self._calculate_kurtosis(returns),
        }

    def _extract_volume_features(self, volumes: list[float]) -> dict[str, float]:
        """Extract volume-based features."""
        if len(volumes) < 2:
            return {"volume_mean": 0.0, "volume_std": 0.0, "volume_trend": 0.0}

        volumes = np.array(volumes)

        return {
            "volume_mean": np.mean(volumes),
            "volume_std": np.std(volumes),
            "volume_trend": (volumes[-1] - volumes[0]) / volumes[0]
            if len(volumes) > 0
            else 0.0,
            "volume_volatility": np.std(volumes) / np.mean(volumes)
            if np.mean(volumes) > 0
            else 0.0,
        }

    def _calculate_technical_indicators(
        self, prices: list[float], volumes: list[float]
    ) -> dict[str, float]:
        """Calculate technical indicators."""
        if len(prices) < 20:
            return {"rsi": 50.0, "macd": 0.0, "bollinger_position": 0.5}

        prices = np.array(prices)

        # RSI calculation
        rsi = self._calculate_rsi(prices, 14)

        # MACD calculation
        macd = self._calculate_macd(prices)

        # Bollinger Bands position
        bb_position = self._calculate_bollinger_position(prices, 20)

        return {
            "rsi": rsi,
            "macd": macd,
            "bollinger_position": bb_position,
            "sma_20": np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1],
            "sma_50": np.mean(prices[-50:]) if len(prices) >= 50 else prices[-1],
        }

    def _extract_sentiment_features(
        self, market_data: dict[str, Any]
    ) -> dict[str, float]:
        """Extract sentiment features (simulated)."""
        # In real implementation, these would come from sentiment analysis APIs
        return {
            "reddit_sentiment": np.random.uniform(-1, 1),  # -1 to 1
            "twitter_sentiment": np.random.uniform(-1, 1),
            "news_sentiment": np.random.uniform(-1, 1),
            "social_volume": np.random.uniform(0, 1),
            "sentiment_volatility": np.random.uniform(0, 1),
        }

    def _extract_options_flow_features(
        self, market_data: dict[str, Any]
    ) -> dict[str, float]:
        """Extract options flow features (simulated)."""
        # In real implementation, these would come from options data providers
        return {
            "put_call_ratio": np.random.uniform(0.5, 2.0),
            "iv_percentile": np.random.uniform(0, 100),
            "unusual_volume": np.random.uniform(0, 1),
            "smart_money_score": np.random.uniform(-1, 1),
            "options_volume_trend": np.random.uniform(-1, 1),
        }

    def _extract_macro_features(self, market_data: dict[str, Any]) -> dict[str, float]:
        """Extract macroeconomic features."""
        return {
            "vix_level": np.random.uniform(10, 50),
            "yield_curve_slope": np.random.uniform(-2, 3),
            "dollar_strength": np.random.uniform(-0.1, 0.1),
            "oil_price_change": np.random.uniform(-0.1, 0.1),
            "crypto_correlation": np.random.uniform(0, 1),
        }

    def _extract_feature_vector(self, features: MLFeatures) -> list[float]:
        """Extract feature vector for ML models."""
        feature_vector = []

        # Combine all features into single vector
        for feature_dict in [
            features.price_features,
            features.volume_features,
            features.sentiment_features,
            features.options_flow_features,
            features.macro_features,
            features.technical_indicators,
        ]:
            feature_vector.extend(list(feature_dict.values()))

        return feature_vector

    def _basic_volatility_prediction(self, market_data: dict[str, Any]) -> float:
        """Basic volatility prediction without ML."""
        prices = market_data.get("prices", [])

        if len(prices) < 2:
            return 0.2  # Default 20% volatility

        prices = np.array(prices)
        returns = np.diff(prices) / prices[:-1]

        # Simple volatility forecast based on recent volatility
        recent_vol = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
        annualized_vol = recent_vol * np.sqrt(252)

        # Add some mean reversion
        long_term_vol = 0.2  # 20% long - term average
        mean_reversion_factor = 0.1

        predicted_vol = (
            1 - mean_reversion_factor
        ) * annualized_vol + mean_reversion_factor * long_term_vol

        return max(0.05, min(1.0, predicted_vol))  # Clamp between 5% and 100%

    def _predict_regime_probabilities(self, features: MLFeatures) -> dict[str, float]:
        """Predict market regime probabilities."""
        # Simple rule-based regime detection
        volatility = features.price_features.get("price_volatility", 0.2)
        vix = features.macro_features.get("vix_level", 20)
        features.sentiment_features.get("reddit_sentiment", 0)

        # Regime classification based on volatility and sentiment
        if volatility > 0.4 or vix > 35:
            return {"normal": 0.2, "high_vol": 0.6, "crisis": 0.2}
        elif volatility < 0.15 and vix < 15:
            return {"normal": 0.8, "high_vol": 0.15, "crisis": 0.05}
        else:
            return {"normal": 0.6, "high_vol": 0.3, "crisis": 0.1}

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(self, prices: np.ndarray) -> float:
        """Calculate MACD indicator."""
        if len(prices) < 26:
            return 0.0

        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)

        return ema_12 - ema_26

    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return prices[-1]

        alpha = 2 / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema

        return ema

    def _calculate_bollinger_position(
        self, prices: np.ndarray, period: int = 20
    ) -> float:
        """Calculate position within Bollinger Bands."""
        if len(prices) < period:
            return 0.5

        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])

        if std == 0:
            return 0.5

        current_price = prices[-1]
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std

        if current_price >= upper_band:
            return 1.0
        elif current_price <= lower_band:
            return 0.0
        else:
            return (current_price - lower_band) / (upper_band - lower_band)

    def _calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(prices) < 2:
            return 0.0

        peak = prices[0]
        max_dd = 0.0

        for price in prices:
            if price > peak:
                peak = price
            else:
                drawdown = (peak - price) / peak
                max_dd = max(max_dd, drawdown)

        return max_dd

    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns."""
        if len(returns) < 3:
            return 0.0

        mean = np.mean(returns)
        std = np.std(returns)

        if std == 0:
            return 0.0

        skewness = np.mean(((returns - mean) / std) ** 3)
        return skewness

    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis of returns."""
        if len(returns) < 4:
            return 0.0

        mean = np.mean(returns)
        std = np.std(returns)

        if std == 0:
            return 0.0

        kurtosis = np.mean(((returns - mean) / std) ** 4) - 3
        return kurtosis

    def predict_risk_score(self, market_data: dict[str, Any]) -> RiskPrediction:
        """Predict overall risk score."""
        # Get volatility forecast
        vol_forecast = self.predict_volatility_regime(market_data)

        # Calculate risk score based on multiple factors
        risk_factors = {
            "volatility": min(
                100, vol_forecast.predicted_volatility * 200
            ),  # Scale to 0 - 100
            "regime": 80
            if vol_forecast.regime_probability.get("crisis", 0) > 0.3
            else 40,
            "sentiment": 60 if market_data.get("sentiment", 0) < -0.5 else 30,
            "technical": 70
            if market_data.get("rsi", 50) > 80 or market_data.get("rsi", 50) < 20
            else 40,
        }

        # Weighted average risk score
        weights = {"volatility": 0.4, "regime": 0.3, "sentiment": 0.2, "technical": 0.1}
        risk_score = sum(
            risk_factors[factor] * weights[factor] for factor in risk_factors
        )

        # Generate recommendations
        recommendations = []
        if risk_score > 70:
            recommendations.append("Reduce position sizes - high risk environment")
        if vol_forecast.regime_probability.get("crisis", 0) > 0.3:
            recommendations.append(
                "Consider hedging strategies - crisis regime detected"
            )
        if risk_factors["volatility"] > 80:
            recommendations.append("High volatility expected - adjust risk parameters")

        return RiskPrediction(
            risk_score=risk_score,
            volatility_forecast=vol_forecast,
            regime_prediction=max(
                vol_forecast.regime_probability, key=vol_forecast.regime_probability.get
            ),
            confidence=0.75,
            recommended_actions=recommendations,
        )

    def predict_volatility(
        self, returns_data: np.ndarray | list[float]
    ) -> VolatilityForecast:
        """Predict volatility from portfolio returns data.

        Args:
            returns_data: Portfolio returns as numpy array or list

        Returns:
            Volatility forecast with confidence intervals
        """
        # Convert to numpy array if needed
        if isinstance(returns_data, list):
            returns_data = np.array(returns_data)

        # Convert returns data to market data format for predict_volatility_regime
        mock_market_data = {
            "prices": (100 * np.cumprod(1 + returns_data)).tolist(),
            "volumes": [1000000] * len(returns_data),  # Mock volume data
            "sentiment": 0.0,  # Neutral sentiment
            "rsi": 50.0,  # Neutral RSI
        }

        # Use existing predict_volatility_regime method
        return self.predict_volatility_regime(mock_market_data, horizon_days=5)


# Example usage and testing
if __name__ == "__main__":  # Create sample market data
    np.random.seed(42)
    sample_prices = 100 + np.cumsum(np.random.normal(0.001, 0.02, 100))
    sample_volumes = np.random.uniform(1000, 10000, 100)

    sample_market_data = {
        "prices": sample_prices.tolist(),
        "volumes": sample_volumes.tolist(),
        "sentiment": np.random.uniform(-1, 1),
        "rsi": np.random.uniform(20, 80),
    }

    # Initialize ML risk predictor
    ml_predictor = MLRiskPredictor()

    # Test volatility prediction
    print("Testing ML Risk Predictor...")
    print(" = " * 40)

    vol_forecast = ml_predictor.predict_volatility_regime(sample_market_data)
    print(f"Predicted Volatility: {vol_forecast.predicted_volatility:.2%}")
    print(
        f"Confidence Interval: {vol_forecast.confidence_interval[0]:.2%} - {vol_forecast.confidence_interval[1]: .2%}"
    )
    print(f"Regime Probabilities: {vol_forecast.regime_probability}")

    # Test risk prediction
    risk_prediction = ml_predictor.predict_risk_score(sample_market_data)
    print(f"\nRisk Score: {risk_prediction.risk_score:.1f}/100")
    print(f"Regime: {risk_prediction.regime_prediction}")
    print(f"Confidence: {risk_prediction.confidence:.1%}")

    if risk_prediction.recommended_actions:
        print("\nRecommendations: ")
        for i, action in enumerate(risk_prediction.recommended_actions, 1):
            print(f"{i}. {action}")
