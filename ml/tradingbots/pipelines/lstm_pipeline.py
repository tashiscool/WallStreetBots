"""
LSTM Price Prediction Pipeline

Integrates the LSTM price predictor with the trading framework for
automated trading decisions based on deep learning predictions.
"""

import datetime
from datetime import timedelta
from typing import Optional, Dict, Any, List

try:
    from backend.settings import BACKEND_ALPACA_ID, BACKEND_ALPACA_KEY
except ImportError:
    import os
    BACKEND_ALPACA_ID = os.getenv('BACKEND_ALPACA_ID', '')
    BACKEND_ALPACA_KEY = os.getenv('BACKEND_ALPACA_KEY', '')

from .pipeline import Pipeline
from ..components.lstm_predictor import LSTMPricePredictor, LSTMConfig, LSTMEnsemble
from ..components.utils import AlpacaFetcher


class LSTMPortfolioManager:
    """
    Portfolio manager that uses LSTM predictions for rebalancing decisions.

    Similar to NaiveHMMPortfolioUpdate but uses deep learning for
    trend prediction instead of Hidden Markov Models.
    """

    def __init__(
        self,
        portfolio: Dict[str, Any],
        data_fetcher: AlpacaFetcher,
        config: Optional[LSTMConfig] = None,
        buffer: float = 0.05,
        use_ensemble: bool = False,
        ensemble_size: int = 3,
        min_training_days: int = 90,
        prediction_threshold: float = 0.5,
    ):
        """
        Initialize the LSTM portfolio manager.

        Args:
            portfolio: Dictionary with 'cash' and 'stocks' keys
            data_fetcher: Data fetcher for price data
            config: LSTM configuration
            buffer: Cash buffer proportion
            use_ensemble: Whether to use ensemble predictions
            ensemble_size: Number of models in ensemble
            min_training_days: Minimum days of data for training
            prediction_threshold: Confidence threshold for predictions
        """
        self.portfolio = portfolio
        self.portfolio_cash = portfolio.get("cash", 0)
        self.portfolio_stocks = portfolio.get("stocks", {})
        self.data_fetcher = data_fetcher
        self.config = config or LSTMConfig()
        self.buffer = buffer
        self.use_ensemble = use_ensemble
        self.ensemble_size = ensemble_size
        self.min_training_days = min_training_days
        self.prediction_threshold = prediction_threshold

        # Initialize models
        self.models: Dict[str, LSTMPricePredictor] = {}
        self.ensembles: Dict[str, LSTMEnsemble] = {}

        # Calculate portfolio metrics
        self.price_dict: Dict[str, float] = {}
        self.total_portfolio_value = 0.0
        self._calculate_portfolio_value()

    def _calculate_portfolio_value(self) -> None:
        """Calculate current portfolio value."""
        self.total_portfolio_value = self.portfolio_cash
        for ticker, qty in self.portfolio_stocks.items():
            try:
                price = self.data_fetcher.get_cur_price(ticker)
                self.price_dict[ticker] = price
                self.total_portfolio_value += price * qty
            except Exception:
                self.price_dict[ticker] = 0

    def train_models(self, verbose: bool = True) -> Dict[str, Dict]:
        """
        Train LSTM models for all stocks in portfolio.

        Returns:
            Dictionary of training metrics per symbol
        """
        training_results = {}

        for ticker in self.portfolio_stocks.keys():
            if verbose:
                print(f"Training LSTM for {ticker}...")

            try:
                # Fetch historical data
                end = datetime.datetime.now(datetime.UTC)
                start = end - timedelta(days=self.min_training_days)
                prices = self.data_fetcher.get_historical_prices(
                    ticker, start.isoformat(), end.isoformat()
                )

                if len(prices) < self.config.seq_length + 50:
                    if verbose:
                        print(f"  Insufficient data for {ticker}")
                    training_results[ticker] = {"error": "insufficient_data"}
                    continue

                # Train model
                if self.use_ensemble:
                    ensemble = LSTMEnsemble(self.ensemble_size, self.config)
                    metrics = ensemble.train(prices, verbose=verbose)
                    self.ensembles[ticker] = ensemble
                else:
                    model = LSTMPricePredictor(self.config)
                    metrics = model.train(prices, verbose=verbose)
                    self.models[ticker] = model

                training_results[ticker] = metrics

            except Exception as e:
                training_results[ticker] = {"error": str(e)}
                if verbose:
                    print(f"  Error training {ticker}: {e}")

        return training_results

    def predict_trends(self) -> Dict[str, Dict[str, Any]]:
        """
        Predict trends for all stocks in portfolio.

        Returns:
            Dictionary with trend predictions per symbol
        """
        predictions = {}

        for ticker in self.portfolio_stocks.keys():
            try:
                # Get recent prices
                end = datetime.datetime.now(datetime.UTC)
                start = end - timedelta(days=self.config.seq_length + 10)
                prices = self.data_fetcher.get_historical_prices(
                    ticker, start.isoformat(), end.isoformat()
                )

                if len(prices) < self.config.seq_length:
                    predictions[ticker] = {
                        "trend": "unknown",
                        "confidence": 0.0,
                        "error": "insufficient_data"
                    }
                    continue

                if self.use_ensemble and ticker in self.ensembles:
                    trend, confidence = self.ensembles[ticker].predict_trend(prices)
                    predicted_price, uncertainty = self.ensembles[ticker].predict(prices)
                    predictions[ticker] = {
                        "trend": trend,
                        "confidence": confidence,
                        "predicted_price": predicted_price,
                        "uncertainty": uncertainty
                    }
                elif ticker in self.models:
                    trend = self.models[ticker].predict_trend(prices)
                    confidence = self.models[ticker].get_trend_confidence(prices)
                    predicted_price = self.models[ticker].predict(prices)
                    predictions[ticker] = {
                        "trend": trend,
                        "confidence": confidence,
                        "predicted_price": predicted_price
                    }
                else:
                    predictions[ticker] = {
                        "trend": "unknown",
                        "confidence": 0.0,
                        "error": "model_not_trained"
                    }

            except Exception as e:
                predictions[ticker] = {
                    "trend": "unknown",
                    "confidence": 0.0,
                    "error": str(e)
                }

        return predictions

    def rebalance(self) -> Dict[str, float]:
        """
        Rebalance portfolio based on LSTM predictions.

        Returns:
            New portfolio allocation
        """
        # Get predictions
        predictions = self.predict_trends()

        # Find stocks predicted to go up with high confidence
        buy_list = []
        for ticker, pred in predictions.items():
            if (pred.get("trend") == "up" and
                    pred.get("confidence", 0) >= self.prediction_threshold):
                buy_list.append((ticker, pred.get("confidence", 0)))

        # Sort by confidence
        buy_list.sort(key=lambda x: x[1], reverse=True)

        # Build new portfolio
        new_portfolio = {}
        if buy_list:
            available_capital = self.total_portfolio_value * (1 - self.buffer)
            capital_per_stock = available_capital / len(buy_list)

            for ticker, _ in buy_list:
                price = self.price_dict.get(ticker, 0)
                if price > 0:
                    qty = round(capital_per_stock / price, 2)
                    new_portfolio[ticker] = qty

        return new_portfolio


class LSTMPipeline(Pipeline):
    """
    Pipeline for LSTM-based trading decisions.

    Extends the base Pipeline class to use deep learning
    predictions for portfolio rebalancing.
    """

    def __init__(
        self,
        name: str,
        portfolio: Dict[str, Any],
        config: Optional[LSTMConfig] = None,
        use_ensemble: bool = False,
        buffer: float = 0.05,
        prediction_threshold: float = 0.5,
    ):
        """
        Initialize the LSTM pipeline.

        Args:
            name: Pipeline name
            portfolio: Portfolio dictionary
            config: LSTM configuration
            use_ensemble: Use ensemble predictions
            buffer: Cash buffer proportion
            prediction_threshold: Confidence threshold
        """
        super().__init__(name, portfolio)
        self.config = config or LSTMConfig()
        self.use_ensemble = use_ensemble
        self.buffer = buffer
        self.prediction_threshold = prediction_threshold
        self.portfolio_manager: Optional[LSTMPortfolioManager] = None

    def pipeline(self) -> Dict[str, float]:
        """
        Execute the LSTM pipeline.

        Returns:
            New portfolio allocation
        """
        # Initialize data fetcher
        data_fetcher = AlpacaFetcher(BACKEND_ALPACA_ID, BACKEND_ALPACA_KEY)

        # Create portfolio manager
        self.portfolio_manager = LSTMPortfolioManager(
            portfolio=self.portfolio,
            data_fetcher=data_fetcher,
            config=self.config,
            buffer=self.buffer,
            use_ensemble=self.use_ensemble,
            prediction_threshold=self.prediction_threshold,
        )

        # Train models
        self.portfolio_manager.train_models(verbose=False)

        # Get rebalanced portfolio
        return self.portfolio_manager.rebalance()

    def get_predictions(self) -> Dict[str, Dict[str, Any]]:
        """Get current predictions for all stocks."""
        if self.portfolio_manager is None:
            return {}
        return self.portfolio_manager.predict_trends()


class LSTMSignalGenerator:
    """
    Generates trading signals based on LSTM predictions.

    Can be used standalone or integrated with the signal
    validation framework.
    """

    def __init__(
        self,
        config: Optional[LSTMConfig] = None,
        data_fetcher: Optional[AlpacaFetcher] = None,
    ):
        """
        Initialize the signal generator.

        Args:
            config: LSTM configuration
            data_fetcher: Optional data fetcher
        """
        self.config = config or LSTMConfig()
        self.data_fetcher = data_fetcher or AlpacaFetcher(
            BACKEND_ALPACA_ID, BACKEND_ALPACA_KEY
        )
        self.models: Dict[str, LSTMPricePredictor] = {}

    def train(self, symbol: str, prices: List[float]) -> Dict[str, Any]:
        """Train a model for a specific symbol."""
        model = LSTMPricePredictor(self.config)
        metrics = model.train(prices, verbose=False)
        self.models[symbol] = model
        return metrics

    def generate_signal(self, symbol: str, prices: List[float]) -> Dict[str, Any]:
        """
        Generate a trading signal for a symbol.

        Args:
            symbol: Stock symbol
            prices: Recent price history

        Returns:
            Signal dictionary with trend, confidence, and recommendation
        """
        if symbol not in self.models:
            return {
                "symbol": symbol,
                "signal": "none",
                "confidence": 0.0,
                "error": "model_not_trained"
            }

        model = self.models[symbol]

        try:
            trend = model.predict_trend(prices)
            confidence = model.get_trend_confidence(prices)
            predicted_price = model.predict(prices)
            current_price = prices[-1]
            pct_change = (predicted_price - current_price) / current_price * 100

            # Generate signal
            if trend == "up" and confidence >= 0.6:
                signal = "strong_buy"
            elif trend == "up" and confidence >= 0.4:
                signal = "buy"
            elif trend == "down" and confidence >= 0.6:
                signal = "strong_sell"
            elif trend == "down" and confidence >= 0.4:
                signal = "sell"
            else:
                signal = "hold"

            return {
                "symbol": symbol,
                "signal": signal,
                "trend": trend,
                "confidence": confidence,
                "predicted_price": predicted_price,
                "current_price": current_price,
                "expected_change_pct": pct_change,
            }

        except Exception as e:
            return {
                "symbol": symbol,
                "signal": "none",
                "confidence": 0.0,
                "error": str(e)
            }
