"""Tests for the production narrative rerate strategy."""

from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest

from backend.tradingbot.core.trading_interface import TradeStatus
from backend.tradingbot.production.core.production_integration import TradeResult
from backend.tradingbot.strategies.production.production_narrative_rerate import (
    NarrativeRerateSignal,
    ProductionNarrativeRerateStrategy,
)


@pytest.fixture
def mock_data_provider():
    provider = AsyncMock()
    provider.is_market_open = AsyncMock(return_value=True)
    provider.get_price_history = AsyncMock(
        return_value=[
            *[Decimal("40") for _ in range(20)],
            *[Decimal(str(40 + i)) for i in range(20)],
            *[Decimal("80") for _ in range(10)],
            *[Decimal("58") for _ in range(10)],
            *[Decimal(str(58 + i * 0.8)) for i in range(30)],
        ]
    )
    provider.get_volume_history = AsyncMock(return_value=[1_500_000] * 30)
    quote = Mock()
    quote.price = Decimal("81.20")
    provider.get_current_price = AsyncMock(return_value=quote)
    provider.get_news_items = AsyncMock(
        return_value=[
            {
                "title": "Figma revenue guidance rises as AI product adoption improves",
                "summary": "Investors cite ai product, revenue growth, and retention.",
                "sentiment_score": 0.72,
            },
            {
                "title": "Analysts say software drawdown fears look overdone",
                "summary": "Margin and guidance commentary support rebound narrative.",
                "sentiment_score": 0.54,
            },
            {
                "title": "Product teams expand Figma usage",
                "summary": "Retention and ai product workflow remain in focus.",
                "sentiment_score": 0.61,
            },
        ]
    )
    return provider


@pytest.fixture
def mock_integration_manager():
    manager = AsyncMock()
    manager.get_account_info = AsyncMock(
        return_value={"cash": 100_000.0, "equity": 100_000.0, "buying_power": 100_000.0}
    )
    trade_result = TradeResult(
        trade_id="narrative_order_1",
        signal=Mock(),
        status=TradeStatus.FILLED,
        filled_quantity=100,
        filled_price=81.20,
        commission=0.0,
        error_message=None,
    )
    manager.execute_trade = AsyncMock(return_value=trade_result)
    return manager


@pytest.fixture
def strategy(mock_integration_manager, mock_data_provider):
    return ProductionNarrativeRerateStrategy(
        mock_integration_manager,
        mock_data_provider,
        {
            "universe": ["FIG", "NVDA"],
            "min_composite_score": 60,
            "max_position_size": 0.15,
            "max_total_exposure": 0.50,
        },
    )


@pytest.mark.asyncio
async def test_scan_returns_ranked_thesis_ready_signal(strategy):
    signals = await strategy.scan_narrative_opportunities()

    assert signals
    signal = signals[0]
    assert isinstance(signal, NarrativeRerateSignal)
    assert signal.ticker in {"FIG", "NVDA"}
    assert signal.composite_score >= 60
    assert signal.target_weight <= 0.15
    assert "rerate" in signal.thesis
    assert "Exit" in signal.invalidation
    assert signal.trim_plan.first_trim_gain == 0.25
    assert signal.metadata["mention_count"] == 3


@pytest.mark.asyncio
async def test_execute_signal_sizes_position_and_preserves_trade_plan(
    strategy, mock_integration_manager
):
    signal = await strategy.evaluate_ticker("FIG")

    executed = await strategy.execute_signal(signal)

    assert executed is True
    assert "FIG" in strategy.active_positions
    trade_signal = mock_integration_manager.execute_trade.call_args.args[0]
    assert trade_signal.strategy_name == "narrative_rerate"
    assert trade_signal.trade_type == "stock"
    assert trade_signal.quantity > 0
    assert trade_signal.metadata["thesis"] == signal.thesis
    assert trade_signal.metadata["trim_plan"]["first_trim_gain"] == 0.25
    assert trade_signal.metadata["scores"]["composite"] == signal.composite_score


@pytest.mark.asyncio
async def test_should_exit_position_handles_invalidation_and_trims(
    strategy, mock_data_provider
):
    signal = await strategy.evaluate_ticker("FIG")
    position = {
        "signal": signal,
        "quantity": 100,
        "entry_price": Decimal("80.00"),
        "market_value": Decimal("8000.00"),
        "trimmed_levels": set(),
    }

    quote = Mock()
    quote.price = signal.invalidation_price - Decimal("0.01")
    mock_data_provider.get_current_price = AsyncMock(return_value=quote)
    exit_decision = await strategy.should_exit_position(position)
    assert exit_decision["action"] == "exit"
    assert exit_decision["reason"] == "INVALIDATION_PRICE"

    quote.price = Decimal("101.00")
    trim_decision = await strategy.should_exit_position(position)
    assert trim_decision["action"] == "trim"
    assert trim_decision["reason"] == "FIRST_TRIM"
    assert trim_decision["trim_fraction"] == 0.25
