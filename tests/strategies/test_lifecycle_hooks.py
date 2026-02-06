"""Tests for Strategy Lifecycle Mixin (Phase 6)."""
import pytest

from backend.tradingbot.strategies.base.lifecycle_mixin import LifecycleMixin


class ConcreteLifecycleStrategy(LifecycleMixin):
    """Concrete strategy that uses LifecycleMixin for testing."""

    def __init__(self):
        self._call_order = []
        super().__init__()

    def initialize(self):
        self._call_order.append("initialize")

    def before_market_opens(self):
        self._call_order.append("before_market_opens")

    def on_trading_iteration(self):
        self._call_order.append("on_trading_iteration")

    def after_market_closes(self):
        self._call_order.append("after_market_closes")

    def before_starting_trading(self):
        self._call_order.append("before_starting_trading")

    def on_abrupt_closing(self):
        self._call_order.append("on_abrupt_closing")

    def trace_stats(self):
        return {"pnl": 100.0, "positions": 3}


class TestLifecycleMixin:
    def test_init(self):
        strategy = ConcreteLifecycleStrategy()
        assert strategy._lifecycle_initialized is False
        assert isinstance(strategy._stats_history, list)
        assert isinstance(strategy._lifecycle_hooks, dict)

    def test_default_methods_dont_raise(self):
        mixin = LifecycleMixin()
        mixin.initialize()
        mixin.before_market_opens()
        mixin.on_trading_iteration()
        mixin.after_market_closes()
        mixin.before_starting_trading()
        mixin.on_abrupt_closing()
        assert mixin.trace_stats() == {}

    def test_run_lifecycle_cycle_first_call(self):
        strategy = ConcreteLifecycleStrategy()
        strategy.run_lifecycle_cycle()

        # First call should trigger initialize + before_starting_trading
        assert "initialize" in strategy._call_order
        assert "before_starting_trading" in strategy._call_order
        assert strategy._lifecycle_initialized is True
        # Then the regular cycle
        assert "before_market_opens" in strategy._call_order
        assert "on_trading_iteration" in strategy._call_order
        assert "after_market_closes" in strategy._call_order

    def test_run_lifecycle_cycle_second_call(self):
        strategy = ConcreteLifecycleStrategy()
        strategy.run_lifecycle_cycle()
        strategy._call_order.clear()
        strategy.run_lifecycle_cycle()

        # Second call should NOT re-initialize
        assert "initialize" not in strategy._call_order
        assert "before_starting_trading" not in strategy._call_order
        # But regular cycle should still run
        assert "before_market_opens" in strategy._call_order
        assert "on_trading_iteration" in strategy._call_order
        assert "after_market_closes" in strategy._call_order

    def test_lifecycle_order(self):
        strategy = ConcreteLifecycleStrategy()
        strategy.run_lifecycle_cycle()

        expected_order = [
            "initialize",
            "before_starting_trading",
            "before_market_opens",
            "on_trading_iteration",
            "after_market_closes",
        ]
        assert strategy._call_order == expected_order

    def test_trace_stats_recorded(self):
        strategy = ConcreteLifecycleStrategy()
        strategy.run_lifecycle_cycle()

        stats = strategy.get_stats_history()
        assert len(stats) == 1
        assert stats[0]["pnl"] == 100.0
        assert stats[0]["positions"] == 3
        assert "timestamp" in stats[0]

    def test_multiple_cycles_accumulate_stats(self):
        strategy = ConcreteLifecycleStrategy()
        strategy.run_lifecycle_cycle()
        strategy.run_lifecycle_cycle()
        strategy.run_lifecycle_cycle()

        stats = strategy.get_stats_history()
        assert len(stats) == 3

    def test_register_hook(self):
        strategy = ConcreteLifecycleStrategy()
        hook_calls = []

        strategy.register_hook("on_trading_iteration", lambda: hook_calls.append("hook1"))
        strategy.register_hook("on_trading_iteration", lambda: hook_calls.append("hook2"))
        strategy.run_lifecycle_cycle()

        assert "hook1" in hook_calls
        assert "hook2" in hook_calls

    def test_register_hook_invalid_event(self):
        strategy = ConcreteLifecycleStrategy()
        # Should not raise, just warn
        strategy.register_hook("nonexistent_event", lambda: None)
        assert "nonexistent_event" not in strategy._lifecycle_hooks

    def test_hook_error_doesnt_crash(self):
        strategy = ConcreteLifecycleStrategy()

        def bad_hook():
            raise ValueError("hook error")

        strategy.register_hook("on_trading_iteration", bad_hook)
        # Should not raise
        strategy.run_lifecycle_cycle()
        # Regular methods should still have executed
        assert "on_trading_iteration" in strategy._call_order

    def test_method_error_doesnt_crash(self):
        class FailingStrategy(LifecycleMixin):
            def before_market_opens(self):
                raise RuntimeError("something went wrong")

            def on_trading_iteration(self):
                pass

        strategy = FailingStrategy()
        # _run_lifecycle should catch the error
        strategy._run_lifecycle("before_market_opens")
        # Should not have crashed

    def test_on_abrupt_closing(self):
        strategy = ConcreteLifecycleStrategy()
        strategy._run_lifecycle("on_abrupt_closing")
        assert "on_abrupt_closing" in strategy._call_order
