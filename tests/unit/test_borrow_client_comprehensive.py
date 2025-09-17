"""Comprehensive tests for Borrow Client to achieve >70% coverage."""
import pytest
from dataclasses import asdict

from backend.tradingbot.borrow.client import LocateQuote, BorrowClient, guard_can_short


class TestLocateQuote:
    """Test LocateQuote dataclass."""

    def test_locate_quote_creation_basic(self):
        """Test basic LocateQuote creation."""
        quote = LocateQuote(
            symbol="AAPL",
            available=True,
            borrow_bps=25.0
        )

        assert quote.symbol == "AAPL"
        assert quote.available is True
        assert quote.borrow_bps == 25.0
        assert quote.reason is None

    def test_locate_quote_creation_with_reason(self):
        """Test LocateQuote creation with reason."""
        quote = LocateQuote(
            symbol="PENNY",
            available=False,
            borrow_bps=0.0,
            reason="Hard to borrow"
        )

        assert quote.symbol == "PENNY"
        assert quote.available is False
        assert quote.borrow_bps == 0.0
        assert quote.reason == "Hard to borrow"

    def test_locate_quote_dataclass_features(self):
        """Test dataclass features of LocateQuote."""
        quote = LocateQuote("MSFT", True, 30.0, "Standard rate")

        # Test string representation
        str_repr = str(quote)
        assert "MSFT" in str_repr
        assert "True" in str_repr
        assert "30.0" in str_repr

        # Test equality
        quote2 = LocateQuote("MSFT", True, 30.0, "Standard rate")
        assert quote == quote2

        # Test inequality
        quote3 = LocateQuote("MSFT", False, 30.0, "Standard rate")
        assert quote != quote3

    def test_locate_quote_conversion_to_dict(self):
        """Test converting LocateQuote to dictionary."""
        quote = LocateQuote("GOOGL", True, 35.5, "Tech stock")
        quote_dict = asdict(quote)

        expected = {
            "symbol": "GOOGL",
            "available": True,
            "borrow_bps": 35.5,
            "reason": "Tech stock"
        }
        assert quote_dict == expected

    def test_locate_quote_edge_cases(self):
        """Test LocateQuote with edge case values."""
        # Test with zero borrow rate
        quote_zero = LocateQuote("FREE", True, 0.0)
        assert quote_zero.borrow_bps == 0.0

        # Test with high borrow rate
        quote_high = LocateQuote("EXPENSIVE", True, 10000.0)
        assert quote_high.borrow_bps == 10000.0

        # Test with negative borrow rate (rebate)
        quote_rebate = LocateQuote("REBATE", True, -50.0)
        assert quote_rebate.borrow_bps == -50.0

    def test_locate_quote_different_symbols(self):
        """Test LocateQuote with different symbol formats."""
        symbols = ["AAPL", "SPY", "TSLA", "BRK.A", "BRK.B", "ARKK", "QQQ"]

        for symbol in symbols:
            quote = LocateQuote(symbol, True, 25.0)
            assert quote.symbol == symbol

    def test_locate_quote_boolean_combinations(self):
        """Test LocateQuote with different boolean combinations."""
        # Available with borrow cost
        quote1 = LocateQuote("STOCK1", True, 50.0)
        assert quote1.available is True

        # Unavailable with zero cost
        quote2 = LocateQuote("STOCK2", False, 0.0)
        assert quote2.available is False

        # Edge case: unavailable but with cost (shouldn't happen in practice)
        quote3 = LocateQuote("STOCK3", False, 100.0)
        assert quote3.available is False
        assert quote3.borrow_bps == 100.0

    def test_locate_quote_reason_variations(self):
        """Test LocateQuote with different reason values."""
        reasons = [
            "Hard to borrow",
            "Microcap restriction",
            "No inventory",
            "Regulatory restriction",
            "Temporarily unavailable",
            "",  # Empty string
            None  # None value
        ]

        for reason in reasons:
            quote = LocateQuote("TEST", False, 0.0, reason)
            assert quote.reason == reason


class TestBorrowClient:
    """Test BorrowClient class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = BorrowClient()

    def test_borrow_client_instantiation(self):
        """Test BorrowClient can be instantiated."""
        client = BorrowClient()
        assert isinstance(client, BorrowClient)

    def test_locate_large_cap_stock(self):
        """Test locate for large cap stocks (short symbols)."""
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]

        for symbol in symbols:
            quote = self.client.locate(symbol, 1000.0)
            assert isinstance(quote, LocateQuote)
            assert quote.symbol == symbol.upper()
            assert quote.available is True
            assert quote.borrow_bps == 30.0
            assert quote.reason is None

    def test_locate_microcap_stocks(self):
        """Test locate for microcap stocks (long symbols)."""
        microcap_symbols = ["MICROCAP", "PENNYSTOCK", "SMALLCAP123", "LONGNAME"]

        for symbol in microcap_symbols:
            quote = self.client.locate(symbol, 1000.0)
            assert isinstance(quote, LocateQuote)
            assert quote.symbol == symbol.upper()
            assert quote.available is False
            assert quote.borrow_bps == 0.0
            assert quote.reason == "Microcap/no locate"

    def test_locate_case_sensitivity(self):
        """Test locate handles case sensitivity correctly."""
        test_cases = [
            ("aapl", "AAPL"),
            ("AAPL", "AAPL"),
            ("AaPl", "AAPL"),
            ("msft", "MSFT"),
            ("TsLa", "TSLA")
        ]

        for input_symbol, expected_symbol in test_cases:
            quote = self.client.locate(input_symbol, 100.0)
            assert quote.symbol == expected_symbol

    def test_locate_quantity_variations(self):
        """Test locate with different quantities."""
        quantities = [1.0, 10.0, 100.0, 1000.0, 10000.0, 0.5, 0.01]

        for qty in quantities:
            quote = self.client.locate("AAPL", qty)
            assert isinstance(quote, LocateQuote)
            # Quantity doesn't affect the result in current implementation
            assert quote.available is True
            assert quote.borrow_bps == 30.0

    def test_locate_boundary_symbol_lengths(self):
        """Test locate with symbols at the boundary of length restrictions."""
        # Symbols with exactly 5 characters (should be available)
        five_char_symbols = ["AAAAA", "BBBBB", "APPLE", "MICRO"]

        for symbol in five_char_symbols:
            quote = self.client.locate(symbol, 1000.0)
            assert quote.available is True
            assert quote.borrow_bps == 30.0
            assert quote.reason is None

        # Symbols with exactly 6 characters (should be unavailable)
        six_char_symbols = ["AAAAAA", "BBBBBB", "MICROS", "PENNY1"]

        for symbol in six_char_symbols:
            quote = self.client.locate(symbol, 1000.0)
            assert quote.available is False
            assert quote.borrow_bps == 0.0
            assert quote.reason == "Microcap/no locate"

    def test_locate_edge_case_symbols(self):
        """Test locate with edge case symbols."""
        edge_cases = [
            ("", True),  # Empty string (length 0 <= 5)
            ("A", True),  # Single character
            ("AB", True),  # Two characters
            ("ABC", True),  # Three characters
            ("ABCD", True),  # Four characters
            ("ABCDE", True),  # Five characters (boundary)
            ("ABCDEF", False),  # Six characters (over boundary)
        ]

        for symbol, should_be_available in edge_cases:
            quote = self.client.locate(symbol, 100.0)
            assert quote.available == should_be_available
            if should_be_available:
                assert quote.borrow_bps == 30.0
                assert quote.reason is None
            else:
                assert quote.borrow_bps == 0.0
                assert quote.reason == "Microcap/no locate"

    def test_locate_special_characters(self):
        """Test locate with symbols containing special characters."""
        # Length 5 symbols (should be available)
        short_special_symbols = ["BRK.A", "BRK.B", "BRK-A", "BRK_A"]

        for symbol in short_special_symbols:
            quote = self.client.locate(symbol, 1000.0)
            # These have length = 5, so should be available
            assert quote.available is True
            assert quote.borrow_bps == 30.0
            assert quote.reason is None

        # Length > 5 symbols (should be unavailable)
        long_special_symbols = ["BRK.AB", "STOCK-1", "SYMBOL_LONG"]

        for symbol in long_special_symbols:
            quote = self.client.locate(symbol, 1000.0)
            # These have length > 5, so should be unavailable
            assert quote.available is False
            assert quote.borrow_bps == 0.0
            assert quote.reason == "Microcap/no locate"

    def test_locate_numeric_symbols(self):
        """Test locate with numeric symbols."""
        numeric_symbols = ["123", "1234", "12345", "123456"]

        for symbol in numeric_symbols:
            quote = self.client.locate(symbol, 1000.0)
            expected_available = len(symbol) <= 5
            assert quote.available == expected_available

    def test_locate_consistency(self):
        """Test locate returns consistent results for same inputs."""
        symbol = "AAPL"
        qty = 1000.0

        # Call locate multiple times with same parameters
        quotes = [self.client.locate(symbol, qty) for _ in range(5)]

        # All results should be identical
        first_quote = quotes[0]
        for quote in quotes[1:]:
            assert quote.symbol == first_quote.symbol
            assert quote.available == first_quote.available
            assert quote.borrow_bps == first_quote.borrow_bps
            assert quote.reason == first_quote.reason


class TestGuardCanShort:
    """Test guard_can_short function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = BorrowClient()

    def test_guard_can_short_success(self):
        """Test guard_can_short with available stocks."""
        symbols = ["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"]

        for symbol in symbols:
            borrow_rate = guard_can_short(self.client, symbol, 1000.0)
            assert borrow_rate == 30.0

    def test_guard_can_short_failure(self):
        """Test guard_can_short with unavailable stocks."""
        microcap_symbols = ["MICROCAP", "PENNYSTOCK", "LONGNAME"]

        for symbol in microcap_symbols:
            with pytest.raises(PermissionError) as exc_info:
                guard_can_short(self.client, symbol, 1000.0)

            assert "Locate failed" in str(exc_info.value)
            assert symbol in str(exc_info.value)
            assert "Microcap/no locate" in str(exc_info.value)

    def test_guard_can_short_different_quantities(self):
        """Test guard_can_short with different quantities."""
        quantities = [1.0, 10.0, 100.0, 1000.0, 5000.0]

        for qty in quantities:
            # Should succeed for large cap
            borrow_rate = guard_can_short(self.client, "AAPL", qty)
            assert borrow_rate == 30.0

            # Should fail for microcap
            with pytest.raises(PermissionError):
                guard_can_short(self.client, "MICROCAP", qty)

    def test_guard_can_short_case_sensitivity(self):
        """Test guard_can_short handles case sensitivity."""
        # Should succeed
        borrow_rate = guard_can_short(self.client, "aapl", 1000.0)
        assert borrow_rate == 30.0

        # Should fail
        with pytest.raises(PermissionError):
            guard_can_short(self.client, "microcap", 1000.0)

    def test_guard_can_short_boundary_cases(self):
        """Test guard_can_short with boundary symbol lengths."""
        # 5 characters - should succeed
        borrow_rate = guard_can_short(self.client, "AAAAA", 1000.0)
        assert borrow_rate == 30.0

        # 6 characters - should fail
        with pytest.raises(PermissionError):
            guard_can_short(self.client, "AAAAAA", 1000.0)

    def test_guard_can_short_error_message_format(self):
        """Test guard_can_short error message format."""
        with pytest.raises(PermissionError) as exc_info:
            guard_can_short(self.client, "LONGSTOCK", 500.0)

        error_message = str(exc_info.value)
        assert error_message.startswith("Locate failed for LONGSTOCK:")
        assert "Microcap/no locate" in error_message

    def test_guard_can_short_return_type(self):
        """Test guard_can_short returns correct type."""
        result = guard_can_short(self.client, "AAPL", 1000.0)
        assert isinstance(result, float)
        assert result == 30.0

    def test_guard_can_short_multiple_calls(self):
        """Test guard_can_short consistency across multiple calls."""
        symbol = "MSFT"
        qty = 1000.0

        # Multiple successful calls
        results = [guard_can_short(self.client, symbol, qty) for _ in range(5)]
        assert all(result == 30.0 for result in results)

        # Multiple failed calls
        for _ in range(3):
            with pytest.raises(PermissionError):
                guard_can_short(self.client, "MICROCAP", qty)


class TestCustomBorrowClient:
    """Test BorrowClient with custom subclassing."""

    def test_custom_borrow_client_subclass(self):
        """Test creating a custom BorrowClient subclass."""
        class CustomBorrowClient(BorrowClient):
            def locate(self, symbol: str, qty: float) -> LocateQuote:
                # Custom implementation: everything costs 100 bps
                return LocateQuote(
                    symbol=symbol.upper(),
                    available=True,
                    borrow_bps=100.0,
                    reason="Custom rate"
                )

        custom_client = CustomBorrowClient()
        quote = custom_client.locate("AAPL", 1000.0)

        assert quote.symbol == "AAPL"
        assert quote.available is True
        assert quote.borrow_bps == 100.0
        assert quote.reason == "Custom rate"

    def test_custom_client_with_guard(self):
        """Test guard_can_short with custom BorrowClient."""
        class AlwaysAvailableClient(BorrowClient):
            def locate(self, symbol: str, qty: float) -> LocateQuote:
                return LocateQuote(
                    symbol=symbol.upper(),
                    available=True,
                    borrow_bps=50.0
                )

        custom_client = AlwaysAvailableClient()

        # Should work even for microcaps with custom client
        borrow_rate = guard_can_short(custom_client, "MICROCAP", 1000.0)
        assert borrow_rate == 50.0

    def test_custom_client_unavailable(self):
        """Test guard_can_short with custom client that denies everything."""
        class NeverAvailableClient(BorrowClient):
            def locate(self, symbol: str, qty: float) -> LocateQuote:
                return LocateQuote(
                    symbol=symbol.upper(),
                    available=False,
                    borrow_bps=0.0,
                    reason="Always denied"
                )

        custom_client = NeverAvailableClient()

        with pytest.raises(PermissionError) as exc_info:
            guard_can_short(custom_client, "AAPL", 1000.0)

        assert "Always denied" in str(exc_info.value)


class TestIntegrationScenarios:
    """Test integration scenarios combining all components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = BorrowClient()

    def test_portfolio_short_screening(self):
        """Test screening multiple stocks for shorting availability."""
        portfolio = {
            "AAPL": 1000,
            "MSFT": 500,
            "GOOGL": 200,
            "MICROCAP": 100,  # Should fail
            "PENNYSTOCK": 50   # Should fail
        }

        available_shorts = {}
        failed_shorts = {}

        for symbol, qty in portfolio.items():
            try:
                borrow_rate = guard_can_short(self.client, symbol, qty)
                available_shorts[symbol] = borrow_rate
            except PermissionError as e:
                failed_shorts[symbol] = str(e)

        # Should have 3 available and 2 failed
        assert len(available_shorts) == 3
        assert len(failed_shorts) == 2

        # All available should have 30 bps rate
        for rate in available_shorts.values():
            assert rate == 30.0

        # All failed should mention locate failure
        for error in failed_shorts.values():
            assert "Locate failed" in error

    def test_borrow_cost_calculation(self):
        """Test calculating total borrow cost for a position."""
        symbol = "AAPL"
        shares = 1000
        days_held = 30

        # Get borrow rate
        borrow_rate_bps = guard_can_short(self.client, symbol, shares)

        # Calculate annual cost (assuming $150 per share)
        share_price = 150.0
        position_value = shares * share_price
        annual_borrow_cost = position_value * (borrow_rate_bps / 10000)
        daily_borrow_cost = annual_borrow_cost / 365
        total_cost = daily_borrow_cost * days_held

        # Verify calculation is reasonable
        assert borrow_rate_bps == 30.0
        assert annual_borrow_cost == 450.0  # $150k * 0.003 = $450
        assert abs(daily_borrow_cost - 1.23) < 0.01  # ~$1.23 per day
        assert abs(total_cost - 36.99) < 0.01  # ~$37 for 30 days

    def test_batch_locate_processing(self):
        """Test processing multiple locate requests efficiently."""
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "SPY", "QQQ", "ARKK"]
        qty = 1000

        quotes = []
        for symbol in symbols:
            quote = self.client.locate(symbol, qty)
            quotes.append(quote)

        # All should be available with same rate
        assert len(quotes) == len(symbols)
        for quote in quotes:
            assert quote.available is True
            assert quote.borrow_bps == 30.0
            assert quote.reason is None

    def test_locate_quote_serialization(self):
        """Test LocateQuote can be serialized/deserialized."""
        original_quote = LocateQuote("AAPL", True, 30.0, "Test quote")

        # Convert to dict
        quote_dict = asdict(original_quote)

        # Recreate from dict
        reconstructed_quote = LocateQuote(**quote_dict)

        # Should be identical
        assert reconstructed_quote == original_quote
        assert reconstructed_quote.symbol == original_quote.symbol
        assert reconstructed_quote.available == original_quote.available
        assert reconstructed_quote.borrow_bps == original_quote.borrow_bps
        assert reconstructed_quote.reason == original_quote.reason