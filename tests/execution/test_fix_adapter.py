"""Tests for FIX Protocol Adapter."""

import pytest

from backend.tradingbot.execution.interfaces import OrderRequest
from backend.tradingbot.execution.fix_adapter import (
    FIXExecutionClient,
    FIXMessage,
    FIXMessageBuilder,
    FIXMsgType,
    FIXOrdType,
    FIXSessionConfig,
    FIXSide,
    FIXTimeInForce,
    FIXVersion,
)


@pytest.fixture
def config():
    return FIXSessionConfig(
        sender_comp_id="TEST_SENDER",
        target_comp_id="TEST_TARGET",
        fix_version=FIXVersion.FIX_44,
    )


@pytest.fixture
def builder(config):
    return FIXMessageBuilder(config)


@pytest.fixture
def client(config):
    return FIXExecutionClient(config)


def _order(symbol="AAPL", qty=100, side="buy", order_type="limit", limit_price=150.0, tif="day"):
    return OrderRequest(
        client_order_id="CLO_001",
        symbol=symbol,
        qty=qty,
        side=side,
        type=order_type,
        limit_price=limit_price,
        time_in_force=tif,
    )


class TestFIXEnums:
    def test_fix_version_values(self):
        assert FIXVersion.FIX_42.value == "FIX.4.2"
        assert FIXVersion.FIX_44.value == "FIX.4.4"

    def test_msg_types(self):
        assert FIXMsgType.NEW_ORDER_SINGLE.value == "D"
        assert FIXMsgType.ORDER_CANCEL.value == "F"
        assert FIXMsgType.EXECUTION_REPORT.value == "8"

    def test_side_values(self):
        assert FIXSide.BUY.value == "1"
        assert FIXSide.SELL.value == "2"

    def test_ord_type_values(self):
        assert FIXOrdType.MARKET.value == "1"
        assert FIXOrdType.LIMIT.value == "2"

    def test_tif_values(self):
        assert FIXTimeInForce.DAY.value == "0"
        assert FIXTimeInForce.GTC.value == "1"


class TestFIXMessage:
    def test_set_and_get_field(self):
        msg = FIXMessage(msg_type="D")
        msg.set_field(55, "AAPL")
        assert msg.get_field(55) == "AAPL"

    def test_get_missing_field(self):
        msg = FIXMessage(msg_type="D")
        assert msg.get_field(999) is None

    def test_to_fix_string(self):
        msg = FIXMessage(msg_type="D")
        msg.set_field(35, "D")
        msg.set_field(55, "AAPL")
        fix_str = msg.to_fix_string()
        assert "35=D" in fix_str
        assert "55=AAPL" in fix_str
        assert fix_str.endswith("\x01")

    def test_chained_set_field(self):
        msg = FIXMessage(msg_type="D")
        result = msg.set_field(55, "AAPL")
        assert result is msg  # Returns self for chaining


class TestFIXMessageBuilder:
    def test_new_order_single_buy(self, builder):
        msg = builder.new_order_single(_order())
        assert msg.msg_type == "D"
        assert msg.get_field(35) == "D"
        assert msg.get_field(55) == "AAPL"
        assert msg.get_field(54) == FIXSide.BUY.value
        assert msg.get_field(38) == "100"
        assert msg.get_field(40) == FIXOrdType.LIMIT.value
        assert msg.get_field(44) == "150.0"

    def test_new_order_single_sell(self, builder):
        msg = builder.new_order_single(_order(side="sell"))
        assert msg.get_field(54) == FIXSide.SELL.value

    def test_market_order_type(self, builder):
        msg = builder.new_order_single(_order(order_type="market"))
        assert msg.get_field(40) == FIXOrdType.MARKET.value

    def test_no_price_for_market_order(self, builder):
        msg = builder.new_order_single(_order(order_type="market", limit_price=None))
        assert msg.get_field(44) is None

    def test_sender_target_comp_id(self, builder):
        msg = builder.new_order_single(_order())
        assert msg.get_field(49) == "TEST_SENDER"
        assert msg.get_field(56) == "TEST_TARGET"

    def test_sequence_numbers_increment(self, builder):
        msg1 = builder.new_order_single(_order())
        msg2 = builder.new_order_single(_order())
        assert int(msg2.get_field(34)) == int(msg1.get_field(34)) + 1

    def test_order_cancel(self, builder):
        msg = builder.order_cancel(
            client_order_id="CXL_001",
            orig_order_id="CLO_001",
            symbol="AAPL",
            side="buy",
        )
        assert msg.msg_type == "F"
        assert msg.get_field(35) == "F"
        assert msg.get_field(11) == "CXL_001"
        assert msg.get_field(41) == "CLO_001"
        assert msg.get_field(55) == "AAPL"

    def test_tif_mapping(self, builder):
        msg_day = builder.new_order_single(_order(tif="day"))
        assert msg_day.get_field(59) == "0"
        msg_gtc = builder.new_order_single(_order(tif="gtc"))
        assert msg_gtc.get_field(59) == "1"
        msg_ioc = builder.new_order_single(_order(tif="ioc"))
        assert msg_ioc.get_field(59) == "3"

    def test_raw_populated(self, builder):
        msg = builder.new_order_single(_order())
        assert len(msg.raw) > 0
        assert "\x01" in msg.raw


class TestFIXExecutionClient:
    def test_connect_stub_mode(self, client):
        assert client.connect() is True
        assert client.validate_connection() is True

    def test_disconnect(self, client):
        client.connect()
        client.disconnect()
        assert client.validate_connection() is False

    def test_place_order(self, client):
        client.connect()
        ack = client.place_order(_order())
        assert ack.accepted is True
        assert ack.client_order_id == "CLO_001"
        assert ack.broker_order_id.startswith("FIX_")

    def test_cancel_order(self, client):
        client.connect()
        ack = client.place_order(_order())
        result = client.cancel_order(ack.broker_order_id)
        assert result is True

    def test_cancel_unknown_order(self, client):
        client.connect()
        assert client.cancel_order("nonexistent") is False

    def test_message_log(self, client):
        client.connect()
        client.place_order(_order())
        log = client.get_message_log()
        assert len(log) == 1
        assert log[0].msg_type == "D"

    def test_parse_execution_report(self, client):
        raw = "35=8\x0155=AAPL\x0154=1\x0138=100\x01"
        fields = client.parse_execution_report(raw)
        assert fields["35"] == "8"
        assert fields["55"] == "AAPL"
        assert fields["54"] == "1"

    def test_place_multiple_orders(self, client):
        client.connect()
        ack1 = client.place_order(_order(symbol="AAPL"))
        ack2 = client.place_order(_order(symbol="GOOGL"))
        assert ack1.broker_order_id != ack2.broker_order_id
        assert len(client.get_message_log()) == 2
