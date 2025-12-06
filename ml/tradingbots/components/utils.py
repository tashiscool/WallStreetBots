import datetime
from datetime import timedelta

try:
    from alpaca_trade_api.rest import TimeFrame
except ImportError:
    # Fallback for when alpaca_trade_api is not available
    class TimeFrame:
        Minute = "1Min"
        Hour = "1Hour"
        Day = "1Day"

try:
    from backend.tradingbot.apimanagers import AlpacaManager
except ImportError:
    AlpacaManager = None


class DataFetcher:
    """generic stock data fetcher."""

    def get_cur_price(self, *args, **kwargs):
        return 0

    def get_past_price(self, *args, **kwargs):
        return 0

    def get_today_news(self, *args, **kwargs):
        return {"headline": "this is the headline"}


class AlpacaFetcher(DataFetcher):
    """wrapper around Alpaca API."""

    TIMESTEP = {
        "MINUTE": TimeFrame.Minute,
        "HOUR": TimeFrame.Hour,
        "DAY": TimeFrame.Day,
    }

    def __init__(self, AlpacaID, AlpacaKey):
        super().__init__()
        if AlpacaManager is None:
            raise ImportError("AlpacaManager is not available. Install required dependencies.")
        self.api = AlpacaManager(AlpacaID, AlpacaKey)
        self.api.validate_api()

    def get_cur_price(self, ticker):
        """Note that I wrapped around get_bar instead of get_price because I
        want to make sure the price is adjusted.
        """
        start = (datetime.datetime.now(datetime.UTC) - timedelta(days=1)).isoformat()
        end = datetime.datetime.now(datetime.UTC).isoformat()
        prices, _ = self.api.get_bar(ticker, TimeFrame.Minute, start, end)
        return prices[0]

    def get_past_price(
        self, ticker, timestep, start, end, price_type="close", adjustment="all"
    ):
        timestep = self.TIMESTEP[timestep]
        prices, times = self.api.get_bar(
            ticker, timestep, start, end, price_type=price_type, adjustment=adjustment
        )
        # prices and times from latest to oldest
        return prices, times


class DummyFetcher(DataFetcher):
    def get_past_price(self, *args, **kwargs):
        return [102, 103, 102, 101, 100, 99], ["t6", "t5", "t4", "t3", "t2", "t1"]

    def get_cur_price(self, *args, **kwargs):
        return 102
