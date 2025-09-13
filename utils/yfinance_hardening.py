import yfinance as yf

# Constants for data validation
MIN_DAILY_DATA_POINTS = 2


def safe_mid(bid: float, ask: float, last: float) -> float:
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    if last > 0:
        return last
    if bid > 0:
        return bid
    if ask > 0:
        return ask
    return 0.01


def fetch_last_and_prior_close(ticker: str):
    tkr = yf.Ticker(ticker)
    dailies = tkr.history(period="7d", interval="1d")
    if dailies is None or len(dailies) < MIN_DAILY_DATA_POINTS:
        return None
    prior_close = float(dailies["Close"].iloc[-2])
    intraday = tkr.history(period="2d", interval="5m")
    if intraday is None or intraday.empty:
        return None
    last = float(intraday["Close"].iloc[-1])
    return {"last": last, "prior_close": prior_close}
