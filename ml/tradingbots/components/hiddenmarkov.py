from datetime import datetime, timedelta

import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
from alpaca_trade_api.rest import TimeFrame
from hmmlearn import hmm


class APImanager:  # API manager for Alpaca
    def __init__(self, API_KEY, SECRET_KEY):
        self.BASE_URL = "https://paper-api.alpaca.markets"
        self.ACCOUNT_URL = f"{self.BASE_URL}/v2/account"
        self.API_KEY = API_KEY
        self.SECRET_KEY = SECRET_KEY
        self.api = tradeapi.REST(API_KEY, SECRET_KEY, self.BASE_URL, api_version="v2")

    def get_bar(
        self, symbol, timestep, start, end, price_type="close", adjustment="all"
    ):
        """Get a list of prices from latest to oldest with a timestep.

        Args:
          symbol: the name of the stock
          timestep: TimeFrame.Day, TimeFrame.Hour, TimeFrame.Minute
          start: Starting time, RFC - 3339 format
          end: Ending time, RFC - 3339 format
          price_type: open, close, high, low

        Returns:
          - a list of prices from latest to oldest with a timestep
          - a list of time associated with each price
        """
        try:
            bars = self.api.get_bars(
                symbol, timestep, start, end, adjustment=adjustment
            ).df
            if bars.empty:
                return [], []
            # print(bars)
            bar_t = list(bars.index)[::-1]
            bar_prices = bars[price_type].tolist()[
                ::-1
            ]  # bars is price data in time step from latest to oldest
            return bar_prices, [t.to_pydatetime() for t in bar_t]
        except Exception as e:
            return "Failed to get bars from Alpaca: " + str(e)

    def get_price(self, symbol):
        """Get get the current price of a stock.

        Args:
          symbol: the name of the stock

        Returns:
          an float of the current price of the stock

        Note now the current price is based on last trade
        """
        try:
            bar = self.api.get_last_trade(symbol)
            price = bar._raw["price"]
            return price
        except Exception as e:
            return "Failed to get price from Alpaca: " + str(e)

    def market_close(self):
        """Checks if market closes.

        Args:
          t: the time to check if the market is closed
          if no argument is passed, check if the market is currently open or closed

        Returns:
          True / False

        To be completed
        """
        api = tradeapi.REST()
        clock = api.get_clock()

        return bool(clock.is_open)


class DataManager:
    def __init__(self, ALPACA_ID, ALPACA_KEY, tname, start_date, end_date):
        self.ALPACA_ID = ALPACA_ID
        self.ALPACA_KEY = ALPACA_KEY
        self.api = APImanager(ALPACA_ID, ALPACA_KEY)
        self.ticker = tname
        self.start_date = start_date
        self.end_date = end_date

        self.open = None
        self.close = None
        self.normalized_close = None
        self.first_min_close = None
        self.last_min_close = None
        self.len_of_data = None

        self.last_datapoint = None
        self.first_day = None
        self.unnormalized_close = None

    def get_data(self, adjustment, open):
        def _normalize_bars(raw_df, target_col):
            """Normalize Alpaca bars frame to the requested column."""
            if raw_df is None or raw_df.empty:
                return pd.DataFrame(columns=[target_col, "timestamp", "datetime", "date"])

            df_local = raw_df.copy()
            if target_col not in df_local.columns:
                fallback_cols = [c for c in ("open", "close", "high", "low") if c in df_local.columns]
                if not fallback_cols:
                    return pd.DataFrame(columns=[target_col, "timestamp", "datetime", "date"])
                df_local[target_col] = df_local[fallback_cols[0]]

            return df_local[[target_col]]

        if open:
            start_date_open = datetime.strptime(
                self.start_date, "%Y-%m-%d"
            ).date() + timedelta(days=1)
            end_date_open = datetime.strptime(
                self.end_date, "%Y-%m-%d"
            ).date() + timedelta(days=1)
            str_start_date_open = start_date_open.strftime("%Y-%m-%d")
            str_end_date_open = end_date_open.strftime("%Y-%m-%d")
            df = self.api.api.get_bars(
                self.ticker,
                TimeFrame.Day,
                str_start_date_open,
                str_end_date_open,
                adjustment=adjustment,
            ).df
            df = _normalize_bars(df, "open")  # get data

            df["timestamp"] = df.index
            df["datetime"] = pd.to_datetime(df["timestamp"])
            df["date"] = df["timestamp"].dt.date  # get date column

            true_start_date = max(
                datetime.strptime(str_start_date_open, "%Y-%m-%d").date(),
                df["date"].iloc[0],
            )
            true_end_date = min(
                datetime.strptime(str_end_date_open, "%Y-%m-%d").date(),
                df["date"].iloc[df.shape[0] - 1],
            )

            true_df = df[
                (df["date"] >= true_start_date) & (df["date"] <= true_end_date)
            ]
            return true_df
        else:
            df = self.api.api.get_bars(
                self.ticker,
                TimeFrame.Minute,
                self.start_date,
                self.end_date,
                adjustment=adjustment,
            ).df
            df = _normalize_bars(df, "close")  # get data

            df["timestamp"] = df.index
            df["datetime"] = pd.to_datetime(df["timestamp"])
            df["date"] = df["timestamp"].dt.date  # get date column

            true_start_date = max(
                datetime.strptime(self.start_date, "%Y-%m-%d").date(),
                df["date"].iloc[0],
            )
            true_end_date = min(
                datetime.strptime(self.end_date, "%Y-%m-%d").date(),
                df["date"].iloc[df.shape[0] - 1],
            )

            true_df = df[
                (df["date"] >= true_start_date) & (df["date"] <= true_end_date)
            ]
            return true_df

    def align_data(self, adjustment):
        open = self.get_data(adjustment, True)
        close = self.get_data(adjustment, False)
        open_cut = list(open["date"])[:-1]
        close_cut = list(close["date"])[1:]
        common_date = list(set(np.setdiff1d(open_cut, close_cut))) + list(
            set(np.setdiff1d(close_cut, open_cut))
        )

        new_open = open[~open["date"].isin(common_date)]
        new_close = close[~close["date"].isin(common_date)]

        if not new_open.empty and not new_close.empty:
            if new_open["date"].iloc[0] == new_close["date"].iloc[0]:
                new_open = new_open.iloc[1:]

        if not new_open.empty and not new_close.empty:
            if new_open["date"].iloc[-1] == new_close["date"].iloc[-1]:
                new_close = new_close.iloc[:-1]

        self.open = new_open
        self.close = new_close
        # return new_open, new_close

    def normalize_helper(self, seq):
        first_price = seq[0]
        return [value - first_price for value in seq]

    def normalize(self):
        normalized_seq = []
        self.first_day = []
        start = 0
        for i in range(len(list(self.close["date"].value_counts().sort_index()))):
            end = start + list(self.close["date"].value_counts().sort_index())[i]
            seq = self.close["close"][start:end].to_numpy()
            self.first_day += [seq[0]]
            normalized_seq += list(self.normalize_helper(seq))
            start = end
        seq = self.close

        self.normalized_close = seq
        # return seq

    def get_last_datapoint(self):
        lst = []
        j = -1

        for i in range(1, len(self.open)):
            j += list(self.close["date"].value_counts())[i]
            last_data = self.close["close"][j] + self.first_day[i]
            lst.append(last_data)

        self.last_datapoint = lst


class HMM:
    def __init__(self, data_manager, num_hidden_states, covar_type, n_iter):
        self.data = data_manager
        self.num_hidden_states = num_hidden_states
        self.covar_type = covar_type
        self.n_iter = n_iter
        self.model = None
        self.transit = None
        self.mean = None
        self.var = None
        self.pred = None
        self.num_uptrend = None
        self.num_pred_acc = None

    def train(self, datamanager):
        model = hmm.GaussianHMM(
            self.num_hidden_states, covariance_type=self.covar_type, n_iter=self.n_iter
        )  # or="full"
        temp_list = list(self.data.close["date"].value_counts().sort_index())
        model.fit(
            np.array(self.data.normalized_close["close"].to_numpy()).reshape(-1, 1),
            lengths=temp_list,
        )
        self.model = model
        transit = np.array(model.transmat_, dtype=float)
        row_sums = transit.sum(axis=1, keepdims=True)
        zero_rows = row_sums.squeeze() <= 0
        if np.any(zero_rows):
            transit[zero_rows, :] = 1.0 / self.num_hidden_states
            row_sums = transit.sum(axis=1, keepdims=True)
        self.transit = transit / row_sums
        self.model.transmat_ = self.transit
        self.mean = model.means_
        self.var = model.covars_

    def evaluation(self, datamanager):
        hidden_states = self.model.predict(
            np.array(self.data.normalized_close["close"].to_numpy()).reshape(-1, 1)
        )
        observed = np.array(self.data.normalized_close["close"].to_numpy(), dtype=float)
        lower_bound = float(np.min(observed) - 50) if observed.size else float("-inf")
        upper_bound = float(np.max(observed) + 50) if observed.size else float("inf")

        pred = []
        j = -1
        counts = list(self.data.close["date"].value_counts())
        for i in range(len(counts)):
            j += counts[i]
            hid = hidden_states[j]
            next_hid = np.argmax(self.transit[hid])
            mean_value = float(np.ravel(self.mean[next_hid])[0])
            var_value = float(np.ravel(self.var[next_hid])[0])
            var_value = max(var_value, 1e-8)
            base_price = (
                self.data.first_day[i]
                if i < len(self.data.first_day)
                else (self.data.first_day[-1] if self.data.first_day else 0.0)
            )

            prediction = base_price + float(np.random.normal(mean_value, var_value, 1)[0])
            prediction = min(max(prediction, lower_bound), upper_bound)
            pred.append(prediction)

        self.pred = pred

    def inference(self):
        c = 0
        u = 0
        compared = 0
        j = -1
        counts = list(self.data.close["date"].value_counts())
        for i in range(1, len(self.pred)):
            if i >= len(counts):
                break
            j += counts[i]

            if i >= len(self.data.open):
                break

            base_price = (
                self.data.first_day[i]
                if i < len(self.data.first_day)
                else (self.data.first_day[-1] if self.data.first_day else 0.0)
            )
            last_data = self.data.close["close"][j] + base_price
            pred_trend = self.pred[i] > last_data  # new_open['open'][i - 1]
            true_trend = (
                self.data.open["open"][i] > last_data
            )  # new_open['open'][i - 1]
            if pred_trend == true_trend:
                c += 1
            if true_trend:
                u += 1
            compared += 1

        denom = compared if compared > 0 else 1
        self.num_uptrend = u / denom
        self.num_pred_acc = c / denom
