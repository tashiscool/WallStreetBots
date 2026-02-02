import datetime
from datetime import timedelta

from .pipeline import Pipeline

from backend.settings import BACKEND_ALPACA_ID, BACKEND_ALPACA_KEY

from ..components.hiddenmarkov import HMM, DataManager
from ..components.naiveportfoliomanager import NaiveHMMPortfolioUpdate
from ..components.utils import AlpacaFetcher


class HMMPipline(Pipeline):
    def pipeline(self):
        data_fetcher = AlpacaFetcher(BACKEND_ALPACA_ID, BACKEND_ALPACA_KEY)
        buffer = 0.05
        start = (datetime.datetime.now(datetime.UTC) - timedelta(days=1)).isoformat()
        end = datetime.datetime.now(datetime.UTC).isoformat()
        num_hidden_states, covar_type, n_iter = 10, "full", 100
        NHPU = NaiveHMMPortfolioUpdate(
            self.portfolio,
            data_fetcher,
            DataManager,
            start,
            end,
            HMM,
            num_hidden_states,
            covar_type,
            n_iter,
            buffer=buffer,
        )
        return NHPU.rebalance()
