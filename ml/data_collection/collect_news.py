import json

import os
import pandas as pd
import requests
from interface import NewsSourceInterface


class NewsSource(NewsSourceInterface):
    # Use environment variable for API token instead of hardcoded value
    API_TOKEN = os.getenv("EOD_API_TOKEN", "demo_token")

    def __init__(self, ticker: str, num_rows: int = 10):
        self.ticker = ticker
        self.num_rows = num_rows

    def get_raw_data(self, date_from=None, date_to=None):
        if date_from is None:
            url = f"https: //eodhistoricaldata.com / api / news?api_token={self.API_TOKEN}&s={self.ticker}&offset = 0 & limit={self.num_rows}"
        else:
            url = f"https: //eodhistoricaldata.com / api / news?api_token={self.API_TOKEN}&s={self.ticker}&from={date_from}&to={date_to}&offset = 0 & limit={self.num_rows}"
        response = requests.get(url)
        content = response.content
        try:
            parsed = json.loads(content)
        except json.decoder.JSONDecodeError:
            parsed = {}

        return parsed

    @staticmethod
    def to_dataframe(parsed):
        d = {"date": [], "title": [], "content": [], "symbols": [], "tags": []}
        for i in range(len(parsed)):
            d["date"].append(parsed[i]["date"])
            d["title"].append(parsed[i]["title"])
            d["content"].append(parsed[i]["content"])
            d["symbols"].append(parsed[i]["symbols"])
            d["tags"].append(parsed[i]["tags"])
        df = pd.DataFrame.from_dict(d)

        return df

    def get_news(self):
        parsed = self.get_raw_data()
        df = self.to_dataframe(parsed)
        df.drop(["content", "symbols", "tags"], axis=1, inplace=True)
        return df


def main():
    news_collector = NewsSource("AAPL", num_rows=10)
    df = news_collector.get_news()
    print(df)


if __name__ == "__main__":
    main()
