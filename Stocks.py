# Import libraries and dependencies
import os
import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi
from MCForecastTools import MCSimulation


import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pandas_datareader import data as pdr

import yfinance as yf

from dotenv import load_dotenv
from dotenv import dotenv_values


# Load .env enviroment variables


load_dotenv()
config = dotenv_values(".env")

api = tradeapi.REST(
    config['ALPACA_API_KEY'],
    config['ALPACA_SECRET_KEY'],
    api_version = "v2"
)


class Stock: 
    def __init__(self, ticker, start_dt, end_dt, timeframe="1D"):
        self.ticker = ticker
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.timeframe = timeframe
        try:
            print("data from alpaca")
            self.ticker_data = self.get_ticker_data_alpaca()
        except Exception as e:
            print(f"Data from alpaca not available, pulling from yahoo. Reason: {e}")
            self.ticker_data = self.get_data_yahoo()

    def get_ticker_data_alpaca(self):
        ticker_data = api.get_bars(
            self.ticker,
            self.timeframe,
            start=self.start_dt,
            end=self.end_dt,
            limit=1000,
        ).df
        self.available_dates = [x.strftime('%Y-%m-%d') for x in ticker_data.index]
        self.first_min_date = [x for x in self.available_dates if x >= self.start_dt][0]
        self.first_max_date = [x for x in self.available_dates if x <= self.end_dt][-1]
        ticker_data = ticker_data.loc[self.first_min_date: self.first_max_date]
        ticker_data = ticker_data[['open', 'high', 'low', 'close', 'volume']]
        ticker_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        ticker_data.index = [x.strftime('%Y-%m-%d') for x in ticker_data.index]
        return ticker_data

    def get_data_yahoo(self):
        import yfinance as yf

        # Some tickers in config may appear without an exchange suffix (e.g. "VFV").
        # If the initial download fails or returns no rows, retry with common suffixes.
        tickers_to_try = [self.ticker]
        if "." not in str(self.ticker):
            tickers_to_try.append(f"{self.ticker}.TO")

        last_error = None
        for ticker_try in tickers_to_try:
            try:
                data = yf.download(ticker_try, start=self.start_dt, end=self.end_dt)
                if data is None or data.empty:
                    continue

                # Rename columns to match your schema
                data = data.rename(columns={
                    'Open': 'Open',
                    'High': 'High',
                    'Low': 'Low',
                    'Close': 'Close',
                    'Volume': 'Volume',
                })

                # Only keep the columns you want, in the right order
                data = data[['Close', 'High', 'Low', 'Open', 'Volume']]
                data.columns = ['Price', 'High', 'Low', 'Open', 'Volume']

                # Set index to string format
                data.index = [x.strftime('%Y-%m-%d') for x in data.index]
                return data
            except Exception as e:
                last_error = e

        if last_error is not None:
            print(f"Yahoo Finance failed for {self.ticker} (tried {tickers_to_try}): {last_error}")
        else:
            print(f"Yahoo Finance returned no data for {self.ticker} (tried {tickers_to_try})")
        return pd.DataFrame()


