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
    def __init__(self,ticker,start_dt,end_dt, timeframe = "1D"):
        
        self.ticker=ticker
        self.start_dt =start_dt
        self.end_dt = end_dt
        self.timeframe = timeframe
        yf.pdr_override()

        try:    
            print("data from alpaca")
            self.ticker_data = self.get_ticker_data_alpaca()    
        except: 
            print("Data from alpaca not available, pulling from yahoo")
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
        
        self.first_min_date =[x for x in self.available_dates if x>=self.start_dt][0]
        self.first_max_date =[x for x in self.available_dates if x<=self.end_dt][-1]
        
        ticker_data = ticker_data.loc[self.first_min_date: self.first_max_date]
        ticker_data = ticker_data[['open','high', 'low', 'close', 'volume']]
        ticker_data.columns = ['Open','High', 'Low', 'Close', 'Volume']
        
        ticker_data.index = [x.strftime('%Y-%m-%d') for x in ticker_data.index]
        return ticker_data 

        
    def get_data_yahoo(self): 
        data = pdr.get_data_yahoo(self.ticker,
            start=self.start_dt, end=self.end_dt)
        data.drop(columns=['Adj Close'], inplace=True)
        data.index = [x.strftime('%Y-%m-%d') for x in data.index]
        
        self.ticker = yf.Ticker(self.ticker)

        return data
            
        
