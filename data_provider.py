import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import os
from pathlib import Path
import logging

def get_ohlcv(
    ticker="BTC-USD", 
    start_date=datetime.datetime(2017, 1, 1).strftime("%Y-%m-%d"),
    end_date=datetime.datetime.now().strftime("%Y-%m-%d"),
    interval="1d"):
    """
    Fetch OHLCV data for a given ticker from Yahoo Finance.

    Parameters:
    ticker (str): The stock ticker symbol.
    start_date (str): The start date for fetching data in 'YYYY-MM-DD' format.
    end_date (str): The end date for fetching data in 'YYYY-MM-DD' format.
    """
    logging.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
    
    # Fetch data
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        raise
    
    # Convert index to datetime
    data.index = pd.to_datetime(data.index)
    
    # Unpack multi-index columns
    columns = ['Close', 'Open', 'High', 'Low', 'Volume']

    unpacked = pd.DataFrame()
    for col in columns:
        unpacked[col.lower()] = data[col][ticker]

    return unpacked