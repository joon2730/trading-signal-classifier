import yfinance as yf
import pandas as pd
import datetime
import logging
import numpy as np

def fetch_ohlcv_online(ticker="BTC-USD", 
    start_date=datetime.datetime(2017, 1, 1).strftime("%Y-%m-%d"),
    end_date=datetime.datetime.now().strftime("%Y-%m-%d"),
    interval="1d"):
    
    # Fetch data
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch data for {ticker}: {e}")
    
    # Convert index to datetime
    data.index = pd.to_datetime(data.index)
    
    # Unpack multi-index columns
    columns = ['Close', 'Open', 'High', 'Low', 'Volume']

    unpacked = pd.DataFrame()
    for col in columns:
        unpacked[col] = data[col][ticker]

    return unpacked

def read_local_data(data_path, columns):
    try:
        data = pd.read_csv(data_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read data from {data_path}: {e}")
    return data[columns]

def read_labels(label_path):
    labels = read_local_data(label_path, ['index', 'label'])
    labels.set_index('index', inplace=True)
    labels.dropna(inplace=True)
    labels = labels.astype(int)
    return labels


    
