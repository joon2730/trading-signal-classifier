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

def read_local_data(data_path, features=['Close', 'Open', 'High', 'Low', 'Volume']):
    try:
        data = pd.read_csv(data_path)
        # data['Close time'] = pd.to_datetime(data['Close time'])
        # data.set_index('Close time', inplace=True)

    except Exception as e:
        raise RuntimeError(f"Failed to read data from {data_path}: {e}")
    
    return data[features]

def read_labels(label_path):
    try:
        labels = pd.read_csv(label_path)
        labels.set_index('index', inplace=True)
        
    except Exception as e:
        raise RuntimeError(f"Failed to read labels from {label_path}: {e}")
    
    return labels


    
