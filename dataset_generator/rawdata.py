import yfinance as yf
import pandas as pd
import datetime
import logging
import numpy as np

def fetch_ohlcv_online(
    ticker="BTC-USD", 
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

def read_ohlcv_from_csv(
    data_path,
    features=['Close', 'Open', 'High', 'Low', 'Volume']):
    try:
        data = pd.read_csv(data_path)
        # data['Close time'] = pd.to_datetime(data['Close time'])
        # data.set_index('Close time', inplace=True)

    except Exception as e:
        raise RuntimeError(f"Failed to read data from {data_path}: {e}")
    
    return data[features]

def append_additional_features(ohclv, to_append={
    'log_return': [1], # 1 for daily log return
    'sma': [20, 50], # 20 and 50 days simple moving average
    'rsi': [14], # 14 days relative strength index
}):
    if 'log_return' in to_append:
        for period in to_append['log_return']:
            ohclv[f'log_return_{period}'] = ohclv['Close'].pct_change(periods=period).apply(lambda x: np.log(1+x))
    
    if 'sma' in to_append:
        for period in to_append['sma']:
            ohclv[f'sma_{period}'] = ohclv['Close'].rolling(window=period).mean()

    if 'rsi' in to_append:
        for period in to_append['rsi']:
            delta = ohclv['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            ohclv[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    ohclv.dropna(inplace=True)
    return ohclv
    
    