import numpy as np
import pandas as pd
import logging
import ta

def label_rsi_signal(data, window=14, high_threshold=70, low_threshold=30):

    data = data.to_frame(name='Close')

    # Calculate RSI
    rsi = ta.momentum.RSIIndicator(close=data['Close'], window=window)
    data['rsi'] = rsi.rsi()

    def rsi_signal(row):
        if row['rsi'] < low_threshold:
            return 0   # Buy signal
        elif row['rsi'] > high_threshold:
            return 1  # Sell signal
        else:
            return 2   # Hold

    data['label'] = data.apply(rsi_signal, axis=1)

    return data[['label']]

def label_trading_signal(index, ohlcv, trading_window, 
                            target_profit, max_drawdown):

    # define the start and end index for the trading window
    start_index = index + 1
    end_index = index + trading_window
    
    # return None if the index + trading_window exceeds the data length
    try:
        assert len(ohlcv.loc[start_index:end_index]) == trading_window
    except Exception as e:
        return None

    # entry price
    entry_price = ohlcv['Close'].loc[index]

    # maximum price in the trading window
    max_price = ohlcv['Close'].loc[start_index:end_index].max()
    # minimum price in the trading window
    min_price = ohlcv['Close'].loc[start_index:end_index].min()

    # maximum return on long position and short position
    long_return = (max_price - entry_price) / entry_price
    short_return = (entry_price - min_price) / entry_price

    if long_return >= target_profit and short_return <= max_drawdown:
        # long position
        return 0
    elif short_return >= target_profit and long_return <= max_drawdown:
        # short position
        return 1
    else:
        # no action
        return 2

