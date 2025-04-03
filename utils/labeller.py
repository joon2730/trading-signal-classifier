import numpy as np
import pandas as pd
import logging
    
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
    max_price = ohlcv['High'].loc[start_index:end_index].max()
    # minimum price in the trading window
    min_price = ohlcv['Low'].loc[start_index:end_index].min()

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

