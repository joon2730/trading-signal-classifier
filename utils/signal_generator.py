import pandas as pd
import ta

def _signal_crossover(data): # dataframe with 'raw_signal' column
    """
    Helper function to determine crossover signals.
    """
    # Trigger signal only at crossover
    data['signal'] = data['raw_signal'].diff()

    # +2 = crossover up (buy) / -2 = crossover down (sell)
    data.loc[data['signal'] == 2, 'signal'] = 1   # Buy signal
    data.loc[data['signal'] == -2, 'signal'] = -1 # Sell signal
    data.loc[~data['signal'].isin([1, -1]), 'signal'] = 0   # All others = 0 (no action)

    return data

def MACD(data, short_window=12, long_window=26, signal_window=9):

    data = data.to_frame(name='Close')

    macd = ta.trend.MACD(close=data['Close'],
            window_slow=long_window,  # long-term EMA
            window_fast=short_window,  # short-term EMA
            window_sign=signal_window)   # signal line EMA
    data['macd'] = macd.macd()
    data['signal_line'] = macd.macd_signal()
    data['macd_diff'] = macd.macd_diff()
    
    # Generate raw continuous signals
    data['raw_signal'] = 0
    data.loc[data['macd_diff'] > 0, 'raw_signal'] = 1   # Buy zone
    data.loc[data['macd_diff'] < 0, 'raw_signal'] = -1  # Sell zone

    # Trigger signal only at crossover
    data = _signal_crossover(data)    

    return data[['Close', 'signal']]


def RSI(data, window=14, low_threshold=30, high_threshold=70):

    data = data.to_frame(name='Close')

    rsi = ta.momentum.RSIIndicator(close=data['Close'], window=14)
    data['rsi'] = rsi.rsi()

    def rsi_signal(row):
        if row['rsi'] < low_threshold:
            return 1   # Buy signal
        elif row['rsi'] > high_threshold:
            return -1  # Sell signal
        else:
            return 0   # Hold

    data['raw_signal'] = data.apply(rsi_signal, axis=1)

    # Trigger signal only at crossover
    data = _signal_crossover(data)

    return data[['Close', 'signal']]

signal_methods_map = {
    'MACD': MACD,
    'RSI': RSI,
}

def generate_signals(data, method, kargs):
    return signal_methods_map[method](data, **kargs)

